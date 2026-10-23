"""Cost-sensitive conflict correction and deterministic full-epoch contracts.

This is supervised binary correction plus a cross-protocol relationship loss,
not the earlier REBEL relative-reward update. Targets are training-only inputs
to these losses; they must never be included in a model prompt.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random

from .experiment import fingerprint
from .grounded_uncertainty import categorical_kl


VERSION = "omni_conflict_cost_v1"
VALUES = {"full": ("yes", "no")}
DEFAULTS = {
    "algorithm": VERSION, "seed": 42, "replication_seed": 43,
    "steps": 420, "epochs": 2, "batch_size": 16,
    "learning_rate": 5e-6, "warmup_steps": 10, "max_grad_norm": 1.,
    "beta": .02, "pair_weight": .1, "pair_margin": 1.,
    "smoke_steps": 40, "budget_hours": 24., "training_cutoff_hours": 20.,
    "video_fps": 2., "video_min_frames": 4, "video_max_frames": 64,
    "video_pixels": 50176, "audio_sampling_rate": 16000,
    "max_prompt_tokens": 6144, "max_completion_length": 128,
    "attn_implementation": "sdpa", "candidate_micro_batch_size": 1,
}


def settings_from(path=None):
    settings = dict(DEFAULTS)
    if path:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict) or set(value) - set(settings):
            raise ValueError("Unknown conflict-cost settings")
        settings.update(value)
    if settings != DEFAULTS:
        raise ValueError("v1 uses the approved fixed configuration; parameter sweeps are prohibited")
    return settings


class EpochSampler:
    """Random access replay: step alone identifies the main batch and extra pair.

    Records are sorted by sample ID before shuffling, so reloading an equivalent
    manifest in another ordering cannot change a resumed draw. Main samples are
    visited exactly once per epoch. Pair draws have a separate cyclic stream.
    """
    def __init__(self, records, seed, batch_size=16):
        records = list(records)
        if not records or not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("A nonempty training set and positive integer batch size are required")
        if any(r.get("split", "train") != "train" for r in records):
            raise ValueError("Epoch sampling accepts training records only")
        self.rows = {r["sample_id"]: r for r in records}
        if len(self.rows) != len(records):
            raise ValueError("Training sample IDs must be unique")
        if any(r["sample_type"] not in ("Conflict", "Aligned") or r["protocol"] not in ("VT", "VA") for r in records):
            raise ValueError("Original conflict labels and VT/VA protocols are required")
        self.seed, self.batch_size = int(seed), batch_size
        self.ids = sorted(self.rows)
        self.counts = dict(Counter(r["sample_type"] for r in records))
        if set(self.counts) != {"Conflict", "Aligned"}:
            raise ValueError("Both original conflict classes must be present")
        self.epoch_updates = math.ceil(len(records) / batch_size)
        groups = defaultdict(list)
        for row in records:
            groups[row["split_group_id"]].append(row)
        self.pairs = []
        for group_id, group in sorted(groups.items()):
            if (len(group) == 2 and {r["protocol"] for r in group} == {"VT", "VA"}
                    and {r["sample_type"] for r in group} == {"Conflict", "Aligned"}):
                conflict = next(r for r in group if r["sample_type"] == "Conflict")
                aligned = next(r for r in group if r["sample_type"] == "Aligned")
                if conflict.get("video") != aligned.get("video"):
                    raise ValueError("A cross-protocol pair must refer to the same actual video")
                self.pairs.append({"pair_id": group_id,
                    "conflict": conflict, "aligned": aligned})
        if len(self.pairs) != 30:
            raise ValueError("Expected the 30 original training-only cross-protocol pairs")
        self.identity = fingerprint({"algorithm": VERSION, "seed": self.seed,
            "batch_size": batch_size,
            "records": [{k: self.rows[sid].get(k) for k in
                ("sample_id", "sample_type", "protocol", "split_group_id", "video")}
                for sid in self.ids]})

    @staticmethod
    def _validate_step(step):
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("Sampler step must be a zero-based nonnegative integer")

    def _permutation(self, values, stream, index):
        values = list(values)
        rng = random.Random(int(fingerprint([self.identity, stream, index]), 16))
        rng.shuffle(values)
        return values

    def batch(self, step):
        self._validate_step(step)
        epoch, offset = divmod(step, self.epoch_updates)
        order = self._permutation(self.ids, "main", epoch)
        start = offset * self.batch_size
        return [{"row": self.rows[sid], "slot": slot,
            "draw_id": f"{self.seed}:epoch:{epoch}:position:{start + slot}",
            "selection": "epoch_uniform"}
            for slot, sid in enumerate(order[start:start + self.batch_size])]

    def pair(self, step):
        self._validate_step(step)
        cycle, offset = divmod(step, len(self.pairs))
        order = self._permutation(range(len(self.pairs)), "pairs", cycle)
        return {**self.pairs[order[offset]], "cycle": cycle,
            "draw_id": f"{self.seed}:pair:{cycle}:{offset}"}


def _binary_scores(scores):
    if scores.ndim != 1 or scores.numel() != 2 or not scores.is_floating_point():
        raise ValueError("Expected a floating point yes/no score vector")
    # Preserve double precision checks; avoid bf16 cancellation in loss arithmetic.
    return scores.float() if scores.element_size() < 4 else scores


def main_loss(scores, target, counts, batch_size):
    """One row's contribution; sum over the current (possibly short) batch.

    E_uniform[weight * logistic] = .6 E_Conflict[logistic] +
    .4 E_Aligned[logistic]. Counts always describe the complete training set.
    """
    import torch.nn.functional as F
    if target not in ("yes", "no"):
        raise ValueError("Binary training target must be yes or no")
    if set(counts) != {"Conflict", "Aligned"} or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in counts.values()):
        raise ValueError("Positive full-training-set counts for both original classes are required")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("Use the positive actual current batch size")
    scores = _binary_scores(scores)
    kind, fraction, sign = ("Conflict", .6, -1.) if target == "yes" else ("Aligned", .4, 1.)
    weight = sum(counts.values()) * fraction / counts[kind] / batch_size
    return weight * F.softplus(sign * (scores[0] - scores[1]))


def pair_loss(conflict_scores, aligned_scores, margin=1.):
    """Unweighted relation term; the caller applies pair_weight exactly once."""
    import torch.nn.functional as F
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("Pair margin must be positive and finite")
    conflict_scores, aligned_scores = _binary_scores(conflict_scores), _binary_scores(aligned_scores)
    dc = conflict_scores[0] - conflict_scores[1]
    da = aligned_scores[0] - aligned_scores[1]
    return F.softplus(margin - (dc - da))


def _valid_score(metrics):
    score = metrics.get("score")
    return isinstance(score, (float, int)) and not isinstance(score, bool) and math.isfinite(score) and 0 <= score <= 1


def choose_baseline(free, new):
    """Select the stronger zero-update baseline; preserve the original on ties."""
    for value in (free, new):
        if not _valid_score(value) or value.get("invalid_outputs", 0) != 0 or value.get("inconsistent_questions", 0) != 0:
            raise ValueError("Both zero-update baselines must have finite valid scores and legal outputs")
    return new if new["score"] > free["score"] else free


def eligible(baseline, candidate, stage="primary"):
    thresholds = {"early": .003, "primary": .015, "replication": .005}
    if stage not in thresholds:
        raise ValueError("Unknown candidate selection stage")
    if not _valid_score(baseline):
        raise ValueError("A valid measured baseline is required")
    valid_score = _valid_score(candidate)
    delta = candidate["score"] - baseline["score"] if valid_score else None
    checks = {"score": valid_score and delta >= thresholds[stage] - 1e-12,
        "output": candidate.get("invalid_outputs", -1) == 0 and candidate.get("inconsistent_questions", -1) == 0,
        "frozen_emotions": candidate.get("emotion_changes", -1) == 0}
    return {"eligible": all(checks.values()), "checks": checks, "score_gain": delta}
