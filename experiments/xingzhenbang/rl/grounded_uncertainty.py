"""CPU contracts, training-only robust sampler, and finite-action REBEL losses."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import random

from .experiment import fingerprint, load_records
from .reward import VALID_EMOTIONS, parse_completion, question_is_consistent

VERSION = "omni_grounded_uncertainty_rebel_v1"
VALUES = {"full": ("yes", "no"), "vision": VALID_EMOTIONS,
          "text": VALID_EMOTIONS, "audio": VALID_EMOTIONS}
BASELINE_SHA = "2446fcec5e105651157cd376d9a80ac5792489c41d8c2f2b70d710a93c4bdf23"
DEFAULTS = {"algorithm": VERSION, "seed": 42, "replication_seed": 43, "steps": 200,
    "batch_size": 16, "block_steps": 40, "learning_rate": 5e-6,
    "warmup_steps": 10, "max_grad_norm": 1., "eta": 2., "aux_weight": .1, "beta": .02,
    "smoke_steps": 20, "budget_hours": 24., "training_cutoff_hours": 20.,
    "video_fps": 2., "video_min_frames": 4, "video_max_frames": 64, "video_pixels": 50176,
    "audio_sampling_rate": 16000, "max_prompt_tokens": 6144,
    "max_completion_length": 128, "attn_implementation": "sdpa", "candidate_micro_batch_size": 1}


def settings_from(path=None):
    settings = dict(DEFAULTS)
    if path:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if set(value) - set(settings):
            raise ValueError("Unknown grounded settings")
        settings.update(value)
    if settings != DEFAULTS:
        raise ValueError("v1 uses the approved fixed configuration; parameter sweeps are prohibited")
    return settings


def load_training(path):
    canonical, duplicates = load_records(path)
    raw = {r["sample_id"]: r for r in read_rows(path)}
    if duplicates or len(raw) != len(canonical):
        raise ValueError("Training rows must be unique")
    for row in canonical:
        views = raw[row["sample_id"]]["views"]
        row["view_targets"] = {mode: {"label": views[view]["label"],
            "clear": views[view].get("is_clear") is True}
            for mode, view in (("vision", "M1"), ("text" if row["protocol"] == "VT" else "audio", "M2"))}
        if any(v["label"] not in VALID_EMOTIONS for v in row["view_targets"].values()):
            raise ValueError("Invalid modality label")
        row["natural"] = row["source_dataset"] in ("ch_sims", "ch_sims_v2")
    return canonical


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def baseline_csv(path, test, *, check_sha=True):
    from .experiment import file_hash
    if check_sha and file_hash(path) != BASELINE_SHA:
        raise ValueError("Baseline CSV does not match the confirmed 0.90813 artifact")
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["sample_id", "pred_emotion", "is_uncertain", "clarifying_question"]:
            raise ValueError("Baseline must have the four official columns")
        rows = list(reader)
    if len(rows) != 734 or [r["sample_id"] for r in rows] != [r["sample_id"] for r in test]:
        raise ValueError("Baseline must preserve all 734 official IDs and their order")
    for row in rows:
        prediction = parse_completion(json.dumps({k: v for k, v in row.items() if k != "sample_id"}))
        if prediction is None or not question_is_consistent(prediction):
            raise ValueError("Illegal baseline output")
    return rows


def main_prefix(emotion):
    if emotion not in VALID_EMOTIONS:
        raise ValueError("Only the frozen model's valid emotion may enter the decision prefix")
    return '{"pred_emotion":"' + emotion + '","is_uncertain":"'


class RobustSampler:
    """Deliberate robust objective, not an importance-corrected uniform sampler."""
    def __init__(self, records, predictions, seed):
        self.rows = {r["sample_id"]: r for r in records}
        if set(predictions) != set(self.rows):
            raise ValueError("Hard pools require complete TRAIN-only predictions")
        self.seed = seed
        self.pools = defaultdict(list)
        groups = defaultdict(list)
        for row in records:
            sid, kind, protocol = row["sample_id"], row["sample_type"], row["protocol"]
            category = "natural" if row["natural"] else "synthetic"
            key = (kind, protocol, category if kind == "Conflict" else "all")
            self.pools[key].append(sid)
            if predictions[sid] != ("yes" if kind == "Conflict" else "no"):
                self.pools[(*key, "hard")].append(sid)
            groups[row["split_group_id"]].append(row)
        self.pairs = []
        for group in groups.values():
            if len(group) == 2 and {r["protocol"] for r in group} == {"VT", "VA"} and {r["sample_type"] for r in group} == {"Conflict", "Aligned"}:
                self.pairs.append(tuple(next(r["sample_id"] for r in group if r["sample_type"] == t) for t in ("Conflict", "Aligned")))
        self.pairs.sort()
        if len(self.pairs) != 30:
            raise ValueError("Expected the 30 original training-only cross-protocol pairs")
        for kind in ("Conflict", "Aligned"):
            for protocol in ("VT", "VA"):
                for category in (("natural", "synthetic") if kind == "Conflict" else ("all",)):
                    if not self.pools[(kind, protocol, category)]:
                        raise ValueError("Missing approved sampler stratum")
        self.identity = fingerprint({"predictions": predictions, "seed": seed})

    def batch(self, step):
        rng = random.Random(int(fingerprint([VERSION, self.seed, step, self.identity]), 16))
        conflict, aligned = self.pairs[rng.randrange(len(self.pairs))]
        slots = []
        def draw(key, hard=False):
            pool = self.pools.get((*key, "hard")) if hard else None
            return rng.choice(pool or self.pools[key])
        for protocol in ("VT", "VA"):
            for kind, category, hard_count, uniform_count, forced in (
                ("Conflict", "natural", 2, 1, conflict),
                ("Aligned", "all", 2, 2, aligned)):
                key = (kind, protocol, category)
                for _ in range(hard_count):
                    slots.append((draw(key, True), "hard_or_uniform_fallback"))
                for index in range(uniform_count):
                    sid = forced if index == 0 and self.rows[forced]["protocol"] == protocol else draw(key)
                    slots.append((sid, "paired" if sid == forced else "uniform"))
            slots.append((draw(("Conflict", protocol, "synthetic")), "synthetic_uniform"))
        rng.shuffle(slots)
        return [{"slot": i, "draw_id": f"{self.seed}:{step}:{i}", "selection": why,
                 "row": self.rows[sid]} for i, (sid, why) in enumerate(slots)]


def pair_loss(scores, old_scores, rewards, eta=2.):
    import torch
    if scores.ndim != 1 or scores.shape != old_scores.shape or scores.shape != rewards.shape:
        raise ValueError("Finite matching score/reward vectors required")
    if scores.numel() not in (2, 3) or not all(torch.isfinite(v).all() for v in (scores, old_scores, rewards)):
        raise ValueError("Invalid action vectors")
    if scores.numel() == 2:
        pairs = [(0, 1)]
    else:
        gold = int(rewards.argmax())
        if rewards[gold] != 1 or rewards.sum() != 1:
            raise ValueError("Unimodal reward must identify one gold label")
        pairs = [(gold, other) for other in range(3) if other != gold]
    errors = [((scores[a]-scores[b])-(old_scores[a]-old_scores[b]).detach()
               -eta*(rewards[a]-rewards[b])) for a, b in pairs]
    return torch.stack([e.square() for e in errors]).mean()


def categorical_kl(scores, reference):
    import torch
    lp = scores.double().log_softmax(-1)
    ref = reference.detach().double().log_softmax(-1)
    return (lp.exp()*(lp-ref)).sum()


def eligible(baseline, candidate, stage="primary"):
    delta = candidate["score"] - baseline["score"]
    counts = candidate["counts"]
    thresholds = {"early": (.003, 8, 0), "primary": (.015, 10, 4), "replication": (.005, 8, 0)}
    gain, natural, va = thresholds[stage]
    checks = {"score": delta >= gain-1e-12,
        "natural_conflict": counts["natural_conflict"]["correct"] >= natural,
        "va_conflict": counts["va_conflict"]["correct"] >= va,
        "false_positives": counts["aligned_false_positive"]["correct"] <= 28,
        "output": candidate["invalid_outputs"] == 0 and candidate["inconsistent_questions"] == 0,
        "frozen_emotions": candidate.get("emotion_changes", -1) == 0}
    return {"eligible": all(checks.values()), "checks": checks, "score_gain": delta}
