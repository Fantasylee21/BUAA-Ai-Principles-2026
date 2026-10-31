"""Frozen-Omni relation learning. No rollout rewards or trainable Omni weights."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random

from .experiment import fingerprint
from .reward import VALID_EMOTIONS

VERSION = "omni_modality_relation_v1"
METHODS = ("binary_control", "joint_relation")
STATES = tuple((v, m) for v in VALID_EMOTIONS for m in VALID_EMOTIONS)
DEFAULTS = {
    "algorithm": VERSION, "seed": 42, "replication_seed": 43, "folds": 3,
    "epochs": 40, "batch_size": 64, "learning_rate": 3e-4,
    "weight_decay": .01, "warmup_epochs": 2, "max_grad_norm": 1.,
    "hidden_size": 2048, "projection_size": 128, "dropout": .1,
    "primary_gain": .015, "replication_gain": .005,
    "video_fps": 2., "video_min_frames": 4, "video_max_frames": 64,
    "video_pixels": 50176, "audio_sampling_rate": 16000,
    "max_prompt_tokens": 6144, "max_completion_length": 128,
    "attn_implementation": "sdpa", "candidate_micro_batch_size": 1,
}


def settings_from(path=None):
    value = dict(DEFAULTS)
    if path:
        supplied = json.loads(Path(path).read_text(encoding="utf-8"))
        if set(supplied) - set(value):
            raise ValueError("Unknown relation settings")
        value.update(supplied)
    if value != DEFAULTS:
        raise ValueError("This version uses the approved fixed configuration")
    return value


def allowed_states(row):
    """Keep the original relation even when one view is not clear."""
    views = row["view_targets"]
    v = views["vision"]
    m = views["text" if row["protocol"] == "VT" else "audio"]
    conflict = row["sample_type"] == "Conflict"
    if (v["label"] != m["label"]) != conflict:
        raise ValueError("Original view labels contradict the original relation")
    mask = [((a != b) == conflict and (not v["clear"] or a == v["label"])
             and (not m["clear"] or b == m["label"])) for a, b in STATES]
    if not any(mask):
        raise ValueError("No compatible relation state")
    return mask


def make_folds(rows, seed=42, n_splits=3):
    """Balance whole-video signatures; paired protocols can never cross a fold.

    Ten signatures occur in this dataset, each in at least three video groups.
    This avoids stratifying individual rows and silently splitting VT/VA pairs.
    """
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate training IDs")
    groups = defaultdict(list)
    for r in rows:
        groups[r["split_group_id"]].append(r)
    strata = defaultdict(list)
    for key, group in sorted(groups.items()):
        if len({r["video"] for r in group}) != 1:
            raise ValueError("Video grouping is inconsistent")
        signature = tuple(sorted((r["source_dataset"], r["protocol"], r["sample_type"]) for r in group))
        strata[signature].append(key)
    assignment, loads = {}, [0] * n_splits
    for signature, keys in sorted(strata.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        rng = random.Random(int(fingerprint([seed, signature]), 16))
        rng.shuffle(keys)
        offset = min(range(n_splits), key=lambda f: (loads[f], f))
        for i, key in enumerate(keys):
            fold = (offset + i) % n_splits
            assignment[key] = fold
            loads[fold] += len(groups[key])
    folds = [assignment[r["split_group_id"]] for r in rows]
    for f in range(n_splits):
        held = [r for r, k in zip(rows, folds) if k == f]
        if {r["sample_type"] for r in held} != {"Conflict", "Aligned"}:
            raise ValueError("Each development fold needs both classes")
        if not any(r["natural"] and r["protocol"] == "VA" and r["sample_type"] == "Conflict" for r in held):
            raise ValueError("A fold lost all natural VA conflicts")
    return folds


def epoch_order(size, seed, epoch):
    order = list(range(size))
    random.Random(int(fingerprint([VERSION, seed, epoch]), 16)).shuffle(order)
    return order


def build_model(method, settings=None):
    import torch
    from torch import nn
    if method not in METHODS:
        raise ValueError("Unknown relation method")
    s = settings or DEFAULTS

    class RelationHead(nn.Module):
        def __init__(self):
            super().__init__()
            d, p = s["hidden_size"], s["projection_size"]
            self.branches = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, p), nn.GELU())
                                          for _ in range(3)])
            self.fusion = nn.Sequential(nn.Linear(5*p+2, p), nn.GELU(), nn.Dropout(s["dropout"]))
            self.output = nn.Linear(p, 9 if method == "joint_relation" else 2)

        def forward(self, x, protocol):
            if x.ndim != 3 or x.shape[1:] != (3, s["hidden_size"]):
                raise ValueError("Expected [batch, vision/second/full, 2048] frozen features")
            v, m, full = [branch(x[:, i].float()) for i, branch in enumerate(self.branches)]
            p = torch.nn.functional.one_hot(protocol.long(), 2).float()
            z = torch.cat((v, m, full, (v-m).abs(), v*m, p), dim=-1)
            return self.output(self.fusion(z)).float()
    return RelationHead()


def row_losses(logits, method, masks, targets):
    import torch
    if method == "joint_relation":
        if logits.shape != masks.shape or not masks.any(-1).all():
            raise ValueError("Invalid allowed-state mask")
        # Log-sum-exp avoids underflow for high-confidence wrong predictions.
        return torch.logsumexp(logits.float(), -1) - torch.logsumexp(
            logits.float().masked_fill(~masks, -torch.inf), -1)
    if method == "binary_control":
        return torch.nn.functional.cross_entropy(logits.float(), targets.long(), reduction="none")
    raise ValueError("Unknown method")


def training_weights(rows):
    counts = Counter(r["sample_type"] for r in rows)
    if set(counts) != {"Conflict", "Aligned"}:
        raise ValueError("Both training classes required")
    return [len(rows) * (.6 if r["sample_type"] == "Conflict" else .4) / counts[r["sample_type"]]
            for r in rows]


def conflict_probability(logits, method):
    p = logits.float().softmax(-1)
    if method == "binary_control":
        return p[:, 1]
    if method == "joint_relation":
        return p[:, [i for i, (v, m) in enumerate(STATES) if v != m]].sum(-1).clamp(0.,1.)
    raise ValueError("Unknown method")


def selection_score(rows, probabilities):
    if len(rows) != len(probabilities) or not rows:
        raise ValueError("Prediction alignment changed")
    counts, correct = Counter(), Counter()
    for r, p in zip(rows, probabilities):
        if not math.isfinite(float(p)) or not 0 <= float(p) <= 1:
            raise ValueError("Invalid conflict score")
        t = r["sample_type"]
        counts[t] += 1
        correct[t] += (float(p) > .5) == (t == "Conflict")
    if not counts["Conflict"] or not counts["Aligned"]:
        raise ValueError("Both classes required for model selection")
    return .3*correct["Conflict"]/counts["Conflict"] + .2*correct["Aligned"]/counts["Aligned"]


def gate(baseline, metrics, threshold):
    keys = ("score", "invalid_outputs", "inconsistent_questions", "emotion_changes")
    if any(k not in metrics for k in keys):
        return {"passed": False, "reason": "output_checks_pending"}
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in (baseline.get("score"),metrics["score"])):
        return {"passed":False,"reason":"invalid_score"}
    gain = metrics["score"] - baseline["score"]
    good = (math.isfinite(gain) and gain >= threshold-1e-12 and
            all(metrics[k] == 0 for k in keys[1:]))
    return {"passed": good, "gain": gain, "required_gain": threshold}


def paired_diagnostics(rows, probabilities):
    groups = defaultdict(list)
    for r, p in zip(rows, probabilities):
        groups[r["split_group_id"]].append((r, p))
    pairs = [g for g in groups.values() if len(g) == 2 and
             {r["protocol"] for r, _ in g} == {"VT", "VA"} and
             {r["sample_type"] for r, _ in g} == {"Conflict", "Aligned"}]
    return {"total": len(pairs), "both_correct": sum(all((float(p) > .5) ==
            (r["sample_type"] == "Conflict") for r, p in g) for g in pairs)}
