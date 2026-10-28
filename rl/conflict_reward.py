"""Contracts and the on-policy, two-action conflict-reward experiment.

Neither sample IDs nor gold labels enter the policy prompt. Rewards use TRAIN
counts; evaluation always uses the original task weights, including for B/C.
"""
from __future__ import annotations

from collections import Counter
import math
import random

VERSION = "conflict_reward_rl_v1"
ARMS = ("A", "B", "C")
SETTINGS = {
    "epochs": 2, "batch_size": 16, "generations": 8,
    "learning_rate": 5e-6, "warmup_fraction": .1, "weight_decay": 0.,
    "beta": .02, "repair_weight": .1, "keep_weight": .1,
    "max_grad_norm": 1., "video_fps": 2., "video_min_frames": 4,
    "video_max_frames": 64, "video_pixels": 50176,
    "audio_sampling_rate": 16000, "max_prompt_tokens": 6144,
    "max_completion_length": 128, "attn_implementation": "sdpa",
    "candidate_micro_batch_size": 1, "save_every": 20,
    "budget_hours": 72, "pilot_hours": 24, "memory_gib": 40,
}


def counts_for(rows):
    counts = Counter(r["sample_type"] for r in rows)
    if set(counts) != {"Conflict", "Aligned"}:
        raise ValueError("Each training partition needs both original sample types")
    return dict(counts)


def reward_weight(kind, counts, arm="A"):
    if arm not in ARMS or kind not in ("Conflict", "Aligned"):
        raise ValueError("Unknown reward arm or sample type")
    if set(counts) != {"Conflict", "Aligned"} or any(type(n) is not int or n <= 0 for n in counts.values()):
        raise ValueError("Rewards require positive TRAIN partition counts")
    return (.3 * (2 if arm != "A" else 1) if kind == "Conflict" else .2) * sum(counts.values()) / counts[kind]


def batches(rows, seed, epoch):
    order = list(range(len(rows)))
    random.Random(f"{VERSION}:{seed}:{epoch}").shuffle(order)
    return [[rows[i] for i in order[j:j + SETTINGS["batch_size"]]]
            for j in range(0, len(order), SETTINGS["batch_size"])]


def anchored_logit(log_odds, reference_log_odds, baseline_yes):
    if type(baseline_yes) is not bool:
        raise ValueError("The anchor must be a prediction, represented as bool")
    return math.log(3.) * (1 if baseline_yes else -1) + log_odds - reference_log_odds


def objective(log_odds, reference_log_odds, baseline_yes, gold_yes, counts, arm,
              *, actions=None, generator=None):
    """One fresh sample group, one update; LOO baseline is detached.

    Actions are Bernoulli draws from THIS transformed policy, not from the
    native LM. Explicit actions are for numerical tests/replaying diagnostics.
    All losses are per-original-example; the trainer divides by batch size.
    """
    import torch
    import torch.nn.functional as F
    s = anchored_logit(log_odds.double(), float(reference_log_odds), baseline_yes)
    p = s.sigmoid()
    s0 = s.new_tensor(math.log(3.) * (1 if baseline_yes else -1))
    p0 = s0.sigmoid()
    if actions is None:
        actions = torch.bernoulli(p.detach().expand(SETTINGS["generations"]), generator=generator)
    else:
        actions = torch.as_tensor(actions, dtype=s.dtype, device=s.device)
    if actions.shape != (SETTINGS["generations"],) or not ((actions == 0) | (actions == 1)).all():
        raise ValueError("Expected exactly eight binary policy actions")
    kind = "Conflict" if gold_yes else "Aligned"
    rewards = (actions == int(gold_yes)).double() * reward_weight(kind, counts, arm)
    advantage = (len(actions) / (len(actions) - 1)) * (rewards - rewards.mean())
    logp = actions * F.logsigmoid(s) + (1 - actions) * F.logsigmoid(-s)
    pg = -(advantage.detach() * logp).mean()
    kl = p * (F.logsigmoid(s) - F.logsigmoid(s0)) + (1-p) * (F.logsigmoid(-s) - F.logsigmoid(-s0))
    reverse_kl = p0 * (F.logsigmoid(s0) - F.logsigmoid(s)) + (1-p0) * (F.logsigmoid(-s0) - F.logsigmoid(-s))
    zero = bool(torch.equal(rewards, rewards[0].expand_as(rewards)))
    repair = F.binary_cross_entropy_with_logits(s, s.new_tensor(float(gold_yes))) * reward_weight(kind, counts, "A") if arm == "C" and zero else s * 0
    keep = reverse_kl if arm == "C" and not gold_yes and not baseline_yes else s * 0
    loss = pg + SETTINGS["beta"] * kl + SETTINGS["repair_weight"] * repair + SETTINGS["keep_weight"] * keep
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite two-action objective")
    return loss, {"loss": float(loss.detach()), "pg_loss": float(pg.detach()),
        "kl": float(kl.detach()), "repair_loss": float(repair.detach()), "keep_loss": float(keep.detach()),
        "zero_advantage": int(zero), "probability": float(p.detach()),
        "expected_reward": float(p.detach() if gold_yes else 1-p.detach()) * reward_weight(kind, counts, arm),
        "actions": actions.detach().cpu().tolist()}


def predicted(baseline, probability, *, delta=0.):
    from scripts.joint_affect_data import EMOTIONS
    if baseline["pred_emotion"] not in EMOTIONS or baseline["is_uncertain"] not in ("yes", "no"):
        raise ValueError("Invalid frozen reference prediction")
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid uncertainty probability")
    return {"sample_id": baseline["sample_id"],
        "emotion_probs": [float(e == baseline["pred_emotion"]) for e in EMOTIONS],
        "uncertain_prob": probability, "delta": float(delta)}


def compare(rows, base, candidate, design=None, *, stage="primary"):
    from scripts.joint_affect_data import metrics
    from scripts.joint_affect_pipeline import public_metrics, independent_audit
    if [r["sample_id"] for r in rows] != [r["sample_id"] for r in base] or [r["sample_id"] for r in rows] != [r["sample_id"] for r in candidate]:
        raise ValueError("Exact, ordered split coverage is required")
    unchanged = all(a["emotion_probs"] == b["emotion_probs"] for a, b in zip(base, candidate))
    a, b = metrics(rows, base), metrics(rows, candidate)
    independent_audit(rows, base, a)
    independent_audit(rows, candidate, b)
    gain = b["selection_score"] - a["selection_score"]
    component_gain = .3 * (b["true_positives"]-a["true_positives"]) / a["n_conflict"] - .2 * (b["false_positives"]-a["false_positives"]) / a["n_aligned"]
    if unchanged and abs(component_gain-gain) > 1e-10:
        raise ValueError("Independent TP/FP score difference disagrees")
    folds = []
    if design:
        for part in design["folds"]:
            ids = part["held"]
            ma = metrics([rows[i] for i in ids], [base[i] for i in ids])
            mb = metrics([rows[i] for i in ids], [candidate[i] for i in ids])
            folds.append({"fold": part["fold"], "gain": mb["selection_score"]-ma["selection_score"]})
    required = {"pilot": .003, "primary": .005, "replication": .003, "pressure": .0025}[stage]
    checks = {"gain": (component_gain if stage == "pressure" else gain) >= required-1e-12,
              "unchanged_emotions": unchanged,
              "natural_va_not_worse": b["natural_va_tp"] >= a["natural_va_tp"]}
    if stage == "primary":
        checks["natural_gain"] = b["natural_tp"] >= a["natural_tp"] + 5
    if stage in ("primary", "replication"):
        checks["two_nondecreasing_folds"] = len(folds) == 3 and sum(f["gain"] >= -1e-12 for f in folds) >= 2
    # The pilot is an economic futility gate on score only.
    if stage == "pilot":
        checks.pop("natural_va_not_worse")
    transitions = Counter()
    for row, x, y in zip(rows, base, candidate):
        ox, oy = x["uncertain_prob"] >= .5, y["uncertain_prob"] >= .5
        if ox != oy:
            correct = oy == (row["sample_type"] == "Conflict")
            transitions["new_correct" if correct else "new_wrong"] += 1
    return {"stage": stage, "control": public_metrics(a), "candidate": public_metrics(b),
        "gain": gain, "uncertainty_gain": component_gain, "folds": folds,
        "changes": dict(transitions), "checks": checks, "passed": all(checks.values())}


def choose(reports):
    candidates = [a for a in ARMS if a in reports and reports[a]["passed"]]
    if not candidates:
        return None
    best = max(reports[a]["candidate"]["selection_score"] for a in candidates)
    return next(a for a in candidates if reports[a]["candidate"]["selection_score"] >= best-.0001-1e-12)


def bootstrap(rows, base, candidate, groups, repeats=2000):
    """Paired content-group bootstrap of the score difference (emotion fixed)."""
    import numpy as np
    if len(groups) != len(rows):
        raise ValueError("One content group per original sample required")
    unique = sorted(set(groups))
    index = {g:i for i,g in enumerate(unique)}
    values = np.zeros((len(unique), 4), dtype=float)
    for row,a,b,g in zip(rows,base,candidate,groups):
        c = row["sample_type"] == "Conflict"
        values[index[g], 0 if c else 1] += 1
        values[index[g], 2 if c else 3] += int(b["uncertain_prob"] >= .5)-int(a["uncertain_prob"] >= .5)
    rng = np.random.default_rng(20261028)
    differences=[]
    for _ in range(repeats):
        n = values[rng.integers(len(unique), size=len(unique))].sum(0)
        if n[0] and n[1]:
            differences.append(.3*n[2]/n[0]-.2*n[3]/n[1])
    if not differences:
        raise ValueError("Bootstrap resamples contain no scoreable partitions")
    return {"unit":"content_group", "seed":20261028, "repeats":len(differences),
        "percentile_95":np.quantile(differences,[.025,.975]).tolist(),
        "selection_adjusted":False, "public_score_guarantee":False}
