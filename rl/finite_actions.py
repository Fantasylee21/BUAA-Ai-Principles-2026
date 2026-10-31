"""Finite-action expected policy gradients; no sampled-response GRPO estimator.

Pure contracts/rewards are CPU-only. The torch objective is imported lazily.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import random

from .experiment import DEFAULT_EXPERIMENT, fingerprint
from .reward import FrozenF1Reference, VALID_EMOTIONS

VERSION = "omni_finite_epg_v1"
ACTIONS = tuple((emotion, uncertain) for emotion in VALID_EMOTIONS for uncertain in ("yes", "no"))
PREFIXES = tuple('{"pred_emotion":"' + e + '","is_uncertain":"' + u + '","clarifying_question":'
                 for e, u in ACTIONS)
DEFAULTS = {
    "algorithm": VERSION, "seed": 42, "batch_size": 16, "epochs": 2,
    "learning_rate": 5e-6, "beta": .02, "warmup_steps": 10,
    "max_grad_norm": 1., "eval_every": 50, "promotion_gain": .005,
    "video_fps": 2., "video_min_frames": 4, "video_max_frames": 64,
    "video_pixels": 50176, "audio_sampling_rate": 16000,
    "max_prompt_tokens": 6144, "max_completion_length": 128,
    "attn_implementation": "sdpa", "candidate_micro_batch_size": 1,
}


def read_settings(path=None):
    settings = dict(DEFAULTS)
    if path:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if set(value) - set(settings):
            raise ValueError("Unknown finite-action configuration keys")
        settings.update(value)
    if settings["algorithm"] != VERSION or settings["epochs"] != 2:
        raise ValueError("This version requires the finite-action algorithm and a two-epoch ceiling")
    for key in ("batch_size", "eval_every", "video_min_frames", "video_max_frames", "video_pixels",
                "audio_sampling_rate", "max_prompt_tokens", "max_completion_length"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if settings["candidate_micro_batch_size"] != 1:
        raise ValueError("This NF4/BF16 version requires candidate microbatch 1 for identical train/inference scoring")
    if type(settings["seed"]) is not int or type(settings["warmup_steps"]) is not int or settings["warmup_steps"] < 0:
        raise ValueError("Invalid seed/warmup")
    for key in ("learning_rate", "beta", "max_grad_norm", "video_fps", "promotion_gain"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if (settings["video_min_frames"] % 2 or settings["video_max_frames"] % 2
            or settings["video_min_frames"] > settings["video_max_frames"] or settings["video_pixels"] % 784):
        raise ValueError("Invalid Qwen video budget")
    return settings


def omni_settings(settings):
    return {**DEFAULT_EXPERIMENT, **settings}


def epoch_batches(records, seed, epoch, batch_size):
    """Each original row exactly once, including a correctly scaled final batch."""
    indices = list(range(len(records)))
    random.Random(int(fingerprint([VERSION, seed, epoch]), 16)).shuffle(indices)
    return [[records[i] for i in indices[j:j + batch_size]] for j in range(0, len(indices), batch_size)]


def reward_table(records, predictions, *, epoch, parent_policy, train_hash):
    if fingerprint(records) != train_hash:
        raise ValueError("Reward background must match the original TRAIN records")
    if len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Reward records must be unique")
    ref = FrozenF1Reference(records, predictions)
    counts = Counter(r["sample_type"] for r in records)
    if set(counts) != {"Conflict", "Aligned"}:
        raise ValueError("Both train types required")
    table = {}
    for row in records:
        kind = row["sample_type"]
        target = "yes" if kind == "Conflict" else "no"
        weight = (.3 if kind == "Conflict" else .2) * len(records) / counts[kind]
        table[row["sample_id"]] = [
            .5 * ref.emotion_reward(row["sample_id"], row["gt_label"], e) + weight * (u == target)
            for e, u in ACTIONS]
    return {"schema": "finite_reward_v1", "split": "train", "epoch": epoch,
            "train_hash": train_hash, "parent_policy": parent_policy,
            "background_predictions": predictions, "background_macro_f1": ref.base_f1,
            "type_counts": dict(counts), "actions": [list(a) for a in ACTIONS], "rewards": table}


def validate_reward(table, records, epoch, parent_policy):
    expected = reward_table(records, table["background_predictions"], epoch=epoch,
                            parent_policy=parent_policy, train_hash=fingerprint(records))
    if table != expected:
        raise ValueError("Reward table identity or values do not match TRAIN-only reconstruction")


def objective(scores, reference_log_probs, rewards, beta):
    """Exact expectation over the six-action policy, with exact categorical KL.

    float64 on six scalars preserves small alternative-action probabilities;
    model token probabilities still use float32 for numerical stability.
    """
    import torch
    if scores.shape[-1] != 6 or scores.shape != reference_log_probs.shape or scores.shape != rewards.shape:
        raise ValueError("Expected matching six-action tensors")
    if not all(torch.isfinite(t).all() for t in (scores, reference_log_probs, rewards)):
        raise ValueError("Nonfinite finite-action inputs")
    lp = scores.double().log_softmax(-1)
    ref = reference_log_probs.detach().double()
    if not torch.allclose(ref.logsumexp(-1), torch.zeros_like(ref[..., 0]), atol=1e-6, rtol=0):
        raise ValueError("Reference probabilities are not normalized")
    probs = lp.exp()
    reward = (probs * rewards.double()).sum(-1)
    kl = (probs * (lp - ref)).sum(-1)
    loss = -reward + beta * kl
    if not torch.isfinite(loss).all():
        raise FloatingPointError("Nonfinite expected policy objective")
    return loss, {"expected_reward": reward, "kl": kl, "entropy": -(probs * lp).sum(-1),
                  "max_probability": probs.max(-1).values, "probabilities": probs}


def selection(baselines, candidate, gain=.005):
    if set(baselines) != {"free", "finite"}:
        raise ValueError("Both zero-update baselines are required")
    scores = [b.get("score") for b in baselines.values()]
    if candidate.get("score") is None or any(s is None for s in scores):
        raise ValueError("Complete validation is required")
    reference = max(scores)
    eligible = (candidate["score"] >= reference + gain - 1e-12
                and candidate.get("invalid_outputs", 0) == 0
                and candidate.get("inconsistent_questions", 0) == 0)
    return {"promote": eligible, "baseline_score": reference,
            "score_gain": candidate["score"] - reference, "required_gain": gain}
