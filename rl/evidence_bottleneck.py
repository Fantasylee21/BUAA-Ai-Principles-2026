"""Isolated emotion evidence and convex, two-offset conflict calibration.

No trainable Omni weights, fused features, rollout rewards, or validation fitting.
All small-model arithmetic is CPU NumPy/SciPy float64.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from .experiment import fingerprint
from .modality_relation import make_folds
from .reward import VALID_EMOTIONS

VERSION = "omni_evidence_bottleneck_v1"
MODES = ("vision", "text", "audio")
DEFAULTS = {
    "algorithm": VERSION, "seed": 42, "replication_seed": 43,
    "folds": 3, "inner_folds": 5, "hidden_size": 2048,
    "regularization": .01, "max_iterations": 300,
    "gradient_tolerance": 1e-5, "function_tolerance": 1e-12,
    "layer_norm_epsilon": 1e-5, "primary_gain": .015, "replication_gain": .005,
    "video_fps": 2., "video_min_frames": 4, "video_max_frames": 64,
    "video_pixels": 50176, "audio_sampling_rate": 16000,
    "max_prompt_tokens": 6144, "max_completion_length": 128,
    "attn_implementation": "sdpa", "candidate_micro_batch_size": 1,
}


def settings_from(path=None):
    settings = dict(DEFAULTS)
    if path:
        supplied = json.loads(Path(path).read_text(encoding="utf-8"))
        if set(supplied)-set(settings):
            raise ValueError("Unknown evidence setting")
        settings.update(supplied)
    if settings != DEFAULTS:
        raise ValueError("This experiment uses the approved fixed configuration")
    return settings


def safe_input(row, mode):
    """Only these values may reach an isolated input or its feature identity."""
    if mode not in MODES or (mode == "text" and row["protocol"] != "VT") or (mode == "audio" and row["protocol"] != "VA"):
        raise ValueError("Wrong isolated modality")
    return {"mode": mode, **({"text": row["text_content"]} if mode == "text" else {"video": row["video"]})}


def feature_key(split, row, mode):
    # Visual sharing is explicit; second-modality samples remain individually traceable.
    unit = row["video"] if mode == "vision" else row["sample_id"]
    return fingerprint([VERSION, split, mode, unit, safe_input(row, mode)])


def tasks_for(rows, split):
    result = {}
    for row in rows:
        for mode in ("vision", "text" if row["protocol"] == "VT" else "audio"):
            key = feature_key(split, row, mode)
            result.setdefault(key, {"key": key, "split": split, "mode": mode, "row": row})
    return list(result.values())


def view_examples(rows, mode):
    unique = {}
    for row in rows:
        if mode not in row["view_targets"]:
            continue
        target = row["view_targets"][mode]
        unit = row["video"] if mode == "vision" else row["sample_id"]
        if unit in unique and unique[unit][1] != target:
            raise ValueError("Shared visual input has inconsistent original annotations")
        unique.setdefault(unit, (row, target))
    return [(row, VALID_EMOTIONS.index(target["label"])) for row, target in unique.values() if target["clear"]]


def layer_norm(h, epsilon=1e-5):
    h = np.asarray(h, dtype=np.float64)
    if h.ndim != 2 or not np.isfinite(h).all():
        raise ValueError("Expected finite hidden-state matrix")
    centered = h-h.mean(-1, keepdims=True)
    return centered/np.sqrt((centered*centered).mean(-1, keepdims=True)+epsilon)


def softmax(scores):
    scores = np.asarray(scores, dtype=np.float64)
    return np.exp(scores-logsumexp(scores, axis=-1, keepdims=True))


def head_objective(parameters, x, native, targets, regularization=.01):
    n, d = x.shape
    values = parameters.reshape(3, d+1)
    scores = native+x@values[:, :d].T+values[:, d]
    logp = scores-logsumexp(scores, axis=1, keepdims=True)
    loss = -logp[np.arange(n), targets].mean()+.5*regularization*np.dot(parameters, parameters)
    error = np.exp(logp)
    error[np.arange(n), targets] -= 1
    gradient = np.concatenate(((error.T@x)/n, error.mean(0)[:, None]), axis=1).ravel()
    return float(loss), gradient+regularization*parameters


def solve(objective, size, settings=None, emit=None):
    s = settings or DEFAULTS
    iteration = 0
    def checked(theta):
        loss, grad = objective(theta)
        if not np.isfinite(loss) or not np.isfinite(grad).all():
            raise FloatingPointError("Nonfinite convex objective; fit not published")
        return loss, grad
    initial = np.zeros(size, dtype=np.float64)
    def callback(theta):
        nonlocal iteration
        iteration += 1
        loss, grad = checked(theta)
        if emit:
            emit(iteration=iteration, max_iterations=s["max_iterations"], loss=loss,
                 data_loss=float(loss-.5*s["regularization"]*np.dot(theta, theta)),
                 gradient_max=float(np.abs(grad).max()))
    outcome = minimize(checked, initial, method="L-BFGS-B", jac=True, callback=callback,
        options={"maxiter": s["max_iterations"], "maxls": 40,
                 "gtol": s["gradient_tolerance"], "ftol": s["function_tolerance"], "maxcor": 20})
    loss, gradient = checked(outcome.x)
    diagnostics = {"success": bool(outcome.success), "message": str(outcome.message),
        "iterations": int(outcome.nit), "evaluations": int(outcome.nfev),
        "loss": loss, "data_loss": float(loss-.5*s["regularization"]*np.dot(outcome.x, outcome.x)),
        "gradient_max": float(np.abs(gradient).max()), "initial_loss": checked(initial)[0]}
    if not outcome.success:
        error = RuntimeError("L-BFGS did not converge: "+json.dumps(diagnostics))
        error.diagnostics = diagnostics
        raise error
    return outcome.x, diagnostics


def fit_head(hidden, native, targets, settings=None, emit=None):
    s = settings or DEFAULTS
    x = layer_norm(hidden, s["layer_norm_epsilon"])
    native = np.asarray(native, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if not len(x) or native.shape != (len(x), 3) or targets.shape != (len(x),) or not np.isfinite(native).all() or not np.isin(targets, range(3)).all():
        raise ValueError("Invalid evidence supervision")
    theta, diagnostics = solve(lambda p: head_objective(p, x, native, targets, s["regularization"]),
                               3*(x.shape[1]+1), s, emit)
    return {"parameters": theta.reshape(3, x.shape[1]+1).tolist(), "solver": diagnostics}


def evidence_probabilities(hidden, native, head=None):
    x = layer_norm(hidden)
    scores = np.asarray(native, dtype=np.float64)
    if head is not None:
        weights = np.asarray(head["parameters"], dtype=np.float64)
        if weights.shape != (3, x.shape[1]+1):
            raise ValueError("Wrong evidence head geometry")
        scores = scores+x@weights[:, :-1].T+weights[:, -1]
    return softmax(scores)


def disagreement(vision, second):
    a, b = np.asarray(vision, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Expected two aligned three-emotion distributions")
    if (a < 0).any() or (b < 0).any() or not np.allclose(a.sum(1), 1) or not np.allclose(b.sum(1), 1):
        raise ValueError("Unnormalized evidence probabilities")
    return np.clip(1-(a*b).sum(1), 0., 1.)


def log_odds(q):
    q = np.asarray(q, dtype=np.float64)
    if not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
        raise ValueError("Invalid disagreement score")
    # Exact endpoints are handled explicitly in decision_scores.
    clipped = np.clip(q, 1e-15, 1-1e-15)
    return np.log(clipped)-np.log1p(-clipped)


def calibration_objective(offsets, odds, protocol, labels, regularization=.01):
    labels = np.asarray(labels, dtype=bool)
    nc, na = labels.sum(), (~labels).sum()
    if not nc or not na:
        raise ValueError("Both classes are required for calibration")
    weight = np.where(labels, .6/nc, .4/na)
    d = odds+offsets[protocol]
    per_row = np.logaddexp(0., d)-labels*d
    gradient = np.bincount(protocol, weights=weight*(expit(d)-labels), minlength=2)
    return (float(np.dot(weight, per_row)+.5*regularization*np.dot(offsets, offsets)),
            gradient+regularization*offsets)


def fit_calibration(q, rows, settings=None, emit=None):
    s = settings or DEFAULTS
    protocol = np.asarray([r["protocol"] == "VA" for r in rows], dtype=np.int64)
    labels = np.asarray([r["sample_type"] == "Conflict" for r in rows])
    odds = log_odds(q)
    if len(odds) != len(rows):
        raise ValueError("Calibration order changed")
    offsets, diagnostics = solve(lambda p: calibration_objective(p, odds, protocol, labels, s["regularization"]),
                                  2, s, emit)
    return {"offsets": offsets.tolist(), "solver": diagnostics}


def decision_scores(q, rows, calibration=None):
    q = np.asarray(q, dtype=np.float64)
    odds = log_odds(q)
    if len(q) != len(rows):
        raise ValueError("Evidence and input counts differ")
    offsets = np.asarray(calibration["offsets"] if calibration else [0., 0.], dtype=np.float64)
    if offsets.shape != (2,) or not np.isfinite(offsets).all():
        raise ValueError("Invalid protocol offsets")
    p = expit(odds+offsets[[int(r["protocol"] == "VA") for r in rows]])
    return np.where(q == 0, 0., np.where(q == 1, 1., p))


def split_design(rows, seed):
    outer = make_folds(rows, seed, 3)
    result = []
    for fold in range(3):
        remaining = [i for i, f in enumerate(outer) if f != fold]
        inner = make_folds([rows[i] for i in remaining], seed+100+fold, 5)
        train = [i for i, f in zip(remaining, inner) if f != 0]
        cal = [i for i, f in zip(remaining, inner) if f == 0]
        held = [i for i, f in enumerate(outer) if f == fold]
        sets = [{rows[i]["video"] for i in ids} for ids in (train, cal, held)]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise ValueError("A video crossed training/calibration/outer folds")
        result.append({"fold": fold, "train": train, "calibration": cal, "held": held})
    return {"seed": seed, "outer": outer, "nested": result,
            "hash": fingerprint([seed, outer, result])}
