"""CPU-only, grouped development of a six-feature evidence calibrator.

Uses frozen base-only TRAIN feature caches; never fits on validation or test.
Run: python -m scripts.train_evidence_calibrator --source-run OLD --output-dir NEW
There is deliberately no test/export command and no threshold search option.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
import os
from pathlib import Path, PurePosixPath
import platform
import time
import traceback
import uuid

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import scipy

from rl.experiment import file_hash, fingerprint, write_json

VERSION = "omni_evidence_calibrator_v1"
FEATURES = ("squared_distribution_distance", "vision_margin", "second_margin",
            "vision_entropy", "second_entropy", "protocol_va")
CONFIG = {"algorithm": VERSION, "seed": 42, "outer_folds": 3,
          "inner_folds": 5, "calibration_fold": 0, "regularization": .01,
          "conflict_class_weight": .6, "aligned_class_weight": .4,
          "threshold": .5, "max_iterations": 500,
          "gradient_tolerance": 1e-8, "function_tolerance": 1e-12,
          "accepted_gradient_max": 1e-4, "pressure_test": True}
NOTE = ("Only the uncertainty evidence branch is group-held-out. Historical v4 "
        "predictions have seen original training samples, so this is not an "
        "independent evaluation of the complete fused system. Validation is "
        "reported once after the model is locked; it is never a selection gate.")


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def immutable_json(path, value):
    """Publish a complete JSON atomically; reject different existing content."""
    path, content = Path(path), json_bytes(value)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("Immutable artifact changed: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            if os.name == "nt":
                # Windows rename is atomic and refuses an existing destination.
                temporary.rename(path)
            else:
                os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError("Immutable artifact changed: " + str(path))
    finally:
        temporary.unlink(missing_ok=True)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def publish_artifact(path, value):
    """The checksum marker is published only after a complete immutable file."""
    path = Path(path)
    immutable_json(path, value)
    immutable_json(str(path) + ".complete.json", {"algorithm": VERSION,
        "sha256": file_hash(path), "payload_fingerprint": fingerprint(value)})


def checked_artifact(path):
    path = Path(path)
    marker = read(str(path) + ".complete.json")
    value = read(path)
    if (marker.get("algorithm") != VERSION or marker.get("sha256") != file_hash(path) or
            marker.get("payload_fingerprint") != fingerprint(value)):
        raise ValueError("Completed calibrator artifact checksum changed: " + str(path))
    return value


def emit(root, phase, **values):
    value = {"phase": phase, "updated": time.time(), "pid": os.getpid(), **values}
    write_json(Path(root) / "status.json", value)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


@contextmanager
def run_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".run.lock"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError as exc:
        raise ValueError("Output is already locked; inspect the existing run before recovery") from exc
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def _validate_rows(rows, require_labels=False):
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("Expected nonempty ordered sample rows")
    ids = [r.get("sample_id") for r in rows]
    if any(not isinstance(sid, str) or not sid for sid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Sample IDs must be unique, nonempty strings")
    if any(r.get("protocol") not in ("VT", "VA") for r in rows):
        raise ValueError("Every row must have a VT or VA protocol")
    if require_labels and any(r.get("sample_type") not in ("Conflict", "Aligned") for r in rows):
        raise ValueError("Original relation labels are required")


def _probabilities(values, n, name):
    p = np.asarray(values, dtype=np.float64)
    if (p.shape != (n, 3) or not np.isfinite(p).all() or
            ((p < 0) | (p > 1)).any() or
            not np.allclose(p.sum(axis=1), 1., rtol=0., atol=1e-6)):
        raise ValueError(name + " must contain finite normalized N x 3 probabilities")
    return p


def six_features(vision, second, rows):
    """Extract label-free evidence summaries, in the fixed FEATURES order."""
    _validate_rows(rows)
    v = _probabilities(vision, len(rows), "vision")
    m = _probabilities(second, len(rows), "second")
    vs, ms = np.sort(v, axis=1), np.sort(m, axis=1)
    entropy = lambda p: -(p * np.log(np.clip(p, 1e-300, 1.))).sum(axis=1)
    return np.column_stack((.5 * ((v - m) ** 2).sum(axis=1),
        vs[:, -1] - vs[:, -2], ms[:, -1] - ms[:, -2],
        entropy(v), entropy(m), [float(r["protocol"] == "VA") for r in rows]))


def disagreement_gate(vision, second, probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected finite calibrated probabilities")
    v = _probabilities(vision, len(p), "vision")
    m = _probabilities(second, len(p), "second")
    return (v.argmax(axis=1) != m.argmax(axis=1)) & (p > CONFIG["threshold"])


def logistic_objective(theta, x, labels):
    """Class-conditional weighted CE and L2 on all coefficients, including bias."""
    x, theta = np.asarray(x, dtype=np.float64), np.asarray(theta, dtype=np.float64)
    y = np.asarray(labels)
    if (x.ndim != 2 or x.shape[1] != len(FEATURES) or
            theta.shape != (len(FEATURES) + 1,) or y.shape != (len(x),) or
            not np.isfinite(x).all() or not np.isfinite(theta).all() or
            not np.isin(y, (0, 1)).all()):
        raise ValueError("Invalid logistic objective geometry or labels")
    y = y.astype(np.float64)
    nc, na = float(y.sum()), float(len(y) - y.sum())
    if nc == 0 or na == 0:
        raise ValueError("Calibration requires both original classes")
    weights = np.where(y == 1, .6 / nc, .4 / na)
    logits = x @ theta[:-1] + theta[-1]
    residual = weights * (expit(logits) - y)
    loss = np.dot(weights, np.logaddexp(0., logits) - y * logits)
    loss += .5 * CONFIG["regularization"] * np.dot(theta, theta)
    gradient = np.r_[x.T @ residual, residual.sum()] + CONFIG["regularization"] * theta
    return float(loss), gradient


def fit_logistic(features, rows, progress=None):
    """Fit the scaler and logistic model on this calibration partition only."""
    _validate_rows(rows, require_labels=True)
    raw = np.asarray(features, dtype=np.float64)
    if raw.shape != (len(rows), len(FEATURES)) or not np.isfinite(raw).all():
        raise ValueError("Invalid six-feature calibration matrix")
    labels = np.array([r["sample_type"] == "Conflict" for r in rows], dtype=np.int64)
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    scale = np.where(scale == 0., 1., scale)
    x = (raw - mean) / scale
    initial = np.zeros(len(FEATURES) + 1, dtype=np.float64)
    objective = lambda theta: logistic_objective(theta, x, labels)
    initial_loss = objective(initial)[0]
    iterations = 0

    def callback(theta):
        nonlocal iterations
        iterations += 1
        if progress:
            loss, grad = objective(theta)
            progress(iteration=iterations, loss=loss, gradient_max=float(np.abs(grad).max()))

    result = minimize(objective, initial, method="L-BFGS-B", jac=True, callback=callback,
        options={"maxiter": CONFIG["max_iterations"], "maxls": 40,
                 "gtol": CONFIG["gradient_tolerance"], "ftol": CONFIG["function_tolerance"]})
    loss, gradient = objective(result.x)
    valid = (bool(result.success) and np.isfinite(result.x).all() and np.isfinite(loss) and
             np.isfinite(gradient).all() and loss <= initial_loss + 1e-10 and
             np.abs(gradient).max() <= CONFIG["accepted_gradient_max"])
    diagnostics = {"success": bool(valid), "message": str(result.message),
        "iterations": int(result.nit), "evaluations": int(result.nfev),
        "loss": loss, "initial_loss": initial_loss,
        "gradient_max": float(np.abs(gradient).max())}
    if not valid:
        error = RuntimeError("Logistic fit failed convergence verification: " + json.dumps(diagnostics))
        error.diagnostics = diagnostics
        raise error
    return {"features": list(FEATURES), "mean": mean.tolist(), "scale": scale.tolist(),
            "coefficients": result.x.tolist(), "solver": diagnostics,
            "examples": [r["sample_id"] for r in rows]}


def predict_logistic(features, model):
    raw = np.asarray(features, dtype=np.float64)
    mean, scale = np.asarray(model["mean"]), np.asarray(model["scale"])
    theta = np.asarray(model["coefficients"])
    if (model.get("features") != list(FEATURES) or model["solver"].get("success") is not True or
            raw.ndim != 2 or raw.shape[1] != len(FEATURES) or
            mean.shape != (6,) or scale.shape != (6,) or theta.shape != (7,) or
            not all(np.isfinite(v).all() for v in (raw, mean, scale, theta)) or (scale <= 0).any()):
        raise ValueError("Invalid or unconverged fixed logistic model")
    return expit(((raw - mean) / scale) @ theta[:-1] + theta[-1])


def _natural(row):
    return row.get("natural") is True or row.get("source_dataset") in ("ch_sims", "ch_sims_v2")


def content_groups(rows, media, pressure=False):
    """Union same video and identical media; pressure also unions parent dirs."""
    _validate_rows(rows, require_labels=True)
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)

    seen = {}
    for i, row in enumerate(rows):
        video = row.get("video")
        if not isinstance(video, str) or not video or video not in media:
            raise ValueError("Every training video needs a verified media SHA-256")
        digest = media[video]
        if (not isinstance(digest, str) or len(digest) != 64 or
                any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("Invalid media SHA-256")
        keys = [("video", video), ("sha256", digest)]
        if pressure:
            keys.append(("parent", str(PurePosixPath(video.replace("\\", "/")).parent)))
        for key in keys:
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    members = defaultdict(list)
    for i in range(len(rows)):
        members[find(i)].append(i)
    identifiers = {k: fingerprint(sorted(rows[i]["sample_id"] for i in values))
                   for k, values in members.items()}
    return [identifiers[find(i)] for i in range(len(rows))]


def _balance_vector(row):
    conflict, natural = row["sample_type"] == "Conflict", _natural(row)
    return np.array([conflict, not conflict, natural and conflict,
                     natural and not conflict, row["protocol"] == "VT", row["protocol"] == "VA"],
                    dtype=np.float64)


def _assign_folds(rows, groups, seed, count):
    members = defaultdict(list)
    for i, group in enumerate(groups):
        members[group].append(i)
    if len(members) < count:
        raise ValueError("Too few independent content groups for the required folds")
    vectors = {g: np.sum([_balance_vector(rows[i]) for i in ix], axis=0)
               for g, ix in members.items()}
    totals = np.maximum(np.sum(list(vectors.values()), axis=0), 1.)
    keys = sorted(members, key=lambda g: (-float(np.max(vectors[g] / totals)),
        -len(members[g]), fingerprint([seed, g])))
    loads = np.zeros((count, len(totals)))
    sizes, assignment = np.zeros(count, dtype=int), {}
    for k, group in enumerate(keys):
        candidates = range(count) if k >= count else [k]

        def cost(fold):
            trial = loads.copy()
            trial[fold] += vectors[group]
            return (float(np.mean(np.var(trial / totals, axis=0))), int(sizes[fold]), fold)

        fold = min(candidates, key=cost)
        assignment[group] = fold
        loads[fold] += vectors[group]
        sizes[fold] += len(members[group])
    result = [assignment[g] for g in groups]
    for fold in range(count):
        subset = [r for r, f in zip(rows, result) if f == fold]
        if {r["sample_type"] for r in subset} != {"Conflict", "Aligned"}:
            raise ValueError("Grouping cannot provide both classes in every fold")
        if any(_natural(r) and r["sample_type"] == "Conflict" for r in rows) and not any(
                _natural(r) and r["sample_type"] == "Conflict" for r in subset):
            raise ValueError("Grouping lost natural conflicts in a development partition")
    return result


def group_design(rows, media, seed=42, pressure=False):
    groups = content_groups(rows, media, pressure)
    outer = _assign_folds(rows, groups, seed, 3)
    nested = []
    for fold in range(3):
        remaining = [i for i, f in enumerate(outer) if f != fold]
        inner = _assign_folds([rows[i] for i in remaining], [groups[i] for i in remaining],
                              seed + 100 + fold, 5)
        parts = {"fold": fold,
            "train": [i for i, f in zip(remaining, inner) if f != 0],
            "calibration": [i for i, f in zip(remaining, inner) if f == 0],
            "held": [i for i, f in enumerate(outer) if f == fold]}
        sets = [{groups[i] for i in parts[name]} for name in ("train", "calibration", "held")]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise RuntimeError("An independent content group crossed fit/calibration/held partitions")
        nested.append(parts)
    result = {"seed": seed, "pressure": pressure, "groups": groups, "outer": outer,
              "nested": nested, "independent_groups": len(set(groups)),
              "supervised_group_independence": True}
    result["hash"] = fingerprint(result)
    return result


def branch_report(rows, yes):
    _validate_rows(rows, require_labels=True)
    yes = np.asarray(yes)
    if yes.shape != (len(rows),) or yes.dtype != np.bool_:
        raise ValueError("Aligned boolean decisions are required")
    conflict = np.array([r["sample_type"] == "Conflict" for r in rows])
    natural_conflict = conflict & np.array([_natural(r) for r in rows])
    nc, na = int(conflict.sum()), int((~conflict).sum())
    tp, fp = int((yes & conflict).sum()), int((yes & ~conflict).sum())
    recall = tp / nc if nc else None
    specificity = (na - fp) / na if na else None
    return {"rows": len(rows), "conflicts": nc, "aligned": na, "true_positives": tp,
        "false_positives": fp, "conflict_recall": recall, "specificity": specificity,
        "task_component": .3 * recall + .2 * specificity if nc and na else None,
        "natural_conflicts": int(natural_conflict.sum()),
        "natural_true_positives": int((yes & natural_conflict).sum()),
        "natural_conflict_recall": float(yes[natural_conflict].mean()) if natural_conflict.any() else None}


def development_gate(rows, control_yes, new_yes, outer):
    if len(outer) != len(rows) or set(outer) != {0, 1, 2}:
        raise ValueError("Gate requires the three complete outer folds")
    control, new = branch_report(rows, control_yes), branch_report(rows, new_yes)
    folds = []
    for fold in range(3):
        ix = np.array([i for i, f in enumerate(outer) if f == fold], dtype=int)
        a = branch_report([rows[i] for i in ix], np.asarray(control_yes)[ix])
        b = branch_report([rows[i] for i in ix], np.asarray(new_yes)[ix])
        if a["task_component"] is None or b["task_component"] is None:
            raise ValueError("Every gate fold needs both classes")
        folds.append({"fold": fold, "control": a, "candidate": b,
                      "gain": b["task_component"] - a["task_component"]})
    gain = new["task_component"] - control["task_component"]
    natural_ok = (control["natural_conflict_recall"] is not None and
                  new["natural_conflict_recall"] >= control["natural_conflict_recall"] - 1e-12)
    checks = {"pooled_task_component_improves": gain > 1e-12,
        "at_least_two_folds_do_not_regress": sum(f["gain"] >= -1e-12 for f in folds) >= 2,
        "natural_conflict_recall_does_not_regress": bool(natural_ok)}
    conflict = np.array([r["sample_type"] == "Conflict" for r in rows])
    before, after = np.asarray(control_yes), np.asarray(new_yes)
    changes = {"conflicts_corrected": int((conflict & ~before & after).sum()),
        "conflicts_lost": int((conflict & before & ~after).sum()),
        "false_positives_added": int((~conflict & ~before & after).sum()),
        "false_positives_removed": int((~conflict & before & ~after).sum())}
    return {"passed": all(checks.values()), "checks": checks, "gain": gain,
            "control": control, "candidate": new, "folds": folds, "changes": changes, "note": NOTE}


def _fit_logistic_saved(root, c, features, rows, job):
    from rl.evidence_training import train_only
    train_only(c, rows)
    signature = fingerprint([c["identity_hash"], CONFIG, FEATURES,
        [(r["sample_id"], r["sample_type"], r["protocol"]) for r in rows],
        np.asarray(features).tolist()])
    path = Path(root) / "calibrators" / (job + ".json")
    if path.exists():
        value = checked_artifact(path)
        if value.get("signature") != signature or value["solver"].get("success") is not True:
            raise ValueError("Completed logistic fit identity changed")
        predict_logistic(features[:1], value)
        return value
    value = fit_logistic(features, rows, lambda **v: emit(root, "fitting_logistic", job=job, **v))
    value.update(algorithm=VERSION, source_identity=c["identity_hash"], signature=signature)
    publish_artifact(path, value)
    return value


def _fit_control_saved(root, c, q, rows, job):
    from rl.evidence_bottleneck import fit_calibration
    from rl.evidence_training import train_only
    train_only(c, rows)
    signature = fingerprint([c["identity_hash"], "fixed_c1_control", CONFIG,
        [(r["sample_id"], r["sample_type"], r["protocol"]) for r in rows], np.asarray(q).tolist()])
    path = Path(root) / "calibrators" / (job + "-control.json")
    if path.exists():
        value = checked_artifact(path)
        if value.get("signature") != signature or value["solver"].get("success") is not True:
            raise ValueError("Completed control fit identity changed")
        return value
    value = fit_calibration(q, rows, c["settings"],
        lambda **v: emit(root, "fitting_control", job=job, **v))
    value.update(algorithm=VERSION, source_identity=c["identity_hash"], signature=signature)
    publish_artifact(path, value)
    return value


def run_development(root, c, bank, design, name):
    """Fit only nested TRAIN/cal rows; write one group-held-out branch result."""
    from rl.evidence_bottleneck import decision_scores
    from rl.evidence_training import fit_evidence, predict_evidence
    path = Path(root) / ("development-" + name + ".json")
    if path.exists():
        value = checked_artifact(path)
        if value["source_identity"] != c["identity_hash"] or value["design_hash"] != design["hash"]:
            raise ValueError("Development identity changed")
        return value
    rows = c["train"]
    new_p, control_p = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    vision, second = np.full((len(rows), 3), np.nan), np.full((len(rows), 3), np.nan)
    assignments = np.zeros(len(rows), dtype=int)
    partitions = []
    for part in design["nested"]:
        train, cal, held = ([rows[i] for i in part[k]] for k in ("train", "calibration", "held"))
        job = f"{name}-nested-fold{part['fold']}"
        heads = fit_evidence(Path(root) / "producer-fits", c, bank, train, job,
            lambda phase, **v: emit(root, phase, **v))
        calibration = predict_evidence(bank, cal, heads)
        logistic = _fit_logistic_saved(root, c,
            six_features(calibration["vision"], calibration["second"], cal), cal, job)
        control = _fit_control_saved(root, c, calibration["q"], cal, job)
        evidence = predict_evidence(bank, held, heads)
        ix = part["held"]
        vision[ix], second[ix] = evidence["vision"], evidence["second"]
        new_p[ix] = predict_logistic(six_features(evidence["vision"], evidence["second"], held), logistic)
        control_p[ix] = decision_scores(evidence["q"], held, control)
        assignments[ix] += 1
        partitions.append({"fold": part["fold"], "training_ids": [r["sample_id"] for r in train],
            "calibration_ids": [r["sample_id"] for r in cal], "held_ids": [r["sample_id"] for r in held],
            "rows": {"train": len(train), "calibration": len(cal), "held": len(held)},
            "calibration_fraction_of_remaining": len(cal) / (len(train) + len(cal)),
            "logistic_signature": logistic["signature"], "control_signature": control["signature"],
            "clear_supervision_counts": {mode: len(head["examples"]) for mode, head in heads.items()},
            "supervised_group_independence": True})
        emit(root, "development_outer_fold_complete", design=name, fold=part["fold"])
    if not (assignments == 1).all() or not all(np.isfinite(v).all() for v in (new_p, control_p, vision, second)):
        raise RuntimeError("Missing, duplicate, or nonfinite outer-fold predictions")
    new_yes = disagreement_gate(vision, second, new_p)
    control_yes = disagreement_gate(vision, second, control_p)
    value = {"algorithm": VERSION, "source_identity": c["identity_hash"],
        "design_hash": design["hash"], "gate": development_gate(rows, control_yes, new_yes, design["outer"]),
        "partitions": partitions, "ids": [r["sample_id"] for r in rows],
        "new_probabilities": new_p.tolist(), "control_probabilities": control_p.tolist(),
        "vision": vision.tolist(), "second": second.tolist(),
        "new_yes": new_yes.tolist(), "control_yes": control_yes.tolist(),
        "supervised_group_independence": True, "note": NOTE}
    publish_artifact(path, value)
    return value


def final_candidate(root, source_run, c, bank, design):
    """Use original TRAIN OOF probabilities, then lock a model before VAL."""
    from rl.evidence_training import fit_evidence, predict_evidence
    from rl.evidence_store import FeatureBank
    from rl.evidence_bottleneck import decision_scores
    from rl.metrics import evaluation_report
    rows = c["train"]
    candidate_path = Path(root) / "candidate.json"
    if candidate_path.exists():
        candidate = checked_artifact(candidate_path)
        if candidate["source_identity"] != c["identity_hash"] or candidate["design_hash"] != design["hash"]:
            raise ValueError("Final candidate identity changed")
    else:
        vision, second = np.full((len(rows), 3), np.nan), np.full((len(rows), 3), np.nan)
        q, coverage = np.full(len(rows), np.nan), np.zeros(len(rows), dtype=int)
        for fold in range(3):
            train_ix = [i for i, f in enumerate(design["outer"]) if f != fold]
            held_ix = [i for i, f in enumerate(design["outer"]) if f == fold]
            heads = fit_evidence(Path(root) / "producer-fits", c, bank, [rows[i] for i in train_ix],
                f"final-oof-fold{fold}", lambda phase, **v: emit(root, phase, **v))
            evidence = predict_evidence(bank, [rows[i] for i in held_ix], heads)
            vision[held_ix], second[held_ix], q[held_ix] = evidence["vision"], evidence["second"], evidence["q"]
            coverage[held_ix] += 1
            emit(root, "final_calibration_oof_complete", fold=fold)
        if not (coverage == 1).all() or not all(np.isfinite(v).all() for v in (vision, second, q)):
            raise RuntimeError("Final calibration requires one independent evidence prediction per training row")
        publish_artifact(Path(root) / "final-calibration-oof.json", {
            "source_identity": c["identity_hash"], "design_hash": design["hash"],
            "ids": [r["sample_id"] for r in rows], "vision": vision.tolist(), "second": second.tolist(),
            "q": q.tolist(), "supervised_group_independence": True})
        logistic = _fit_logistic_saved(root, c, six_features(vision, second, rows), rows, "final")
        control = _fit_control_saved(root, c, q, rows, "final")
        heads = fit_evidence(Path(root) / "producer-fits", c, bank, rows, "final",
            lambda phase, **v: emit(root, phase, **v))
        candidate = {"algorithm": VERSION, "source_identity": c["identity_hash"],
            "design_hash": design["hash"], "config": CONFIG, "logistic": logistic,
            "control": control, "heads": heads, "locked_before_validation": True, "note": NOTE}
        publish_artifact(candidate_path, candidate)
    val_path = Path(root) / "validation-report.json"
    if val_path.exists():
        result = checked_artifact(val_path)
        if result["candidate_sha256"] != file_hash(candidate_path):
            raise ValueError("Validation report candidate identity changed")
        return result
    if "data_root" in c:
        from rl.grounded_uncertainty import load_training
        raw_val = Path(c["data_root"]) / "val.jsonl"
        if (file_hash(raw_val) != c["identity"]["raw_split_hashes"]["val"] or
                load_training(raw_val) != c["val"]):
            raise ValueError("Original validation labels or inputs changed before locked-model reporting")
    # This is the only VAL FeatureBank call, after immutable model locking.
    val, val_bank = c["val"], FeatureBank(source_run, c, "val")
    evidence = predict_evidence(val_bank, val, candidate["heads"])
    p = predict_logistic(six_features(evidence["vision"], evidence["second"], val), candidate["logistic"])
    cp = decision_scores(evidence["q"], val, candidate["control"])
    added, c_added = (disagreement_gate(evidence["vision"], evidence["second"], score) for score in (p, cp))
    old_yes = np.array([c["baselines"]["val"][r["sample_id"]]["is_uncertain"] == "yes" for r in val])

    def full_report(yes):
        records = [{**r, "pred_emotion": c["baselines"]["val"][r["sample_id"]]["pred_emotion"],
                    "is_uncertain": "yes" if value else "no"} for r, value in zip(val, yes)]
        return evaluation_report(records)

    result = {"algorithm": VERSION, "candidate_sha256": file_hash(candidate_path),
        "candidate_locked_before_validation": True, "validation_used_for_selection": False,
        "baseline": full_report(old_yes), "control_fused": full_report(old_yes | c_added),
        "candidate_fused": full_report(old_yes | added),
        "candidate_branch": branch_report(val, added), "control_branch": branch_report(val, c_added),
        "ids": [r["sample_id"] for r in val], "probabilities": p.tolist(), "control_probabilities": cp.tolist(),
        "added_yes": [r["sample_id"] for r, add, old in zip(val, added, old_yes) if add and not old],
        "baseline_reference": "historical_v4_validation_predictions; not the newer public submission",
        "output_checks_pending": True, "note": NOTE}
    publish_artifact(val_path, result)
    return result


def run(args):
    from rl.evidence_store import context_for, FeatureBank
    from rl.grounded_uncertainty import load_training
    source, root = Path(args.source_run).resolve(), Path(args.output_dir).resolve()
    if root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Calibrator output must be separate from the immutable producer run")
    with run_lock(root):
        try:
            emit(root, "verifying_producer")
            c = context_for(source)
            if c["settings"] != c["identity"]["settings"]:
                raise ValueError("Frozen evidence settings changed")
            raw_train = Path(c["data_root"]) / "train.jsonl"
            if (file_hash(raw_train) != c["identity"]["raw_split_hashes"]["train"] or
                    load_training(raw_train) != c["train"]):
                raise ValueError("Training rows differ from the original supervised data")
            rows, media = c["train"], c["identity"]["media"]
            design = group_design(rows, media)
            try:
                pressure = group_design(rows, media, pressure=True)
                pressure_error = None
            except ValueError as exc:
                pressure, pressure_error = None, str(exc)
            identity = {"algorithm": VERSION, "config": CONFIG,
                "source_identity": c["identity_hash"], "source_context_sha256": file_hash(source / "context.json"),
                "train_features_manifest_sha256": file_hash(source / "features/train/complete.json"),
                "training_rows_hash": fingerprint(rows), "design_hash": design["hash"],
                "pressure_design_hash": pressure["hash"] if pressure else None,
                "pressure_design_error": pressure_error,
                "script_source_hash": fingerprint(Path(__file__).read_text(encoding="utf-8")),
                "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}}
            immutable_json(root / "identity.json", {"identity": identity, "identity_hash": fingerprint(identity)})
            immutable_json(root / "folds.json", {"primary": design, "pressure": pressure,
                "pressure_design_error": pressure_error})
            bank = FeatureBank(source, c, "train")
            development = run_development(root, c, bank, design, "media")
            if pressure is not None:
                pressure_result = run_development(root, c, bank, pressure, "parent-pressure")
                pressure_summary = {"status": "complete", "gate": pressure_result["gate"],
                    "supervised_group_independence": True, "used_for_model_selection": False}
            else:
                pressure_summary = {"status": "not_evaluable", "reason": pressure_error,
                    "used_for_model_selection": False, "supervised_group_independence": None}
            validation = final_candidate(root, source, c, bank, design) if development["gate"]["passed"] else None
            result = {"algorithm": VERSION, "identity_hash": fingerprint(identity), "status": "complete",
                "decision": "candidate_locked" if validation is not None else "primary_gate_not_met",
                "primary_gate": development["gate"], "parent_pressure": pressure_summary,
                "candidate_locked": validation is not None, "validation_reported": validation is not None,
                "test_inference_performed": False, "submitted_to_kaggle": False, "note": NOTE}
            publish_artifact(root / "final-report.json", result)
            emit(root, "complete", gate_passed=development["gate"]["passed"], validation_reported=validation is not None)
            return result
        except Exception as exc:
            write_json(root / "error.json", {"type": type(exc).__name__, "message": str(exc),
                                            "traceback": traceback.format_exc()})
            emit(root, "failed", error=str(exc))
            raise


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source-run", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    return value


def main(argv=None):
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    main()
