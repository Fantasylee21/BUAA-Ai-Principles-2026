"""Independent TRAIN-only comparison of Omni and Qwen3 visual evidence.

Called by ``python -m scripts.vision_expert evaluate ...`` after full extraction.
No model loading, GPU use, validation/test inference, or Kaggle submission.
"""
from __future__ import annotations

from pathlib import Path
import traceback

import numpy as np

from rl.evidence_bottleneck import (decision_scores, fit_calibration, fit_head,
                                    view_examples)
from rl.evidence_store import FeatureBank, saved_fit, save_fit
from rl.evidence_training import fit_evidence, predict_evidence, train_only
from rl.experiment import file_hash, fingerprint, write_json
from scripts import vision_expert as cache
from scripts.train_evidence_calibrator import (branch_report, checked_artifact,
    disagreement_gate, emit, group_design, immutable_json, publish_artifact, run_lock)

VERSION = "qwen3vl8b_visual_expert_evaluation_v1"
CONFIG = {"algorithm": VERSION, "seed": 42, "outer_folds": 3, "inner_folds": 5,
    "calibration_fold": 0, "threshold": .5, "regularization": .01,
    "minimum_primary_gain": .01, "natural_recall_must_not_regress": True,
    "parent_pressure_must_not_regress": True,
    "calibration": "original_q_log_odds_plus_protocol_offsets",
    "decision": "argmax_diff_and_calibrated_probability_gt_half",
    "second_modality_heads": "shared_control_heads_fitted_only_on_current_training_partition"}
NOTE = ("Independent group-held-out TRAIN evidence-branch comparison, not a complete "
    "Kaggle metric or an estimate of public leaderboard improvement. No v4 yes/emotion "
    "outputs, validation labels or test samples enter fitting or selection. Only the "
    "visual expert and its fold-fitted residual head change; text/audio features and "
    "their fold-fitted residual heads are shared. Video preprocessing also differs, "
    "so this is not a representation-only ablation.")


def comparison_gate(primary, pressure):
    a, b = primary["control"], primary["expert"]
    gain = b["task_component"] - a["task_component"]
    pressure_gain = None if pressure is None else pressure["expert"]["task_component"] - pressure["control"]["task_component"]
    checks = {"primary_gain_at_least_0_01": gain >= .01 - 1e-12,
        "primary_natural_recall_does_not_regress": a["natural_conflict_recall"] is not None and
            b["natural_conflict_recall"] is not None and
            b["natural_conflict_recall"] >= a["natural_conflict_recall"] - 1e-12,
        "parent_pressure_does_not_regress": pressure_gain is not None and pressure_gain >= -1e-12}
    return {"passed": all(checks.values()), "checks": checks, "primary_gain": gain,
            "parent_pressure_gain": pressure_gain, "note": NOTE}


def fit_context(c, evaluation_identity, variant):
    if variant not in ("control", "expert"):
        raise ValueError("Unknown evidence producer variant")
    identity = {"algorithm": VERSION, "evaluation_identity": evaluation_identity,
                "old_feature_producer": c["identity_hash"], "variant": variant}
    # The original helper validates TRAIN row equality. Bind its saved-fit
    # signature to this new feature producer, not merely the old sample IDs.
    return {"train": c["train"], "settings": c["settings"],
            "identity": identity, "identity_hash": fingerprint(identity)}


def fit_expert_vision(root, c, bank, rows, job):
    """Use the original generic residual-head objective with the new width."""
    train_only(c, rows)
    if bank.split != "train":
        raise ValueError("Expert supervision requires TRAIN-only features")
    examples = view_examples(rows, "vision")
    if not examples:
        raise ValueError("No clear visual supervision in this training partition")
    selected, labels = zip(*examples)
    signature = fingerprint([c["identity_hash"], "vision", [(r["sample_id"], y) for r, y in examples]])
    path = Path(root) / "producer-fits/expert/heads" / job / "vision.json"
    result = saved_fit(path, c, signature)
    if result is None:
        hidden, native = bank.get(selected, "vision")
        result = fit_head(hidden, native, labels, c["settings"],
            lambda **v: emit(root, "fitting_expert_vision", job=job, covered=len(examples), **v))
        result.update(mode="vision", identity_hash=c["identity_hash"], signature=signature,
                      examples=[r["sample_id"] for r in selected], labels=list(labels))
        save_fit(path, c, signature, result)
    return result


def calibration(root, c, q, rows, job, variant):
    train_only(c, rows)
    signature = fingerprint([c["identity_hash"], CONFIG,
        [(r["sample_id"], r["sample_type"], r["protocol"]) for r in rows], np.asarray(q).tolist()])
    path = Path(root) / "calibration" / (job + "-" + variant + ".json")
    if path.exists():
        result = checked_artifact(path)
        if result.get("signature") != signature or result["solver"].get("success") is not True:
            raise ValueError("Completed visual-expert calibration identity changed")
        return result
    result = fit_calibration(q, rows, c["settings"],
        lambda **v: emit(root, "fitting_conflict_calibration", job=job, variant=variant, **v))
    result.update(signature=signature, identity_hash=c["identity_hash"],
                  examples=[r["sample_id"] for r in rows])
    publish_artifact(path, result)
    return result


def _stratified_reports(rows, yes):
    result = {"by_protocol": {}, "by_natural": {}}
    for protocol in ("VT", "VA"):
        ix = [i for i, r in enumerate(rows) if r["protocol"] == protocol]
        if ix:
            result["by_protocol"][protocol] = branch_report([rows[i] for i in ix], yes[ix])
    for natural, label in ((True, "natural"), (False, "synthetic")):
        ix = [i for i, r in enumerate(rows) if r["natural"] == natural]
        if ix:
            result["by_natural"][label] = branch_report([rows[i] for i in ix], yes[ix])
    return result


def development(root, c, control_bank, expert_bank, contexts, design, name):
    path = Path(root) / ("development-" + name + ".json")
    identity = fingerprint([contexts["control"]["identity_hash"], contexts["expert"]["identity_hash"], design["hash"]])
    if path.exists():
        result = checked_artifact(path)
        if result.get("identity_hash") != identity:
            raise ValueError("Completed visual-expert development identity changed")
        return result
    rows = c["train"]
    vision = {k: np.full((len(rows), 3), np.nan) for k in ("control", "expert")}
    second = np.full((len(rows), 3), np.nan)
    scores = {k: np.full(len(rows), np.nan) for k in ("control", "expert")}
    coverage, partitions = np.zeros(len(rows), dtype=int), []
    for part in design["nested"]:
        train, cal, held = ([rows[i] for i in part[k]] for k in ("train", "calibration", "held"))
        job = f"{name}-fold{part['fold']}"
        control_heads = fit_evidence(Path(root) / "producer-fits/control", contexts["control"],
            control_bank, train, job, lambda phase, **v: emit(root, phase, variant="control", **v))
        expert_heads = {**control_heads, "vision": fit_expert_vision(root, contexts["expert"], expert_bank, train, job)}
        cal_evidence = {"control": predict_evidence(control_bank, cal, control_heads),
                        "expert": predict_evidence(expert_bank, cal, expert_heads)}
        held_evidence = {"control": predict_evidence(control_bank, held, control_heads),
                         "expert": predict_evidence(expert_bank, held, expert_heads)}
        if (not np.array_equal(cal_evidence["control"]["second"], cal_evidence["expert"]["second"]) or
                not np.array_equal(held_evidence["control"]["second"], held_evidence["expert"]["second"])):
            raise RuntimeError("The visual replacement changed the shared second-modality branch")
        for variant in ("control", "expert"):
            bias = calibration(root, contexts[variant], cal_evidence[variant]["q"], cal, job, variant)
            evidence = held_evidence[variant]
            scores[variant][part["held"]] = decision_scores(evidence["q"], held, bias)
            vision[variant][part["held"]] = evidence["vision"]
        second[part["held"]] = held_evidence["control"]["second"]
        coverage[part["held"]] += 1
        counts = {mode: len(head["examples"]) for mode, head in control_heads.items()}
        partitions.append({"fold": part["fold"], "training_ids": [r["sample_id"] for r in train],
            "calibration_ids": [r["sample_id"] for r in cal], "held_ids": [r["sample_id"] for r in held],
            "clear_supervision_counts": counts, "shared_text_audio_heads": True,
            "supervised_group_independence": True})
        emit(root, "outer_fold_complete", design=name, fold=part["fold"])
    if not (coverage == 1).all() or not all(np.isfinite(v).all() for v in (*vision.values(), *scores.values(), second)):
        raise RuntimeError("Incomplete or duplicate visual-expert outer-fold predictions")
    yes = {variant: disagreement_gate(vision[variant], second, scores[variant]) for variant in ("control", "expert")}
    metrics = {variant: branch_report(rows, yes[variant]) for variant in yes}
    folds = []
    for fold in range(3):
        ix = [i for i, f in enumerate(design["outer"]) if f == fold]
        per_fold = {k: branch_report([rows[i] for i in ix], yes[k][ix]) for k in yes}
        folds.append({"fold": fold, **per_fold,
                      "gain": per_fold["expert"]["task_component"] - per_fold["control"]["task_component"]})
    result = {"algorithm": VERSION, "identity_hash": identity, "design_hash": design["hash"],
        **metrics, "gain": metrics["expert"]["task_component"] - metrics["control"]["task_component"],
        "folds": folds, "partitions": partitions, "ids": [r["sample_id"] for r in rows],
        "stratified": {k: _stratified_reports(rows, yes[k]) for k in yes},
        "vision": {k: value.tolist() for k, value in vision.items()}, "second": second.tolist(),
        "probabilities": {k: value.tolist() for k, value in scores.items()},
        "is_uncertain": {k: value.tolist() for k, value in yes.items()},
        "supervised_group_independence": True, "note": NOTE}
    publish_artifact(path, result)
    return result


def evaluate_run(args):
    source, cache_root = Path(args.source_run).resolve(), Path(args.output_dir).resolve()
    root = cache_root / "evaluation"
    if cache_root.is_relative_to(source) or source.is_relative_to(cache_root):
        raise ValueError("Expert run must be separate from the old evidence producer")
    with run_lock(root):
        try:
            emit(root, "verifying_frozen_training_features")
            c = cache.checked_source(source)
            gpu = cache.read(cache_root / "identity.json")
            tasks = cache.video_tasks(c)
            if (gpu["identity_hash"] != fingerprint(gpu["identity"]) or
                    gpu["identity"]["algorithm"] != cache.VERSION or
                    gpu["identity"]["source_producer_identity"] != c["identity_hash"] or
                    gpu["identity"]["source_context_sha256"] != file_hash(source / "context.json") or
                    gpu["identity"]["training_tasks_hash"] != fingerprint(tasks) or
                    gpu["identity"]["extractor_source_hash"] != fingerprint(Path(cache.__file__).read_text(encoding="utf-8"))):
                raise ValueError("Frozen visual extraction identity changed")
            control_bank = FeatureBank(source, c, "train")
            visual_bank = cache.VisionBank(cache_root, gpu, tasks)
            expert_bank = cache.HybridBank(control_bank, visual_bank)
            design = group_design(c["train"], c["identity"]["media"])
            try:
                pressure = group_design(c["train"], c["identity"]["media"], pressure=True)
                pressure_error = None
            except ValueError as exc:
                pressure, pressure_error = None, str(exc)
            identity = {"algorithm": VERSION, "config": CONFIG, "old_producer_identity": c["identity_hash"],
                "visual_cache_identity": gpu["identity_hash"],
                "visual_complete_sha256": file_hash(cache_root / "features/train/complete.json"),
                "old_train_complete_sha256": file_hash(source / "features/train/complete.json"),
                "frozen_model_sha256": cache.read(cache_root / "model-frozen.json")["frozen_sha256"],
                "primary_design_hash": design["hash"], "pressure_design_hash": pressure["hash"] if pressure else None,
                "pressure_design_error": pressure_error,
                "cpu_source_hash": fingerprint(Path(__file__).read_text(encoding="utf-8")),
                "grouping_source_hash": fingerprint(Path(__file__).with_name("train_evidence_calibrator.py").read_text(encoding="utf-8")),
                "software": cache._software()}
            identity_hash = fingerprint(identity)
            immutable_json(root / "identity.json", {"identity": identity, "identity_hash": identity_hash})
            immutable_json(root / "folds.json", {"primary": design, "parent_pressure": pressure,
                                                 "pressure_design_error": pressure_error})
            contexts = {k: fit_context(c, identity_hash, k) for k in ("control", "expert")}
            primary = development(root, c, control_bank, expert_bank, contexts, design, "primary")
            stressed = development(root, c, control_bank, expert_bank, contexts, pressure, "parent-pressure") if pressure else None
            gate = comparison_gate(primary, stressed)
            result = {"algorithm": VERSION, "identity_hash": identity_hash, "status": "complete",
                "gate": gate, "primary": {k: primary[k] for k in ("control", "expert", "gain", "folds")},
                "parent_pressure": {k: stressed[k] for k in ("control", "expert", "gain", "folds")} if stressed else None,
                "pressure_design_error": pressure_error, "validation_used": False, "test_used": False,
                "omni_model_weights_changed": False, "qwen_model_weights_changed": False,
                "residual_heads_trained": True, "complete_model_or_kaggle_promotion_claimed": False,
                "next_step": "Eligible for a separately locked final fit and one validation report" if gate["passed"] else "Do not promote the visual replacement",
                "note": NOTE}
            publish_artifact(root / "final-report.json", result)
            emit(root, "complete", gate_passed=gate["passed"], primary_gain=gate["primary_gain"],
                 pressure_gain=gate["parent_pressure_gain"], validation_used=False, test_used=False)
            return result
        except Exception as exc:
            write_json(root / "error.json", {"type": type(exc).__name__, "message": str(exc),
                                            "traceback": traceback.format_exc()})
            emit(root, "failed", error=str(exc))
            raise
