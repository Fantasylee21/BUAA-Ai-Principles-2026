"""Recover only the frozen visual experiment's CPU optimizer iteration budget.

The original feature producer, GPU extractor, objectives, folds and promotion
gate remain immutable. Both comparison arms restart their residual heads and
calibrators from zero with a common 3000-iteration ceiling. No GPU, validation,
test inference or Kaggle submission is performed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import traceback

from rl.evidence_store import FeatureBank
from rl.experiment import file_hash, fingerprint, write_json
from scripts import vision_expert as cache
from scripts import vision_expert_cpu as original
from scripts.train_evidence_calibrator import (emit, group_design,
    immutable_json, publish_artifact, run_lock)

VERSION = "qwen3vl8b_visual_expert_cpu_recovery_v1"
MAX_ITERATIONS = 3000
NOTE = ("Only the CPU convex solver iteration ceiling is recovered from 300 to 3000. "
    "Frozen GPU features are read in place. Both arms and all residual heads and "
    "calibrators restart from zero with unchanged objectives, tolerances, splits "
    "and promotion gate. This is not new backbone training or a Kaggle result.")


def recovery_settings(c):
    """Never mutate the producer context checked by the old feature loader."""
    if c["settings"] != c["identity"]["settings"] or c["settings"]["max_iterations"] != 300:
        raise ValueError("Recovery requires the original immutable 300-iteration producer")
    settings = deepcopy(c["settings"])
    settings["max_iterations"] = MAX_ITERATIONS
    return settings


def optimizer_policy(c):
    settings = recovery_settings(c)
    return {"optimizer": "L-BFGS-B", "old_max_iterations": 300,
        "max_iterations": MAX_ITERATIONS, "max_line_search": 40, "max_corrections": 20,
        "gradient_tolerance": settings["gradient_tolerance"],
        "function_tolerance": settings["function_tolerance"],
        "regularization": settings["regularization"],
        "layer_norm_epsilon": settings["layer_norm_epsilon"], "initialization": "zeros",
        "acceptance": "unchanged_original_solve_requires_scipy_success_and_finite_objective_gradient",
        "settings": settings}


def fit_context(c, evaluation_identity, variant):
    if variant not in ("control", "expert"):
        raise ValueError("Unknown comparison arm")
    settings = recovery_settings(c)
    identity = {"algorithm": VERSION, "evaluation_identity": evaluation_identity,
        "old_feature_producer": c["identity_hash"], "variant": variant, "fit_settings": settings}
    return {"train": c["train"], "settings": settings, "identity": identity,
            "identity_hash": fingerprint(identity)}


def checked_failure(cache_root):
    path = Path(cache_root) / "evaluation/error.json"
    content = path.read_bytes()
    failure = json.loads(content)
    prefix = "L-BFGS did not converge: "
    if failure.get("type") != "RuntimeError" or not failure.get("message", "").startswith(prefix):
        raise ValueError("Recovery requires the preserved original L-BFGS failure")
    diagnostics = json.loads(failure["message"][len(prefix):])
    if (diagnostics.get("success") is not False or diagnostics.get("iterations") != 300 or
            not all(isinstance(diagnostics.get(k), (int, float)) and
                    math.isfinite(diagnostics[k]) for k in ("loss", "data_loss", "gradient_max", "initial_loss"))):
        raise ValueError("Original failure is not the finite 300-iteration exhaustion")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(content).hexdigest(), "diagnostics": diagnostics}


def checked_cache_identity(source, cache_root, c):
    source, cache_root = Path(source), Path(cache_root)
    gpu = cache.read(cache_root / "identity.json")
    tasks = cache.video_tasks(c)
    raw = gpu["identity"]
    if (gpu["identity_hash"] != fingerprint(raw) or raw["algorithm"] != cache.VERSION or
            raw["source_producer_identity"] != c["identity_hash"] or
            raw["source_context_sha256"] != file_hash(source / "context.json") or
            raw["training_tasks_hash"] != fingerprint(tasks) or
            raw["extractor_source_hash"] != fingerprint(Path(cache.__file__).read_text(encoding="utf-8"))):
        raise ValueError("Frozen visual extraction identity changed")
    return gpu, tasks


def verified_inputs(source, cache_root):
    c = cache.checked_source(source)
    # Strict identity/source/media checks run on the unchanged producer before
    # constructing any fit-only contexts with the larger iteration ceiling.
    recovery_settings(c)
    gpu, tasks = checked_cache_identity(source, cache_root, c)
    failure = checked_failure(cache_root)
    control = FeatureBank(source, c, "train")
    visual = cache.VisionBank(cache_root, gpu, tasks)
    hybrid = cache.HybridBank(control, visual)
    manifests = {"omni_train": Path(source) / "features/train/complete.json",
                 "visual_train": Path(cache_root) / "features/train/complete.json"}
    provenance = {"source_producer_identity": c["identity_hash"],
        "source_context_sha256": file_hash(Path(source) / "context.json"),
        "visual_cache_identity": gpu["identity_hash"],
        "feature_manifest_sha256": {name: file_hash(path) for name, path in manifests.items()},
        # Each complete manifest binds every per-feature archive SHA; both
        # banks above have just checked every feature and its embedded identity.
        "complete_feature_manifests_hash": fingerprint({name: cache.read(path) for name, path in manifests.items()}),
        "frozen_model_sha256": cache.read(Path(cache_root) / "model-frozen.json")["frozen_sha256"],
        "original_failure": failure, "feature_arrays_copied": False,
        "training_videos": len(tasks), "training_rows": len(c["train"])}
    return c, control, hybrid, provenance


def evaluate_recovery(source_run, cache_root, output_dir):
    source, frozen_cache, output = (Path(p).resolve() for p in (source_run, cache_root, output_dir))
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (source, frozen_cache)):
        raise ValueError("Recovery output must be separate from both immutable feature runs")
    if source.is_relative_to(frozen_cache) or frozen_cache.is_relative_to(source):
        raise ValueError("The two feature producers must be separate")
    root = output / "evaluation"
    # This process uses NumPy/SciPy only; prevent accidental CUDA discovery in
    # optional transitive runtime imports as well.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    with run_lock(root):
        try:
            write_json(output / "pipeline.status.json", {"phase": "running", "algorithm": VERSION,
                "stage": "cpu_optimizer_recovery", "gpu_used": False})
            emit(root, "verifying_frozen_training_features")
            c, control_bank, expert_bank, provenance = verified_inputs(source, frozen_cache)
            design = group_design(c["train"], c["identity"]["media"])
            try:
                pressure = group_design(c["train"], c["identity"]["media"], pressure=True)
                pressure_error = None
            except ValueError as exc:
                pressure, pressure_error = None, str(exc)
            identity = {"algorithm": VERSION, "recovery_scope": "cpu_iteration_ceiling_only",
                "provenance": provenance, "solver_policy": optimizer_policy(c),
                "comparison_config": deepcopy(original.CONFIG),
                "primary_design_hash": design["hash"],
                "pressure_design_hash": pressure["hash"] if pressure else None,
                "pressure_design_error": pressure_error,
                "recovery_source_hash": fingerprint(Path(__file__).read_text(encoding="utf-8")),
                "original_cpu_source_hash": fingerprint(Path(original.__file__).read_text(encoding="utf-8")),
                "grouping_source_hash": fingerprint(Path(__file__).with_name("train_evidence_calibrator.py").read_text(encoding="utf-8")),
                "software": {**cache._software(), "scipy": importlib.metadata.version("scipy")}}
            identity_hash = fingerprint(identity)
            immutable_json(root / "identity.json", {"identity": identity, "identity_hash": identity_hash})
            immutable_json(root / "folds.json", {"primary": design, "parent_pressure": pressure,
                                                 "pressure_design_error": pressure_error})
            contexts = {k: fit_context(c, identity_hash, k) for k in ("control", "expert")}
            primary = original.development(root, c, control_bank, expert_bank, contexts, design, "primary")
            stressed = original.development(root, c, control_bank, expert_bank, contexts, pressure, "parent-pressure") if pressure else None
            gate = original.comparison_gate(primary, stressed)
            # The failure and complete feature publications are immutable
            # inputs; do not publish success if they changed during fitting.
            if file_hash(provenance["original_failure"]["path"]) != provenance["original_failure"]["sha256"]:
                raise ValueError("Preserved original CPU failure changed during recovery")
            for name, path in (("omni_train", source / "features/train/complete.json"),
                               ("visual_train", frozen_cache / "features/train/complete.json")):
                if file_hash(path) != provenance["feature_manifest_sha256"][name]:
                    raise ValueError("Frozen feature publication changed during recovery")
            result = {"algorithm": VERSION, "identity_hash": identity_hash, "status": "complete",
                "gate": gate, "primary": {k: primary[k] for k in ("control", "expert", "gain", "folds")},
                "parent_pressure": {k: stressed[k] for k in ("control", "expert", "gain", "folds")} if stressed else None,
                "pressure_design_error": pressure_error, "validation_used": False, "test_used": False,
                "gpu_used": False, "omni_model_weights_changed": False, "qwen_model_weights_changed": False,
                "residual_heads_trained": True, "complete_model_or_kaggle_promotion_claimed": False,
                "submitted_to_kaggle": False, "recovery_scope": "cpu_iteration_ceiling_only",
                "solver_policy": identity["solver_policy"], "provenance": provenance,
                "note": NOTE + " " + original.NOTE}
            publish_artifact(root / "final-report.json", result)
            write_json(output / "pipeline.status.json", {"phase": "complete", "algorithm": VERSION,
                "identity_hash": identity_hash, "gate_passed": gate["passed"], "gpu_used": False,
                "final_report_sha256": file_hash(root / "final-report.json")})
            emit(root, "complete", gate_passed=gate["passed"], validation_used=False, test_used=False)
            return result
        except Exception as exc:
            write_json(root / "error.json", {"type": type(exc).__name__, "message": str(exc),
                "traceback": traceback.format_exc()})
            write_json(output / "pipeline.status.json", {"phase": "failed", "algorithm": VERSION,
                "reason": str(exc), "gpu_used": False})
            emit(root, "failed", error=str(exc))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "cache-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    return evaluate_recovery(args.source_run, args.cache_root, args.output_dir)


if __name__ == "__main__":
    main()
