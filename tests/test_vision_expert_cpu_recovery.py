"""CPU recovery contracts; synthetic arrays do not establish model quality."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rl.evidence_bottleneck import DEFAULTS, fit_head, fit_calibration
from rl.experiment import file_hash, fingerprint
from scripts import vision_expert_cpu_recovery as app


def context(videos=45):
    rows, media = [], {}
    for i in range(videos):
        video = f"videos/parent-{i // 3}/clip-{i}.mp4"
        media[video] = hashlib.sha256(video.encode()).hexdigest()
        label = "positive" if i % 2 else "negative"
        other = label if i % 3 else ("negative" if label == "positive" else "positive")
        for protocol, mode in (("VT", "text"), ("VA", "audio")):
            rows.append({"sample_id": f"synthetic-{i}-{protocol}", "video": video,
                "protocol": protocol, "text_content": "never in fit identity", "gt_label": label,
                "sample_type": "Conflict" if i % 3 == 0 else "Aligned", "natural": True,
                "source_dataset": "ch_sims_v2", "split_group_id": video,
                "view_targets": {"vision": {"label": label, "clear": True},
                                 mode: {"label": other, "clear": True}}})
    identity = {"media": media, "settings": copy.deepcopy(DEFAULTS)}
    return {"identity": identity, "identity_hash": fingerprint(identity),
        "settings": copy.deepcopy(DEFAULTS), "train": rows,
        "val": ["MUST_NOT_FIT"], "test": ["MUST_NOT_FIT"]}


class SyntheticBank:
    split = "train"

    def __init__(self, width):
        self.width = width

    def get(self, rows, mode):
        values = [int(r["sample_id"].split("-")[1]) for r in rows]
        hidden = np.array([[np.sin((i + 1) * (j + 1)) for j in range(self.width)] for i in values])
        native = np.zeros((len(rows), 3))
        for i, row in enumerate(rows):
            native[i, ("positive", "negative", "neutral").index(row["view_targets"][mode]["label"])] = 1.5
        return hidden, native


def failure():
    diagnostics = {"success": False, "message": "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT",
        "iterations": 300, "evaluations": 347, "loss": .34808015498,
        "data_loss": .30921262658, "gradient_max": 7.54556e-5, "initial_loss": 3.85214}
    return {"type": "RuntimeError", "message": "L-BFGS did not converge: " + json.dumps(diagnostics)}


class RecoveryTests(unittest.TestCase):
    def test_fit_contexts_raise_only_shared_budget_without_mutating_producer(self):
        c = context()
        before = copy.deepcopy(c)
        arms = [app.fit_context(c, "new-evaluation", arm) for arm in ("control", "expert")]
        self.assertEqual(c, before)
        self.assertNotEqual(arms[0]["identity_hash"], arms[1]["identity_hash"])
        for fitted in arms:
            self.assertEqual({k for k in DEFAULTS if fitted["settings"][k] != DEFAULTS[k]}, {"max_iterations"})
            self.assertEqual(fitted["settings"]["max_iterations"], 3000)
            self.assertNotIn("val", fitted)
            self.assertNotIn("test", fitted)
        arms[0]["settings"]["max_iterations"] = 1
        self.assertEqual(arms[1]["settings"]["max_iterations"], 3000)
        self.assertEqual(c["settings"]["max_iterations"], 300)
        self.assertEqual(app.original.CONFIG["threshold"], .5)
        policy = app.optimizer_policy(c)
        self.assertEqual((policy["gradient_tolerance"], policy["function_tolerance"], policy["regularization"]), (1e-5, 1e-12, .01))

    def test_preserved_failure_must_be_finite_300_iteration_nonconvergence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evaluation/error.json"
            path.parent.mkdir()
            path.write_text(json.dumps(failure()))
            checked = app.checked_failure(tmp)
            self.assertEqual(checked["sha256"], file_hash(path))
            for changed in ({"type": "ValueError", "message": "invalid features"},
                {"type": "RuntimeError", "message": "L-BFGS did not converge: " + json.dumps({"success": True, "iterations": 300})}):
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    app.checked_failure(tmp)

    def test_cache_identity_binds_same_source_tasks_and_extractor(self):
        c = context(3)
        with tempfile.TemporaryDirectory() as tmp:
            source, cache_root = Path(tmp) / "source", Path(tmp) / "cache"
            source.mkdir()
            cache_root.mkdir()
            (source / "context.json").write_text(json.dumps(c))
            raw = {"algorithm": app.cache.VERSION, "source_producer_identity": c["identity_hash"],
                "source_context_sha256": file_hash(source / "context.json"),
                "training_tasks_hash": fingerprint(app.cache.video_tasks(c)),
                "extractor_source_hash": fingerprint(Path(app.cache.__file__).read_text(encoding="utf-8"))}
            path = cache_root / "identity.json"
            path.write_text(json.dumps({"identity": raw, "identity_hash": fingerprint(raw)}))
            _, tasks = app.checked_cache_identity(source, cache_root, c)
            self.assertEqual(len(tasks), 3)
            for field in ("source_producer_identity", "source_context_sha256", "training_tasks_hash", "extractor_source_hash"):
                changed = {**raw, field: "changed"}
                path.write_text(json.dumps({"identity": changed, "identity_hash": fingerprint(changed)}))
                with self.assertRaises(ValueError):
                    app.checked_cache_identity(source, cache_root, c)

    def test_actual_nested_solvers_receive_3000_for_both_arms_and_calibration(self):
        c = context()
        omni, expert = SyntheticBank(5), SyntheticBank(9)
        hybrid = app.cache.HybridBank(omni, expert)
        contexts = {arm: app.fit_context(c, "synthetic-dimension-contract", arm) for arm in ("control", "expert")}
        design = app.group_design(c["train"], c["identity"]["media"])
        with tempfile.TemporaryDirectory() as tmp, patch("rl.evidence_training.fit_head", wraps=fit_head) as control_fit, \
                patch.object(app.original, "fit_head", wraps=fit_head) as expert_fit, \
                patch.object(app.original, "fit_calibration", wraps=fit_calibration) as calibration_fit, \
                patch.object(app.original, "emit"):
            result = app.original.development(tmp, c, omni, hybrid, contexts, design, "primary")
            self.assertEqual((control_fit.call_count, expert_fit.call_count, calibration_fit.call_count), (9, 3, 6))
            for call in control_fit.call_args_list + expert_fit.call_args_list:
                self.assertEqual(call.args[3]["max_iterations"], 3000)
                self.assertEqual(call.args[3]["gradient_tolerance"], 1e-5)
            for call in calibration_fit.call_args_list:
                self.assertEqual(call.args[2]["max_iterations"], 3000)
            self.assertTrue(result["supervised_group_independence"])
            self.assertTrue(all(part["shared_text_audio_heads"] for part in result["partitions"]))
            self.assertEqual(c["settings"]["max_iterations"], 300)

    def test_failed_recovery_never_publishes_completed_report(self):
        c = context()
        with tempfile.TemporaryDirectory() as tmp:
            source, cache_root, output = (Path(tmp) / p for p in ("source", "cache", "recovery"))
            source.mkdir()
            cache_root.mkdir()
            provenance = {"original_failure": {"path": "not-reached"}}
            with patch.object(app, "verified_inputs", return_value=(c, SyntheticBank(5), SyntheticBank(9), provenance)), \
                    patch.object(app.original, "development", side_effect=RuntimeError("L-BFGS did not converge")), \
                    patch.object(app, "emit"):
                with self.assertRaisesRegex(RuntimeError, "did not converge"):
                    app.evaluate_recovery(source, cache_root, output)
            self.assertEqual(app.cache.read(output / "pipeline.status.json")["phase"], "failed")
            self.assertFalse((output / "evaluation/final-report.json").exists())
            self.assertFalse((output / "evaluation/final-report.json.complete.json").exists())
            self.assertEqual(list(source.iterdir()), [])
            self.assertEqual(list(cache_root.iterdir()), [])

    def test_lifecycle_completion_is_collector_compatible_and_marks_gate_failure(self):
        # Stub only the completed-development lifecycle here; the preceding
        # test runs the real nested SciPy fits. Neither test models 8B quality.
        from scripts.server_training_batch import checked_visual_result
        c = context()
        with tempfile.TemporaryDirectory() as tmp:
            source, cache_root, output = (Path(tmp) / p for p in ("source", "cache", "recovery"))
            for directory in (source, cache_root):
                (directory / "features/train").mkdir(parents=True)
                (directory / "features/train/complete.json").write_text('{"fixture": true}')
            old_error = cache_root / "evaluation/error.json"
            old_error.parent.mkdir()
            old_error.write_text(json.dumps(failure()))
            provenance = {"original_failure": app.checked_failure(cache_root),
                "feature_manifest_sha256": {"omni_train": file_hash(source / "features/train/complete.json"),
                                            "visual_train": file_hash(cache_root / "features/train/complete.json")}}
            metrics = {"task_component": .3, "natural_conflict_recall": .5}
            development = {"control": metrics, "expert": metrics, "gain": 0., "folds": []}
            with patch.object(app, "verified_inputs", return_value=(c, SyntheticBank(5), SyntheticBank(9), provenance)), \
                    patch.object(app.original, "development", return_value=development), patch.object(app, "emit"):
                result = app.evaluate_recovery(source, cache_root, output)
            self.assertEqual(checked_visual_result(output), result)
            self.assertEqual(app.cache.read(output / "pipeline.status.json")["phase"], "complete")
            self.assertFalse(result["gate"]["passed"])
            self.assertFalse(result["validation_used"])
            self.assertFalse(result["test_used"])
            self.assertFalse(result["gpu_used"])
            self.assertFalse((output / "features").exists())
            self.assertEqual(result["solver_policy"]["max_iterations"], 3000)


if __name__ == "__main__":
    unittest.main()
