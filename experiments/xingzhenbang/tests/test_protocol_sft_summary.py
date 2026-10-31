"""Strict pairing and the predeclared score/F1/natural-conflict promotion gate."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from rl.experiment import file_hash, fingerprint
from scripts import protocol_sft_summary as app


def predictions():
    result = []
    for i, (emotion, kind, natural, protocol) in enumerate((
        ("positive", "Conflict", True, "VT"), ("negative", "Conflict", True, "VA"),
        ("neutral", "Conflict", False, "VA"), ("positive", "Aligned", True, "VT"),
        ("negative", "Aligned", True, "VA"), ("neutral", "Aligned", False, "VA"))):
        uncertain = "yes" if i in (1, 4) else "no"
        result.append({"sample_id": str(i), "gt_label": emotion, "sample_type": kind,
            "source_dataset": "ch_sims_v2" if natural else "accept_a_va", "protocol": protocol,
            "pred_emotion": emotion, "is_uncertain": uncertain,
            "clarifying_question": "你能说明你的感受吗？" if uncertain == "yes" else "none"})
    experiment = deepcopy(result)
    for row in experiment:
        row["is_uncertain"] = "yes" if row["sample_type"] == "Conflict" else "no"
        row["clarifying_question"] = "你能说明你的感受吗？" if row["is_uncertain"] == "yes" else "none"
    return result, experiment


class SummaryTests(unittest.TestCase):
    def test_gate_and_attribution_for_paired_improvement(self):
        control, experiment = predictions()
        summary = app.paired_report(control, experiment)
        self.assertTrue(summary["gate"]["passed"])
        self.assertEqual(summary["experiment"]["score"], 1.)
        self.assertEqual(summary["changes"]["conflicts_corrected"], 2)
        self.assertEqual(summary["changes"]["false_positives_removed"], 1)
        self.assertEqual(summary["experiment"]["natural_conflict"]["true_positives"], 2)
        self.assertEqual(set(summary["by_protocol"]), {"VT", "VA"})

    def test_natural_regression_or_f1_regression_prevents_promotion(self):
        control, experiment = predictions()
        control[0].update(is_uncertain="yes", clarifying_question="请说明感受？")
        experiment[0].update(is_uncertain="no", clarifying_question="none")
        summary = app.paired_report(control, experiment)
        self.assertFalse(summary["gate"]["passed"])
        self.assertFalse(summary["gate"]["checks"]["natural_conflict_tp_does_not_regress"])
        control, experiment = predictions()
        experiment[3]["pred_emotion"] = "negative"
        summary = app.paired_report(control, experiment)
        self.assertFalse(summary["gate"]["checks"]["macro_f1_drop_at_most_0_005"])
        self.assertFalse(summary["gate"]["passed"])

    def test_pairing_rejects_changed_ids_or_gold_labels(self):
        control, experiment = predictions()
        for key, value in (("sample_id", "wrong"), ("gt_label", "negative"), ("sample_type", "Aligned")):
            changed = deepcopy(experiment)
            changed[0][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "identical holdout"):
                app.paired_report(control, changed)

    def fixture(self, root):
        control, experiment = predictions()
        raw = [{"sample_id": r["sample_id"], "split": "train", "protocol": r["protocol"],
                "source_dataset": r["source_dataset"], "sample_type": r["sample_type"],
                "views": {"M12": {"label": r["gt_label"]}}} for r in control]
        design = {"split_seed": 42, "fixture": True}
        identity = {"algorithm": app.VERSION, "training_seed": 42, "split_hash": fingerprint(design)}
        identity_hash = fingerprint(identity)
        root.mkdir(parents=True, exist_ok=True)
        (root/"identity.json").write_bytes(app.json_bytes({"identity": identity, "identity_hash": identity_hash}))
        (root/"raw").mkdir()
        (root/"raw/holdout.jsonl").write_text("\n".join(json.dumps(r) for r in raw), encoding="utf-8")
        split = {"identity_hash": identity_hash, "design": design, "partitions": {"holdout": {"rows": 6}}}
        (root/"split-manifest.json").write_bytes(app.json_bytes(split))
        manifest = {"algorithm": app.VERSION, "identity_hash": identity_hash, "official_val_read": False,
                    "official_test_read": False, "artifacts": {name: file_hash(root/name) for name in ("raw/holdout.jsonl", "split-manifest.json")}}
        (root/"prepare-manifest.json").write_bytes(app.json_bytes(manifest))
        for arm, records in (("control", control), ("experiment", experiment)):
            folder = root/arm
            evaluation = folder/"evaluation"
            evaluation.mkdir(parents=True)
            checkpoint = str(folder/"checkpoints/checkpoint-10")
            selected = {"algorithm": app.VERSION, "identity_hash": identity_hash,
                "holdout_used_for_selection": False, "selected_by": "training-only development Score", "checkpoint": checkpoint}
            (folder/"selected-checkpoint.json").write_bytes(app.json_bytes(selected))
            values = app.report_metrics(records)
            metrics = {"samples": 6, "inference_errors": 0, "Score": values["score"],
                "Macro-F1": values["macro_f1"], "Conflict_accuracy": values["conflict_accuracy"], "Aligned_calibration": values["aligned_calibration"]}
            (evaluation/"checkpoint-10_holdout_metrics.json").write_bytes(app.json_bytes(metrics))
            details = [{**r, "true_emotion": r["gt_label"], "true_sample_type": r["sample_type"], "inference_error": None} for r in records]
            (evaluation/"checkpoint-10_holdout_details.jsonl").write_text("\n".join(json.dumps(r) for r in details), encoding="utf-8")
            with (evaluation/"checkpoint-10_holdout_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=app.FIELDS)
                writer.writeheader()
                writer.writerows({key: r[key] for key in app.FIELDS} for r in records)
            report = {"algorithm": app.VERSION, "identity_hash": identity_hash, "arm": arm,
                "checkpoint": checkpoint, "used_for_selection": False, "official_val_read": False,
                "official_test_read": False, "metrics": metrics}
            (folder/"holdout-report.json").write_bytes(app.json_bytes(report))

    def test_real_schema_end_to_end_and_immutable_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"run"
            self.fixture(root)
            summary = app.summarize(root)
            self.assertTrue(summary["advance_to_seed43"])
            self.assertEqual(summary["next_action"], "run_seed43_replication")
            self.assertTrue((root/"pair-summary.md").exists())
            original = (root/"pair-summary.json").read_bytes()
            app.summarize(root)
            self.assertEqual((root/"pair-summary.json").read_bytes(), original)
            path = root/"pair-summary.md"
            path.write_text("Existing report")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                app.summarize(root)
            self.assertEqual(path.read_text(), "Existing report")

    def test_holdout_label_mismatch_or_reported_metric_tampering_fails(self):
        for tamper_label in (True, False):
            with self.subTest(tamper_label=tamper_label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)/"run"
                self.fixture(root)
                if tamper_label:
                    path = root/"experiment/evaluation/checkpoint-10_holdout_details.jsonl"
                    content = app.read_rows(path)
                    content[0]["true_emotion"] = "negative"
                    path.write_text("\n".join(json.dumps(r) for r in content), encoding="utf-8")
                else:
                    path = root/"experiment/evaluation/checkpoint-10_holdout_metrics.json"
                    metrics = app.read(path)
                    metrics["Score"] = .99
                    path.write_bytes(app.json_bytes(metrics))
                    report_path = root/"experiment/holdout-report.json"
                    report = app.read(report_path)
                    report["metrics"] = metrics
                    report_path.write_bytes(app.json_bytes(report))
                with self.assertRaises(ValueError):
                    app.summarize(root)
                self.assertFalse((root/"pair-summary.json").exists())

    def test_native_windows_manifest_paths_are_normalized_for_portable_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"run"
            self.fixture(root)
            path = root/"prepare-manifest.json"
            manifest = app.read(path)
            manifest["artifacts"] = {name.replace("/", "\\"): value for name, value in manifest["artifacts"].items()}
            path.write_bytes(app.json_bytes(manifest))
            summary = app.summarize(root)
            self.assertTrue(summary["gate"]["passed"])
            self.assertIn("raw/holdout.jsonl", summary["input_sha256"])
            self.assertTrue(all("\\" not in name for name in summary["input_sha256"]))


if __name__ == "__main__":
    unittest.main()
