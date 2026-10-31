"""Full-training lifecycle contracts; fixture weights/media are never trained."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from rl.experiment import artifact_hash, file_hash
from scripts import protocol_sft_final as app


class FinalSFTTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="protocol-final-fixture-", dir=app.ROOT / "runs")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data, self.audio, self.questions = (self.root / name for name in ("Dataset", "audio-cache", "questions"))
        self.model = self.root / "models/Qwen2.5-Omni-3B"
        for path in (self.data, self.audio, self.questions, self.model):
            path.mkdir(parents=True)
        (self.model / "config.json").write_text("{}", encoding="utf-8")
        (self.model / "model.safetensors").write_bytes(b"FIXTURE BASE WEIGHTS; NOT A TRAINED MODEL")
        raw, group_questions, sample_questions = [], [], []
        for i in range(30):
            video = f"videos/parent-{i}/clip.mp4"
            path = self.data / video
            path.parent.mkdir(parents=True)
            path.write_bytes(f"fixture-video-{i}".encode())
            audio = app.ablation.sft.extracted_audio_path(self.audio, video)
            audio.parent.mkdir(parents=True)
            audio.write_bytes(f"fixture-audio-{i}".encode())
            group_questions.append({"split_group_id": str(i), "questions": [f"Could you clarify feeling {k}?" for k in range(5)]})
            for protocol, mode in (("VT", "text"), ("VA", "audio")):
                sid, conflict = f"fixture-{i}-{protocol}", i % 3 == 0
                raw.append({"sample_id": sid, "split_group_id": str(i), "split": "train", "video": video,
                    "protocol": protocol, "sample_type": "Conflict" if conflict else "Aligned", "text_content": "PRIVATE TRANSCRIPT",
                    "source_dataset": "ch_sims_v2", "use_in_main": False, "views": {
                        "M1": {"modality": "vision", "label": "negative", "is_clear": True},
                        "M2": {"modality": mode, "label": "positive" if conflict else "negative", "is_clear": True},
                        "M12": {"modality": "vision+" + mode, "label": "negative", "is_clear": True}}})
                sample_questions.append({"sample_id": sid, "questions": [f"Could you clarify feeling {k}?" for k in range(5)]})
        (self.data / "train.jsonl").write_bytes(app.ablation.jsonl_bytes(raw))
        # Preparation must not parse either official file.
        (self.data / "val.jsonl").write_text("OFFICIAL VALIDATION MUST NOT BE READ BY PREPARE", encoding="utf-8")
        (self.data / "test_release.jsonl").write_text("OFFICIAL TEST MUST NOT BE READ BY PREPARE", encoding="utf-8")
        for name, values in (("clarifying_questions_by_group.jsonl", group_questions),
                             ("clarifying_questions_by_sample_id.jsonl", sample_questions)):
            (self.questions / name).write_bytes(app.ablation.jsonl_bytes(values))
        self.seed_roots = []
        for seed, epoch in ((42, 1), (43, 2)):
            root = self.root / ("seed" + str(seed))
            app.ablation.prepare(root, self.data / "train.jsonl", self.data, self.model, self.questions,
                                 self.audio, training_seed=seed, expected_rows=None)
            self.add_seed_results(root, seed, epoch)
            self.seed_roots.append(root)
        self.output = self.root / "final"
        self.row_patch = patch.object(app, "TRAIN_ROWS", 60)
        self.row_patch.start()
        self.addCleanup(self.row_patch.stop)

    def add_seed_results(self, root, seed, experiment_epoch):
        identity = app.read(root / "identity.json")["identity_hash"]
        held = app.ablation.sft.read_jsonl(root / "raw/holdout.jsonl")
        for arm in ("control", "experiment"):
            epoch = experiment_epoch if arm == "experiment" else 1
            adapter = root / arm / "checkpoints" / ("checkpoint-" + str(epoch * 10))
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(f"FIXTURE ADAPTER {seed} {arm}".encode())
            (adapter / "trainer_state.json").write_text(json.dumps({"epoch": float(epoch)}), encoding="utf-8")
            selected = {"algorithm": app.ablation.VERSION, "identity_hash": identity,
                "checkpoint": str(adapter), "adapter_sha256": artifact_hash(adapter, adapter=True),
                "selected_by": "training-only development Score", "holdout_used_for_selection": False}
            (root / arm / "selected-checkpoint.json").write_bytes(app.ablation.json_bytes(selected))
            details, records = [], []
            for row in held:
                yes = arm == "experiment" and row["sample_type"] == "Conflict"
                prediction = {"pred_emotion": "negative", "is_uncertain": "yes" if yes else "no",
                              "clarifying_question": "Could you clarify your emotion?" if yes else "none"}
                details.append({"sample_id": row["sample_id"], "true_emotion": row["views"]["M12"]["label"],
                    "true_sample_type": row["sample_type"], "protocol": row["protocol"],
                    "source_dataset": row["source_dataset"], "inference_error": None, **prediction})
                records.append({"sample_id": row["sample_id"], "gt_label": row["views"]["M12"]["label"],
                    "sample_type": row["sample_type"], "protocol": row["protocol"], "source_dataset": row["source_dataset"], **prediction})
            measured = app.paired.report_metrics(records)
            metrics = {"samples": len(held), "inference_errors": 0, "Score": measured["score"],
                "Macro-F1": measured["macro_f1"], "Conflict_accuracy": measured["conflict_accuracy"],
                "Aligned_calibration": measured["aligned_calibration"]}
            prefix = root / arm / "evaluation" / (adapter.name + "_holdout")
            prefix.parent.mkdir(parents=True)
            prefix.with_name(prefix.name + "_details.jsonl").write_bytes(app.ablation.jsonl_bytes(details))
            prefix.with_name(prefix.name + "_metrics.json").write_bytes(app.ablation.json_bytes(metrics))
            self.csv(prefix.with_name(prefix.name + "_predictions.csv"),
                     [{"sample_id": row["sample_id"], **{k: row[k] for k in app.FIELDS[1:]}} for row in details])
            report = {"algorithm": app.ablation.VERSION, "identity_hash": identity, "arm": arm,
                "checkpoint": str(adapter), "metrics": metrics, "used_for_selection": False,
                "official_val_read": False, "official_test_read": False, "submitted_to_kaggle": False}
            (root / arm / "holdout-report.json").write_bytes(app.ablation.json_bytes(report))
        self.assertTrue(app.paired.summarize(root)["gate"]["passed"])

    def csv(self, path, rows):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=app.FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(stream.getvalue().encode("utf-8"))

    def prepare(self):
        return app.prepare_final(*self.seed_roots, self.output, self.data, self.model, self.audio, self.questions)

    def test_epoch_rule_is_fixed_ceiling_of_development_selected_epochs(self):
        for first, second, expected in ((1, 1, 1), (1, 2, 2), (1, 3, 2), (2, 3, 3), (3, 3, 3)):
            self.assertEqual(app.final_epochs(first, second), expected)
        for invalid in (0, 4, 1.5, True):
            with self.assertRaises(ValueError):
                app.final_epochs(invalid, 2)

    def test_full_preparation_preserves_all_rows_and_never_reads_official_splits(self):
        manifest = self.prepare()
        self.assertEqual(manifest["epochs"], 2)
        self.assertEqual(manifest["rows"], 60)
        self.assertEqual(manifest["sampled_training_rows"], manifest["rows"] + manifest["conflicts"])
        config = yaml.safe_load((self.output / "config.yaml").read_text())
        self.assertFalse(config["do_eval"])
        self.assertEqual(config["eval_strategy"], "no")
        self.assertNotIn("eval_dataset", config)
        self.assertNotIn("adapter_name_or_path", config)
        self.assertEqual((config["num_train_epochs"], config["flash_attn"], config["learning_rate"]), (2., "sdpa", 1e-4))
        records = (app.ablation.sft.read_jsonl(self.output / "dataset/aligned.jsonl") +
                   app.ablation.sft.read_jsonl(self.output / "dataset/conflict.jsonl"))
        self.assertEqual(len({r["sample_id"] for r in records}), 60)
        self.assertTrue(all(r["use_in_main"] is False for r in records))
        self.assertTrue(all("PRIVATE TRANSCRIPT" not in r["messages"][0]["content"] for r in records if r["source_protocol"] == "VA"))
        app.checked_final(self.output)

    def test_full_preparation_requires_both_seed_gates(self):
        original = app.paired.summarize

        def fail_second(root):
            result = deepcopy(original(root))
            if Path(root).resolve() == self.seed_roots[1].resolve():
                result["gate"]["passed"] = False
            return result

        with patch.object(app.paired, "summarize", side_effect=fail_second), self.assertRaisesRegex(ValueError, "gates must pass"):
            self.prepare()
        self.assertFalse((self.output / "prepare-manifest.json").exists())

    def test_no_candidate_or_earlier_checkpoint_can_be_selected_on_official_validation(self):
        self.prepare()
        official = {"val": [{"sample_id": f"val-{i}"} for i in range(6)],
                    "test": [{"sample_id": f"test-{i}", "views": "DO NOT ACCESS TEST LABELS"} for i in range(8)]}
        for split, filename in (("val", "val.jsonl"), ("test", "test_release.jsonl")):
            (self.data / filename).write_bytes(app.ablation.jsonl_bytes(official[split]))
        requests = []

        def execute(command, **kwargs):
            self.assertIn("set_per_process_memory_fraction", command[2])
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1")
            if "--split" not in command:
                requests.append("train")
                checkpoints = self.output / "checkpoints"
                for epoch in (1, 2):
                    path = checkpoints / ("checkpoint-" + str(epoch * 10))
                    path.mkdir(parents=True)
                    (path / "adapter_config.json").write_text("{}")
                    (path / "adapter_model.safetensors").write_bytes(f"FIXTURE FINAL ADAPTER E{epoch}".encode())
                    (path / "trainer_state.json").write_text(json.dumps({"epoch": epoch}))
                (checkpoints / "train_results.json").write_text("{}")
                return
            split = command[command.index("--split") + 1]
            requests.append(split)
            selected = app.read(self.output / "selected-checkpoint.json")
            self.assertEqual(selected["epoch"], 2)
            self.assertEqual(Path(command[command.index("--adapter-path") + 1]).name, "checkpoint-20")
            rows = [{"sample_id": r["sample_id"], "pred_emotion": "negative", "is_uncertain": "no", "clarifying_question": "none"}
                    for r in official[split]]
            self.csv(self.output / "evaluation" / ("locked_final_" + split + "_predictions.csv"), rows)
            # A poor official validation score must still never reselect E1.
            metric_path = self.output / "evaluation" / ("locked_final_" + split + "_metrics.json")
            metric_path.write_bytes(app.ablation.json_bytes({"samples": len(rows), "inference_errors": 0, "Score": .01}))

        with patch.object(app, "VAL_ROWS", 6), patch.object(app, "TEST_ROWS", 8), patch.object(app.subprocess, "run", side_effect=execute):
            result = app.train_final(self.output)
            repeated = app.train_final(self.output)
        self.assertEqual(requests, ["train", "val", "test"])
        self.assertEqual(result, repeated)
        self.assertFalse(result["submitted_to_kaggle"])
        self.assertFalse(result["official_val_used_for_selection"])
        self.assertEqual(result["epochs"], 2)
        self.assertEqual(len(app.checked_csv(self.output / "candidate.csv", [r["sample_id"] for r in official["test"]])), 8)

    def test_identity_detects_changed_prepared_config_and_csv_schema_errors(self):
        self.prepare()
        path = self.output / "config.yaml"
        path.write_text(path.read_text() + "\nlearning_rate: 0.9\n")
        with self.assertRaises(ValueError):
            app.checked_final(self.output)
        candidate = self.root / "bad.csv"
        row = {"sample_id": "s", "pred_emotion": "negative", "is_uncertain": "yes", "clarifying_question": "none"}
        self.csv(candidate, [row])
        with self.assertRaises(ValueError):
            app.checked_csv(candidate, ["s"])
        row.update(is_uncertain="no", clarifying_question="A wrong question?")
        self.csv(candidate, [row])
        with self.assertRaises(ValueError):
            app.checked_csv(candidate, ["s"])
        row.update(clarifying_question="none")
        self.csv(candidate, [row])
        with self.assertRaises(ValueError):
            app.checked_csv(candidate, ["different-id"])


if __name__ == "__main__":
    unittest.main()
