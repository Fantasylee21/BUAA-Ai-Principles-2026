"""TRAIN-only SFT preparation contracts; fake media are never decoded or trained."""
from collections import defaultdict
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from scripts import protocol_sft_ablation as app


def fixture_split(rows, media):
    # Production reuses the separately tested C2 grouping. This fixture replaces
    # its SciPy-importing module while exercising the remaining preparation path.
    groups = [media[r["video"]] for r in rows]
    folds_by_group = {g: i % 5 for i, g in enumerate(sorted(set(groups)))}
    folds = [folds_by_group[g] for g in groups]
    names = ["train" if f < 3 else "dev" if f == 3 else "holdout" for f in folds]
    return {"split_seed": 42, "requested_ratios": [.6, .2, .2], "groups": groups,
            "five_folds": folds, "assignment": names,
            "partitions": {name: [i for i, part in enumerate(names) if part == name]
                           for name in ("train", "dev", "holdout")}}


class ProtocolPreparationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="protocol-fixture-", dir=app.ROOT/"runs")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data, self.audio, self.questions = self.root/"Dataset", self.root/"audio-cache", self.root/"questions"
        self.model, self.output = self.root/"models/Qwen2.5-Omni-3B", self.root/"prepared"
        for path in (self.data, self.questions, self.model):
            path.mkdir(parents=True)
        (self.model/"config.json").write_text("{}", encoding="utf-8")
        (self.model/"model.safetensors").write_bytes(b"FAKE UNIT TEST BASE WEIGHTS")
        self.raw, group_questions, sample_questions = [], [], []
        for group in range(30):
            video = f"videos/{group}.mp4"
            path = self.data/video
            path.parent.mkdir(parents=True, exist_ok=True)
            # Different names with equal media bytes must stay together.
            path.write_bytes(f"fake-media-{group if group != 1 else 0}".encode())
            audio = app.sft.extracted_audio_path(self.audio, video)
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(b"FAKE UNIT TEST FLAC")
            choices = [f"你能说明第{i}种感受吗？" for i in range(5)]
            group_questions.append({"split_group_id": str(group), "questions": choices})
            for protocol in ("VT", "VA"):
                conflict = group % 5 == 1 or (group % 3 == 0 and protocol == "VT")
                mode = "text" if protocol == "VT" else "audio"
                sid = f"s-{group}-{protocol}"
                self.raw.append({"sample_id": sid, "split_group_id": str(group), "protocol": protocol,
                    "split": "train", "sample_type": "Conflict" if conflict else "Aligned",
                    "video": video, "text_content": "SECRET TRANSCRIPT", "source_dataset": "ch_sims_v2",
                    "use_in_main": False, "views": {
                        "M1": {"modality": "vision", "label": "negative", "is_clear": group % 2 == 0},
                        "M2": {"modality": mode, "label": "positive" if conflict else "negative", "is_clear": True},
                        "M12": {"modality": "vision+"+mode, "label": "negative", "is_clear": True}}})
                sample_questions.append({"sample_id": sid, "questions": choices})
        self.train = self.data/"train.jsonl"
        self.train.write_bytes(app.jsonl_bytes(self.raw))
        # Any accidental official-split read fails parsing immediately.
        (self.data/"val.jsonl").write_text("DO NOT READ OFFICIAL VALIDATION", encoding="utf-8")
        (self.data/"test_release.jsonl").write_text("DO NOT READ OFFICIAL TEST", encoding="utf-8")
        for name, values in (("clarifying_questions_by_group.jsonl", group_questions),
                             ("clarifying_questions_by_sample_id.jsonl", sample_questions)):
            (self.questions/name).write_bytes(app.jsonl_bytes(values))

    def prepare(self):
        return app.prepare(self.output, self.train, self.data, self.model, self.questions,
                           self.audio, expected_rows=None, split_fn=fixture_split)

    def test_preparation_keeps_original_labels_and_changes_only_training_protocol(self):
        manifest = self.prepare()
        split = app.read(self.output/"split-manifest.json")
        self.assertFalse(manifest["trained"])
        self.assertFalse(manifest["official_val_read"])
        self.assertFalse(manifest["official_test_read"])
        self.assertEqual(split["full_source"]["rows"], 60)
        self.assertEqual(split["full_source"]["use_in_main_false"], 60)
        self.assertGreater(split["full_source_conflicts"]["control"], split["full_source_conflicts"]["experiment"])
        groups = split["design"]["groups"]
        partitions = split["design"]["partitions"]
        sets = [{groups[i] for i in ids} for ids in partitions.values()]
        self.assertFalse(any(sets[i]&sets[j] for i in range(3) for j in range(i)))
        self.assertEqual(groups[0], groups[2])
        configs = {arm: yaml.safe_load((self.output/arm/"config.yaml").read_text()) for arm in app.ARMS}
        for arm, config in configs.items():
            self.assertEqual(config["learning_rate"], 1e-4)
            self.assertEqual(config["lora_rank"], 16)
            self.assertEqual(config["flash_attn"], "sdpa")
            self.assertEqual(config["num_train_epochs"], 3.)
            self.assertEqual(config["model_name_or_path"], str(self.model.resolve()))
            self.assertNotIn("adapter_name_or_path", config)
            self.assertEqual(config["eval_dataset"], "protocol_dev")
            info = app.read(self.output/arm/"dataset/dataset_info.json")
            self.assertEqual(set(info), {"protocol_train_aligned", "protocol_train_conflict", "protocol_dev"})
            conflicts = app.sft.read_jsonl(self.output/arm/"dataset/conflict.jsonl")
            self.assertEqual(info["protocol_train_conflict"]["num_samples"], len(conflicts)*2)
            records = conflicts+app.sft.read_jsonl(self.output/arm/"dataset/aligned.jsonl")
            for record in records:
                self.assertFalse(record["use_in_main"])
                if record["source_protocol"] == "VA":
                    prompt = record["messages"][0]["content"]
                    self.assertEqual("Supplied text:" in prompt, arm == "control")
                    self.assertTrue(record["audios"])
            dev = app.sft.read_jsonl(self.output/arm/"dataset/dev.jsonl")
            self.assertTrue(all("Supplied text:" not in r["messages"][0]["content"] for r in dev if r["source_protocol"] == "VA"))
        self.assertEqual(configs["control"]["task_balanced_loss"], configs["experiment"]["task_balanced_loss"])
        attention = app.read(self.output/"identity.json")["identity"]["attention"]
        self.assertEqual(attention["shared_flash_attn"], "sdpa")
        self.assertEqual(attention["historical_v4_flash_attn"], "fa2")
        self.assertTrue(attention["differs_from_historical_v4"])

    def test_immutable_prepare_and_raw_source_integrity_before_training(self):
        self.prepare()
        first = (self.output/"prepare-manifest.json").read_bytes()
        self.prepare()
        self.assertEqual((self.output/"prepare-manifest.json").read_bytes(), first)
        with patch.object(app.subprocess, "run", side_effect=AssertionError("Dry run launched a subprocess")), redirect_stdout(io.StringIO()):
            result = app.train(self.output, "control", "1", dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertNotIn("resume_from_checkpoint", " ".join(result["training_command"]))
        self.train.write_bytes(self.train.read_bytes()+b"\n")
        with self.assertRaisesRegex(ValueError, "Raw training file changed"):
            app.checked_run(self.output)

    def test_attention_cannot_change_within_a_frozen_preparation(self):
        self.prepare()
        original = (self.output/"identity.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            app.prepare(self.output, self.train, self.data, self.model, self.questions,
                        self.audio, flash_attn="fa2", expected_rows=None, split_fn=fixture_split)
        self.assertEqual((self.output/"identity.json").read_bytes(), original)
        for arm in app.ARMS:
            self.assertEqual(yaml.safe_load((self.output/arm/"config.yaml").read_text())["flash_attn"], "sdpa")

    def test_grouped_split_reuses_C2_five_folds_and_never_splits_a_content_group(self):
        rows = [{"sample_id": str(i)} for i in range(10)]
        groups = [f"g-{i//2}" for i in range(10)]
        def balance(given_rows, given_groups, seed, count):
            self.assertEqual((seed, count), (42, 5))
            return [i//2 for i in range(10)]
        helper = SimpleNamespace(content_groups=lambda r, m: groups, _assign_folds=balance)
        with patch.dict("sys.modules", {"scripts.train_evidence_calibrator": helper}):
            design = app.grouped_split(rows, {})
        self.assertEqual([len(design["partitions"][p]) for p in ("train", "dev", "holdout")], [6, 2, 2])

    def test_rejects_official_split_and_changed_prepared_config(self):
        with self.assertRaisesRegex(ValueError, "only the original train"):
            app.verified_training(self.data/"val.jsonl", self.data, None)
        self.prepare()
        path = self.output/"experiment/config.yaml"
        path.write_text(path.read_text()+"\nlearning_rate: 0.9\n")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            app.checked_run(self.output)

    def test_evaluation_uses_private_development_or_holdout_directory(self):
        command = app.evaluation_command(self.output, "experiment", self.root/"checkpoint-10", "holdout")
        self.assertIn(str(self.output/"evaluation-inputs/holdout"), command)
        self.assertNotIn(str(self.data), command)
        self.assertEqual(command[command.index("--max-frames")+1], "64")
        self.assertEqual(command[command.index("--max-visual-tokens")+1], "64")
        self.assertIn("set_per_process_memory_fraction", command[2])

    def test_train_selects_on_all_dev_epochs_then_evaluates_only_winner_on_holdout(self):
        self.prepare()
        arm_root = self.output/"experiment"
        checkpoints = arm_root/"checkpoints"
        for step in (10, 20, 30):
            path = checkpoints/f"checkpoint-{step}"
            path.mkdir(parents=True)
            (path/"adapter_config.json").write_text("{}")
            (path/"adapter_model.safetensors").write_bytes(f"fake-adapter-{step}".encode())
        (checkpoints/"train_results.json").write_text("{}")
        counts = app.read(self.output/"split-manifest.json")["partitions"]
        requests = []
        def evaluate(command, **kwargs):
            adapter = Path(command[command.index("--adapter-path")+1])
            partition = Path(command[command.index("--data-root")+1]).name
            requests.append((adapter.name, partition))
            if partition == "holdout":
                self.assertTrue((arm_root/"selected-checkpoint.json").exists())
                self.assertEqual(adapter.name, "checkpoint-20")
            score = {"checkpoint-10": .5, "checkpoint-20": .8, "checkpoint-30": .6}[adapter.name]
            metric_path = arm_root/"evaluation"/(adapter.name+"_"+partition+"_metrics.json")
            metric_path.parent.mkdir(parents=True, exist_ok=True)
            metric_path.write_bytes(app.json_bytes({"samples": counts[partition]["rows"], "inference_errors": 0, "Score": score}))
        with patch.object(app.subprocess, "run", side_effect=evaluate):
            report = app.train(self.output, "experiment", "1")
        self.assertEqual(requests, [("checkpoint-10", "dev"), ("checkpoint-20", "dev"),
                                    ("checkpoint-30", "dev"), ("checkpoint-20", "holdout")])
        self.assertFalse(report["used_for_selection"])
        with patch.object(app.subprocess, "run", side_effect=AssertionError("Holdout must not be rerun")):
            self.assertEqual(app.train(self.output, "experiment", "1"), report)


if __name__ == "__main__":
    unittest.main()
