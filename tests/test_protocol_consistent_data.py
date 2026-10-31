"""Original-label Omni preparation with isolated, reproducible SFT outputs."""
from __future__ import annotations

from collections import defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
import io
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data import prepare_training_data as clean
from scripts import prepare_sft_data as sft


ROOT = Path(__file__).resolve().parents[1]


def raw_pair(split="train", group="clip"):
    records = []
    for index, protocol in enumerate(("VT", "VA"), 1):
        second = "text" if protocol == "VT" else "audio"
        records.append({
            "sample_id": f"source:{protocol}:{group}", "split_group_id": group,
            "protocol": protocol, "split": split, "source_dataset": "source",
            "sample_type": "Conflict" if protocol == "VT" else "Aligned",
            "video": f"videos/{group}.mp4", "text_content": "SECRET_SUPPLIED_TRANSCRIPT",
            "use_in_main": False, "_source_line": index,
            "views": {
                "M1": {"modality": "vision", "label": "negative", "is_clear": False},
                "M2": {"modality": second, "label": "positive" if protocol == "VT" else "negative", "is_clear": True},
                "M12": {"modality": "vision+" + second, "label": "negative", "is_clear": True},
            },
        })
    return records


def groups_for(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[row["split_group_id"]][row["protocol"]] = row
    return groups


class ProtocolConsistentDataTests(unittest.TestCase):
    def test_original_protocol_targets_and_clarity_survive_without_union(self):
        rows = raw_pair()
        before = deepcopy(rows)
        output, _ = clean.build_omni_records(rows, groups_for(rows), omni_input_policy="protocol_consistent")
        self.assertEqual([r["sample_type"] for r in output], ["Conflict", "Aligned"])
        self.assertEqual([r["is_uncertain"] for r in output], ["yes", "no"])
        self.assertEqual(output[1]["input_modalities"], ["vision", "audio"])
        for original, result in zip(rows, output):
            self.assertEqual(result["views"], original["views"])
            self.assertFalse(result["use_in_main"])
            self.assertNotIn("vision", result["view_supervision"])
            self.assertEqual(result["view_supervision"], {original["views"]["M2"]["modality"]: original["views"]["M2"]["label"]})
        output[0]["views"]["M1"]["label"] = "neutral"
        self.assertEqual(rows, before)

    def test_baseline_validation_policy_does_not_union_original_va_label(self):
        rows = raw_pair("val")
        output, _ = clean.build_omni_records(rows, groups_for(rows),
            baseline_aligned_validation=True, omni_input_policy="protocol_consistent")
        self.assertEqual(output[1]["sample_type"], "Aligned")
        self.assertEqual(output[1]["is_uncertain"], "no")
        self.assertEqual(output[1]["views"], rows[1]["views"])

    def test_default_legacy_training_and_validation_union_remain_unchanged(self):
        rows = raw_pair()
        legacy, _ = clean.build_omni_records(rows, groups_for(rows))
        validation, _ = clean.build_omni_records(rows, groups_for(rows), baseline_aligned_validation=True)
        self.assertEqual(legacy[1]["sample_type"], "Conflict")
        self.assertEqual(legacy[1]["training_protocol"], "VTA")
        self.assertEqual(legacy[1]["input_modalities"], ["vision", "text", "audio"])
        self.assertEqual(validation[1]["sample_type"], "Conflict")
        self.assertEqual(validation[1]["input_modalities"], ["vision", "audio"])
        self.assertNotIn("omni_input_policy", legacy[1])
        self.assertNotIn("views", legacy[1])

    def test_new_policy_rejects_union_cleaned_legacy_rows(self):
        rows = raw_pair()
        legacy, _ = clean.build_omni_records(rows, groups_for(rows))
        with self.assertRaises(ValueError):
            sft.validate_omni_input_policy(legacy, "protocol_consistent")

    def test_clean_cli_keeps_use_in_main_false_main_examples(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            root = Path(directory)
            video = root / "videos/clip.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"fixture")
            raw = root / "train.jsonl"
            raw.write_text("".join(json.dumps(r) + "\n" for r in raw_pair()), encoding="utf-8")
            args = ["prepare_training_data.py", "--input", str(raw), "--dataset-root", str(root),
                    "--output-dir", str(root / "clean"), "--omni-input-policy", "protocol_consistent"]
            with patch("sys.argv", args), redirect_stdout(io.StringIO()):
                clean.main()
            output = sft.read_jsonl(root / "clean/train_omni_clean.jsonl")
            self.assertEqual(len(output), 2)
            self.assertTrue(all(r["use_in_main"] is False for r in output))
            self.assertEqual([r["sample_type"] for r in output], ["Conflict", "Aligned"])

    def test_va_sft_prompt_withholds_supplied_text_and_keeps_raw_metadata(self):
        rows = raw_pair()
        cleaned, _ = clean.build_omni_records(rows, groups_for(rows), omni_input_policy="protocol_consistent")
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            root = Path(directory)
            video = root / "videos/clip.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"fixture")
            audio = root / "audio.flac"
            audio.write_bytes(b"fixture")
            questions = {rows[0]["sample_id"]: {"questions": [f"Could you explain context {i}?" for i in range(5)]}}
            records = sft.make_sft_records(cleaned, root, root, "omni", "train", {}, questions,
                "single", "seed", {"videos/clip.mp4": "audio.flac"}, "full")
            sft.validate_sft_records(records, "omni", root, "full")
            by_id = {r["sample_id"]: r for r in records}
            va = by_id[rows[1]["sample_id"]]
            self.assertNotIn("SECRET_SUPPLIED_TRANSCRIPT", va["messages"][0]["content"])
            self.assertNotIn("Supplied text:", va["messages"][0]["content"])
            self.assertIn("<audio>", va["messages"][0]["content"])
            self.assertEqual(json.loads(va["messages"][1]["content"])["is_uncertain"], "no")
            self.assertEqual(va["views"], rows[1]["views"])
            self.assertFalse(va["use_in_main"])
            self.assertIn("SECRET_SUPPLIED_TRANSCRIPT", by_id[rows[0]["sample_id"]]["messages"][0]["content"])

    @unittest.skipUnless((ROOT / "Dataset/train.jsonl").is_file() and (ROOT / "Dataset/val.jsonl").is_file(),
                         "Optional local official dataset check")
    def test_original_official_counts_are_411_and_74_conflicts(self):
        for split, count, conflicts in (("train", 3354, 411), ("val", 666, 74)):
            with self.subTest(split=split):
                rows = clean.read_jsonl(ROOT / "Dataset" / f"{split}.jsonl")
                output, _ = clean.build_omni_records(rows, groups_for(rows),
                    baseline_aligned_validation=split == "val", omni_input_policy="protocol_consistent")
                self.assertEqual(len(output), count)
                self.assertEqual(sum(r["sample_type"] == "Conflict" for r in output), conflicts)
                self.assertEqual([r["views"] for r in output], [r["views"] for r in rows])

    def test_isolated_generation_and_existing_resampler_accept_new_policy(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            root = Path(directory)
            dataset = root / "Dataset"
            train = raw_pair(group="train")
            aligned = deepcopy(train[0])
            aligned.update(sample_id="source:VT:aligned", split_group_id="aligned",
                           sample_type="Aligned", video="videos/aligned.mp4", _source_line=3)
            aligned["views"]["M2"]["label"] = "negative"
            train.append(aligned)
            val = raw_pair("val", "val")
            for row in [*train, *val]:
                path = dataset / row["video"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            for split, raw in (("train", train), ("val", val)):
                grouped = groups_for(raw)
                omni, _ = clean.build_omni_records(raw, grouped, omni_input_policy="protocol_consistent")
                vl, _ = (clean.build_vl_records(grouped) if split == "train" else
                         clean.build_vl_records_preserving_source(raw, grouped))
                destination = root / "data/cleaned/protocol_consistent" / split
                for family, records in (("vl", vl), ("omni", omni)):
                    sft.write_jsonl(destination / f"{split}_{family}_clean.jsonl", records)
            global_info = root / "data/llamafactory/dataset_info.json"
            sft.write_json(global_info, sft.dataset_info())
            before = global_info.read_bytes()
            audio = root / "fixture.flac"
            audio.write_bytes(b"fixture")
            audio_map = {r["video"]: "fixture.flac" for r in [*train, *val] if r["protocol"] == "VA"}
            question = {"questions": [f"Could you explain context {i}?" for i in range(5)]}
            argv = ["prepare_sft_data.py", "--repo-root", str(root),
                    "--omni-input-policy", "protocol_consistent"]
            with patch("sys.argv", argv), patch.object(sft, "prepare_audio", return_value=(audio_map, [])), \
                    patch.object(sft, "load_questions", return_value=({"train": question}, {train[0]["sample_id"]: question})), \
                    redirect_stdout(io.StringIO()):
                sft.main()
            generated = root / "data/llamafactory/generated/protocol_consistent"
            self.assertEqual(global_info.read_bytes(), before)
            info = json.loads((generated / "dataset_info.json").read_text())
            self.assertEqual(info["affect_sft_omni_train"]["file_name"], "affect_sft_omni_train.jsonl")
            manifest = json.loads((generated / "generation_manifest.json").read_text())
            self.assertEqual(manifest["partitions"]["train"]["input_policy"]["omni"]["VA"], ["vision", "audio"])
            with patch.dict("sys.modules", {"prepare_sft_data": sft}):
                resampler = importlib.import_module("scripts.prepare_resampled_sft_data")
                summary = resampler.prepare(root, generated, generated / "resampled_2x", 2)
            self.assertEqual(summary["omni"]["original_conflict"], 1)
            self.assertEqual(summary["omni"]["sampled_conflict"], 2)
            self.assertEqual(summary["omni"]["sampled_train"], 4)
            self.assertEqual(global_info.read_bytes(), before)

    def test_v5_only_changes_v4_dataset_and_output_locations(self):
        import yaml
        config_dir = ROOT / "configs/sft/qwen25omni"
        v4 = yaml.safe_load((config_dir / "qlora_sft_v4_resample2x.yaml").read_text())
        v5 = yaml.safe_load((config_dir / "qlora_sft_v5_protocol.yaml").read_text())
        changed = {k for k in v4.keys() | v5.keys() if v4.get(k) != v5.get(k)}
        self.assertEqual(changed, {"dataset_dir", "output_dir"})
        self.assertIn("protocol_consistent", v5["dataset_dir"])
        self.assertEqual(sft.dataset_info("")["affect_sft_omni_train"]["file_name"], "affect_sft_omni_train.jsonl")


if __name__ == "__main__":
    unittest.main()
