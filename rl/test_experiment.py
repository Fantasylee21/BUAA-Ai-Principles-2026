import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rl.experiment import (BalancedSampler, DEFAULT_EXPERIMENT, DEFAULT_RUNTIME,
                           canonical_record, check_disjoint, file_hash, fingerprint,
                           group_advantages, load_records, read_config, validate_config, write_json)
from rl.omni_io import messages_for
from rl.omni_runtime import select_gpu
from rl.state import validate_checkpoint, validate_f1_cache, validate_run_directory


def rows():
    return [canonical_record({"sample_id": f"{kind}-{i}", "video": f"videos/{kind}-{i}.mp4",
                              "protocol": "VA" if i % 2 else "VT", "sample_type": kind,
                              "text_content": "a transcript", "views": {"M12": {"label": "positive"}}})
            for kind in ("Conflict", "Aligned") for i in range(10)]


class ExperimentTests(unittest.TestCase):
    def test_resumable_sampler_and_exact_type_balance(self):
        data = rows()
        original = BalancedSampler(data, 42, 8)
        restored = BalancedSampler(data, 42, 8)
        for step in range(50):
            batch = original.batch(step)
            self.assertEqual(batch, restored.batch(step))
            self.assertEqual(len({r["sample_id"] for r in batch}), 8)
            self.assertEqual(sum(r["sample_type"] == "Conflict" for r in batch), 4)
        self.assertNotEqual(original.batch(5), original.batch(6))

    def test_group_centering_preserves_reward_magnitude(self):
        self.assertEqual(group_advantages([0, .4]), [-.2, .2])
        self.assertEqual(group_advantages([0, .6]), [-.3, .3])
        self.assertEqual(group_advantages([1, 1, 1, 1]), [0, 0, 0, 0])

    def test_gold_metadata_cannot_change_prompt_or_leak_VA_transcript(self):
        row = rows()[1]
        modified = {**row, "gt_label": "negative", "sample_type": "Aligned"}
        before = messages_for(row, "videos/example.mp4")
        self.assertEqual(before, messages_for(modified, "videos/example.mp4"))
        self.assertNotIn("a transcript", json.dumps(before))
        vt = {**row, "protocol": "VT"}
        self.assertIn("a transcript", json.dumps(messages_for(vt, "videos/example.mp4")))

    def test_duplicate_data_and_leakage_detection(self):
        data = rows()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "train.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in [*data, data[0]]), encoding="utf-8")
            loaded, duplicates = load_records(path)
            self.assertEqual(loaded, data)
            self.assertEqual(duplicates, 1)
            path.write_text(json.dumps(data[0]) + "\n" + json.dumps({**data[0], "gt_label": "negative"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_records(path)
        for field in ("sample_id", "video", "split_group_id"):
            val = {**data[-1], field: data[0][field]}
            with self.assertRaises(ValueError):
                check_disjoint([data[0]], [val])

    def test_test_labels_are_explicitly_ignored(self):
        record = canonical_record(rows()[0], require_labels=False)
        self.assertIsNone(record["gt_label"])
        self.assertIsNone(record["sample_type"])

    def test_official_test_protocol_is_inferred_from_input_id_only(self):
        for sid, expected in (("ch_sims_v2:VT:video_0001:0006", "VT"),
                              ("ch_sims_v2:VA:video_0001:0006", "VA"),
                              ("syn:accept_a_va:S0001", "VA"),
                              ("syn:accept_c_svt:S0001", "VT")):
            record = canonical_record({"sample_id": sid, "video": "videos/x.mp4", "text": "input"},
                                      require_labels=False)
            self.assertEqual(record["protocol"], expected)
            self.assertIsNone(record["gt_label"])

    def test_runtime_changes_do_not_change_experiment_identity(self):
        experiment = dict(DEFAULT_EXPERIMENT)
        before = fingerprint(experiment)
        for runtime in ({**DEFAULT_RUNTIME, "gpu": "GPU-new", "micro_batch_size": 2}, DEFAULT_RUNTIME):
            validate_config(experiment, runtime)
            self.assertEqual(before, fingerprint(experiment))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            write_json(path, {"experiment": {"num_generation": 4}})
            with self.assertRaises(ValueError):
                read_config(path)

    def test_untrusted_or_stale_f1_cache_rejected(self):
        identity = {"train": "abc", "base": "base", "sft": "adapter"}
        cache = {"schema": "train_f1_cache_v1", "split": "train", "baseline_identity": identity}
        validate_f1_cache(cache, identity)
        for changed in ({**cache, "split": "val"}, {**cache, "baseline_identity": {"train": "other"}}):
            with self.assertRaises(ValueError):
                validate_f1_cache(changed, identity)

    def test_incomplete_or_corrupted_checkpoint_rejected_before_loading_pickle(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "training_state.pt").write_bytes(b"test state")
            adapter = path / "adapter/policy"
            adapter.mkdir(parents=True)
            (adapter / "adapter_model.safetensors").write_bytes(b"test adapter")
            (adapter / "adapter_config.json").write_text("{}")
            identity = {"experiment": "frozen"}
            with self.assertRaises(FileNotFoundError):
                validate_checkpoint(path, identity)
            write_json(path / "complete.json", {"schema": "omni_checkpoint_v1", "identity": identity,
                "completed_steps": 10, "policy_digest": "p", "reference_digest": "r",
                "files": {str(p.relative_to(path)).replace('\\', '/'): file_hash(p)
                          for p in path.rglob("*") if p.is_file()}})
            validate_checkpoint(path, identity)
            with self.assertRaises(ValueError):
                validate_checkpoint(path, {"experiment": "different"})
            (path / "training_state.pt").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                validate_checkpoint(path, identity)

    def test_resume_cannot_reuse_another_run_or_rewind_past_evaluations(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            identity = {"run": "one"}
            validate_run_directory(path, identity)
            checkpoint = path / "checkpoint-000020"
            checkpoint.mkdir()
            write_json(checkpoint / "complete.json", {"completed_steps": 20})
            validate_run_directory(path, identity, {"completed_steps": 20})
            with self.assertRaises(ValueError):
                validate_run_directory(path, identity, {"completed_steps": 10})
            with self.assertRaises(ValueError):
                validate_run_directory(path, {"run": "other"}, {"completed_steps": 20})

    def test_gpu_selection_uses_actual_free_memory_and_avoids_busy_cards(self):
        inventory = "0, GPU-zero, RTX 4090, 49140, 49100\n1, GPU-one, RTX 5090, 32768, 32000\n2, GPU-two, RTX 4090, 49140, 48000\n"
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            with patch("subprocess.check_output", side_effect=[inventory, "GPU-zero\n"]):
                selected = select_gpu("auto")
            self.assertEqual(selected["uuid"], "GPU-two")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-two")

    def test_gpu_selection_respects_parent_allocation(self):
        inventory = "0, GPU-zero, RTX 4090, 49140, 49100\n1, GPU-one, RTX 5090, 32768, 32000\n"
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
            with patch("subprocess.check_output", side_effect=[inventory, ""]):
                self.assertEqual(select_gpu("auto")["uuid"], "GPU-one")
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
            with patch("subprocess.check_output", return_value=inventory):
                with self.assertRaises(ValueError):
                    select_gpu("0")


if __name__ == "__main__":
    unittest.main()
