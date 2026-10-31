"""Recovery corruption must be visible; imported caches retain exact provenance."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rl.experiment import file_hash, write_json
from rl.grounded_recovery import (prepare_environment, tensor_difference, compare_snapshots,
                                  task_differences, configure_torch)
from rl.grounded_store import cache_contract, import_verified_cache, task_key
from rl.experiment import ROOT


class RecoveryContracts(unittest.TestCase):
    def test_replay_difference_detects_parameter_and_optimizer_corruption(self):
        import torch
        left = {"parameters": {"LoRA": torch.tensor([.1, .2])},
                "optimizer": {"0/exp_avg": torch.tensor([.01, .02])}, "metadata": {"step": 10}}
        identical = {key: {n: v.clone() for n, v in values.items()} if key != "metadata" else dict(values)
                     for key, values in left.items()}
        self.assertTrue(compare_snapshots(torch, left, identical)["exact"])
        identical["optimizer"]["0/exp_avg"][0] += .001
        diagnosis = compare_snapshots(torch, left, identical)
        self.assertTrue(diagnosis["parameters"]["exact"])
        self.assertFalse(diagnosis["optimizer"]["exact"])
        self.assertGreater(diagnosis["optimizer"]["max_abs"], .0009)
        self.assertFalse(diagnosis["exact"])
        self.assertFalse(tensor_difference(torch, left["parameters"], {})["exact"])

    def test_recovery_checks_same_tasks_and_native_value_scores(self):
        original = [{"sample_id": "train-a", "draw_id": "42:11:0", "mode": "full", "target": "yes",
                     "selection": "missed", "weight": .075, "scores": [4., 3.]}]
        copy = json.loads(json.dumps(original))
        self.assertTrue(task_differences(original, copy)["exact"])
        copy[0]["scores"][0] += .01
        result = task_differences(original, copy)
        self.assertTrue(result["same_samples_modes_targets"])
        self.assertFalse(result["exact"])
        self.assertAlmostEqual(result["maximum_score_difference"], .01)
        copy[0]["sample_id"] = "validation-input"
        self.assertFalse(task_differences(original, copy)["same_samples_modes_targets"])

    def test_deterministic_runtime_configuration_is_explicit(self):
        import torch
        previous = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,
                    torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32)
        try:
            with patch.dict(os.environ, {}, clear=False):
                prepare_environment()
                self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
                configure_torch(torch)
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                self.assertTrue(torch.backends.cudnn.deterministic)
                self.assertFalse(torch.backends.cudnn.benchmark)
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        finally:
            torch.use_deterministic_algorithms(previous[0])
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = previous[1:3]
            torch.backends.cuda.matmul.allow_tf32 = previous[3]

    def cache_fixture(self, root):
        source, run = root / "source", root / "new"
        for folder in (source / "cache", run / "cache"):
            folder.mkdir(parents=True)
        train = [{"sample_id": "train-vt", "view_targets": {"vision": {}, "text": {}}},
                 {"sample_id": "train-va", "view_targets": {"vision": {}, "audio": {}}}]
        val = [{"sample_id": "val-vt"}]
        entries = {}
        for split, rows in (("train", train), ("val", val)):
            for row in rows:
                for mode in (["full", *row["view_targets"]] if split == "train" else ["full"]):
                    key = task_key(row["sample_id"], mode)
                    path = source / "cache" / (key+".pt")
                    path.write_bytes((row["sample_id"]+mode).encode())
                    entries[key] = {"sample_id": row["sample_id"], "mode": mode, "split": split,
                        "file": path.name, "sha256": file_hash(path), "identity_hash": "producer"}
        write_json(source / "context.json", {"identity_hash": "producer"})
        write_json(source / "cache/complete.json", {"identity_hash": "producer", "entries": entries})
        parent = {"run": str(source), "identity_hash": "producer", "cache_contract": cache_contract(ROOT),
            "context_sha256": file_hash(source / "context.json"),
            "manifest_sha256": file_hash(source / "cache/complete.json")}
        return run, {"identity_hash": "new-experiment", "identity": {"cache_parent": parent}, "train": train, "val": val}, entries

    def test_cache_reuse_verifies_every_blob_and_preserves_original_files(self):
        with tempfile.TemporaryDirectory() as folder:
            run, context, original = self.cache_fixture(Path(folder))
            progress = []
            imported = import_verified_cache(run, context, lambda n, total: progress.append((n, total)))
            self.assertEqual(progress[-1], (7, 7))
            self.assertEqual(set(imported["entries"]), set(original))
            for key, entry in imported["entries"].items():
                self.assertEqual(entry["identity_hash"], "new-experiment")
                self.assertEqual(entry["producer_identity"], "producer")
                old = Path(context["identity"]["cache_parent"]["run"]) / "cache" / entry["file"]
                self.assertTrue(os.path.samefile(old, run / "cache" / entry["file"]))
            self.assertEqual(json.loads((Path(context["identity"]["cache_parent"]["run"]) / "cache/complete.json").read_text())["entries"], original)
            first = next(iter(imported["entries"].values()))
            (run / "cache" / first["file"]).write_bytes(b"corrupt blob")
            with self.assertRaisesRegex(ValueError, "checksum failed"):
                import_verified_cache(run, context)

    def test_import_rejects_cross_split_metadata_even_with_updated_manifest_checksum(self):
        with tempfile.TemporaryDirectory() as folder:
            run, context, entries = self.cache_fixture(Path(folder))
            parent = context["identity"]["cache_parent"]
            entry = next(iter(entries.values()))
            entry["split"] = "val" if entry["split"] == "train" else "train"
            manifest = Path(parent["run"]) / "cache/complete.json"
            write_json(manifest, {"identity_hash": "producer", "entries": entries})
            parent["manifest_sha256"] = file_hash(manifest)
            with self.assertRaisesRegex(ValueError, "sample/mode/split boundary"):
                import_verified_cache(run, context)
            self.assertFalse((run / "cache/complete.json").exists())


if __name__ == "__main__":
    unittest.main()
