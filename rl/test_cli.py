"""Exercise real CLI/data flow without importing the training stack."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rl.experiment import ROOT
from rl.test_experiment import rows


class CLITests(unittest.TestCase):
    def test_dry_run_and_data_preparation_are_independent_of_torch_and_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train, val = root / "train.jsonl", root / "val.jsonl"
            train.write_text("\n".join(json.dumps(row) for row in rows()), encoding="utf-8")
            validation = [{**rows()[i % 20], "sample_id": f"val-{i}", "video": f"val/{i}.mp4",
                           "split_group_id": f"val/{i}.mp4"} for i in range(666)]
            val.write_text("\n".join(json.dumps(row) for row in validation), encoding="utf-8")
            code = "import sys; from rl.train_grpo import main; main(sys.argv[1:]); assert 'torch' not in sys.modules"
            result = subprocess.run([
                sys.executable, "-B", "-c", code, "--dry-run", "--data", str(train),
                "--val-file", str(val), "--model-path", str(root / "missing-base"),
                "--sft-adapter-path", str(root / "missing-adapter"),
                "--run-dir", str(root / "not-created")], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["unique_train"], 20)
            self.assertEqual(report["validation"], 666)
            self.assertEqual(report["completions_per_update"], 32)
            self.assertFalse((root / "not-created").exists())
            prepared = root / "prepared.jsonl"
            result = subprocess.run([
                sys.executable, "-B", "-m", "rl.prepare_rl_data", "--input", str(train),
                "--val-file", str(val), "--output", str(prepared)],
                cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(prepared.read_text(encoding="utf-8").splitlines()), 20)

    def test_test_inputs_cannot_be_used_for_training(self):
        result = subprocess.run([
            sys.executable, "-B", "-m", "rl.train_grpo", "--split", "test",
            "--model-path", "unused-base", "--sft-adapter-path", "unused-adapter",
            "--run-dir", "runs/must-not-be-created"], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Test inputs cannot be used", result.stderr)
        self.assertFalse((ROOT / "runs/must-not-be-created").exists())


if __name__ == "__main__":
    unittest.main()
