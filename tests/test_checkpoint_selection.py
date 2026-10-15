import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from select_sft_checkpoint import select_checkpoint


class CheckpointSelectionTests(unittest.TestCase):
    def fixture(self, base):
        output = base / "adapters"
        for step in (10, 20):
            adapter = output / f"checkpoint-{step}"
            adapter.mkdir(parents=True)
            (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
        config = base / "config.yaml"
        config.write_text(yaml.safe_dump({"model_name_or_path": "models/Qwen2.5-Omni-3B",
                                         "output_dir": str(output), "stage": "sft",
                                         "finetuning_type": "lora", "quantization_bit": 4}))
        return config, output, base / "evaluation"

    def evaluator(self, evaluation, samples=666):
        def execute(command, **kwargs):
            name = command[command.index("--run-name") + 1]
            score = 0.7 if "checkpoint-20" in name else 0.6
            result = {"samples": samples, "inference_errors": 0, "Macro-F1": 0.5,
                      "Conflict_accuracy": 0.8, "Aligned_calibration": 0.9, "Score": score}
            (evaluation / f"{name}_metrics.json").write_text(json.dumps(result))
        return execute

    def test_selects_highest_score_from_complete_epoch_evaluations(self):
        with tempfile.TemporaryDirectory() as directory:
            config, output, evaluation = self.fixture(Path(directory))
            with patch("select_sft_checkpoint.subprocess.run", side_effect=self.evaluator(evaluation)) as execute:
                selected = select_checkpoint(config, evaluation)
            self.assertEqual(execute.call_count, 2)
            self.assertTrue(selected["checkpoint"].endswith("checkpoint-20"))
            self.assertEqual(selected["Score"], 0.7)
            self.assertEqual(json.loads((output / "best_checkpoint.json").read_text())["samples"], 666)

    def test_partial_validation_is_not_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            config, output, evaluation = self.fixture(Path(directory))
            with patch("select_sft_checkpoint.subprocess.run", side_effect=self.evaluator(evaluation, 665)):
                with self.assertRaisesRegex(RuntimeError, "Incomplete validation"):
                    select_checkpoint(config, evaluation)
            self.assertFalse((output / "best_checkpoint.json").exists())

    def test_failed_evaluator_does_not_select_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            config, output, evaluation = self.fixture(Path(directory))
            with patch("select_sft_checkpoint.subprocess.run", side_effect=subprocess.CalledProcessError(1, "evaluate")):
                with self.assertRaises(subprocess.CalledProcessError):
                    select_checkpoint(config, evaluation)
            self.assertFalse((output / "best_checkpoint.json").exists())


if __name__ == "__main__":
    unittest.main()
