"""Runtime artifact and read-only progress contracts, without torch/GPU loads."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import joint_affect_pipeline as pipeline
from scripts import joint_affect_progress as progress
from tests.test_joint_affect_data import temporary_directory


class RuntimeContracts(unittest.TestCase):
    def test_raw_backbone_identity_detects_changed_shards_and_runtime(self):
        with temporary_directory() as tmp:
            root = Path(tmp)
            for name in ("Qwen2.5-Omni-3B", "Qwen3-VL-8B-Instruct"):
                model = root / "models" / name
                model.mkdir(parents=True)
                (model / "config.json").write_text("{}", encoding="utf-8")
                (model / "model-00001.safetensors").write_bytes(b"test-shard")
            identity = {"base_artifacts": pipeline.base_inventory(root), "runtime": pipeline.runtime_identity()}
            pipeline.verify_environment(root, identity)
            model.joinpath("model-00001.safetensors").write_bytes(b"changed-shard")
            with self.assertRaisesRegex(ValueError, "backbone artifact"):
                pipeline.verify_environment(root, identity)
            with self.assertRaisesRegex(ValueError, "runtime changed"):
                pipeline.verify_environment(root, {"runtime": {"different": True}})

    def test_readonly_dashboard_shows_real_training_steps_and_telemetry(self):
        with temporary_directory() as tmp:
            root = Path(tmp)
            job = root / "cohort-64/jobs/C1"
            job.mkdir(parents=True)
            record = {"current_steps": 2, "total_steps": 10, "loss": 1.2,
                      "remaining_seconds_estimate": 180, "module_gradients": {"audio_perception": .003},
                      "training_observations": {"vision_n": 32, "vision_correct": 16}}
            log = job / "trainer_log.jsonl"
            log.write_text(json.dumps(record) + "\n{partial", encoding="utf-8")
            snapshot = {"status": {"phase": "content_C1"}, "clock": {},
                        "children": {"cohort-64/jobs/C1": {"phase": "training"}}}
            with patch.object(Path, "write_text", side_effect=AssertionError("Read-only")):
                rendered = progress.render(root, snapshot)
            for text in ("2/10", "20.0%", "loss=1.2000", "audio_perception", "16/32", "ETA: 00:03:00"):
                self.assertIn(text, rendered)
            snapshot["children"]["cohort-64/jobs/C1"]["phase"] = "predicting_held"
            self.assertNotIn("This training job ETA", progress.render(root, snapshot))


if __name__ == "__main__":
    unittest.main()
