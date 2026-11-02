"""Promotion and delivery contracts for expensive detached server jobs."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts import server_training_batch as batch


class ServerBatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = SimpleNamespace(project_root=self.root, batch_dir=self.root / "runs/batch",
            seed42_run=self.root / "runs/seed42", seed43_run=self.root / "runs/seed43",
            final_run=self.root / "runs/final", vision_run=self.root / "runs/vision",
            audio_cache_dir=self.root / "audio", gpu="1", command="protocol", poll_seconds=.001, max_wait_hours=1.)
        batch.check_paths(self.args)
        self.args.seed42_run.mkdir()
        self.write(self.args.seed42_run / "runner.status.json", {"phase": "complete_two_arm_comparison"})

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def run_protocol(self, gates):
        with patch.object(batch, "plan", return_value={"frozen": True}), \
                patch("scripts.protocol_sft_summary.summarize", side_effect=[{"gate": {"passed": x}} for x in gates]), \
                patch.object(batch, "run_command") as command, patch.object(batch, "require_free_gpu"), \
                patch.object(batch, "checked_final_result", return_value={"status": "complete"}):
            result = batch.protocol(self.args)
        return result, command

    def test_rejected_seed42_never_launches_more_training(self):
        result, command = self.run_protocol([False])
        self.assertEqual(result["reason"], "seed42_gate_not_met")
        self.assertFalse(result["final_model_trained"])
        command.assert_not_called()

    def test_rejected_replication_cannot_start_full_training(self):
        result, command = self.run_protocol([True, False])
        self.assertEqual(result["reason"], "seed43_gate_not_met")
        self.assertTrue(all("scripts.protocol_sft_final" not in call.args[1] for call in command.call_args_list))

    def test_full_training_requires_both_gates_and_has_no_submission_command(self):
        result, command = self.run_protocol([True, True])
        self.assertTrue(result["final_model_trained"])
        launched = [call.args[1] for call in command.call_args_list]
        self.assertEqual(sum("scripts.protocol_sft_final" in value for value in launched), 1)
        self.assertFalse(result["submitted_to_kaggle"])
        self.assertFalse(any("kaggle" in token for value in launched for token in value))

    def test_prior_failed_training_does_not_launch_a_duplicate(self):
        self.write(self.args.seed42_run / "runner.status.json", {"phase": "failed"})
        result, command = self.run_protocol([])
        self.assertEqual(result["status"], "failed")
        command.assert_not_called()

    def test_transient_status_write_is_not_a_failure(self):
        path = self.root / "partial.json"
        path.write_text('{"phase":')
        self.assertEqual(batch.read_status(path), {})

    def test_archive_keeps_weights_media_out_and_projects_detail_fields(self):
        self.args.command = "collect"
        self.write(self.args.batch_dir / "protocol-result.json", {"status": "complete", "reason": "seed42_gate_not_met"})
        self.write(self.args.vision_run / "pipeline.status.json", {"phase": "complete"})
        report = {"status": "complete", "validation_used": False, "test_used": False}
        report_path = self.args.vision_run / "evaluation/final-report.json"
        self.write(report_path, report)
        self.write(Path(str(report_path) + ".complete.json"), {"sha256": batch.file_hash(report_path),
                   "payload_fingerprint": batch.fingerprint(report)})
        self.write(self.args.seed42_run / "control/holdout-report.json", {"score": .7})
        weights = self.args.seed42_run / "control/checkpoints/adapter_model.safetensors"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"MODEL-MUST-STAY-ON-SERVER")
        media = self.args.seed42_run / "shared-media/video.mp4"
        media.parent.mkdir()
        media.write_bytes(b"MEDIA-MUST-STAY-ON-SERVER")
        details = self.args.seed42_run / "control/evaluation/checkpoint-1_holdout_details.jsonl"
        self.write(details, {"sample_id": "x", "pred_emotion": "negative", "input_prompt": "PRIVATE_FULL_PROMPT", "raw_response": "LONG_RESPONSE"})
        result = batch.collect(self.args)
        self.assertEqual(result["status"], "complete")
        with tarfile.open(self.args.batch_dir / "review-results.tar.gz") as tar:
            names = tar.getnames()
            self.assertFalse(any(name.endswith((".safetensors", ".mp4")) for name in names))
            projected = next(name for name in names if name.endswith("_details.jsonl"))
            content = tar.extractfile(projected).read()
            self.assertIn(b'"sample_id": "x"', content)
            self.assertNotIn(b"PRIVATE_FULL_PROMPT", content)
            self.assertNotIn(b"LONG_RESPONSE", content)
        self.assertEqual(batch.collect(self.args), result)

    def test_status_without_visual_report_cannot_publish_complete(self):
        self.args.command = "collect"
        self.write(self.args.batch_dir / "protocol-result.json", {"status": "complete"})
        self.write(self.args.vision_run / "pipeline.status.json", {"phase": "complete"})
        with self.assertRaises(FileNotFoundError):
            batch.collect(self.args)
        self.assertFalse((self.args.batch_dir / "COMPLETE.json").exists())

    def test_claimed_final_model_without_verified_files_cannot_publish_complete(self):
        self.args.command = "collect"
        self.write(self.args.batch_dir / "protocol-result.json", {"status": "complete", "final_model_trained": True})
        self.write(self.args.vision_run / "pipeline.status.json", {"phase": "failed"})
        with self.assertRaises(FileNotFoundError):
            batch.collect(self.args)
        self.assertFalse((self.args.batch_dir / "COMPLETE.json").exists())


if __name__ == "__main__":
    unittest.main()
