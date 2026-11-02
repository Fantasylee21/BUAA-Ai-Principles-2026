"""Controller gates must prevent an incomplete stage from launching training."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_rl_f1_pipeline import build_plan, execute_plan


class F1PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        rows = [{"sample_id": str(i), "video": f"{i}.mp4", "protocol": "VT",
                 "gt_label": "positive", "sample_type": kind}
                for i, kind in enumerate(("Conflict", "Aligned"))]
        (self.root / "train.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        self.plan = build_plan(self.root, self.root / "base", self.root / "sft",
                               "GPU-allocated", self.root / "pipeline")

    def complete_stage(self, stage, log):
        directory = Path(stage["run_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        status = {"status": "complete", "completed_steps": stage["target"]}
        (directory / "status.json").write_text(json.dumps(status), encoding="utf-8")
        if stage["stage"] == "cache_train":
            (directory / "f1_cache.json").write_text(json.dumps({
                "schema": "train_f1_cache_v1", "split": "train",
                "predictions": {"0": "positive", "1": "positive"}}), encoding="utf-8")
        else:
            marker = directory / f"checkpoint-{stage['target']:06d}" / "complete.json"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("{}", encoding="utf-8")

    def test_gpu_and_fresh_sft_start_are_preserved(self):
        cache, smoke, reload, formal = self.plan["stages"]
        for stage in self.plan["stages"]:
            command = stage["command"]
            self.assertEqual(command[command.index("--gpu") + 1], "GPU-allocated")
            self.assertIn(str((self.root / "sft").resolve()), command)
        self.assertEqual(formal["target"], 200)
        self.assertIn("--f1-cache", formal["command"])
        self.assertNotIn("--resume-from-checkpoint", formal["command"])
        self.assertNotIn("--policy-path", formal["command"])
        self.assertIn("--resume-from-checkpoint", reload["command"])
        self.assertEqual(smoke["run_dir"], reload["run_dir"])
        self.assertNotEqual(smoke["run_dir"], formal["run_dir"])

    def test_successful_stages_run_in_order(self):
        called = []
        def launch(stage, log):
            called.append(stage["stage"])
            self.complete_stage(stage, log)
        execute_plan(self.plan, launch)
        self.assertEqual(called, ["cache_train", "smoke", "smoke_reload", "formal_f1"])
        status = json.loads((self.root / "pipeline/pipeline-status.json").read_text())
        self.assertEqual(status["status"], "complete")

    def test_child_failure_stops_the_pipeline(self):
        called = []
        def launch(stage, log):
            called.append(stage["stage"])
            raise RuntimeError("cache inference failed")
        with self.assertRaisesRegex(RuntimeError, "cache inference failed"):
            execute_plan(self.plan, launch)
        self.assertEqual(called, ["cache_train"])
        status = json.loads((self.root / "pipeline/pipeline-status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["stage"], "cache_train")

    def test_zero_exit_but_incomplete_cache_does_not_start_smoke(self):
        called = []
        def launch(stage, log):
            called.append(stage["stage"])
            self.complete_stage(stage, log)
            path = Path(stage["run_dir"]) / "f1_cache.json"
            cache = json.loads(path.read_text())
            cache["predictions"].pop("1")
            path.write_text(json.dumps(cache), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "cache is missing or incomplete"):
            execute_plan(self.plan, launch)
        self.assertEqual(called, ["cache_train"])

    def test_stopped_smoke_does_not_start_formal_training(self):
        called = []
        def launch(stage, log):
            called.append(stage["stage"])
            self.complete_stage(stage, log)
            if stage["stage"] == "smoke":
                (Path(stage["run_dir"]) / "status.json").write_text(
                    json.dumps({"status": "stopped", "completed_steps": 3}), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "did not complete"):
            execute_plan(self.plan, launch)
        self.assertEqual(called, ["cache_train", "smoke"])

    def test_existing_pipeline_is_never_overwritten(self):
        directory = self.root / "pipeline"
        directory.mkdir()
        sentinel = directory / "keep.txt"
        sentinel.write_text("existing run", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            execute_plan(self.plan, self.complete_stage)
        self.assertEqual(sentinel.read_text(), "existing run")


if __name__ == "__main__":
    unittest.main()
