import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.conflict_dashboard import render, render_parameters, valid_candidate


class ConflictDashboardTests(unittest.TestCase):
    def test_unlimited_context_shows_no_countdown_or_time_cutoff(self):
        self.write("context.json", {"started": 100, "deadline": None, "training_cutoff": None,
            "settings": {"steps": 420, "budget_hours": None, "training_cutoff_hours": None}})
        with patch("scripts.conflict_dashboard.time.time", return_value=10**12):
            text = render(self.run)
            parameters = render_parameters(self.run)
        self.assertIn("时间限制: 无", text)
        self.assertNotIn("剩余预算", text)
        self.assertNotIn("24h截止", text)
        self.assertIn("时间限制：无", parameters)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.write("context.json", {"started": 100, "deadline": 86500,
                                   "settings": {"steps": 420, "batch_size": 16}})
        self.gpu = patch("scripts.conflict_dashboard.gpu_summary", return_value="GPU mocked")
        self.gpu.start()
        self.addCleanup(self.gpu.stop)
        self.terminal = patch("scripts.conflict_dashboard.shutil.get_terminal_size",
                              return_value=type("Terminal", (), {"columns": 250})())
        self.terminal.start()
        self.addCleanup(self.terminal.stop)

    def write(self, path, value):
        target = self.run / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value), encoding="utf-8")

    def candidate(self, score=.74, **updates):
        metrics = {"score": score, "invalid_outputs": 0,
                   "inconsistent_questions": 0, "emotion_changes": 0}
        metrics.update(updates)
        return {"metrics": metrics, "checkpoint": "/run/seed-42/checkpoint-000105",
                "name": "seed-42-000105-1", "primary": {"eligible": True}}

    def test_invalid_or_missing_constraints_never_claim_pass(self):
        broken = self.candidate(emotion_changes=1)
        self.write("leaderboard-42.json", {"candidates": [broken]})
        self.write("train-42.done.json", {"reason": "primary_gate_not_met"})
        text = render(self.run)
        self.assertIn("seed42 未达标", text)
        self.assertIn("无效/约束未核验", text)
        self.assertIn("本轮最佳有效训练分: --", text)
        del broken["metrics"]["emotion_changes"]
        self.assertFalse(valid_candidate(broken))
        self.assertFalse(valid_candidate(self.candidate(score=float("nan"))))

    def test_best_and_baselines_remain_distinct_and_completed_keeps_loss(self):
        self.write("results/baseline-free.json", {"metrics": {"score": .71}})
        self.write("results/baseline-new.json", {"metrics": {"score": .72}})
        self.write("results/baseline.json", {"metrics": {"score": .72}})
        self.write("leaderboard-42.json", {"candidates": [self.candidate()]})
        self.write("train.status.json", {"phase": "complete", "step": 210, "seed": 42})
        (self.run / "train-42.jsonl").write_text(json.dumps({"loss": .45,
            "main_loss": .4, "pair_loss": .3, "kl": .05, "unique_main_seen": 3354})+"\n", encoding="utf-8")
        self.write("pipeline.status.json", {"phase": "complete", "reason": "primary_gate_not_met"})
        text = render(self.run)
        self.assertIn("LOSS: 0.450000", text)
        self.assertIn("独立覆盖: 3354/3354", text)
        self.assertIn("自由生成 0.710000", text)
        self.assertIn("字段判别 0.720000", text)
        self.assertIn("Δ +0.020000", text)
        self.assertIn("KAGGLE 已确认 Public: 0.90813", text)

    def test_smoke_uses_forty_updates_and_does_not_claim_formal_training(self):
        self.write("pipeline.status.json", {"phase": "40_step_learning_and_resume_check"})
        self.write("smoke.status.json", {"step": 12, "loss": .51})
        text = render(self.run)
        self.assertIn("临时可学习性检查: 参数更新 12/40", text)
        self.assertIn("40步可学习性: 进行中 12/40", text)
        self.assertNotIn("正式训练 seed", text)
        self.assertIn("seed42 等待", text)

    def test_budget_stop_is_explicit_and_never_a_quality_pass(self):
        self.write("pipeline.status.json", {"phase": "stopped", "reason": "training_cutoff"})
        with patch("scripts.conflict_dashboard.time.time", return_value=90100):
            text = render(self.run)
        self.assertIn("剩余预算 0.00 h", text)
        self.assertIn("第 20 小时，不再启动新训练阶段", text)
        self.assertIn("seed43复验 等待", text)
        self.assertIn("CSV: 未生成", text)
        parameters = render_parameters(self.run)
        self.assertIn("cosine horizon 420", parameters)
        self.assertIn("临时40更新", parameters)

    def test_rejected_smoke_keeps_its_metrics_and_never_claims_formal_start(self):
        self.write("pipeline.status.json", {"phase": "stopped", "reason": "smoke_rejected"})
        self.write("smoke.json", {"passed": False})
        self.write("smoke.status.json", {"phase": "rejected", "passed": False})
        (self.run / "smoke-updates.jsonl").write_text(json.dumps({"step": 40,
            "loss": .2, "main_loss": .19, "pair_loss": 0., "kl": .5})+"\n", encoding="utf-8")
        text = render(self.run)
        self.assertIn("临时可学习性检查: 参数更新 40/40", text)
        self.assertIn("LOSS: 0.200000", text)
        self.assertIn("40步可学习性: 未通过", text)
        self.assertIn("未启动正式训练", text)

    def test_replication_has_selected_limit_and_initial_epoch_stop_is_explained(self):
        self.write("pipeline.status.json", {"phase": "training_and_validation", "seed": 43, "maximum_steps": 105})
        self.write("train.status.json", {"phase": "complete", "step": 420, "seed": 42})
        text = render(self.run)
        self.assertIn("正式训练 seed 43: 参数更新 0/105", text)
        self.write("pipeline.status.json", {"phase": "complete", "reason": "primary_gate_not_met"})
        self.write("train.status.json", {"phase": "complete", "step": 210, "seed": 42,
                                         "reason": "first_epoch_gate_not_met"})
        text = render(self.run)
        self.assertIn("首轮验证收益不足 +0.003", text)
        self.assertNotIn("ERROR", text)

    def test_diagnostics_status_and_error_are_visible(self):
        self.write("pipeline.status.json", {"phase": "selected_modality_diagnostics"})
        self.write("diagnostics.status.json", {"phase": "unimodal_training_diagnostic", "completed": 50, "total": 3354})
        text = render(self.run)
        self.assertIn("unimodal_training_diagnostic 50/3354", text)
        self.write("diagnostics.error.json", {"error": "diagnostic model load failed"})
        self.assertIn("ERROR diagnostics: diagnostic model load failed", render(self.run))

    def test_post_smoke_training_only_diagnostic_shows_its_progress(self):
        self.write("pipeline.status.json", {"phase": "40_step_learning_and_resume_check"})
        self.write("smoke.status.json", {"phase": "training_only_diagnostic", "completed": 150, "total": 3322})
        (self.run / "smoke-updates.jsonl").write_text(json.dumps({"step": 40,
            "loss": .2})+"\n", encoding="utf-8")
        text = render(self.run)
        self.assertIn("参数更新 40/40", text)
        self.assertIn("training_only_diagnostic 150/3322", text)
        self.assertIn("LOSS: 0.200000", text)


if __name__ == "__main__":
    unittest.main()
