"""CPU checks for frozen-fit verification, resumable questions, and run locks."""
import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from rl.experiment import file_hash, fingerprint
from scripts import run_evidence_supplement as app


class RunnerTests(unittest.TestCase):
    def test_calibration_endpoints_protocol_offsets_and_invalid_values(self):
        rows = [{"protocol": p} for p in ("VT", "VA", "VT", "VA")]
        q = np.array([0., 1., .5, .5])
        actual = app.calibrated_scores(q, rows, {"offsets": [1., -1.]})
        np.testing.assert_allclose(actual, [0., 1., 1/(1+np.exp(-1)), 1/(1+np.exp(1))])
        for invalid in ([0., 1.], [0., 1., float("nan"), .5], [0., 1., -.01, .5]):
            with self.subTest(q=invalid), self.assertRaises(ValueError):
                app.calibrated_scores(invalid, rows, {"offsets": [0., 0.]})

    def test_fit_marker_cannot_authorize_changed_parameter_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"head.json"
            c = {"identity_hash": "producer"}
            fit = {"identity_hash": "producer", "signature": "sig", "solver": {"success": True}, "parameters": [[0.]]}
            path.write_bytes(app.json_bytes(fit))
            marker = {"algorithm": app.PRODUCER_VERSION, "identity_hash": "producer", "signature": "sig", "sha256": file_hash(path)}
            Path(str(path)+".complete.json").write_bytes(app.json_bytes(marker))
            self.assertEqual(app.checked_fit(path, c, "sig"), fit)
            fit["parameters"][0][0] = 1.
            path.write_bytes(app.json_bytes(fit))
            with self.assertRaisesRegex(ValueError, "checksum"):
                app.checked_fit(path, c, "sig")

    def test_question_cache_reuses_exact_answer_without_loading_a_model(self):
        baseline = [{"sample_id": "a", "pred_emotion": "negative", "is_uncertain": "no", "clarifying_question": "none"}]
        row = {"sample_id": "a", "protocol": "VA", "video": "v.mp4", "text_content": "MUST NOT REACH MODEL", "gt_label": "positive"}
        decision = {"added_yes": True}
        called = []

        class Policy:
            model = SimpleNamespace(named_parameters=lambda: [("weight", SimpleNamespace(_version=0, requires_grad=False))])
            def answer(self, safe_row, split, c, source, old, uncertain):
                called.append(safe_row)
                self_check.assertEqual(set(safe_row), {"sample_id", "protocol", "video"})
                self_check.assertEqual((split, source, uncertain), ("test", None, "yes"))
                return json.dumps({"pred_emotion": "negative", "is_uncertain": "yes", "clarifying_question": "How are you feeling?"})

        self_check = self
        with tempfile.TemporaryDirectory() as tmp:
            identity = {"identity_hash": "run-a"}
            one = app.QuestionCache(tmp, {}, identity, baseline, Policy, lambda *a, **k: None)
            answer = one(row, decision)
            two = app.QuestionCache(tmp, {}, identity, baseline,
                lambda: self.fail("Resume should not load a model"), lambda *a, **k: None)
            self.assertEqual(two(row, decision), answer)
            self.assertEqual(len(called), 1)
            three = app.QuestionCache(tmp, {}, {"identity_hash": "run-b"}, baseline, Policy, lambda *a, **k: None)
            with self.assertRaisesRegex(ValueError, "different evidence/baseline"):
                three(row, decision)

    def test_question_cache_rejects_changed_emotion_or_model_weights(self):
        baseline = [{"sample_id": "a", "pred_emotion": "negative", "is_uncertain": "no", "clarifying_question": "none"}]
        row = {"sample_id": "a", "protocol": "VT", "video": "v", "text_content": "Good"}
        for change_emotion in (True, False):
            parameter = SimpleNamespace(_version=0, requires_grad=False)
            class Policy:
                model = SimpleNamespace(named_parameters=lambda: [("weight", parameter)])
                def answer(self, *args):
                    if not change_emotion:
                        parameter._version += 1
                    return json.dumps({"pred_emotion": "positive" if change_emotion else "negative",
                                       "is_uncertain": "yes", "clarifying_question": "How do you feel?"})
            with self.subTest(change_emotion=change_emotion), tempfile.TemporaryDirectory() as tmp:
                cache = app.QuestionCache(tmp, {}, {"identity_hash": "r"}, baseline, Policy, lambda *a, **k: None)
                with self.assertRaises((ValueError, RuntimeError)):
                    cache(row, {"added_yes": True})
                self.assertEqual(list(Path(tmp).glob("questions/*.json")), [])

    def test_context_and_candidate_validation_require_no_gpu_or_scipy(self):
        # The production audit reads cached arrays; the fit checker itself should
        # stay usable in the lightweight NumPy-only inspection environment.
        result = subprocess.run([sys.executable, "-c",
            "import sys; from scripts import run_evidence_supplement; "
            "assert 'torch' not in sys.modules; assert 'scipy' not in sys.modules"],
            cwd=app.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_baseline_snapshot_is_pinned_to_run_identity_before_gpu_work(self):
        from scripts import evidence_supplement
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/"source.csv"
            source.write_bytes(b"snapshot A")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            with patch.object(evidence_supplement, "read_baseline_snapshot", return_value=([{"sample_id": "a"}], digest, {"name": "confirmed"})):
                baseline, frozen, _ = app.freeze_baseline(root/"run", source, [], digest)
                self.assertEqual(frozen.read_bytes(), b"snapshot A")
                source.write_bytes(b"snapshot B")
                self.assertEqual(frozen.read_bytes(), b"snapshot A")
                with self.assertRaisesRegex(ValueError, "changed"):
                    app.freeze_baseline(root/"other", source, [], digest)
                self.assertFalse((root/"other/baseline-frozen.csv").exists())

    def test_run_lock_prevents_a_second_controller(self):
        with tempfile.TemporaryDirectory() as tmp, app.run_lock(tmp):
            with self.assertRaises(OSError):
                with app.run_lock(tmp):
                    self.fail("A second controller acquired the same run")

    def test_validation_audit_marks_question_checks_pending(self):
        rows = [{"sample_id": "a", "protocol": "VT", "sample_type": "Conflict", "source_dataset": "ch_sims", "gt_label": "negative"},
                {"sample_id": "b", "protocol": "VA", "sample_type": "Aligned", "source_dataset": "other", "gt_label": "positive"}]
        baseline = {r["sample_id"]: {"pred_emotion": r["gt_label"], "is_uncertain": "no", "clarifying_question": "none", "completion": "legacy"} for r in rows}
        a = np.array([[.8, .1, .1], [.8, .1, .1]])
        b = np.array([[.1, .8, .1], [.8, .1, .1]])
        q = 1-(a*b).sum(1)
        candidate = {"vision_probabilities": a.tolist(), "second_probabilities": b.tolist(),
                     "raw_disagreement": q.tolist(), "calibration": {"offsets": [0., 0.]}}
        candidate["val_probabilities"] = app.calibrated_scores(q, rows, candidate["calibration"]).tolist()
        audit = app.validation_audit({"val": rows, "baselines": {"val": baseline}, "identity_hash": "producer"}, candidate, {})
        for key in ("baseline", "supplement", "evidence_replacement"):
            self.assertIsNone(audit[key]["invalid_outputs"])
            self.assertIsNone(audit[key]["inconsistent_questions"])
            self.assertTrue(audit[key]["output_checks_pending"])
        self.assertTrue(audit["posthoc"])
        self.assertFalse(audit["independent_validation"])
        self.assertEqual(audit["supplement"]["counts"]["conflict_tp"], 1)

    def test_cli_rejects_unapproved_memory_and_output_inside_producer(self):
        with tempfile.TemporaryDirectory() as tmp:
            for extra in (["--memory-gib", "33"], ["--cpu-threads", "0"]):
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    app.main(["audit", "--source-run", tmp, "--output-dir", tmp, *extra])
            with self.assertRaisesRegex(ValueError, "immutable producer"):
                app.main(["audit", "--source-run", tmp, "--output-dir", str(Path(tmp)/"child")])


if __name__ == "__main__":
    unittest.main()
