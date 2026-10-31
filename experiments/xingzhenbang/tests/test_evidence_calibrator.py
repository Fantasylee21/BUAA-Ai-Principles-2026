"""Numerical, leakage, grouping and lifecycle tests for CPU calibration."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import train_evidence_calibrator as app


def corpus(groups=60, prefix="train"):
    rows, media = [], {}
    for i in range(groups):
        video = f"videos/{prefix}-parent-{i // 3}/clip-{i}.mp4"
        media[video] = hashlib.sha256(video.encode()).hexdigest()
        vision = "positive" if i % 2 else "negative"
        conflict = i % 3 == 0
        other = ("negative" if vision == "positive" else "positive") if conflict else vision
        for protocol, mode in (("VT", "text"), ("VA", "audio")):
            rows.append({"sample_id": f"{prefix}-{i}-{protocol}", "protocol": protocol,
                "video": video, "text_content": "Supplied evidence", "source_dataset": "ch_sims_v2",
                "natural": True, "sample_type": "Conflict" if conflict else "Aligned",
                "gt_label": vision, "split_group_id": video, "use_in_main": False,
                "view_targets": {"vision": {"label": vision, "clear": True},
                    mode: {"label": other, "clear": True}}})
    return rows, media


class TinyBank:
    """Deterministic synthetic frozen features; records every requested ID."""
    def __init__(self, split="train", locked_candidate=None):
        self.split, self.requests, self.locked_candidate = split, [], locked_candidate

    def get(self, rows, mode):
        if self.split == "val" and self.locked_candidate is not None:
            if not Path(self.locked_candidate).exists():
                raise AssertionError("Validation accessed before candidate lock")
        self.requests.append((mode, [r["sample_id"] for r in rows]))
        hidden, native = [], []
        for row in rows:
            seed = int(hashlib.sha256((row["video"] + mode).encode()).hexdigest()[:8], 16)
            hidden.append(np.random.default_rng(seed).normal(size=6))
            scores = np.full(3, -2.)
            scores[("positive", "negative", "neutral").index(row["view_targets"][mode]["label"])] = 2.
            native.append(scores)
        return np.array(hidden), np.array(native)


class CalibratorTests(unittest.TestCase):
    def test_features_measure_disagreement_without_confusing_shared_entropy(self):
        rows, _ = corpus(1)
        v = np.array([[.8, .2, 0.], [1., 0., 0.]])
        m = np.array([[.8, .2, 0.], [0., 1., 0.]])
        f = app.six_features(v, m, rows)
        self.assertEqual(f.shape, (2, 6))
        self.assertEqual(f[0, 0], 0.)
        self.assertGreater(f[0, 3], 0.)
        self.assertEqual(f[1, 0], 1.)
        np.testing.assert_allclose(f[:, -1], [0., 1.])
        dirty = copy.deepcopy(rows)
        for row in dirty:
            row.update(gt_label="neutral", sample_type="Aligned", source_dataset="different", natural=False)
        np.testing.assert_array_equal(f, app.six_features(v, m, dirty))
        for bad in (np.ones((2, 3)), np.ones((2, 2)), [[float("nan"), 0., 0.], [1., 0., 0.]]):
            with self.assertRaises(ValueError):
                app.six_features(bad, m, rows)

    def test_argmax_gate_uses_strict_half_and_first_maximum(self):
        v = np.array([[.5, .5, 0.]] * 4)
        m = np.array([[.5, 0., .5], [0., .5, .5], [0., .5, .5], [0., .5, .5]])
        p = [.99, .5, np.nextafter(.5, 1.), np.nextafter(.5, 0.)]
        np.testing.assert_array_equal(app.disagreement_gate(v, m, p), [False, False, True, False])
        with self.assertRaises(ValueError):
            app.disagreement_gate(v, m, .7)

    def test_analytic_gradient_and_class_conditional_weighting(self):
        rng = np.random.default_rng(10)
        x, theta = rng.normal(size=(14, 6)), rng.normal(size=7)
        y = np.array([1] * 3 + [0] * 11)
        loss, gradient = app.logistic_objective(theta, x, y)
        numeric = np.zeros_like(theta)
        for i in range(len(theta)):
            delta = np.zeros_like(theta)
            delta[i] = 1e-6
            numeric[i] = (app.logistic_objective(theta + delta, x, y)[0] -
                          app.logistic_objective(theta - delta, x, y)[0]) / 2e-6
        np.testing.assert_allclose(gradient, numeric, atol=2e-8, rtol=2e-7)
        duplicate_x = np.concatenate((x[:3], x[:3], x[:3], x[3:]))
        duplicate_y = np.array([1] * 9 + [0] * 11)
        loss2, grad2 = app.logistic_objective(theta, duplicate_x, duplicate_y)
        self.assertAlmostEqual(loss, loss2, places=12)
        np.testing.assert_allclose(gradient, grad2, atol=1e-12)

    def test_calibration_scaler_has_no_held_out_information(self):
        rows, _ = corpus(9)
        x = np.zeros((len(rows), 6))
        x[:, 0] = [1. if r["sample_type"] == "Conflict" else 0. for r in rows]
        x[:, 2] = np.linspace(0., 1., len(rows))
        model = app.fit_logistic(x, rows)
        np.testing.assert_allclose(model["mean"], x.mean(axis=0))
        self.assertEqual(model["scale"][1], 1.)
        frozen = copy.deepcopy(model)
        probabilities = app.predict_logistic(np.array([[0., 0., .5, 0., 0., 0.],
                                                       [1., 0., .5, 0., 0., 0.]]), model)
        self.assertLess(probabilities[0], .5)
        self.assertGreater(probabilities[1], .5)
        app.predict_logistic(np.full((3, 6), 1e6), model)
        self.assertEqual(model, frozen)
        with self.assertRaises(ValueError):
            app.fit_logistic(x[:2], rows[:2])  # Only conflict rows.

    def test_optimizer_failure_is_not_published_as_a_success(self):
        rows, _ = corpus(6)
        x = np.zeros((len(rows), 6))
        failed = SimpleNamespace(success=False, x=np.zeros(7), message="iteration limit", nit=1, nfev=1)
        with patch.object(app, "minimize", return_value=failed), self.assertRaises(RuntimeError):
            app.fit_logistic(x, rows)
        failed.success = True  # A claimed success with a large gradient is still rejected.
        with patch.object(app, "minimize", return_value=failed), self.assertRaises(RuntimeError):
            app.fit_logistic(x, rows)

    def test_hash_video_and_parent_groups_are_transitive_and_never_cross_partitions(self):
        rows, media = corpus(75)
        # Identical bytes across two different parents must union both components.
        media[rows[8]["video"]] = media[rows[0]["video"]]
        for pressure in (False, True):
            design = app.group_design(rows, media, pressure=pressure)
            self.assertEqual(design, app.group_design(rows, media, pressure=pressure))
            self.assertEqual(design["groups"][0], design["groups"][1])
            self.assertEqual(design["groups"][0], design["groups"][8])
            if pressure:
                self.assertEqual(design["groups"][0], design["groups"][2])
                self.assertEqual(design["groups"][0], design["groups"][6])
            for part in design["nested"]:
                sets = [{design["groups"][i] for i in part[key]}
                        for key in ("train", "calibration", "held")]
                self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
                indices = [i for key in ("train", "calibration", "held") for i in part[key]]
                self.assertEqual(sorted(indices), list(range(len(rows))))
        broken = dict(media)
        broken[rows[0]["video"]] = "unknown"
        with self.assertRaises(ValueError):
            app.group_design(rows, broken)

    def test_gate_requires_pooled_gain_two_folds_and_natural_recall(self):
        rows, _ = corpus(9)
        outer = [i // 2 // 3 for i in range(len(rows))]
        control = np.zeros(len(rows), dtype=bool)
        candidate = np.array([r["sample_type"] == "Conflict" for r in rows])
        gate = app.development_gate(rows, control, candidate, outer)
        self.assertTrue(gate["passed"])
        self.assertGreater(gate["gain"], 0.)
        self.assertFalse(app.development_gate(rows, candidate, candidate, outer)["passed"])
        # A gain from synthetic conflicts cannot compensate a natural-conflict loss.
        changed = copy.deepcopy(rows)
        for i, row in enumerate(changed):
            row["natural"] = i == 0
            row["source_dataset"] = "ch_sims_v2" if i == 0 else "synthetic"
        before, after = np.zeros(len(rows), dtype=bool), candidate.copy()
        before[0], after[0] = True, False
        failed = app.development_gate(changed, before, after, outer)
        self.assertGreater(failed["gain"], 0.)
        self.assertFalse(failed["checks"]["natural_conflict_recall_does_not_regress"])
        self.assertFalse(failed["passed"])

    def test_one_winning_fold_cannot_pass_even_with_positive_pooled_gain(self):
        rows, _ = corpus(9)
        outer = [i // 6 for i in range(len(rows))]
        control, candidate = np.zeros(len(rows), dtype=bool), np.zeros(len(rows), dtype=bool)
        candidate[:2] = True  # Fold 0 corrects all conflicts: +.3.
        candidate[8:10] = True  # Fold 1 adds false positives: -.1.
        candidate[14:16] = True  # Fold 2 adds false positives: -.1.
        gate = app.development_gate(rows, control, candidate, outer)
        self.assertGreater(gate["gain"], 0.)
        self.assertTrue(gate["checks"]["natural_conflict_recall_does_not_regress"])
        self.assertFalse(gate["checks"]["at_least_two_folds_do_not_regress"])
        self.assertFalse(gate["passed"])

    def test_nontraining_rows_cannot_reach_either_calibration_objective(self):
        rows, _ = corpus(6)
        held, _ = corpus(6, prefix="held")
        c = {"identity_hash": "synthetic", "train": rows}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                app._fit_logistic_saved(tmp, c, np.zeros((len(held), 6)), held, "wrong-split")
            with self.assertRaises(ValueError):
                app._fit_control_saved(tmp, c, np.zeros(len(held)), held, "wrong-split")
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_immutable_complete_markers_detect_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fit.json"
            value = {"model": [1., 2.], "solver": {"success": True}}
            app.publish_artifact(path, value)
            app.publish_artifact(path, value)
            self.assertEqual(app.checked_artifact(path), value)
            with self.assertRaises(ValueError):
                app.publish_artifact(path, {"model": [2.]})
            path.write_text(json.dumps({"model": [7.]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                app.checked_artifact(path)

    def test_nested_development_reuses_old_evidence_fit_without_supervision_leakage(self):
        from rl.evidence_bottleneck import DEFAULTS
        rows, media = corpus(45)
        # Unclear examples remain in relation calibration and outer evaluation.
        for row in rows[:4]:
            for view in row["view_targets"].values():
                view["clear"] = False
        c = {"identity_hash": "synthetic-source", "train": rows, "settings": dict(DEFAULTS)}
        bank, design = TinyBank(), app.group_design(rows, media)
        with tempfile.TemporaryDirectory() as tmp, patch.object(app, "emit"):
            result = app.run_development(tmp, c, bank, design, "media")
            self.assertEqual(result["ids"], [r["sample_id"] for r in rows])
            self.assertEqual(len(result["new_probabilities"]), len(rows))
            self.assertTrue(result["supervised_group_independence"])
            for part, plan in zip(result["partitions"], design["nested"]):
                self.assertFalse(set(part["training_ids"]) & set(part["calibration_ids"]))
                self.assertFalse(set(part["training_ids"]) & set(part["held_ids"]))
                job = f"media-nested-fold{plan['fold']}"
                model = app.checked_artifact(Path(tmp) / "calibrators" / (job + ".json"))
                self.assertEqual(model["examples"], part["calibration_ids"])
                expected_clear = sum(rows[i]["protocol"] == "VT" and
                    rows[i]["view_targets"]["text"]["clear"] for i in plan["train"])
                self.assertEqual(part["clear_supervision_counts"]["text"], expected_clear)
            requests = len(bank.requests)
            repeated = app.run_development(tmp, c, bank, design, "media")
            self.assertEqual(repeated, result)
            self.assertEqual(len(bank.requests), requests)

    def test_final_model_is_locked_before_single_validation_report(self):
        from rl.evidence_bottleneck import DEFAULTS
        rows, media = corpus(30)
        val, _ = corpus(6, prefix="val")
        baseline = {r["sample_id"]: {"pred_emotion": r["gt_label"], "is_uncertain": "no"} for r in val}
        baseline[val[0]["sample_id"]]["is_uncertain"] = "yes"
        c = {"identity_hash": "synthetic-source", "train": rows, "val": val,
             "baselines": {"val": baseline}, "settings": dict(DEFAULTS)}
        design, bank = app.group_design(rows, media), TinyBank()
        with tempfile.TemporaryDirectory() as tmp, patch.object(app, "emit"):
            val_bank = TinyBank("val", Path(tmp) / "candidate.json")
            with patch("rl.evidence_store.FeatureBank", return_value=val_bank):
                result = app.final_candidate(tmp, "unused-source", c, bank, design)
                self.assertTrue(result["candidate_locked_before_validation"])
                self.assertFalse(result["validation_used_for_selection"])
                self.assertEqual(result["candidate_fused"]["macro_f1"], result["baseline"]["macro_f1"])
                self.assertEqual(result["candidate_fused"]["n_records"], len(val))
                locked = app.checked_artifact(Path(tmp) / "candidate.json")
                self.assertEqual(locked["logistic"]["examples"], [r["sample_id"] for r in rows])
                oof = app.checked_artifact(Path(tmp) / "final-calibration-oof.json")
                self.assertEqual(oof["ids"], [r["sample_id"] for r in rows])
                calls = len(val_bank.requests)
                repeated = app.final_candidate(tmp, "unused-source", c, bank, design)
                self.assertEqual(result, repeated)
                self.assertEqual(len(val_bank.requests), calls)
                self.assertFalse({sid for _, ids in bank.requests for sid in ids} & {r["sample_id"] for r in val})


if __name__ == "__main__":
    unittest.main()
