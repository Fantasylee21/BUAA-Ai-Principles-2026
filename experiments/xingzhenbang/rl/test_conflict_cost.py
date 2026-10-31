"""Objective and data invariants independent of the large Omni model."""
from collections import Counter
import json
import os
from pathlib import Path
import tempfile
import unittest

from rl.conflict_cost import (DEFAULTS, EpochSampler, choose_baseline, eligible,
                             main_loss, pair_loss, settings_from)


def synthetic_training():
    records = []
    for i in range(30):
        for protocol, kind in (("VT", "Conflict"), ("VA", "Aligned")):
            records.append({"sample_id": f"{i:02d}:{protocol}", "protocol": protocol,
                "sample_type": kind, "split_group_id": f"group-{i:02d}",
                "video": f"video-{i}.mp4", "split": "train"})
    return records


class CostObjectiveTests(unittest.TestCase):
    def test_uniform_weighted_sum_matches_full_class_risk_and_gradient(self):
        import torch
        import torch.nn.functional as F
        scores = torch.tensor([[-3., 2.], [4., -1.], [1., 3.], [6., 1.], [-8., 2.]],
                              dtype=torch.float64, requires_grad=True)
        counts = {"Conflict": 2, "Aligned": 3}
        calculated = sum(main_loss(row, "yes" if i < 2 else "no", counts, 5)
                         for i, row in enumerate(scores))
        d = scores[:, 0] - scores[:, 1]
        exact = .6 * F.softplus(-d[:2]).mean() + .4 * F.softplus(d[2:]).mean()
        self.assertTrue(torch.allclose(calculated, exact, atol=1e-14, rtol=0))
        actual_grad, = torch.autograd.grad(calculated, scores, retain_graph=True)
        exact_grad, = torch.autograd.grad(exact, scores)
        self.assertTrue(torch.allclose(actual_grad, exact_grad, atol=1e-14, rtol=0))
        # A short final batch is normalized by its actual size, not padded to 16.
        short = main_loss(scores[0], "yes", counts, 1)
        self.assertAlmostEqual(float(short.detach()), 5 * float(main_loss(scores[0], "yes", counts, 5).detach()))

    def test_extreme_wrong_logits_keep_correcting_gradients(self):
        import torch
        for target, values in (("yes", [-1000., 1000.]), ("no", [1000., -1000.])):
            score = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            loss = main_loss(score, target, {"Conflict": 1, "Aligned": 1}, 1)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            correct = 0 if target == "yes" else 1
            self.assertLess(float(score.grad[correct]), -.7)
            self.assertGreater(float(score.grad[1-correct]), .7)
        correct = torch.tensor([1000., -1000.], requires_grad=True)
        main_loss(correct, "yes", {"Conflict": 1, "Aligned": 1}, 1).backward()
        self.assertEqual(float(correct.grad.abs().max()), 0.)

    def test_pair_gradient_and_two_pass_vjp_match_joint_computation(self):
        import torch
        # Shared parameters make this representative of scoring pair members in
        # separate forward/backward passes without retaining two model graphs.
        matrix = torch.tensor([[1., 2.], [-1., .3]], dtype=torch.float64)
        wc = torch.tensor([.2, -.7], dtype=torch.float64)
        wa = torch.tensor([-1., .6], dtype=torch.float64)
        theta = matrix.clone().requires_grad_()
        c, a = theta @ wc, theta @ wa
        expected_loss = .1 * pair_loss(c, a)
        expected_loss.backward()
        expected_grad = theta.grad.clone()
        detached_c = c.detach().requires_grad_()
        detached_a = a.detach().requires_grad_()
        scalar = .1 * pair_loss(detached_c, detached_a)
        gc, ga = torch.autograd.grad(scalar, (detached_c, detached_a))
        theta.grad = None
        (theta @ wc).backward(gc)
        (theta @ wa).backward(ga)
        self.assertTrue(torch.allclose(theta.grad, expected_grad, atol=1e-14, rtol=0))
        self.assertLess(float(gc[0]), 0.)
        self.assertGreater(float(gc[1]), 0.)
        self.assertGreater(float(ga[0]), 0.)
        self.assertLess(float(ga[1]), 0.)

    def test_loss_rejects_wrong_targets_or_incomplete_counts(self):
        import torch
        scores = torch.tensor([0., 1.])
        for target, counts, batch in (("unknown", {"Conflict": 1, "Aligned": 1}, 1),
                ("yes", {"Conflict": 1}, 1), ("yes", {"Conflict": 1, "Aligned": 0}, 1),
                ("yes", {"Conflict": 1, "Aligned": 1}, 0)):
            with self.assertRaises(ValueError):
                main_loss(scores, target, counts, batch)


class CostSamplingTests(unittest.TestCase):
    def test_epochs_cover_each_row_once_and_resume_without_rng_state(self):
        rows = synthetic_training()
        sampler = EpochSampler(rows, 42)
        resumed = EpochSampler(list(reversed(rows)), 42)
        self.assertEqual(sampler.epoch_updates, 4)
        draw_ids = []
        orders = []
        for epoch in range(2):
            draws = [slot for step in range(epoch * 4, (epoch + 1) * 4) for slot in sampler.batch(step)]
            ids = [s["row"]["sample_id"] for s in draws]
            self.assertEqual(Counter(ids), Counter(r["sample_id"] for r in rows))
            self.assertEqual(len(sampler.batch(epoch * 4 + 3)), 12)
            self.assertTrue(all(s["selection"] == "epoch_uniform" for s in draws))
            draw_ids.extend(s["draw_id"] for s in draws)
            orders.append(ids)
        self.assertNotEqual(*orders)
        self.assertEqual(len(set(draw_ids)), 120)
        for step in (0, 3, 4, 7, 209, 210, 419):
            self.assertEqual(sampler.batch(step), resumed.batch(step))
            self.assertEqual(sampler.pair(step), resumed.pair(step))
        self.assertNotEqual(sampler.batch(0), EpochSampler(rows, 43).batch(0))

    def test_pair_cycles_visit_all_30_with_opposite_labels_same_video(self):
        sampler = EpochSampler(synthetic_training(), 42)
        cycles = []
        for cycle in range(2):
            pairs = [sampler.pair(cycle * 30 + i) for i in range(30)]
            self.assertEqual(len({p["pair_id"] for p in pairs}), 30)
            for pair in pairs:
                self.assertEqual(pair["cycle"], cycle)
                self.assertEqual(pair["conflict"]["sample_type"], "Conflict")
                self.assertEqual(pair["aligned"]["sample_type"], "Aligned")
                self.assertNotEqual(pair["conflict"]["protocol"], pair["aligned"]["protocol"])
                self.assertEqual(pair["conflict"]["video"], pair["aligned"]["video"])
            cycles.append([p["pair_id"] for p in pairs])
        self.assertNotEqual(*cycles)

    def test_invalid_membership_duplicate_and_pair_media_are_rejected(self):
        for change in (lambda r: r + [r[0]], lambda r: [{**r[0], "split": "val"}] + r[1:],
                       lambda r: [{**r[0], "video": "different.mp4"}] + r[1:], lambda r: r[:-2]):
            with self.assertRaises(ValueError):
                EpochSampler(change(synthetic_training()), 42)
        sampler = EpochSampler(synthetic_training(), 42)
        for step in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                sampler.batch(step)

    def test_actual_dataset_full_epoch_and_original_pairs(self):
        from rl.grounded_uncertainty import load_training
        path = Path(os.environ.get("BUAA_DATA_ROOT", Path(__file__).resolve().parents[1] / "Dataset")) / "train.jsonl"
        if not path.is_file():
            self.skipTest("Original training data is unavailable")
        sampler = EpochSampler(load_training(path), 42)
        self.assertEqual(sampler.counts, {"Conflict": 411, "Aligned": 2943})
        self.assertEqual(sampler.epoch_updates, 210)
        ids = [slot["row"]["sample_id"] for step in range(210) for slot in sampler.batch(step)]
        self.assertEqual(len(ids), 3354)
        self.assertEqual(len(set(ids)), 3354)
        self.assertEqual(len(sampler.batch(209)), 10)


class CostSelectionTests(unittest.TestCase):
    def test_select_stronger_baseline_and_fixed_settings(self):
        free, new = {"score": .7}, {"score": .72}
        self.assertIs(choose_baseline(free, new), new)
        self.assertIs(choose_baseline(free, {"score": .7}), free)
        for value in (float("nan"), float("inf"), -.1, 1.1):
            with self.assertRaises(ValueError):
                choose_baseline(free, {"score": value})
        self.assertEqual(settings_from(), DEFAULTS)
        config = Path(__file__).resolve().parents[1] / "configs/rl/conflict_cost_v1.json"
        self.assertEqual(settings_from(config), DEFAULTS)
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "wrong.json"
            file.write_text(json.dumps({"beta": .04}), encoding="utf-8")
            with self.assertRaises(ValueError):
                settings_from(file)

    def test_score_tradeoff_has_no_fixed_false_positive_or_subgroup_gate(self):
        base = {"score": .7}
        candidate = {"score": .715, "invalid_outputs": 0, "inconsistent_questions": 0,
            "emotion_changes": 0, "counts": {"natural_conflict": {"correct": 0},
            "va_conflict": {"correct": 0}, "aligned_false_positive": {"correct": 40}}}
        self.assertTrue(eligible(base, candidate, "primary")["eligible"])
        for stage, minimum in (("early", .003), ("primary", .015), ("replication", .005)):
            self.assertTrue(eligible(base, {**candidate, "score": .7 + minimum}, stage)["eligible"])
            self.assertFalse(eligible(base, {**candidate, "score": .7 + minimum - .00001}, stage)["eligible"])
        for change in ({"emotion_changes": 1}, {"invalid_outputs": 1}, {"inconsistent_questions": 1},
                       {"score": float("nan")}, {"score": float("inf")}):
            self.assertFalse(eligible(base, {**candidate, **change})["eligible"])


if __name__ == "__main__":
    unittest.main()
