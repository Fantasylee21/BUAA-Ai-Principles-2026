"""Reward invariants, metric parity and model-selection regressions."""
import itertools
import json
import random
import unittest

from rl.metrics import compute_competition_score, evaluation_report, macro_f1, promotion_decision
from rl.reward import FrozenF1Reference, RewardEngine, parse_completion


def completion(emotion="positive", uncertain="no", question=None):
    return json.dumps({"pred_emotion": emotion, "is_uncertain": uncertain,
                       "clarifying_question": question if question is not None else
                       ("none" if uncertain == "no" else "你现在的感受是什么？")})


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.population = {"Conflict": 411/3354, "Aligned": 2943/3354}
        self.engine = RewardEngine(self.population)

    def test_whole_object_and_duplicate_keys(self):
        text = completion()
        self.assertIsNotNone(parse_completion(" \n" + text + "\n"))
        for value in ("Answer: " + text, text + text, "```json\n" + text + "\n```", "[]",
                      text.replace('"positive"', '"happy"'), text.replace('"no"', 'false'),
                      text[:-1], text.replace('"pred_emotion":', '"pred_emotion":"negative","pred_emotion":')):
            self.assertIsNone(parse_completion(value), value)

    def test_sampling_correction_preserves_expected_task_weights(self):
        for kind, expected in (("Conflict", .6), ("Aligned", .4)):
            uncertain = "yes" if kind == "Conflict" else "no"
            score = self.engine.score(completion(uncertain=uncertain), "positive", kind)
            self.assertAlmostEqual(score["uncertainty"], expected)
        expected_emotion = sum(.5*self.engine.score(completion(), "positive", t)["emotion"]
                               for t in self.population)
        self.assertAlmostEqual(expected_emotion, .5)
        # Constant-yes/no cannot obtain the uncertainty reward of a correct classifier.
        for strategy, expected in (("yes", .3), ("no", .2)):
            got = sum(.5*self.engine.score(completion(uncertain=strategy), "negative", t)["total"]
                      for t in self.population)
            self.assertAlmostEqual(got, expected)

    def test_invalid_output_never_beats_a_valid_wrong_answer(self):
        invalid = self.engine.score("garbage", "positive", "Conflict")["total"]
        for gold, pred, kind, uncertain, question in itertools.product(
                ("positive", "negative", "neutral"), ("positive", "negative", "neutral"),
                ("Conflict", "Aligned"), ("yes", "no"), ("none", "", "你怎么想？")):
            result = self.engine.score(completion(pred, uncertain, question), gold, kind)
            self.assertGreater(result["total"], invalid)

    def test_question_constraint_is_only_a_small_penalty(self):
        valid = self.engine.score(completion(uncertain="yes"), "positive", "Conflict")
        bad = self.engine.score(completion(uncertain="yes", question="none"), "positive", "Conflict")
        self.assertAlmostEqual(valid["total"]-bad["total"], .02)
        self.assertEqual(bad["emotion"], valid["emotion"])
        self.assertEqual(bad["uncertainty"], valid["uncertainty"])

    def test_reward_metadata_fails_closed(self):
        for gold, kind in ((None, "Conflict"), ("positive", None), ("happy", "Aligned")):
            with self.assertRaises(ValueError):
                self.engine.score("bad", gold, kind)
        with self.assertRaises(ValueError):
            RewardEngine(self.population, {"Conflict": .2, "Aligned": .8})
        with self.assertRaises(ValueError):
            self.engine(["prompt"], [completion()], [], ["Aligned"], ["id"])
        with self.assertRaises(ValueError):
            RewardEngine(self.population, version="f1_delta_v1")

    def test_conversational_completions(self):
        self.assertEqual(parse_completion([{"role": "assistant", "content": [
            {"type": "text", "text": completion()}]}]), json.loads(completion()))

    def test_f1_reference_is_frozen_and_correct_label_is_never_worse(self):
        rng = random.Random(42)
        labels = ("positive", "negative", "neutral")
        for _ in range(40):
            records = [{"sample_id": str(i), "gt_label": rng.choice(labels)} for i in range(30)]
            predictions = {r["sample_id"]: rng.choice([*labels, None]) for r in records}
            reference = FrozenF1Reference(records, predictions)
            snapshot = json.dumps(reference.matrix)
            for row in records:
                values = {c: reference.emotion_reward(row["sample_id"], row["gt_label"], c) for c in labels}
                self.assertGreaterEqual(values[row["gt_label"]] + 1e-12, max(values.values()))
                self.assertTrue(all(-1 <= value <= 1 for value in values.values()))
            self.assertEqual(json.dumps(reference.matrix), snapshot)
        with self.assertRaises(ValueError):
            FrozenF1Reference(records, {**predictions, "test_id": "positive"})


class MetricTests(unittest.TestCase):
    def test_original_ch_sims_v2_source_is_included_in_natural_conflict_guard(self):
        rows = [
            {"gt_label": "positive", "pred_emotion": "positive", "sample_type": "Conflict",
             "is_uncertain": "no", "source_dataset": "ch_sims_v2", "protocol": "VA"},
            {"gt_label": "negative", "pred_emotion": "negative", "sample_type": "Aligned",
             "is_uncertain": "no", "source_dataset": "ch_sims_v2", "protocol": "VT"},
            {"gt_label": "positive", "pred_emotion": "positive", "sample_type": "Conflict",
             "is_uncertain": "yes", "source_dataset": "synthetic", "protocol": "VT"}]
        result = evaluation_report(rows)
        self.assertEqual(result["natural_conflict_recall"], 0)
        self.assertEqual(result["conflict_accuracy"], .5)

    def test_fixed_class_metric_cannot_silently_turn_into_present_class_metric(self):
        rows = [{"gt_label": "positive", "pred_emotion": "positive", "sample_type": "Conflict", "is_uncertain": "yes"},
                {"gt_label": "negative", "pred_emotion": "negative", "sample_type": "Aligned", "is_uncertain": "no"}]
        metrics = compute_competition_score(rows)
        self.assertAlmostEqual(metrics["macro_f1"], 2/3)
        self.assertEqual(metrics["macro_f1_union_diagnostic"], 1)
        self.assertAlmostEqual(metrics["score"], 5/6)
        self.assertEqual(metrics["score_union_diagnostic"], 1)

    def test_invalid_predictions_stay_in_denominators(self):
        self.assertAlmostEqual(macro_f1(["positive", "positive"], ["positive", None]), 2/9)
        rows = [{"gt_label": "positive", "pred_emotion": None, "sample_type": "Conflict", "is_uncertain": None}]
        result = compute_competition_score(rows)
        self.assertEqual(result["conflict_accuracy"], 0)
        self.assertIsNone(result["score"])
        self.assertEqual(result["invalid_outputs"], 1)
        self.assertIsNone(compute_competition_score([])["score"])

    def test_selection_requires_actual_score_gain_and_no_quality_regression(self):
        baseline = {"score": .72, "macro_f1": .63, "aligned_calibration": .95,
                    "natural_conflict_recall": .4}
        candidate = {**baseline, "score": .73}
        self.assertTrue(promotion_decision(baseline, candidate)["promote"])
        for change in ({"score": .722}, {"macro_f1": .62}, {"aligned_calibration": .93},
                       {"natural_conflict_recall": .3}, {"invalid_outputs": 1},
                       {"inconsistent_questions": 1}):
            self.assertFalse(promotion_decision(baseline, {**candidate, **change})["promote"])


if __name__ == "__main__":
    unittest.main()
