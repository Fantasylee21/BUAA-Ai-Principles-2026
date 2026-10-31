"""Frozen-baseline supplement contracts, including immutable submission output."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import evidence_supplement as app


def inputs(count=4):
    rows = [{"sample_id": f"s-{i}", "protocol": "VT" if i % 2 == 0 else "VA",
             "video": f"videos/{i}.mp4", "text_content": "A supplied sentence"}
            for i in range(count)]
    baseline = [{"sample_id": row["sample_id"],
                 "pred_emotion": ("positive", "negative", "neutral")[i % 3],
                 "is_uncertain": "no", "clarifying_question": "none"}
                for i, row in enumerate(rows)]
    vision = np.tile([.8, .1, .1], (count, 1))
    second = np.tile([.1, .8, .1], (count, 1))
    scores = np.full(count, .75)
    return rows, baseline, vision, second, scores


def callback(row, decision):
    return f"Could you clarify how you feel about sample {row['sample_id']}?"


def baseline_map(baseline):
    return {row["sample_id"]: row for row in baseline}


class SupplementTests(unittest.TestCase):
    def test_fixed_or_truth_table_preserves_every_emotion_and_old_yes(self):
        cases = [(old_yes, different, score)
                 for old_yes in (False, True)
                 for different in (False, True)
                 for score in (.49, .5, .51)]
        rows, baseline, vision, second, scores = inputs(len(cases))
        for i, (old_yes, different, score) in enumerate(cases):
            baseline[i]["is_uncertain"] = "yes" if old_yes else "no"
            baseline[i]["clarifying_question"] = f" Existing question {i}?  " if old_yes else "none"
            second[i] = [.1, .8, .1] if different else [.8, .1, .1]
            scores[i] = score
        original = copy.deepcopy((rows, baseline))
        arrays = [value.copy() for value in (vision, second, scores)]
        called = []

        def generate(row, decision):
            called.append(row["sample_id"])
            self.assertTrue(decision["added_yes"])
            return callback(row, decision)

        outputs, changes = app.build_submission(rows, baseline, vision, second, scores, generate)
        expected_added = []
        for i, (old_yes, different, score) in enumerate(cases):
            expected = old_yes or (different and score > .5)
            self.assertEqual(outputs[i]["sample_id"], rows[i]["sample_id"])
            self.assertEqual(outputs[i]["pred_emotion"], baseline[i]["pred_emotion"])
            self.assertEqual(outputs[i]["is_uncertain"], "yes" if expected else "no")
            self.assertEqual(set(outputs[i]), set(app.FIELDS))
            if old_yes or not expected:
                self.assertEqual(outputs[i], baseline[i])
            else:
                expected_added.append(rows[i]["sample_id"])
                self.assertEqual(outputs[i]["clarifying_question"], callback(rows[i], {}))
        self.assertEqual(called, expected_added)
        self.assertEqual([r["sample_id"] for r in changes], expected_added)
        self.assertEqual((rows, baseline), original)
        for actual, before in zip((vision, second, scores), arrays):
            np.testing.assert_array_equal(actual, before)

    def test_half_boundary_is_strict_and_argmax_ties_are_deterministic(self):
        rows, baseline, vision, second, scores = inputs(5)
        scores[:] = [.5, np.nextafter(.5, 1.), np.nextafter(.5, 0.), .99, .99]
        vision[3:] = [.5, .5, 0.]
        second[3] = [.5, 0., .5]  # Both choose first emotion on a tie.
        second[4] = [0., .5, .5]  # First maxima differ.
        decisions = app.supplement_decisions(rows, baseline_map(baseline), vision, second, scores)
        self.assertEqual([r["is_uncertain"] for r in decisions], ["no", "yes", "no", "no", "yes"])
        self.assertEqual(decisions[3]["vision_emotion"], "positive")
        self.assertEqual(decisions[3]["second_emotion"], "positive")
        self.assertEqual(decisions[4]["second_emotion"], "negative")

    def test_high_entropy_same_argmax_never_creates_uncertainty(self):
        rows, baseline, vision, second, scores = inputs(2)
        vision[:] = [.34, .33, .33]
        second[:] = [.35, .33, .32]
        scores[:] = [.99, 1.]
        with patch.object(app, "THRESHOLD", .5):
            outputs, changes = app.build_submission(
                rows, baseline, vision, second, scores,
                lambda *_: self.fail("No added yes means no question generation"))
        self.assertEqual(outputs, baseline)
        self.assertEqual(changes, [])

    def test_baseline_mapping_may_omit_redundant_sample_id(self):
        rows, baseline, vision, second, scores = inputs(2)
        mapping = {row["sample_id"]: {k: v for k, v in row.items() if k != "sample_id"}
                   for row in baseline}
        decisions = app.supplement_decisions(rows, mapping, vision, second, scores)
        self.assertEqual([r["sample_id"] for r in decisions], [r["sample_id"] for r in rows])

    def test_question_generator_receives_only_label_free_input(self):
        rows, baseline, vision, second, scores = inputs(1)
        rows[0].update(gt_label="negative", sample_type="Conflict", source_dataset="private",
                       views={"M1": {"label": "negative"}}, natural=True)
        received = []

        def generate(row, decision):
            received.append(row)
            self.assertEqual(set(row), {"sample_id", "protocol", "video", "text_content"})
            self.assertFalse({"gt_label", "sample_type", "source_dataset", "views", "natural"} & set(decision))
            return callback(row, decision)

        app.build_submission(rows, baseline, vision, second, scores, generate)
        self.assertEqual(len(received), 1)

    def test_va_question_generator_never_receives_supplied_text(self):
        rows, baseline, vision, second, scores = inputs(2)
        rows[1].update(text="FORBIDDEN VA TEXT", gt_label="negative", source_dataset="private")
        received = []
        def generate(row, decision):
            received.append(row)
            return callback(row, decision)
        app.build_submission(rows, baseline, vision, second, scores, generate)
        self.assertIn("text_content", received[0])
        self.assertEqual(set(received[1]), {"sample_id", "protocol", "video"})

    def test_confirmed_baseline_switch_during_questions_cancels_all_artifacts(self):
        rows, baseline, vision, second, scores = inputs(734)
        scores[:] = .5
        scores[0] = .75
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/"source.csv"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=app.FIELDS)
                writer.writeheader()
                writer.writerows(baseline)
            first = source.read_bytes()
            second_bytes = first.replace(b"positive", b"negative", 1)
            confirmed = {hashlib.sha256(data).hexdigest(): {"name": name, "public_score": .9}
                         for data, name in ((first, "A.csv"), (second_bytes, "B.csv"))}
            reads = []
            original_read = Path.read_bytes
            def count_reads(path):
                if path == source:
                    reads.append(path)
                return original_read(path)
            with patch.dict(app.CONFIRMED_BASELINES, confirmed), patch.object(Path, "read_bytes", count_reads):
                snapshot, digest, _ = app.read_baseline_snapshot(source, rows)
                self.assertEqual(len(reads), 1)
                self.assertEqual(snapshot, baseline)
                self.assertEqual(digest, hashlib.sha256(first).hexdigest())
                def switch(row, decision):
                    source.write_bytes(second_bytes)
                    return callback(row, decision)
                with self.assertRaisesRegex(ValueError, "changed"):
                    app.export_submission(root/"candidate", rows, source, vision, second, scores, switch, {})
            self.assertEqual(list((root/"candidate").glob("submission*")), [])

    def test_invalid_probability_geometry_or_values_fail_before_questions(self):
        rows, baseline, vision, second, scores = inputs(2)
        invalid = [np.ones((2, 2)), np.ones((1, 3)), np.ones((2, 3, 1)),
                   np.array([[.8, .1, .1], [.8, -.1, .3]]),
                   np.array([[.8, .1, .1], [.8, .1, .2]]),
                   np.array([[.8, .1, .1], [float("nan"), .1, .1]]),
                   np.array([[.8, .1, .1], [float("inf"), .1, .1]])]
        for which in ("vision", "second"):
            for bad in invalid:
                with self.subTest(which=which, shape=bad.shape, value=bad.tolist()):
                    arguments = (bad, second) if which == "vision" else (vision, bad)
                    with self.assertRaises(ValueError):
                        app.build_submission(rows, baseline, *arguments, scores,
                                             lambda *_: self.fail("Invalid evidence reached generation"))

    def test_invalid_calibrated_probability_fails_before_questions(self):
        rows, baseline, vision, second, _ = inputs(2)
        for scores in ([.6], [[.6], [.7]], [.6, -.01], [.6, 1.01],
                       [.6, float("nan")], [.6, float("inf")]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                app.build_submission(rows, baseline, vision, second, scores,
                                     lambda *_: self.fail("Invalid scores reached generation"))

    def test_duplicate_missing_extra_and_reordered_ids_fail(self):
        rows, baseline, vision, second, scores = inputs(3)
        bad_baselines = [baseline[:-1], baseline + [dict(baseline[0], sample_id="extra")],
                         [baseline[0], baseline[0], baseline[2]], list(reversed(baseline)),
                         [dict(baseline[0], sample_id="unknown"), *baseline[1:]]]
        for bad in bad_baselines:
            with self.subTest(ids=[r["sample_id"] for r in bad]), self.assertRaises(ValueError):
                app.build_submission(rows, bad, vision, second, scores,
                                     lambda *_: self.fail("Bad IDs reached question generation"))
        for bad in ([rows[0], rows[0], rows[2]],
                    [{k: v for k, v in rows[0].items() if k != "sample_id"}, *rows[1:]]):
            with self.subTest(rows=bad), self.assertRaises(ValueError):
                app.build_submission(bad, baseline, vision, second, scores, callback)

    def test_baseline_extra_fields_and_invalid_question_are_rejected(self):
        rows, baseline, vision, second, scores = inputs(2)
        for changed in (dict(baseline[0], debug="unexpected"),
                        dict(baseline[0], pred_emotion="unknown"),
                        dict(baseline[0], is_uncertain="maybe"),
                        dict(baseline[0], is_uncertain="yes", clarifying_question="none"),
                        dict(baseline[0], clarifying_question="A question on a no?")):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                app.build_submission(rows, [changed, baseline[1]], vision, second, scores, callback)

    def test_invalid_new_question_cannot_escape_to_output(self):
        rows, baseline, vision, second, scores = inputs(1)
        for question in ("", "   ", "none", "NONE", None, 3):
            with self.subTest(question=question), self.assertRaises(ValueError):
                app.build_submission(rows, baseline, vision, second, scores,
                                     lambda *_, value=question: value)

    def test_immutable_write_is_idempotent_and_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.csv"
            before = b"sample_id,pred_emotion\ns-0,positive\n"
            app.write_immutable(path, before)
            written_ns = path.stat().st_mtime_ns
            app.write_immutable(path, before)
            self.assertEqual(path.stat().st_mtime_ns, written_ns)
            with self.assertRaises((ValueError, FileExistsError)):
                app.write_immutable(path, before.replace(b"positive", b"negative"))
            self.assertEqual(path.read_bytes(), before)

    def test_export_keeps_baseline_and_publishes_one_auditable_candidate(self):
        rows, baseline, vision, second, scores = inputs(734)
        # One existing yes and one newly added yes; all remaining rows stay no.
        baseline[0].update(is_uncertain="yes", clarifying_question=" Existing question? ")
        scores[:] = .5
        scores[1] = .75
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = root / "baseline.csv"
            with frozen.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=app.FIELDS)
                writer.writeheader()
                writer.writerows(baseline)
            original = frozen.read_bytes()
            digest = hashlib.sha256(original).hexdigest()
            with patch.object(app, "BASELINE_SHA", digest):
                app.export_submission(root / "candidate", rows, frozen, vision, second,
                                      scores, callback, {"source": "unit-test"})
            self.assertEqual(frozen.read_bytes(), original)
            folder = root / "candidate"
            with (folder / "submission.csv").open(encoding="utf-8-sig", newline="") as stream:
                exported = list(csv.DictReader(stream))
            self.assertEqual(len(exported), 734)
            self.assertEqual([r["sample_id"] for r in exported], [r["sample_id"] for r in rows])
            self.assertEqual(exported[0], baseline[0])
            self.assertEqual(exported[1]["is_uncertain"], "yes")
            self.assertEqual(exported[2:], baseline[2:])
            for name in ("submission-changes.json", "submission-evidence.json", "submission-manifest.json"):
                self.assertTrue(json.loads((folder / name).read_text(encoding="utf-8")))

    def test_export_rejects_baseline_destination_and_changed_existing_candidate(self):
        rows, baseline, vision, second, scores = inputs(734)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = root / "submission.csv"
            with frozen.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=app.FIELDS)
                writer.writeheader()
                writer.writerows(baseline)
            original = frozen.read_bytes()
            with patch.object(app, "BASELINE_SHA", hashlib.sha256(original).hexdigest()):
                with self.assertRaises((ValueError, FileExistsError)):
                    app.export_submission(root, rows, frozen, vision, second,
                                          scores, callback, {"source": "unit-test"})
                self.assertEqual(frozen.read_bytes(), original)
                folder = root / "candidate"
                app.export_submission(folder, rows, frozen, vision, second,
                                      scores, callback, {"source": "unit-test"})
                published = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
                scores[0] = .5
                with self.assertRaises((ValueError, FileExistsError)):
                    app.export_submission(folder, rows, frozen, vision, second,
                                          scores, callback, {"source": "unit-test"})
                self.assertEqual({p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}, published)


if __name__ == "__main__":
    unittest.main()
