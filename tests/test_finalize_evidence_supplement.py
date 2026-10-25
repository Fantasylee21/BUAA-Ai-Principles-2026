"""Question-only review must preserve predictions and the original evidence."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rl.experiment import file_hash, fingerprint, load_records
from scripts import evidence_supplement as core
from scripts import finalize_evidence_supplement as app


class FinalizeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.raw, self.output = self.root/"raw", self.root/"reviewed"
        self.raw.mkdir()
        self.test = self.root/"test.jsonl"
        source_rows = [{"sample_id": f"s-{i}", "protocol": "VT" if i % 2 == 0 else "VA",
                        "video": f"videos/{i}.mp4", "text": "A supplied statement"} for i in range(734)]
        self.test.write_text("\n".join(json.dumps(r) for r in source_rows), encoding="utf-8")
        self.rows, _ = load_records(self.test, require_labels=False)
        self.baseline = [{"sample_id": r["sample_id"], "pred_emotion": "positive",
                          "is_uncertain": "no", "clarifying_question": "none"} for r in self.rows]
        self.baseline[0].update(is_uncertain="yes", clarifying_question=" Preserve this existing question? ")
        self.base = self.root/"baseline.csv"
        with self.base.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=core.FIELDS)
            writer.writeheader()
            writer.writerows(self.baseline)
        self.base_hash = file_hash(self.base)
        patched = patch.object(core, "BASELINE_SHA", self.base_hash)
        patched.start()
        self.addCleanup(patched.stop)
        (self.raw/"baseline-frozen.csv").write_bytes(self.base.read_bytes())
        identity = {"algorithm": core.VERSION, "command": "export", "threshold": .5,
                    "train_models": False, "baseline_sha256": self.base_hash,
                    "producer_identity": "producer", "artifacts": {"candidate": "frozen"}}
        self.run_hash = fingerprint(identity)
        (self.raw/"run-identity.json").write_bytes(app.json_bytes({"identity": identity, "identity_hash": self.run_hash}))
        vision, second = np.tile([.8, .1, .1], (734, 1)), np.tile([.1, .8, .1], (734, 1))
        scores = np.full(734, .5)
        scores[1:3] = .75
        core.export_submission(self.raw, self.rows, self.base, vision, second, scores,
            lambda *_: "What relationship do those people have?",
            {"run_identity": self.run_hash, "producer_identity": "producer",
             "artifacts": identity["artifacts"], "expected_baseline_sha256": self.base_hash})

    def finalize(self):
        return app.finalize(self.raw, self.base, self.test, self.output)

    def test_replaces_only_new_yes_questions_with_user_facing_protocol_templates(self):
        original = {p.name: p.read_bytes() for p in self.raw.iterdir()}
        raw_rows = app.csv_rows(original["submission.csv"])
        manifest = self.finalize()
        reviewed = app.csv_rows((self.output/"submission.csv").read_bytes())
        self.assertEqual(len(reviewed), 734)
        self.assertEqual(reviewed[0], self.baseline[0])
        self.assertEqual(reviewed[1]["clarifying_question"], app.QUESTIONS["VA"])
        self.assertEqual(reviewed[2]["clarifying_question"], app.QUESTIONS["VT"])
        self.assertEqual(reviewed[3:], self.baseline[3:])
        for raw, final in zip(raw_rows, reviewed):
            for field in ("sample_id", "pred_emotion", "is_uncertain"):
                self.assertEqual(raw[field], final[field])
        self.assertTrue(all("你" in reviewed[i]["clarifying_question"] for i in (1, 2)))
        self.assertEqual({p.name: p.read_bytes() for p in self.raw.iterdir()}, original)
        self.assertEqual(manifest["provenance"]["question_policy"], app.POLICY)
        self.assertEqual(manifest["provenance"]["raw_manifest_sha256"], file_hash(self.raw/"submission-manifest.json"))
        self.assertFalse(manifest["submitted_to_kaggle"])

    def test_tampered_raw_csv_hash_fails_before_publishing(self):
        path = self.raw/"submission.csv"
        path.write_bytes(path.read_bytes()+b"\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.finalize()
        self.assertFalse(self.output.exists())

    def test_corrupt_run_identity_fails_before_publishing(self):
        path = self.raw/"run-identity.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["identity"]["producer_identity"] = "changed"
        path.write_bytes(app.json_bytes(value))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.finalize()
        self.assertFalse(self.output.exists())

    def test_source_change_during_question_replacement_cancels_publication(self):
        generate = app.neutral_question
        def change_source(row, decision):
            path = self.raw/"submission.csv"
            path.write_bytes(path.read_bytes()+b"\n")
            return generate(row, decision)
        with patch.object(app, "neutral_question", side_effect=change_source):
            with self.assertRaisesRegex(ValueError, "changed"):
                self.finalize()
        self.assertFalse(self.output.exists())

    def test_output_is_immutable_and_identical_retry_is_idempotent(self):
        self.finalize()
        original = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.output.iterdir()}
        self.finalize()
        self.assertEqual({p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.output.iterdir()}, original)
        path = self.output/"submission.csv"
        path.write_bytes(b"Existing user data")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.finalize()
        self.assertEqual(path.read_bytes(), b"Existing user data")

    def test_output_cannot_alias_input_or_its_ancestor(self):
        for output in (self.raw, self.root, self.raw/"child"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "separate"):
                app.finalize(self.raw, self.base, self.test, output)


if __name__ == "__main__":
    unittest.main()
