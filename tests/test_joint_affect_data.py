"""Clarity, grouped training-only remixing, score conventions and exact gates."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import joint_affect_data as app


def temporary_directory():
    root = Path(__file__).resolve().parents[1] / "tmp"
    root.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root)


def corpus(groups=30):
    rows, media = [], {}
    for g in range(groups):
        video = f"media/parent-{g//2}/video-{g}.mp4"
        media[video] = hashlib.sha256(f"media-{g if g != 1 else 0}".encode()).hexdigest()
        for protocol in ("VT", "VA"):
            first = "negative" if g % 2 == 0 else "positive"
            conflict = g % 3 == 0 or (protocol == "VA" and g % 5 == 0)
            second = ("positive" if first == "negative" else "negative") if conflict else first
            rows.append({"sample_id": f"s-{g}-{protocol}", "split": "train", "protocol": protocol,
                "video": video, "text_content": f"text {g}", "split_group_id": str(g),
                "sample_type": "Conflict" if conflict else "Aligned", "use_in_main": False,
                "source_dataset": "ch_sims_v2" if g < groups*2//3 else "accept_a_va",
                "views": {"M1": {"modality": "vision", "label": first, "is_clear": g % 7 != 0},
                    "M2": {"modality": "text" if protocol == "VT" else "audio", "label": second, "is_clear": g % 11 != 0},
                    "M12": {"modality": "vision+text" if protocol == "VT" else "vision+audio", "label": first}}})
    return rows, media


def predict(rows, yes=None):
    result = []
    for i, row in enumerate(rows):
        target = app.targets(row)
        probs = [0., 0., 0.]
        probs[target["emotion"]] = 1.
        v, m = [0.]*4, [0.]*4
        v[target["vision"]], m[target["second"]] = 1., 1.
        uncertain = row["sample_type"] == "Conflict" if yes is None else yes[i]
        result.append({"sample_id": row["sample_id"], "emotion_probs": probs,
                       "uncertain_prob": .8 if uncertain else .2, "vision_probs": v, "second_probs": m})
    return result


class JointDataTests(unittest.TestCase):
    def test_label_order_unclear_is_separate_and_real_M12_survives(self):
        rows, _ = corpus(1)
        row = rows[0]
        row["views"]["M1"].update(label="neutral", is_clear=False)
        self.assertEqual(app.EMOTIONS, ("negative", "neutral", "positive"))
        self.assertEqual(app.targets(row), {"vision": 3, "second": 3, "emotion": 0, "relation": 1, "emotion_mask": 1})
        del row["views"]["M1"]["is_clear"]
        with self.assertRaisesRegex(ValueError, "boolean is_clear"):
            app.targets(row)
        inferred = app.canonical_inference({"sample_id": "ch_sims_v2:VA:v:1", "video": "v.mp4"})
        self.assertEqual(inferred["protocol"], "VA")
        self.assertNotIn("views", inferred)

    def test_content_union_hash_pairs_and_parent_pressure_are_isolated(self):
        rows, media = corpus()
        groups = app.content_groups(rows, media)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[0], groups[2])  # Different videos, identical bytes.
        primary = app.fold_manifest(rows, media)
        self.assertEqual(primary, app.fold_manifest(rows, media))
        held = []
        for partition in primary["folds"]:
            held += partition["held"]
            self.assertFalse({groups[i] for i in partition["train"]} & {groups[i] for i in partition["held"]})
        self.assertEqual(sorted(held), list(range(len(rows))))
        parent = app.fold_manifest(rows, media, pressure=True)
        self.assertLessEqual(parent["independent_groups"], primary["independent_groups"])
        with self.assertRaisesRegex(ValueError, "split is fixed"):
            app.fold_manifest(rows, media, seed=43)

    def test_remix_uses_only_train_clear_different_media_and_never_fabricates_M12(self):
        rows, media = corpus()
        indices = app.fold_manifest(rows, media)["folds"][0]["train"]
        original = deepcopy(rows)
        remixed = app.remix_training(rows, indices, media, 42, epoch=0)
        self.assertEqual(len(remixed), len(indices)//2)
        self.assertEqual(remixed, app.remix_training(rows, indices, media, 42, epoch=0))
        same = sum(r["_remix"]["same_emotion"] for r in remixed)
        self.assertLessEqual(abs(same - (len(remixed)-same)), 1)
        for row in remixed:
            a, b = row["_vision_source_idx"], row["_second_source_idx"]
            self.assertIn(a, indices); self.assertIn(b, indices)
            self.assertEqual(rows[a]["protocol"], rows[b]["protocol"])
            self.assertNotEqual(media[rows[a]["video"]], media[rows[b]["video"]])
            self.assertTrue(row["views"]["M1"]["is_clear"])
            self.assertTrue(row["views"]["M2"]["is_clear"])
            self.assertNotIn("M12", row["views"])
            self.assertEqual(app.targets(row)["emotion"], -100)
            self.assertEqual(app.targets(row)["emotion_mask"], 0)
            self.assertIs(app.second_source(row, rows), rows[b])
        examples = app.training_examples(rows, indices, media, 42)
        self.assertEqual(len(examples), len(indices)+len(remixed))
        self.assertEqual(rows, original)
        self.assertTrue(all(not r["use_in_main"] for r in examples[:len(indices)]))

    def test_load_manifest_hashes_media_and_does_not_open_official_splits(self):
        rows, _ = corpus(12)
        with temporary_directory() as tmp:
            root = Path(tmp)
            for i in range(12):
                path = root/f"media/parent-{i//2}/video-{i}.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"media-{i if i != 1 else 0}".encode())
            path = root/"train.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            (root/"val.jsonl").write_text("DO NOT READ")
            (root/"test_release.jsonl").write_text("DO NOT READ")
            manifest = app.prepare_manifest(path, root, expected_rows=None)
            self.assertEqual(manifest["identity"]["budget_hours"], 120)
            self.assertEqual(len(manifest["train"]), len(rows))
            self.assertTrue(all(not r["use_in_main"] for r in manifest["train"]))
            self.assertEqual(manifest, app.prepare_manifest(path, root, expected_rows=None))
            self.assertIs(app.validate_manifest(manifest), manifest)
            counts = app.class_counts(manifest["train"])
            self.assertEqual(counts["real_total"], len(rows))
            self.assertEqual(counts["unique_videos"], 12)
            self.assertEqual(sum(counts["video_frequency"].values()), len(rows))
            changed = deepcopy(manifest)
            changed["train"][0]["views"]["M12"]["label"] = "positive"
            with self.assertRaisesRegex(ValueError, "detached"):
                app.validate_manifest(changed)
            changed = deepcopy(manifest)
            changed["folds"]["content"]["folds"][0]["held"].append(0)
            with self.assertRaisesRegex(ValueError, "fold design changed"):
                app.validate_manifest(changed)
            with self.assertRaisesRegex(ValueError, "Only original train"):
                app.load_training(root/"val.jsonl", root, expected_rows=None)

    def test_missing_neutral_uses_PN_selection_and_diagnostics_rejection(self):
        rows, _ = corpus(6)
        predictions = predict(rows)
        report = app.metrics(rows, predictions)
        self.assertAlmostEqual(report["selection_score"], 1.)
        self.assertAlmostEqual(report["competition_score_fixed3"], 5/6)
        aligned = next(i for i, r in enumerate(rows) if r["sample_type"] == "Aligned")
        predictions[aligned]["uncertain_prob"] = .5
        report = app.metrics(rows, predictions)
        self.assertEqual(report["false_positives"], 1)
        self.assertEqual(report["evidence_correct_but_rejected"], 1)
        self.assertEqual(report["aligned_evidence_correct_but_uncertain"], 1)
        self.assertGreater(report["opposite_pair_count"], 0)
        self.assertLessEqual(report["opposite_pair_both_correct"], report["opposite_pair_count"])

    def gate_fixture(self):
        folds = []
        all_rows, all_control, all_candidate = [], [], []
        for fold in range(3):
            rows, _ = corpus(3)
            # Three natural conflicts and three ordinary samples per held fold.
            rows = rows[:6]
            for i, row in enumerate(rows):
                row["sample_id"] = f"f{fold}-{i}"
                row["video"] = f"fold-{fold}/v-{i}.mp4"
                row["source_dataset"] = "ch_sims_v2"
                row["sample_type"] = "Conflict" if i < 3 else "Aligned"
            control = predict(rows, [False]*6)
            candidate = predict(rows, [True]*3+[False]*3)
            folds.append({"fold": fold, "control": app.metrics(rows, control), "candidate": app.metrics(rows, candidate)})
            all_rows += rows; all_control += control; all_candidate += candidate
        pressure = {"control": app.metrics(all_rows, all_control), "candidate": app.metrics(all_rows, all_candidate)}
        return folds, pressure

    def test_exact_primary_pressure_and_replication_gates(self):
        folds, pressure = self.gate_fixture()
        primary = app.promotion_gate(folds)
        self.assertTrue(primary["primary_passed"])
        self.assertIsNone(primary["pressure_passed"])
        self.assertFalse(primary["passed"])
        result = app.promotion_gate(folds, pressure)
        self.assertTrue(result["passed"])
        self.assertFalse(result["ready_for_full_training"])
        self.assertTrue(app.promotion_gate(folds, pressure, {"primary_passed": True})["ready_for_full_training"])
        bad = deepcopy(folds)
        for fold in bad:
            # One extra ordinary false positive per fold exceeds 1% of aligned.
            fold["candidate"]["_records"][-1]["is_uncertain"] = "yes"
        rejected = app.promotion_gate(bad, pressure)
        self.assertFalse(rejected["checks"]["false_positive_increase_at_most_1_percent_aligned"])
        self.assertFalse(rejected["passed"])
        # Better emotion alone must not pass the uncertainty pressure gate.
        emotion_only = deepcopy(pressure)
        for a, b in zip(emotion_only["control"]["_records"], emotion_only["candidate"]["_records"]):
            a["pred_emotion"] = "negative" if a["gt_label"] == "positive" else "positive"
            b["is_uncertain"] = a["is_uncertain"]
        self.assertFalse(app.promotion_gate(folds, emotion_only)["pressure_passed"])

    def test_export_strict_ID_coverage_threshold_and_immutable_output(self):
        rows, _ = corpus(2)
        inputs = [{k: r[k] for k in ("sample_id", "video", "protocol", "text_content")} for r in rows]
        predictions = predict(rows)
        predictions[0]["uncertain_prob"] = .5
        with temporary_directory() as tmp:
            path = Path(tmp)/"submission.csv"
            result = app.export_csv(path, inputs, predictions, expected_rows=4)
            self.assertEqual(result["rows"], 4)
            with path.open(encoding="utf-8", newline="") as stream:
                parsed = list(csv.DictReader(stream))
            self.assertEqual(parsed[0]["is_uncertain"], "yes")
            self.assertIn("你", parsed[0]["clarifying_question"])
            original = path.read_bytes()
            app.export_csv(path, inputs, predictions, expected_rows=4)
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "exact sample"):
                app.export_csv(path, inputs, list(reversed(predictions)), expected_rows=4)
            predictions[0]["uncertain_prob"] = 0.
            with self.assertRaisesRegex(ValueError, "different"):
                app.export_csv(path, inputs, predictions, expected_rows=4)


if __name__ == "__main__":
    unittest.main()
