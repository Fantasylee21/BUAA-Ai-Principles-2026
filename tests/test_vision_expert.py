"""Visual-expert cache/input contracts; these tests do not simulate model quality.

The real NF4 Qwen3 checkpoint must separately pass GPU preflight. CPU synthetic
arrays below exercise serialization and generic residual-head dimensions only.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from rl.experiment import fingerprint
from scripts import vision_expert as app


def context(videos=3):
    rows, media = [], {}
    for i in range(videos):
        video = f"videos/parent-{i // 3}/clip-{i}.mp4"
        media[video] = hashlib.sha256(video.encode()).hexdigest()
        visual_label = "positive" if i % 2 else "negative"
        other = visual_label if i % 3 else ("positive" if visual_label == "negative" else "negative")
        for protocol, mode in (("VT", "text"), ("VA", "audio")):
            rows.append({"sample_id": f"PRIVATE-ID-{i}-{protocol}", "video": video,
                "protocol": protocol, "text_content": "PRIVATE-TRANSCRIPT", "gt_label": visual_label,
                "sample_type": "Conflict" if i % 3 == 0 else "Aligned",
                "source_dataset": "ch_sims_v2", "natural": True, "split_group_id": video,
                "view_targets": {"vision": {"label": visual_label, "clear": True},
                    mode: {"label": other, "clear": True}}})
    return {"identity_hash": "original-producer", "train": rows, "identity": {"media": media}}


def cache_identity(width=11):
    raw = {"hidden_size": width, "model_artifact_sha256": "model-weights-hash",
           "extractor_source_hash": "extractor-source-hash", "settings": app.SETTINGS,
           "source_producer_identity": "original-producer"}
    return {"identity": raw, "identity_hash": fingerprint(raw)}


def audit():
    ix = app.sample_indices(100, 25.)
    result = {"decode_stream": {"video": 0}, "audio_decoded": False,
        "source_frames": 100, "source_fps": 25., "sampled_frames": len(ix),
        "sample_indices": ix.tolist(), "source_frame_timestamps": (ix / 25.).tolist(),
        "processor_timestamps": (ix / 25.).tolist(), "source_frame_height": 480,
        "source_frame_width": 640, "elapsed_seconds": .25,
        "peak_allocated_bytes": 1024, "peak_reserved_bytes": 2048,
        "supplied_text_in_prompt": False, "gold_or_id_in_prompt": False,
        "prompt_hash": fingerprint([app.SYSTEM_PROMPT, app.INSTRUCTION, app.PREFIX])}
    result.update(app.validate_processed_geometry([[len(ix) // 2, 14, 14]], len(ix), 512))
    return result


def populate(root, c, identity):
    tasks, entries = app.video_tasks(c), []
    app.json_once(Path(root) / "model-frozen.json", {"identity_hash": identity["identity_hash"],
                                                   "frozen_sha256": "packed-frozen-weights"})
    for i, task in enumerate(tasks):
        h = np.arange(identity["identity"]["hidden_size"], dtype=np.float32) + i / 10
        native = np.array([1., 0., -1.], dtype=np.float32)
        marker, _, _ = app.save_feature(root, identity, task, "packed-frozen-weights", h, native, audit())
        entries.append(marker)
    app.json_once(Path(root) / "features/train/complete.json", {
        "identity_hash": identity["identity_hash"], "keys": [t["key"] for t in tasks], "entries": entries})
    return tasks


class RawOmniBank:
    split = "train"

    def get(self, rows, mode):
        hidden = np.array([[float(i), 1., 2., 3., 4.] for i in range(len(rows))])
        native = np.zeros((len(rows), 3))
        for i, row in enumerate(rows):
            native[i, ("positive", "negative", "neutral").index(row["view_targets"][mode]["label"])] = 2.
        return hidden, native


class VisionExpertTests(unittest.TestCase):
    def test_video_dedup_and_keys_ignore_text_gold_and_protocol(self):
        c = context()
        tasks = app.video_tasks(c)
        self.assertEqual(len(tasks), 3)
        self.assertEqual(tasks, app.video_tasks(copy.deepcopy(c)))
        for row in c["train"]:
            row.update(text_content="another private utterance", gt_label="neutral",
                       sample_type="Aligned", source_dataset="secret-source")
        self.assertEqual(tasks, app.video_tasks(c))
        self.assertTrue(all(set(t) == {"key", "video", "media_sha256", "split", "mode"} for t in tasks))
        c["train"].append(dict(c["train"][0]))
        with self.assertRaises(ValueError):
            app.video_tasks(c)

    def test_visual_messages_have_no_supplied_text_audio_gold_or_sample_ids(self):
        row = context()["train"][0]
        messages = app.vision_messages(row["video"])
        text = " ".join(part["text"] for message in messages for part in message["content"] if part["type"] == "text")
        self.assertNotIn(row["sample_id"], text)
        self.assertNotIn(row["text_content"], text)
        self.assertNotIn("Conflict", text)
        parts = [part["type"] for message in messages for part in message["content"]]
        self.assertEqual(parts.count("video"), 1)
        self.assertNotIn("audio", parts)
        self.assertEqual(app.PREFIX, '{"pred_emotion":"')

    def test_sampling_is_deterministic_and_keeps_fixed_limits(self):
        np.testing.assert_array_equal(app.sample_indices(100, 25.), [0, 14, 28, 42, 57, 71, 85, 99])
        capped = app.sample_indices(10000, 25.)
        self.assertEqual(len(capped), 64)
        self.assertEqual((capped[0], capped[-1]), (0, 9999))
        self.assertEqual(len(app.sample_indices(10, 25.)), 4)
        for total, fps in ((3, 25.), (100, 0.), (100, float("nan")), (100, None)):
            with self.assertRaises(ValueError):
                app.sample_indices(total, fps)

    def test_qwen3_total_pixel_budget_uses_padded_frames_not_temporal_grid(self):
        for n in (4, 5, 32, 63, 64):
            kwargs = app.processor_video_kwargs(n)
            padded = 2 * ((n + 1) // 2)
            self.assertEqual(kwargs["size"]["longest_edge"], padded * 50176)
            self.assertEqual(kwargs["size"]["longest_edge"] / padded, 50176)
            self.assertFalse(kwargs["do_sample_frames"])
        geometry = app.validate_processed_geometry([[32, 14, 14]], 64, 2000)
        self.assertEqual(geometry["processed_pixels_per_frame"], 50176)
        for grid, frames, tokens in (([[32, 16, 14]], 64, 2000), ([[33, 14, 14]], 64, 2000),
                                    ([[32, 14, 14]], 64, 6145), ([[16, 14, 14]], 64, 2000)):
            with self.assertRaises(ValueError):
                app.validate_processed_geometry(grid, frames, tokens)

    def test_decoder_requests_video_stream_zero_without_any_audio_decode(self):
        calls = []

        class Frame:
            def __init__(self, i):
                self.time = i / 25.

            def to_ndarray(self, format):
                self_format = format
                if self_format != "rgb24":
                    raise AssertionError("Not RGB")
                return np.zeros((32, 64, 3), dtype=np.uint8)

        class Container:
            streams = SimpleNamespace(video=[SimpleNamespace(average_rate=25., frames=100)])

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def decode(self, **kwargs):
                calls.append(kwargs)
                if kwargs != {"video": 0}:
                    raise AssertionError("Audio or a wrong stream was decoded")
                return iter(Frame(i) for i in range(100))

        fake_av = SimpleNamespace(open=lambda *_a, **_k: Container())
        with patch.dict("sys.modules", {"av": fake_av}):
            frames, metadata, diagnostics = app.decode_silent_video("not-a-real-model-test.mp4")
        self.assertEqual(frames.shape[0], 8)
        self.assertEqual(calls, [{"video": 0}])
        self.assertFalse(diagnostics["audio_decoded"])
        self.assertEqual(metadata["frames_indices"], app.sample_indices(100, 25.).tolist())

    def test_cache_round_trip_preserves_arrays_and_embedded_provenance(self):
        c, identity = context(), cache_identity()
        task = app.video_tasks(c)[0]
        h, native = np.arange(11, dtype=np.float32), np.array([.3, -.4, .2], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            marker, values, meta = app.save_feature(tmp, identity, task, "packed-frozen-weights", h, native, audit())
            np.testing.assert_array_equal(values[0], h)
            np.testing.assert_array_equal(values[1], native)
            self.assertEqual(meta["media_sha256"], task["media_sha256"])
            self.assertEqual(meta["frozen_sha256"], "packed-frozen-weights")
            path, _ = app.feature_paths(tmp, task)
            before = path.read_bytes()
            app.save_feature(tmp, identity, task, "packed-frozen-weights", h, native, audit())
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(marker["archive_sha256"], hashlib.sha256(before).hexdigest())

    def test_cache_rejects_changed_model_source_settings_frozen_weights_and_media(self):
        c, identity = context(), cache_identity()
        task = app.video_tasks(c)[0]
        with tempfile.TemporaryDirectory() as tmp:
            app.save_feature(tmp, identity, task, "packed-frozen-weights", np.zeros(11), np.zeros(3), audit())
            for key, replacement in (("model_artifact_sha256", "wrong-model"),
                                     ("extractor_source_hash", "wrong-code"),
                                     ("settings", {"video_fps": 1.})):
                wrong = copy.deepcopy(identity)
                wrong["identity"][key] = replacement
                wrong["identity_hash"] = fingerprint(wrong["identity"])
                with self.assertRaises(ValueError):
                    app.load_feature(tmp, wrong, task, "packed-frozen-weights", required=True)
            with self.assertRaises(ValueError):
                app.load_feature(tmp, identity, task, "changed-quantized-weights", required=True)
            changed = dict(task, media_sha256="f" * 64)
            with self.assertRaises(ValueError):
                app.load_feature(tmp, identity, changed, "packed-frozen-weights", required=True)
            path, _ = app.feature_paths(tmp, task)
            content = path.read_bytes()
            path.write_bytes(content[:-10] + b"corruption")
            with self.assertRaises(ValueError):
                app.load_feature(tmp, identity, task, "packed-frozen-weights", required=True)

    def test_invalid_input_audit_fails_before_creating_any_feature(self):
        c, identity = context(), cache_identity()
        task = app.video_tasks(c)[0]
        for key, value in (("audio_decoded", True), ("supplied_text_in_prompt", True),
                           ("gold_or_id_in_prompt", True), ("processor_timestamps", [0.]),
                           ("processed_frame_height", 999), ("input_tokens_including_prefix", 6145)):
            invalid = audit()
            invalid[key] = value
            with tempfile.TemporaryDirectory() as tmp, self.subTest(key=key):
                with self.assertRaises(ValueError):
                    app.save_feature(tmp, identity, task, "packed-frozen-weights", np.zeros(11), np.zeros(3), invalid)
                self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_incomplete_cache_and_different_existing_artifacts_fail_closed(self):
        task, identity = app.video_tasks(context())[0], cache_identity()
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = app.feature_paths(tmp, task)
            path.parent.mkdir(parents=True)
            path.write_bytes(b"interrupted")
            with self.assertRaises(ValueError):
                app.load_feature(tmp, identity, task, "packed-frozen-weights")
            with self.assertRaises(ValueError):
                app.bytes_once(path, b"different")
            self.assertEqual(path.read_bytes(), b"interrupted")

    def test_hybrid_bank_replaces_only_vision_and_accepts_different_hidden_widths(self):
        c, identity = context(6), cache_identity(width=13)
        with tempfile.TemporaryDirectory() as tmp:
            tasks = populate(tmp, c, identity)
            expert, omni = app.VisionBank(tmp, identity, tasks), RawOmniBank()
            hybrid = app.HybridBank(omni, expert)
            self.assertEqual(hybrid.get(c["train"], "vision")[0].shape, (12, 13))
            vt = [r for r in c["train"] if r["protocol"] == "VT"]
            va = [r for r in c["train"] if r["protocol"] == "VA"]
            np.testing.assert_array_equal(hybrid.get(vt, "text")[0], omni.get(vt, "text")[0])
            np.testing.assert_array_equal(hybrid.get(va, "audio")[1], omni.get(va, "audio")[1])
            with self.assertRaises(ValueError):
                hybrid.get(c["train"], "full")
            with self.assertRaises(ValueError):
                expert.get([{"video": "validation-only.mp4"}], "vision")
            # Actual generic old fit functions, using synthetic arrays only.
            from rl.evidence_bottleneck import DEFAULTS
            from rl.evidence_training import fit_evidence
            fitted_context = {**c, "identity_hash": "independent-hybrid-fit", "settings": dict(DEFAULTS)}
            heads = fit_evidence(Path(tmp) / "fits", fitted_context, hybrid, c["train"], "dimensional-contract", lambda *_a, **_k: None)
            self.assertEqual(np.array(heads["vision"]["parameters"]).shape, (3, 14))
            self.assertEqual(np.array(heads["text"]["parameters"]).shape, (3, 6))
            self.assertEqual(np.array(heads["audio"]["parameters"]).shape, (3, 6))

    def test_publication_requires_full_video_coverage_and_original_order(self):
        c, identity = context(), cache_identity()
        with tempfile.TemporaryDirectory() as tmp:
            tasks = populate(tmp, c, identity)
            app.VisionBank(tmp, identity, tasks)
            with self.assertRaises(ValueError):
                app.VisionBank(tmp, identity, list(reversed(tasks)))

    def test_preflight_includes_maximum_frame_clip_without_reading_labels(self):
        c = context(40)
        identity, tasks = cache_identity(), app.video_tasks(c)
        with tempfile.TemporaryDirectory() as tmp:
            c["data_root"] = tmp
            for task in tasks:
                path = Path(tmp) / task["video"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"metadata-probe-contract-only")

            def probe(path):
                index = int(path.stem.split("-")[-1])
                return (1600 if index == 39 else 100), 25.

            with patch.object(app, "probe_video", side_effect=probe):
                chosen, plan = app.preflight_plan(tmp, c, tasks, identity)
            self.assertEqual(len(chosen), 32)
            self.assertIn(tasks[39], chosen)
            self.assertTrue(plan["covers_64_frame_cap"])
            self.assertFalse(plan["labels_used_for_selection"])
            self.assertFalse(plan["audio_decoded"])

    def test_cpu_fit_identities_separate_model_producers_and_hide_val_test(self):
        from scripts import vision_expert_cpu as cpu
        from rl.evidence_bottleneck import DEFAULTS
        c = {**context(), "settings": dict(DEFAULTS), "val": ["must-not-fit"], "test": ["must-not-fit"]}
        control = cpu.fit_context(c, "evaluation-identity", "control")
        expert = cpu.fit_context(c, "evaluation-identity", "expert")
        changed = cpu.fit_context(c, "different-visual-cache-identity", "expert")
        self.assertNotEqual(control["identity_hash"], expert["identity_hash"])
        self.assertNotEqual(expert["identity_hash"], changed["identity_hash"])
        self.assertNotIn("val", expert)
        self.assertNotIn("test", expert)
        self.assertNotIn("baselines", expert)

    def test_cpu_gate_requires_gain_natural_recall_and_pressure_stability(self):
        from scripts import vision_expert_cpu as cpu

        def report(component, recall):
            return {"task_component": component, "natural_conflict_recall": recall}

        primary = {"control": report(.3, .6), "expert": report(.31, .6)}
        pressure = {"control": report(.25, .5), "expert": report(.25, .5)}
        self.assertTrue(cpu.comparison_gate(primary, pressure)["passed"])
        for which in ("primary_gain", "natural", "pressure", "missing_pressure"):
            p, stress = copy.deepcopy(primary), copy.deepcopy(pressure)
            if which == "primary_gain":
                p["expert"]["task_component"] = .309
            elif which == "natural":
                p["expert"]["natural_conflict_recall"] = .59
            elif which == "pressure":
                stress["expert"]["task_component"] = .249
            else:
                stress = None
            with self.subTest(which=which):
                self.assertFalse(cpu.comparison_gate(p, stress)["passed"])

    def test_cpu_nested_real_head_fits_share_second_branch_and_hold_out_groups(self):
        from scripts import vision_expert_cpu as cpu
        from scripts.train_evidence_calibrator import group_design, checked_artifact
        from rl.evidence_bottleneck import DEFAULTS
        c = {**context(45), "settings": dict(DEFAULTS)}
        identity = cache_identity(width=9)
        with tempfile.TemporaryDirectory() as tmp:
            tasks = populate(tmp, c, identity)
            omni = RawOmniBank()
            hybrid = app.HybridBank(omni, app.VisionBank(tmp, identity, tasks))
            contexts = {k: cpu.fit_context(c, "synthetic-dimensional-contract", k) for k in ("control", "expert")}
            design = group_design(c["train"], c["identity"]["media"])
            destination = Path(tmp) / "evaluation"
            with patch.object(cpu, "emit"):
                result = cpu.development(destination, c, omni, hybrid, contexts, design, "primary")
                repeated = cpu.development(destination, c, omni, hybrid, contexts, design, "primary")
            self.assertEqual(result, repeated)
            self.assertEqual(result["ids"], [r["sample_id"] for r in c["train"]])
            self.assertTrue(result["supervised_group_independence"])
            self.assertEqual(result["control"]["rows"], len(c["train"]))
            self.assertEqual(result["expert"]["rows"], len(c["train"]))
            for part in result["partitions"]:
                self.assertTrue(part["shared_text_audio_heads"])
                self.assertFalse(set(part["training_ids"]) & set(part["held_ids"]))
                self.assertFalse(set(part["calibration_ids"]) & set(part["held_ids"]))
                job = f"primary-fold{part['fold']}"
                for variant in ("control", "expert"):
                    fitted = checked_artifact(destination / "calibration" / (job + "-" + variant + ".json"))
                    self.assertEqual(fitted["examples"], part["calibration_ids"])
            self.assertEqual(len(result["probabilities"]["expert"]), len(c["train"]))
            # These synthetic arrays verify fitting/data contracts, not that the
            # actual new Qwen3 model passes the gain gate or improves Kaggle.


if __name__ == "__main__":
    unittest.main()
