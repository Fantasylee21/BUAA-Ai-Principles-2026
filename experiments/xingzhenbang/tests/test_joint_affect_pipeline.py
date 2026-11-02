"""CPU orchestration tests. These do not establish model quality or GPU fit."""
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
from pathlib import Path
import tarfile
import tempfile
import threading
import unittest
from unittest.mock import patch

from scripts import joint_affect_pipeline as p
from scripts.joint_affect_data import canonical_inference


def temporary_directory():
    root = p.SOURCE / "tmp" / "joint-affect-tests"
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root)


def prediction(row, uncertain=None):
    label = row.get("views", {}).get("M12", {}).get("label", "positive")
    return {"sample_id": row["sample_id"], "emotion_probs": [float(label == e) for e in p.LABELS],
            "uncertain_prob": float(row.get("sample_type") == "Conflict") if uncertain is None else uncertain}


def fixture_rows():
    result = []
    for i in range(18):
        protocol = "VT" if i % 2 == 0 else "VA"
        emotion = "positive" if i % 2 == 0 else "negative"
        result.append({"sample_id": "example:" + protocol + ":" + str(i), "split": "train",
            "video": "videos/parent" + str(i // 6) + "/" + str(i) + ".mp4",
            "protocol": protocol, "text_content": "original supplied text",
            "source_dataset": "ch_sims", "sample_type": "Conflict" if i % 6 < 2 else "Aligned",
            "use_in_main": False,
            "views": {"M1": {"label": emotion, "is_clear": True, "modality": "vision"},
                "M2": {"label": emotion, "is_clear": True, "modality": "text" if protocol == "VT" else "audio"},
                "M12": {"label": emotion, "is_clear": True}}})
    return result


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def pipeline_fixture(folder, runner=None):
    project = Path(folder).resolve()
    root = project / "runs" / "joint"
    root.mkdir(parents=True)
    rows = fixture_rows()
    design = {"folds": [{"fold": f, "held": list(range(6*f, 6*f+6)),
                         "train": [i for i in range(18) if i // 6 != f]} for f in range(3)]}
    manifest = {"train": rows, "folds": {"content": design, "parent": design}}
    p.immutable(root / "train-manifest.json", manifest)
    identity = {"algorithm": p.VERSION, "project_root": str(project), "source_sha256": {},
                "train_manifest_sha256": p.digest(root / "train-manifest.json")}
    p.immutable(root / "plan.json", {"identity": identity, "identity_hash": p.fingerprint(identity)})
    write_jsonl(project / "Dataset/val.jsonl", [{**r, "split": "val"} for r in rows])
    # Mirrors real release rows: protocol is inferred from the existing ID tag.
    write_jsonl(project / "Dataset/test_release.jsonl", [
        {"sample_id": r["sample_id"], "video": r["video"], "text": "input only"} for r in rows])
    return p.Pipeline(project, root, runner=runner)


class FakeTrainer:
    def __init__(self, bigger=False, fail=None):
        self.calls, self.bigger, self.fail = [], bigger, fail
        self.lock = threading.Lock()
        self.pipeline = None

    def __call__(self, command, directory, output_marker):
        def arg(name):
            return command[command.index(name) + 1]
        action, fold = command[3], arg("--fold")
        job = self.pipeline.spec(arg("--variant"), "full" if fold == "full" else int(fold),
            arg("--grouping"), int(arg("--seed")), arg("--vision-model"))
        with self.lock:
            self.calls.append((action, job, list(command)))
        if self.fail:
            self.fail(action, job, directory)
        if job["vision_model"] == "qwen3vl8b":
            assert arg("--gpus") == "1,2" and "--gpu" not in command
            expected = self.pipeline.second_initialization(job)["checkpoint"]
            assert arg("--init-second-from") == expected
        else:
            assert arg("--gpu") in ("1", "2") and "--gpus" not in command
        if action == "preflight":
            p.immutable(output_marker, {"job": job, "samples": 32, "peak_reserved_gib": 30.,
                "actual_media_audit": [{"video_frames": 32}], "gradient_and_update_audit": {"adapter": {"delta": 1.}},
                "losses_finite": True, "estimated_epochs_seconds": 10.})
        elif action == "train":
            checkpoint = directory / "checkpoint-final"
            checkpoint.mkdir(exist_ok=True)
            (checkpoint / "weights.bin").write_bytes(b"fixture-not-model-weights")
            p.immutable(output_marker, {"job": job, "checkpoint": str(checkpoint.resolve()),
                                       "checkpoint_sha256": "a" * 64, "epochs": 3})
        elif action == "predict":
            split = arg("--split")
            if split == "held":
                held = self.pipeline.designs[job["grouping"]]["folds"][job["fold"]]["held"]
                rows = [self.pipeline.rows[i] for i in held]
            else:
                # Both locked final checkpoints must exist before first access.
                lock = directory / "final_model_lock.json"
                assert lock.is_file() and len(p.read(lock)["models"]) == 2
                name = "val.jsonl" if split == "val" else "test_release.jsonl"
                rows = [canonical_inference(json.loads(line)) for line in
                        (self.pipeline.project / "Dataset" / name).read_text(encoding="utf-8").splitlines()]
            values = []
            for row in rows:
                value = prediction(row, 0. if job["variant"] == "SFT-ref" else None)
                if self.bigger and job["vision_model"] == "omni" and row["sample_id"] == "example:VA:1":
                    value["uncertain_prob"] = 0.
                values.append(value)
            write_jsonl(Path(arg("--output")), values)
            trained = p.read(directory / "train.complete.json")
            p.immutable(Path(arg("--output") + ".complete.json"), {"job": job, "split": split,
                "rows": len(rows), "sha256": p.digest(arg("--output")),
                "checkpoint_sha256": trained["checkpoint_sha256"]})
        else:
            raise AssertionError(action)


class BudgetTests(unittest.TestCase):
    def test_monotonic_and_resumed_wall_are_charged_without_reset(self):
        with temporary_directory() as folder:
            wall, mono = [100.], [5.]
            budget = p.Budget(folder, 100, wall=lambda: wall[0], mono=lambda: mono[0])
            mono[0] += 12
            self.assertEqual(budget.tick(), 188.)
            wall[0] += 40
            resumed = p.Budget(folder, 9999, wall=lambda: wall[0], mono=lambda: 0.)
            self.assertEqual(resumed.started, 100.)
            self.assertEqual(resumed.deadline, 200.)
            self.assertEqual(resumed.remaining, 60.)
            wall[0] = 201
            with self.assertRaises(TimeoutError):
                resumed.tick()

    def test_clock_rollback_refused(self):
        with temporary_directory() as folder:
            p.Budget(folder, 10, wall=lambda: 100., mono=lambda: 0.)
            with self.assertRaisesRegex(RuntimeError, "backwards"):
                p.Budget(folder, wall=lambda: 90., mono=lambda: 0.)


class ContractTests(unittest.TestCase):
    def test_exact_predictions_and_candidate_threshold(self):
        rows = fixture_rows()[:2]
        values = [prediction(r, .5 if i == 0 else .49) for i, r in enumerate(rows)]
        with temporary_directory() as folder:
            path = Path(folder) / "predictions.jsonl"
            write_jsonl(path, values)
            self.assertEqual(p.checked_predictions(path, rows), values)
            saved = list(csv.DictReader(io.StringIO(p.candidate_csv(rows, values).decode("utf-8"))))
            self.assertEqual(tuple(saved[0]), p.FIELDS)
            self.assertEqual([r["is_uncertain"] for r in saved], ["yes", "no"])
            self.assertEqual(saved[1]["clarifying_question"], "none")
            self.assertIn("画面", saved[0]["clarifying_question"])
            write_jsonl(path, list(reversed(values)))
            with self.assertRaises(ValueError):
                p.checked_predictions(path, rows)
            values[0]["uncertain_prob"] = float("nan")
            write_jsonl(path, values)
            with self.assertRaises(ValueError):
                p.checked_predictions(path, rows)

    def test_probability_average_and_identity(self):
        a = [{"sample_id": "one", "emotion_probs": [.8, .1, .1], "uncertain_prob": .2}]
        b = [{"sample_id": "one", "emotion_probs": [.1, .1, .8], "uncertain_prob": .8}]
        result = p.ensemble(a, b)[0]
        self.assertEqual(result["uncertain_prob"], .5)
        self.assertAlmostEqual(result["emotion_probs"][0], .45)
        b[0]["sample_id"] = "other"
        with self.assertRaises(ValueError):
            p.ensemble(a, b)

    def test_pooled_gate_natural_va_and_integer_fp_cap(self):
        rows = fixture_rows()
        design = {"folds": [{"fold": f, "held": list(range(6*f, 6*f+6))} for f in range(3)]}
        reference = [prediction(r, 0.) for r in rows]
        candidate = [prediction(r) for r in rows]
        report = p.compare(rows, reference, candidate, design)
        self.assertTrue(report["passed"])
        self.assertEqual(report["candidate"]["natural_tp"], 6)
        self.assertEqual(report["candidate"]["natural_va_tp"], 3)
        # 12 aligned rows permit zero additional false positives, not one.
        candidate[2]["uncertain_prob"] = .5
        changed = p.compare(rows, reference, candidate, design)
        self.assertFalse(changed["checks"]["fp_increase_within_one_percent"])
        self.assertFalse(changed["passed"])
        self.assertNotIn("_records", report["candidate"])

    def test_preflight_requires_actual_audit_finite_memory(self):
        with temporary_directory() as folder:
            pipeline = pipeline_fixture(folder)
            job = pipeline.spec("C0", 0, "content", 42, "omni")
            path = pipeline.root / "marker.json"
            path.write_text(json.dumps({"job": job, "samples": 32, "peak_reserved_gib": float("nan")}), encoding="utf-8")
            with self.assertRaises(ValueError):
                pipeline.validate_marker(path, job, "preflight")

    def test_dual_gpu_allocation_is_atomic(self):
        with temporary_directory() as folder:
            pipeline = pipeline_fixture(folder)
            dual = pipeline.spec("C0", 0, "content", 42, "qwen3vl8b")
            barrier = threading.Barrier(2)
            def work():
                barrier.wait()
                with pipeline.resources(dual) as cards:
                    self.assertEqual(cards, (1, 2))
                    self.assertEqual(pipeline.available_gpus, set())
            with ThreadPoolExecutor(max_workers=2) as executor:
                list(executor.map(lambda _: work(), range(2)))
            self.assertEqual(pipeline.available_gpus, {1, 2})

    def test_only_explicit_cuda_oom_triggers_fallback(self):
        self.assertTrue(p.is_cuda_oom("torch.cuda.OutOfMemoryError: CUDA out of memory"))
        self.assertFalse(p.is_cuda_oom("MemoryError: CPU allocation failed"))
        self.assertFalse(p.is_cuda_oom("CUBLAS_STATUS_EXECUTION_FAILED"))

    def test_oom_log_detection_excludes_historical_failure(self):
        with temporary_directory() as folder:
            path = Path(folder) / "reference-train.log"
            path.write_text("CUDA out of memory\n", encoding="utf-8")
            previous = path.stat().st_size
            with path.open("a", encoding="utf-8") as stream:
                stream.write("new attempt failed with invalid label\n")
            self.assertFalse(p.is_cuda_oom(p.new_log_tail(path, previous)))
            with path.open("a", encoding="utf-8") as stream:
                stream.write("torch.cuda.OutOfMemoryError: current attempt\n")
            self.assertTrue(p.is_cuda_oom(p.new_log_tail(path, previous)))


class PipelineTests(unittest.TestCase):
    def setup_pipeline(self, folder, trainer):
        pipeline = pipeline_fixture(folder, trainer)
        trainer.pipeline = pipeline
        return pipeline

    def tiny_official(self):
        return patch.object(p, "OFFICIAL_SPLITS", (("val", "val.jsonl", 18), ("test", "test_release.jsonl", 18)))

    def test_whole_pipeline_ties_locked_inference_and_archive(self):
        with temporary_directory() as folder, self.tiny_official():
            trainer = FakeTrainer()
            pipeline = self.setup_pipeline(folder, trainer)
            report = pipeline.run()
            self.assertEqual(report["status"], "complete")
            self.assertTrue(report["final_model_trained"])
            self.assertEqual(report["winner"]["variant"], "C0")
            self.assertEqual(report["winner"]["vision_model"], "omni")
            self.assertEqual(len([x for x in trainer.calls if x[0] == "predict" and x[1]["fold"] == "full"]), 4)
            candidate = pipeline.root / "cohort-64/candidate.csv"
            with candidate.open(encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 18)
            with tarfile.open(pipeline.root / "review-results.tar.gz") as archive:
                names = archive.getnames()
            self.assertIn("cohort-64/candidate.csv", names)
            self.assertFalse(any("weights.bin" in name or "train-manifest" in name or "Dataset/" in name for name in names))
            before = len(trainer.calls)
            self.assertEqual(pipeline.run(), report)
            self.assertEqual(len(trainer.calls), before)
            self.assertFalse(p.read(pipeline.root / "COMPLETE.json")["submitted_to_kaggle"])

    def test_8b_initializes_same_fold_group_seed_second_stream(self):
        with temporary_directory() as folder, self.tiny_official():
            trainer = FakeTrainer(bigger=True)
            pipeline = self.setup_pipeline(folder, trainer)
            report = pipeline.run()
            self.assertEqual(report["winner"]["vision_model"], "qwen3vl8b")
            self.assertTrue(report["final_model_trained"])
            for action, job, command in trainer.calls:
                if job["vision_model"] != "qwen3vl8b":
                    continue
                checkpoint = Path(command[command.index("--init-second-from") + 1])
                native = p.read(checkpoint.parent / "train.complete.json")["job"]
                self.assertEqual(native, {**job, "vision_model": "omni"})
            self.assertTrue(any(j["grouping"] == "parent" and j["vision_model"] == "omni" and j["variant"] == "C0"
                                for _, j, _ in trainer.calls))

    def test_gate_failure_never_reads_official_or_trains_full(self):
        with temporary_directory() as folder:
            trainer = FakeTrainer()
            pipeline = self.setup_pipeline(folder, trainer)
            (pipeline.project / "Dataset/val.jsonl").unlink()
            (pipeline.project / "Dataset/test_release.jsonl").unlink()
            # No advantage: every architecture emits the same reference output.
            original = trainer.__call__
            def same(command, directory, marker):
                original(command, directory, marker)
                if command[3] == "predict":
                    path = Path(command[command.index("--output") + 1])
                    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                    for value in values:
                        value["uncertain_prob"] = 0.
                    write_jsonl(path, values)
                    completion = p.read(str(path) + ".complete.json")
                    completion["sha256"] = p.digest(path)
                    p.atomic(str(path) + ".complete.json", completion)
            pipeline.runner = same
            report = pipeline.run()
            self.assertEqual(report["reason"], "primary_gate_not_met")
            self.assertFalse(report["final_model_trained"])
            self.assertFalse(any(j["fold"] == "full" for _, j, _ in trainer.calls))

    def test_oom_restarts_all_jobs_in_new_cohort_budget_unchanged(self):
        def fail(action, job, directory):
            if job["max_frames"] == 64 and action == "train" and job["variant"] == "C1":
                raise p.ChildFailure("CUDA out of memory", oom=True)
        with temporary_directory() as folder, self.tiny_official():
            trainer = FakeTrainer(fail=fail)
            pipeline = self.setup_pipeline(folder, trainer)
            report = pipeline.run()
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["max_frames"], 32)
            self.assertTrue((pipeline.root / "cohort-failures/frames64.json").is_file())
            self.assertTrue((pipeline.root / "cohort-64/jobs/content/omni/C0/seed42/fold0/train.complete.json").is_file())
            self.assertTrue((pipeline.root / "cohort-32/jobs/content/omni/C0/seed42/fold0/train.complete.json").is_file())
            clock = p.read(pipeline.root / "clock.json")
            self.assertEqual(clock["deadline_unix"] - clock["started_unix"], 120 * 3600)
            self.assertEqual(report["winner"]["max_frames"], 32)

    def test_non_oom_failure_does_not_restart_or_claim_training_complete(self):
        def fail(action, job, directory):
            if action == "preflight":
                raise p.ChildFailure("Invalid audio protocol")
        with temporary_directory() as folder:
            trainer = FakeTrainer(fail=fail)
            pipeline = self.setup_pipeline(folder, trainer)
            report = pipeline.run()
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["final_model_trained"])
            self.assertFalse((pipeline.root / "cohort-32").exists())
            self.assertEqual(p.read(pipeline.root / "COMPLETE.json")["outcome"], "failed")

    def test_incomplete_prediction_prefix_is_not_reused_as_complete(self):
        with temporary_directory() as folder:
            trainer = FakeTrainer()
            pipeline = self.setup_pipeline(folder, trainer)
            job = pipeline.spec("C0", 0, "content", 42, "omni")
            directory = pipeline.job_path(job)
            directory.mkdir(parents=True)
            write_jsonl(directory / "predictions.jsonl", [prediction(pipeline.rows[0])])
            result = pipeline.job(job)
            self.assertEqual(len(result), 6)
            calls = len(trainer.calls)
            self.assertEqual(pipeline.job(job), result)
            self.assertEqual(len(trainer.calls), calls)
            (directory / "predictions.jsonl").write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                pipeline.job(job)

    def test_changed_source_and_missing_lock_block_work(self):
        with temporary_directory() as folder:
            trainer = FakeTrainer()
            pipeline = self.setup_pipeline(folder, trainer)
            pipeline.identity["source_sha256"] = {"scripts/joint_affect_pipeline.py": "wrong"}
            with self.assertRaisesRegex(ValueError, "Source changed"):
                pipeline.child(["unused"], pipeline.root)
            with self.assertRaisesRegex(ValueError, "locked final"):
                pipeline.official([], {"variant": "C0", "vision_model": "omni"}, pipeline.root / "cohort-64")


if __name__ == "__main__":
    unittest.main()
