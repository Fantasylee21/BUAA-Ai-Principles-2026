"""CPU contracts for fold isolation, state persistence and real/remix batches."""
from copy import deepcopy
import json
from pathlib import Path
import pickle
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import joint_affect_data as data
from scripts import train_joint_affect as app
from tests.test_joint_affect_data import corpus, temporary_directory


class FakeOptimizer:
    def __init__(self):
        self.value, self.pending, self.updates = {"momentum": [1, 2]}, [], []

    def state_dict(self):
        return deepcopy(self.value)

    def load_state_dict(self, value):
        self.value = value

    def zero_grad(self, **kwargs):
        self.pending.clear()

    def step(self):
        self.updates.append(sum(self.pending))


class FakeScheduler:
    def __init__(self):
        self.value = {"last_epoch": 0}

    def state_dict(self):
        return deepcopy(self.value)

    def load_state_dict(self, value):
        self.value = value

    def step(self):
        self.value["last_epoch"] += 1

    def get_last_lr(self):
        return [1e-4]


class FakeTorch:
    def __init__(self):
        self.rng, self.cuda_rng = b"torch-rng", [b"cuda-rng"]
        self.cuda = SimpleNamespace(get_rng_state_all=lambda: self.cuda_rng,
            set_rng_state_all=lambda value: setattr(self, "cuda_rng", value),
            max_memory_reserved=lambda device: 1024)
        self.nn = SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=lambda *a, **k: .5))

    def get_rng_state(self):
        return self.rng

    def set_rng_state(self, value):
        self.rng = value

    @staticmethod
    def isfinite(value):
        return SimpleNamespace(all=lambda: True)

    @staticmethod
    def save(value, path):
        Path(path).write_bytes(pickle.dumps(value))

    @staticmethod
    def load(path, **kwargs):
        return pickle.loads(Path(path).read_bytes())


class FakeModel:
    def save_checkpoint(self, path, metadata):
        Path(path, "trainable.json").write_text(json.dumps({"metadata": metadata}), encoding="utf-8")
        Path(path, "trainable.pt").write_bytes(b"trainable-adapter")

    def load_checkpoint(self, path):
        self.loaded = Path(path)

    def parameters(self):
        return [SimpleNamespace(requires_grad=True)]

    def trainable_audit(self):
        return {"trainable": 2}

    def train(self):
        pass


class Loss(float):
    def __new__(cls, value, optimizer):
        result = float.__new__(cls, value)
        result.optimizer = optimizer
        return result

    def __truediv__(self, count):
        return Loss(float(self) / count, self.optimizer)

    def backward(self):
        self.optimizer.pending.append(float(self))

    def detach(self):
        return self

    def cpu(self):
        return self


class TrainingContracts(unittest.TestCase):
    def fixture(self, root):
        dataset = root / "Dataset"
        rows, _ = corpus(30)
        for i in range(30):
            path = dataset / f"media/parent-{i//2}/video-{i}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"media-{i if i != 1 else 0}".encode())
        source = dataset / "train.jsonl"
        source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        for name in ("val.jsonl", "test_release.jsonl"):
            (dataset / name).write_text("OFFICIAL MUST NOT BE READ", encoding="utf-8")
        manifest = data.prepare_manifest(source, dataset, expected_rows=None)
        path = root / "runs/train-manifest.json"
        app.atomic(path, manifest)
        args = SimpleNamespace(project_root=root, run_dir=root / "runs/content/fold0/C1",
            manifest=path, fold=0, grouping="content", seed=42, variant="C1",
            vision_model="omni", max_frames=64, init_second_from=None, gpu="1", gpus=None,
            deadline_unix=float("inf"), split="held")
        return args, manifest

    def test_manifest_identity_frames_and_held_inference_are_train_only(self):
        with temporary_directory() as tmp:
            args, manifest = self.fixture(Path(tmp))
            _, rows, train, held, identity = app.load_context(args)
            self.assertFalse(set(train) & set(held))
            self.assertEqual(app.inference_rows(args, manifest, rows, held), [rows[i] for i in held])
            self.assertEqual(identity["effective_batch"], 16)
            args.max_frames = 32
            with self.assertRaisesRegex(ValueError, "Immutable training artifact"):
                app.load_context(args)
            args.max_frames = 64
            changed = deepcopy(manifest)
            changed["train"][0]["views"]["M12"]["label"] = "positive"
            app.atomic(args.manifest, changed)
            with self.assertRaisesRegex(ValueError, "detached"):
                app.load_context(args)

    def test_gpu_selection_and_full_official_lock_are_explicit(self):
        args = SimpleNamespace(gpus=None, gpu="1", vision_model="omni")
        self.assertEqual(app.devices(args), ["1"])
        for gpus, vision in (("0", "omni"), ("1,1", "qwen3vl8b"), ("1", "qwen3vl8b"), ("1,2", "omni")):
            args.gpus, args.vision_model = gpus, vision
            with self.assertRaises(ValueError):
                app.devices(args)
        with temporary_directory() as tmp:
            args, manifest = self.fixture(Path(tmp))
            args.fold, args.split = "full", "test"
            with self.assertRaisesRegex(ValueError, "locked full-data"):
                app.inference_rows(args, manifest, manifest["train"], [])

    def test_checkpoint_roundtrip_restores_optimizer_scheduler_rng_and_hashes(self):
        with temporary_directory() as tmp:
            args, _ = self.fixture(Path(tmp))
            args.run_dir.mkdir(parents=True)
            torch, model, opt, sched = FakeTorch(), FakeModel(), FakeOptimizer(), FakeScheduler()
            position = {"epoch": 1, "offset": 16, "global_step": 4}
            sched.value["last_epoch"] = 4
            rng = app.state_rng(torch)
            marker = app.save_state(args, model, opt, sched, torch, position, rng)
            opt.value, sched.value = {}, {}
            torch.rng, torch.cuda_rng = b"changed", []
            random.seed(100)
            self.assertEqual(app.load_state(args, model, opt, sched, torch), position)
            self.assertEqual(opt.value, {"momentum": [1, 2]})
            self.assertEqual(sched.value, {"last_epoch": 4})
            self.assertEqual(app.state_rng(torch), rng)
            Path(marker["checkpoint"], "trainable.pt").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                app.load_state(args, model, opt, sched, torch)

    def test_probe_covers_unclear_neutral_and_never_held(self):
        rows, _ = corpus(30)
        train = list(range(40))
        rows[10]["views"]["M12"]["label"] = "neutral"
        selected = app.probe_indices(rows, train, priority=[55, 6])
        self.assertEqual(len(selected), 32)
        self.assertEqual(selected[0], 6)
        self.assertTrue(set(selected) <= set(train))
        self.assertIn(10, selected)
        self.assertTrue(any(not rows[i]["views"]["M1"]["is_clear"] for i in selected))
        self.assertTrue(any(not rows[i]["views"]["M2"]["is_clear"] for i in selected))

    def test_checkpoint_position_is_mathematically_consistent(self):
        self.assertEqual(app.validate_position({"epoch": 1, "offset": 16, "global_step": 3}, 18)["global_step"], 3)
        for position in ({"epoch": 3, "offset": 16, "global_step": 7},
                         {"epoch": 1, "offset": 17, "global_step": 3},
                         {"epoch": 2, "offset": 0, "global_step": 3},
                         {"epoch": True, "offset": 0, "global_step": 2}):
            with self.assertRaises(ValueError):
                app.validate_position(position, 18)

    def test_official_lock_binds_directory_and_marker_structurally(self):
        with temporary_directory() as tmp:
            args, manifest = self.fixture(Path(tmp))
            args.fold, args.split = "full", "test"
            args.run_dir.mkdir(parents=True)
            app.atomic(args.run_dir / "train.complete.json", {"complete": True})
            sha = app.digest(args.run_dir / "train.complete.json")
            app.atomic(args.run_dir / "final_model_lock.json", {"note": sha, "models": []})
            with self.assertRaisesRegex(ValueError, "does not bind"):
                app.inference_rows(args, manifest, manifest["train"], [])
            app.atomic(args.run_dir / "final_model_lock.json", {"models": [{"directory": str(args.run_dir), "complete_sha256": sha}]})
            with patch.object(data, "load_test", return_value=["locked-input"]) as loader:
                self.assertEqual(app.inference_rows(args, manifest, manifest["train"], []), ["locked-input"])
                self.assertEqual(loader.call_count, 1)

    def test_fixed_reference_uses_final_adapter_not_earlier_periodic_save(self):
        with temporary_directory() as tmp:
            root = Path(tmp)
            app.atomic(root / "trainer_state.json", {"epoch": 3., "global_step": 34})
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                (root / name).write_bytes(b"final")
            periodic = root / "checkpoint-25"
            periodic.mkdir()
            (periodic / "adapter_model.safetensors").write_bytes(b"earlier")
            final = app.reference_final_checkpoint(root)
            self.assertEqual((final / "adapter_model.safetensors").read_bytes(), b"final")
            self.assertEqual(app.reference_final_checkpoint(root), final)
            (root / "adapter_model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                app.reference_final_checkpoint(root)
            app.atomic(root / "trainer_state.json", {"epoch": 2.8})
            with self.assertRaises(InterruptedError):
                app.reference_final_checkpoint(root)

    def test_reference_preparation_uses_only_train_protocol_and_native_2x(self):
        import yaml
        from scripts import prepare_sft_data as sft
        with temporary_directory() as tmp:
            args, manifest = self.fixture(Path(tmp))
            args.run_dir.mkdir(parents=True)
            rows = manifest["train"]
            train = manifest["folds"]["content"]["folds"][0]["train"]
            questions = {rows[i]["sample_id"]: {"questions": [f"你愿意补充感受{j}吗？" for j in range(5)]} for i in train}
            audio = Path(tmp) / "cached.flac"
            audio.write_bytes(b"only-a-cache-fixture")
            with patch.object(sft, "load_questions", return_value=({}, questions)), \
                    patch.object(sft, "extracted_audio_path", return_value=audio), \
                    patch.object(sft, "extract_one_audio", side_effect=AssertionError("No extraction in CPU test")):
                path = app.reference_configuration(args, manifest, rows, train)
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(config["num_train_epochs"], 3.)
            self.assertEqual(config["flash_attn"], "sdpa")
            self.assertFalse(config["do_eval"])
            self.assertNotIn("eval_dataset", config)
            self.assertEqual(config["quantization_bit"], 4)
            self.assertEqual(config["lora_rank"], 16)
            self.assertEqual(config["gradient_accumulation_steps"], 16)
            directory = args.run_dir / "reference-data"
            records = [json.loads(line) for name in ("aligned.jsonl", "conflict.jsonl")
                for line in (directory / name).read_text(encoding="utf-8").splitlines()]
            self.assertEqual({r["sample_id"] for r in records}, {rows[i]["sample_id"] for i in train})
            self.assertTrue(all(not r["use_in_main"] for r in records))
            for row in records:
                self.assertEqual(row["input_modalities"], ["text", "vision"] if row["source_protocol"] == "VT" else ["audio", "vision"])
                self.assertEqual("Supplied text:" in row["messages"][0]["content"], row["source_protocol"] == "VT")
            count = sum(json.loads(r["messages"][-1]["content"])["is_uncertain"] == "yes" for r in records)
            self.assertEqual(app.read(directory / "dataset_info.json")["joint_conflict"]["num_samples"], 2 * count)

    def test_real_and_remix_losses_have_separate_batch_normalization(self):
        with temporary_directory() as tmp:
            args, manifest = self.fixture(Path(tmp))
            args.run_dir.mkdir(parents=True)
            rows, train = manifest["train"], list(range(18))
            torch, model, opt, sched = FakeTorch(), FakeModel(), FakeOptimizer(), FakeScheduler()
            observed = []
            def sample(model, row, all_rows, counts, phase):
                observed.append((phase, bool(row.get("_remix"))))
                loss = Loss(.25 if row.get("_remix") else 1., opt)
                return {"warmup_loss": loss, "main_loss": loss, "remix_loss": loss}, {}
            with patch.object(app, "setup_torch", return_value=torch), \
                    patch.object(app, "make_model", return_value=model), \
                    patch.object(app, "optimizer_for", return_value=(opt, sched)), \
                    patch.object(app, "sample_loss", side_effect=sample), \
                    patch.object(app, "emit"), patch.object(app, "stopped", return_value=False):
                result = app.train_new(args, manifest, rows, train, {"test": True})
            self.assertEqual(len(opt.updates), 6)
            for value, expected in zip(opt.updates, [1., 1., 1.25, 1.25, 1.25, 1.25]):
                self.assertAlmostEqual(value, expected)
            self.assertEqual(observed.count(("evidence_warmup", False)), 18)
            self.assertEqual(observed.count(("main", False)), 36)
            self.assertEqual(observed.count(("main", True)), 18)
            self.assertEqual(result["optimizer_steps"], 6)
            self.assertEqual(app.read(args.run_dir / "latest.json")["position"],
                             {"epoch": 3, "offset": 0, "global_step": 6})


if __name__ == "__main__":
    unittest.main()
