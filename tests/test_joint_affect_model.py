"""CPU differentiability/contracts only; no HF weight load or efficacy claim."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from scripts import joint_affect_model as app


def row(protocol="VT", clear=True, conflict=True):
    return {"sample_id": "PRIVATE_SAMPLE_ID", "protocol": protocol, "video": "video.mp4",
        "text_content": "PRIVATE_SUPPLIED_TEXT", "source_dataset": "ch_sims_v2",
        "sample_type": "Conflict" if conflict else "Aligned", "views": {
            "M1": {"label": "negative" if clear else "neutral", "is_clear": clear},
            "M2": {"label": "positive" if conflict else "negative", "is_clear": True},
            "M12": {"label": "positive" if conflict else "negative", "is_clear": True}}}


class FakeBackend(nn.Module):
    def __init__(self, visual_width=6):
        super().__init__()
        self.vision_hidden_size, self.second_hidden_size = visual_width, 8
        self.vision_model = nn.Linear(5, visual_width)
        self.second_model = nn.Linear(5, 8)
        self.calls = []

    def encode_vision(self, record):
        self.calls.append(("vision", record["sample_id"]))
        tokens = self.vision_model(torch.arange(20, dtype=torch.float32).reshape(1, 4, 5) / 20)
        return app.TokenBatch(tokens, audit={"mode": "vision", "audio": False})

    def encode_second(self, record):
        self.calls.append(("second", record["sample_id"]))
        tokens = self.second_model(torch.arange(15, dtype=torch.float32).reshape(1, 3, 5) / 15)
        return app.TokenBatch(tokens, audit={"mode": "audio" if record["protocol"] == "VA" else "text"})

    def parameter_group(self, name):
        return "vision_perception" if name.startswith("vision_model") else "audio_perception"


def model(variant="C1", visual_width=6):
    return app.JointAffectModel(FakeBackend(visual_width), variant,
        app.ModelConfig(width=16, queries=4, dropout=0.))


COUNTS = {"real_total": 100, "aligned": 80, "natural_conflict": 10,
          "synthetic_conflict": 10, "unique_videos": 50, "video_frequency": {"video.mp4": 2}}


class JointAffectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)

    def test_joint_distribution_avoids_independent_marginal_entropy_trap(self):
        logits = torch.full((1, 16), -100.)
        logits[0, 0] = logits[0, 10] = 0.
        joint, ulogit, probability = app.joint_uncertainty(logits)
        self.assertLess(float(probability), 1e-20)
        marginal = joint.sum(2)
        old_q = 1 - (marginal * joint.sum(1)).sum()
        self.assertAlmostEqual(float(old_q), .5, places=6)
        self.assertTrue(torch.allclose(probability, ulogit.sigmoid(), atol=1e-7))
        logits.fill_(-100.)
        logits[0, 15] = 0.
        self.assertAlmostEqual(float(app.joint_uncertainty(logits)[2]), 1., places=6)

    def test_shapes_and_actual_gradients_reach_both_streams_and_interaction(self):
        current = model()
        output = current.forward_row(row())
        self.assertEqual(tuple(output["joint_probs"].shape), (1, 4, 4))
        self.assertEqual(tuple(output["emotion_probs"].shape), (1, 3))
        self.assertEqual(tuple(output["vision_probs"].shape), (1, 4))
        loss = app.compute_objectives(output, app.make_targets(row()), COUNTS)["main_loss"]
        loss.backward()
        for module in (current.backbone.vision_model, current.backbone.second_model,
                       current.vision_pool, current.second_pool, current.fusion, current.joint_head,
                       current.emotion_head, current.vision_head, current.second_head):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
        with self.assertRaises(ValueError):
            current.forward_tokens(app.TokenBatch(torch.ones(1, 2, 6), torch.zeros(1, 2, dtype=torch.bool)), torch.ones(1, 2, 8))

    def test_four_state_clarity_and_remix_never_fabricate_emotion_labels(self):
        original = row(clear=False)
        target = app.make_targets(original)
        self.assertEqual(target["vision"], 3)
        self.assertEqual(target["second"], 2)
        self.assertEqual(target["joint"], 14)
        self.assertEqual(target["emotion"], 2)
        mixed = copy.deepcopy(original)
        mixed.update(is_remix=True, emotion_mask=0)
        del mixed["views"]["M12"]
        target = app.make_targets(mixed)
        self.assertEqual((target["emotion"], target["emotion_mask"]), (-100, 0))
        output = model("C0").forward_row(mixed)
        output["emotion_logits"].retain_grad()
        loss = app.compute_objectives(output, target, COUNTS, remix=True)
        self.assertTrue(torch.allclose(loss["remix_loss"], .25 * loss["relation_loss"]))
        loss["remix_loss"].backward()
        self.assertTrue(output["emotion_logits"].grad is None or output["emotion_logits"].grad.abs().max() == 0)

    def test_global_condition_weights_survive_single_row_microbatches(self):
        def outputs():
            return {"variant": "C0", "emotion_logits": torch.zeros(1, 3, requires_grad=True),
                "vision_logits": torch.zeros(1, 4, requires_grad=True),
                "second_logits": torch.zeros(1, 4, requires_grad=True),
                "uncertain_logit": torch.zeros(1, requires_grad=True)}
        expected = {"aligned": .5, "natural_conflict": 3., "synthetic_conflict": 3.}
        values = {}
        for category, factor in expected.items():
            target = {"vision": 0, "second": 0, "emotion": 0, "emotion_mask": 1,
                "relation": int(category != "aligned"), "category": category, "video": "video.mp4"}
            result = app.compute_objectives(outputs(), target, COUNTS)
            relation_term = result["main_loss"] - torch.log(torch.tensor(3.))
            self.assertAlmostEqual(float(relation_term), factor * float(torch.log(torch.tensor(2.))), places=6)
            values[category] = float(relation_term)
            self.assertEqual(result["visual_weight"], 1.)
        mean = (80 * values["aligned"] + 10 * values["natural_conflict"] + 10 * values["synthetic_conflict"]) / 100
        self.assertAlmostEqual(mean, float(torch.log(torch.tensor(2.))), places=6)

    def test_c0_warmup_trains_auxiliary_heads_but_main_does_not(self):
        current = model("C0")
        target = app.make_targets(row())
        warmup = app.compute_objectives(current.forward_row(row()), target, COUNTS, phase="evidence_warmup")
        warmup["warmup_loss"].backward()
        self.assertGreater(float(current.vision_head.weight.grad.abs().sum()), 0)
        self.assertGreater(float(current.second_head.weight.grad.abs().sum()), 0)
        current.zero_grad(set_to_none=True)
        app.compute_objectives(current.forward_row(row()), target, COUNTS)["main_loss"].backward()
        self.assertTrue(current.vision_head.weight.grad is None or current.vision_head.weight.grad.abs().max() == 0)
        self.assertGreater(float(current.uncertainty_head.weight.grad.abs().sum()), 0)

    def test_prompt_and_remix_routing_are_isolated_from_annotations(self):
        original = row("VA")
        for mode in ("vision", "audio"):
            messages = app.isolated_messages(original, mode, "/media/file.mp4")
            text = " ".join(p.get("text", "") for message in messages for p in message["content"])
            self.assertNotIn(original["sample_id"], text)
            self.assertNotIn(original["text_content"], text)
            kinds = [p["type"] for message in messages for p in message["content"]]
            self.assertEqual(kinds.count("video"), int(mode == "vision"))
            self.assertEqual(kinds.count("audio"), int(mode == "audio"))
        other = copy.deepcopy(original)
        other["sample_id"] = "SECOND_SOURCE"
        current = model()
        current.forward_row(original, second_row=other)
        self.assertEqual(current.backbone.calls, [("vision", "PRIVATE_SAMPLE_ID"), ("second", "SECOND_SOURCE")])

    def test_optimizer_audit_and_checkpoint_roundtrip_and_visual_upgrade(self):
        current = model().eval()
        audit = current.trainable_audit()
        self.assertEqual(set(audit["parameter_roles"]), {n for n, p in current.named_parameters() if p.requires_grad})
        groups = current.optimizer_groups(backbone_lr=1e-4, head_lr=3e-4)
        self.assertTrue(all(g["lr"] == 2e-5 for g in groups if "perception" in g["name"]))
        optimizer = torch.optim.AdamW(groups)
        app.compute_objectives(current.forward_row(row()), app.make_targets(row()), COUNTS)["main_loss"].backward()
        optimizer.step()
        expected = current.forward_row(row())["emotion_probs"].detach()
        with tempfile.TemporaryDirectory() as tmp:
            current.save_checkpoint(tmp, {"fold": 0, "kind": "CPU_TEST_ONLY"})
            restored = model().eval()
            restored.load_checkpoint(tmp)
            self.assertTrue(torch.equal(expected, restored.forward_row(row())["emotion_probs"]))
            upgraded = model(visual_width=10)
            visual_before = {n: p.detach().clone() for n, p in upgraded.named_parameters()
                             if n.startswith(("backbone.vision_model.", "vision_pool."))}
            report = upgraded.init_second_from(tmp)
            self.assertTrue(report["loaded"])
            for name, p in upgraded.named_parameters():
                if name in visual_before:
                    self.assertTrue(torch.equal(p, visual_before[name]))
            self.assertTrue(torch.equal(upgraded.backbone.second_model.weight, current.backbone.second_model.weight))
            path = Path(tmp) / "trainable.pt"
            path.write_bytes(path.read_bytes()[:-5] + b"WRONG")
            with self.assertRaises(ValueError):
                restored.load_checkpoint(tmp)

    def test_exact_hf_module_plan_selects_top_four_and_dense_connectors(self):
        toy = nn.Module()
        toy.model = nn.Module()
        toy.model.layers = nn.ModuleList([nn.ModuleDict({"q_proj": nn.Linear(2, 2), "up_proj": nn.Linear(2, 2)})])
        toy.visual = nn.Module()
        toy.visual.blocks = nn.ModuleList([nn.ModuleDict({"attn": nn.ModuleDict({"qkv": nn.Linear(2, 2)})}) for _ in range(6)])
        toy.visual.merger = nn.Sequential(nn.Linear(2, 2))
        toy.audio_tower = nn.Module()
        toy.audio_tower.layers = nn.ModuleList([nn.ModuleDict({"fc1": nn.Linear(2, 2)}) for _ in range(7)])
        toy.audio_tower.proj = nn.Linear(2, 2)
        plan = app.module_plan(toy, "both")
        self.assertEqual(plan["top_block_ids"], {"vision": [2, 3, 4, 5], "audio": [3, 4, 5, 6]})
        self.assertEqual(plan["connectors"], ["audio_tower.proj", "visual.merger"])
        self.assertEqual(plan["decoder"], ["model.layers.0.q_proj", "model.layers.0.up_proj"])
        self.assertEqual(len(plan["perception"]), 8)

    def test_omni_capture_preserves_gradients_and_projects_zero_vocab_positions(self):
        class Causal(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder, self.lm_head = nn.Linear(4, 4), nn.Linear(4, 7)
                self.last_logit_shape = None
            def get_base_model(self):
                return self
            def get_output_embeddings(self):
                return self.lm_head
            def forward(self, x, **kwargs):
                logits = self.lm_head(self.encoder(x))
                self.last_logit_shape = logits.shape
                return SimpleNamespace(logits=logits)
        causal = Causal()
        hidden = app.capture_omni_states(causal, {"x": torch.ones(1, 3, 4)})
        self.assertEqual(tuple(causal.last_logit_shape), (1, 0, 7))
        hidden.square().mean().backward()
        self.assertGreater(float(causal.encoder.weight.grad.abs().sum()), 0)
        self.assertEqual(len(causal.lm_head._forward_pre_hooks), 0)

    def test_omni_processor_metadata_is_verified_and_removed_before_forward(self):
        metadata = {"total_num_frames": 100, "fps": 25., "frames_indices": [0, 33, 66, 99]}
        class Processor:
            list_output = False
            video_processor = SimpleNamespace(patch_size=2, temporal_patch_size=2, merge_size=2)
            def apply_chat_template(self, *args, **kwargs):
                return "static isolated video prompt"
            def __call__(self, **kwargs):
                result = {"input_ids": torch.tensor([[99] * 8 + [1]]),
                    "attention_mask": torch.ones(1, 9, dtype=torch.long),
                    "pixel_values_videos": torch.zeros(32, 24),
                    "video_grid_thw": torch.tensor([[2, 4, 4]]),
                    "video_second_per_grid": [1.], "video_metadata": [copy.deepcopy(metadata)]}
                if self.list_output:
                    result["mm_token_type_ids"] = [[2] * 8 + [0]]
                    result["input_ids"] = result["input_ids"].tolist()
                    result["attention_mask"] = result["attention_mask"].tolist()
                return result
        backend = app.HFBackend.__new__(app.HFBackend)
        nn.Module.__init__(backend)
        backend.config = app.HFBackendConfig("unused", "unused", ".", vision_device="cpu", second_device="cpu")
        backend.media_audit = []
        backend.vision_processor = Processor()
        backend.vision_info = {"kind": "omni"}
        backend.vision_model = SimpleNamespace(get_base_model=lambda: SimpleNamespace(config=SimpleNamespace(video_token_id=99)))
        backend._media_path = lambda record: Path("unused-video.mp4")
        backend._decode_video = lambda path: ([None] * 4,
            {"total_num_frames": 100, "fps": 25., "frames_indices": [0, 33, 66, 99]}, {"sampled_frames": 4})
        inputs, mask, audit = backend.prepare_inputs(row(), "vision")
        self.assertNotIn("video_metadata", inputs)
        self.assertEqual(audit["processor_video_metadata"], metadata)
        self.assertEqual(int(mask.sum()), 8)
        self.assertFalse(inputs["use_audio_in_video"])
        metadata["frames_indices"][-1] = 98
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            backend.prepare_inputs(row(), "vision")
        metadata["frames_indices"][-1] = 99
        backend.vision_info = {"kind": "qwen3_vl"}
        backend.vision_processor.list_output = True
        inputs, mask, _ = backend.prepare_inputs(row(), "vision")
        self.assertTrue(torch.is_tensor(inputs["input_ids"]))
        self.assertTrue(torch.is_tensor(inputs["mm_token_type_ids"]))
        self.assertEqual(int(mask.sum()), 8)


if __name__ == "__main__":
    unittest.main()
