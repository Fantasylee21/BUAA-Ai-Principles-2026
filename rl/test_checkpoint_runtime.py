"""Real PEFT safetensors + optimizer/RNG round-trip on a tiny CPU model."""
import copy
import importlib.util
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

HAS_STACK = all(importlib.util.find_spec(name) is not None for name in ("torch", "peft", "transformers"))


@unittest.skipUnless(HAS_STACK, "CPU torch/transformers/peft are required")
class CheckpointRuntimeTests(unittest.TestCase):
    def test_peft_checkpoint_roundtrip_and_continued_optimizer_step(self):
        import numpy as np
        import torch
        from peft import (LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict,
                          set_peft_model_state_dict)
        from transformers import GPT2Config, GPT2LMHeadModel
        from rl.omni_runtime import OmniPolicy
        from rl.state import (adapter_digest, restore_rng, save_checkpoint, validate_checkpoint)

        torch.manual_seed(42)
        base = GPT2LMHeadModel(GPT2Config(vocab_size=24, n_embd=16, n_layer=1, n_head=2,
                                         resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
        original_base = copy.deepcopy(base)
        config = LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], fan_in_fan_out=True,
                            task_type="CAUSAL_LM", lora_dropout=0)
        model = get_peft_model(base, config, adapter_name="policy")
        sft_state = copy.deepcopy(get_peft_model_state_dict(model, adapter_name="policy"))
        model.add_adapter("reference", copy.deepcopy(config))
        set_peft_model_state_dict(model, sft_state, adapter_name="reference")
        model.set_adapter("policy")
        for name, param in model.named_parameters():
            param.requires_grad = ".policy." in name and "lora_" in name
        model.eval()

        class ProcessorStub:
            def save_pretrained(self, path):
                Path(path).mkdir(parents=True)
                (Path(path) / "test_processor.json").write_text("{}", encoding="utf-8")

        policy = SimpleNamespace(torch=torch, model=model, processor=ProcessorStub())
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=.01, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        input_ids = torch.tensor([[1, 3, 4, 5]])

        def update(m, opt, sched):
            opt.zero_grad()
            m(input_ids=input_ids, use_cache=False).logits.float().square().mean().backward()
            opt.step()
            sched.step()

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model.save_pretrained(root / "sft", selected_adapters=["policy"], safe_serialization=True)
            OmniPolicy.verify_adapter(policy, root / "sft/policy", "policy")
            reference_digest = adapter_digest(policy, "reference")
            update(model, optimizer, scheduler)
            identity = {"test": "real_peft_roundtrip"}
            checkpoint = save_checkpoint(root, policy, optimizer, scheduler, 1, identity)
            marker = validate_checkpoint(checkpoint, identity)
            expected_rng = (random.random(), np.random.rand(), torch.rand(1))

            restored = PeftModel.from_pretrained(copy.deepcopy(original_base), checkpoint / "adapter/policy",
                                                 adapter_name="policy", is_trainable=True, local_files_only=True)
            restored.load_adapter(root / "sft/policy", adapter_name="reference", is_trainable=False,
                                  local_files_only=True)
            restored.set_adapter("policy")
            for name, param in restored.named_parameters():
                param.requires_grad = ".policy." in name and "lora_" in name
            restored.eval()
            reloaded = SimpleNamespace(torch=torch, model=restored)
            self.assertEqual(adapter_digest(reloaded, "policy"), marker["policy_digest"])
            self.assertEqual(adapter_digest(reloaded, "reference"), reference_digest)
            opt2 = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=.01, weight_decay=0)
            sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda _: 1)
            state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
            opt2.load_state_dict(state["optimizer"])
            sched2.load_state_dict(state["scheduler"])
            restore_rng(torch, state["rng"])
            self.assertEqual(random.random(), expected_rng[0])
            self.assertEqual(np.random.rand(), expected_rng[1])
            self.assertTrue(torch.equal(torch.rand(1), expected_rng[2]))
            update(model, optimizer, scheduler)
            update(restored, opt2, sched2)
            for left, right in zip(model.parameters(), restored.parameters()):
                self.assertTrue(torch.allclose(left, right, atol=1e-7, rtol=1e-6))
            self.assertEqual(adapter_digest(reloaded, "reference"), reference_digest)

    def test_installed_omni_forward_accepts_required_media_fields(self):
        import inspect
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration
        fields = set(inspect.signature(Qwen2_5OmniThinkerForConditionalGeneration.forward).parameters)
        required = {"pixel_values_videos", "video_grid_thw", "input_features",
                    "feature_attention_mask", "use_audio_in_video"}
        self.assertFalse(required - fields, required - fields)

    def test_legacy_output_head_suffix_preserves_probabilities_and_gradients(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
        from rl.omni_runtime import OmniPolicy

        torch.manual_seed(7)
        base = GPT2LMHeadModel(GPT2Config(vocab_size=24, n_embd=16, n_layer=1, n_head=2,
                                         resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
        base.eval()

        class LegacyForward(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def get_base_model(self):
                return self

            def get_output_embeddings(self):
                return self.model.get_output_embeddings()

            def forward(self, input_ids, attention_mask, use_cache=False, return_dict=True):
                return self.model(input_ids=input_ids, attention_mask=attention_mask,
                                  use_cache=use_cache, return_dict=return_dict)

        model = LegacyForward(base)
        policy = object.__new__(OmniPolicy)
        policy.torch, policy.model = torch, model
        inputs = {"input_ids": torch.tensor([[1, 3, 4, 5, 2]]),
                  "attention_mask": torch.ones((1, 5), dtype=torch.long)}
        targets = inputs["input_ids"][:, -2:]
        full = model(**inputs).logits[:, -3:-1].log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        full.sum().backward()
        expected_gradients = {n: p.grad.clone() for n, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        projected_lengths = []
        observer = base.lm_head.register_forward_hook(lambda m, args, out: projected_lengths.append(out.shape[1]))
        try:
            suffix = policy.log_probabilities(inputs, targets)
        finally:
            observer.remove()
        self.assertEqual(projected_lengths, [3])
        self.assertTrue(torch.allclose(full, suffix, atol=1e-7))
        suffix.sum().backward()
        for name, param in model.named_parameters():
            self.assertTrue(torch.allclose(param.grad, expected_gradients[name], atol=1e-6, rtol=1e-5), name)
        self.assertEqual(model(**inputs).logits.shape[1], 5)

    def test_generation_does_not_inherit_probability_warpers(self):
        import torch
        from transformers import GenerationConfig
        from rl.omni_runtime import OmniPolicy
        from rl.test_omni_runtime import FakeOmni, make_prefix
        from rl.experiment import DEFAULT_EXPERIMENT

        policy = object.__new__(OmniPolicy)
        policy.torch, policy.settings = torch, dict(DEFAULT_EXPERIMENT)
        policy.training, policy.pad_id = True, 0
        policy.model = FakeOmni()
        policy.model.generation_config = GenerationConfig(eos_token_id=2, repetition_penalty=1.2,
                                                          suppress_tokens=[7], num_beams=4)
        policy.processor = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *a, **kw: "output"))
        captured = {}
        def generate(**kwargs):
            captured.update(kwargs)
            from transformers.generation.utils import GenerationMixin
            captured["resolved_config"], _ = GenerationMixin._prepare_generation_config(
                policy.model, kwargs["generation_config"])
            return torch.cat([kwargs["input_ids"], torch.tensor([[7, 2]])], dim=1)
        policy.model.generate = generate
        policy.generate(make_prefix(), sample=True)
        settings = captured["resolved_config"]
        self.assertTrue(settings.do_sample)
        self.assertEqual(settings.repetition_penalty, 1)
        self.assertIsNone(settings.suppress_tokens)
        self.assertEqual(settings.num_beams, 1)
        self.assertEqual((settings.temperature, settings.top_p, settings.top_k), (1, 1, 0))
        self.assertEqual(policy.model.generation_config.repetition_penalty, 1.2)
        self.assertEqual(policy.model.generation_config.suppress_tokens, [7])


if __name__ == "__main__":
    unittest.main()
