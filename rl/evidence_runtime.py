"""Native Omni with no PEFT adapter: isolated, read-only feature extraction."""
from __future__ import annotations

import time

from .finite_runtime import FinitePolicy
from .omni_io import prepare_task_inputs
from .reward import VALID_EMOTIONS


class BaseEvidenceExtractor(FinitePolicy):
    def __init__(self, model_path, settings):
        import torch
        from transformers import BitsAndBytesConfig, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        from peft import prepare_model_for_kbit_training
        self.torch, self.settings = torch, settings
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path, local_files_only=True)
        omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, local_files_only=True,
            torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}, low_cpu_mem_usage=True,
            attn_implementation=settings["attn_implementation"],
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
        omni.disable_talker()
        thinker = prepare_model_for_kbit_training(omni.thinker, use_gradient_checkpointing=False)
        del omni
        # A structural wrapper shares only helper signatures; no adapter is constructed or loaded.
        class PlainThinker(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.thinker = model
            def forward(self, **kwargs):
                return self.thinker(**kwargs)
            def get_base_model(self):
                return self.thinker
        self.model = PlainThinker(thinker)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self.value_ids = [self.processor.tokenizer.encode(v, add_special_tokens=False) for v in VALID_EMOTIONS]
        if any(len(ids) != 1 for ids in self.value_ids):
            raise ValueError("Emotion values must each be one token")
        self.value_ids = [ids[0] for ids in self.value_ids]
        self.prefix_ids = self.processor.tokenizer.encode('{"pred_emotion":"', add_special_tokens=False)
        self.check_frozen()

    def check_frozen(self):
        if any(p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise RuntimeError("Base Omni acquired a gradient")
        if any("lora_" in n or "adapter" in n for n, _ in self.model.named_parameters()):
            raise RuntimeError("A forbidden adapter entered the evidence extractor")
        return {n: int(p._version) for n, p in self.model.named_parameters()}

    def hidden(self, prefix, features=None):
        torch = self.torch
        suffix = torch.tensor([self.prefix_ids], dtype=prefix["input_ids"].dtype, device="cuda:0")
        inputs = {**prefix, "input_ids": torch.cat((prefix["input_ids"], suffix), -1),
            "attention_mask": torch.cat((prefix["attention_mask"], torch.ones_like(suffix)), -1)}
        captured = []
        def capture(module, args):
            captured.append(args[0][0, -1, :].detach().float().clone())
        self.clear_positions()
        with self.cached_perception(features), self.suffix_projection(1) as projection:
            handle = self.model.get_base_model().get_output_embeddings().register_forward_pre_hook(capture)
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    result = self.model(**inputs, **projection, use_cache=False, return_dict=True)
            finally:
                handle.remove()
        if len(captured) != 1 or captured[0].shape != (2048,):
            raise RuntimeError("Wrong base hidden-state extraction position")
        logits = result.logits[0, -1, self.value_ids].detach().float()
        return captured[0], logits

    def extract(self, row, data_root, mode, cached=True):
        started = time.monotonic()
        inputs = prepare_task_inputs(self.processor, row, data_root, self.settings, mode)
        perception = self.perception(inputs) if cached else None
        hidden, logits = self.hidden(inputs, perception)
        return hidden.cpu().numpy(), logits.cpu().numpy(), time.monotonic()-started
