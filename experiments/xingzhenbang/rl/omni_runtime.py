"""Omni-specific single-GPU rollout and probability path.

Upstream TRL's generic VLM trainer does not forward Omni audio/video through
all probability paths. The companion loop uses TRL GRPOConfig and implements
its on-policy, one-iteration Dr.GRPO objective explicitly, with no value model.
"""
from __future__ import annotations

import contextlib
import csv
import inspect
import io
import os
import subprocess


def select_gpu(requested="auto", min_free_memory_gib=12):
    imported_torch = __import__("sys").modules.get("torch")
    if imported_torch is not None and imported_torch.cuda.is_initialized():
        raise RuntimeError("Select the GPU before CUDA initialization")
    text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free",
        "--format=csv,noheader,nounits"], text=True)
    devices = []
    for row in csv.reader(io.StringIO(text)):
        index, uuid, name, total, free = [v.strip() for v in row]
        devices.append({"index": index, "uuid": uuid, "name": name,
                        "total_mib": int(total), "free_mib": int(free)})
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        allowed = [value.strip() for value in visible.split(",") if value.strip()]
        devices = [d for d in devices if any(value == d["index"] or d["uuid"].startswith(value)
                                             for value in allowed)]
    if requested == "auto":
        processes = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True)
        busy = {line.strip() for line in processes.splitlines()}
        candidates = [d for d in devices if d["uuid"] not in busy
                      and d["free_mib"] >= min_free_memory_gib * 1024]
        if not candidates:
            raise RuntimeError("No idle GPU satisfies the free-memory requirement; specify an allocated --gpu")
        chosen = sorted(candidates, key=lambda d: (-d["free_mib"], int(d["index"])))[0]
    else:
        candidates = [d for d in devices if requested in (d["index"], d["uuid"])]
        if len(candidates) != 1:
            raise ValueError(f"Unknown or ambiguous GPU: {requested}")
        chosen = candidates[0]
        if chosen["free_mib"] < min_free_memory_gib * 1024:
            raise RuntimeError("The selected GPU has insufficient free memory")
    os.environ["CUDA_VISIBLE_DEVICES"] = chosen["uuid"]
    return chosen


def batch_for_completions(torch, prefix, completions, pad_id):
    """Duplicate one prompt's media, right-pad only its completion suffixes."""
    count = len(completions)
    if not count or any(tokens.numel() == 0 for tokens in completions):
        raise ValueError("Completions must contain at least an EOS token")
    prompt_ids = prefix["input_ids"]
    width = max(tokens.numel() for tokens in completions)
    ids = torch.full((count, width), pad_id, dtype=prompt_ids.dtype, device=prompt_ids.device)
    mask = torch.zeros_like(ids)
    for i, tokens in enumerate(completions):
        ids[i, :tokens.numel()] = tokens
        mask[i, :tokens.numel()] = 1
    inputs = {}
    for key, value in prefix.items():
        if key in ("input_ids", "attention_mask"):
            continue
        if key in ("token_type_ids", "position_ids"):
            raise RuntimeError(f"Unsupported processor field requires explicit extension: {key}")
        if torch.is_tensor(value):
            inputs[key] = value.repeat((count,) + (1,) * (value.ndim - 1))
        elif isinstance(value, (list, tuple)):
            inputs[key] = list(value) * count
        else:
            inputs[key] = value
    inputs["input_ids"] = torch.cat([prompt_ids.repeat(count, 1), ids], dim=1)
    inputs["attention_mask"] = torch.cat([prefix["attention_mask"].repeat(count, 1), mask], dim=1)
    return inputs, ids, mask


def dr_grpo_loss(torch, logps, reference_logps, mask, advantages, *, beta,
                 max_completion_length, completions_per_update):
    """Exactly on-policy, one optimizer iteration: old logps = detach(current).

    Sampling uses temperature=1, top_p=1, top_k=0, and disabled dropout. Thus
    the ratio is numerically 1 with the correct policy-gradient derivative.
    A second optimizer iteration on these rollouts is deliberately unsupported.
    """
    ratio = torch.exp(logps - logps.detach())
    difference = reference_logps - logps
    kl = torch.exp(difference) - difference - 1
    per_token = -ratio * advantages.unsqueeze(1) + beta * kl
    loss = (per_token * mask).sum() / (max_completion_length * completions_per_update)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite GRPO loss; the update was not applied")
    mean_kl = (kl.detach() * mask).sum() / mask.sum().clamp(min=1)
    return loss, mean_kl


class OmniPolicy:
    def __init__(self, model_path, sft_adapter_path, settings, *, policy_path=None, training=True):
        import torch
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        self.torch, self.settings = torch, settings
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("A CUDA GPU with BF16 support is required")
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path, local_files_only=True)
        tokenizer = self.processor.tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            raise RuntimeError("Tokenizer needs a pad or EOS token")
        self.eos_ids = tokenizer.eos_token_id
        omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, low_cpu_mem_usage=True,
            attn_implementation=settings["attn_implementation"],
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16))
        omni.disable_talker()
        thinker = omni.thinker
        del omni
        # Keep frozen nonquantized modules in the same dtype in training,
        # baseline evaluation, reward caching and final test inference.
        thinker = prepare_model_for_kbit_training(
            thinker, use_gradient_checkpointing=training,
            gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model = PeftModel.from_pretrained(
            thinker, policy_path or sft_adapter_path, adapter_name="policy",
            is_trainable=training, local_files_only=True)
        self.model.load_adapter(sft_adapter_path, adapter_name="reference", is_trainable=False,
                                local_files_only=True)
        self.verify_adapter(policy_path or sft_adapter_path, "policy")
        self.verify_adapter(sft_adapter_path, "reference")
        self.training = training
        self.model.set_adapter("policy")
        self._set_trainability()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        forward = inspect.signature(self.model.get_base_model().forward).parameters
        required = {"pixel_values_videos", "video_grid_thw", "input_features",
                    "feature_attention_mask", "use_audio_in_video"}
        if required - set(forward):
            raise RuntimeError(f"Omni forward is incompatible: missing {sorted(required-set(forward))}")
        self.model.eval()

    def verify_adapter(self, path, name):
        """Detect missing/prefix-mismatched SFT tensors, not just a nonzero parameter count."""
        from pathlib import Path
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file
        expected = load_file(str(Path(path) / "adapter_model.safetensors"), device="cpu")
        loaded = get_peft_model_state_dict(self.model, adapter_name=name)
        if set(expected) != set(loaded):
            raise RuntimeError(f"{name} adapter tensor names do not match the supplied artifact")
        for key, value in expected.items():
            if not self.torch.equal(value.float(), loaded[key].detach().cpu().float()):
                raise RuntimeError(f"{name} adapter weight was not loaded exactly: {key}")

    def _set_trainability(self):
        for name, parameter in self.model.named_parameters():
            language = not any(part in name.lower() for part in ("visual", "audio", "projector"))
            parameter.requires_grad = bool(self.training and ".policy." in name
                                           and "lora_" in name and language)
        if self.training and not any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("No language-policy LoRA parameters loaded")

    @contextlib.contextmanager
    def adapter(self, name):
        previous = self.model.active_adapter
        self.model.set_adapter(name)
        self._set_trainability()
        # Checkpointed backward is only used on the policy, never reference.
        try:
            yield
        finally:
            self.model.set_adapter(previous)
            self._set_trainability()

    def clear_positions(self):
        base = self.model.get_base_model()
        if hasattr(base, "rope_deltas"):
            base.rope_deltas = None

    def prepare(self, row, data_root):
        from .omni_io import prepare_omni_inputs
        return prepare_omni_inputs(self.processor, row, data_root, self.settings)

    @contextlib.contextmanager
    def suffix_projection(self, count):
        """Project only needed suffix states without dropping prompt/media context.

        Omni 5.8 lacks logits_to_keep. Its output Linear sits outside the
        checkpointed decoder blocks, so a scoped input hook avoids projecting
        the entire prompt vocabulary while preserving exact suffix gradients.
        """
        base = self.model.get_base_model()
        if "logits_to_keep" in inspect.signature(base.forward).parameters:
            yield {"logits_to_keep": count}
            return
        head = base.get_output_embeddings()
        if head is None:
            raise RuntimeError("Omni needs an output head for suffix projection")

        def trim_states(module, args):
            if not args or args[0].ndim != 3:
                raise RuntimeError("Unexpected output-head hidden-state shape")
            return (args[0][:, -count:, :], *args[1:])

        handle = head.register_forward_pre_hook(trim_states)
        try:
            yield {}
        finally:
            handle.remove()

    @contextlib.contextmanager
    def generation_settings(self, config):
        # Transformers 5.8 fills unspecified settings from the model even when
        # a fresh GenerationConfig is supplied. Replace that fallback too.
        base = self.model.get_base_model()
        previous = base.generation_config
        base.generation_config = config
        try:
            yield
        finally:
            base.generation_config = previous

    def generate(self, prefix, *, sample=False, adapter="policy"):
        from transformers import GenerationConfig
        torch = self.torch
        self.model.eval()
        self.clear_positions()
        # Do not inherit repetition penalties, suppression or other warpers from
        # the downloaded generation_config: they would invalidate on-policy logps.
        config = GenerationConfig(
            max_new_tokens=self.settings["max_completion_length"], do_sample=sample,
            use_cache=True, pad_token_id=self.pad_id,
            eos_token_id=self.model.generation_config.eos_token_id,
            num_beams=1, repetition_penalty=1.0,
            return_dict_in_generate=False)
        if sample:
            config.temperature, config.top_p, config.top_k = 1.0, 1.0, 0
        device_type = prefix["input_ids"].device.type
        with self.adapter(adapter), self.generation_settings(config), self.suffix_projection(1) as projection, \
                torch.no_grad(), torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            generated = self.model.generate(**prefix, generation_config=config, **projection)
        tokens = generated[0, prefix["input_ids"].shape[1]:].detach()
        if not tokens.numel():
            raise RuntimeError("Omni generated no completion tokens")
        return tokens, self.processor.tokenizer.decode(tokens, skip_special_tokens=True)

    def log_probabilities(self, inputs, completion_ids):
        torch = self.torch
        self.clear_positions()
        length = completion_ids.shape[1]
        with self.suffix_projection(length + 1) as projection, torch.autocast(
                inputs["input_ids"].device.type, dtype=torch.bfloat16,
                enabled=inputs["input_ids"].device.type == "cuda"):
            output = self.model(**inputs, **projection, use_cache=False, return_dict=True)
        logits = output.logits[:, -(length + 1):-1].float()
        if logits.shape[:2] != completion_ids.shape:
            raise RuntimeError("Completion logits are not autoregressively aligned")
        return logits.log_softmax(-1).gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)

    def backward_group(self, prefix, completions, advantages, micro_batch_size):
        torch = self.torch
        config = self.settings
        kl_total, loss_total = 0.0, 0.0
        for start in range(0, len(completions), micro_batch_size):
            chunk = completions[start:start + micro_batch_size]
            inputs, ids, mask = batch_for_completions(torch, prefix, chunk, self.pad_id)
            self.model.eval()
            with self.adapter("reference"), torch.no_grad():
                reference = self.log_probabilities(inputs, ids)
            # Switch before building the graph; keep policy active through checkpointed backward.
            self.model.train()
            # Frozen perception must also stay in eval mode (e.g. layerdrop or
            # audio augmentation), not merely have requires_grad=False.
            for name, module in self.model.named_modules():
                if any(part in name.lower() for part in ("visual", "audio", "projector")):
                    module.eval()
            logps = self.log_probabilities(inputs, ids)
            adv = torch.tensor(advantages[start:start + len(chunk)], device=ids.device)
            loss, kl = dr_grpo_loss(
                torch, logps, reference, mask, adv, beta=config["beta"],
                max_completion_length=config["max_completion_length"],
                completions_per_update=config["prompts_per_update"] * config["num_generations"])
            loss.backward()
            loss_total += float(loss.detach())
            kl_total += float(kl) * len(chunk) / len(completions)
        return {"loss": loss_total, "kl": kl_total}
