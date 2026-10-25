"""GRPO training script skeleton for the RL post-training stage.

Post-trains Qwen2.5-Omni-3B (with an existing SFT LoRA adapter) for emotion
recognition with uncertainty expression, using TRL's GRPOTrainer + PEFT.
"""

import argparse
import json
import os
import sys

# Make sibling modules (reward.py) importable regardless of the cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reward import (
    build_reward_funcs,
    reward_weights_for_phase,
)

MAX_FRAMES = 8


def freeze_by_name_substring(model, substrings):
    """Freeze every parameter whose qualified name contains any substring."""
    n_frozen = 0
    for name, param in model.named_parameters():
        if any(sub in name for sub in substrings):
            param.requires_grad = False
            n_frozen += 1
    return n_frozen


def parse_args():
    parser = argparse.ArgumentParser(
        description="GRPO RL stage for emotion recognition with uncertainty."
    )
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-Omni-3B",
                        help="Base model path or HF id (Qwen2.5-Omni-3B).")
    parser.add_argument("--sft-adapter-path", default=None,
                        help="Path to the existing SFT LoRA adapter to continue training from.")
    parser.add_argument("--data", default="rl/data/rl_prompts.jsonl",
                        help="RL prompt JSONL produced by prepare_rl_data.py.")
    parser.add_argument("--output-dir", default="runs/grpo",
                        help="Checkpoint / log output directory.")
    parser.add_argument("--curriculum-phase", choices=["warmup", "full"],
                        default="warmup",
                        help="warmup = format+emotion rewards only; full = all components.")
    parser.add_argument("--use-vllm", action="store_true",
                        help="Use vLLM colocate generation for rollouts.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data, print one sample, and exit.")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="LoRA learning rate (default: 1e-5).")
    parser.add_argument("--num-generations", type=int, default=8,
                        help="Completions sampled per prompt (default: 8).")
    parser.add_argument("--beta", type=float, default=0.01,
                        help="KL coefficient to the SFT reference policy (default: 0.01).")
    parser.add_argument("--max-completion-length", type=int, default=128,
                        help="Max new tokens per completion (default: 128).")
    parser.add_argument("--max-prompt-length", type=int, default=None,
                        help="Optional prompt truncation length (tokens).")
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_prompt_dataset(path):
    with open(path, "r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def main():
    args = parse_args()

    rows = load_prompt_dataset(args.data)
    print(f"Loaded {len(rows)} RL prompts from {args.data}")

    if args.dry_run:
        sample = rows[0]
        print("--- dry run: sample row ---")
        print(f"sample_id   : {sample['sample_id']}")
        print(f"sample_type : {sample['sample_type']}")
        print(f"gt_label    : {sample['gt_label']}")
        print("dry run OK; exiting before trainer init.")
        return

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import (
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniProcessor,
        TrainerCallback,
    )
    from trl import GRPOConfig, GRPOTrainer

    # ------------------------------------------------------------------
    # Robust fused_lm_head patch for Qwen2.5-Omni / PEFT models
    # ------------------------------------------------------------------
    import types
    import torch.nn.functional as F
    from dataclasses import dataclass
    from transformers.utils import ModelOutput
    import trl.trainer.utils as _trl_utils
    import trl.trainer.grpo_trainer as _trl_grpo

    @dataclass
    class _SafeFusedCausalLMOutput(ModelOutput):
        loss: torch.Tensor | None = None
        log_probs: torch.Tensor | None = None
        entropy: torch.Tensor | None = None
        log_sum_sq_probs: torch.Tensor | None = None
        mean_logits: torch.Tensor | None = None
        is_top1: torch.Tensor | None = None
        label_mask: torch.Tensor | None = None
        aux_loss: torch.Tensor | None = None

    def _safe_add_fused_lm_head(target_model, temperature=1.0, **kwargs):
        orig_forward = target_model.forward

        def _safe_fused_forward(
            self,
            *args,
            fused_lm_head=False,
            labels=None,
            shift_labels=None,
            num_items_in_batch=None,
            **fwd_kwargs,
        ):
            if not fused_lm_head:
                if labels is not None:
                    fwd_kwargs["labels"] = labels
                if shift_labels is not None:
                    fwd_kwargs["shift_labels"] = shift_labels
                if num_items_in_batch is not None:
                    fwd_kwargs["num_items_in_batch"] = num_items_in_batch
                return orig_forward(*args, **fwd_kwargs)

            fwd_kwargs["use_cache"] = False
            out = orig_forward(*args, labels=None, **fwd_kwargs)

            if hasattr(out, "log_probs") and out.log_probs is not None:
                return out

            logits = getattr(out, "logits", None)
            if logits is None:
                hs = getattr(out, "last_hidden_state", None)
                if hs is None and hasattr(out, "hidden_states") and out.hidden_states:
                    hs = out.hidden_states[-1]
                if hs is not None and hasattr(self, "get_output_embeddings"):
                    logits = self.get_output_embeddings()(hs)
                else:
                    raise AttributeError(
                        f"Cannot compute log_probs: model output has neither 'logits' nor 'last_hidden_state'."
                    )

            scaled_logits = logits.float()
            if temperature != 1.0 and temperature > 0:
                scaled_logits = scaled_logits / temperature

            if shift_labels is None and labels is not None:
                shift_logits = scaled_logits[..., :-1, :].contiguous()
                target_labels = labels[..., 1:].contiguous()
            else:
                shift_logits = scaled_logits
                target_labels = shift_labels

            if target_labels is not None:
                mask = (target_labels != -100)
                safe_labels = target_labels.clone()
                safe_labels[~mask] = 0

                log_probs_full = F.log_softmax(shift_logits, dim=-1)
                per_token_logps = log_probs_full.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
                per_token_logps = per_token_logps.masked_fill(~mask, 0.0)

                probs = torch.softmax(shift_logits, dim=-1)
                entropy = -(probs * log_probs_full).sum(dim=-1).masked_fill(~mask, 0.0)
            else:
                per_token_logps = None
                entropy = None
                mask = None

            return _SafeFusedCausalLMOutput(
                loss=getattr(out, "loss", None),
                log_probs=per_token_logps,
                entropy=entropy,
                label_mask=mask,
                aux_loss=getattr(out, "aux_loss", None),
            )

        target_model.forward = types.MethodType(_safe_fused_forward, target_model)

    for _mod in (_trl_utils, _trl_grpo):
        if hasattr(_mod, "add_fused_lm_head"):
            _mod.add_fused_lm_head = _safe_add_fused_lm_head
        if hasattr(_mod, "patch_fused_lm_head"):
            _mod.patch_fused_lm_head = _safe_add_fused_lm_head
    import trl as _trl
    if hasattr(_trl, "add_fused_lm_head"):
        _trl.add_fused_lm_head = _safe_add_fused_lm_head
    if hasattr(_trl, "patch_fused_lm_head"):
        _trl.patch_fused_lm_head = _safe_add_fused_lm_head

    for _mod_name, _mod in list(sys.modules.items()):
        if _mod_name == "trl" or _mod_name.startswith("trl."):
            for _attr in ("add_fused_lm_head", "patch_fused_lm_head", "create_fused_lm_head"):
                if hasattr(_mod, _attr):
                    setattr(_mod, _attr, _safe_add_fused_lm_head)

    # ------------------------------------------------------------------
    # PyAV compatibility shim
    # ------------------------------------------------------------------
    import av as _av
    try:
        _av_major = int(str(_av.__version__).split(".")[0])
    except (ValueError, IndexError):
        _av_major = 0
    if _av_major >= 15:
        _orig_av_open = _av.open

        def _av_open_compat(*a, **kw):
            kw.pop("metadata_errors", None)
            kw.pop("metadata_encoding", None)
            return _orig_av_open(*a, **kw)

        _av.open = _av_open_compat
        print(f"Note: PyAV {_av.__version__} >= 15 patched for torchvision compatibility.")

    # ------------------------------------------------------------------
    # Model + adapter path auto-detection
    # ------------------------------------------------------------------
    if args.sft_adapter_path is None and os.path.isdir(args.model_path):
        adapter_cfg_file = os.path.join(args.model_path, "adapter_config.json")
        if os.path.isfile(adapter_cfg_file):
            print(f"[Auto-detect] Treating --model-path ({args.model_path}) as --sft-adapter-path.")
            args.sft_adapter_path = args.model_path
            try:
                with open(adapter_cfg_file, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                base_name = cfg_data.get("base_model_name_or_path")
                if base_name and (os.path.exists(base_name) or "/" in base_name):
                    args.model_path = base_name
                else:
                    args.model_path = "Qwen/Qwen2.5-Omni-3B"
            except Exception:
                args.model_path = "Qwen/Qwen2.5-Omni-3B"

    # ------------------------------------------------------------------
    # Processor & Video Sampling
    # ------------------------------------------------------------------
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    _vp = processor.video_processor
    _vp.do_sample_frames = True
    _vp.fps = 2.0
    print(
        "Video frame sampling: "
        f"do_sample_frames={_vp.do_sample_frames}, fps={_vp.fps}, "
        f"min_frames={getattr(_vp, 'min_frames', None)}, "
        f"max_frames={getattr(_vp, 'max_frames', None)}"
    )

    _orig_apply_chat_template = processor.apply_chat_template

    def _cleaning_apply_chat_template(conversation, **kwargs):
        def clean_conv(conv):
            cleaned = []
            for m in conv:
                content = m.get("content")
                if isinstance(content, list):
                    content = [
                        {k: v for k, v in p.items() if v is not None}
                        if isinstance(p, dict) else p
                        for p in content
                    ]
                cleaned.append({**m, "content": content})
            return cleaned

        is_batched = (
            isinstance(conversation, (list, tuple))
            and len(conversation) > 0
            and isinstance(conversation[0], (list, tuple))
        )
        cleaned_conversation = (
            [clean_conv(c) for c in conversation] if is_batched else clean_conv(conversation)
        )
        return _orig_apply_chat_template(cleaned_conversation, **kwargs)

    processor.apply_chat_template = _cleaning_apply_chat_template

    # ------------------------------------------------------------------
    # Model Loading (Omni composite -> Thinker extraction)
    # ------------------------------------------------------------------
    omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype="bfloat16",
        attn_implementation="sdpa",
    )
    omni.disable_talker()

    if args.sft_adapter_path:
        omni.thinker = PeftModel.from_pretrained(
            omni.thinker, args.sft_adapter_path, is_trainable=True
        )
    else:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        omni.thinker = get_peft_model(omni.thinker, lora_config)

    omni.thinker.print_trainable_parameters()
    model = omni.thinker

    n_frozen = freeze_by_name_substring(model, ["visual", "audio", "projector"])
    print(f"Froze {n_frozen} parameters (vision/audio/projector towers).")

    reward_funcs = build_reward_funcs(args.curriculum_phase)
    reward_weights = reward_weights_for_phase(args.curriculum_phase)
    print(f"Curriculum phase: {args.curriculum_phase}")
    print(f"Reward funcs : {[f.__name__ for f in reward_funcs]}")
    print(f"Reward weights: {reward_weights}")

    # ------------------------------------------------------------------
    # GRPO Config
    # ------------------------------------------------------------------
    grpo_kwargs = dict(
        output_dir=args.output_dir,
        temperature=1.0000001,
        top_p=0.99,
        num_generations=args.num_generations,
        beta=args.beta,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        bf16=True,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=5,
        save_steps=100,
        seed=args.seed,
        reward_weights=reward_weights,
        report_to=[],
    )
    import inspect
    valid_params = set(inspect.signature(GRPOConfig.__init__).parameters)
    dropped = sorted(k for k, v in grpo_kwargs.items() if k not in valid_params)
    if dropped:
        print(f"Note: GRPOConfig of this TRL version ignores: {dropped}")
    grpo_kwargs = {k: v for k, v in grpo_kwargs.items()
                   if k in valid_params and v is not None}
    grpo_config = GRPOConfig(**grpo_kwargs)

    class UncertaintyMonitorCallback(TrainerCallback):
        def __init__(self):
            self._buffer = []

        def on_log(self, args_, state, control, logs=None, **kwargs):
            logs = logs or {}
            share = logs.get("is_uncertain_yes_share")
            if share is not None:
                self._buffer.append(share)
                running = sum(self._buffer) / len(self._buffer)
                print(
                    f"[step {state.global_step}] is_uncertain='yes' share: "
                    f"{share:.3f} (running mean {running:.3f})"
                )
            return control

    # ------------------------------------------------------------------
    # Dataset + Trainer
    # ------------------------------------------------------------------
    train_dataset = Dataset.from_list(rows)

    _temp = getattr(grpo_config, "temperature", 1.0)
    _safe_add_fused_lm_head(model, temperature=_temp)
    if hasattr(model, "get_base_model"):
        try:
            _safe_add_fused_lm_head(model.get_base_model(), temperature=_temp)
        except Exception:
            pass

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_dataset,
        callbacks=[UncertaintyMonitorCallback()],
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Done. Checkpoints saved under {args.output_dir}")


if __name__ == "__main__":
    main()
