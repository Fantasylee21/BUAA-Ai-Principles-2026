"""GRPO training script skeleton for the RL post-training stage.

Post-trains Qwen2.5-Omni-3B (with an existing SFT LoRA adapter) for emotion
recognition with uncertainty expression, using TRL's GRPOTrainer + PEFT.

Heavy dependencies (torch / transformers / peft / trl) are imported lazily
inside ``main()`` so that ``python rl/train_grpo.py --help`` and
``--dry-run`` work on machines without a GPU stack installed.

Usage:
    python rl/train_grpo.py --help
    python rl/train_grpo.py --dry-run --data rl/data/rl_prompts.jsonl
    python rl/train_grpo.py --model-path <base> --sft-adapter-path <adapter> \
        --data rl/data/rl_prompts.jsonl --output-dir runs/grpo_full \
        --curriculum-phase full
"""

import argparse
import json
import os
import sys

# Make sibling modules (reward.py) importable regardless of the cwd the
# script is launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reward import (  # noqa: E402  (pure-python module, safe at top level)
    build_reward_funcs,
    reward_weights_for_phase,
)

# Frame sampling is configured on the processor's video_processor right
# after loading (see main()): fps=2.0 + do_sample_frames=True, matching the
# SFT evaluation conditions in scripts/evaluate_sft.py. Bounding the visual
# token count is critical for GRPO where every prompt is expanded into
# `num_generations` completions — with the shipped processor defaults (no
# sampling at all) a single prompt measured ~77k tokens on the server.
MAX_FRAMES = 8  # kept for reference/ablation; not wired in (fps sampling is)


def freeze_by_name_substring(model, substrings):
    """Freeze every parameter whose qualified name contains any substring.

    For Qwen2.5-Omni the frozen parts are the vision tower, the audio tower
    and their projectors — only the language model's LoRA adapter should
    train. To discover the exact module names on your checkpoint, run:

        python -c "from transformers import AutoModelForImageTextToText; \\
            m = AutoModelForImageTextToText.from_pretrained('<path>'); \\
            [print(n) for n, _ in m.named_parameters()]" | sort -u | less

    and adjust the substrings accordingly (typical matches contain
    "visual", "audio" / "audio_tower", and "projector" / "multi_modal_projector").
    Returns the number of frozen parameters.
    """
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
                        help="Path to the existing SFT LoRA adapter to continue "
                             "training from.")
    parser.add_argument("--data", default="rl/data/rl_prompts.jsonl",
                        help="RL prompt JSONL produced by prepare_rl_data.py.")
    parser.add_argument("--output-dir", default="runs/grpo",
                        help="Checkpoint / log output directory.")
    parser.add_argument("--curriculum-phase", choices=["warmup", "full"],
                        default="warmup",
                        help="warmup = format+emotion rewards only (~first 30%% "
                             "of training); full = all four components. Switch "
                             "phases by restarting with this flag and resuming "
                             "from the previous checkpoint.")
    parser.add_argument("--use-vllm", action="store_true",
                        help="Use vLLM colocate generation for rollouts "
                             "(~10x faster than HF generate) — requires the "
                             "vllm package. Default off: HF generate works "
                             "everywhere, use this for the full run.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data, print one batch-shaped sample, and exit "
                             "before model/trainer initialization.")
    # Common hyperparameters (GRPOConfig defaults below carry the rationale).
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="LoRA learning rate (default: 1e-5).")
    parser.add_argument("--num-generations", type=int, default=8,
                        help="Completions sampled per prompt (default: 8).")
    parser.add_argument("--beta", type=float, default=0.01,
                        help="KL coefficient to the SFT reference policy "
                             "(default: 0.01).")
    parser.add_argument("--max-completion-length", type=int, default=128,
                        help="Max new tokens per completion (default: 128; the "
                             "output contract is tiny).")
    parser.add_argument("--max-prompt-length", type=int, default=None,
                        help="Optional prompt truncation length (tokens).")
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_prompt_dataset(path):
    """Load the RL prompt JSONL into a list of dicts.

    Each row keeps its extra columns (gt_label, sample_type) — TRL forwards
    them to the reward functions via **kwargs.
    """
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
        # Print one batch-shaped sample and exit before touching torch/trl.
        sample = rows[0]
        print("--- dry run: sample row ---")
        print(f"sample_id   : {sample['sample_id']}")
        print(f"sample_type : {sample['sample_type']}")
        print(f"gt_label    : {sample['gt_label']}")
        print(f"prompt roles: {[m['role'] for m in sample['prompt']]}")
        content_types = [c['type'] for c in sample['prompt'][0]['content']]
        print(f"content types: {content_types}")
        text_part = next(
            c['text'] for c in sample['prompt'][0]['content'] if c['type'] == 'text'
        )
        print(f"instruction (first 200 chars): {text_part[:200]!r}")
        print(f"curriculum_phase: {args.curriculum_phase} -> reward funcs: "
              f"{[f.__name__ for f in build_reward_funcs(args.curriculum_phase)]}")
        print("dry run OK; exiting before trainer init.")
        return

    # ------------------------------------------------------------------
    # Lazy heavy imports: torch / transformers / peft / trl may be absent
    # on dev machines (e.g. this Mac); fail with a friendly message.
    # ------------------------------------------------------------------
    try:
        import torch  # noqa: F401
        from datasets import Dataset
        from peft import PeftModel
        from transformers import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
            TrainerCallback,
        )
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependency: "
            f"{exc.name or exc}. The GRPO stage requires a GPU environment with "
            "torch, transformers, datasets, peft and trl installed "
            "(e.g. `pip install trl peft datasets transformers accelerate`). "
            "Use --dry-run or --help to inspect this script without them."
        )

    # ------------------------------------------------------------------
    # PyAV compatibility shim.
    # torchvision 0.21's read_video() calls av.open(..., metadata_errors=
    # "ignore"), but PyAV >= 15 removed that kwarg — and the AutoDL pip
    # mirror ships no prebuilt av<15 wheel for py3.12, so downgrading is
    # not always possible. When the installed PyAV is >= 15, wrap av.open
    # to drop the removed kwargs; torchvision looks the attribute up on the
    # module at call time, so the patch takes effect. Dormant on av < 15.
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
        print(
            f"Note: PyAV {_av.__version__} >= 15 removed the metadata_errors/"
            "metadata_encoding kwargs; patched av.open so torchvision video "
            "decoding works. (Install av<15 from a mirror with wheels to make "
            "this shim unnecessary.)"
        )

    # ------------------------------------------------------------------
    # Model + processor
    # ------------------------------------------------------------------
    # Qwen2.5-Omni is a composite model: an outer container holding the
    # "thinker" (text/vision/audio understanding) and the "talker" (speech
    # synthesis, unused here). AutoModelForImageTextToText cannot resolve the
    # composite Qwen2_5OmniConfig, so load via the dedicated classes,
    # following the team's own scripts/evaluate_sft.py: disable the talker
    # and train the thinker submodel (which is also what LLaMAFactory used
    # as the PEFT base during SFT). If this TRL path proves fragile,
    # ms-swift and NeMo-RL ship dedicated Qwen2.5-Omni GRPO recipes and are
    # the planned fallback frameworks.
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)

    # --- frame sampling (CRITICAL) -----------------------------------------
    # With the processor's shipped defaults (do_sample_frames=False, fps=None)
    # the video processor decodes EVERY frame of each video — measured on the
    # server: a single prompt ballooned to ~77k tokens. Configure instance
    # defaults so preprocess() samples frames exactly like the SFT evaluation
    # did (scripts/evaluate_sft.py: fps=2.0, do_sample_frames=True; the
    # processor's own min_frames=4 / max_frames=768 bounds then apply).
    _vp = processor.video_processor
    _vp.do_sample_frames = True
    _vp.fps = 2.0
    print(
        "Video frame sampling: "
        f"do_sample_frames={_vp.do_sample_frames}, fps={_vp.fps}, "
        f"min_frames={getattr(_vp, 'min_frames', None)}, "
        f"max_frames={getattr(_vp, 'max_frames', None)}"
    )

    # --- Arrow None-key cleaner (CRITICAL for this model's chat template) ---
    # Dataset.from_list stores the "prompt" column in Arrow, which UNIFIES the
    # keys of the content-part dicts across a column: every text part gains a
    # phantom "video": None and every video part gains "text": None (verified
    # locally). This model's LLaMAFactory-exported chat template decides each
    # block's type by key PRESENCE ('video' in content), so a polluted text
    # part renders an EXTRA <|vision_bos|><|VIDEO|><|vision_eos|> — the batch
    # then has twice as many <|VIDEO|> tokens as decoded videos, and the
    # processor dies with StopIteration in replace_multimodal_special_tokens.
    # Fix: strip None-valued keys from content parts before the template and
    # the video collector see them. The StopIteration probe is kept as a
    # safety net: if it ever fires again, it dumps the batch composition.
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
        try:
            return _orig_apply_chat_template(cleaned_conversation, **kwargs)
        except StopIteration:
            convs = cleaned_conversation if is_batched else [cleaned_conversation]
            print(f"[probe] StopIteration STILL occurred: batch of {len(convs)} conversation(s):")
            for i, conv in enumerate(convs[:8]):
                parts = [
                    part
                    for m in conv
                    for part in (m.get("content") or [])
                    if isinstance(part, dict)
                ]
                print(f"[probe] conv {i}: parts={[p.get('type') for p in parts]} "
                      f"keys={[sorted(p.keys()) for p in parts]}")
            raise

    processor.apply_chat_template = _cleaning_apply_chat_template

    omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype="bfloat16",
        attn_implementation="sdpa",  # flash_attention_2 is faster but the
                                     # flash-attn wheel is not installed here;
                                     # sdpa is correct on A800, just slower
    )
    omni.disable_talker()  # text-only task: drop the speech-synthesis stack

    if args.sft_adapter_path:
        # Continue training the existing SFT LoRA adapter. It is mounted on
        # the THINKER (the LLaMAFactory PEFT base), not the outer container —
        # mounting on the container silently leaves every tensor unloaded
        # (see scripts/evaluate_sft.py). Do NOT merge the adapter: beta>0
        # keeps a frozen reference copy for the KL term via PEFT.
        omni.thinker = PeftModel.from_pretrained(
            omni.thinker, args.sft_adapter_path, is_trainable=True
        )
    else:
        # No adapter: the HF team repo ships a MERGED SFT model, so start
        # from it directly and add a fresh LoRA on the thinker, matching the
        # SFT-stage hyperparameters (r=16, alpha=32, dropout=0.05).
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
    model = omni.thinker  # GRPO trains the thinker, exactly like SFT did

    # Freeze vision tower / audio tower / projectors: GRPO should only move
    # the language-model LoRA weights; the perception stacks were already
    # aligned during SFT and moving them wastes memory and risks drift.
    n_frozen = freeze_by_name_substring(model, ["visual", "audio", "projector"])
    print(f"Froze {n_frozen} parameters (vision/audio/projector towers).")

    # ------------------------------------------------------------------
    # Reward wiring
    # ------------------------------------------------------------------
    reward_funcs = build_reward_funcs(args.curriculum_phase)
    reward_weights = reward_weights_for_phase(args.curriculum_phase)
    print(f"Curriculum phase: {args.curriculum_phase}")
    print(f"Reward funcs : {[f.__name__ for f in reward_funcs]}")
    print(f"Reward weights: {reward_weights}")

    # ------------------------------------------------------------------
    # GRPO configuration (defaults carry inline rationale)
    # ------------------------------------------------------------------
    grpo_kwargs = dict(
        output_dir=args.output_dir,
        # --- generation -------------------------------------------------
        temperature=1.0000001,  # TRL issue #5783: temperature exactly 1.0 with
                                # Qwen models silently falls back to greedy
                                # decoding; 1.0000001 keeps true sampling while
                                # staying numerically at temperature 1.
        top_p=0.99,             # light tail trimming for sampling stability
        num_generations=args.num_generations,  # 8 completions per prompt group
        beta=args.beta,         # 0.01: KL to the frozen SFT reference policy —
                                # the main guard against reward hacking (e.g.
                                # always answering is_uncertain=yes)
        max_completion_length=args.max_completion_length,  # 128: the JSON
                                # contract is tiny; anything longer is waste
        max_prompt_length=args.max_prompt_length,
        # --- vLLM -------------------------------------------------------
        use_vllm=args.use_vllm,  # vLLM generation is ~an order of magnitude
                                 # faster than HF generate for GRPO rollouts;
                                 # off by default until vllm is installed
        vllm_mode="colocate",   # single-GPU setup: vLLM shares the training GPU
        vllm_gpu_memory_utilization=0.3,  # leave headroom for training state
        # --- batching ---------------------------------------------------
        per_device_train_batch_size=1,   # video tokens are huge; 1 per device
        gradient_accumulation_steps=16,  # effective batch 16 prompts x 8 gens
        # --- precision / memory -----------------------------------------
        bf16=True,
        gradient_checkpointing=True,
        # --- optimization / logging --------------------------------------
        learning_rate=args.learning_rate,  # 1e-5 is a safe LoRA GRPO rate
        num_train_epochs=args.num_train_epochs,
        logging_steps=5,
        save_steps=100,
        seed=args.seed,
        reward_weights=reward_weights,
        report_to=[],           # add "wandb" / "tensorboard" as desired
    )
    # TRL's GRPOConfig signature drifts between versions (e.g. TRL 1.16
    # dropped max_prompt_length). Filter to what the installed version
    # actually accepts instead of pinning the code to one TRL release.
    import inspect
    valid_params = set(inspect.signature(GRPOConfig.__init__).parameters)
    dropped = sorted(k for k, v in grpo_kwargs.items() if k not in valid_params)
    if dropped:
        print(f"Note: GRPOConfig of this TRL version ignores: {dropped}")
    grpo_kwargs = {k: v for k, v in grpo_kwargs.items()
                   if k in valid_params and v is not None}
    grpo_config = GRPOConfig(**grpo_kwargs)

    # ------------------------------------------------------------------
    # Monitoring callback
    # ------------------------------------------------------------------
    class UncertaintyMonitorCallback(TrainerCallback):
        """Logs the running mean share of is_uncertain='yes' predictions.

        TRL already logs every reward component under
        ``rewards/<fn_name>/mean`` — watching ``rewards/reward_uncertainty/mean``
        is the primary reward-hacking sensor: if the policy collapses to
        always answering is_uncertain=yes (to farm conflict reward), the
        uncertainty component mean and the is_uncertain share will spike
        together. This callback prints the is_uncertain share whenever the
        trainer's log payload carries it (e.g. from a custom metric);
        otherwise rely on the TRL reward-component logs above.
        """

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
    # Dataset + trainer
    # ------------------------------------------------------------------
    train_dataset = Dataset.from_list(rows)

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
