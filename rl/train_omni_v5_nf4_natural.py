#!/usr/bin/env python3
"""Small on-policy group-relative RL run from the selected Omni 7B SFT adapter.

Uses the same multimodal processor/prompt as scripts/evaluate_sft.py. This is
policy-gradient RL with a within-group reward baseline and a frozen BF16 base;
only the existing LoRA is updated.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from reward import compute_sample_reward  # noqa: E402
from reward_metric import metric_reward  # noqa: E402


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="Check files and Python stack without loading the model")
    p.add_argument("--model", type=Path, default=ROOT.parent / "ckpt/Qwen2.5-Omni-7B")
    p.add_argument("--adapter", type=Path, default=ROOT / "output/sft/qwen25omni/qlora_sft_v5_targeted_7b/checkpoint-118")
    p.add_argument("--train-data", type=Path, default=ROOT / "Dataset/train.jsonl")
    p.add_argument("--data-root", type=Path, default=ROOT / "Dataset")
    p.add_argument("--output", type=Path, default=ROOT / "output/rl/omni_v5_metric_lora_bf16")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--generations", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--reward-scheme", choices=("legacy", "metric"), default="metric")
    p.add_argument("--sampling", choices=("shuffle", "stratified"), default="stratified")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def preflight(a):
    import torch
    import transformers
    import peft
    import av

    for path in (a.model / "config.json", a.adapter / "adapter_config.json",
                 a.adapter / "adapter_model.safetensors", a.train_data,
                 a.data_root / "test_release.jsonl"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if a.steps < 1 or a.generations < 2 or a.max_new_tokens < 16 or a.lr <= 0:
        raise ValueError("steps >= 1, generations >= 2, max-new-tokens >= 16, lr > 0 required")
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(json.dumps({"python": sys.executable, "torch": torch.__version__,
                      "transformers": transformers.__version__, "peft": peft.__version__,
                      "training_base_precision": "nf4", "av": av.__version__,
                      "cuda": torch.cuda.is_available(), "devices": torch.cuda.device_count()},
                     ensure_ascii=False), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA driver is not visible. Run on the A100 host with the GPU device exposed; nvidia-smi must work first.")
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "5,6" and torch.cuda.device_count() != 2:
        raise RuntimeError("Expected physical GPUs 5 and 6 to be visible as two CUDA devices")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This run requires a BF16-capable GPU")
    if torch.cuda.get_device_properties(0).total_memory < 40 * 1024**3:
        raise RuntimeError("Omni 7B BF16 multimodal RL requires at least 40 GiB GPU memory")


def rows_for_training(a):
    rows = [json.loads(line) for line in a.train_data.open(encoding="utf-8") if line.strip()]
    if not rows:
        raise ValueError("Empty training set")
    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate training sample_id")
    # Legacy shuffled training repeats conflict rows. Stratified scheduling
    # already balances the two types, so each source row appears once here.
    pool = ([r for r in rows for _ in range(2 if r["sample_type"] == "Conflict" else 1)]
            if a.sampling == "shuffle" else rows)
    random.Random(a.seed).shuffle(pool)
    return pool


def training_schedule(pool, steps, world, seed, sampling):
    if sampling == "shuffle":
        return [pool[i % len(pool)] for i in range(steps * world)]
    groups = {
        "natural_vt": [r for r in pool if r["sample_type"] == "Conflict" and str(r["sample_id"]).startswith("ch_sims_v2:") and r["protocol"] == "VT"],
        "natural_va": [r for r in pool if r["sample_type"] == "Conflict" and str(r["sample_id"]).startswith("ch_sims_v2:") and r["protocol"] == "VA"],
        "synthetic": [r for r in pool if r["sample_type"] == "Conflict" and not str(r["sample_id"]).startswith("ch_sims_v2:")],
        "aligned": [r for r in pool if r["sample_type"] == "Aligned"],
    }
    if not all(groups.values()):
        raise ValueError("Missing a required stratified training group")
    rng = random.Random(seed)
    for rows in groups.values():
        rng.shuffle(rows)
    return [groups[name][step % len(groups[name])] for step in range(steps)
            for name in ("natural_vt", "natural_va", "synthetic", "aligned")]



def main():
    a = arguments()
    preflight(a)
    pool = rows_for_training(a)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if a.sampling == "stratified" and world != 4 and not a.check:
        raise ValueError("Stratified training requires four GPUs")
    schedule = training_schedule(pool, a.steps, world, a.seed, a.sampling)
    from collections import Counter
    labels = Counter(row["views"]["M12"]["label"] for row in pool)
    type_frequency = ({"Conflict": 0.75, "Aligned": 0.25}
                      if a.sampling == "stratified" and world == 4
                      else {key: value / len(pool)
                            for key, value in Counter(row["sample_type"] for row in pool).items()})
    class_frequency = {key: value / len(pool) for key, value in labels.items()}
    if a.check:
        return

    import torch
    import torch.distributed as dist
    from peft import PeftModel
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, BitsAndBytesConfig
    from scripts import evaluate_sft as ev

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = f"cuda:{local_rank}"
    if rank == 0:
        print(f"Training pool: {len(pool)} rows; {a.steps} optimizer steps; {world} GPUs", flush=True)

    if rank == 0 and a.output.exists() and any(a.output.iterdir()) and not a.resume:
        raise FileExistsError(f"{a.output} is nonempty; pass --resume to continue")
    if rank == 0:
        a.output.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    state_file = a.output / "rl_state.json"
    state = json.loads(state_file.read_text()) if a.resume and state_file.is_file() else {"step": 0}
    adapter_path = a.output / "latest" if state["step"] else a.adapter
    if a.resume and state["step"] == 0:
        raise RuntimeError("No checkpoint to resume")

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    processor = Qwen2_5OmniProcessor.from_pretrained(a.model, local_files_only=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        a.model, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": device}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"))
    model.disable_talker()
    model.thinker = PeftModel.from_pretrained(model.thinker, adapter_path,
                                              is_trainable=True, local_files_only=True)
    model.eval()
    # SFT froze these modules; keep the same trainable surface for RL.
    for name, param in model.thinker.named_parameters():
        if any(part in name for part in ("visual", "audio", "projector")):
            param.requires_grad_(False)
    model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in model.thinker.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable adapter parameters")
    unexpected = [name for name, param in model.thinker.named_parameters()
                  if param.requires_grad and "lora_" not in name]
    if unexpected:
        raise RuntimeError(f"Non-LoRA parameters are trainable: {unexpected[:5]}")
    optimizer = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    if state["step"]:
        if (state.get("seed") != a.seed or state.get("sft_adapter") != str(a.adapter)
                or state.get("model") != str(a.model)
                or state.get("reward_scheme") != a.reward_scheme
                or state.get("sampling") != a.sampling
                or state.get("training_base_precision") != "nf4"):
            raise ValueError("Resume settings differ from saved run")
        opt_file = a.output / "optimizer.pt"
        if not opt_file.is_file():
            raise FileNotFoundError(opt_file)
        optimizer.load_state_dict(torch.load(opt_file, map_location="cpu", weights_only=True))
    if rank == 0:
        print(f"Trainable parameters: {sum(p.numel() for p in params):,}", flush=True)

    ev_args = argparse.Namespace(data_root=a.data_root, model_key="qwen25omni",
                                 fps=None, min_frames=None, max_frames=None,
                                 min_visual_tokens=None, max_visual_tokens=None, device=device)
    log_file = a.output / f"train_log_rank{rank}.jsonl"
    for step in range(state["step"], a.steps):
        row = schedule[step * world + rank]
        inputs, _ = ev.prepare_inputs(ev_args, processor, row)
        use_audio = bool(inputs.pop("use_audio_in_video"))
        prompt_len = inputs["input_ids"].shape[1]
        completions = []
        rewards = []
        model.thinker.eval()
        with torch.no_grad():
            for _ in range(a.generations):
                ids = model.generate(**inputs, use_audio_in_video=use_audio,
                                     thinker_max_new_tokens=a.max_new_tokens,
                                     return_audio=False, do_sample=True,
                                     temperature=0.9, top_p=0.95, use_cache=True)
                generated = ids[0, prompt_len:] if torch.equal(ids[0, :prompt_len], inputs["input_ids"][0]) else ids[0]
                completions.append(generated.detach())
                raw = processor.decode(generated, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)
                if a.reward_scheme == "metric":
                    reward = metric_reward(raw, row["views"]["M12"]["label"],
                                           row["sample_type"],
                                           class_frequency=class_frequency,
                                           type_frequency=type_frequency)["total"]
                else:
                    reward = compute_sample_reward(raw, row["views"]["M12"]["label"],
                                                   row["sample_type"])["total"]
                rewards.append(reward)
        mean_reward = sum(rewards) / len(rewards)
        advantages = [reward - mean_reward for reward in rewards]
        optimizer.zero_grad(set_to_none=True)
        updated = max(rewards) != min(rewards)
        if updated:
            model.thinker.train()
            for generated, advantage in zip(completions, advantages):
                if not generated.numel() or advantage == 0:
                    continue
                full_ids = torch.cat((inputs["input_ids"], generated.unsqueeze(0)), dim=1)
                labels = full_ids.clone()
                labels[:, :prompt_len] = -100
                forward_inputs = {k: v for k, v in inputs.items() if k != "input_ids"}
                forward_inputs["attention_mask"] = torch.ones_like(full_ids)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model.thinker(input_ids=full_ids, labels=labels,
                                         use_audio_in_video=use_audio,
                                         use_cache=False, **forward_inputs).loss
                (loss * advantage / len(completions)).backward()
        if world > 1:
            # Both ranks take the same optimizer step, including when one rank
            # sampled equal rewards and therefore contributed zero gradient.
            update_count = torch.tensor(int(updated), device=device)
            dist.all_reduce(update_count, op=dist.ReduceOp.SUM)
            for param in params:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(world)
            any_updated = bool(update_count.item())
        else:
            any_updated = updated
        if any_updated:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
        record = {"step": step + 1, "sample_id": row["sample_id"],
                  "sample_type": row["sample_type"], "rewards": rewards,
                  "updated": updated, "rank": rank}
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if rank == 0:
            print(json.dumps(record, ensure_ascii=False), flush=True)
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            # Save adapter before state so interrupted saves are not marked complete.
            if rank == 0:
                model.thinker.save_pretrained(a.output / f"checkpoint-{step + 1:04d}")
                model.thinker.save_pretrained(a.output / "latest")
                torch.save(optimizer.state_dict(), a.output / "optimizer.pt")
                state_file.write_text(json.dumps({"step": step + 1, "seed": a.seed,
                                                   "sft_adapter": str(a.adapter),
                                                   "model": str(a.model),
                                                   "reward_scheme": a.reward_scheme,
                                                   "sampling": a.sampling,
                                                   "training_base_precision": "nf4"}, indent=2))
            if world > 1:
                dist.barrier()
    if rank == 0:
        print(f"Saved RL adapter to {a.output / 'latest'}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
