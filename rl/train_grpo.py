#!/usr/bin/env python3
"""Single-GPU Omni Dr.GRPO runner with explicit video/audio probability paths.

Run as python -m rl.train_grpo. Heavy dependencies are imported only after
--dry-run. See rl/README.md for the distinction from generic TRL GRPOTrainer.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "rl"

from .experiment import (ROOT, BalancedSampler, artifact_hash, check_disjoint,
                         fingerprint, group_advantages, load_records, read_config,
                         validate_config, write_json)
from .metrics import evaluation_report, prediction_record, promotion_decision
from .reward import FrozenF1Reference, RewardEngine
from .state import (adapter_digest, capture_rng, code_identity, restore_rng,
                    save_checkpoint, software_versions, validate_checkpoint,
                    validate_f1_cache, validate_run_directory)

_ACTIVE_RUN_DIR = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rl/correctness_v1.json")
    parser.add_argument("--mode", choices=("train", "smoke", "evaluate", "cache"), default="train")
    parser.add_argument("--data-root", type=Path, default=ROOT / "Dataset")
    parser.add_argument("--data", type=Path, help="Original train.jsonl or exact duplicate legacy prompts")
    parser.add_argument("--val-file", type=Path)
    parser.add_argument("--test-file", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sft-adapter-path", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, help="Trained adapter for evaluate only")
    parser.add_argument("--run-dir", "--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--f1-cache", type=Path)
    parser.add_argument("--gpu", help="Physical GPU index/UUID or auto; changes execution only")
    parser.add_argument("--micro-batch-size", type=int, choices=(1, 2, 4))
    parser.add_argument("--max-runtime-hours", type=float)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--smoke-updates", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def resolved(path):
    return (path if path.is_absolute() else ROOT / path).resolve()


def append_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def evaluate(policy, records, data_root, destination, *, adapter="policy", labelled=True):
    """One common greedy inference path for SFT baseline and RL candidates."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial.jsonl")
    if partial.exists():
        # Previous incomplete evaluation is kept as a diagnostic; do not append to it.
        partial = destination.with_name(destination.stem + f".{time.time_ns()}.partial.jsonl")
    rng = capture_rng(policy.torch)
    predictions = []
    try:
        for index, row in enumerate(records, 1):
            inputs = policy.prepare(row, data_root)
            tokens, completion = policy.generate(inputs, adapter=adapter)
            record = prediction_record(row, completion)
            record["completion_tokens"] = tokens.numel()
            append_jsonl(partial, record)
            predictions.append(record)
            del inputs, tokens
            if index % 25 == 0:
                print(f"evaluation {index}/{len(records)}", flush=True)
    finally:
        restore_rng(policy.torch, rng)
    partial.replace(destination)
    if labelled:
        metrics = evaluation_report(predictions)
        write_json(destination.with_suffix(".metrics.json"), metrics)
        return metrics
    return predictions


def smoke_media_checks(policy, records, data_root):
    """Check real media influence and identical initial reference logits."""
    torch = policy.torch
    for protocol in ("VT", "VA"):
        rows = [row for row in records if row["protocol"] == protocol]
        if not rows:
            raise ValueError(f"Smoke requires {protocol} examples")
        # Probe durations once to include the longest training input of each protocol.
        import subprocess
        durations = {}
        for row in rows:
            video = row["video"]
            if video not in durations:
                value = subprocess.check_output([
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", str(Path(data_root) / video)], text=True)
                durations[video] = float(value.strip())
        row = max(rows, key=lambda r: durations[r["video"]])
        prefix = policy.prepare(row, data_root)
        tokens, _ = policy.generate(prefix)
        from .omni_runtime import batch_for_completions
        inputs, ids, _ = batch_for_completions(torch, prefix, [tokens], policy.pad_id)
        policy.model.eval()
        with torch.no_grad():
            with policy.adapter("policy"):
                initial = policy.log_probabilities(inputs, ids)
            with policy.adapter("reference"):
                reference = policy.log_probabilities(inputs, ids)
            if not torch.allclose(initial, reference, atol=2e-4, rtol=1e-4):
                raise RuntimeError("Initial policy/reference token probabilities differ")
            for key in (("pixel_values_videos", "input_features") if protocol == "VA" else ("pixel_values_videos",)):
                altered = dict(inputs)
                altered[key] = torch.zeros_like(inputs[key])
                changed = policy.log_probabilities(altered, ids)
                if torch.allclose(initial, changed, atol=1e-8, rtol=0):
                    raise RuntimeError(f"No measurable {key} influence in {protocol} probability path")
        print(f"media path verified: {protocol}, duration={durations[row['video']]:.2f}s", flush=True)


def main(argv=None):
    global _ACTIVE_RUN_DIR
    _ACTIVE_RUN_DIR = None
    args = parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, resolved(value))
    settings, runtime = read_config(args.config)
    for key in ("gpu", "micro_batch_size", "max_runtime_hours"):
        value = getattr(args, key)
        if value is not None:
            runtime[key] = value
    validate_config(settings, runtime)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This runner is single-GPU; launch each independent experiment separately")
    if args.policy_path and args.mode != "evaluate":
        raise ValueError("--policy-path is for evaluate only; training starts from SFT or an explicit resume")
    if args.resume_from_checkpoint and args.mode not in ("train", "smoke"):
        raise ValueError("Checkpoint resume is only for train/smoke")
    if args.split == "test" and args.mode != "evaluate":
        raise ValueError("Test inputs cannot be used for training, reward caching or model selection")
    if args.smoke_updates < 1:
        raise ValueError("smoke-updates must be positive")
    train, duplicates, val = [], 0, []
    if args.mode in ("train", "smoke", "cache"):
        train, duplicates = load_records(args.data or args.data_root / "train.jsonl")
        BalancedSampler(train, settings["seed"], settings["prompts_per_update"])
    if args.mode in ("train", "smoke") or (args.mode == "evaluate" and args.split == "val"):
        val, val_duplicates = load_records(args.val_file or args.data_root / "val.jsonl")
        if val_duplicates:
            raise ValueError("Official validation must not contain repeated IDs")
        if train:
            check_disjoint(train, val)
        if len(val) != 666:
            raise ValueError("Full official validation must contain exactly 666 unique samples")
    test = []
    if args.mode == "evaluate" and args.split == "test":
        test, repeated = load_records(args.test_file or args.data_root / "test_release.jsonl", require_labels=False)
        if repeated or len(test) != 734:
            raise ValueError("Official test input must contain exactly 734 unique samples")
    if args.dry_run:
        print(json.dumps({"experiment": settings, "runtime": runtime,
                          "unique_train": len(train), "ignored_exact_duplicates": duplicates,
                          "validation": len(val), "test": len(test),
                          "completions_per_update": settings["prompts_per_update"] * settings["num_generations"],
                          "software": software_versions()}, ensure_ascii=False, indent=2))
        return
    if args.run_dir.exists() and any(args.run_dir.iterdir()) and not args.resume_from_checkpoint:
        raise ValueError("Run directory is nonempty: use a new run-dir or an explicit checkpoint resume")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    from .omni_runtime import OmniPolicy, select_gpu
    gpu = select_gpu(runtime["gpu"], runtime["min_free_memory_gib"])
    code = code_identity()
    base_hash = artifact_hash(args.model_path)
    sft_hash = artifact_hash(args.sft_adapter_path, adapter=True)
    input_keys = ("video_fps", "video_min_frames", "video_max_frames", "video_pixels", "audio_sampling_rate",
                  "max_prompt_tokens", "max_completion_length", "attn_implementation")
    baseline_identity = {"train": fingerprint(train), "base": base_hash, "sft": sft_hash,
                         "inputs": {k: settings[k] for k in input_keys}, "source": code["source_hash"]}
    f1_reference, cache_hash = None, None
    if settings["reward_version"] == "f1_delta_v1" and args.mode in ("train", "smoke"):
        if args.f1_cache is None:
            raise ValueError("f1_delta_v1 needs --f1-cache generated by --mode cache")
        cache = json.loads(args.f1_cache.read_text(encoding="utf-8"))
        validate_f1_cache(cache, baseline_identity)
        f1_reference = FrozenF1Reference(train, cache["predictions"])
        cache_hash = fingerprint(cache)
    import numpy as np
    import torch
    from transformers import get_cosine_schedule_with_warmup
    from trl import GRPOConfig
    software = software_versions()
    identity = {"schema": "omni_dr_grpo_v1", "mode": args.mode, "experiment": settings,
                "baseline": baseline_identity, "validation": fingerprint(val),
                "evaluation_policy": (artifact_hash(args.policy_path, adapter=True)
                                      if args.policy_path else sft_hash) if args.mode == "evaluate" else None,
                "f1_cache": cache_hash, "software": {k: v for k, v in software.items()
                                                     if k not in ("platform", "python")}}
    marker = validate_checkpoint(args.resume_from_checkpoint, identity) if args.resume_from_checkpoint else None
    validate_run_directory(args.run_dir, identity, marker)
    _ACTIVE_RUN_DIR = args.run_dir
    write_json(args.run_dir / "status.json", {"status": "initializing", "mode": args.mode,
               "completed_steps": marker["completed_steps"] if marker else 0})
    policy_path = args.resume_from_checkpoint / "adapter/policy" if marker else args.policy_path
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed_all(settings["seed"])
    manifest = {"identity": identity, "runtime": runtime, "gpu": gpu,
                "software": software, "code": code, "ignored_exact_duplicates": duplicates,
                "inference_adapter": str(args.policy_path or args.sft_adapter_path),
                "note": "Dedicated one-iteration Dr.GRPO loop with TRL GRPOConfig; not generic GRPOTrainer"}
    write_json(args.run_dir / f"manifest-{time.time_ns()}.json", manifest)
    policy = OmniPolicy(args.model_path, args.sft_adapter_path, settings, policy_path=policy_path,
                        training=args.mode in ("train", "smoke"))
    if args.mode == "evaluate":
        output = args.run_dir / (args.split + "_predictions.jsonl")
        evaluate(policy, val if args.split == "val" else test, args.data_root,
                 output, labelled=args.split == "val")
        if args.split == "test":
            import subprocess
            subprocess.run([sys.executable, str(ROOT / "rl/make_submission.py"), "--strict",
                            "--input", str(output), "--test-file",
                            str(args.test_file or args.data_root / "test_release.jsonl"),
                            "--output", str(args.run_dir / "submission.csv")], check=True)
        write_json(args.run_dir / "status.json", {"status": "complete", "mode": args.mode})
        return
    if args.mode == "cache":
        output = args.run_dir / "train_predictions.jsonl"
        evaluate(policy, train, args.data_root, output, adapter="reference")
        predictions = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        write_json(args.run_dir / "f1_cache.json", {"schema": "train_f1_cache_v1", "split": "train",
                   "baseline_identity": baseline_identity,
                   "predictions": {r["sample_id"]: r["pred_emotion"] for r in predictions}})
        write_json(args.run_dir / "status.json", {"status": "complete", "mode": args.mode})
        return
    # Use TRL's public configuration contract. No silent unsupported-option filtering.
    grpo_config = GRPOConfig(
        output_dir=str(args.run_dir), num_generations=settings["num_generations"],
        per_device_train_batch_size=runtime["micro_batch_size"],
        gradient_accumulation_steps=settings["prompts_per_update"] * settings["num_generations"] // runtime["micro_batch_size"],
        max_steps=settings["max_steps"], max_completion_length=settings["max_completion_length"],
        learning_rate=settings["learning_rate"], beta=settings["beta"],
        scale_rewards="none", loss_type="dr_grpo", num_iterations=1,
        temperature=1.0, top_p=1.0, top_k=0, bf16=True, use_vllm=False,
        gradient_checkpointing=True, report_to="none", seed=settings["seed"])
    if grpo_config.num_iterations != 1:
        raise ValueError("Only fresh, one-iteration on-policy updates are supported")
    engine = RewardEngine.from_records(train, version=settings["reward_version"], f1_reference=f1_reference)
    sampler = BalancedSampler(train, settings["seed"], settings["prompts_per_update"])
    parameters = [p for p in policy.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"], weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, settings["warmup_steps"], settings["max_steps"])
    completed = 0
    if marker:
        # Only load checkpoints that passed the file manifest verification above.
        state = torch.load(args.resume_from_checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
        completed = state["completed_steps"]
        if completed != marker["completed_steps"]:
            raise ValueError("Checkpoint marker and optimizer state disagree")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(torch, state["rng"])
        if adapter_digest(policy, "reference") != marker["reference_digest"]:
            raise ValueError("SFT reference differs after reload")
        if adapter_digest(policy, "policy") != marker["policy_digest"]:
            raise ValueError("Policy differs after reload")
    elif adapter_digest(policy, "reference") != adapter_digest(policy, "policy"):
        raise RuntimeError("Training did not initialize from the supplied SFT adapter")
    reference_digest = adapter_digest(policy, "reference")
    baseline_path = args.run_dir / "baseline.metrics.json"
    baseline = None
    if args.mode == "smoke" and not marker:
        smoke_media_checks(policy, train, args.data_root)
    if args.mode == "train":
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        else:
            baseline = evaluate(policy, val, args.data_root, args.run_dir / "baseline.jsonl", adapter="reference")
    if not marker:
        save_checkpoint(args.run_dir, policy, optimizer, scheduler, completed, identity)
    stop_requested = []
    def stop_at_boundary(signum, frame):
        stop_requested.append(signum)
        print("Stop requested; finishing the current optimizer update before saving.", flush=True)
    previous_handlers = {sig: signal.signal(sig, stop_at_boundary) for sig in (signal.SIGINT, signal.SIGTERM)}
    started = time.monotonic()
    target = min(settings["max_steps"], args.smoke_updates) if args.mode == "smoke" else settings["max_steps"]

    def evaluate_checkpoint(step):
        destination = args.run_dir / f"evaluation/step-{step:06d}.jsonl"
        metrics_path = destination.with_suffix(".metrics.json")
        metrics = (json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists()
                   else evaluate(policy, val, args.data_root, destination))
        decision = promotion_decision(baseline, metrics)
        write_json(destination.with_suffix(".selection.json"), decision)
        if decision["promote"]:
            best_path = args.run_dir / "best_checkpoint.json"
            best = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else None
            if best is None or metrics["score"] > best["metrics"]["score"]:
                write_json(best_path, {"checkpoint": f"checkpoint-{step:06d}",
                                      "adapter": f"checkpoint-{step:06d}/adapter/policy", "metrics": metrics})

    try:
        if completed and args.mode == "train" and (completed % settings["eval_every"] == 0 or completed >= target):
            evaluate_checkpoint(completed)
        for step in range(completed, target):
            update_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            totals = {"loss": 0.0, "kl": 0.0, "reward": 0.0, "valid_json": 0.0,
                      "emotion": 0.0, "uncertainty": 0.0, "question_penalty": 0.0,
                      "zero_advantage_groups": 0, "is_uncertain_yes_share": 0.0}
            prompt_count, generations = settings["prompts_per_update"], settings["num_generations"]
            for row in sampler.batch(step):
                prefix = policy.prepare(row, args.data_root)
                completions, scores, raw = [], [], []
                for generation in range(generations):
                    tokens, text = policy.generate(prefix, sample=True)
                    score = engine.score(text, row["gt_label"], row["sample_type"], row["sample_id"])
                    completions.append(tokens)
                    scores.append(score)
                    raw.append(text)
                    append_jsonl(args.run_dir / "rollouts.jsonl", {
                        "update": step+1, "sample_id": row["sample_id"], "generation": generation,
                        "sample_type": row["sample_type"], "completion": text, "reward": score,
                        "completion_tokens": tokens.numel(),
                        "reached_token_limit": tokens.numel() >= settings["max_completion_length"]})
                advantages = group_advantages([s["total"] for s in scores])
                totals["zero_advantage_groups"] += int(max(advantages) - min(advantages) < 1e-12)
                losses = policy.backward_group(prefix, completions, advantages, runtime["micro_batch_size"])
                totals["loss"] += losses["loss"]
                totals["kl"] += losses["kl"] / prompt_count
                for score, text in zip(scores, raw):
                    totals["reward"] += score["total"] / (prompt_count * generations)
                    for key in ("valid_json", "emotion", "uncertainty", "question_penalty"):
                        totals[key] += score[key] / (prompt_count * generations)
                    parsed = prediction_record(row, text)
                    totals["is_uncertain_yes_share"] += (parsed["is_uncertain"] == "yes") / (prompt_count * generations)
                del prefix, completions, scores
            norm = torch.nn.utils.clip_grad_norm_(parameters, settings["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            completed = step + 1
            record = {"step": completed, **totals, "grad_norm": float(norm),
                      "fraction_zero_advantage_groups": totals["zero_advantage_groups"] / prompt_count,
                      "learning_rate": scheduler.get_last_lr()[0],
                      "seconds": time.monotonic() - update_started,
                      "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                      "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
            append_jsonl(args.run_dir / "training.jsonl", record)
            write_json(args.run_dir / "status.json", {"status": "running", **record})
            print(json.dumps(record), flush=True)
            deadline = runtime["max_runtime_hours"]
            timed_out = deadline is not None and time.monotonic() - started >= deadline * 3600
            save = completed % settings["save_every"] == 0 or completed == target or stop_requested or timed_out
            evaluate_now = args.mode == "train" and (completed % settings["eval_every"] == 0 or completed == target)
            if save or evaluate_now:
                if adapter_digest(policy, "reference") != reference_digest:
                    raise RuntimeError("Frozen SFT reference changed")
                save_checkpoint(args.run_dir, policy, optimizer, scheduler, completed, identity)
            if evaluate_now and not (stop_requested or timed_out):
                evaluate_checkpoint(completed)
            if stop_requested or timed_out:
                break
        if args.mode == "smoke" and completed >= target:
            if adapter_digest(policy, "policy") == reference_digest:
                raise RuntimeError("Smoke produced no policy update; inspect reward diversity and gradients")
        evaluation_pending = args.mode == "train" and completed >= target and not (
            args.run_dir / f"evaluation/step-{completed:06d}.metrics.json").exists()
        write_json(args.run_dir / "status.json", {"status": "complete" if completed >= target and not evaluation_pending else "stopped",
                   "completed_steps": completed, "target_steps": target,
                   "note": "No eligible checkpoint means SFT remains the selected model"})
    except BaseException as exc:
        write_json(args.run_dir / "status.json", {"status": "failed", "completed_steps": completed,
                   "error": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def entrypoint():
    try:
        main()
    except Exception as exc:
        if _ACTIVE_RUN_DIR is not None:
            path = _ACTIVE_RUN_DIR / "status.json"
            try:
                status = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                status = {}
            write_json(path, {**status, "status": "failed", "error": str(exc),
                              "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    entrypoint()
