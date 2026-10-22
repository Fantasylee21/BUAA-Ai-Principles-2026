#!/usr/bin/env python3
"""Read-only Xshell progress for the bounded grounded experiment."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time


def read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def configuration_lines(settings):
    return [
        f"AdamW | base LR {settings['learning_rate']:g} | KL {settings['beta']:g} | aux {settings['aux_weight']:g} | REBEL eta {settings['eta']:g}",
        f"Batch {settings['batch_size']} | max {settings['steps']} updates/seed | warmup {settings['warmup_steps']} + cosine | old/eval/pools every {settings['block_steps']}",
        "Batch: natural Conflict VT3/VA3 + synthetic Conflict VT1/VA1 + Aligned VT4/VA4",
        "Train language LoRA only | emotion/reference frozen | seed 42 -> qualified seed 43",
    ]


def render_parameters(run, gpus=("1", "2")):
    context = read(run / "context.json")
    if not context:
        return "Training parameters: waiting for the immutable run context."
    settings = context["settings"]
    pipeline = read(run / "pipeline.status.json")
    train = read(run / "train.status.json")
    current_lr = f"{train['learning_rate']:g}" if "learning_rate" in train else "-- (no optimizer update yet)"
    cache_progress = []
    for shard, gpu in enumerate(gpus):
        status = read(run / f"cache-{shard}.status.json")
        if status:
            cache_progress.append(f"GPU {gpu} cache {status.get('completed', 0)}/{status.get('total', '--')}")
    phase = "Phase: "+pipeline.get("phase", "initializing")
    if cache_progress:
        phase += " | "+"; ".join(cache_progress)
    return "\n".join([
        "Omni v4 | ACTUAL RUN PARAMETERS | grounded uncertainty REBEL v1",
        phase,
        f"Training seed {train.get('seed', settings['seed'])} | update {train.get('step', 0)}/{settings['steps']} | current LR {current_lr}",
        "",
        *configuration_lines(settings),
        f"Weight decay 0 | gradient clip {settings['max_grad_norm']:g} | dropout 0 | no parameter sweeps",
        "Loss = 0.6 mean(Conflict) + 0.4 mean(Aligned) + 0.1 mean(views) + 0.02 KL_U",
        "Reward: correct=1 / wrong=0; regress change in action log-odds against frozen old",
        "Main action yes/no | auxiliary positive/negative/neutral | native value-token logits",
        "Views: silent video M1; VT text M2 / VA audio M2; unclear auxiliary targets skipped",
        "",
        "Per protocol: natural C = 1 uniform + 2 missed; Aligned = 2 uniform + 2 false positives",
        "Each batch includes an opposite-label VT/VA pair; hard pools use TRAIN only",
        f"Media: {settings['video_fps']:g} fps, {settings['video_min_frames']}-{settings['video_max_frames']} frames, {settings['video_pixels']} pixels/frame, audio {settings['audio_sampling_rate']} Hz",
        f"Input max {settings['max_prompt_tokens']} tokens | question max {settings['max_completion_length']} tokens | NF4/BF16",
        f"GPUs {gpus[0]}/{gpus[1]} parallel cache; then {gpus[0]} train / {gpus[1]} validate + pools | 32 GiB allocator cap each",
        "",
        "Before training: exact 666-row zero-update check + disposable 32-example / 20-update learning check",
        "Learning check: both hard-group median log-odds gains >=0.25; main loss drop >=10%; exact resume",
        "Primary gate: local gain >=0.015; natural >=10/25; VA >=4/10; aligned FP <=28/592",
        "Seed 43 only after primary gate; replication gain >=0.005, natural >=8/25, FP <=28",
        "From step 80: stop if no evaluated candidate has gain >=0.003, natural >=8/25, FP <=28",
        "All candidates require frozen emotions + valid outputs; CSV only after both seeds pass",
        f"Budget {settings['budget_hours']:g} h total; no new training after {settings['training_cutoff_hours']:g} h; remaining {max(0,context['deadline']-time.time())/3600:.2f} h",
        "",
        "Ctrl+B then 0: live progress | 2: training logs | 8: parameters | D: detach safely",
    ])


def render(run, gpus=("1", "2")):
    pipeline = read(run / "pipeline.status.json")
    lines = ["Omni v4 | frozen emotion + grounded REBEL RL | target Public 0.92",
             datetime.now(timezone(timedelta(hours=8))).strftime("Beijing %Y-%m-%d %H:%M:%S"),
             "Phase: "+pipeline.get("phase", "initializing data/model checksums"),
             "Two allocated GPUs | 32 GiB allocator limit each | max 24 hours", ""]
    context = read(run / "context.json")
    if context:
        lines.extend(configuration_lines(context["settings"]))
        lines.append(f"Wall-clock budget remaining: {max(0,context['deadline']-time.time())/3600:.2f} h")
    for role, name in (("probe", "Numerical check"), ("cache-0", f"Cache GPU {gpus[0]}"), ("cache-1", f"Cache GPU {gpus[1]}"), ("smoke", "Learning / exact resume check")):
        status = read(run / f"{role}.status.json")
        if status:
            lines.append(f"{name}: {status.get('phase','--')} | {status.get('completed',status.get('step','--'))}/{status.get('total','--')}")
    zero = read(run / "zero-check.json")
    if zero:
        lines.append(f"Zero-update identical to frozen v4: {zero['passed']} ({len(zero['mismatches'])} mismatches)")
    train = read(run / "train.status.json")
    if train:
        lines += ["", f"TRAIN seed {train.get('seed','--')}: {train.get('phase','--')} | update {train.get('step','--')}/200"]
        lines.append("Loss: {} | expected correctness: {} | KL: {}".format(*[
            f"{train[k]:.6f}" if k in train else "--" for k in ("loss", "expected_reward", "kl")]))
        lines.append(f"Main correct: {train.get('main_correct','--')}/16 | auxiliary: {train.get('aux_correct','--')}/{train.get('aux_count','--')}")
        if "learning_rate" in train:
            lines.append(f"Current LR: {train['learning_rate']:g}")
        if "estimated_remaining_hours" in train:
            lines.append(f"Estimated remaining training: {train['estimated_remaining_hours']:.2f} h (includes previous waits)")
    worker = read(run / "worker.status.json")
    if worker:
        lines += ["", f"GPU {gpus[1]}: {worker.get('phase','--')} | {worker.get('completed','--')}/{worker.get('total','--')}"]
    baseline = read(run / "results/baseline.json")
    if baseline:
        lines.append(f"Local original-v4 score: {baseline['metrics']['score']:.6f}")
    for seed in (42, 43):
        board = read(run / f"leaderboard-{seed}.json")
        for candidate in board.get("candidates", [])[:2]:
            m = candidate["metrics"]
            lines.append(f"seed {seed} {candidate['name']}: {m['score']:.6f}, delta {candidate['primary']['score_gain']:+.6f}, natural {m['counts']['natural_conflict']['correct']}/25, FP {m['counts']['aligned_false_positive']['correct']}/592")
    if pipeline.get("reason"):
        lines.append("STOP REASON: "+pipeline["reason"])
    for role in ("run", "probe", "cache", "smoke", "train", "worker", "export"):
        error = read(run / f"{role}.error.json")
        if error:
            lines.append("ERROR "+role+": "+error["error"][:220])
    lines += ["", "Ctrl+B then 0 progress | 1 pipeline | 2 train | 3 validation | 4 numeric | 5/6 cache | 7 learning | 8 parameters",
              "Ctrl+B then D: detach; server keeps running. Emotion output remains frozen.",
              "A second seed and test CSV require the agreed gates; no automatic Kaggle upload."]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--view", choices=("progress", "parameters"), default="progress")
    parser.add_argument("--gpus", nargs=2, default=["1", "2"])
    args = parser.parse_args()
    while True:
        render_view = render_parameters if args.view == "parameters" else render
        print("\033[2J\033[H"+render_view(args.run, args.gpus), flush=True)
        time.sleep(5)
