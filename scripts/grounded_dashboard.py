#!/usr/bin/env python3
"""Read-only Xshell progress for the bounded grounded experiment."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import unicodedata


def read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def last_metrics(path):
    """Read the last complete update even while the trainer appends a record."""
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            position, data = stream.tell(), b""
            while position and data.count(b"\n") < 2:
                start = max(0, position-8192)
                stream.seek(start)
                data = stream.read(position-start)+data
                position = start
        for line in reversed(data.splitlines()):
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    return value
            except (ValueError, UnicodeDecodeError):
                pass
    except OSError:
        pass
    return {}


def number(value, spec=".6f"):
    return format(value, spec) if isinstance(value, (int, float)) and math.isfinite(value) else "--"


def count(metrics, key, total):
    value = metrics.get("counts", {}).get(key, {})
    return f"{value.get('correct', '--')}/{value.get('total', total)}"


def gpu_summary(gpus):
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True, capture_output=True, timeout=2, check=True)
        values = {}
        for line in output.stdout.splitlines():
            index, memory, utilization = (v.strip() for v in line.split(","))
            values[index] = f"GPU {index}: {float(memory)/1024:.1f} GiB / {utilization}%"
        return " | ".join(values.get(g, f"GPU {g}: --") for g in gpus)+" (整卡含其他用户)"
    except (OSError, subprocess.SubprocessError, ValueError):
        return "GPU 状态暂不可读；我方每卡 PyTorch 分配上限 32 GiB"


def fit_line(value, columns):
    value = str(value).replace("\n", " ").replace("\r", " ").replace("\t", " ")
    result, width = [], 0
    for character in value:
        if unicodedata.category(character).startswith("C"):
            continue
        size = 2 if unicodedata.east_asian_width(character) in ("W", "F") else 1
        if width+size > columns-3:
            return "".join(result)+"..."
        result.append(character)
        width += size
    return "".join(result)


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
    now = time.time()
    pipeline = read(run / "pipeline.status.json")
    context = read(run / "context.json")
    phase = pipeline.get("phase", "initializing")
    settings = context.get("settings", {})
    train = read(run / "train.status.json")
    smoke = read(run / "smoke.status.json")
    seed = pipeline.get("seed", train.get("seed", settings.get("seed", 42)))
    persisted = last_metrics(run / f"train-{seed}.jsonl")
    training = {**persisted, **{k: v for k, v in train.items() if k != "seed" or v == seed}} if train.get("seed", seed) == seed else persisted
    learning = phase == "20_step_learning_and_resume_check" and "loss" in smoke
    active = smoke if learning else training
    source = "临时可学习性检查" if learning else f"正式训练 seed {seed}"
    step, limit = active.get("step", 0), (20 if learning else settings.get("steps", 200))
    waiting = "等待首次参数更新" if "loss" not in active else "最近一次更新"
    elapsed = number(max(0, now-context["started"])/3600, ".2f") if context else "--"
    remaining = number(max(0, context["deadline"]-now)/3600, ".2f") if context else "--"
    zero = read(run / "zero-check.json")
    probe = read(run / "probe.status.json")
    smoke_result = read(run / "smoke.json")
    baseline = read(run / "results/baseline.json").get("metrics", {})
    boards = {s: read(run / f"leaderboard-{s}.json").get("candidates", []) for s in (42, 43)}
    candidates = [{**c, "seed": s} for s, values in boards.items() for c in values]
    valid = [c for c in candidates if isinstance(c.get("metrics", {}).get("score"), (int, float))
             and c["metrics"].get("invalid_outputs") == 0 and c["metrics"].get("inconsistent_questions") == 0
             and c["metrics"].get("emotion_changes") == 0]
    best = max(valid, key=lambda c: c["metrics"]["score"], default=None)
    latest = max(candidates, key=lambda c: (c["seed"], c["name"]), default=None)
    best_metrics = best["metrics"] if best else {}
    def candidate_text(candidate):
        if not candidate:
            return "-- (等待完整验证)"
        metrics = candidate["metrics"]
        delta = metrics["score"]-baseline["score"] if baseline else None
        checkpoint = Path(candidate["checkpoint"]).name
        return f"{number(metrics['score'])} | seed {candidate['seed']} {checkpoint} | 较 v4 {number(delta, '+.6f')}"
    overall = "-- (等待本地基线验证)"
    if baseline:
        overall = f"{number(baseline['score'])} (原始 v4)"
        if best and best_metrics["score"] > baseline["score"]:
            overall = candidate_text(best)
    def gate_label(s, stage):
        if any(c.get(stage, {}).get("eligible") for c in boards[s]):
            return "通过"
        return "未达标" if (run / f"train-{s}.done.json").exists() else "等待"
    worker = read(run / "worker.status.json")
    shards = [read(run / f"cache-{i}.status.json") for i in (0, 1)]
    cache_progress = " | ".join(f"GPU {gpu}: {s.get('completed', 0)}/{s.get('total', 2010)}" for gpu, s in zip(gpus, shards))
    completed = sum(s.get("completed", 0) for s in shards)
    total = sum(s.get("total", 2010) for s in shards)
    cache_eta = "--"
    if phase == "parallel_task_cache" and completed and pipeline.get("time"):
        cache_eta = number(max(0, total-completed)*(now-pipeline["time"])/completed/3600, ".2f")+" h (估计)"
    errors = []
    for role in ("run", "probe", "cache", "smoke", "train", "worker", "export"):
        error = read(run / f"{role}.error.json")
        if error:
            errors.append(role+": "+error["error"])
    check_label = lambda value: "通过" if value.get("passed") is True else "未通过" if value.get("passed") is False else "等待"
    status_text = pipeline.get("reason", "运行中；缓存后先做检查，通过后自动开始正式训练")
    if errors:
        status_text = "ERROR "+"; ".join(errors)
    elif phase in ("complete", "stopped", "failed"):
        status_text = "流程已结束，等待分析 | "+pipeline.get("reason", phase)
    elif phase == "training_and_validation":
        status_text = "训练、验证与难例刷新按固定流程执行"
    learning_check = f"进行中 {smoke.get('step', 0)}/20" if phase == "20_step_learning_and_resume_check" and not smoke_result else check_label(smoke_result)
    validation_progress = f"{worker.get('phase', '等待训练 checkpoint')} {worker.get('completed', '--')}/{worker.get('total', '--')}"
    active_lr = number(active.get("learning_rate"), ".3g")
    lines = [
        "Omni v4 | 实验总览 | "+datetime.now(timezone(timedelta(hours=8))).strftime("北京时间 %Y-%m-%d %H:%M:%S"),
        f"阶段: {phase} | 已用 {elapsed} h | 剩余预算 {remaining} h",
        f"{source}: 参数更新 {step}/{limit} | {active.get('phase', '准备中，尚未训练')}",
        "",
        f"LOSS: {number(active.get('loss'))} | 主任务回归: {number(active.get('pair_loss'))} | {waiting}",
        f"Reward P(correct): {number(active.get('expected_reward'))} | KL: {number(active.get('kl'))} | LR: {active_lr}",
        f"梯度范数: {number(active.get('grad_norm'), '.4f')} | 主任务答对 {active.get('main_correct', '--')}/16 | 辅助 {active.get('aux_correct', '--')}/{active.get('aux_count', '--')}",
        f"训练剩余估计: {number(train.get('estimated_remaining_hours'), '.2f')} h | 每 40 步完整验证 + 难例刷新",
        "",
        "KAGGLE 已确认 Public: 0.90813 | 目标: 0.92000 | 本轮新线上成绩: 未录入",
        f"本地 v4 基线: {number(baseline.get('score'))} | Macro-F1: {number(baseline.get('macro_f1'))}",
        "本地最高有效分 (含基线): "+overall,
        "本轮最佳有效 RL 验证分: "+candidate_text(best),
        "最近一次完整验证: "+candidate_text(latest),
        f"最佳 RL: 自然冲突 {count(best_metrics, 'natural_conflict', 25)} | VA 冲突 {count(best_metrics, 'va_conflict', 10)} | Aligned 误报 {count(best_metrics, 'aligned_false_positive', 592)}",
        f"最佳 RL: F1 {number(best_metrics.get('macro_f1'))} | 冲突召回 {number(best_metrics.get('conflict_accuracy'), '.4f')} | 对齐正确率 {number(best_metrics.get('aligned_calibration'), '.4f')}",
        f"最佳 RL 输出: 非法 {best_metrics.get('invalid_outputs', '--')} | 问题不合规 {best_metrics.get('inconsistent_questions', '--')} | 情感变化 {best_metrics.get('emotion_changes', '--')}",
        f"晋级: seed 42 {gate_label(42, 'primary')} | seed 43 复验 {gate_label(43, 'replication')} | CSV: {'已生成，未自动提交' if (run/'submission-manifest.json').exists() else '等待双种子通过'}",
        "",
        f"缓存: {cache_progress} | 剩余约 {cache_eta}",
        "验证/难例任务: "+validation_progress,
        f"检查: 数值 {probe.get('phase', '等待')} | 零更新一致性 {check_label(zero)} | 20 步可学习性 {learning_check}",
        gpu_summary(gpus),
        "状态: "+status_text,
        "",
        "Ctrl+B 后按 0 总览 | 2 训练日志 | 3 验证 | 8 参数；也可鼠标点击底部窗口名",
        "每 5 秒刷新；本地验证分与 Kaggle Public 分分别显示。Ctrl+B 后按 D 可安全离开。",
    ]
    columns = max(40, shutil.get_terminal_size((118, 30)).columns-1)
    return "\n".join(fit_line(line, columns) for line in lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--view", choices=("progress", "parameters"), default="progress")
    parser.add_argument("--gpus", nargs=2, default=["1", "2"])
    args = parser.parse_args()
    while True:
        render_view = render_parameters if args.view == "parameters" else render
        print("\033[2J\033[H"+render_view(args.run, args.gpus), end="", flush=True)
        time.sleep(5)
