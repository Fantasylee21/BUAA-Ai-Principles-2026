#!/usr/bin/env python3
"""Read-only progress and actual configuration for conflict-cost training."""
import argparse
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import shutil
import time

try:
    from .grounded_dashboard import count, fit_line, gpu_summary, last_metrics, number, read
except ImportError:
    from grounded_dashboard import count, fit_line, gpu_summary, last_metrics, number, read


REASONS = {
    "first_epoch_gate_not_met": "首轮验证收益不足 +0.003，按约定提前停止",
    "early_gate_not_met": "首轮验证收益不足 +0.003，按约定提前停止",
    "primary_gate_not_met": "第一种子未达到 +0.015 晋级要求，已停止",
    "replication_gate_not_met": "第二种子未达到 +0.005 复验要求，已停止",
    "learning_check_failed": "40 步可学习性检查未通过，未启动正式训练",
    "smoke_failed": "40 步可学习性检查未通过，未启动正式训练",
    "smoke_rejected": "40 步可学习性检查未通过，未启动正式训练",
    "budget_exhausted": "24 小时预算已用完，流程结束",
    "deadline": "24 小时预算已用完，流程结束",
    "training_cutoff": "已到第 20 小时，不再启动新训练阶段",
    "training_cutoff_reached": "已到第 20 小时，不再启动新训练阶段",
    "training_time_cutoff": "已到第 20 小时，不再启动新训练阶段",
    "insufficient_replication_budget": "剩余预算不足以复验，保留结果并停止",
    "insufficient_budget": "剩余预算不足，保留结果并停止",
    "exported": "双种子通过，提交文件已生成，等待手动提交 Kaggle",
    "qualified_submission_ready": "双种子通过，提交文件已生成，等待手动提交 Kaggle",
}


def valid_candidate(candidate):
    metrics = candidate.get("metrics", {})
    score = metrics.get("score")
    return (isinstance(score, (int, float)) and math.isfinite(score)
            and all(metrics.get(key) == 0 for key in
                    ("invalid_outputs", "inconsistent_questions", "emotion_changes")))


def load_candidates(run):
    return {seed: [{**item, "seed": seed} for item in
                   read(run / f"leaderboard-{seed}.json").get("candidates", [])]
            for seed in (42, 43)}


def candidate_step(candidate):
    value = candidate.get("step")
    if isinstance(value, int):
        return value
    # Existing immutable checkpoint names end in a six-digit update number.
    for part in reversed(str(candidate.get("checkpoint", candidate.get("name", ""))).split("-")):
        if part.isdigit() and len(part) >= 3:
            return int(part)
    return -1


def active_metrics(run, context, pipeline):
    status = read(run / "train.status.json")
    settings = context.get("settings", {})
    seed = pipeline.get("seed", status.get("seed", settings.get("seed", 42)))
    persisted = last_metrics(run / f"train-{seed}.jsonl")
    training = {**persisted, **status} if status.get("seed", seed) == seed else persisted
    smoke = {**last_metrics(run / "smoke-updates.jsonl"), **read(run / "smoke.status.json")}
    phase = pipeline.get("phase", "initializing")
    learning = (phase == "40_step_learning_and_resume_check" or
                (not persisted and (pipeline.get("reason") == "smoke_rejected" or
                 (phase == "failed" and (run / "smoke.error.json").exists()))))
    active = smoke if learning else training
    limit = (settings.get("smoke_steps", 40) if learning else
             active.get("target_steps", active.get("limit_steps", active.get("maximum_steps",
                 pipeline.get("maximum_steps", settings.get("steps", 420))))))
    return active, seed, learning, limit


def check_label(value):
    return "通过" if value.get("passed") is True else "未通过" if value.get("passed") is False else "等待"


def render(run, gpus=("1", "2")):
    context, pipeline = read(run / "context.json"), read(run / "pipeline.status.json")
    settings = context.get("settings", {})
    phase = pipeline.get("phase", "initializing")
    active, seed, learning, limit = active_metrics(run, context, pipeline)
    now = time.time()
    elapsed = number(max(0, now-context["started"])/3600, ".2f") if "started" in context else "--"
    remaining = number(max(0, context["deadline"]-now)/3600, ".2f") if "deadline" in context else "--"
    baseline = read(run / "results/baseline.json").get("metrics", {})
    free = read(run / "results/baseline-free.json").get("metrics", {})
    new = read(run / "results/baseline-new.json").get("metrics", {})
    boards = load_candidates(run)
    candidates = [item for items in boards.values() for item in items]
    best = max((item for item in candidates if valid_candidate(item)),
               key=lambda item: item["metrics"]["score"], default=None)
    latest = max(candidates, key=lambda item: (item["seed"], candidate_step(item), item.get("name", "")), default=None)
    metrics = best.get("metrics", {}) if best else {}
    changes = best.get("comparison", {}) if best else {}

    def candidate_text(candidate):
        if not candidate:
            return "-- (等待完整验证)"
        current = candidate.get("metrics", {})
        score, base = current.get("score"), baseline.get("score")
        delta = score-base if isinstance(score, (int, float)) and isinstance(base, (int, float)) else None
        validity = "有效" if valid_candidate(candidate) else "无效/约束未核验"
        return f"{number(score)} | seed {candidate['seed']} 步 {candidate_step(candidate)} | Δ {number(delta, '+.6f')} | {validity}"

    def gate_label(stage_seed, stage):
        if any(valid_candidate(item) and item.get(stage, {}).get("eligible") is True
               for item in boards[stage_seed]):
            return "通过"
        return "未达标" if (run / f"train-{stage_seed}.done.json").exists() else "等待"

    overall = number(baseline.get("score"))+" (较高零更新基线)"
    if best and (not isinstance(baseline.get("score"), (int, float)) or metrics["score"] > baseline["score"]):
        overall = candidate_text(best)
    smoke = read(run / "smoke.json")
    learning_check = (f"进行中 {active.get('step', 0)}/40" if learning and not smoke
                      else check_label(smoke))
    if learning and read(run / "smoke.status.json").get("phase") == "training_only_diagnostic":
        worker = read(run / "smoke.status.json")
    else:
        worker = read(run / ("diagnostics.status.json" if phase == "selected_modality_diagnostics"
                             else "worker.status.json"))
    caches = [read(run / f"cache-{index}.status.json") for index in (0, 1)]
    cache_text = " | ".join(f"GPU {gpu}: {value.get('completed', 0)}/{value.get('total', '--')}"
                            for gpu, value in zip(gpus, caches))
    errors = []
    for role in ("run", "probe", "cache", "cache-0", "cache-1", "smoke", "train", "worker", "export", "diagnostics"):
        value = read(run / f"{role}.error.json")
        if value:
            errors.append(role+": "+str(value.get("error", value.get("reason", "未知错误"))))
    reason = pipeline.get("reason", read(run / "train.status.json").get("reason"))
    train_reason = read(run / "train.status.json").get("reason")
    if reason == "primary_gate_not_met" and train_reason == "first_epoch_gate_not_met":
        reason = train_reason
    status_text = REASONS.get(reason, reason or "按固定流程执行；先通过检查，再从原始 v4 正式训练")
    if errors:
        status_text = "ERROR "+"; ".join(errors)
    elif phase in ("complete", "stopped", "failed"):
        status_text = "流程已结束 | "+status_text
    source = "临时可学习性检查" if learning else f"正式训练 seed {seed}"
    lines = [
        "Omni v4 | 冲突直接纠错总览 | "+datetime.now(timezone(timedelta(hours=8))).strftime("北京时间 %Y-%m-%d %H:%M:%S"),
        f"阶段: {phase} | 已用 {elapsed} h | 剩余预算 {remaining} h",
        f"{source}: 参数更新 {active.get('step', 0)}/{limit} | {active.get('phase', '准备中')} | epoch {active.get('epoch', '--')}",
        "",
        f"LOSS: {number(active.get('loss'))} | 主任务 CE: {number(active.get('main_loss'))} | 配对损失: {number(active.get('pair_loss'))}",
        f"KL: {number(active.get('kl'))} | P(correct): {number(active.get('expected_reward'))} | LR: {number(active.get('learning_rate'), '.3g')} | 梯度: {number(active.get('grad_norm'), '.4f')}",
        f"本批主任务正确: {active.get('main_correct', '--')}/{active.get('main_count', '--')} | 独立覆盖: {active.get('unique_main_seen', '--')}/3354 | 本轮覆盖: {active.get('epoch_seen', '--')}/3354",
        f"累计主任务抽取: {active.get('main_draws', '--')} | 配对抽取: {active.get('pair_draws', '--')} | 每轮210步；验证0/105/210/315/420",
        "",
        "KAGGLE 已确认 Public: 0.90813 | 目标: 0.92000 | 本轮新线上成绩: 未录入",
        f"本地零更新: 自由生成 {number(free.get('score'))} | 新提示词字段判别 {number(new.get('score'))} | 晋级基线 {number(baseline.get('score'))}",
        "本地最高有效分 (含基线): "+overall,
        "本轮最佳有效训练分: "+candidate_text(best),
        "最近一次完整验证: "+candidate_text(latest),
        f"最佳: 自然冲突 {count(metrics, 'natural_conflict', 25)} | VA冲突 {count(metrics, 'va_conflict', 10)} | Aligned误报 {count(metrics, 'aligned_false_positive', 592)}",
        f"较原始自由生成: 冲突纠正/丢失 {changes.get('conflicts_corrected', '--')}/{changes.get('conflicts_lost', '--')} | 误报新增/减少 {changes.get('false_positives_added', '--')}/{changes.get('false_positives_removed', '--')}",
        f"最佳: Macro-F1 {number(metrics.get('macro_f1'))} | 非法 {metrics.get('invalid_outputs', '--')} | 问题不合规 {metrics.get('inconsistent_questions', '--')} | 情感变化 {metrics.get('emotion_changes', '--')}",
        f"晋级: seed42 {gate_label(42, 'primary')} | seed43复验 {gate_label(43, 'replication')} | CSV: {'已生成，未自动提交' if (run/'submission-manifest.json').exists() else '未生成；等待双种子通过'}",
        "",
        f"缓存: {cache_text}",
        f"验证/诊断: {worker.get('phase', '等待checkpoint')} {worker.get('completed', '--')}/{worker.get('total', '--')} | 数值检查: {read(run/'probe.status.json').get('phase', '等待')}",
        f"40步可学习性: {learning_check} | 第210步收益须 ≥0.003；晋级 ≥0.015；复验 ≥0.005",
        gpu_summary(gpus),
        "状态: "+status_text,
        "",
        "Ctrl+B 后按 0 总览 | 2 训练日志 | 3 验证 | 8 参数；也可鼠标点击底部窗口名",
        "每5秒刷新；本地分与Public分别显示。Ctrl+B 后按 D 安全离开；20h后不启新训练，24h截止。",
    ]
    columns = max(40, shutil.get_terminal_size((140, 30)).columns-1)
    return "\n".join(fit_line(line, columns) for line in lines)


def render_parameters(run, gpus=("1", "2")):
    context = read(run / "context.json")
    if not context:
        return "训练参数：等待创建不可变运行配置。"
    settings = context.get("settings", {})
    active, seed, learning, limit = active_metrics(run, context, read(run / "pipeline.status.json"))
    return "\n".join([
        "Omni v4 | 实际运行参数 | omni_conflict_cost_v1",
        f"阶段: {read(run/'pipeline.status.json').get('phase', '等待')} | seed {seed} | 更新 {active.get('step', 0)}/{limit}",
        f"AdamW | LR {settings.get('learning_rate', 5e-6):g} | 当前LR {number(active.get('learning_rate'), '.3g')} | weight decay 0 | clip {settings.get('max_grad_norm', 1):g}",
        f"Batch {settings.get('batch_size', 16)} | dropout 0 | warmup {settings.get('warmup_steps', 10)} | cosine horizon {settings.get('steps', 420)} updates",
        "",
        "主损失 = 0.6 mean_C softplus(-d) + 0.4 mean_A softplus(d)",
        "d = logit(yes) - logit(no)；使用完整训练集类别比例加权",
        f"总损失 = 主损失 + {settings.get('pair_weight', .1):g} 配对损失 + {settings.get('beta', .02):g} KL(original v4)",
        "配对损失 = softplus(1 - [d(Conflict)-d(Aligned)])；真实同视频VT/VA相反标签",
        "方法：成本敏感监督纠错 + 模态配对约束；不使用旧策略固定增量回归",
        "",
        "主任务完整3354条，不放回打乱遍历；每轮210更新，最多2轮",
        "每次额外一对真实VT/VA样本；30对打乱循环；主任务覆盖与配对抽取分别记录",
        "仅语言LoRA训练；v4情感/问题/reference冻结；显式实际两模态情感冲突提示词",
        "完整验证：0、105、210、315、420；两种零更新基线取较高者用于所有晋级",
        "",
        "先检查32条训练样本、临时40更新：漏判/误报各至少改对4/8，容易C/A各新增错误≤1/8",
        "同时检查数值、模态隔离、参考冻结、缓存一致性与精确恢复；通过后丢弃临时权重",
        "首轮门槛 +0.003；seed42晋级 +0.015；seed43相同候选更新数独立复验 +0.005",
        "选模要求情感不变、输出合法；完整报告自然/VA冲突与误报，不设旧误报28条硬门槛",
        "双种子通过才生成734行CSV；保留原始ID及情感；用户手动提交Kaggle",
        "",
        f"GPU {gpus[0]}/{gpus[1]}并行准备缓存，随后{gpus[0]}训练/{gpus[1]}验证；各32GiB PyTorch分配上限",
        f"输入：{settings.get('video_fps', '--')} fps | 最多{settings.get('video_max_frames', '--')}帧 | {settings.get('video_pixels', '--')}像素/帧 | prompt最多{settings.get('max_prompt_tokens', '--')} tokens",
        f"总预算 {settings.get('budget_hours', 24):g}h；{settings.get('training_cutoff_hours', 20):g}h后不启新训练；剩余 {number(max(0, context.get('deadline', time.time())-time.time())/3600, '.2f')}h",
        "",
        "Ctrl+B 后按0总览，2训练日志，8参数；Ctrl+B 后按D安全离开",
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--view", choices=("progress", "parameters"), default="progress")
    parser.add_argument("--gpus", nargs=2, default=["1", "2"])
    args = parser.parse_args()
    while True:
        renderer = render_parameters if args.view == "parameters" else render
        print("\033[2J\033[H"+renderer(args.run, args.gpus), end="", flush=True)
        time.sleep(5)
