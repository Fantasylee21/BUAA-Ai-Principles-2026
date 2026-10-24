#!/usr/bin/env python3
"""Evidence pipeline: actual CE/calibration losses, scores and explicit status."""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import time

try:
    from .grounded_dashboard import read, number, count, fit_line, gpu_summary
except ImportError:
    from grounded_dashboard import read, number, count, fit_line, gpu_summary


REASONS = {"primary_gate_not_met": "正常结束：seed42 未达到 +0.015；没有启动复验/测试",
    "replication_gate_not_met": "正常结束：分组种子43未达到 +0.005；没有生成 CSV",
    "qualified_submission_ready": "正常结束：两组分折均通过，CSV已生成；等待手动提交"}


def render(run, gpus=("1", "2"), parameters=False):
    run = Path(run)
    c = read(run/"context.json")
    pipeline = read(run/"pipeline.status.json")
    losses = read(run/"losses.json")
    fit = [read(run/f"fit-{seed}.status.json") for seed in (42, 43)]
    latest = max(fit, key=lambda x: x.get("updated", 0))
    base = read(run/"results/baseline.json").get("metrics", {})
    primary = read(run/"results/evidence-seed42.json")
    replication = read(run/"results/evidence-seed43.json")
    candidates = [v for v in (primary, replication) if v]
    best = max(candidates, key=lambda x: x["metrics"]["score"], default={})
    metrics = best.get("metrics", {})
    changes = best.get("comparison", {})
    dev = read(run/"development.json")
    pre = read(run/"preflight.json")
    error = read(run/"pipeline.error.json") if pipeline.get("phase") == "failed" else {}
    reason = "程序故障："+str(error.get("error", pipeline.get("error"))) if error or pipeline.get("phase") == "failed" else REASONS.get(pipeline.get("reason"), "运行中："+pipeline.get("phase", "准备中"))
    delta = metrics.get("score", base.get("score", 0))-base.get("score", 0) if base and metrics else None
    ce = " | ".join(f"{mode}: {number(losses.get(mode,{}).get('data_loss'))}" for mode in ("vision", "text", "audio"))
    lines = [
        "Omni 原始基座 | 双路情感证据 + 成本校准 | "+datetime.now(timezone(timedelta(hours=8))).strftime("北京时间 %m-%d %H:%M:%S"),
        f"阶段: {pipeline.get('phase','准备中')} | 时间限制: 无 | 提取用GPU；小模型拟合用CPU/FP64",
        "LOSS 交叉熵: "+ce+" | 校准: "+number(losses.get("calibration",{}).get("data_loss")),
        f"当前拟合: {latest.get('job','--')} / {latest.get('mode','--')} | 迭代 {latest.get('iteration','--')}/300 | 目标损失 {number(latest.get('loss'))} | 梯度 {number(latest.get('gradient_max'),'.3g')}",
        "特征进度: "+" | ".join(f"GPU{gpus[i]} {read(run/f'cache-{i}.status.json').get('completed',0)}/{read(run/f'cache-{i}.status.json').get('total','--')}" for i in (0,1)),
        f"本地v4: {number(base.get('score'))} | 本轮最高正式验证: {number(metrics.get('score'))} | Δ {number(delta,'+.6f')} | 分组seed {best.get('seed','--')}",
        f"开发外折冲突任务分: {number(dev.get('task_component'))}/0.5 | seed42验证: {number(primary.get('metrics',{}).get('score'))} | seed43验证: {number(replication.get('metrics',{}).get('score'))}",
        f"自然冲突 {count(metrics,'natural_conflict',25)} | VA冲突 {count(metrics,'va_conflict',10)} | 自然VA {count(metrics,'natural_va_conflict',5)} | 误报 {count(metrics,'aligned_false_positive',592)}",
        f"相对v4：冲突纠正/丢失 {changes.get('conflicts_corrected','--')}/{changes.get('conflicts_lost','--')} | 误报新增/消除 {changes.get('false_positives_added','--')}/{changes.get('false_positives_removed','--')}",
        "Kaggle 已确认 Public 0.90813 | 目标0.92000 | 本轮线上未提交；本地分不能直接换算线上分",
        "参数：原始Omni完全冻结 | 18441个残差参数+2个校准偏置 | L2=0.01 | L-BFGS最多300迭代/拟合",
        f"真实检查: {'通过' if pre.get('passed') else '等待'} | 双卡特征估时 {number(pre.get('estimated_feature_hours_two_gpus'),'.2f')}h | 峰值 {number(pre.get('peak_allocated_gib'),'.2f')}GiB",
        gpu_summary(gpus),
        "状态: "+reason,
        "Ctrl+B 后按 0 总览 | 1 流程 | 2/3 特征日志 | 4 CPU拟合日志 | 5 参数 | D 安全离开",
    ]
    if parameters:
        lines = [lines[0],
            "原始Qwen2.5-Omni-3B：NF4/BF16，只读；证据提取不加载任何v4 adapter",
            "视觉/文本/音频：各用2048维最后token表征+原生三情感logits；无全模态分支",
            "s = 原生logits + W*无参数LayerNorm(h) + c；W,c初始为0，共18441个参数",
            "各模态：平均交叉熵 + 0.01/2*(||W||²+||c||²)；只用clear标注；视觉去重",
            "q=1-dot(pV,pM)；d=logit(q)+协议偏置；d>0为yes；没有额外阈值搜索",
            "校准：0.6*Conflict条件CE+0.4*Aligned条件CE+0.01/2*||偏置||²",
            "CPU/FP64 L-BFGS；maxiter300、gtol1e-5、ftol1e-12；不收敛即保存故障并停止",
            "三折视频外折；剩余数据20%为内部校准；最终偏置仅拟合折外分数",
            "seed42提高>=0.015且输出合法后，才做seed43分折稳定性复验（>=0.005）",
            "情感与问题：冻结v4；不修改原始734行的ID、顺序和情感列",
            "每卡分配上限32GiB；无墙钟时限；CPU阶段GPU低占用正常",
            "不加载旧RL优化器；未完成小任务重算；完成任务校验身份后复用",
            "配置与身份："+str(run/"context.json"), "Ctrl+B 后按0返回总览；D安全离开"]
    terminal = shutil.get_terminal_size((150, 35))
    height = max(10, terminal.lines-1)
    if len(lines) > height:
        lines = lines[:height-2]+["状态: "+reason, "Ctrl+B 0总览 | 1流程 | 2/3特征 | 4拟合 | 5参数 | D离开"]
    return "\n".join(fit_line(line, max(40, terminal.columns-1)) for line in lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--gpus", nargs=2, default=["1", "2"])
    p.add_argument("--parameters", action="store_true")
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    while True:
        print(("" if args.once else "\033[2J\033[H")+render(args.run, args.gpus, args.parameters), flush=True)
        if args.once:
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
