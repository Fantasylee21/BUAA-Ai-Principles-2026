#!/usr/bin/env python3
"""Read-only relation experiment overview with loss, scores and stop reasons."""
import argparse
from datetime import datetime,timedelta,timezone
from pathlib import Path
import shutil
import time

try:
    from .grounded_dashboard import read,number,count,fit_line,gpu_summary
except ImportError:
    from grounded_dashboard import read,number,count,fit_line,gpu_summary

REASONS={"primary_gate_not_met":"seed42 未达到 +0.015，正常结束；未启动复验/测试",
         "replication_gate_not_met":"seed43 未达到 +0.005，正常结束；未生成 CSV",
         "qualified_submission_ready":"双种子通过，CSV 已生成；等待你手动提交 Kaggle"}


def render(run,gpus=("1","2"),parameters=False):
    run=Path(run)
    c=read(run/"context.json")
    pipeline=read(run/"pipeline.status.json")
    s=c.get("settings",{})
    base=read(run/"results/baseline.json").get("metrics",{})
    selected=read(run/"selected.json")
    candidates=[read(run/"results"/f"{m}-seed42.json") for m in ("binary_control","joint_relation")]
    candidates=[v for v in candidates if v]
    best=max(candidates,key=lambda v:v["metrics"]["score"],default={})
    metrics=best.get("metrics",{})
    delta=metrics.get("score",base.get("score",0))-base.get("score",0) if base else None
    changes=best.get("comparison",{})
    status=[]
    for path in run.glob("*.status.json"):
        v=read(path)
        if path.name!="pipeline.status.json" and v:
            status.append((path.stem.replace(".status",""),v))
    active=sorted(((k,v) for k,v in status if v.get("phase") not in ("complete","failed")),
                  key=lambda kv:kv[1].get("updated",0),reverse=True)
    training=[(k,v) for k,v in status if "loss" in v]
    latest=max(training,key=lambda kv:kv[1].get("updated",0),default=("--",{}))
    v=latest[1]
    error=read(run/"pipeline.error.json")
    if pipeline.get("phase")!="failed" and error.get("time",0)<pipeline.get("updated",0):
        error={}
    development=read(run/"development.json")
    lines=[
        "Omni v4 | 冻结特征 + 模态关系/二分类对照 | "+datetime.now(timezone(timedelta(hours=8))).strftime("北京时间 %Y-%m-%d %H:%M:%S"),
        f"阶段: {pipeline.get('phase','准备中')} | 时间限制: 无 | 全部 Omni/LoRA 冻结，仅训练约百万参数判别器",
        f"LOSS: {number(v.get('loss'))} | E[任务收益]: {number(v.get('expected_reward'))} | LR: {number(v.get('learning_rate'),'.3g')} | 梯度: {number(v.get('grad_norm'),'.4f')}",
        f"最近训练: {latest[0]} | epoch {v.get('epoch','--')}/{v.get('epochs',40)} | 更新 {v.get('step','--')}/{v.get('total_steps','--')} | 本轮覆盖 {v.get('epoch_seen','--')}/{v.get('epoch_total','--')}",
        "双卡特征: "+" | ".join(f"分片{i} {read(run/f'cache-{i}.status.json').get('completed',0)}/{read(run/f'cache-{i}.status.json').get('total','--')}" for i in (0,1)),
        "KAGGLE 已确认 Public: 0.90813 | 目标 0.92000 | 本轮线上成绩: 尚未提交",
        f"本地原始 v4: {number(base.get('score'))} | 历次已评测最好: 0.714419 | 本轮最好判别分: {number(metrics.get('score'))} | Δ {number(delta,'+.6f')}",
        f"最好方法: {best.get('method','--')} | 输出约束: {'等待问题生成核验' if metrics.get('output_checks_pending',True) else '已核验'}",
        f"自然冲突 {count(metrics,'natural_conflict',25)} | VA冲突 {count(metrics,'va_conflict',10)} | 自然VA {count(metrics,'natural_va_conflict',5)} | 误报 {count(metrics,'aligned_false_positive',592)}",
        f"相对v4: 冲突纠正/丢失 {changes.get('conflicts_corrected','--')}/{changes.get('conflicts_lost','--')} | 误报新增/消除 {changes.get('false_positives_added','--')}/{changes.get('false_positives_removed','--')}",
        "开发三折分只用于新判别器选epoch；与666条正式验证分分开显示。",
    ]
    for method in ("binary_control","joint_relation"):
        choice=development.get("choices",{}).get(method,{})
        result=next((x for x in candidates if x["method"]==method),{})
        lines.append(f"{method}: 开发任务分 {number(choice.get('selection_score'))} /0.5 | 选定epoch {choice.get('epoch','--')} | 正式验证 {number(result.get('metrics',{}).get('score'))}")
    lines.append("")
    for name,value in active[:3]:
        lines.append(f"运行 {name}: {value.get('phase')} | {value.get('completed',value.get('epoch','--'))}/{value.get('total',value.get('epochs','--'))}")
    for i in (0,1):
        cache=read(run/f"cache-{i}.status.json")
        lines.append(f"特征分片 {i}: {cache.get('completed',0)}/{cache.get('total','--')} | {cache.get('phase','等待')} ")
    pre=read(run/"preflight.json")
    lines.extend([
        f"真实模型检查: {'通过' if pre.get('passed') else '等待'} | 特征估时 {number(pre.get('estimated_feature_hours_two_gpus'),'.2f')} h | 峰值分配 {number(pre.get('peak_allocated_gib'),'.2f')} GiB",
        "CSV: "+("已生成，未自动提交" if (run/"submission-manifest.json").exists() else "未生成；须通过 seed42 +0.015、seed43 +0.005 及输出约束"),
        gpu_summary(gpus),
        "状态: "+("ERROR "+str(error.get("error")) if error else REASONS.get(pipeline.get("reason"),pipeline.get("phase","准备中"))),
        "",
        f"参数: AdamW LR {s.get('learning_rate',.0003):g} | wd {s.get('weight_decay',.01):g} | batch 64 | dropout 0.1 | clip 1",
        "最多40 epoch/开发任务 | warmup 2 epoch + cosine(40) | 3折按视频分组 | 阈值0.5 | 类别目标0.6/0.4",
        "小网络训练阶段显存低是正常现象；32 GiB是上限，不是预留。旧实验不会被覆盖。",
        "Ctrl+B 后按 0 总览 | 1 流程 | 2/3 特征与训练日志 | 4 参数；也可鼠标点底部窗口名",
        "Ctrl+B 后按 D 安全离开。无24小时限制；质量未达标和程序故障分别显示。",
    ])
    if parameters:
        lines=[lines[0],lines[1],
            f"实际参数: AdamW LR {s.get('learning_rate',.0003):g} | wd {s.get('weight_decay',.01):g} | batch 64 | dropout 0.1 | clip 1",
            "每个开发任务40 epoch；完整训练使用OOF选定epoch；warmup 2 + cosine固定40 epoch",
            "3折按完整视频分组；seed42两种方法；仅获胜且达标者运行seed43",
            "网络: 3 x LayerNorm(2048)+Linear(2048,128)+GELU；融合642->128->2/9",
            "完整模态表征使用原始v4提示词和预测情感前缀；没有真实标签输入。",
            "joint: 类别加权的允许状态负对数似然；binary: 相同类别权重的交叉熵。",
            "无GRPO采样、无固定F1增量、无old-policy回归、无旧判断KL；完整情感和问题由冻结v4提供。",
            "冲突/对齐条件目标0.6/0.4；冲突分数>0.5输出yes，否则no",
            "每卡PyTorch分配上限32 GiB；无墙钟时间限制；不操作其他用户进程",
            "晋级: seed42本地增益>=0.015；复验>=0.005；输出合法且情感不变",
            "实际配置: "+str(run/"context.json"),"Ctrl+B 后按0返回总览；Ctrl+B 后按D离开"]
    terminal=shutil.get_terminal_size((150,35))
    columns=max(40,terminal.columns-1)
    height=max(10,getattr(terminal,"lines",35)-1)
    if len(lines)>height:
        footer=[gpu_summary(gpus),"状态: "+("ERROR "+str(error.get("error")) if error else
            REASONS.get(pipeline.get("reason"),pipeline.get("phase","准备中"))),
            "Ctrl+B 后按 0 总览 | 1 流程 | 2/3 日志 | 4 参数 | D 安全离开"]
        lines=lines[:height-len(footer)]+footer
    return "\n".join(fit_line(line,columns) for line in lines)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("run",type=Path)
    p.add_argument("--gpus",nargs=2,default=["1","2"])
    p.add_argument("--parameters",action="store_true")
    p.add_argument("--once",action="store_true")
    args=p.parse_args()
    while True:
        text=render(args.run,args.gpus,args.parameters)
        print(("" if args.once else "\033[2J\033[H")+text,flush=True)
        if args.once: break
        time.sleep(5)


if __name__=="__main__": main()
