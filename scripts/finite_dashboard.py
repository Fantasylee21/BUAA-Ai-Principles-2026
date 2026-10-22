#!/usr/bin/env python3
"""Compact read-only Xshell dashboard, independent of the training process."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def render(run):
    pipeline = read(run / "pipeline.status.json")
    lines = ["Omni v4 | finite-action RL | bounded to 2 epochs",
             datetime.now(timezone(timedelta(hours=8))).strftime("Beijing %Y-%m-%d %H:%M:%S"),
             "Phase: " + pipeline.get("phase", "starting"), ""]
    for i in (0,1):
        c = read(run / f"cache-{i}.status.json")
        if c:
            lines.append(f"Cache GPU {i+1}: {c.get('completed',0)}/{c.get('total','--')} | {c.get('phase','--')}")
    p = read(run / "preflight.status.json")
    if p:
        lines.append("Numerical checks: " + p.get("phase", "--"))
    t = read(run / "train.status.json")
    if t:
        lines.extend(["", f"TRAIN | {t.get('phase','--')} | update {t.get('step',0)}/{t.get('maximum_steps',420)} | epoch {t.get('epoch','--')}",
                      "Loss: {} | E[reward]: {} | KL: {}".format(*[
                          f"{t[k]:.6f}" if k in t else "--" for k in ("loss","expected_reward","kl")])])
        if "inputs_per_second" in t:
            lines.append(f"Throughput: {t['inputs_per_second']:.3f} input/s | epoch remaining estimate: {t['estimated_epoch_remaining_hours']:.2f} h")
    w = read(run / "worker.status.json")
    if w:
        lines.append(f"VALIDATION | {w.get('phase','--')} | {w.get('completed','--')}/{w.get('total','--')}")
    board = read(run / "leaderboard.json")
    if board:
        lines.append("")
        lines.append("Zero-update scores: " + " | ".join(f"{k}={v['score']:.6f}" for k,v in board["baselines"].items()))
        for candidate in board["candidates"][:3]:
            lines.append(f"{candidate['name']}: score={candidate['metrics']['score']:.6f}, gain={candidate['score_gain']:+.6f}, eligible={candidate['promote']}")
    for role in ("run","preflight","train","worker","cache"):
        error = read(run / (role + ".error.json"))
        if error:
            lines.append("ERROR " + role + ": " + str(error["error"])[:180])
    lines.extend(["", "Ctrl+B then 1: pipeline | 2: training | 3: validation | 4: preflight",
                  "5/6: cache logs | Ctrl+B then D: detach (server keeps running)",
                  "First epoch: 210 updates. Second only if valid gain >= 0.005 vs both baselines."])
    return "\n".join(lines)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    args=parser.parse_args()
    while True:
        print("\033[2J\033[H" + render(args.run),flush=True)
        time.sleep(5)
