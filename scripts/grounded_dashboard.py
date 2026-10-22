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


def render(run):
    pipeline = read(run / "pipeline.status.json")
    lines = ["Omni v4 | frozen emotion + grounded REBEL RL | target Public 0.92",
             datetime.now(timezone(timedelta(hours=8))).strftime("Beijing %Y-%m-%d %H:%M:%S"),
             "Phase: "+pipeline.get("phase", "initializing data/model checksums"),
             "Two allocated GPUs | 32 GiB allocator limit each | max 24 hours", ""]
    context = read(run / "context.json")
    if context:
        lines.append(f"Wall-clock budget remaining: {max(0,context['deadline']-time.time())/3600:.2f} h")
    for role, name in (("probe", "Numerical check"), ("cache-0", "Cache GPU 1"), ("cache-1", "Cache GPU 2"), ("smoke", "Learning / exact resume check")):
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
        if "estimated_remaining_hours" in train:
            lines.append(f"Estimated remaining training: {train['estimated_remaining_hours']:.2f} h (includes previous waits)")
    worker = read(run / "worker.status.json")
    if worker:
        lines += ["", f"GPU 2: {worker.get('phase','--')} | {worker.get('completed','--')}/{worker.get('total','--')}"]
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
    lines += ["", "Ctrl+B then 1 pipeline | 2 train | 3 validation | 4 numeric | 5/6 cache | 7 learning",
              "Ctrl+B then D: detach; server keeps running. Emotion output remains frozen.",
              "A second seed and test CSV require the agreed gates; no automatic Kaggle upload."]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    while True:
        print("\033[2J\033[H"+render(args.run), flush=True)
        time.sleep(5)
