#!/usr/bin/env python3
"""Read-only, standard-library tmux dashboard for the server training batch."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time


def read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def tail(path, size=32768):
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - size))
            return stream.read(size).decode("utf-8", errors="replace")
    except OSError:
        return ""


def records(path):
    result = []
    for line in tail(path).splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                result.append(value)
        except ValueError:
            pass
    return result


def bar(done, total, width=32):
    if not total:
        return "[waiting for progress]"
    ratio = max(0., min(1., done / total))
    filled = int(width * ratio)
    return f"[{'#' * filled}{'-' * (width - filled)}] {done}/{total} ({ratio:.1%})"


def number(value):
    return format(value, ".6g") if isinstance(value, (int, float)) else "--"


def processes():
    if not hasattr(os, "getuid"):
        return []
    try:
        value = subprocess.run(["ps", "-u", str(os.getuid()), "-o", "args="],
                               capture_output=True, text=True, timeout=2, check=True)
        return [shlex.split(line) for line in value.stdout.splitlines()
                if "/scripts/evaluate_sft.py" in line or "/scripts/train_sft.py" in line]
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def option(command, name):
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError):
        return ""


def completed_ids(path):
    ids = set()
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    if row.get("sample_id") is not None:
                        ids.add(str(row["sample_id"]))
                except ValueError:
                    pass
    except OSError:
        pass
    return len(ids)


def render(project, batch):
    plan = read(batch / "batch-plan.json")
    protocol = read(batch / "protocol.status.json")
    result = read(batch / "protocol-result.json")
    seed42 = Path(plan.get("seed42_run", project / "runs/protocol-sft-ablation-seed42-20261025"))
    seed43 = Path(plan.get("seed43_run", project / "runs/protocol-sft-ablation-seed43-20261025"))
    final = Path(plan.get("final_run", project / "runs/protocol-sft-final-20261025"))
    vision = Path(plan.get("vision_run", project / "runs/visual-expert-v2-20261025"))
    phase = protocol.get("phase", "waiting_for_batch")
    run = final if phase == "final_full_training_and_locked_evaluation" else seed43 if "seed43" in phase else seed42
    terminal = phase in ("complete", "failed")
    if terminal:
        run = final if (final / "status.json").exists() else seed43 if (seed43 / "identity.json").exists() else seed42
    commands = [c for c in processes() if any(str(run) in arg for arg in c)]
    evaluation = next((c for c in commands if any(a.endswith("/scripts/evaluate_sft.py") for a in c)), None)
    training = next((c for c in commands if any(a.endswith("/scripts/train_sft.py") for a in c)), None)
    arm = ""
    if run != final:
        relevant = evaluation or training or []
        arm = next((a for a in ("control", "experiment")
                    if any(str(run / a) in part for part in relevant)), "")
        if not arm:
            arm = ("experiment" if "experiment" in read(run / "runner.status.json").get("phase", phase)
                   or (run / "control/holdout-report.json").exists() else "control")
    arm_root = run / arm if arm else run
    checkpoints = arm_root / "checkpoints"
    values = records(checkpoints / "trainer_log.jsonl")
    latest = values[-1] if values else {}
    loss = next((v for v in reversed(values) if v.get("loss") is not None), {})
    lr = next((v for v in reversed(values) if v.get("lr") is not None), {})
    trained = (checkpoints / "train_results.json").is_file()
    step, total = latest.get("current_steps", 0), latest.get("total_steps", 0)
    if trained:
        step = total or read(checkpoints / "trainer_state.json").get("global_step", 0)
        total = step
    label = ("generation evaluation" if evaluation else "post-training loss evaluation" if training and trained
             else "training / epoch validation" if training else "waiting / stage transition")
    if terminal:
        label = "A route " + phase
    lines = ["SERVER MODEL TRAINING  |  " + datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S UTC+8"),
             "=" * 92, f"A / GPU 1: {run.name}  {arm}", f"Current: {label}" + ("; metrics below are historical" if terminal else ""),
             "Train " + bar(step, total) + ("  TRAINING FINISHED" if trained else ""),
             f"loss={number(loss.get('loss'))} (logged step {loss.get('current_steps', '--')})"
             f"   lr={number(lr.get('lr'))}   epoch={number(latest.get('epoch'))}"]
    if training and not trained:
        lines.append(f"Training logged elapsed={latest.get('elapsed_time', '--')}"
                     f"   remaining estimate={latest.get('remaining_time', '--')}")
    else:
        lines.append("Training ETA: -- (not currently updating weights)")
    if evaluation:
        name = option(evaluation, "--run-name")
        out = Path(option(evaluation, "--output-dir"))
        partition = "holdout" if name.endswith("_holdout") else "dev"
        counts = read(run / "split-manifest.json").get("partitions", {})
        expected = counts.get(partition, {}).get("rows", 0) if run != final else (734 if name.endswith("_test") else 666)
        done = completed_ids(out / (name + "_details.jsonl"))
        lines += [f"Evaluate {name}", "Saved rows " + bar(done, expected) + "  (buffered updates)"]
    if run != final:
        dev_count = len(list((arm_root / "evaluation").glob("checkpoint-*_dev_metrics.json")))
        lines.append(f"Development checkpoints finished: {dev_count}/3"
                     f"   selected holdout finished: {(arm_root / 'holdout-report.json').is_file()}")
    if result:
        lines.append("A outcome: " + str(result.get("reason", result.get("status", "--"))))
    if run == seed42:
        log = run / "logs" / (arm + ".log")
    elif run == seed43:
        log = batch / "logs" / ("seed43-" + arm + ".log")
    else:
        log = batch / "logs/final-training.log"
    lines += ["", "Latest progress in selected log:"]
    progress = []
    for row in re.split(r"[\r\n]", tail(log)):
        clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", row).strip()
        if re.search(r"\d+%\|.*\|\s*\d+/\d+", clean):
            progress.append(clean)
    lines.append(progress[-1][:110] if progress else "No tqdm update yet (loading/preparing may be in progress).")
    try:
        lines.append(f"Log last changed {max(0, int(time.time() - log.stat().st_mtime))}s ago; latest bar may belong to prior stage.")
    except OSError:
        pass
    lines += ["", "B / GPU 2: " + read(vision / "pipeline.status.json").get("phase", "waiting")]
    visual_rows = records(vision / "logs/extract.log")
    cached = next((v for v in reversed(visual_rows) if "completed" in v), {})
    if cached:
        lines.append("Visual features " + bar(cached["completed"], cached.get("total", 0)))
    if read(vision / "pipeline.status.json").get("phase") == "failed":
        errors = tail(vision / "logs/evaluate.log").splitlines() + tail(vision / "logs/extract.log").splitlines()
        error = next((line for line in reversed(errors) if line.startswith(("RuntimeError:", "ValueError:", "AssertionError:"))), "See preserved B logs for failure reason.")
        lines.append(error[:110])
    complete = read(batch / "COMPLETE.json")
    lines += ["", "Batch: " + ("ENDED; inspect FINAL_REPORT.json for outcomes" if complete else read(batch / "collect.status.json").get("phase", "waiting")),
              "Bars describe individual stages; full batch ETA is unknown.",
              "Refresh 3s | Ctrl+b then d: detach and keep training"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    batch = args.batch_dir or args.project_root / "runs/model-training-batch-20261025-r2"
    try:
        while True:
            screen = render(args.project_root, batch)
            print(("" if args.once else "\033[H\033[J") + screen, flush=True)
            if args.once:
                return
            time.sleep(3)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
