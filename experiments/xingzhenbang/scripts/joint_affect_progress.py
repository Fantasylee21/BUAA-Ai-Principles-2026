"""Read-only terminal dashboard for the joint-affect tmux batch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from scripts.joint_affect_pipeline import progress_snapshot


def last_record(path):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(max(0, Path(path).stat().st_size - 65536))
            lines = stream.read().decode("utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            try:
                return json.loads(line)
            except (ValueError, TypeError):
                pass
    except FileNotFoundError:
        pass
    return {}


def duration(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}"


def bar(current, total):
    fraction = min(1., max(0., current / total)) if total else 0.
    filled = int(fraction * 32)
    return "[" + "#" * filled + "." * (32 - filled) + f"] {current}/{total} ({fraction:.1%})"


def render(root, snapshot, gpu_text=""):
    lines = ["JOINT AFFECT | server training | " + time.strftime("%Y-%m-%d %H:%M:%S"),
             "Batch: " + str(Path(root).resolve()), "Phase: " + snapshot.get("status", {}).get("phase", "waiting")]
    clock = snapshot.get("clock", {})
    if clock:
        lines.append("Batch elapsed " + duration(clock["elapsed_wall_seconds"]) +
                     " | remaining budget " + duration(clock["budget_seconds_available_now"]) + " (budget, not ETA)")
    if gpu_text:
        lines += ["", gpu_text.strip()]
    for name, status in snapshot.get("children", {}).items():
        directory = Path(root) / name
        phase = status.get("phase", "initializing")
        record = last_record(directory / "trainer_log.jsonl")
        reference = last_record(directory / "reference-checkpoints/trainer_log.jsonl")
        if phase == "training_sft_reference":
            record = reference
        lines += ["", name, "  " + phase]
        current = record.get("current_steps", status.get("completed", 0))
        total = record.get("total_steps", status.get("total", 0))
        if total:
            lines.append("  " + bar(current, total))
        if phase in ("training", "training_sft_reference"):
            details = [f"{key}={record[key]:.4f}" for key in ("epoch", "loss", "real_loss", "remix_loss", "vision_loss", "second_loss", "relation_loss", "joint_loss", "grad_norm")
                       if isinstance(record.get(key), (int, float))]
            lines.append("  " + " | ".join(details))
            remaining = record.get("remaining_seconds_estimate")
            if remaining is not None:
                lines.append("  This training job ETA: " + duration(remaining))
            observations = record.get("training_observations", {})
            values = []
            for branch in ("vision", "second", "natural_va"):
                n = observations.get(branch + "_n", 0)
                if n:
                    k = observations[branch + "_correct"]
                    values.append(f"{branch} {k}/{n}={k/n:.1%}")
            if values:
                lines.append("  Running TRAIN observations (not held-out): " + " | ".join(values))
            gradients = record.get("module_gradients", {})
            if gradients:
                lines.append("  Module gradient norms: " + " | ".join(f"{k} {v:.2e}" for k, v in sorted(gradients.items())))
        if phase == "real_preflight":
            lines.append("  Backward/update check: " + status.get("stage", ""))
        if status.get("error"):
            lines.append("  ERROR: " + status["error"])
    if not snapshot.get("children"):
        lines.append("\nNo active child. See phase and FINAL_REPORT.json for completed/failed batches.")
    lines.append("\nRead-only panel; Ctrl-C here stops only this display. No Kaggle submissions.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        try:
            gpu = subprocess.run(["nvidia-smi", "--id=1,2", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                                  "--format=csv"], capture_output=True, text=True, timeout=5).stdout
            output = render(args.run_dir, progress_snapshot(args.run_dir), gpu)
        except Exception as exc:
            output = "Progress read pending: " + str(exc)
        print(("" if args.once else "\x1b[2J\x1b[H") + output, flush=True)
        if args.once:
            return
        time.sleep(3)


if __name__ == "__main__":
    main()
