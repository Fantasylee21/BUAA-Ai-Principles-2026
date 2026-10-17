#!/usr/bin/env python3
"""Keep the existing v3 queue running independently of the chat connection."""

import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "runs/sft/v3_queue.log"
STATUS = ROOT / "runs/sft/v3_queue_status.json"
COMMANDS = {"run_sft_v3.py", "run_sft_training.sh", "train_sft.py",
            "evaluate_sft.py", "select_sft_checkpoint.py"}


def queue_processes_exist():
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            command = (directory / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if not any(Path(arg).name in COMMANDS for arg in command if arg):
                continue
            if (directory / "cwd").resolve() == ROOT:
                return True
        except (OSError, PermissionError):
            continue
    return False


def main():
    offset = max(0, LOG.stat().st_size - 1800) if LOG.exists() else 0
    recovered = None
    output = None
    print("Monitoring the existing v3 queue; current training is not restarted.", flush=True)
    while True:
        if LOG.exists():
            with LOG.open(encoding="utf-8", errors="replace") as stream:
                stream.seek(offset)
                delta = stream.read()
                offset = stream.tell()
            if delta:
                print(delta, end="", flush=True)
        if recovered is not None and recovered.poll() is not None:
            print(f"\nRecovered queue finished with exit code {recovered.returncode}.", flush=True)
            output.close()
            return recovered.returncode
        if not queue_processes_exist():
            status = json.loads(STATUS.read_text())
            states = [item["state"] for item in status["models"].values()]
            if all(state == "complete" for state in states):
                print("\nAll v3 models and checkpoint evaluations finished.", flush=True)
                return 0
            if any(state.endswith("failed") for state in states):
                print("\nQueue reported a failure; inspect v3_queue.log before retrying.", flush=True)
                return 1
            print("\nOriginal queue exited unexpectedly; resuming saved checkpoints in tmux.", flush=True)
            output = LOG.open("a")
            recovered = subprocess.Popen([sys.executable, str(ROOT / "scripts/run_sft_v3.py"),
                                          "--gpu", str(status["gpu"])], cwd=ROOT,
                                         stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
