#!/usr/bin/env python3
"""Train SFT adapters, then compare all epoch checkpoints on validation."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--version", choices=["v3", "v4_resample2x"], default="v3")
    parser.add_argument("--models", nargs="+", choices=["qwen25omni", "qwen3vl", "qwen25vl"],
                        default=["qwen25omni", "qwen3vl", "qwen25vl"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "CONDA_ENV": "fwt_temp", "CUDA_VISIBLE_DEVICES": args.gpu,
           "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"}
    run_name = f"qlora_sft_{args.version}"
    status_file = root / f"runs/sft/{args.version}_queue_status.json"
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.with_name(f"{args.version}_queue.pid").write_text(str(os.getpid()) + "\n")
    status = {"gpu": args.gpu, "models": {key: {"state": "queued"} for key in args.models}}

    def update(key, state, **extra):
        status["models"][key].update(state=state, **extra)
        status["updated_at"] = datetime.now().astimezone().isoformat()
        status_file.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"model": key, "state": state, **extra}, ensure_ascii=False), flush=True)

    trained = []
    for key in args.models:
        config = root / f"configs/sft/{key}/{run_name}.yaml"
        if (root / f"output/sft/{key}/{run_name}/train_results.json").is_file():
            update(key, "trained", config=str(config), reused_completed_training=True)
            trained.append((key, config))
            continue
        update(key, "training", config=str(config))
        result = subprocess.run(["bash", "scripts/run_sft_training.sh", "qlora", str(config), args.gpu],
                                cwd=root, env=env)
        if result.returncode:
            update(key, "training_failed", exit_code=result.returncode)
            continue
        update(key, "trained")
        trained.append((key, config))

    for key, config in trained:
        update(key, "evaluating_epoch_checkpoints")
        result = subprocess.run([sys.executable, "scripts/select_sft_checkpoint.py", "--config", str(config)],
                                cwd=root, env=env)
        update(key, "complete" if result.returncode == 0 else "evaluation_failed", exit_code=result.returncode)
    return int(any(item["state"] != "complete" for item in status["models"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
