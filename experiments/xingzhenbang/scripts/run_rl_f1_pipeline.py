#!/usr/bin/env python3
"""Run train-only F1 caching, smoke/reload checks and a fresh formal experiment.

Launch this controller in tmux with the same environment as run_rl.sh. It only
orchestrates the existing single-GPU runner; it does not change training code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl.experiment import fingerprint, load_records, read_config, write_json
from rl.state import code_identity


def build_plan(data_root, model_path, sft_adapter_path, gpu, pipeline_dir, python=sys.executable):
    """Keep both reward experiments identical except for their emotion reward."""
    correctness_config = ROOT / "configs/rl/correctness_v1.json"
    f1_config = ROOT / "configs/rl/f1_delta_v1.json"
    correctness, correctness_runtime = read_config(correctness_config)
    f1, f1_runtime = read_config(f1_config)
    if correctness["reward_version"] != "correctness_v1" or f1["reward_version"] != "f1_delta_v1":
        raise ValueError("Unexpected reward configuration")
    if ({k: v for k, v in correctness.items() if k != "reward_version"}
            != {k: v for k, v in f1.items() if k != "reward_version"}
            or correctness_runtime != f1_runtime):
        raise ValueError("Paired experiments must differ only in reward_version")
    records, duplicates = load_records(Path(data_root) / "train.jsonl")
    if duplicates:
        raise ValueError("Use the original unique training data for the paired experiment")
    pipeline_dir = Path(pipeline_dir).resolve()
    cache = pipeline_dir / "cache"
    smoke = pipeline_dir / "smoke"
    formal = pipeline_dir / "train"
    common = [str(python), "-u", "-m", "rl.train_grpo",
              "--data-root", str(Path(data_root).resolve()),
              "--model-path", str(Path(model_path).resolve()),
              "--sft-adapter-path", str(Path(sft_adapter_path).resolve()),
              "--gpu", str(gpu), "--micro-batch-size", str(f1_runtime["micro_batch_size"])]
    f1_arguments = ["--config", str(f1_config), "--f1-cache", str(cache / "f1_cache.json")]
    stages = [
        {"stage": "cache_train", "run_dir": str(cache), "target": len(records),
         "command": common + ["--mode", "cache", "--config", str(correctness_config),
                              "--run-dir", str(cache)]},
        {"stage": "smoke", "run_dir": str(smoke), "target": 10,
         "command": common + f1_arguments + ["--mode", "smoke", "--smoke-updates", "10",
                                             "--run-dir", str(smoke)]},
        {"stage": "smoke_reload", "run_dir": str(smoke), "target": 10,
         "command": common + f1_arguments + ["--mode", "smoke", "--smoke-updates", "10",
                  "--run-dir", str(smoke), "--resume-from-checkpoint", str(smoke / "checkpoint-000010")]},
        {"stage": "formal_f1", "run_dir": str(formal), "target": f1["max_steps"],
         "command": common + f1_arguments + ["--mode", "train", "--run-dir", str(formal)]},
    ]
    return {"schema": "f1_pipeline_v1", "pipeline_dir": str(pipeline_dir),
            "gpu": str(gpu), "experiment": f1, "runtime": {**f1_runtime, "gpu": str(gpu)},
            "training_data_hash": fingerprint(records), "unique_train": len(records), "stages": stages}


def launch_stage(stage, pipeline_log):
    """Tee child output to tmux, the aggregate log and a per-run console log."""
    console_path = Path(stage["run_dir"] + ".console.log")
    with console_path.open("a", encoding="utf-8") as console:
        process = subprocess.Popen(stage["command"], cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace", bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                for stream in (console, pipeline_log):
                    stream.write(line)
                    stream.flush()
        finally:
            process.stdout.close()
        result = process.wait()
    if result:
        raise RuntimeError(f"Stage {stage['stage']} exited with code {result}")


def validate_stage(stage):
    run_dir = Path(stage["run_dir"])
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise RuntimeError(f"Stage {stage['stage']} did not complete: {status}")
    if stage["stage"] == "cache_train":
        cache = json.loads((run_dir / "f1_cache.json").read_text(encoding="utf-8"))
        if (cache.get("schema") != "train_f1_cache_v1" or cache.get("split") != "train"
                or len(cache.get("predictions", {})) != stage["target"]):
            raise RuntimeError("Train-only F1 cache is missing or incomplete")
    else:
        if status.get("completed_steps") != stage["target"]:
            raise RuntimeError(f"Stage {stage['stage']} stopped before its target")
        marker = run_dir / f"checkpoint-{stage['target']:06d}" / "complete.json"
        if not marker.is_file():
            raise RuntimeError(f"Stage {stage['stage']} has no completed checkpoint")


def execute_plan(plan, launch=launch_stage):
    directory = Path(plan["pipeline_dir"])
    # Exclusive creation prevents duplicate launches and accidental overwrite/resume.
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "plan.json", {**plan, "code": code_identity()})
    stage = None
    with (directory / "pipeline.log").open("a", encoding="utf-8") as log:
        try:
            for stage in plan["stages"]:
                status = {"stage": stage["stage"], "status": "running", "run_dir": stage["run_dir"],
                          "target": stage["target"], "gpu": plan["gpu"], "started_at": time.time()}
                write_json(directory / "pipeline-status.json", status)
                message = f"Starting {stage['stage']}: {stage['run_dir']}\n"
                print(message, end="", flush=True)
                log.write(message)
                log.flush()
                launch(stage, log)
                validate_stage(stage)
            write_json(directory / "pipeline-status.json", {
                **status, "stage": "finished", "status": "complete", "finished_at": time.time()})
            print("F1 cache, smoke, reload and formal experiment completed.", flush=True)
        except BaseException as exc:
            write_json(directory / "pipeline-status.json", {
                "stage": stage["stage"] if stage else "initializing", "status": "failed",
                "run_dir": stage["run_dir"] if stage else None,
                "gpu": plan["gpu"], "error": str(exc), "traceback": traceback.format_exc()})
            log.write(traceback.format_exc())
            log.flush()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "model-path", "sft-adapter-path", "pipeline-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu", required=True, help="Allocated physical GPU index or UUID")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.gpu == "auto":
        parser.error("Bind this pipeline to one allocated GPU so all four stages use the same card")
    plan = build_plan(args.data_root, args.model_path, args.sft_adapter_path, args.gpu, args.pipeline_dir)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    else:
        execute_plan(plan)


if __name__ == "__main__":
    main()
