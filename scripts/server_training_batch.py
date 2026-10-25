"""Durable server continuation and result collection for the fixed training batch.

Run protocol and collect in separate tmux windows. This module never submits to
Kaggle. Existing seed-42 training is observed, never launched a second time.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
import traceback

from rl.experiment import artifact_hash, file_hash, fingerprint
from rl.reward import parse_completion, question_is_consistent
from scripts.evidence_supplement import FIELDS, write_immutable
from scripts.run_evidence_supplement import run_lock

VERSION = "server_model_training_batch_v1"
SOURCE = Path(__file__).resolve().parents[1]
DETAIL_FIELDS = ("sample_id", "protocol", "source_dataset", "true_emotion", "true_sample_type",
                 "pred_emotion", "is_uncertain", "clarifying_question", "parse_status",
                 "inference_error", "elapsed_seconds", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_status(path):
    # Older status writers replace their contents in place. A transient partial
    # write is not evidence that the experiment itself has failed.
    try:
        return read(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def data_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def status(args, phase, **extra):
    value = {"algorithm": VERSION, "phase": phase, "updated_unix": time.time(),
             "pid": os.getpid(), "submitted_to_kaggle": False, **extra}
    path = args.batch_dir / (args.command + ".status.json")
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(data_bytes(value))
    temporary.replace(path)
    print(json.dumps(value, ensure_ascii=False), flush=True)


@contextmanager
def lock(path):
    with run_lock(Path(str(path) + ".d")):
        yield


def check_paths(args):
    project = args.project_root.resolve()
    for name in ("batch_dir", "seed42_run", "seed43_run", "final_run", "vision_run"):
        path = getattr(args, name).resolve()
        if not path.is_relative_to((project / "runs").resolve()):
            raise ValueError(name + " must stay under the server project's runs directory")
        setattr(args, name, path)
    args.project_root = project
    if len({args.batch_dir, args.seed42_run, args.seed43_run, args.final_run, args.vision_run}) != 5:
        raise ValueError("Each stage needs a distinct output directory")
    args.batch_dir.mkdir(parents=True, exist_ok=True)
    (args.batch_dir / "logs").mkdir(exist_ok=True)


def plan(args):
    value = {"algorithm": VERSION, "project_root": str(args.project_root), "source_root": str(SOURCE),
             "seed42_run": str(args.seed42_run), "seed43_run": str(args.seed43_run),
             "final_run": str(args.final_run), "vision_run": str(args.vision_run),
             "audio_cache_dir": str(args.audio_cache_dir.resolve()), "gpu": args.gpu,
             "max_wait_hours": args.max_wait_hours, "poll_seconds": args.poll_seconds,
             "split_seed": 42, "replication_training_seed": 43, "shared_attention": "sdpa",
             "gate": {"score_gain_min": .01, "macro_f1_drop_max": .005, "natural_recall_non_decrease": True},
             "final_epoch_rule": "ceil(mean(two experiment development-selected epoch numbers))",
             "official_validation_for_selection": False, "kaggle_submission": False,
             "source_sha256": {str(path.relative_to(SOURCE)): file_hash(path) for path in (
                 Path(__file__), SOURCE / "scripts/protocol_sft_summary.py", SOURCE / "scripts/protocol_sft_final.py")}}
    write_immutable(args.batch_dir / "batch-plan.json", data_bytes(value))
    return value


def wait_seed42(args, deadline):
    while True:
        path = args.seed42_run / "runner.status.json"
        current = read_status(path)
        if current.get("phase") == "failed":
            raise RuntimeError("Existing seed42 pipeline failed; inspect its preserved logs")
        if current.get("phase") == "complete_two_arm_comparison":
            return
        if time.monotonic() > deadline:
            raise TimeoutError("Seed42 did not finish within the declared wait limit")
        time.sleep(args.poll_seconds)


def run_command(args, command, log_name, deadline):
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                   "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    environment["PATH"] = str(args.project_root / "media-tools/bin") + os.pathsep + environment.get("PATH", "")
    with (args.batch_dir / "logs" / log_name).open("ab") as log:
        process = subprocess.Popen(command, cwd=SOURCE, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=max(1., deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            # The new session is exclusively this command and its descendants.
            if os.getpgid(process.pid) != process.pid:
                raise RuntimeError("Unexpected child process group; refusing broad termination")
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise TimeoutError("The batch's bounded continuation exceeded its deadline")
    if code:
        raise RuntimeError(f"Command failed with exit {code}; see {log_name}")


def require_free_gpu(args):
    value = subprocess.check_output(["nvidia-smi", "--id=" + str(args.gpu), "--query-gpu=memory.used",
                                    "--format=csv,noheader,nounits"], text=True).strip()
    if not value.isdigit() or int(value) > 512:
        raise RuntimeError("Selected GPU is occupied; other processes will remain untouched")


def checked_final_result(root):
    root = Path(root).resolve()
    report = read(root / "final-report.json")
    candidate = root / "candidate.csv"
    checkpoint = Path(report["checkpoint"]).resolve()
    if (report.get("status") != "complete" or report.get("official_test_rows") != 734 or
            report.get("submitted_to_kaggle") is not False or report.get("official_val_used_for_selection") is not False or
            Path(report["candidate"]).resolve() != candidate or report.get("candidate_sha256") != file_hash(candidate) or
            not checkpoint.is_relative_to(root / "checkpoints") or
            report.get("adapter_sha256") != artifact_hash(checkpoint, adapter=True)):
        raise ValueError("Final result lacks a complete, locked adapter and candidate")
    with candidate.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Final candidate schema changed")
        rows = list(reader)
    if len(rows) != 734 or len({r.get("sample_id") for r in rows}) != 734 or any(not r.get("sample_id") for r in rows):
        raise ValueError("Final candidate does not contain 734 unique IDs")
    for row in rows:
        value = {key: row[key] for key in FIELDS[1:]}
        if parse_completion(json.dumps(value)) is None or not question_is_consistent(value):
            raise ValueError("Final candidate contains an invalid prediction")
    return report


def checked_visual_result(root):
    path = Path(root) / "evaluation/final-report.json"
    report, marker = read(path), read(str(path) + ".complete.json")
    if (report.get("status") != "complete" or report.get("validation_used") is not False or report.get("test_used") is not False or
            marker.get("sha256") != file_hash(path) or marker.get("payload_fingerprint") != fingerprint(report)):
        raise ValueError("Visual completion lacks its verified evaluation report")
    return report


def protocol(args):
    from scripts.protocol_sft_summary import summarize
    result_path = args.batch_dir / "protocol-result.json"
    if result_path.exists():
        result = read(result_path)
        if result.get("plan_hash") != fingerprint(plan(args)):
            raise ValueError("Existing result belongs to another frozen plan")
        return result
    with lock(args.batch_dir / "protocol.lock"):
        frozen = plan(args)
        deadline = time.monotonic() + args.max_wait_hours * 3600
        result = {"algorithm": VERSION, "plan_hash": fingerprint(frozen), "submitted_to_kaggle": False}
        try:
            status(args, "waiting_for_existing_seed42_training")
            wait_seed42(args, deadline)
            first = summarize(args.seed42_run)
            result["seed42_gate"] = first["gate"]
            if not first["gate"]["passed"]:
                result.update(status="complete", reason="seed42_gate_not_met", final_model_trained=False)
            else:
                status(args, "preparing_seed43_replication")
                run_command(args, [sys.executable, "-m", "scripts.protocol_sft_ablation", "prepare",
                    "--output-dir", str(args.seed43_run), "--model-path", str(args.project_root / "models/Qwen2.5-Omni-3B"),
                    "--dataset-root", str(args.project_root / "Dataset"), "--train-jsonl", str(args.project_root / "Dataset/train.jsonl"),
                    "--audio-cache-dir", str(args.audio_cache_dir), "--training-seed", "43", "--flash-attn", "sdpa"],
                    "seed43-prepare.log", deadline)
                for arm in ("control", "experiment"):
                    require_free_gpu(args)
                    status(args, "training_and_evaluating_seed43_" + arm)
                    run_command(args, [sys.executable, "-m", "scripts.protocol_sft_ablation", "train", "--output-dir",
                        str(args.seed43_run), "--arm", arm, "--gpu", str(args.gpu)], "seed43-" + arm + ".log", deadline)
                second = summarize(args.seed43_run)
                result["seed43_gate"] = second["gate"]
                if not second["gate"]["passed"]:
                    result.update(status="complete", reason="seed43_gate_not_met", final_model_trained=False)
                else:
                    require_free_gpu(args)
                    status(args, "final_full_training_and_locked_evaluation")
                    run_command(args, [sys.executable, "-m", "scripts.protocol_sft_final", "run", "--seed42-run",
                        str(args.seed42_run), "--seed43-run", str(args.seed43_run), "--output-dir", str(args.final_run),
                        "--dataset-root", str(args.project_root / "Dataset"), "--model-path", str(args.project_root / "models/Qwen2.5-Omni-3B"),
                        "--audio-cache-dir", str(args.audio_cache_dir), "--gpu", str(args.gpu)], "final-training.log", deadline)
                    checked_final_result(args.final_run)
                    result.update(status="complete", reason="both_seed_gates_passed_and_final_run_completed",
                                  final_model_trained=True, final_report=str(args.final_run / "final-report.json"))
        except Exception as exc:
            result.update(status="failed", reason=str(exc), error_type=type(exc).__name__, traceback=traceback.format_exc())
        write_immutable(result_path, data_bytes(result))
        status(args, result["status"], reason=result["reason"])
        return result


def artifact_paths(root):
    """Small review artifacts only: never model tensors, media or feature arrays."""
    root = Path(root)
    found = set()
    if not root.is_dir():
        return []
    for pattern in ("*.json", "*.md", "*.txt", "*.sha256", "config.yaml", "candidate.csv", "logs/*.log", "evaluation/*.json", "evaluation/*.csv",
                    "evaluation/*.md", "control/*.json", "experiment/*.json", "control/config.yaml", "experiment/config.yaml",
                    "control/evaluation/*_metrics.json", "control/evaluation/*_run_config.json", "control/evaluation/*.csv",
                    "experiment/evaluation/*_metrics.json", "experiment/evaluation/*_run_config.json", "experiment/evaluation/*.csv",
                    "control/checkpoints/*.json", "control/checkpoints/*.jsonl", "control/checkpoints/*.png",
                    "experiment/checkpoints/*.json", "experiment/checkpoints/*.jsonl", "experiment/checkpoints/*.png",
                    "control/checkpoints/checkpoint-*/trainer_state.json", "experiment/checkpoints/checkpoint-*/trainer_state.json",
                    "checkpoints/*.json", "checkpoints/*.jsonl", "checkpoints/*.png", "checkpoints/checkpoint-*/trainer_state.json"):
        found.update(p for p in root.glob(pattern) if p.is_file() and not p.is_symlink())
    return sorted(found)


def detail_projection(source, destination):
    records = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            records.append({key: value[key] for key in DETAIL_FIELDS if key in value})
    content = "".join(json.dumps(v, ensure_ascii=False, allow_nan=False) + "\n" for v in records).encode()
    write_immutable(destination, content)
    return destination


def collect(args):
    with lock(args.batch_dir / "collect.lock"):
        done = args.batch_dir / "COMPLETE.json"
        if done.exists():
            previous = read(done)
            for name, digest in previous["files"].items():
                if file_hash(args.batch_dir / name) != digest:
                    raise ValueError("Completed delivery checksum changed")
            return previous
        deadline = time.monotonic() + args.max_wait_hours * 3600
        status(args, "waiting_for_protocol_and_visual_results")
        while True:
            protocol_path = args.batch_dir / "protocol-result.json"
            vision_path = args.vision_run / "pipeline.status.json"
            visual = read_status(vision_path)
            if protocol_path.is_file() and visual.get("phase") in ("complete", "failed"):
                break
            if time.monotonic() > deadline:
                status(args, "collection_wait_limit_reached", files_still_on_server=True)
                return None
            time.sleep(args.poll_seconds)
        protocol_result = read(protocol_path)
        visual_report = checked_visual_result(args.vision_run) if visual["phase"] == "complete" else None
        if protocol_result.get("final_model_trained") is True:
            checked_final_result(args.final_run)
        runs = {"seed42": args.seed42_run, "seed43": args.seed43_run, "final": args.final_run, "vision": args.vision_run}
        files = {}
        for label, root in runs.items():
            for path in artifact_paths(root):
                files[label + "/" + str(path.relative_to(root))] = path
            for path in root.glob("**/*_details.jsonl") if root.is_dir() else ():
                if "/evaluation/" not in path.as_posix() or path.is_symlink():
                    continue
                projected = args.batch_dir / "projected-details" / label / path.relative_to(root)
                detail_projection(path, projected)
                files["details/" + label + "/" + str(path.relative_to(root))] = projected
        for name in ("batch-plan.json", "protocol-result.json", "IMPLEMENTATION_PLAN.md", "source-manifest.json"):
            if (args.batch_dir / name).is_file():
                files["batch/" + name] = args.batch_dir / name
        for path in (args.batch_dir / "logs").glob("*.log"):
            if path.name not in ("protocol-driver.log", "collect-driver.log") and path.is_file():
                files["batch/logs/" + path.name] = path
        listing = {name: {"sha256": file_hash(path), "bytes": path.stat().st_size} for name, path in sorted(files.items())}
        report = {"algorithm": VERSION, "protocol": protocol_result, "vision_pipeline": visual,
                  "visual_report": visual_report,
                  "previous_public_best": .92313, "target_public_score": .95, "new_public_score": None,
                  "submitted_to_kaggle": False, "files": listing}
        report_path = args.batch_dir / "FINAL_REPORT.json"
        write_immutable(report_path, data_bytes(report))
        archive = args.batch_dir / "review-results.tar.gz"
        if archive.exists():
            raise ValueError("Existing unpublished result archive must be inspected, not overwritten")
        temporary_archive = archive.with_suffix(archive.suffix + ".partial")
        with tarfile.open(temporary_archive, "x:gz") as tar:
            tar.add(report_path, arcname="FINAL_REPORT.json", recursive=False)
            for name, path in sorted(files.items()):
                tar.add(path, arcname=name, recursive=False)
        temporary_archive.rename(archive)
        completed = {"algorithm": VERSION, "status": "complete", "protocol_status": protocol_result["status"],
                     "visual_status": visual["phase"], "submitted_to_kaggle": False,
                     "files": {path.name: file_hash(path) for path in (report_path, archive)},
                     "model_weights_retained_on_server": True}
        write_immutable(done, data_bytes(completed))
        status(args, "complete", protocol_status=protocol_result["status"], visual_status=visual["phase"])
        return completed


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("command", choices=("protocol", "collect"))
    value.add_argument("--project-root", type=Path, required=True)
    for name in ("batch-dir", "seed42-run", "seed43-run", "final-run", "vision-run", "audio-cache-dir"):
        value.add_argument("--" + name, type=Path, required=True)
    value.add_argument("--gpu", default="1")
    value.add_argument("--poll-seconds", type=float, default=20.)
    value.add_argument("--max-wait-hours", type=float, default=72.)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if args.poll_seconds <= 0 or args.max_wait_hours <= 0 or str(args.gpu) != "1":
        raise ValueError("This frozen batch uses GPU1 and positive bounded waits")
    check_paths(args)
    try:
        return protocol(args) if args.command == "protocol" else collect(args)
    except Exception as exc:
        error = {"algorithm": VERSION, "status": "failed", "reason": str(exc),
                 "traceback": traceback.format_exc(), "submitted_to_kaggle": False}
        write_immutable(args.batch_dir / (args.command + "-error.json"), data_bytes(error))
        if args.command == "protocol" and not (args.batch_dir / "protocol-result.json").exists():
            write_immutable(args.batch_dir / "protocol-result.json", data_bytes(error))
        status(args, "failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
