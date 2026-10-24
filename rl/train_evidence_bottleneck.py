"""Bounded base-only emotion evidence, OOF cost calibration and gated export."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from .evidence_bottleneck import (decision_scores, feature_key, safe_input, softmax, tasks_for)
from .evidence_store import (FeatureBank, context_for, feature_entry, initialize, publish,
                             read, save_feature)
from .experiment import file_hash, fingerprint, write_json
from .finite_reports import comparison
from .modality_relation import gate
from .reward import parse_completion, question_is_consistent
from .train_modality_relation import baseline_records, extra_report


def emit(run, role, phase, **values):
    value = {"phase": phase, "pid": os.getpid(), "updated": time.time(), **values}
    write_json(Path(run)/(role+".status.json"), value)
    if role.startswith("fit-") and "loss" in values:
        target = Path(run)/"losses.json"
        history = read(target) if target.exists() else {}
        mode = values.get("mode", "calibration")
        history[mode] = value
        write_json(target, history)
    line = __import__("json").dumps({"role": role, **value}, ensure_ascii=False)
    slot = os.environ.get("BUAA_EVIDENCE_GPU_SLOT")
    if slot in ("0", "1"):
        path = Path(run)/"logs"/f"gpu-{slot}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line+"\n")
    print(line, flush=True)


def torch_device(args):
    from .grounded_recovery import prepare_environment, configure_torch
    from .omni_runtime import select_gpu
    prepare_environment()
    gpu = select_gpu(args.gpu, min_free_memory_gib=12)
    import torch
    torch.set_num_threads(args.cpu_threads)
    torch.cuda.set_per_process_memory_fraction(min(1., args.memory_gib*1024/gpu["total_mib"]), 0)
    configure_torch(torch)
    print({"selected_gpu": gpu, "allocator_limit_gib": args.memory_gib}, flush=True)
    return torch


def verify_media(c, row, mode, checked):
    if mode == "text" or row["video"] in checked:
        return
    root = Path(c["data_root"]).resolve()
    path = (root/row["video"]).resolve()
    if not path.is_relative_to(root) or file_hash(path) != c["identity"]["media"][row["video"]]:
        raise ValueError("Original media checksum changed or escaped data root")
    checked.add(row["video"])


def preflight(args, c):
    from .evidence_runtime import BaseEvidenceExtractor
    from .omni_io import task_messages
    torch = torch_device(args)
    policy = BaseEvidenceExtractor(c["model"], c["settings"])
    before = policy.check_frozen()
    chosen = [next(r for r in c["train"] if r["protocol"] == p and r["sample_type"] == t)
              for p in ("VT", "VA") for t in ("Conflict", "Aligned")]
    tasks = tasks_for(chosen, "train")
    checks, checked, timings = [], set(), {mode: [] for mode in ("vision", "text", "audio")}
    for i, task in enumerate(tasks):
        row, mode = task["row"], task["mode"]
        verify_media(c, row, mode, checked)
        h, logits, seconds = policy.extract(row, c["data_root"], mode, cached=True)
        raw_h, raw_l, _ = policy.extract(row, c["data_root"], mode, cached=False)
        if not np.allclose(h, raw_h, atol=.01, rtol=1e-4) or not np.allclose(logits, raw_l, atol=.01, rtol=1e-4):
            raise RuntimeError("Native and cached-perception evidence paths differ: "+mode)
        changed = {**row, "view_targets": {}, "sample_type": "FORBIDDEN_LABEL", "gt_label": "FORBIDDEN_GOLD",
                   "source_dataset": "FORBIDDEN_SOURCE", "is_clear": False, "natural": False}
        path = Path(c["data_root"])/row["video"]
        if safe_input(row, mode) != safe_input(changed, mode) or task_messages(row, path, mode) != task_messages(changed, path, mode):
            raise RuntimeError("Annotation metadata entered the evidence input")
        if mode == "vision":
            changed["text_content"] = "Completely different transcript"
            changed["protocol"] = "VA" if row["protocol"] == "VT" else "VT"
            if task_messages(row, path, mode) != task_messages(changed, path, mode):
                raise RuntimeError("Visual branch depends on the second modality")
        save_feature(args.run_dir, c, task, h, logits, seconds)
        restored = feature_entry(args.run_dir, c, task, required=True)[1]
        from .evidence_bottleneck import evidence_probabilities
        if not np.allclose(evidence_probabilities(restored[0][None], restored[1][None]), softmax(logits[None]), atol=1e-12):
            raise RuntimeError("Zero residual did not reproduce native decision probabilities")
        checks.append({"mode": mode, "key": task["key"], "hidden_max_abs": float(np.max(np.abs(h-raw_h))),
            "logit_max_abs": float(np.max(np.abs(logits-raw_l)))})
        timings[mode].append(seconds)
        emit(args.run_dir, "preflight", "checking_base_only_inputs", completed=i+1, total=len(tasks))
    if policy.check_frozen() != before:
        raise RuntimeError("Base parameters changed during feature extraction")
    per_mode = {mode: sum(values)/len(values) for mode, values in timings.items()}
    counts = Counter(t["mode"] for split in ("train", "val") for t in tasks_for(c[split], split))
    estimate = sum(counts[mode]*per_mode[mode] for mode in counts)/2/3600
    result = {"identity_hash": c["identity_hash"], "passed": True, "active_adapter": None,
        "frozen": True, "checks": checks, "seconds_per_mode": per_mode, "tasks_by_mode": dict(counts),
        "estimated_feature_hours_two_gpus": estimate, "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
        "note": "Measured isolated VT/VA inputs; shared-GPU contention, initialization, fitting and question generation add time."}
    write_json(Path(args.run_dir)/"preflight.json", result)
    emit(args.run_dir, "preflight", "complete", **result)


def cache(args, c):
    if args.split == "test":
        qualified(args.run_dir, c)
    from .evidence_runtime import BaseEvidenceExtractor
    torch_device(args)
    policy = BaseEvidenceExtractor(c["model"], c["settings"])
    before = policy.check_frozen()
    splits = ("test",) if args.split == "test" else ("train", "val")
    tasks = [task for split in splits for task in tasks_for(c[split], split)][args.shard::2]
    checked = set()
    for i, task in enumerate(tasks):
        if feature_entry(args.run_dir, c, task) is None:
            verify_media(c, task["row"], task["mode"], checked)
            hidden, logits, seconds = policy.extract(task["row"], c["data_root"], task["mode"])
            save_feature(args.run_dir, c, task, hidden, logits, seconds)
        emit(args.run_dir, args.role, "extracting_base_evidence", completed=i+1, total=len(tasks), mode=task["mode"])
    if policy.check_frozen() != before:
        raise RuntimeError("Base changed during cache building")
    emit(args.run_dir, args.role, "complete", completed=len(tasks), total=len(tasks))


def fitting(args, c):
    from .evidence_training import development, final_candidate
    bank = FeatureBank(args.run_dir, c, "train")
    output = lambda phase, **v: emit(args.run_dir, args.role, phase, **v)
    if args.seed == 42:
        development(args.run_dir, c, bank, output)
    final_candidate(args.run_dir, c, bank, args.seed, output)
    emit(args.run_dir, args.role, "complete", seed=args.seed)


def question_extractor(args, c):
    torch_device(args)
    from .relation_runtime import RelationExtractor
    from .grounded_store import TaskCache
    old = read(Path(c["source_run"])/"context.json")
    if (old["identity_hash"] != c["identity"]["question_source_identity"] or
            file_hash(Path(c["source_run"])/"cache/complete.json") != c["identity"]["question_source_manifest_sha256"]):
        raise ValueError("Frozen v4 question cache identity changed")
    policy = RelationExtractor(c["model"], c["adapter"], c["settings"])
    return policy, TaskCache(c["source_run"], old, policy.torch)


def validate_outputs(args, c):
    from .metrics import prediction_record
    from .train_modality_relation import frozen_state
    candidate = read(args.candidate)
    if candidate["identity_hash"] != c["identity_hash"] or candidate["seed"] not in (42, 43):
        raise ValueError("Wrong candidate for output validation")
    policy, source = question_extractor(args, c)
    before = frozen_state(policy)
    predictions = []
    for i, (row, p) in enumerate(zip(c["val"], candidate["val_probabilities"])):
        baseline = c["baselines"]["val"][row["sample_id"]]
        answer = policy.answer(row, "val", c, source, baseline, "yes" if p > .5 else "no")
        prediction = {**prediction_record(row, answer), "split_group_id": row["split_group_id"], "conflict_score": p}
        if prediction["pred_emotion"] != baseline["pred_emotion"]:
            raise RuntimeError("Frozen emotion changed during question completion")
        predictions.append(prediction)
        emit(args.run_dir, args.role, "validating_questions", completed=i+1, total=len(c["val"]))
    if len(predictions) != 666 or before != frozen_state(policy):
        raise RuntimeError("Incomplete predictions or modified v4")
    metrics = extra_report(predictions)
    if abs(metrics["score"]-candidate["metrics"]["score"]) > 1e-12:
        raise RuntimeError("Question generation changed decision score")
    metrics.update(emotion_changes=0, output_checks_pending=False)
    candidate.update(metrics=metrics, comparison=comparison(baseline_records(c), predictions))
    write_json(args.candidate.with_suffix(".predictions.json"), predictions)
    write_json(args.candidate, candidate)
    emit(args.run_dir, args.role, "complete", metrics=metrics)


def qualified(run, c):
    primary = read(Path(run)/"results/evidence-seed42.json")
    secondary = read(Path(run)/"results/evidence-seed43.json")
    base = extra_report(baseline_records(c))
    for candidate, seed, gain in ((primary, 42, .015), (secondary, 43, .005)):
        if candidate["identity_hash"] != c["identity_hash"] or candidate["seed"] != seed or not gate(base, candidate["metrics"], gain)["passed"]:
            raise ValueError("Both partition seeds and legal outputs must pass before TEST")
    return primary, secondary


def export(args, c):
    from .evidence_training import predict_evidence
    from .train_modality_relation import frozen_state
    primary, _ = qualified(args.run_dir, c)
    evidence = predict_evidence(FeatureBank(args.run_dir, c, "test"), c["test"], primary["heads"])
    scores = decision_scores(evidence["q"], c["test"], primary["calibration"])
    policy, source = question_extractor(args, c)
    before = frozen_state(policy)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["sample_id", "pred_emotion", "is_uncertain", "clarifying_question"])
    writer.writeheader()
    changes = []
    for i, (row, baseline, p) in enumerate(zip(c["test"], c["baseline_test"], scores)):
        if row["sample_id"] != baseline["sample_id"]:
            raise ValueError("Public baseline/test order changed")
        answer = parse_completion(policy.answer(row, "test", c, source, baseline, "yes" if p > .5 else "no"))
        if answer is None or not question_is_consistent(answer) or answer["pred_emotion"] != baseline["pred_emotion"]:
            raise ValueError("Illegal final CSV prediction")
        writer.writerow({"sample_id": row["sample_id"], **answer})
        if answer["is_uncertain"] != baseline["is_uncertain"]:
            changes.append({"sample_id": row["sample_id"], "before": baseline["is_uncertain"],
                "after": answer["is_uncertain"], "score": float(p)})
        emit(args.run_dir, args.role, "serializing_submission", completed=i+1, total=734)
    if before != frozen_state(policy) or i != 733:
        raise RuntimeError("Wrong test count or v4 modified")
    target = Path(args.run_dir)/"submission.csv"
    content = buffer.getvalue().encode("utf-8")
    if target.exists() and target.read_bytes() != content:
        raise ValueError("Refusing to overwrite a different submission")
    temporary = target.with_suffix(".partial")
    temporary.write_bytes(content)
    temporary.replace(target)
    write_json(Path(args.run_dir)/"submission-evidence.json", {"ids": [r["sample_id"] for r in c["test"]],
        "vision": evidence["vision"].tolist(), "second": evidence["second"].tolist(),
        "disagreement": evidence["q"].tolist(), "scores": scores.tolist()})
    write_json(Path(args.run_dir)/"submission-manifest.json", {"identity_hash": c["identity_hash"],
        "file": str(target), "sha256": file_hash(target), "rows": 734, "changes": changes,
        "baseline_sha256": c["identity"]["baseline_csv_sha256"], "submitted_to_kaggle": False})
    emit(args.run_dir, args.role, "complete", rows=734, changes=len(changes))


def run_jobs(args, jobs, cpu=False):
    active, pending = [], list(jobs)
    slots = ["cpu"] if cpu else args.gpus
    logs = Path(args.run_dir)/"logs"
    logs.mkdir(parents=True, exist_ok=True)
    try:
        while pending or active:
            for slot in slots:
                if not pending or any(x["slot"] == slot for x in active):
                    continue
                command, role, extra = pending.pop(0)
                status = Path(args.run_dir)/(role+".status.json")
                if status.exists():
                    state = read(status)
                    proc = Path(f"/proc/{state.get('pid')}/cmdline")
                    if state.get("phase") not in ("complete", "failed") and proc.exists() and b"rl.train_evidence_bottleneck" in proc.read_bytes():
                        raise RuntimeError("Owned worker still exists: "+role)
                stream = (logs/(role+".log")).open("a", encoding="utf-8")
                cmd = [sys.executable, "-u", "-m", "rl.train_evidence_bottleneck", command,
                    "--run-dir", str(args.run_dir), "--role", role, "--cpu-threads", str(args.cpu_threads)]
                env = {**os.environ, "OMP_NUM_THREADS": str(args.cpu_threads), "MKL_NUM_THREADS": str(args.cpu_threads),
                       "OPENBLAS_NUM_THREADS": str(args.cpu_threads)}
                if cpu:
                    env["CUDA_VISIBLE_DEVICES"] = ""
                    env.pop("BUAA_EVIDENCE_GPU_SLOT", None)
                else:
                    cmd += ["--gpu", slot, "--memory-gib", str(args.memory_gib)]
                    env["BUAA_EVIDENCE_GPU_SLOT"] = str(args.gpus.index(slot))
                process = subprocess.Popen(cmd+extra, stdout=stream, stderr=subprocess.STDOUT, env=env)
                active.append({"slot": slot, "process": process, "log": stream, "role": role})
            for item in active[:]:
                code = item["process"].poll()
                if code is not None:
                    item["log"].close()
                    active.remove(item)
                    if code:
                        marker = Path(args.run_dir)/(item["role"]+".error.json")
                        detail = read(marker)["error"] if marker.exists() else "see worker log"
                        raise RuntimeError(f"{item['role']} exited {code}: {detail}")
            if active:
                time.sleep(1)
    finally:
        for item in active:
            item["process"].terminate()
            try:
                item["process"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                item["process"].kill()
                item["process"].wait()
            item["log"].close()


def finish(run, c, reason):
    primary = read(Path(run)/"results/evidence-seed42.json")
    secondary = Path(run)/"results/evidence-seed43.json"
    value = {"identity_hash": c["identity_hash"], "reason": reason,
        "baseline": extra_report(baseline_records(c)), "primary": primary,
        "replication": read(secondary) if secondary.exists() else None,
        "development": read(Path(run)/"development.json"),
        "confirmed_kaggle_public": .90813, "new_kaggle_score": None,
        "wall_clock_hours": (time.time()-c["started"])/3600,
        "note": "Independent evidence-branch development; v4 emotion remains fixed. No automatic Kaggle submission."}
    write_json(Path(run)/"final-report.json", value)
    emit(run, "pipeline", "complete", reason=reason, score=primary["metrics"]["score"])


def pipeline(args):
    import fcntl
    root = Path(args.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    with (root/".controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c = initialize(args)
        if (root/"final-report.json").exists():
            previous = read(root/"final-report.json")
            if previous["identity_hash"] != c["identity_hash"]:
                raise ValueError("Completed pipeline identity changed")
            emit(root, "pipeline", "complete", reason=previous["reason"], already_completed=True)
            return
        base = extra_report(baseline_records(c))
        write_json(root/"results/baseline.json", {"metrics": base})
        if not (root/"preflight.json").exists():
            emit(root, "pipeline", "base_preflight")
            run_jobs(args, [("preflight", "preflight", [])])
        pre = read(root/"preflight.json")
        if pre["identity_hash"] != c["identity_hash"] or not pre["passed"] or pre["active_adapter"] is not None:
            raise ValueError("Base-only preflight did not pass")
        emit(root, "pipeline", "feature_extraction")
        run_jobs(args, [("cache", f"cache-{i}", ["--shard", str(i)]) for i in (0, 1)])
        publish(root, c, ("train", "val"))
        emit(root, "pipeline", "nested_development_and_seed42")
        run_jobs(args, [("fit", "fit-42", ["--seed", "42"])], cpu=True)
        primary_path = root/"results/evidence-seed42.json"
        primary = read(primary_path)
        if primary["metrics"]["score"]-base["score"] < .015-1e-12:
            finish(root, c, "primary_gate_not_met")
            return
        emit(root, "pipeline", "primary_output_validation")
        run_jobs(args, [("questions", "questions-42", ["--candidate", str(primary_path)])])
        if not gate(base, read(primary_path)["metrics"], .015)["passed"]:
            finish(root, c, "primary_gate_not_met")
            return
        emit(root, "pipeline", "partition_replication_seed43")
        run_jobs(args, [("fit", "fit-43", ["--seed", "43"])], cpu=True)
        secondary_path = root/"results/evidence-seed43.json"
        if read(secondary_path)["metrics"]["score"]-base["score"] < .005-1e-12:
            finish(root, c, "replication_gate_not_met")
            return
        run_jobs(args, [("questions", "questions-43", ["--candidate", str(secondary_path)])])
        if not gate(base, read(secondary_path)["metrics"], .005)["passed"]:
            finish(root, c, "replication_gate_not_met")
            return
        emit(root, "pipeline", "qualified_test_features")
        run_jobs(args, [("cache", f"test-cache-{i}", ["--split", "test", "--shard", str(i)]) for i in (0, 1)])
        publish(root, c, ("test",))
        emit(root, "pipeline", "submission_generation")
        run_jobs(args, [("export", "export", [])])
        finish(root, c, "qualified_submission_ready")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("run", "prepare", "preflight", "cache", "fit", "questions", "export"))
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    for name in ("config", "data-root", "model-path", "sft-adapter-path", "baseline-csv", "source-run", "candidate"):
        p.add_argument("--"+name, type=Path)
    p.add_argument("--gpu", default="auto")
    p.add_argument("--gpus", nargs=2, default=["1", "2"])
    p.add_argument("--memory-gib", type=float, default=32.)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--role", default="worker")
    p.add_argument("--seed", type=int, choices=(42, 43), default=42)
    p.add_argument("--shard", type=int, choices=(0, 1), default=0)
    p.add_argument("--split", choices=("trainval", "test"), default="trainval")
    return p


def main():
    args = parser().parse_args()
    role = "pipeline" if args.command in ("run", "prepare") else "preflight" if args.command == "preflight" else args.role
    try:
        if args.memory_gib != 32 or len(set(args.gpus)) != 2:
            raise ValueError("Use two distinct allocated GPUs, each with the approved 32 GiB cap")
        if args.command == "run":
            pipeline(args)
        elif args.command == "prepare":
            c = initialize(args)
            emit(args.run_dir, "pipeline", "prepared", identity_hash=c["identity_hash"])
        else:
            c = context_for(args.run_dir)
            if args.command == "preflight": preflight(args, c)
            elif args.command == "cache": cache(args, c)
            elif args.command == "fit": fitting(args, c)
            elif args.command == "questions": validate_outputs(args, c)
            elif args.command == "export": export(args, c)
    except Exception as error:
        value = {"error": str(error), "traceback": traceback.format_exc(), "diagnostics": getattr(error, "diagnostics", None)}
        write_json(Path(args.run_dir)/(role+".error.json"), value)
        emit(args.run_dir, role, "failed", error=str(error))
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
