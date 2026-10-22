"""Bounded six-action Omni RL pipeline. Run with python -m rl.train_finite_actions."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

from .experiment import ROOT, file_hash, fingerprint, load_records, write_json
from .finite_actions import ACTIONS, epoch_batches, reward_table, selection, validate_reward
from .finite_store import (Cache, context_for, initialize, load_policy, merge_cache, read_json,
                           save_cache, save_checkpoint, validate_checkpoint)
from .metrics import prediction_record
from .state import adapter_digest, restore_rng


def emit(run, role, **values):
    value = {"time": time.time(), "role": role, **values}
    write_json(Path(run) / (role + ".status.json"), value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def prediction_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def make_policy(args, context, training):
    from .omni_runtime import select_gpu
    device = select_gpu(args.gpu, min_free_memory_gib=args.memory_gib + 2)
    import numpy as np
    import torch
    torch.set_num_threads(args.cpu_threads)
    torch.cuda.set_per_process_memory_fraction(args.memory_gib * 1024 / device["total_mib"], 0)
    random.seed(context["identity"]["settings"]["seed"])
    np.random.seed(context["identity"]["settings"]["seed"])
    torch.manual_seed(context["identity"]["settings"]["seed"])
    torch.cuda.manual_seed_all(context["identity"]["settings"]["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    from .finite_runtime import FinitePolicy
    print(json.dumps({"gpu": device, "allocator_limit_gib": args.memory_gib, "training": training}), flush=True)
    return FinitePolicy(context["model"], context["adapter"], context["identity"]["settings"], training=training)


def cache_shard(args, c):
    from .finite_runtime import pack_cache
    role = f"cache-{args.shard}"
    policy = make_policy(args, c, False)
    torch = policy.torch
    records = [(split, row) for split in ("train", "val") for row in c[split]]
    records = records[args.shard::2]
    entries = []
    journal = args.run_dir / f"cache/shard-{args.shard}.journal.jsonl"
    previous = {e["sample_id"]: e for e in prediction_rows(journal)} if journal.exists() else {}
    checked_media = set()
    started = time.monotonic()
    for index, (split, row) in enumerate(records, 1):
        # Check media content before using either existing or new cache entries.
        name = row["video"]
        if name not in checked_media:
            if file_hash(Path(c["data_root"]) / name) != c["identity"]["media"][name]:
                raise ValueError("Input video changed after run initialization")
            checked_media.add(name)
        old = previous.get(row["sample_id"])
        if old is not None:
            from .finite_store import cache_path
            if (old["identity_hash"] != c["identity_hash"] or old["split"] != split
                    or file_hash(cache_path(args.run_dir, row["sample_id"])) != old["sha256"]):
                raise ValueError("Cannot resume an incompatible cache shard")
            entries.append(old)
            continue
        prefix = policy.prepare(row, c["data_root"])
        features = policy.perception(prefix)
        with torch.no_grad():
            scores = policy.scores(prefix, features, adapter="reference")
        log_probs = scores.double().log_softmax(-1).tolist()
        entry = save_cache(args.run_dir, row, split, pack_cache(torch, prefix, features),
                           log_probs, torch, c["identity_hash"])
        append(journal, entry)
        entries.append(entry)
        del prefix, features, scores
        if index % 10 == 0 or index == len(records):
            emit(args.run_dir, role, phase="reference_features", completed=index, total=len(records),
                 elapsed_seconds=time.monotonic()-started)
    write_json(args.run_dir / f"cache/shard-{args.shard}.json",
               {"shard": args.shard, "identity_hash": c["identity_hash"], "entries": entries})
    emit(args.run_dir, role, phase="complete", completed=len(records), total=len(records))


def evaluate(policy, c, cache, run, name, *, free=False, checkpoint=None):
    from .finite_reports import report
    result_path = run / f"results/{name}.json"
    if result_path.exists():
        previous = read_json(result_path)
        if previous["identity_hash"] != c["identity_hash"] or previous["checkpoint"] != checkpoint:
            raise ValueError("Stale evaluation result")
        if file_hash(run / previous["predictions"]) != previous["predictions_sha256"]:
            raise ValueError("Evaluation predictions were modified")
        return previous
    rows = []
    out = run / f"results/{name}.predictions.jsonl"
    temporary = out.with_suffix(".partial.jsonl")
    with temporary.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(c["val"], 1):
            prefix, features = cache.load(row, "val")
            if free:
                with policy.cached_perception(features):
                    _, completion = policy.generate(prefix, adapter="reference")
                probabilities = None
            else:
                action = None
                if checkpoint is None:
                    # The new zero-update baseline uses the very same cached
                    # reference distribution which constrains training.
                    ref = cache.entries[row["sample_id"]]["reference_log_probs"]
                    action = max(range(6), key=ref.__getitem__)
                completion, probabilities = policy.answer(prefix, features, action=action,
                                                         adapter="reference" if checkpoint is None else "policy")
            record = prediction_record(row, completion)
            record["split_group_id"] = row["split_group_id"]
            record["probabilities"] = probabilities
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows.append(record)
            del prefix, features
            if index % 10 == 0 or index == len(c["val"]):
                emit(run, "worker", phase=name, completed=index, total=len(c["val"]))
    temporary.replace(out)
    result = {"identity_hash": c["identity_hash"], "checkpoint": checkpoint,
              "predictions": str(out.relative_to(run)), "predictions_sha256": file_hash(out),
              "metrics": report(rows)}
    write_json(result_path, result)
    return result


def queue(run, name, kind, **values):
    value = {"kind": kind, **values}
    path = run / f"requests/{name}.json"
    if path.exists() and read_json(path) != value:
        raise ValueError("Request name collision")
    write_json(path, value)


def wait_result(run, name):
    path = run / f"results/{name}.json"
    while not path.exists():
        if (run / "worker.error.json").exists():
            raise RuntimeError("Evaluation worker failed; see worker.error.json")
        time.sleep(5)
    return read_json(path)


def leaderboard(run, c, *, persist=True):
    baselines = {kind: read_json(run / f"results/baseline-{kind}.json")["metrics"] for kind in ("free", "finite")}
    entries = []
    for path in sorted((run / "results").glob("eval-*.json")):
        result = read_json(path)
        if result["identity_hash"] != c["identity_hash"]:
            raise ValueError("Mismatched evaluation identity")
        entries.append({"name": path.stem, "checkpoint": result["checkpoint"], "metrics": result["metrics"],
                        **selection(baselines, result["metrics"], c["identity"]["settings"]["promotion_gain"])})
    entries.sort(key=lambda e: (-e["metrics"]["score"], e["name"]))
    best = next((e for e in entries if e["promote"]), None)
    value = {"identity_hash": c["identity_hash"], "baselines": baselines, "candidates": entries, "best": best}
    if persist:
        write_json(run / "leaderboard.json", value)
    return value


def refresh_background(policy, c, cache, run, checkpoint):
    marker = load_policy(policy, checkpoint, c)
    predictions, diagnostics = {}, []
    for index, row in enumerate(c["train"], 1):
        prefix, features = cache.load(row, "train")
        with policy.torch.no_grad():
            scores = policy.scores(prefix, features)
        action = int(scores.argmax())
        predictions[row["sample_id"]] = ACTIONS[action][0]
        diagnostics.append({"sample_id": row["sample_id"], "gt_label": row["gt_label"],
                            "sample_type": row["sample_type"], "action": list(ACTIONS[action])})
        del prefix, features, scores
        if index % 10 == 0:
            emit(run, "worker", phase="refresh_train_background", completed=index, total=len(c["train"]))
    table = reward_table(c["train"], predictions, epoch=1, parent_policy=marker["policy_digest"],
                         train_hash=c["identity"]["train_hash"])
    write_json(run / "train-background-epoch1.json", diagnostics)
    write_json(run / "reward-epoch1.json", table)
    write_json(run / "results/refresh-epoch1.json", {"reward_hash": fingerprint(table),
               "parent_policy": marker["policy_digest"], "checkpoint": checkpoint})


def export_best(policy, c, run, best):
    """Only a promoted validation candidate may produce the one test CSV."""
    checkpoint = best["checkpoint"]
    load_policy(policy, checkpoint, c)
    test, repeated = load_records(Path(c["data_root"]) / "test_release.jsonl", require_labels=False)
    if repeated or len(test) != 734:
        raise ValueError("Expected all 734 official test inputs")
    from .reward import parse_completion, question_is_consistent
    output = run / "submission.csv"
    temporary = output.with_suffix(".partial.csv")
    fields = ("sample_id", "pred_emotion", "is_uncertain", "clarifying_question")
    records = []
    with temporary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(test, 1):
            prefix = policy.prepare(row, c["data_root"])
            features = policy.perception(prefix)
            answer, probabilities = policy.answer(prefix, features)
            parsed = parse_completion(answer)
            if parsed is None or not question_is_consistent(parsed):
                raise ValueError(f"Invalid test question/output: {row['sample_id']}; CSV not published")
            writer.writerow({"sample_id": row["sample_id"], **parsed})
            records.append({"sample_id": row["sample_id"], "probabilities": probabilities, **parsed})
            del prefix, features
            if index % 10 == 0:
                emit(run, "worker", phase="selected_test_submission", completed=index, total=734)
    temporary.replace(output)
    write_json(run / "test-predictions.json", records)
    write_json(run / "submission-manifest.json", {"checkpoint": checkpoint, "rows": len(records),
               "test_input_hash": fingerprint(test), "csv_sha256": file_hash(output), "uploaded": False})


def worker(args, c):
    from .finite_reports import comparison
    run = args.run_dir
    policy = make_policy(args, c, False)
    cache = Cache(run, c, policy.torch)
    reference_digest = adapter_digest(policy, "reference")
    for kind in ("free", "finite"):
        evaluate(policy, c, cache, run, "baseline-" + kind, free=kind == "free")
    baseline_rows = {k: prediction_rows(run / f"results/baseline-{k}.predictions.jsonl") for k in ("free", "finite")}
    while True:
        requests = sorted((run / "requests").glob("*.json"))
        for path in requests:
            if (run / f"results/{path.stem}.json").exists():
                continue
            request = read_json(path)
            checkpoint = request["checkpoint"]
            if request["kind"] == "evaluate":
                load_policy(policy, checkpoint, c)
                result = evaluate(policy, c, cache, run, path.stem, checkpoint=checkpoint)
                rows = prediction_rows(run / result["predictions"])
                result["vs_baselines"] = {k: comparison(b, rows) for k, b in baseline_rows.items()}
                write_json(run / f"results/{path.stem}.json", result)
                leaderboard(run, c)
            elif request["kind"] == "refresh":
                refresh_background(policy, c, cache, run, checkpoint)
            else:
                raise ValueError("Unknown worker request")
            if adapter_digest(policy, "reference") != reference_digest:
                raise RuntimeError("Reference adapter changed")
        if (run / "train.done.json").exists():
            board = leaderboard(run, c)
            if board["best"] and not (run / "submission-manifest.json").exists():
                export_best(policy, c, run, board["best"])
            emit(run, "worker", phase="complete", best=board["best"])
            return
        if (run / "train.error.json").exists():
            raise RuntimeError("Training failed; evaluation is stopping")
        time.sleep(5)


def evaluation_steps(n, every):
    # Avoid a redundant 200-step evaluation immediately before the 210-step
    # epoch boundary. The agreed schedule is 50, 100, 150, 210 per epoch.
    return [*[i for i in range(every, n, every) if n-i >= every/2], n]


def train(args, c):
    from transformers import get_cosine_schedule_with_warmup
    run, settings = args.run_dir, c["identity"]["settings"]
    policy = make_policy(args, c, True)
    torch = policy.torch
    cache = Cache(run, c, torch)
    parameters = [p for p in policy.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"], weight_decay=0)
    nsteps = math.ceil(len(c["train"]) / settings["batch_size"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, settings["warmup_steps"], nsteps*2)
    step = 0
    ref_digest = adapter_digest(policy, "reference")
    if (run / "latest.json").exists():
        latest = read_json(run / "latest.json")
        marker = load_policy(policy, latest["checkpoint"], c)
        # This pickle is produced locally by this run and verified by the full
        # checkpoint manifest before loading optimizer/RNG state.
        state = torch.load(Path(latest["checkpoint"]) / "training_state.pt", map_location="cpu", weights_only=False)
        step = state["step"]
        if step != marker["step"] or fingerprint(state["reward"]) != marker["reward_hash"]:
            raise ValueError("Checkpoint training-state mismatch")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(torch, state["rng"])
    checkpoint = None
    for epoch in range(2):
        if epoch == 0:
            parent = ref_digest
        else:
            # Epoch 1 rewards always come from the end-of-epoch-0 policy.
            checkpoint = str(run / f"checkpoints/step-{nsteps:06d}")
            marker = validate_checkpoint(checkpoint, c)
            parent = marker["policy_digest"]
            queue(run, "refresh-epoch1", "refresh", checkpoint=checkpoint)
            emit(run, "train", phase="waiting_for_train_background", step=step, maximum_steps=nsteps*2)
            wait_result(run, "refresh-epoch1")
        table = read_json(run / f"reward-epoch{epoch}.json")
        validate_reward(table, c["train"], epoch, parent)
        if step and step > epoch*nsteps and step <= (epoch+1)*nsteps:
            latest = read_json(run / "latest.json")
            marker = validate_checkpoint(latest["checkpoint"], c)
            if marker["reward_hash"] != fingerprint(table):
                raise ValueError("Resume reward background differs from checkpoint")
        batches = epoch_batches(c["train"], settings["seed"], epoch, settings["batch_size"])
        scheduled = evaluation_steps(nsteps, settings["eval_every"])
        started, processed = time.monotonic(), 0
        for batch_index, rows in enumerate(batches):
            current = epoch*nsteps + batch_index + 1
            if current <= step:
                if batch_index+1 in scheduled:
                    queue(run, f"eval-{current:06d}", "evaluate",
                          checkpoint=str(run / f"checkpoints/step-{current:06d}"))
                continue
            optimizer.zero_grad(set_to_none=True)
            metrics = []
            for row in rows:
                prefix, features = cache.load(row, "train")
                metrics.append(policy.backward_actions(prefix, features,
                    cache.entries[row["sample_id"]]["reference_log_probs"], table["rewards"][row["sample_id"]], divisor=len(rows)))
                del prefix, features
            grad = torch.nn.utils.clip_grad_norm_(parameters, settings["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step = current
            processed += len(rows)
            elapsed = time.monotonic()-started
            average = {key: sum(m[key] for m in metrics)/len(metrics) for key in metrics[0] if key != "action"}
            row_metrics = [{"sample_id": r["sample_id"], "sample_type": r["sample_type"],
                            "gt_label": r["gt_label"], "previous_hard_case": r["sample_id"] in c["diagnostic_ids"],
                            "emotion_correct": ACTIONS[m["action"]][0] == r["gt_label"],
                            "uncertainty_correct": ACTIONS[m["action"]][1] == ("yes" if r["sample_type"] == "Conflict" else "no"),
                            **m} for r, m in zip(rows, metrics)]
            record = {"step": step, "epoch": epoch, "batch_index": batch_index, "batch_size": len(rows),
                      "learning_rate": scheduler.get_last_lr()[0], "grad_norm": float(grad),
                      "reward_hash": fingerprint(table), "mean": average, "samples": row_metrics}
            append(run / "train-metrics.jsonl", record)
            emit(run, "train", phase="training", step=step, maximum_steps=nsteps*2, epoch=epoch+1,
                 epoch_updates=nsteps, learning_rate=record["learning_rate"], **average,
                 inputs_per_second=processed/elapsed,
                 estimated_epoch_remaining_hours=max(0, len(c["train"])-(batch_index+1)*settings["batch_size"])*elapsed/max(1,processed)/3600)
            if batch_index+1 in scheduled:
                if adapter_digest(policy, "reference") != ref_digest:
                    raise RuntimeError("Reference weights changed during training")
                checkpoint = save_checkpoint(run, policy, optimizer, scheduler, step, c, table)
                queue(run, f"eval-{step:06d}", "evaluate", checkpoint=str(checkpoint))
        emit(run, "train", phase="waiting_for_validation", step=step, maximum_steps=nsteps*2)
        for local_step in scheduled:
            wait_result(run, f"eval-{epoch*nsteps+local_step:06d}")
        for kind in ("free", "finite"):
            wait_result(run, f"baseline-{kind}")
        board = leaderboard(run, c, persist=False)
        if epoch == 0 and board["best"] is None:
            write_json(run / "train.done.json", {"step": step, "reason": "no_valid_gain_at_least_0.005_vs_both_baselines"})
            emit(run, "train", phase="complete", step=step, reason="promotion_gate_not_met")
            return
    write_json(run / "train.done.json", {"step": step, "reason": "two_epoch_ceiling"})
    emit(run, "train", phase="complete", step=step, reason="two_epoch_ceiling")


def controller(args):
    run = args.run_dir
    if (run / "context.json").exists():
        if not args.resume:
            raise ValueError("Use --resume for an existing run")
        c = context_for(run)
    else:
        emit(run, "pipeline", phase="checking_data_and_model_hashes", gpus=args.gpus)
        c = initialize(run, args.data_root, args.model_path, args.sft_adapter_path, args.config, args.diagnostic_predictions)
    children = []
    handles = []
    def launch(role, gpu, *extra):
        log = (run / f"logs/{role}{'-'+extra[-1] if role == 'cache' else ''}.log").open("a", encoding="utf-8")
        handles.append(log)
        command = [sys.executable, "-u", "-m", "rl.train_finite_actions", role, "--run-dir", str(run),
                   "--gpu", gpu, "--memory-gib", str(args.memory_gib), "--cpu-threads", str(args.cpu_threads), *extra]
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        children.append(process)
        return process
    def wait_all(processes):
        while any(p.poll() is None for p in processes):
            failed = [p for p in processes if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"Worker exited with code {failed[0].returncode}; see logs")
            time.sleep(3)
        if any(p.returncode for p in processes):
            raise RuntimeError("Worker failed; see logs")
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Controller interrupted; only its own subprocesses will stop")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if not (run / "preflight.json").exists():
            emit(run, "pipeline", phase="preflight", gpus=args.gpus)
            wait_all([launch("preflight", args.gpus[0])])
        if not read_json(run / "preflight.json").get("passed"):
            raise RuntimeError("Numerical preflight did not pass")
        if not (run / "cache/complete.json").exists():
            emit(run, "pipeline", phase="parallel_cache", gpus=args.gpus)
            processes = [launch("cache", gpu, "--shard", str(i)) for i, gpu in enumerate(args.gpus)
                         if not (run / f"cache/shard-{i}.json").exists()]
            wait_all(processes)
            manifest = merge_cache(run, c)
        else:
            manifest = read_json(run / "cache/complete.json")
        if not (run / "reward-epoch0.json").exists():
            predictions = {}
            for row in c["train"]:
                lp = manifest["entries"][row["sample_id"]]["reference_log_probs"]
                predictions[row["sample_id"]] = ACTIONS[max(range(6), key=lp.__getitem__)][0]
            parent = read_json(run / "preflight.json")["reference_digest"]
            write_json(run / "reward-epoch0.json", reward_table(c["train"], predictions, epoch=0,
                parent_policy=parent, train_hash=c["identity"]["train_hash"]))
        # Clear only previous error markers of this explicit resume; preserve
        # their contents in logs and never overwrite model artifacts.
        for role in ("train", "worker"):
            old = run / f"{role}.error.json"
            if old.exists():
                old.rename(run / f"{role}.error.{time.time_ns()}.json")
        emit(run, "pipeline", phase="train_and_parallel_validation", gpus=args.gpus)
        processes = [launch("worker", args.gpus[1])]
        if not (run / "train.done.json").exists():
            processes.append(launch("train", args.gpus[0]))
        wait_all(processes)
        emit(run, "pipeline", phase="complete", result=str(run / "leaderboard.json"))
    finally:
        # These handles are exclusively ours. Never enumerate/kill other users.
        for process in children:
            if process.poll() is None:
                process.terminate()
        for process in children:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for handle in handles:
            handle.close()


def dashboard(run):
    while True:
        print("\033[2J\033[HOmni v4 finite-action RL | two GPUs | max 2 epochs", flush=True)
        for role in ("pipeline", "preflight", "cache-0", "cache-1", "train", "worker"):
            path = run / (role + ".status.json")
            if path.exists():
                print(role + ": " + json.dumps(read_json(path), ensure_ascii=False), flush=True)
        if (run / "leaderboard.json").exists():
            board = read_json(run / "leaderboard.json")
            print("Baselines:", {k: v["score"] for k, v in board["baselines"].items()}, flush=True)
            print("Best eligible:", None if not board["best"] else {"name": board["best"]["name"],
                  "score": board["best"]["metrics"]["score"], "gain": board["best"]["score_gain"]}, flush=True)
        print("Logs: tmux window 1 pipeline; 2 training; 3 validation. Ctrl+B D detaches.", flush=True)
        time.sleep(5)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("run", "preflight", "cache", "train", "worker", "dashboard"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "Dataset")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--sft-adapter-path", type=Path)
    parser.add_argument("--diagnostic-predictions", type=Path, help="Prior train-only v4 greedy predictions; diagnostic membership only")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rl/finite_actions_v1.json")
    parser.add_argument("--gpu", default="auto")
    parser.add_argument("--gpus", nargs=2, default=["1", "2"])
    parser.add_argument("--memory-gib", type=float, default=32)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--shard", type=int, choices=(0, 1))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.run_dir = args.run_dir.resolve()
    if args.memory_gib <= 0 or args.cpu_threads < 1 or len(set(args.gpus)) != 2:
        parser.error("Need a positive memory/thread budget and two distinct allocated GPUs")
    if args.role == "cache" and args.shard is None:
        parser.error("cache requires --shard 0 or 1")
    if args.role == "run" and not args.resume and (not args.model_path or not args.sft_adapter_path):
        parser.error("A new run needs --model-path and --sft-adapter-path")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        from .finite_actions import read_settings
        settings = read_settings(args.config)
        train_rows, td = load_records(args.data_root / "train.jsonl")
        val, vd = load_records(args.data_root / "val.jsonl")
        from .experiment import check_disjoint
        check_disjoint(train_rows, val)
        if td or vd or len(train_rows) != 3354 or len(val) != 666:
            raise ValueError("Unexpected original dataset counts")
        print(json.dumps({"settings": settings, "train": len(train_rows), "val": len(val),
                          "updates_per_epoch": math.ceil(len(train_rows)/settings["batch_size"]),
                          "gpus": args.gpus, "memory_gib": args.memory_gib}, indent=2))
        return
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    if args.role == "dashboard":
        dashboard(args.run_dir)
        return
    try:
        if args.role == "run":
            import fcntl
            args.run_dir.mkdir(parents=True, exist_ok=True)
            with (args.run_dir / ".controller.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                controller(args)
        else:
            c = context_for(args.run_dir)
            if args.role == "preflight":
                from .finite_preflight import preflight
                emit(args.run_dir, "preflight", phase="loading_model")
                result = preflight(make_policy(args, c, True), c)
                write_json(args.run_dir / "preflight.json", result)
                emit(args.run_dir, "preflight", phase="passed", **result)
            else:
                {"cache": cache_shard, "train": train, "worker": worker}[args.role](args, c)
    except BaseException as error:
        if args.run_dir.exists():
            write_json(args.run_dir / f"{args.role}.error.json",
                       {"error": str(error), "traceback": traceback.format_exc(), "time": time.time()})
            emit(args.run_dir, "pipeline" if args.role == "run" else args.role, phase="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
