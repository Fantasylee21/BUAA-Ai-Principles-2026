"""Bounded two-GPU evidence-assisted Omni v4 REBEL training and verification."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback

from .experiment import ROOT, file_hash, fingerprint, write_json
from .grounded_store import (TaskCache, context_for, initialize, load_checkpoint, merge_tasks, read,
                             save_checkpoint, save_task, task_key)
from .grounded_uncertainty import (VERSION, VALUES, RobustSampler, eligible, main_prefix, read_rows,
                                  pair_loss, settings_from)
from .metrics import prediction_record
from .finite_reports import report, comparison
from .reward import parse_completion, question_is_consistent
from .state import adapter_digest
from .train_finite_actions import append, emit


def remaining(c):
    return c["deadline"]-time.time()


def check_budget(c):
    if remaining(c) <= 0:
        raise TimeoutError("The approved 24-hour wall-clock budget has ended")


def make_policy(args, c, training):
    # Bind CUDA before importing model/scheduler dependencies.
    from .omni_runtime import select_gpu
    device = select_gpu(args.gpu, min_free_memory_gib=args.memory_gib+2)
    import torch
    import numpy as np
    torch.set_num_threads(args.cpu_threads)
    torch.cuda.set_per_process_memory_fraction(args.memory_gib*1024/device["total_mib"], 0)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    from .grounded_runtime import GroundedPolicy
    print(json.dumps({"gpu": device, "allocator_limit_gib": args.memory_gib,
                      "training": training, "seed": seed}), flush=True)
    return GroundedPolicy(c["model"], c["adapter"], c["settings"], training=training)


def numerical_probe(args, c):
    from .finite_runtime import pack_cache, unpack_cache
    from .finite_preflight import gradient_vector, close_gradient
    policy = make_policy(args, c, True)
    torch = policy.torch
    reference = adapter_digest(policy, "reference")
    policy.snapshot_old()
    checks = []
    for protocol in ("VT", "VA"):
        row = next(r for r in c["train"] if r["protocol"] == protocol)
        baseline = None
        for mode in ("full", "vision", "text" if protocol == "VT" else "audio"):
            check_budget(c)
            raw = policy.prepare_task(row, c["data_root"], mode)
            features = policy.perception(raw)
            cached, features = unpack_cache(torch, pack_cache(torch, raw, features))
            if mode == "full":
                with policy.cached_perception(features):
                    _, completion = policy.generate(cached, adapter="reference")
                baseline = parse_completion(completion)
                if baseline is None or not question_is_consistent(baseline):
                    raise RuntimeError("Frozen v4 emitted an invalid probe baseline")
                emotion = baseline["pred_emotion"]
                target = "yes" if row["sample_type"] == "Conflict" else "no"
            else:
                emotion = None
                target = row["view_targets"][mode]["label"]
            with torch.no_grad():
                original = policy.value_scores(raw, None, mode, emotion)
                reused = policy.value_scores(cached, features, mode, emotion)
                old = policy.value_scores(cached, features, mode, emotion, adapter="old")
            if not torch.allclose(original, reused, atol=.01, rtol=1e-4):
                raise RuntimeError(f"{protocol}/{mode} cached value scores changed")
            if mode == "full" and VALUES[mode][int(reused.argmax())] != baseline["is_uncertain"]:
                raise RuntimeError("Native next-value decoding differs from free-generation probe")
            rewards = torch.tensor([float(v == target) for v in VALUES[mode]], device="cuda:0")
            policy.model.zero_grad(set_to_none=True)
            pair_loss(policy.value_scores(raw, None, mode, emotion), old, rewards).backward()
            direct = gradient_vector(policy)
            policy.model.zero_grad(set_to_none=True)
            pair_loss(policy.value_scores(cached, features, mode, emotion), old, rewards).backward()
            gradient = close_gradient(torch, gradient_vector(policy), direct, f"{protocol}/{mode}")
            checks.append({"protocol": protocol, "mode": mode, "gradient": gradient,
                           "max_score_difference": float((original-reused).abs().max())})
            policy.model.zero_grad(set_to_none=True)
            emit(args.run_dir, "probe", phase="numerical_checks", completed=len(checks), total=6)
            del raw, cached, features, direct, original, reused, old
            torch.cuda.empty_cache()
    if adapter_digest(policy, "reference") != reference:
        raise RuntimeError("Numerical probe changed the reference")
    write_json(args.run_dir / "probe.json", {"passed": True, "checks": checks, "reference_digest": reference})
    emit(args.run_dir, "probe", phase="passed", completed=6, total=6)


def cache_shard(args, c):
    from .finite_runtime import pack_cache
    from .finite_store import Cache as LegacyCache
    policy = make_policy(args, c, False)
    torch = policy.torch
    legacy = Path(c["legacy_run"])
    old_context = read(legacy / "context.json")
    source = LegacyCache(legacy, old_context, torch)
    journal = args.run_dir / f"cache/shard-{args.shard}.journal.jsonl"
    previous = {task_key(e["sample_id"], e["mode"]): e for e in read_rows(journal)} if journal.exists() else {}
    selected = [(split, row) for split in ("train", "val") for row in c[split]][args.shard::2]
    entries, checked_media = [], set()
    for index, (split, row) in enumerate(selected, 1):
        check_budget(c)
        if row["video"] not in checked_media:
            if file_hash(Path(c["data_root"]) / row["video"]) != c["identity"]["media"][row["video"]]:
                raise ValueError("Media changed during task preparation")
            checked_media.add(row["video"])
        baseline = None
        for mode in (["full", *row["view_targets"]] if split == "train" else ["full"]):
            key = task_key(row["sample_id"], mode)
            if key in previous:
                entry = previous[key]
                if entry["identity_hash"] != c["identity_hash"] or entry["split"] != split or file_hash(args.run_dir / "cache" / entry["file"]) != entry["sha256"]:
                    raise ValueError("Incompatible partial task cache")
            else:
                if mode == "full":
                    # Reuse verified frozen tensors, never legacy six-action probabilities.
                    prefix, features = source.load(row, split)
                    with policy.cached_perception(features):
                        _, completion = policy.generate(prefix, adapter="reference")
                    baseline = parse_completion(completion)
                    if baseline is None or not question_is_consistent(baseline) or not completion.startswith(main_prefix(baseline["pred_emotion"])):
                        raise RuntimeError(f"Noncanonical frozen baseline: {row['sample_id']}")
                    baseline = {**baseline, "completion": completion}
                else:
                    prefix = policy.prepare_task(row, c["data_root"], mode)
                    features = policy.perception(prefix)
                with torch.no_grad():
                    scores = policy.value_scores(prefix, features, mode,
                        baseline["pred_emotion"] if mode == "full" else None, adapter="reference")
                entry = save_task(args.run_dir, row, split, mode, pack_cache(torch, prefix, features),
                    scores.cpu().tolist(), baseline if mode == "full" else None, torch, c["identity_hash"])
                append(journal, entry)
                del prefix, features, scores
            entries.append(entry)
            if mode == "full":
                baseline = entry["baseline"]
        if index % 5 == 0 or index == len(selected):
            emit(args.run_dir, f"cache-{args.shard}", phase="task_cache", completed=index, total=len(selected), tasks=len(entries))
    write_json(args.run_dir / f"cache/shard-{args.shard}.json", {"identity_hash": c["identity_hash"], "entries": entries})
    emit(args.run_dir, f"cache-{args.shard}", phase="complete", completed=len(selected), total=len(selected))


def zero_check(run, c):
    manifest = read(run / "cache/complete.json")["entries"]
    rows, mismatches = [], []
    for row in c["val"]:
        entry = manifest[task_key(row["sample_id"], "full")]
        baseline = entry["baseline"]
        decision = VALUES["full"][max(range(2), key=entry["reference_scores"].__getitem__)]
        if decision != baseline["is_uncertain"]:
            mismatches.append(row["sample_id"])
        result = prediction_record(row, baseline["completion"])
        result["split_group_id"] = row["split_group_id"]
        rows.append(result)
    output = run / "results/baseline.predictions.jsonl"
    output.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")
    metrics = report(rows)
    write_json(run / "results/baseline.json", {"identity_hash": c["identity_hash"], "metrics": metrics})
    write_json(run / "zero-check.json", {"passed": not mismatches, "mismatches": mismatches,
                                         "emotion_changes": 0, "records": len(rows)})
    if mismatches:
        raise RuntimeError(f"Strict zero-update consistency failed on {len(mismatches)}/666 rows; no training allowed")


def optimizer_for(policy, c):
    from transformers import get_cosine_schedule_with_warmup
    parameters = [p for p in policy.model.parameters() if p.requires_grad]
    optimizer = policy.torch.optim.AdamW(parameters, lr=c["settings"]["learning_rate"], weight_decay=0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, c["settings"]["warmup_steps"], c["settings"]["steps"])
    return optimizer, scheduler, parameters


def update(policy, cache, slots, old_cache, optimizer, scheduler, parameters, c):
    optimizer.zero_grad(set_to_none=True)
    aux_count = sum(target["clear"] for slot in slots for target in slot["row"]["view_targets"].values())
    type_counts = {kind: sum(s["row"]["sample_type"] == kind for s in slots) for kind in ("Conflict", "Aligned")}
    metrics = []
    for slot in slots:
        row = slot["row"]
        baseline = cache.entry(row, "full")["baseline"]
        for mode in ["full", *(m for m, t in row["view_targets"].items() if t["clear"])]:
            prefix, features = cache.load(row, "train", mode)
            emotion = baseline["pred_emotion"] if mode == "full" else None
            key = task_key(row["sample_id"], mode)
            if key not in old_cache:
                with policy.torch.no_grad():
                    old_cache[key] = policy.value_scores(prefix, features, mode, emotion, adapter="old").cpu().tolist()
            if mode == "full":
                target = "yes" if row["sample_type"] == "Conflict" else "no"
                weight = (.6 if row["sample_type"] == "Conflict" else .4)/type_counts[row["sample_type"]]
                kl_weight = c["settings"]["beta"]/len(slots)
                reference = cache.entry(row, mode)["reference_scores"]
            else:
                target = row["view_targets"][mode]["label"]
                weight, kl_weight, reference = c["settings"]["aux_weight"]/max(aux_count, 1), 0., None
            value = policy.backward_task(prefix, features, mode, emotion, target,
                old=old_cache[key], reference=reference, weight=weight, kl_weight=kl_weight)
            metrics.append({"sample_id": row["sample_id"], "draw_id": slot["draw_id"], "selection": slot["selection"],
                            "mode": mode, "target": target, "weight": weight, **value})
            del prefix, features
    grad = policy.torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    main = [m for m in metrics if m["mode"] == "full"]
    return {"loss": sum(m["loss"] for m in metrics), "pair_loss": statistics.mean(m["pair_loss"] for m in main),
            "expected_reward": statistics.mean(m["expected_reward"] for m in main),
            "kl": statistics.mean(m["kl"] for m in main), "grad_norm": float(grad),
            "learning_rate": scheduler.get_last_lr()[0], "main_correct": sum(m["target"] == m["decision"] for m in main),
            "aux_correct": sum(m["target"] == m["decision"] for m in metrics if m["mode"] != "full"),
            "aux_count": aux_count, "samples": metrics}


def smoke_rows(c, cache):
    result = []
    for cohort in ("missed_natural", "false_positive", "easy_aligned", "easy_conflict"):
        for protocol in ("VT", "VA"):
            candidates = []
            for row in c["train"]:
                if row["protocol"] != protocol:
                    continue
                target = "yes" if row["sample_type"] == "Conflict" else "no"
                prediction = cache.entry(row, "full")["baseline"]["is_uncertain"]
                match = ((cohort == "missed_natural" and row["natural"] and target == "yes" and prediction != target)
                    or (cohort == "false_positive" and target == "no" and prediction != target)
                    or (cohort == "easy_aligned" and target == "no" and prediction == target)
                    or (cohort == "easy_conflict" and target == "yes" and prediction == target))
                if match:
                    candidates.append(row)
            if not candidates:
                raise RuntimeError(f"No training-only smoke examples for {cohort}/{protocol}")
            rng = random.Random(int(fingerprint([VERSION, cohort, protocol]), 16))
            selected = rng.sample(candidates, 4) if len(candidates) >= 4 else rng.choices(candidates, k=4)
            result.extend({"row": row, "cohort": cohort} for row in selected)
    return result


def measure_smoke(policy, cache, selected):
    margins, losses = {}, []
    for index, value in enumerate(selected):
        row = value["row"]
        prefix, features = cache.load(row, "train")
        emotion = cache.entry(row, "full")["baseline"]["pred_emotion"]
        with policy.torch.no_grad():
            scores = policy.value_scores(prefix, features, "full", emotion)
            old = policy.value_scores(prefix, features, "full", emotion, adapter="old")
            gold = 0 if row["sample_type"] == "Conflict" else 1
            reward = policy.torch.tensor([float(i == gold) for i in range(2)], device=scores.device)
            losses.append(float(pair_loss(scores, old, reward)))
            margins[str(index)] = float(scores[gold]-scores[1-gold])
        del prefix, features
    return margins, statistics.mean(losses)


def smoke(args, c):
    policy = make_policy(args, c, True)
    cache = TaskCache(args.run_dir, c, policy.torch)
    optimizer, scheduler, parameters = optimizer_for(policy, c)
    policy.snapshot_old()
    reference = adapter_digest(policy, "reference")
    selected = smoke_rows(c, cache)
    initial, initial_loss = measure_smoke(policy, cache, selected)
    old_cache = {}
    started = time.monotonic()
    resume_check = None
    for step in range(1, 21):
        check_budget(c)
        # Two members from each cohort/protocol stratum: exactly 8 C + 8 A.
        offset = (step % 2)*2
        indices = [start+offset+j for start in range(0, 32, 4) for j in (0, 1)]
        random.Random(42+step).shuffle(indices)
        slots = [{"row": selected[i]["row"], "draw_id": f"smoke:{step}:{j}", "selection": selected[i]["cohort"]}
                 for j, i in enumerate(indices)]
        value = update(policy, cache, slots, old_cache, optimizer, scheduler, parameters, c)
        if step == 10:
            checkpoint = save_checkpoint(args.run_dir, policy, optimizer, scheduler, c, seed=42, step=10,
                                         old_block=0, pool={"smoke_ids": [v["row"]["sample_id"] for v in selected]}, smoke=True)
        if step == 11:
            expected = adapter_digest(policy, "policy")
            load_checkpoint(policy, checkpoint, c, optimizer=optimizer, scheduler=scheduler)
            repeated = update(policy, cache, slots, old_cache, optimizer, scheduler, parameters, c)
            resume_check = expected == adapter_digest(policy, "policy") and repeated["loss"] == value["loss"]
            if not resume_check:
                raise RuntimeError("Real-model optimizer/block restoration changed the next update")
        emit(args.run_dir, "smoke", phase="learning_check", step=step, total=20,
             **{k: v for k, v in value.items() if k != "samples"})
    final, final_loss = measure_smoke(policy, cache, selected)
    gain = {cohort: statistics.median(final[str(i)]-initial[str(i)] for i, v in enumerate(selected) if v["cohort"] == cohort)
            for cohort in ("missed_natural", "false_positive")}
    reduction = 1-final_loss/max(initial_loss, 1e-12)
    frozen = adapter_digest(policy, "reference") == reference
    passed = min(gain.values()) >= .25 and reduction >= .1 and frozen and resume_check
    result = {"passed": passed, "median_margin_gain": gain, "main_loss_reduction": reduction,
              "reference_frozen": frozen, "resume_exact": resume_check,
              "seconds_per_update": (time.monotonic()-started)/20,
              "peak_allocated_gib": policy.torch.cuda.max_memory_allocated()/2**30,
              "samples": [{"sample_id": v["row"]["sample_id"], "cohort": v["cohort"]} for v in selected]}
    write_json(args.run_dir / "smoke.json", result)
    emit(args.run_dir, "smoke", phase="passed" if passed else "rejected", **result)


def request(run, name, kind, checkpoint, seed):
    value = {"kind": kind, "checkpoint": str(checkpoint), "seed": seed}
    path = run / f"requests/{name}.json"
    if path.exists() and read(path) != value:
        raise ValueError("Request collision")
    write_json(path, value)


def wait_result(run, name, c):
    path = run / f"results/{name}.json"
    while not path.exists():
        check_budget(c)
        if (run / "worker.error.json").exists():
            raise RuntimeError("Validation worker failed")
        time.sleep(3)
    return read(path)


def candidates(run, c, seed):
    baseline = read(run / "results/baseline.json")["metrics"]
    result = []
    for path in sorted((run / "results").glob(f"seed-{seed}-*-1-eval.json")):
        value = read(path)
        if value["identity_hash"] != c["identity_hash"]:
            raise ValueError("Evaluation belongs to another run")
        result.append({"name": path.stem, "checkpoint": value["checkpoint"], "metrics": value["metrics"],
                       **{stage: eligible(baseline, value["metrics"], stage) for stage in ("early", "primary", "replication")}})
    return sorted(result, key=lambda r: -r["metrics"]["score"])


def train(args, c):
    run = args.run_dir
    policy = make_policy(args, c, True)
    optimizer, scheduler, parameters = optimizer_for(policy, c)
    cache = TaskCache(run, c, policy.torch)
    pool = {r["sample_id"]: cache.entry(r, "full")["baseline"]["is_uncertain"] for r in c["train"]}
    step, old_block, old_cache = 0, -1, {}
    if (run / f"latest-{args.seed}.json").exists():
        latest = read(run / f"latest-{args.seed}.json")
        state = load_checkpoint(policy, latest["checkpoint"], c, optimizer=optimizer, scheduler=scheduler)
        if state["seed"] != args.seed:
            raise ValueError("Cannot resume another seed")
        step, old_block, pool = state["step"], state["old_block"], state["pool"]
    reference = adapter_digest(policy, "reference")
    started = time.monotonic()
    completed_this_launch = 0
    reason = "maximum_steps"
    while step < 200:
        check_budget(c)
        if time.time() >= c["training_cutoff"]:
            reason = "training_time_cutoff"
            break
        if step and step % 40 == 0:
            checkpoint = run / f"checkpoints/seed-{args.seed}/step-{step:06d}"
            stem = f"seed-{args.seed}-{step:06d}"
            request(run, stem+"-1-eval", "evaluate", checkpoint, args.seed)
            request(run, stem+"-0-refresh", "refresh", checkpoint, args.seed)
            emit(run, "train", phase="waiting_for_train_pool", seed=args.seed, step=step, maximum_steps=200)
            refreshed = wait_result(run, stem+"-0-refresh", c)
            pool = refreshed["predictions"]
            if step >= 80:
                wait_result(run, stem+"-1-eval", c)
                if not any(v["early"]["eligible"] for v in candidates(run, c, args.seed)):
                    reason = "early_gate_not_met"
                    break
        block = step//40
        if old_block != block:
            policy.snapshot_old()
            old_block, old_cache = block, {}
        sampler = RobustSampler(c["train"], pool, args.seed)
        slots = sampler.batch(step)
        value = update(policy, cache, slots, old_cache, optimizer, scheduler, parameters, c)
        step += 1
        completed_this_launch += 1
        append(run / f"train-{args.seed}.jsonl", {"seed": args.seed, "step": step, "old_block": old_block,
            "pool_hash": fingerprint(pool), **value})
        elapsed = time.monotonic()-started
        emit(run, "train", phase="training", seed=args.seed, step=step, maximum_steps=200,
             estimated_remaining_hours=(200-step)*elapsed/max(1, completed_this_launch)/3600,
             **{k: v for k, v in value.items() if k != "samples"})
        if step % 20 == 0:
            if adapter_digest(policy, "reference") != reference:
                raise RuntimeError("Frozen reference changed")
            checkpoint = save_checkpoint(run, policy, optimizer, scheduler, c, seed=args.seed,
                                         step=step, old_block=old_block, pool=pool)
            if step % 40 == 0:
                request(run, f"seed-{args.seed}-{step:06d}-1-eval", "evaluate", checkpoint, args.seed)
    if step and step % 20:
        save_checkpoint(run, policy, optimizer, scheduler, c, seed=args.seed, step=step, old_block=old_block, pool=pool)
    write_json(run / f"train-{args.seed}.done.json", {"step": step, "reason": reason,
               "duration_seconds": time.monotonic()-started})
    emit(run, "train", phase="complete", seed=args.seed, step=step, reason=reason)


def worker(args, c):
    run = args.run_dir
    policy = make_policy(args, c, False)
    cache = TaskCache(run, c, policy.torch)
    baseline_rows = read_rows(run / "results/baseline.predictions.jsonl")
    frozen = adapter_digest(policy, "reference")
    while True:
        pending = sorted((run / "requests").glob(f"seed-{args.seed}-*.json"))
        for path in pending:
            out = run / f"results/{path.stem}.json"
            if out.exists():
                continue
            check_budget(c)
            job = read(path)
            marker = load_checkpoint(policy, job["checkpoint"], c)
            if marker["seed"] != args.seed:
                raise ValueError("Worker received another seed's checkpoint")
            if job["kind"] == "refresh":
                predictions = {}
                for i, row in enumerate(c["train"], 1):
                    check_budget(c)
                    prefix, features = cache.load(row, "train")
                    baseline = cache.entry(row, "full")["baseline"]
                    with policy.torch.no_grad():
                        scores = policy.value_scores(prefix, features, "full", baseline["pred_emotion"])
                    predictions[row["sample_id"]] = VALUES["full"][int(scores.argmax())]
                    del prefix, features, scores
                    if i % 50 == 0 or i == len(c["train"]):
                        emit(run, "worker", phase="refresh_training_only", seed=args.seed, completed=i, total=len(c["train"]))
                value = {"predictions": predictions}
            elif job["kind"] == "evaluate":
                rows = []
                for i, row in enumerate(c["val"], 1):
                    check_budget(c)
                    prefix, features = cache.load(row, "val")
                    baseline = cache.entry(row, "full")["baseline"]
                    answer, probabilities = policy.answer_frozen_emotion(prefix, features, baseline)
                    record = prediction_record(row, answer)
                    record.update(split_group_id=row["split_group_id"], probabilities=probabilities)
                    rows.append(record)
                    del prefix, features
                    if i % 20 == 0 or i == len(c["val"]):
                        emit(run, "worker", phase=path.stem, seed=args.seed, completed=i, total=len(c["val"]))
                metrics = report(rows)
                metrics["emotion_changes"] = sum(b["pred_emotion"] != r["pred_emotion"] for b, r in zip(baseline_rows, rows))
                predictions_path = out.with_suffix(".predictions.jsonl")
                predictions_path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")
                value = {"metrics": metrics, "comparison": comparison(baseline_rows, rows),
                         "predictions_sha256": file_hash(predictions_path)}
            else:
                raise ValueError("Unknown worker job")
            if adapter_digest(policy, "reference") != frozen:
                raise RuntimeError("Worker reference changed")
            write_json(out, {"identity_hash": c["identity_hash"], "checkpoint": job["checkpoint"], **value})
            if job["kind"] == "evaluate":
                write_json(run / f"leaderboard-{args.seed}.json", {"seed": args.seed, "candidates": candidates(run, c, args.seed)})
        if (run / f"train-{args.seed}.done.json").exists():
            unresolved = [p for p in (run / "requests").glob(f"seed-{args.seed}-*.json") if not (run / f"results/{p.stem}.json").exists()]
            if not unresolved:
                emit(run, "worker", phase="complete", seed=args.seed)
                return
        if (run / "train.error.json").exists():
            raise RuntimeError("Trainer failed; stopping only this run's worker")
        time.sleep(3)


def export(args, c):
    selected = read(args.run_dir / "selection.json")
    if not selected.get("replication_passed") or not selected.get("primary_passed"):
        raise ValueError("Only a replicated qualified candidate may produce a CSV")
    policy = make_policy(args, c, False)
    load_checkpoint(policy, selected["best"]["checkpoint"], c)
    from .grounded_uncertainty import baseline_csv
    baseline = baseline_csv(args.run_dir / "baseline_090813.csv", c["test"])
    output = args.run_dir / "submission.csv"
    if output.exists():
        raise ValueError("The selected test CSV is generated only once")
    temporary = output.with_suffix(".partial.csv")
    predictions, changes = [], []
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "pred_emotion", "is_uncertain", "clarifying_question"])
        writer.writeheader()
        for i, (row, original) in enumerate(zip(c["test"], baseline), 1):
            check_budget(c)
            if file_hash(Path(c["data_root"]) / row["video"]) != c["identity"]["media"][row["video"]]:
                raise ValueError("Test media changed")
            prefix = policy.prepare_task(row, c["data_root"], "full")
            features = policy.perception(prefix)
            answer, probabilities = policy.answer_frozen_emotion(prefix, features, original)
            parsed = parse_completion(answer)
            if parsed is None or not question_is_consistent(parsed) or parsed["pred_emotion"] != original["pred_emotion"]:
                raise ValueError("Illegal output or frozen emotion changed; CSV not published")
            record = {"sample_id": original["sample_id"], **parsed}
            writer.writerow(record)
            predictions.append({**record, "probabilities": probabilities})
            changes.append({"sample_id": original["sample_id"],
                "pred_emotion": original["pred_emotion"], "emotion_changed": False,
                "old_is_uncertain": original["is_uncertain"], "new_is_uncertain": parsed["is_uncertain"],
                "uncertainty_changed": original["is_uncertain"] != parsed["is_uncertain"],
                "old_question": original["clarifying_question"], "new_question": parsed["clarifying_question"]})
            del prefix, features
            if i % 20 == 0 or i == 734:
                emit(args.run_dir, "worker", phase="selected_test_submission", completed=i, total=734)
    temporary.replace(output)
    write_json(args.run_dir / "test-predictions.json", predictions)
    write_json(args.run_dir / "test-changes.json", changes)
    write_json(args.run_dir / "submission-manifest.json", {"checkpoint": selected["best"]["checkpoint"],
        "csv_sha256": file_hash(output), "baseline_sha256": c["identity"]["baseline_csv_sha256"],
        "rows": 734, "emotion_changes": 0,
        "uncertainty_changes": sum(v["uncertainty_changed"] for v in changes), "uploaded": False})


def controller(args):
    run = args.run_dir
    c = initialize(args)
    children, handles = [], []
    def launch(role, gpu, seed=42, extra=()):
        log_name = f"cache-{extra[1]}" if role == "cache" else role
        log = (run / f"logs/{log_name}.log").open("a", buffering=1)
        handles.append(log)
        command = [sys.executable, "-u", "-m", "rl.train_grounded_uncertainty", role,
                   "--run-dir", str(run), "--gpu", gpu, "--seed", str(seed),
                   "--memory-gib", str(args.memory_gib), "--cpu-threads", str(args.cpu_threads), *extra]
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        children.append(process)
        return process
    def wait(processes):
        while any(p.poll() is None for p in processes):
            check_budget(c)
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("Experiment worker failed; see role.error.json and logs")
            time.sleep(3)
        if any(p.returncode for p in processes):
            raise RuntimeError("Experiment worker exited unsuccessfully")
    try:
        if not (run / "probe.json").exists():
            emit(run, "pipeline", phase="numerical_probe")
            wait([launch("probe", args.gpus[0])])
        if not read(run / "probe.json")["passed"]:
            raise RuntimeError("Numerical preflight was not passed")
        if not (run / "cache/complete.json").exists():
            emit(run, "pipeline", phase="parallel_task_cache", gpus=args.gpus)
            processes = [launch("cache", gpu, extra=("--shard", str(i))) for i, gpu in enumerate(args.gpus)
                         if not (run / f"cache/shard-{i}.json").exists()]
            wait(processes)
            merge_tasks(run, c)
        zero_check(run, c)
        if not (run / "smoke.json").exists():
            emit(run, "pipeline", phase="20_step_learning_and_resume_check")
            wait([launch("smoke", args.gpus[0])])
        if not read(run / "smoke.json")["passed"]:
            emit(run, "pipeline", phase="stopped", reason="learning_check_rejected_no_formal_training")
            return
        for seed in (42, 43):
            if time.time() >= c["training_cutoff"]:
                emit(run, "pipeline", phase="stopped", reason="no_new_training_after_20_hours")
                return
            if seed == 43:
                primary = [v for v in candidates(run, c, 42) if v["primary"]["eligible"]]
                if not primary:
                    emit(run, "pipeline", phase="complete", reason="primary_gate_not_met", submission=False)
                    return
                duration = read(run / "train-42.done.json")["duration_seconds"]
                if remaining(c) < 1.2*duration+3600:
                    emit(run, "pipeline", phase="stopped", reason="insufficient_budget_for_replication_and_export", submission=False)
                    return
            for role in ("worker", "train"):
                path = run / f"{role}.error.json"
                if path.exists():
                    path.rename(run / f"{role}.error.{time.time_ns()}.json")
            emit(run, "pipeline", phase="training_and_validation", seed=seed, maximum_steps=200)
            processes = [launch("worker", args.gpus[1], seed)]
            if not (run / f"train-{seed}.done.json").exists():
                processes.append(launch("train", args.gpus[0], seed))
            wait(processes)
        primary = [v for v in candidates(run, c, 42) if v["primary"]["eligible"]]
        replication = [v for v in candidates(run, c, 43) if v["replication"]["eligible"]]
        options = primary+[v for v in candidates(run, c, 43) if v["primary"]["eligible"]]
        selection = {"primary_passed": bool(primary), "replication_passed": bool(replication),
                     "best": max(options, key=lambda v: v["metrics"]["score"]) if options else None}
        write_json(run / "selection.json", selection)
        if primary and replication:
            emit(run, "pipeline", phase="exporting_selected_model")
            if not (run / "submission-manifest.json").exists():
                wait([launch("export", args.gpus[1])])
        emit(run, "pipeline", phase="complete", submission=(run / "submission-manifest.json").exists())
    except TimeoutError:
        emit(run, "pipeline", phase="stopped", reason="24_hour_wall_clock_budget_exhausted", submission=False)
    finally:
        # Only child handles created by this controller are ever signaled.
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("run", "probe", "cache", "smoke", "train", "worker", "export"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "Dataset")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--sft-adapter-path", type=Path)
    parser.add_argument("--baseline-csv", type=Path)
    parser.add_argument("--legacy-run", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rl/grounded_uncertainty_v1.json")
    parser.add_argument("--gpu", default="auto")
    parser.add_argument("--gpus", nargs=2, default=["1", "2"])
    parser.add_argument("--memory-gib", type=float, default=32)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, choices=(42, 43), default=42)
    parser.add_argument("--shard", type=int, choices=(0, 1))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--budget-start", type=float, help="Preserve an earlier start time when replacing a failed preflight release")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.run_dir = args.run_dir.resolve()
    if args.memory_gib != 32 or args.cpu_threads < 1 or len(set(args.gpus)) != 2:
        parser.error("This experiment uses two distinct GPUs with 32 GiB allocator limits")
    if args.role == "cache" and args.shard is None:
        parser.error("cache requires --shard")
    if args.budget_start is not None and not 0 < args.budget_start <= time.time():
        parser.error("Budget start must be an actual timestamp in the past")
    if args.role == "run" and not args.resume and not all((args.model_path, args.sft_adapter_path, args.baseline_csv, args.legacy_run)):
        parser.error("A new run requires model, v4 adapter, baseline CSV and legacy cache run")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        settings = settings_from(args.config)
        from .grounded_uncertainty import load_training, baseline_csv
        from .experiment import load_records
        training_rows = load_training(args.data_root / "train.jsonl")
        test, _ = load_records(args.data_root / "test_release.jsonl", require_labels=False)
        baseline_csv(args.baseline_csv, test)
        print(json.dumps({"settings": settings, "training_samples": len(training_rows), "test_samples": len(test),
                          "baseline_sha256": file_hash(args.baseline_csv), "gpu_memory_gib_each": args.memory_gib}, indent=2))
        return
    os.environ.update(OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads),
                      TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1")
    try:
        if args.role == "run":
            import fcntl
            args.run_dir.mkdir(parents=True, exist_ok=True)
            with (args.run_dir / ".controller.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                controller(args)
        else:
            c = context_for(args.run_dir)
            emit(args.run_dir, args.role, phase="loading_model", seed=args.seed)
            {"probe": numerical_probe, "cache": cache_shard, "smoke": smoke,
             "train": train, "worker": worker, "export": export}[args.role](args, c)
    except BaseException as error:
        write_json(args.run_dir / f"{args.role}.error.json", {"error": str(error), "traceback": traceback.format_exc(), "time": time.time()})
        emit(args.run_dir, "pipeline" if args.role == "run" else args.role, phase="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
