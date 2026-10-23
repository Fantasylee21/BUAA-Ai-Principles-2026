"""Bounded Omni conflict correction: full epochs, real modality pairs, score gates."""
from __future__ import annotations

import argparse
from collections import Counter
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
from .conflict_cost import (VERSION, VALUES, EpochSampler, main_loss, pair_loss,
                            categorical_kl, choose_baseline, eligible, settings_from)
from .conflict_store import (TaskCache, context_for, initialize, load_checkpoint,
                             merge_tasks, read, save_checkpoint, save_task, task_key)
from .grounded_uncertainty import baseline_csv, read_rows
from .finite_reports import report, comparison
from .metrics import prediction_record
from .reward import parse_completion, question_is_consistent
from .state import adapter_digest
from .train_finite_actions import append, emit


def remaining(c):
    return c["deadline"] - time.time()


def check_budget(c):
    if remaining(c) <= 0:
        raise TimeoutError("The approved 24-hour wall-clock budget has ended")


def make_policy(args, c, training):
    from .grounded_recovery import prepare_environment, configure_torch, EXECUTION
    prepare_environment()
    from .omni_runtime import select_gpu
    device = select_gpu(args.gpu, min_free_memory_gib=args.memory_gib + 2)
    import torch
    import numpy as np
    torch.set_num_threads(args.cpu_threads)
    torch.cuda.set_per_process_memory_fraction(args.memory_gib * 1024 / device["total_mib"], 0)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    configure_torch(torch)
    from .conflict_runtime import ConflictPolicy
    print(json.dumps({"gpu": device, "allocator_limit_gib": args.memory_gib,
        "training": training, "seed": args.seed, "execution": EXECUTION}), flush=True)
    return ConflictPolicy(c["model"], c["adapter"], c["settings"], training=training)


def source_cache(c, torch):
    from .grounded_store import TaskCache as SourceCache
    source = Path(c["source_run"])
    return SourceCache(source, read(source / "context.json"), torch)


def pair_backward(policy, items, settings):
    """Two detached scores followed by two exact chain-rule graph replays."""
    torch = policy.torch
    with torch.no_grad():
        first = [policy.value_scores(p, f, "full", emotion) for p, f, emotion in items]
    leaves = [s.detach().double().requires_grad_(True) for s in first]
    unweighted = pair_loss(*leaves, settings["pair_margin"])
    loss = settings["pair_weight"] * unweighted
    derivatives = torch.autograd.grad(loss, leaves)
    for (prefix, features, emotion), expected, derivative in zip(items, first, derivatives):
        replay = policy.value_scores(prefix, features, "full", emotion)
        if not torch.allclose(replay.detach(), expected, atol=.01, rtol=1e-4):
            raise RuntimeError("Pair scores changed between no-update passes")
        (replay.double() * derivative.detach()).sum().backward()
    return {"pair_loss": float(unweighted.detach()), "pair_contribution": float(loss.detach()),
        "pair_scores": [s.detach().cpu().tolist() for s in first],
        "pair_gap": float((first[0][0]-first[0][1])-(first[1][0]-first[1][1]))}


def numerical_probe(args, c):
    from .finite_runtime import pack_cache, unpack_cache
    from .finite_preflight import gradient_vector, close_gradient
    policy = make_policy(args, c, True)
    torch = policy.torch
    source = source_cache(c, torch)
    reference = adapter_digest(policy, "reference")
    counts = dict(Counter(r["sample_type"] for r in c["train"]))
    checks = []
    for protocol in ("VT", "VA"):
        check_budget(c)
        row = next(r for r in c["train"] if r["protocol"] == protocol)
        original, features = source.load(row, "train")
        raw = policy.prepare_conflict(row, c["data_root"])
        policy.validate_perception_reuse(raw, original, features)
        prefix, features = unpack_cache(torch, pack_cache(torch, raw, features))
        baseline = source.entry(row, "full")["baseline"]
        emotion = baseline["pred_emotion"]
        target = "yes" if row["sample_type"] == "Conflict" else "no"
        with torch.no_grad():
            direct = policy.value_scores(raw, None, "full", emotion)
            cached = policy.value_scores(prefix, features, "full", emotion)
        if not torch.allclose(direct, cached, atol=.01, rtol=1e-4):
            raise RuntimeError("Changed-prompt raw and reused perception scores differ")
        policy.model.zero_grad(set_to_none=True)
        main_loss(policy.value_scores(raw, None, "full", emotion), target, counts, 1).backward()
        direct_gradient = gradient_vector(policy)
        policy.model.zero_grad(set_to_none=True)
        main_loss(policy.value_scores(prefix, features, "full", emotion), target, counts, 1).backward()
        gradient = close_gradient(torch, gradient_vector(policy), direct_gradient, protocol)
        checks.append({"protocol": protocol, "gradient": gradient,
            "max_score_difference": float((direct-cached).abs().max()),
            "new_prompt_tokens": int(prefix["input_ids"].shape[-1]),
            "old_prompt_tokens": int(original["input_ids"].shape[-1])})
        policy.model.zero_grad(set_to_none=True)
        emit(args.run_dir, "probe", phase="raw_cached_gradient_check", completed=len(checks), total=3)
        del raw, prefix, features, original, direct, cached, direct_gradient
        torch.cuda.empty_cache()
    sampler = EpochSampler(c["train"], args.seed)
    pair = sampler.pair(0)
    items = []
    for row in (pair["conflict"], pair["aligned"]):
        original, features = source.load(row, "train")
        raw = policy.prepare_conflict(row, c["data_root"])
        policy.validate_perception_reuse(raw, original, features)
        prefix, features = unpack_cache(torch, pack_cache(torch, raw, features))
        items.append((prefix, features, source.entry(row, "full")["baseline"]["pred_emotion"]))
        del original, raw
    policy.model.zero_grad(set_to_none=True)
    scores = [policy.value_scores(p, f, "full", e) for p, f, e in items]
    (c["settings"]["pair_weight"] * pair_loss(*scores, c["settings"]["pair_margin"])).backward()
    direct_gradient = gradient_vector(policy)
    policy.model.zero_grad(set_to_none=True)
    measured = pair_backward(policy, items, c["settings"])
    gradient = close_gradient(torch, gradient_vector(policy), direct_gradient, "cross_protocol_pair")
    checks.append({"pair_id": pair["pair_id"], "gradient": gradient, **measured})
    if adapter_digest(policy, "reference") != reference:
        raise RuntimeError("Numerical checks changed the frozen reference")
    write_json(args.run_dir / "probe.json", {"passed": True, "checks": checks, "reference_digest": reference})
    emit(args.run_dir, "probe", phase="passed", completed=3, total=3)


def cache_shard(args, c):
    from .finite_runtime import pack_cache, unpack_cache
    policy = make_policy(args, c, False)
    torch = policy.torch
    source = source_cache(c, torch)
    journal = args.run_dir / f"cache/shard-{args.shard}.journal.jsonl"
    previous = {task_key(e["sample_id"], "full"): e for e in read_rows(journal)} if journal.exists() else {}
    selected = [(split, row) for split in ("train", "val") for row in c[split]][args.shard::2]
    entries, checked_media = [], set()
    for index, (split, row) in enumerate(selected, 1):
        check_budget(c)
        key = task_key(row["sample_id"], "full")
        if key in previous:
            entry = previous[key]
            if entry["identity_hash"] != c["identity_hash"] or entry["split"] != split or file_hash(args.run_dir / "cache" / entry["file"]) != entry["sha256"]:
                raise ValueError("Partial conflict cache identity or checksum changed")
        else:
            if row["video"] not in checked_media:
                if file_hash(Path(c["data_root"]) / row["video"]) != c["identity"]["media"][row["video"]]:
                    raise ValueError("Media changed before conflict preprocessing")
                checked_media.add(row["video"])
            original, features = source.load(row, split)
            raw = policy.prepare_conflict(row, c["data_root"])
            policy.validate_perception_reuse(raw, original, features)
            packed = pack_cache(torch, raw, features)
            prefix, features = unpack_cache(torch, packed)
            baseline = source.entry(row, "full")["baseline"]
            with torch.no_grad():
                scores = policy.value_scores(prefix, features, "full", baseline["pred_emotion"], adapter="reference")
            entry = save_task(args.run_dir, row, split, "full", packed, scores.cpu().tolist(),
                              baseline, torch, c["identity_hash"])
            if split == "val":
                answer, probabilities = policy.answer_frozen_emotion(prefix, features, baseline,
                    adapter="reference", question_prefix=original)
                parsed = parse_completion(answer)
                if parsed is None or not question_is_consistent(parsed) or parsed["pred_emotion"] != baseline["pred_emotion"]:
                    raise RuntimeError("New zero-update baseline is not a legal frozen-emotion output")
                entry["zero_completion"] = answer
                entry["zero_probabilities"] = probabilities
            append(journal, entry)
            del original, raw, packed, prefix, features, scores
        entries.append(entry)
        emit(args.run_dir, f"cache-{args.shard}", phase="building_new_prompt_cache", completed=index, total=len(selected))
    write_json(args.run_dir / f"cache/shard-{args.shard}.json", {"identity_hash": c["identity_hash"], "entries": entries})
    emit(args.run_dir, f"cache-{args.shard}", phase="complete", completed=len(selected), total=len(selected))


def save_predictions(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False)+"\n" for r in rows), encoding="utf-8")
    temporary.replace(path)


def zero_baselines(run, c):
    entries = read(run / "cache/complete.json")["entries"]
    free, new = [], []
    for row in c["val"]:
        e = entries[task_key(row["sample_id"], "full")]
        b = e["baseline"]
        old_answer = b.get("completion") or json.dumps({k: b[k] for k in ("pred_emotion", "is_uncertain", "clarifying_question")})
        for records, answer in ((free, old_answer), (new, e["zero_completion"])):
            p = prediction_record(row, answer)
            p["split_group_id"] = row["split_group_id"]
            records.append(p)
        new[-1]["probabilities"] = e["zero_probabilities"]
        if free[-1]["pred_emotion"] != new[-1]["pred_emotion"]:
            raise ValueError("A zero-update baseline changed the frozen emotion")
    values = {}
    for name, rows in (("free", free), ("new", new)):
        metrics = report(rows)
        metrics["emotion_changes"] = 0
        values[name] = metrics
        path = run / f"results/baseline-{name}.predictions.jsonl"
        save_predictions(path, rows)
        write_json(run / f"results/baseline-{name}.json", {"identity_hash": c["identity_hash"],
            "metrics": metrics, "predictions_sha256": file_hash(path)})
    chosen = choose_baseline(values["free"], values["new"])
    name = "new" if chosen is values["new"] else "free"
    write_json(run / "results/baseline.json", {"identity_hash": c["identity_hash"], "selected": name,
        "metrics": chosen, "comparison": comparison(free, new)})
    write_json(run / "zero-check.json", {"passed": True, "records": len(new), "selected": name,
        "uncertainty_changes": sum(a["is_uncertain"] != b["is_uncertain"] for a, b in zip(free, new)),
        "emotion_changes": 0, "requires_uncertainty_equality": False})
    # The unchanged single-modality prompts and frozen v4 scores are diagnostic
    # evidence, not targets for the new conflict training path.
    from .grounded_uncertainty import VALUES as MODAL_VALUES
    original = read(Path(c["source_run"]) / "cache/complete.json")["entries"]
    groups = {}
    for row in c["train"]:
        for mode, target in row["view_targets"].items():
            if not target["clear"]:
                continue
            scores = original[task_key(row["sample_id"], mode)]["reference_scores"]
            correct = MODAL_VALUES[mode][max(range(3), key=scores.__getitem__)] == target["label"]
            for key in (mode, mode+"/"+row["source_dataset"], mode+"/"+row["sample_type"]):
                group = groups.setdefault(key, {"correct": 0, "total": 0})
                group["correct"] += int(correct)
                group["total"] += 1
    write_json(run / "results/unimodal-reference.json", {"identity_hash": c["identity_hash"],
        "groups": groups, "source_identity": c["identity"]["cache_parent"]["identity_hash"],
        "note": "Frozen training-only diagnostics, excluded from the conflict objective."})


def optimizer_for(policy, c):
    from transformers import get_cosine_schedule_with_warmup
    parameters = [p for p in policy.model.parameters() if p.requires_grad]
    optimizer = policy.torch.optim.AdamW(parameters, lr=c["settings"]["learning_rate"], weight_decay=0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, c["settings"]["warmup_steps"], c["settings"]["steps"])
    return optimizer, scheduler, parameters


def update(policy, cache, slots, pair, optimizer, scheduler, parameters, c, *, counts=None):
    torch = policy.torch
    counts = counts or dict(Counter(r["sample_type"] for r in c["train"]))
    optimizer.zero_grad(set_to_none=True)
    metrics = []
    for slot in slots:
        row = slot["row"]
        prefix, features = cache.load(row, "train")
        entry = cache.entry(row, "full")
        target = "yes" if row["sample_type"] == "Conflict" else "no"
        scores = policy.value_scores(prefix, features, "full", entry["baseline"]["pred_emotion"])
        task = main_loss(scores, target, counts, len(slots))
        reference = torch.as_tensor(entry["reference_scores"], device=scores.device)
        kl = categorical_kl(scores, reference)
        loss = task + c["settings"]["beta"] * kl / len(slots)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite conflict correction objective")
        loss.backward()
        metrics.append({"sample_id": row["sample_id"], "draw_id": slot["draw_id"],
            "selection": slot["selection"], "mode": "full", "target": target,
            "main_loss": float(task.detach()), "loss": float(loss.detach()), "kl": float(kl.detach()),
            "decision": VALUES["full"][int(scores.detach().argmax())],
            "expected_reward": float(scores.detach().double().softmax(-1)[VALUES["full"].index(target)]),
            "scores": scores.detach().cpu().tolist()})
        del prefix, features, scores, loss, task, kl, reference
    paired = {"pair_loss": 0., "pair_contribution": 0., "pair_scores": [], "pair_gap": None}
    if pair is not None:
        items = []
        for row in (pair["conflict"], pair["aligned"]):
            prefix, features = cache.load(row, "train")
            items.append((prefix, features, cache.entry(row, "full")["baseline"]["pred_emotion"]))
        paired = pair_backward(policy, items, c["settings"])
        del items, prefix, features
    gradient = torch.nn.utils.clip_grad_norm_(parameters, c["settings"]["max_grad_norm"], error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    return {"loss": sum(m["loss"] for m in metrics) + paired["pair_contribution"],
        "main_loss": sum(m["main_loss"] for m in metrics), **paired,
        "pair_id": pair["pair_id"] if pair else None, "pair_draw_id": pair["draw_id"] if pair else None,
        "expected_reward": statistics.mean(m["expected_reward"] for m in metrics),
        "kl": statistics.mean(m["kl"] for m in metrics), "grad_norm": float(gradient),
        "learning_rate": scheduler.get_last_lr()[0],
        "main_correct": sum(m["target"] == m["decision"] for m in metrics),
        "main_count": len(slots), "samples": metrics}


def smoke_rows(c, cache):
    selected = []
    for cohort in ("missed_conflict", "false_positive", "easy_conflict", "easy_aligned"):
        for protocol in ("VT", "VA"):
            pool = []
            for row in c["train"]:
                if row["protocol"] != protocol:
                    continue
                target = "yes" if row["sample_type"] == "Conflict" else "no"
                scores = cache.entry(row, "full")["reference_scores"]
                decision = VALUES["full"][max(range(2), key=scores.__getitem__)]
                difficult = decision != target
                kind = ("missed_conflict" if difficult else "easy_conflict") if target == "yes" else (
                    "false_positive" if difficult else "easy_aligned")
                if kind == cohort:
                    pool.append(row)
            if len(pool) < 4:
                raise ValueError(f"Need four distinct training samples for {cohort}/{protocol}; gate cannot be fabricated")
            rng = random.Random(int(fingerprint([VERSION, "smoke", cohort, protocol]), 16))
            selected.extend({"row": r, "cohort": cohort} for r in rng.sample(sorted(pool, key=lambda r: r["sample_id"]), 4))
    return selected


def measure_decisions(policy, cache, rows, c, *, role="smoke", adapter="policy"):
    values = []
    for i, row in enumerate(rows, 1):
        check_budget(c)
        prefix, features = cache.load(row, "train")
        with policy.torch.no_grad():
            scores = policy.value_scores(prefix, features, "full", cache.entry(row, "full")["baseline"]["pred_emotion"], adapter=adapter)
        target = "yes" if row["sample_type"] == "Conflict" else "no"
        margin = float(scores[0]-scores[1]) * (1 if target == "yes" else -1)
        decision = VALUES["full"][int(scores.argmax())]
        values.append({"sample_id": row["sample_id"], "target": target, "decision": decision,
            "correct": target == decision, "correct_margin": margin,
            "protocol": row["protocol"], "source_dataset": row["source_dataset"]})
        del prefix, features, scores
        if i % 50 == 0 or i == len(rows):
            emit(cache.run, role, phase="training_only_diagnostic", completed=i, total=len(rows))
    return values


def smoke(args, c):
    from .grounded_recovery import snapshot, compare_snapshots, task_differences
    policy = make_policy(args, c, True)
    cache = TaskCache(args.run_dir, c, policy.torch)
    selected = smoke_rows(c, cache)
    membership = {"identity_hash": c["identity_hash"], "samples": [
        {"sample_id": v["row"]["sample_id"], "cohort": v["cohort"]} for v in selected]}
    selection_path = args.run_dir / "smoke-selection.json"
    if selection_path.exists() and read(selection_path) != membership:
        raise ValueError("Smoke membership changed")
    write_json(selection_path, membership)
    optimizer, scheduler, parameters = optimizer_for(policy, c)
    reference = adapter_digest(policy, "reference")
    initial = measure_decisions(policy, cache, [v["row"] for v in selected], c)
    counts = dict(Counter(v["row"]["sample_type"] for v in selected))
    # The temporary check trains only these 32 rows. Cross-protocol graph replay
    # is independently checked by the real-GPU numerical probe before caching.
    started = time.monotonic()
    resume_check, start_step = False, 0
    checkpoint = args.run_dir / "checkpoints/smoke/step-000010"
    completed = args.run_dir / "checkpoints/smoke/step-000040"
    if (completed / "complete.json").exists():
        if not read(args.run_dir / "recovery-check.json").get("passed"):
            raise ValueError("Completed temporary checkpoint lacks recovery proof")
        load_checkpoint(policy, completed, c, optimizer=optimizer, scheduler=scheduler)
        resume_check, start_step = True, 40
    elif (checkpoint / "complete.json").exists():
        load_checkpoint(policy, checkpoint, c, optimizer=optimizer, scheduler=scheduler)
        start_step = 10
    for step in range(start_step+1, c["settings"]["smoke_steps"] + 1):
        check_budget(c)
        offset = (step % 2) * 2
        indices = [start+offset+j for start in range(0, 32, 4) for j in (0, 1)]
        random.Random(42+step).shuffle(indices)
        slots = [{"row": selected[i]["row"], "draw_id": f"smoke:{step}:{j}", "selection": selected[i]["cohort"]}
                 for j, i in enumerate(indices)]
        if step == 11:
            before = snapshot(policy, optimizer, scheduler)
        tick = time.monotonic()
        value = update(policy, cache, slots, None, optimizer, scheduler, parameters, c, counts=counts)
        append(args.run_dir / "smoke-updates.jsonl", {"step": step,
            "resume_from_step": start_step, "update_seconds": time.monotonic()-tick, **value})
        if step == 10:
            checkpoint = save_checkpoint(args.run_dir, policy, optimizer, scheduler, c, seed=42, step=10, smoke=True)
        if step == 11:
            after = snapshot(policy, optimizer, scheduler, gradients=True)
            load_checkpoint(policy, checkpoint, c, optimizer=optimizer, scheduler=scheduler)
            restored_before = snapshot(policy, optimizer, scheduler)
            repeated = update(policy, cache, slots, None, optimizer, scheduler, parameters, c, counts=counts)
            restored_after = snapshot(policy, optimizer, scheduler, gradients=True)
            diagnosis = {"before": compare_snapshots(policy.torch, before, restored_before),
                "after": compare_snapshots(policy.torch, after, restored_after),
                "tasks": task_differences(value["samples"], repeated["samples"]),
                "original_loss": value["loss"], "restored_loss": repeated["loss"]}
            resume_check = diagnosis["before"]["exact"] and diagnosis["after"]["exact"] and diagnosis["tasks"]["exact"] and value == repeated
            diagnosis["passed"] = resume_check
            write_json(args.run_dir / "recovery-check.json", diagnosis)
            del before, after, restored_before, restored_after
            if not resume_check:
                raise RuntimeError("Exact temporary update recovery failed")
        emit(args.run_dir, "smoke", phase="learning_check", step=step, total=40,
             **{k: v for k, v in value.items() if k != "samples"})
    if not (completed / "complete.json").exists():
        save_checkpoint(args.run_dir, policy, optimizer, scheduler, c, seed=42, step=40, smoke=True)
    by_step = {v["step"]: v for v in read_rows(args.run_dir / "smoke-updates.jsonl")}
    seconds_per_update = statistics.mean(v["update_seconds"] for v in by_step.values())
    final = measure_decisions(policy, cache, [v["row"] for v in selected], c)
    groups = {}
    for cohort in ("missed_conflict", "false_positive", "easy_conflict", "easy_aligned"):
        indices = [i for i, v in enumerate(selected) if v["cohort"] == cohort]
        groups[cohort] = {"total": len(indices), "initial_correct": sum(initial[i]["correct"] for i in indices),
            "final_correct": sum(final[i]["correct"] for i in indices),
            "corrected": sum(not initial[i]["correct"] and final[i]["correct"] for i in indices),
            "regressed": sum(initial[i]["correct"] and not final[i]["correct"] for i in indices)}
    selected_ids = {v["row"]["sample_id"] for v in selected}
    selected_groups = {v["row"]["split_group_id"] for v in selected}
    rest = [r for r in c["train"] if r["sample_id"] not in selected_ids]
    rest_values = measure_decisions(policy, cache, rest, c)
    for value, row in zip(rest_values, rest):
        s = cache.entry(row, "full")["reference_scores"]
        value["baseline_decision"] = VALUES["full"][max(range(2), key=s.__getitem__)]
        value["video_group_used_in_smoke"] = row["split_group_id"] in selected_groups
    save_predictions(args.run_dir / "smoke-remaining-train.predictions.jsonl", rest_values)
    def summarize(values):
        return {"records": len(values), "corrected": sum(v["baseline_decision"] != v["target"] and v["correct"] for v in values),
            "regressed": sum(v["baseline_decision"] == v["target"] and not v["correct"] for v in values),
            "baseline_correct": sum(v["baseline_decision"] == v["target"] for v in values),
            "final_correct": sum(v["correct"] for v in values)}
    frozen = adapter_digest(policy, "reference") == reference
    passed = (all(groups[k]["corrected"] >= 4 for k in ("missed_conflict", "false_positive")) and
        all(groups[k]["regressed"] <= 1 for k in ("easy_conflict", "easy_aligned")) and frozen and resume_check)
    result = {"passed": passed, "groups": groups, "reference_frozen": frozen, "resume_exact": resume_check,
        "seconds_per_update": seconds_per_update, "peak_allocated_gib": policy.torch.cuda.max_memory_allocated()/2**30,
        "remaining_train": summarize(rest_values),
        "remaining_video_groups": summarize([v for v in rest_values if not v["video_group_used_in_smoke"]]),
        "diagnostic_note": "Not independent validation: original v4 SFT has already used the training split.",
        "initial": initial, "final": final, **membership}
    write_json(args.run_dir / "smoke.json", result)
    emit(args.run_dir, "smoke", phase="passed" if passed else "rejected", **{k: v for k, v in result.items() if k not in ("initial", "final", "samples")})


def request(run, name, kind, checkpoint, seed):
    value = {"kind": kind, "checkpoint": str(checkpoint), "seed": seed}
    path = run / f"requests/{name}.json"
    if path.exists() and read(path) != value:
        raise ValueError("Checkpoint request collision")
    write_json(path, value)


def wait_result(run, name, c):
    path = run / f"results/{name}.json"
    while not path.exists():
        check_budget(c)
        if (run / "worker.error.json").exists():
            raise RuntimeError("Validation worker failed")
        time.sleep(3)
    value = read(path)
    if value["identity_hash"] != c["identity_hash"]:
        raise ValueError("Evaluation identity differs")
    return value


def candidates(run, c, seed):
    baseline = read(run / "results/baseline.json")["metrics"]
    result = []
    for path in sorted((run / "results").glob(f"seed-{seed}-*-eval.json")):
        value = read(path)
        if value["identity_hash"] != c["identity_hash"]:
            raise ValueError("Candidate belongs to another experiment")
        result.append({"name": path.stem, "checkpoint": value["checkpoint"], "step": value["step"],
            "metrics": value["metrics"], "comparison": value["comparison"],
            **{stage: eligible(baseline, value["metrics"], stage) for stage in ("early", "primary", "replication")}})
    return sorted(result, key=lambda r: -r["metrics"]["score"])


def coverage(step, sampler):
    n, size = len(sampler.rows), sampler.batch_size
    epoch, position = divmod(step, sampler.epoch_updates)
    return {"epoch": (step-1)//sampler.epoch_updates+1 if step else 0,
        "epoch_seen": min(n, (((step-1) % sampler.epoch_updates)+1)*size) if step else 0,
        "unique_main_seen": min(n, step*size), "main_draws": epoch*n + min(n, position*size),
        "pair_draws": 2*step, "pair_updates": step, "pair_unique_seen": min(step, len(sampler.pairs))}


def train(args, c):
    run = args.run_dir
    policy = make_policy(args, c, True)
    optimizer, scheduler, parameters = optimizer_for(policy, c)
    cache = TaskCache(run, c, policy.torch)
    sampler = EpochSampler(c["train"], args.seed, c["settings"]["batch_size"])
    limit = args.target_steps or c["settings"]["steps"]
    if args.seed == 43:
        if limit != read(run / "primary-selection.json")["step"]:
            raise ValueError("Replication must use the selected primary update count")
    elif limit != c["settings"]["steps"]:
        raise ValueError("Primary schedule must retain its 420-update ceiling")
    step = 0
    if (run / f"latest-{args.seed}.json").exists():
        state = load_checkpoint(policy, read(run / f"latest-{args.seed}.json")["checkpoint"], c,
                                optimizer=optimizer, scheduler=scheduler)
        if state["seed"] != args.seed or state["step"] > limit:
            raise ValueError("Restored seed or data position exceeds this phase")
        step = state["step"]
    reference = adapter_digest(policy, "reference")
    started, launch_step = time.monotonic(), step
    reason = "maximum_steps"
    evaluation_steps = {105, 210, 315, 420} if args.seed == 42 else {limit}
    while step < limit:
        check_budget(c)
        # The 20-hour cutoff prevents NEW epochs, not finishing an epoch already
        # underway. The independent 24-hour deadline remains a strict upper bound.
        if (step == 0 or (args.seed == 42 and step == sampler.epoch_updates)) and time.time() >= c["training_cutoff"]:
            reason = "training_time_cutoff"
            break
        if args.seed == 42 and step == sampler.epoch_updates:
            for evaluated in (105, 210):
                request(run, f"seed-42-{evaluated:06d}-eval", "evaluate",
                        run / f"checkpoints/seed-42/step-{evaluated:06d}", 42)
                emit(run, "train", phase="waiting_first_epoch_validation", seed=42, step=step, maximum_steps=limit)
                wait_result(run, f"seed-42-{evaluated:06d}-eval", c)
            if not any(v["early"]["eligible"] for v in candidates(run, c, 42)):
                reason = "first_epoch_gate_not_met"
                break
        value = update(policy, cache, sampler.batch(step), sampler.pair(step), optimizer, scheduler, parameters, c)
        step += 1
        metadata = coverage(step, sampler)
        append(run / f"train-{args.seed}.jsonl", {"seed": args.seed, "step": step,
            "sampler_identity": sampler.identity, **metadata, **value})
        emit(run, "train", phase="training", seed=args.seed, step=step, maximum_steps=limit,
            target_steps=limit, limit_steps=limit, **metadata,
            estimated_remaining_hours=(limit-step)*(time.monotonic()-started)/max(1, step-launch_step)/3600,
            **{k: v for k, v in value.items() if k != "samples"})
        if step % 20 == 0 or step in evaluation_steps or step == limit:
            if adapter_digest(policy, "reference") != reference:
                raise RuntimeError("Original v4 reference changed")
            checkpoint = save_checkpoint(run, policy, optimizer, scheduler, c, seed=args.seed, step=step)
            if step in evaluation_steps:
                request(run, f"seed-{args.seed}-{step:06d}-eval", "evaluate", checkpoint, args.seed)
    if step and not (run / f"checkpoints/seed-{args.seed}/step-{step:06d}/complete.json").exists():
        save_checkpoint(run, policy, optimizer, scheduler, c, seed=args.seed, step=step)
    # Regenerate a request after interruption between atomic checkpoint and queue write.
    for evaluated in sorted(evaluation_steps):
        checkpoint = run / f"checkpoints/seed-{args.seed}/step-{evaluated:06d}"
        if evaluated <= step and (checkpoint / "complete.json").exists():
            request(run, f"seed-{args.seed}-{evaluated:06d}-eval", "evaluate", checkpoint, args.seed)
    write_json(run / f"train-{args.seed}.done.json", {"step": step, "target_steps": limit, "reason": reason,
        "duration_seconds": time.monotonic()-started, **coverage(step, sampler)})
    emit(run, "train", phase="complete", seed=args.seed, step=step, target_steps=limit, reason=reason)


def assert_score_attribution(baseline, candidate):
    if baseline["macro_f1"] != candidate["macro_f1"]:
        raise ValueError("Frozen emotion Macro-F1 changed")
    before, after = baseline["counts"], candidate["counts"]
    delta = (.3*(after["conflict"]["correct"]-before["conflict"]["correct"])/baseline["n_conflict"]
             -.2*(after["aligned_false_positive"]["correct"]-before["aligned_false_positive"]["correct"])/baseline["n_aligned"])
    if abs((candidate["score"]-baseline["score"])-delta) > 1e-10:
        raise ValueError("Score change does not reconstruct from conflict corrections and false positives")
    return delta


def worker(args, c):
    run = args.run_dir
    policy = make_policy(args, c, False)
    cache = TaskCache(run, c, policy.torch)
    source = source_cache(c, policy.torch)
    baseline_rows = {name: read_rows(run / f"results/baseline-{name}.predictions.jsonl") for name in ("free", "new")}
    baseline_metrics = {name: read(run / f"results/baseline-{name}.json")["metrics"] for name in ("free", "new")}
    frozen = adapter_digest(policy, "reference")
    while True:
        for path in sorted((run / "requests").glob(f"seed-{args.seed}-*.json")):
            out = run / f"results/{path.stem}.json"
            if out.exists():
                continue
            check_budget(c)
            job = read(path)
            marker = load_checkpoint(policy, job["checkpoint"], c)
            if marker["seed"] != args.seed or job["kind"] != "evaluate":
                raise ValueError("Unexpected validation request")
            rows = []
            for i, row in enumerate(c["val"], 1):
                check_budget(c)
                prefix, features = cache.load(row, "val")
                original, _ = source.load(row, "val")
                baseline = cache.entry(row, "full")["baseline"]
                answer, probabilities = policy.answer_frozen_emotion(prefix, features, baseline, question_prefix=original)
                record = prediction_record(row, answer)
                record.update(split_group_id=row["split_group_id"], probabilities=probabilities)
                rows.append(record)
                del prefix, features, original
                if i % 20 == 0 or i == len(c["val"]):
                    emit(run, "worker", phase=path.stem, seed=args.seed, completed=i, total=len(c["val"]))
            metrics = report(rows)
            metrics["emotion_changes"] = sum(b["pred_emotion"] != r["pred_emotion"] for b, r in zip(baseline_rows["free"], rows))
            # Preserve invalid candidates for diagnosis and reject them through gates.
            attribution = None
            if not metrics["emotion_changes"] and not metrics["invalid_outputs"]:
                attribution = {name: assert_score_attribution(base, metrics) for name, base in baseline_metrics.items()}
            predictions_path = out.with_suffix(".predictions.jsonl")
            save_predictions(predictions_path, rows)
            if adapter_digest(policy, "reference") != frozen:
                raise RuntimeError("Validation changed the reference")
            write_json(out, {"identity_hash": c["identity_hash"], "checkpoint": job["checkpoint"], "step": marker["step"],
                "metrics": metrics, "comparison": comparison(baseline_rows["free"], rows),
                "comparison_new_prompt": comparison(baseline_rows["new"], rows),
                "score_attribution": attribution, "predictions_sha256": file_hash(predictions_path)})
            write_json(run / f"leaderboard-{args.seed}.json", {"seed": args.seed, "candidates": candidates(run, c, args.seed)})
        if (run / f"train-{args.seed}.done.json").exists():
            pending = [p for p in (run / "requests").glob(f"seed-{args.seed}-*.json") if not (run / f"results/{p.stem}.json").exists()]
            if not pending:
                emit(run, "worker", phase="complete", seed=args.seed)
                return
        if (run / "train.error.json").exists():
            raise RuntimeError("Trainer failed; stopping this run's validation worker")
        check_budget(c)
        time.sleep(3)


def diagnostics(args, c):
    """Single-modality measurements are diagnostics, never training rewards."""
    from .grounded_uncertainty import VALUES as MODAL_VALUES
    policy = make_policy(args, c, False)
    if args.checkpoint:
        load_checkpoint(policy, args.checkpoint, c)
    source = source_cache(c, policy.torch)
    groups = {}
    for index, row in enumerate(c["train"], 1):
        for mode, target in row["view_targets"].items():
            if not target["clear"]:
                continue
            check_budget(c)
            prefix, features = source.load(row, "train", mode)
            with policy.torch.no_grad():
                scores = policy.value_scores(prefix, features, mode)
            correct = MODAL_VALUES[mode][int(scores.argmax())] == target["label"]
            for key in (mode, mode+"/"+row["source_dataset"], mode+"/"+row["sample_type"]):
                group = groups.setdefault(key, {"correct": 0, "total": 0})
                group["correct"] += int(correct)
                group["total"] += 1
            del prefix, features, scores
        if index % 50 == 0:
            emit(args.run_dir, "diagnostics", phase="unimodal_training_diagnostic", completed=index, total=len(c["train"]))
    write_json(args.run_dir / "results/unimodal-selected.json", {"identity_hash": c["identity_hash"],
        "checkpoint": str(args.checkpoint) if args.checkpoint else None, "groups": groups,
        "note": "Training-only diagnostic; no auxiliary loss or rule-based disagreement decoder."})


def export(args, c):
    selected = read(args.run_dir / "selection.json")
    if not selected.get("primary_passed") or not selected.get("replication_passed"):
        raise ValueError("Both score gates must pass before export")
    policy = make_policy(args, c, False)
    marker = load_checkpoint(policy, selected["best"]["checkpoint"], c)
    if marker["seed"] != 42:
        raise ValueError("Export only the preselected seed-42 candidate")
    baseline = baseline_csv(args.run_dir / "baseline_090813.csv", c["test"])
    output = args.run_dir / "submission.csv"
    if output.exists():
        raise ValueError("Generate the selected submission only once")
    temporary = output.with_suffix(".partial.csv")
    predictions, changes = [], []
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "pred_emotion", "is_uncertain", "clarifying_question"])
        writer.writeheader()
        for i, (row, original) in enumerate(zip(c["test"], baseline), 1):
            check_budget(c)
            if file_hash(Path(c["data_root"]) / row["video"]) != c["identity"]["media"][row["video"]]:
                raise ValueError("Test media changed")
            prefix = policy.prepare_conflict(row, c["data_root"])
            features = policy.perception(prefix)
            original_prefix = policy.original_task(row, c["data_root"])
            policy.validate_perception_reuse(prefix, original_prefix, features)
            answer, probabilities = policy.answer_frozen_emotion(prefix, features, original, question_prefix=original_prefix)
            parsed = parse_completion(answer)
            if parsed is None or not question_is_consistent(parsed) or parsed["pred_emotion"] != original["pred_emotion"]:
                raise ValueError("Illegal selected output; submission is not published")
            record = {"sample_id": original["sample_id"], **parsed}
            writer.writerow(record)
            predictions.append({**record, "probabilities": probabilities})
            changes.append({"sample_id": original["sample_id"], "emotion_changed": False,
                "old_is_uncertain": original["is_uncertain"], "new_is_uncertain": parsed["is_uncertain"],
                "uncertainty_changed": original["is_uncertain"] != parsed["is_uncertain"],
                "old_question": original["clarifying_question"], "new_question": parsed["clarifying_question"]})
            del prefix, features, original_prefix
            if i % 20 == 0 or i == 734:
                emit(args.run_dir, "worker", phase="selected_test_submission", completed=i, total=734)
    # Re-read actual bytes to check the external four-column contract and ordering.
    with temporary.open(encoding="utf-8", newline="") as stream:
        actual = list(csv.DictReader(stream))
    if len(actual) != 734 or any((a["sample_id"], a["pred_emotion"]) != (b["sample_id"], b["pred_emotion"]) for a, b in zip(actual, baseline)):
        raise ValueError("Submission row count, order, or frozen emotions changed")
    temporary.replace(output)
    write_json(args.run_dir / "test-predictions.json", predictions)
    write_json(args.run_dir / "test-changes.json", changes)
    write_json(args.run_dir / "submission-manifest.json", {"checkpoint": selected["best"]["checkpoint"],
        "csv_sha256": file_hash(output), "baseline_sha256": c["identity"]["baseline_csv_sha256"],
        "rows": 734, "emotion_changes": 0, "uncertainty_changes": sum(v["uncertainty_changed"] for v in changes), "uploaded": False})


def controller(args):
    run = args.run_dir
    c = initialize(args)
    if args.resume:
        # Preserve previous failure evidence but do not let stale errors abort
        # newly resumed workers. No checkpoint, cache, or training result is removed.
        for role in ("run", "probe", "cache", "smoke", "train", "worker", "export", "diagnostics"):
            old_error = run / f"{role}.error.json"
            if old_error.exists():
                old_error.rename(run / f"{role}.error.{time.time_ns()}.json")
    children, handles = [], []
    def launch(role, gpu, seed=42, extra=()):
        log_name = f"cache-{extra[1]}" if role == "cache" else role
        handle = (run / f"logs/{log_name}.log").open("a", buffering=1)
        handles.append(handle)
        command = [sys.executable, "-u", "-m", "rl.train_conflict_cost", role,
            "--run-dir", str(run), "--gpu", str(gpu), "--seed", str(seed),
            "--memory-gib", str(args.memory_gib), "--cpu-threads", str(args.cpu_threads), *extra]
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
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
            raise RuntimeError("Numerical preflight failed")
        if not (run / "cache/complete.json").exists():
            emit(run, "pipeline", phase="parallel_task_cache")
            wait([launch("cache", gpu, extra=("--shard", str(i))) for i, gpu in enumerate(args.gpus)
                  if not (run / f"cache/shard-{i}.json").exists()])
            merge_tasks(run, c)
        emit(run, "pipeline", phase="zero_baselines")
        zero_baselines(run, c)
        if not (run / "smoke.json").exists():
            if not (run / "checkpoints/smoke/step-000010/complete.json").exists() and time.time() >= c["training_cutoff"]:
                emit(run, "pipeline", phase="stopped", reason="training_time_cutoff", submission=False)
                return
            emit(run, "pipeline", phase="40_step_learning_and_resume_check")
            wait([launch("smoke", args.gpus[0])])
        if not read(run / "smoke.json")["passed"]:
            emit(run, "pipeline", phase="stopped", reason="smoke_rejected", submission=False)
            return
        for seed in (42, 43):
            limit = c["settings"]["steps"]
            if seed == 43:
                primary = [v for v in candidates(run, c, 42) if v["primary"]["eligible"]]
                if not primary:
                    emit(run, "pipeline", phase="complete", reason="primary_gate_not_met", submission=False)
                    return
                chosen = primary[0]
                write_json(run / "primary-selection.json", chosen)
                limit = chosen["step"]
                duration = read(run / "train-42.done.json")["duration_seconds"]
                if not (run / "train-43.done.json").exists() and remaining(c) < duration*1.2 + 3600:
                    emit(run, "pipeline", phase="stopped", reason="insufficient_replication_budget", submission=False)
                    return
            if (not (run / f"train-{seed}.done.json").exists()
                    and not (run / f"latest-{seed}.json").exists()
                    and time.time() >= c["training_cutoff"]):
                emit(run, "pipeline", phase="stopped", reason="training_time_cutoff", submission=False)
                return
            emit(run, "pipeline", phase="training_and_validation", seed=seed, maximum_steps=limit)
            processes = [launch("worker", args.gpus[1], seed)]
            if not (run / f"train-{seed}.done.json").exists():
                processes.append(launch("train", args.gpus[0], seed, extra=("--target-steps", str(limit))))
            wait(processes)
        primary = read(run / "primary-selection.json")
        replication = [v for v in candidates(run, c, 43) if v["step"] == primary["step"] and v["replication"]["eligible"]]
        selection = {"primary_passed": primary["primary"]["eligible"], "replication_passed": bool(replication),
            "best": primary, "replication": replication[0] if replication else None}
        write_json(run / "selection.json", selection)
        if not selection["replication_passed"]:
            emit(run, "pipeline", phase="complete", reason="replication_gate_not_met", submission=False)
            return
        emit(run, "pipeline", phase="exporting_selected_model")
        if not (run / "submission-manifest.json").exists():
            wait([launch("export", args.gpus[1])])
        if remaining(c) > 3600 and not (run / "results/unimodal-selected.json").exists():
            emit(run, "pipeline", phase="selected_modality_diagnostics")
            wait([launch("diagnostics", args.gpus[1], extra=("--checkpoint", primary["checkpoint"]))])
        emit(run, "pipeline", phase="complete", reason="qualified_submission_ready", submission=True)
    except TimeoutError:
        emit(run, "pipeline", phase="stopped", reason="budget_exhausted", submission=False)
    finally:
        # Never inspect, kill, or signal other users' jobs: these are owned handles.
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
    parser.add_argument("role", choices=("run", "probe", "cache", "smoke", "train", "worker", "export", "diagnostics"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "Dataset")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--sft-adapter-path", type=Path)
    parser.add_argument("--baseline-csv", type=Path)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--source-release", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rl/conflict_cost_v1.json")
    parser.add_argument("--gpu", default="auto")
    parser.add_argument("--gpus", nargs=2, default=["1", "2"])
    parser.add_argument("--memory-gib", type=float, default=32)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, choices=(42, 43), default=42)
    parser.add_argument("--shard", type=int, choices=(0, 1))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--budget-start", type=float)
    parser.add_argument("--target-steps", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.run_dir = args.run_dir.resolve()
    if args.memory_gib != 32 or args.cpu_threads < 1 or len(set(args.gpus)) != 2:
        parser.error("Use distinct allocated GPUs with 32 GiB per-process allocator limits")
    if args.role == "cache" and args.shard is None:
        parser.error("cache requires --shard")
    if args.target_steps is not None and not 1 <= args.target_steps <= 420:
        parser.error("Target updates must be between 1 and 420")
    if args.budget_start is not None and not 0 < args.budget_start <= time.time():
        parser.error("Budget start must be a real earlier timestamp")
    if args.role == "run" and not args.resume and not all((args.model_path, args.sft_adapter_path, args.baseline_csv, args.source_run, args.source_release)):
        parser.error("A new run requires original model/adapter, baseline CSV, source run and immutable source release")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        from .grounded_uncertainty import load_training
        from .experiment import load_records
        rows = load_training(args.data_root / "train.jsonl")
        sampler = EpochSampler(rows, args.seed)
        test, _ = load_records(args.data_root / "test_release.jsonl", require_labels=False)
        baseline_csv(args.baseline_csv, test)
        print(json.dumps({"settings": settings_from(args.config), "training_samples": len(rows),
            "type_counts": sampler.counts, "epoch_updates": sampler.epoch_updates, "real_pairs": len(sampler.pairs),
            "test_samples": len(test), "baseline_sha256": file_hash(args.baseline_csv)}, indent=2))
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
            {"probe": numerical_probe, "cache": cache_shard, "smoke": smoke, "train": train,
             "worker": worker, "export": export, "diagnostics": diagnostics}[args.role](args, c)
    except BaseException as error:
        write_json(args.run_dir / f"{args.role}.error.json", {"error": str(error), "traceback": traceback.format_exc(), "time": time.time()})
        emit(args.run_dir, "pipeline" if args.role == "run" else args.role, phase="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
