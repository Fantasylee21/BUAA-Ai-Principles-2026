#!/usr/bin/env python3
"""Fork a step-zero failed run, reusing verified caches after the import-order fix.

The original run is preserved. This tool permits exactly the known scheduler
import move, not arbitrary source changes or optimizer/checkpoint migration.
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.experiment import ROOT, file_hash, fingerprint, write_json
from rl.finite_store import read_json, source_hash

OLD = '''def train(args, c):
    from transformers import get_cosine_schedule_with_warmup
    run, settings = args.run_dir, c["identity"]["settings"]
    policy = make_policy(args, c, True)
'''
NEW = '''def train(args, c):
    run, settings = args.run_dir, c["identity"]["settings"]
    policy = make_policy(args, c, True)
    # Some optional scheduler dependencies initialize CUDA during import.
    # Bind the assigned physical GPU before importing that dependency stack.
    from transformers import get_cosine_schedule_with_warmup
'''


def source_texts(root):
    paths = [*sorted((root / "rl").glob("*.py")), root / "sft_prompt.py"]
    return {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in paths}


def verify_import_only_change(before, after):
    name = "rl/train_finite_actions.py"
    if set(before) != set(after) or any(before[p] != after[p] for p in before if p != name):
        raise ValueError("Cache reuse prohibited: another computation source changed")
    if before[name].count(OLD) != 1 or before[name].replace(OLD, NEW, 1) != after[name]:
        raise ValueError("Cache reuse prohibited: change exceeds the scheduler import-order repair")


def repair(old_run, old_source, new_run):
    old_run, old_source, new_run = [Path(p).resolve() for p in (old_run, old_source, new_run)]
    if new_run.exists():
        raise ValueError("Recovery destination must not exist; the original run is never overwritten")
    before, after = source_texts(old_source), source_texts(ROOT)
    verify_import_only_change(before, after)
    context = read_json(old_run / "context.json")
    if context["identity_hash"] != fingerprint(context["identity"]) or fingerprint(before) != context["identity"]["source"]:
        raise ValueError("Original run/source identity mismatch")
    if (old_run / "latest.json").exists() or list((old_run / "checkpoints").glob("*/complete.json")):
        raise ValueError("Only a run with zero completed parameter updates can use this repair")
    metrics = old_run / "train-metrics.jsonl"
    if metrics.exists() and metrics.stat().st_size:
        raise ValueError("Training metrics show updates; refusing step-zero recovery")
    manifest = read_json(old_run / "cache/complete.json")
    if manifest["identity_hash"] != context["identity_hash"] or manifest["schema"] != "finite_cache_v1":
        raise ValueError("Original cache identity mismatch")
    expected = {r["sample_id"]: split for split in ("train", "val") for r in context[split]}
    if {sid: e["split"] for sid,e in manifest["entries"].items()} != expected:
        raise ValueError("Original cache lacks the full training/validation set")
    new_context = copy.deepcopy(context)
    new_context["identity"]["source"] = source_hash()
    new_context["identity_hash"] = fingerprint(new_context["identity"])
    identity = new_context["identity_hash"]
    # Validate every feature artifact before creating the recovery run.
    for index, (sid, entry) in enumerate(manifest["entries"].items(), 1):
        path = old_run / "cache" / entry["file"]
        if (entry["file"] != fingerprint(sid)+".pt" or entry["identity_hash"] != context["identity_hash"]
                or file_hash(path) != entry["sha256"]):
            raise ValueError("Cached feature file failed integrity validation")
        if index % 250 == 0:
            print(f"verified cache {index}/{len(expected)}", flush=True)
    new_run.mkdir()
    for name in ("cache", "logs", "requests", "results", "checkpoints"):
        (new_run / name).mkdir()
    new_manifest = copy.deepcopy(manifest)
    new_manifest["identity_hash"] = identity
    linked = copied = 0
    for entry in new_manifest["entries"].values():
        original, destination = old_run / "cache" / entry["file"], new_run / "cache" / entry["file"]
        try:
            os.link(original, destination)
            linked += 1
        except OSError:
            shutil.copyfile(original, destination)
            if file_hash(destination) != entry["sha256"]:
                raise ValueError("Copied feature failed integrity validation")
            copied += 1
        entry["identity_hash"] = identity
    write_json(new_run / "context.json", new_context)
    write_json(new_run / "cache/complete.json", new_manifest)
    for shard in (0,1):
        part = read_json(old_run / f"cache/shard-{shard}.json")
        part["identity_hash"] = identity
        for entry in part["entries"]:
            entry["identity_hash"] = identity
        write_json(new_run / f"cache/shard-{shard}.json", part)
    preflight = read_json(old_run / "preflight.json")
    if not preflight.get("passed"):
        raise ValueError("Original numerical preflight was not complete")
    preflight["reused_from"] = str(old_run)
    preflight["reuse_reason"] = "Only scheduler import moved after GPU binding; all forward/reward code unchanged"
    write_json(new_run / "preflight.json", preflight)
    shutil.copyfile(old_run / "reward-epoch0.json", new_run / "reward-epoch0.json")
    recovery = {"reason": "cuda_initialized_before_gpu_selection", "old_run": str(old_run),
                "old_source": str(old_source), "new_source": str(ROOT),
                "old_identity_hash": context["identity_hash"], "new_identity_hash": identity,
                "old_source_hash": fingerprint(before), "new_source_hash": source_hash(),
                "cached_feature_files_verified": len(expected), "hardlinked": linked, "copied": copied,
                "cache_tensor_bytes_changed": False, "original_run_preserved": True,
                "training_settings_changed": False, "completed_updates_before_recovery": 0}
    write_json(new_run / "recovery.json", recovery)
    print(recovery, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--old-source", type=Path, required=True)
    parser.add_argument("--new-run", type=Path, required=True)
    args = parser.parse_args()
    repair(args.old_run, args.old_source, args.new_run)
