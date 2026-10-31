"""Immutable evidence identities, deduplicated base-only features and fit markers."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

import numpy as np

from .evidence_bottleneck import (VERSION, feature_key, safe_input, settings_from,
                                  split_design, tasks_for, view_examples)
from .experiment import (ROOT, artifact_hash, check_disjoint, file_hash, fingerprint,
                         load_records, write_json)
from .grounded_store import source_hash_at
from .grounded_uncertainty import baseline_csv, load_training
from .state import software_versions


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def context_for(run):
    c = read(Path(run)/"context.json")
    if c["identity_hash"] != fingerprint(c["identity"]) or c["identity"]["algorithm"] != VERSION:
        raise ValueError("Corrupt or wrong evidence algorithm identity")
    if c["identity"]["source"] != source_hash_at(ROOT):
        raise ValueError("Evidence source code changed after preparation")
    if file_hash(Path(run)/"baseline_090813.csv") != c["identity"]["baseline_csv_sha256"]:
        raise ValueError("Frozen public baseline changed")
    return c


def initialize(args):
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    if (run/"context.json").exists():
        if not args.resume:
            raise ValueError("Existing evidence run requires --resume")
        return context_for(run)
    old_path = Path(args.source_run)/"context.json"
    old = read(old_path)
    if fingerprint(old["identity"]) != old["identity_hash"]:
        raise ValueError("Corrupt frozen-baseline context")
    data = Path(args.data_root).resolve()
    train, val = load_training(data/"train.jsonl"), load_training(data/"val.jsonl")
    test, duplicate = load_records(data/"test_release.jsonl", require_labels=False)
    if (len(train), len(val), len(test)) != (3354, 666, 734) or duplicate:
        raise ValueError("Official splits changed")
    for left, right in ((train, val), (train, test), (val, test)):
        check_disjoint(left, right)
    for split, rows in (("train", train), ("val", val)):
        if rows != old[split]:
            raise ValueError("Original rows or labels changed since baseline")
        for row in rows:
            v, m = row["view_targets"].values()
            if (v["label"] != m["label"]) != (row["sample_type"] == "Conflict"):
                raise ValueError("Original view labels contradict original relation")
    settings = settings_from(args.config)
    for key in ("video_fps", "video_min_frames", "video_max_frames", "video_pixels",
                "audio_sampling_rate", "max_prompt_tokens", "attn_implementation"):
        if settings[key] != old["settings"][key]:
            raise ValueError("Media processing contract changed")
    model, adapter = str(Path(args.model_path).resolve()), str(Path(args.sft_adapter_path).resolve())
    split_hashes = {s: file_hash(data/name) for s, name in
        (("train", "train.jsonl"), ("val", "val.jsonl"), ("test", "test_release.jsonl"))}
    base_hash, v4_hash = artifact_hash(model), artifact_hash(adapter, adapter=True)
    if base_hash != old["identity"]["base"] or v4_hash != old["identity"]["sft"] or split_hashes != old["identity"]["raw_split_hashes"]:
        raise ValueError("Base, v4 or raw data differs from baseline identity")
    frozen_test = baseline_csv(args.baseline_csv, test)
    designs = {str(seed): split_design(train, seed) for seed in (42, 43)}
    counts = {mode: len(view_examples(train, mode)) for mode in ("vision", "text", "audio")}
    if counts != {"vision": 2084, "text": 1730, "audio": 1561}:
        raise ValueError("Expected clear, deduplicated supervision counts changed")
    import importlib.metadata
    software = {**software_versions(), "scipy": importlib.metadata.version("scipy")}
    identity = {"algorithm": VERSION, "source": source_hash_at(ROOT), "settings": settings,
        "base": base_hash, "sft_for_output_only": v4_hash, "raw_split_hashes": split_hashes,
        "media": old["identity"]["media"], "baseline_csv_sha256": file_hash(args.baseline_csv),
        "baseline_context_sha256": file_hash(old_path), "baseline_identity": old["identity_hash"],
        "question_source_identity": old["identity"]["source_identity"],
        "question_source_manifest_sha256": old["identity"]["source_manifest_sha256"],
        "software": software, "fold_designs": {k: v["hash"] for k, v in designs.items()},
        "feature_contract": {"base_adapter": None, "hidden_size": 2048, "modes": ["vision", "text", "audio"],
            "position": "last_token_before_emotion_value", "storage": "float32_npz",
            "compute": "nf4_bfloat16", "label_free_inputs": True}}
    import time
    c = {"identity": identity, "identity_hash": fingerprint(identity), "settings": settings,
        "model": model, "adapter": adapter, "data_root": str(data),
        "baseline_context": str(old_path.resolve()), "source_run": old["source_run"],
        "baselines": old["baselines"], "baseline_test": frozen_test,
        "train": train, "val": val, "test": test, "designs": designs, "clear_counts": counts,
        "started": time.time(), "deadline": None, "training_cutoff": None}
    frozen = run/"baseline_090813.csv"
    shutil.copyfile(args.baseline_csv, frozen)
    frozen.chmod(0o444)
    write_json(run/"context.json", c)
    write_json(run/"folds.json", {"identity_hash": c["identity_hash"], "designs": designs})
    return c


def feature_path(run, task):
    return Path(run)/"features"/task["split"]/(task["key"]+".npz")


def feature_entry(run, c, task, required=False):
    path = feature_path(run, task)
    marker = path.with_suffix(".json")
    if not marker.exists():
        if required:
            raise ValueError("Incomplete evidence feature: "+task["key"])
        return None
    e = read(marker)
    if (e.get("identity_hash"), e.get("key"), e.get("mode"), e.get("split"), e.get("input_hash")) != (
        c["identity_hash"], task["key"], task["mode"], task["split"], fingerprint(safe_input(task["row"], task["mode"]))):
        raise ValueError("Evidence feature crossed model, mode, input or split")
    if e["sha256"] != file_hash(path):
        raise ValueError("Evidence feature checksum failed")
    with np.load(path, allow_pickle=False) as data:
        h, logits = data["hidden"].copy(), data["native"].copy()
    if h.shape != (2048,) or logits.shape != (3,) or h.dtype != np.float32 or logits.dtype != np.float32 or not np.isfinite(h).all() or not np.isfinite(logits).all():
        raise ValueError("Invalid base-only evidence feature geometry/precision")
    return e, (h, logits)


def save_feature(run, c, task, hidden, native, seconds):
    path = feature_path(run, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+".partial-"+uuid.uuid4().hex)
    h, logits = np.asarray(hidden, dtype=np.float32), np.asarray(native, dtype=np.float32)
    try:
        with tmp.open("wb") as stream:
            np.savez(stream, hidden=h, native=logits)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    e = {"identity_hash": c["identity_hash"], "key": task["key"], "mode": task["mode"],
        "split": task["split"], "input_hash": fingerprint(safe_input(task["row"], task["mode"])),
        "file": path.name, "sha256": file_hash(path), "seconds": seconds, "base_adapter": None}
    write_json(path.with_suffix(".json"), e)
    _, (restored_h, restored_l) = feature_entry(run, c, task, required=True)
    if not np.array_equal(h, restored_h) or not np.array_equal(logits, restored_l):
        raise RuntimeError("Evidence disk round trip changed values")


def publish(run, c, splits):
    for split in splits:
        entries = [feature_entry(run, c, t, required=True)[0] for t in tasks_for(c[split], split)]
        write_json(Path(run)/"features"/split/"complete.json", {
            "identity_hash": c["identity_hash"], "entries": entries})


class FeatureBank:
    def __init__(self, run, c, split):
        self.c, self.split, self.values = c, split, {}
        manifest = read(Path(run)/"features"/split/"complete.json")
        tasks = tasks_for(c[split], split)
        if manifest["identity_hash"] != c["identity_hash"] or [e["key"] for e in manifest["entries"]] != [t["key"] for t in tasks]:
            raise ValueError("Published evidence order or model identity changed")
        for task, expected in zip(tasks, manifest["entries"]):
            e, values = feature_entry(run, c, task, required=True)
            if e != expected:
                raise ValueError("Evidence metadata changed after publication")
            self.values[task["key"]] = values

    def get(self, rows, mode):
        values = [self.values[feature_key(self.split, row, mode)] for row in rows]
        if not values:
            raise ValueError("Empty evidence batch")
        return np.stack([v[0] for v in values]), np.stack([v[1] for v in values])


def saved_fit(path, c, signature):
    path = Path(path)
    if not path.exists() or not Path(str(path)+".complete.json").exists():
        return None
    marker = read(str(path)+".complete.json")
    if (marker["identity_hash"], marker["signature"], marker["sha256"]) != (c["identity_hash"], signature, file_hash(path)):
        raise ValueError("Completed convex fit identity changed")
    result = read(path)
    if not result["solver"]["success"]:
        raise ValueError("Unconverged fit cannot be reused")
    return result


def save_fit(path, c, signature, value):
    write_json(path, value)
    write_json(str(path)+".complete.json", {"algorithm": VERSION, "identity_hash": c["identity_hash"],
        "signature": signature, "sha256": file_hash(path)})
