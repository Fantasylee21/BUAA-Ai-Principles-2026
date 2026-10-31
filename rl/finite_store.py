"""Versioned artifacts for finite-action runs, separate from GRPO checkpoints."""
from __future__ import annotations

import json
import math
from pathlib import Path
import uuid

from .experiment import ROOT, artifact_hash, check_disjoint, file_hash, fingerprint, load_records, write_json
from .finite_actions import PREFIXES, VERSION, read_settings
from .state import adapter_digest, capture_rng, software_versions


def source_hash():
    paths = [*sorted((ROOT / "rl").glob("*.py")), ROOT / "sft_prompt.py"]
    return fingerprint({p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8") for p in paths})


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def initialize(run, data_root, model, adapter, config, diagnostic_predictions=None):
    run, data_root = Path(run), Path(data_root).resolve()
    if run.exists() and any(p.name not in ("logs", ".controller.lock", "pipeline.status.json") for p in run.iterdir()):
        raise ValueError("A new experiment needs an empty run directory")
    train, td = load_records(data_root / "train.jsonl")
    val, vd = load_records(data_root / "val.jsonl")
    if td or vd or len(train) != 3354 or len(val) != 666:
        raise ValueError("Expected original 3354 train / 666 validation rows, without duplicates")
    check_disjoint(train, val)
    settings = read_settings(config)
    diagnostic_ids = []
    if diagnostic_predictions:
        previous = [json.loads(line) for line in Path(diagnostic_predictions).read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {r["sample_id"]: r for r in previous}
        if len(by_id) != len(previous) or set(by_id) != {r["sample_id"] for r in train}:
            raise ValueError("Diagnostic predictions must cover exactly the original training set")
        for row in train:
            old = by_id[row["sample_id"]]
            if old["gt_label"] != row["gt_label"] or old["sample_type"] != row["sample_type"]:
                raise ValueError("Diagnostic gold labels differ from original train")
            if (old["pred_emotion"] != row["gt_label"]
                    or old["is_uncertain"] != ("yes" if row["sample_type"] == "Conflict" else "no")):
                diagnostic_ids.append(row["sample_id"])
    media = {}
    for row in train + val:
        name = row["video"]
        path = (data_root / name).resolve()
        if not path.is_relative_to(data_root):
            raise ValueError("Media path escapes dataset")
        if name not in media:
            media[name] = file_hash(path)
    identity = {"algorithm": VERSION, "settings": settings, "prefixes": list(PREFIXES),
                "train_hash": fingerprint(train), "val_hash": fingerprint(val), "media": media,
                "base": artifact_hash(model), "sft": artifact_hash(adapter, adapter=True),
                "diagnostic_ids_hash": fingerprint(diagnostic_ids),
                "source": source_hash(), "software": software_versions()}
    run.mkdir(parents=True, exist_ok=True)
    context = {"identity": identity, "identity_hash": fingerprint(identity),
               "data_root": str(data_root), "model": str(Path(model).resolve()),
               "adapter": str(Path(adapter).resolve()), "train": train, "val": val, "diagnostic_ids": diagnostic_ids}
    write_json(run / "context.json", context)
    for directory in ("cache", "requests", "results", "logs", "checkpoints"):
        (run / directory).mkdir(exist_ok=True)
    return context


def context_for(run):
    c = read_json(Path(run) / "context.json")
    if c["identity_hash"] != fingerprint(c["identity"]) or c["identity"]["algorithm"] != VERSION:
        raise ValueError("Invalid finite-action experiment identity")
    if c["identity"]["source"] != source_hash():
        raise ValueError("Code changed since initialization; create a new run")
    if c["identity"]["software"] != software_versions():
        raise ValueError("Runtime versions changed since initialization")
    if (artifact_hash(c["model"]) != c["identity"]["base"]
            or artifact_hash(c["adapter"], adapter=True) != c["identity"]["sft"]):
        raise ValueError("Base or original v4 weights changed")
    if fingerprint(c["diagnostic_ids"]) != c["identity"]["diagnostic_ids_hash"]:
        raise ValueError("Diagnostic membership changed")
    for name, digest in c["identity"]["media"].items():
        if file_hash(Path(c["data_root"]) / name) != digest:
            raise ValueError("Dataset media changed")
    for split in ("train", "val"):
        if fingerprint(c[split]) != c["identity"][split + "_hash"]:
            raise ValueError("Dataset context was modified")
        actual, duplicates = load_records(Path(c["data_root"]) / (split + ".jsonl"))
        if duplicates or actual != c[split]:
            raise ValueError("Dataset annotations changed since initialization")
    return c


def cache_path(run, sid):
    return Path(run) / "cache" / (fingerprint(sid) + ".pt")


def save_cache(run, row, split, packed, log_probs, torch, identity_hash):
    path = cache_path(run, row["sample_id"])
    temporary = path.with_suffix(".partial")
    torch.save(packed, temporary)
    temporary.replace(path)
    return {"sample_id": row["sample_id"], "split": split, "file": path.name,
            "sha256": file_hash(path), "reference_log_probs": log_probs,
            "identity_hash": identity_hash}


def merge_cache(run, context, shards=2):
    entries = {}
    for shard in range(shards):
        part = read_json(Path(run) / f"cache/shard-{shard}.json")
        if part["identity_hash"] != context["identity_hash"] or part["shard"] != shard:
            raise ValueError("Cache shard belongs to another experiment")
        for entry in part["entries"]:
            sid = entry["sample_id"]
            if sid in entries or entry["identity_hash"] != context["identity_hash"]:
                raise ValueError("Duplicated or incompatible cached record")
            if cache_path(run, sid).name != entry["file"] or file_hash(cache_path(run, sid)) != entry["sha256"]:
                raise ValueError("Corrupt cached feature file")
            lp = entry["reference_log_probs"]
            if len(lp) != 6 or not all(math.isfinite(v) for v in lp) or abs(sum(math.exp(v) for v in lp)-1) > 1e-8:
                raise ValueError("Invalid cached reference probability distribution")
            entries[sid] = entry
    expected = {r["sample_id"]: split for split in ("train", "val") for r in context[split]}
    if {sid: e["split"] for sid, e in entries.items()} != expected:
        raise ValueError("Cache is incomplete or crosses dataset splits")
    manifest = {"schema": "finite_cache_v1", "identity_hash": context["identity_hash"], "entries": entries}
    write_json(Path(run) / "cache/complete.json", manifest)
    return manifest


class Cache:
    def __init__(self, run, context, torch):
        self.run, self.torch = Path(run), torch
        self.manifest = read_json(self.run / "cache/complete.json")
        if (self.manifest["schema"] != "finite_cache_v1"
                or self.manifest["identity_hash"] != context["identity_hash"]):
            raise ValueError("Invalid cache identity")
        self.entries = self.manifest["entries"]
        self.checked = set()

    def load(self, row, split):
        from .finite_runtime import unpack_cache
        sid = row["sample_id"]
        e = self.entries[sid]
        path = cache_path(self.run, sid)
        if e["split"] != split or e["file"] != path.name:
            raise ValueError("Cached sample crossed a split boundary")
        if sid not in self.checked:
            if file_hash(path) != e["sha256"]:
                raise ValueError("Cached feature checksum mismatch")
            self.checked.add(sid)
        packed = self.torch.load(path, map_location="cpu", weights_only=True)
        return unpack_cache(self.torch, packed)


def validate_checkpoint(path, context):
    path = Path(path).resolve()
    m = read_json(path / "complete.json")
    required = {"training_state.pt", "adapter/policy/adapter_model.safetensors", "adapter/policy/adapter_config.json"}
    if (m.get("schema") != VERSION or m.get("identity_hash") != context["identity_hash"]
            or type(m.get("step")) is not int or not 0 <= m["step"] <= 420
            or not all(m.get(k) for k in ("reward_hash", "policy_digest", "reference_digest"))
            or not required <= set(m.get("files", {}))):
        raise ValueError("Wrong algorithm, identity or incomplete checkpoint; GRPO resume is prohibited")
    for name, digest in m["files"].items():
        p = (path / name).resolve()
        if not p.is_relative_to(path) or file_hash(p) != digest:
            raise ValueError("Checkpoint file integrity failure")
    return m


def save_checkpoint(run, policy, optimizer, scheduler, step, context, reward):
    root = Path(run) / "checkpoints"
    dest = root / f"step-{step:06d}"
    digest = adapter_digest(policy, "policy")
    if dest.exists():
        m = validate_checkpoint(dest, context)
        if m["policy_digest"] != digest or m["reward_hash"] != fingerprint(reward):
            raise ValueError("Cannot overwrite a different completed checkpoint")
        return dest
    tmp = root / (dest.name + ".partial-" + uuid.uuid4().hex)
    tmp.mkdir()
    policy.model.save_pretrained(tmp / "adapter", selected_adapters=["policy"], safe_serialization=True)
    policy.processor.save_pretrained(tmp / "processor")
    state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "rng": capture_rng(policy.torch), "step": step, "reward": reward}
    policy.torch.save(state, tmp / "training_state.pt")
    write_json(tmp / "complete.json", {"schema": VERSION, "identity_hash": context["identity_hash"],
               "step": step, "reward_hash": fingerprint(reward), "policy_digest": digest,
               "reference_digest": adapter_digest(policy, "reference"),
               "files": {p.relative_to(tmp).as_posix(): file_hash(p) for p in tmp.rglob("*") if p.is_file()}})
    tmp.rename(dest)
    write_json(Path(run) / "latest.json", {"checkpoint": str(dest), "step": step})
    return dest


def load_policy(policy, checkpoint, context):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    marker = validate_checkpoint(checkpoint, context)
    path = Path(checkpoint) / "adapter/policy"
    set_peft_model_state_dict(policy.model, load_file(str(path / "adapter_model.safetensors")), adapter_name="policy")
    policy.verify_adapter(path, "policy")
    policy._set_trainability()
    if adapter_digest(policy, "policy") != marker["policy_digest"]:
        raise ValueError("Loaded policy hash mismatch")
    if adapter_digest(policy, "reference") != marker["reference_digest"]:
        raise ValueError("Frozen reference was changed")
    return marker
