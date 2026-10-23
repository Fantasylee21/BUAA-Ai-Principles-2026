"""Immutable cost-sensitive experiments, full-input caches and policy-only state."""
from __future__ import annotations

import math
from pathlib import Path
import shutil
import time
import uuid

from .conflict_cost import VERSION, EpochSampler, settings_from
from .experiment import ROOT, artifact_hash, check_disjoint, file_hash, fingerprint, load_records, write_json
from .grounded_recovery import EXECUTION
from .grounded_store import read, source_hash_at, task_key
from .grounded_uncertainty import baseline_csv, load_training
from .state import adapter_digest, capture_rng, restore_rng, software_versions


# Prompt changes are intentional. These unchanged producer modules determine
# media decoding, frozen encoder features, precision and original-v4 baselines.
PERCEPTION_MODULES = ("rl/omni_io.py", "rl/omni_runtime.py", "rl/finite_runtime.py",
                      "rl/finite_actions.py", "rl/grounded_runtime.py",
                      "rl/grounded_uncertainty.py", "sft_prompt.py")
INPUT_KEYS = ("video_fps", "video_min_frames", "video_max_frames", "video_pixels",
              "audio_sampling_rate", "max_prompt_tokens", "max_completion_length", "attn_implementation")


def code_hash():
    return source_hash_at(ROOT)


def perception_contract(root):
    return {name: fingerprint((Path(root) / name).read_text(encoding="utf-8")) for name in PERCEPTION_MODULES}


def verify_source(source_run, source_release, identity):
    """Permit perception reuse, never reuse old prompt scores as new references."""
    source, release = Path(source_run).resolve(), Path(source_release).resolve()
    old = read(source / "context.json")
    manifest = read(source / "cache/complete.json")
    if old["identity_hash"] != fingerprint(old["identity"]) or manifest["identity_hash"] != old["identity_hash"]:
        raise ValueError("Source cache identity is corrupt")
    if source_hash_at(release) != old["identity"]["source"]:
        raise ValueError("Source release does not match its recorded code")
    if perception_contract(release) != perception_contract(ROOT):
        raise ValueError("Frozen perception or original-v4 prompt contract changed")
    if old["identity"].get("execution") != EXECUTION:
        raise ValueError("Source cache deterministic execution contract changed")
    for key in ("software", "base", "sft", "media", "split_hashes", "raw_split_hashes", "baseline_csv_sha256"):
        if old["identity"].get(key) != identity[key]:
            raise ValueError(f"Source cache data/model/runtime identity differs: {key}")
    if any(old["identity"]["settings"].get(k) != identity["settings"][k] for k in INPUT_KEYS):
        raise ValueError("Source cache media/precision settings differ")
    for split in ("train", "val"):
        for row in old[split]:
            key = task_key(row["sample_id"], "full")
            entry = manifest["entries"].get(key, {})
            if (entry.get("split"), entry.get("sample_id"), entry.get("mode"), entry.get("identity_hash"), entry.get("file")) != (
                    split, row["sample_id"], "full", old["identity_hash"], key+".pt"):
                raise ValueError("Source cache crossed a sample/mode/split boundary")
    return {"run": str(source), "release": str(release), "identity_hash": old["identity_hash"],
            "context_sha256": file_hash(source / "context.json"),
            "manifest_sha256": file_hash(source / "cache/complete.json"),
            "perception_contract": perception_contract(ROOT)}


def initialize(args):
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    if (run / "context.json").exists():
        if not args.resume:
            raise ValueError("Existing conflict-cost run requires explicit --resume")
        return context_for(run)
    if any(p.name not in ("logs", ".controller.lock", "pipeline.status.json") for p in run.iterdir()):
        raise ValueError("New conflict-cost run directory is not empty")
    data_root = Path(args.data_root).resolve()
    train = load_training(data_root / "train.jsonl")
    val, vd = load_records(data_root / "val.jsonl")
    test, td = load_records(data_root / "test_release.jsonl", require_labels=False)
    if (len(train), len(val), len(test)) != (3354, 666, 734) or vd or td:
        raise ValueError("Unexpected official split counts")
    for a, b in ((train, val), (train, test), (val, test)):
        check_disjoint(a, b)
    baseline = baseline_csv(args.baseline_csv, test)
    settings = settings_from(args.config)
    # This verifies the exact 30 same-video opposite-label TRAIN pairs at setup.
    EpochSampler(train, settings["seed"], settings["batch_size"])
    media = {}
    for row in train+val+test:
        name = row["video"]
        path = (data_root / name).resolve()
        if not path.is_relative_to(data_root) or not path.is_file():
            raise ValueError("Media is missing or outside official data root")
        if name not in media:
            media[name] = file_hash(path)
    identity = {"algorithm": VERSION, "settings": settings, "source": code_hash(),
        "software": software_versions(), "execution": EXECUTION,
        "base": artifact_hash(args.model_path), "sft": artifact_hash(args.sft_adapter_path, adapter=True),
        "media": media, "split_hashes": {"train": fingerprint(train), "val": fingerprint(val), "test": fingerprint(test)},
        "raw_split_hashes": {name: file_hash(data_root / filename) for name, filename in
            (("train", "train.jsonl"), ("val", "val.jsonl"), ("test", "test_release.jsonl"))},
        "baseline_csv_sha256": file_hash(args.baseline_csv)}
    identity["cache_parent"] = verify_source(args.source_run, args.source_release, identity)
    dest = run / "baseline_090813.csv"
    shutil.copyfile(args.baseline_csv, dest)
    dest.chmod(0o444)
    started = args.budget_start if args.budget_start is not None else time.time()
    c = {"identity": identity, "identity_hash": fingerprint(identity), "settings": settings,
        "model": str(Path(args.model_path).resolve()), "adapter": str(Path(args.sft_adapter_path).resolve()),
        "data_root": str(data_root), "source_run": str(Path(args.source_run).resolve()),
        "source_release": str(Path(args.source_release).resolve()), "train": train, "val": val, "test": test,
        "baseline_test": baseline, "started": started,
        "deadline": started+settings["budget_hours"]*3600,
        "training_cutoff": started+settings["training_cutoff_hours"]*3600}
    for name in ("cache", "requests", "results", "checkpoints", "logs"):
        (run / name).mkdir(exist_ok=True)
    write_json(run / "context.json", c)
    return c


def context_for(run):
    run = Path(run)
    c = read(run / "context.json")
    identity = c["identity"]
    if identity["algorithm"] != VERSION or c["identity_hash"] != fingerprint(identity) or identity["source"] != code_hash():
        raise ValueError("Conflict-cost algorithm/source/identity mismatch")
    if identity["software"] != software_versions() or settings_from() != c["settings"] or identity.get("execution") != EXECUTION:
        raise ValueError("Runtime or approved conflict-cost settings changed")
    if file_hash(run / "baseline_090813.csv") != identity["baseline_csv_sha256"]:
        raise ValueError("Frozen public baseline changed")
    if artifact_hash(c["model"]) != identity["base"] or artifact_hash(c["adapter"], adapter=True) != identity["sft"]:
        raise ValueError("Original model weights changed")
    for split, filename in (("train", "train.jsonl"), ("val", "val.jsonl"), ("test", "test_release.jsonl")):
        if file_hash(Path(c["data_root"]) / filename) != identity["raw_split_hashes"][split] or fingerprint(c[split]) != identity["split_hashes"][split]:
            raise ValueError("Dataset split changed")
    parent = identity["cache_parent"]
    source = Path(c["source_run"])
    if (str(source.resolve()) != parent["run"] or str(Path(c["source_release"]).resolve()) != parent["release"]
            or file_hash(source / "context.json") != parent["context_sha256"]
            or file_hash(source / "cache/complete.json") != parent["manifest_sha256"]
            or perception_contract(ROOT) != parent["perception_contract"]):
        raise ValueError("Source cache provenance changed")
    return c


def save_task(run, row, split, mode, packed, reference, baseline, torch, identity):
    if mode != "full" or split not in ("train", "val"):
        raise ValueError("New conflict cache contains train/val full tasks only")
    if len(reference) != 2 or not all(math.isfinite(v) for v in reference):
        raise ValueError("New-prompt reference requires two finite native field scores")
    key = task_key(row["sample_id"], mode)
    dest = Path(run) / "cache" / (key+".pt")
    if dest.exists():
        # A process may finish the blob and fail before journaling its metadata
        # (for example, while generating the zero-update question). Reuse only
        # an exactly regenerated tensor tree, preserving the existing bytes.
        previous = torch.load(dest, map_location="cpu", weights_only=True)
        if not _same_tensor_tree(torch, previous, packed):
            raise ValueError("Refusing to overwrite a conflicting immutable cache tensor")
    else:
        temp = dest.with_suffix(".partial")
        torch.save(packed, temp)
        temp.replace(dest)
    return {"sample_id": row["sample_id"], "mode": mode, "split": split,
        "file": dest.name, "sha256": file_hash(dest), "reference_scores": reference,
        "baseline": baseline, "identity_hash": identity}


def _same_tensor_tree(torch, a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return (torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype
                and a.shape == b.shape and torch.equal(a.cpu(), b.cpu()))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_same_tensor_tree(torch, a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and type(a) is type(b):
        return len(a) == len(b) and all(_same_tensor_tree(torch, x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def _validate_entry(key, entry, c):
    expected = {r["sample_id"]: split for split in ("train", "val") for r in c[split]}
    sid = entry["sample_id"]
    if (entry.get("mode") != "full" or key != task_key(sid, "full") or entry.get("file") != key+".pt"
            or entry.get("split") != expected.get(sid) or entry.get("identity_hash") != c["identity_hash"]):
        raise ValueError("Conflict cache crossed a sample/mode/split/identity boundary")
    scores = entry.get("reference_scores", [])
    if len(scores) != 2 or not all(math.isfinite(v) for v in scores):
        raise ValueError("Conflict cache lacks valid new-prompt reference scores")


def merge_tasks(run, c):
    run = Path(run)
    entries = {}
    for shard in (0, 1):
        part = read(run / f"cache/shard-{shard}.json")
        if part["identity_hash"] != c["identity_hash"]:
            raise ValueError("Conflict shard belongs to another run")
        for entry in part["entries"]:
            key = task_key(entry["sample_id"], entry["mode"])
            _validate_entry(key, entry, c)
            if key in entries or file_hash(run / "cache" / entry["file"]) != entry["sha256"]:
                raise ValueError("Duplicate or corrupt conflict task entry")
            entries[key] = entry
    expected = {task_key(row["sample_id"], "full") for split in ("train", "val") for row in c[split]}
    if set(entries) != expected:
        raise ValueError("Conflict shards are incomplete")
    value = {"schema": VERSION, "identity_hash": c["identity_hash"], "entries": entries,
             "perception_source": c["identity"]["cache_parent"]}
    write_json(run / "cache/complete.json", value)
    return value


class TaskCache:
    def __init__(self, run, c, torch):
        self.run, self.c, self.torch = Path(run), c, torch
        manifest = read(self.run / "cache/complete.json")
        if manifest.get("schema") != VERSION or manifest.get("identity_hash") != c["identity_hash"]:
            raise ValueError("Wrong algorithm or identity for conflict task cache")
        self.entries, self.checked = manifest["entries"], set()
        expected = {task_key(r["sample_id"], "full") for split in ("train", "val") for r in c[split]}
        if set(self.entries) != expected:
            raise ValueError("Conflict task cache membership is incomplete")
        for key, entry in self.entries.items():
            _validate_entry(key, entry, c)

    def entry(self, row, mode="full"):
        return self.entries[task_key(row["sample_id"], mode)]

    def load(self, row, split, mode="full", *, device="cuda:0"):
        from .finite_runtime import unpack_cache
        entry = self.entry(row, mode)
        if entry["split"] != split or mode != "full":
            raise ValueError("Conflict task cache crossed a sample/mode/split")
        path = self.run / "cache" / entry["file"]
        if path.name not in self.checked:
            if file_hash(path) != entry["sha256"]:
                raise ValueError("Corrupt conflict task cache tensor")
            self.checked.add(path.name)
        return unpack_cache(self.torch, self.torch.load(path, map_location="cpu", weights_only=True), device)


def sampler_position(c, seed, step, smoke=False, smoke_selection=None):
    if smoke:
        if not smoke_selection:
            raise ValueError("Smoke recovery requires its immutable selected-32 identity")
        return {"sequence": "fixed_smoke32", "step": step, "seed": seed,
                "selection_sha256": smoke_selection, "pair_sequence": None}
    sampler = EpochSampler(c["train"], seed, c["settings"]["batch_size"])
    epoch, offset = divmod(step, sampler.epoch_updates)
    return {"identity": sampler.identity, "step": step, "seed": seed, "smoke": smoke,
            "epoch": epoch, "main_position": offset*sampler.batch_size,
            "pair_cycle": step//len(sampler.pairs), "pair_position": step % len(sampler.pairs)}


def save_checkpoint(run, policy, optimizer, scheduler, c, *, seed, step, smoke=False):
    root = Path(run) / "checkpoints" / ("smoke" if smoke else f"seed-{seed}")
    root.mkdir(parents=True, exist_ok=True)
    dest = root / f"step-{step:06d}"
    if dest.exists():
        raise ValueError("Refusing to overwrite a completed conflict checkpoint")
    temp = root / (dest.name+".partial-"+uuid.uuid4().hex)
    temp.mkdir()
    policy.model.save_pretrained(temp / "adapters", selected_adapters=["policy"], safe_serialization=True)
    smoke_selection = file_hash(Path(run) / "smoke-selection.json") if smoke else None
    state = {"algorithm": VERSION, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "rng": capture_rng(policy.torch), "step": step, "seed": seed,
        "sampler": sampler_position(c, seed, step, smoke, smoke_selection)}
    policy.torch.save(state, temp / "training_state.pt")
    marker = {"schema": VERSION, "identity_hash": c["identity_hash"], "seed": seed, "step": step,
        "smoke": smoke, "sampler": state["sampler"], "policy_digest": adapter_digest(policy, "policy"),
        "reference_digest": adapter_digest(policy, "reference"),
        "files": {p.relative_to(temp).as_posix(): file_hash(p) for p in temp.rglob("*") if p.is_file()}}
    write_json(temp / "complete.json", marker)
    temp.rename(dest)
    if not smoke:
        write_json(Path(run) / f"latest-{seed}.json", {"checkpoint": str(dest), "step": step})
    return dest


def validate_checkpoint(path, c):
    path = Path(path).resolve()
    marker = read(path / "complete.json")
    required = {"training_state.pt", "adapters/policy/adapter_model.safetensors", "adapters/policy/adapter_config.json"}
    if (marker.get("schema") != VERSION or marker.get("identity_hash") != c["identity_hash"]
            or not required <= set(marker.get("files", {}))
            or any(name.startswith("adapters/old/") for name in marker.get("files", {}))):
        raise ValueError("Wrong algorithm or incomplete conflict checkpoint")
    smoke_selection = file_hash(path.parents[2] / "smoke-selection.json") if marker["smoke"] else None
    if marker.get("sampler") != sampler_position(c, marker["seed"], marker["step"], marker["smoke"], smoke_selection):
        raise ValueError("Conflict checkpoint sampler/data position mismatch")
    for name, expected in marker["files"].items():
        candidate = (path / name).resolve()
        if not candidate.is_relative_to(path) or file_hash(candidate) != expected:
            raise ValueError("Conflict checkpoint integrity failure")
    return marker


def load_checkpoint(policy, path, c, *, optimizer=None, scheduler=None):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    path = Path(path)
    marker = validate_checkpoint(path, c)
    weights = load_file(str(path / "adapters/policy/adapter_model.safetensors"))
    set_peft_model_state_dict(policy.model, weights, adapter_name="policy")
    policy._set_trainability()
    if adapter_digest(policy, "policy") != marker["policy_digest"]:
        raise ValueError("Conflict policy adapter did not restore exactly")
    if adapter_digest(policy, "reference") != marker["reference_digest"]:
        raise ValueError("Frozen original v4 changed")
    if optimizer is None:
        return marker
    if scheduler is None:
        raise ValueError("Optimizer resume requires its fixed scheduler")
    state = policy.torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    if (state.get("algorithm") != VERSION or state.get("step") != marker["step"] or state.get("seed") != marker["seed"]
            or state.get("sampler") != marker["sampler"] or "old_block" in state or "pool" in state):
        raise ValueError("Conflict optimizer/sampler state mismatch or obsolete algorithm")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng(policy.torch, state["rng"])
    return state
