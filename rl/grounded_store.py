"""Strict immutable identities, task caches, and block-exact resumable checkpoints."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
import uuid

from .experiment import ROOT, artifact_hash, check_disjoint, file_hash, fingerprint, load_records, write_json
from .grounded_uncertainty import VERSION, baseline_csv, load_training, settings_from
from .state import adapter_digest, capture_rng, restore_rng, software_versions


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def code_hash():
    paths = [*sorted((ROOT / "rl").glob("*.py")), ROOT / "sft_prompt.py"]
    return fingerprint({p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8") for p in paths})


def initialize(args):
    started = time.time()
    run = args.run_dir
    if (run / "context.json").exists():
        if not args.resume:
            raise ValueError("Existing run requires explicit --resume")
        c = context_for(run)
        return c
    train = load_training(args.data_root / "train.jsonl")
    val, vd = load_records(args.data_root / "val.jsonl")
    test, td = load_records(args.data_root / "test_release.jsonl", require_labels=False)
    if (len(train), len(val), len(test)) != (3354, 666, 734) or vd or td:
        raise ValueError("Unexpected official split counts")
    for a, b in ((train, val), (train, test), (val, test)):
        check_disjoint(a, b)
    baseline = baseline_csv(args.baseline_csv, test)
    if any(p.name not in ("logs", ".controller.lock", "pipeline.status.json") for p in run.iterdir()):
        raise ValueError("New grounded run directory is not empty")
    settings = settings_from(args.config)
    media = {}
    for row in train+val+test:
        name = row["video"]
        path = (args.data_root / name).resolve()
        if not path.is_relative_to(args.data_root.resolve()):
            raise ValueError("Media outside official data root")
        if name not in media:
            media[name] = file_hash(path)
    old = read(args.legacy_run / "context.json")
    old_manifest = read(args.legacy_run / "cache/complete.json")
    if old_manifest["identity_hash"] != old["identity_hash"]:
        raise ValueError("Legacy cache identity mismatch")
    base, adapter = artifact_hash(args.model_path), artifact_hash(args.sft_adapter_path, adapter=True)
    if old["identity"]["base"] != base or old["identity"]["sft"] != adapter:
        raise ValueError("Legacy feature weights differ from original v4")
    input_keys = ("video_fps", "video_min_frames", "video_max_frames", "video_pixels", "audio_sampling_rate",
                  "max_prompt_tokens", "max_completion_length", "attn_implementation")
    if any(old["identity"]["settings"][k] != settings[k] for k in input_keys) or old["identity"]["software"] != software_versions():
        raise ValueError("Legacy feature precision/input/runtime compatibility check failed")
    for row in train+val:
        if old["identity"]["media"].get(row["video"]) != media[row["video"]]:
            raise ValueError("Legacy media changed")
    identity = {"algorithm": VERSION, "settings": settings, "source": code_hash(), "software": software_versions(),
                "base": base, "sft": adapter, "media": media,
                "split_hashes": {"train": fingerprint(train), "val": fingerprint(val), "test": fingerprint(test)},
                "raw_split_hashes": {name: file_hash(args.data_root / filename) for name, filename in
                    (("train", "train.jsonl"), ("val", "val.jsonl"), ("test", "test_release.jsonl"))},
                "baseline_csv_sha256": file_hash(args.baseline_csv), "legacy_identity": old["identity_hash"],
                "legacy_manifest_sha256": file_hash(args.legacy_run / "cache/complete.json")}
    dest = run / "baseline_090813.csv"
    shutil.copyfile(args.baseline_csv, dest)
    dest.chmod(0o444)
    c = {"identity": identity, "identity_hash": fingerprint(identity), "settings": settings,
         "model": str(args.model_path.resolve()), "adapter": str(args.sft_adapter_path.resolve()),
         "data_root": str(args.data_root.resolve()), "legacy_run": str(args.legacy_run.resolve()),
         "train": train, "val": val, "test": test, "baseline_test": baseline,
         "started": started, "deadline": started+settings["budget_hours"]*3600,
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
        raise ValueError("Grounded algorithm/source/identity mismatch")
    if identity["software"] != software_versions() or settings_from() != c["settings"]:
        raise ValueError("Runtime or approved settings changed")
    if file_hash(run / "baseline_090813.csv") != identity["baseline_csv_sha256"]:
        raise ValueError("Frozen public baseline changed")
    if artifact_hash(c["model"]) != identity["base"] or artifact_hash(c["adapter"], adapter=True) != identity["sft"]:
        raise ValueError("Original model weights changed")
    for name, filename in (("train", "train.jsonl"), ("val", "val.jsonl"), ("test", "test_release.jsonl")):
        if file_hash(Path(c["data_root"]) / filename) != identity["raw_split_hashes"][name] or fingerprint(c[name]) != identity["split_hashes"][name]:
            raise ValueError("Dataset split changed")
    return c


def task_key(sid, mode):
    return fingerprint([sid, mode])


def save_task(run, row, split, mode, packed, reference, baseline, torch, identity):
    key = task_key(row["sample_id"], mode)
    dest = Path(run) / "cache" / (key+".pt")
    temp = dest.with_suffix(".partial")
    torch.save(packed, temp)
    temp.replace(dest)
    return {"sample_id": row["sample_id"], "mode": mode, "split": split,
            "file": dest.name, "sha256": file_hash(dest), "reference_scores": reference,
            "baseline": baseline, "identity_hash": identity}


class TaskCache:
    def __init__(self, run, c, torch):
        self.run, self.torch, self.c = Path(run), torch, c
        manifest = read(self.run / "cache/complete.json")
        if manifest["identity_hash"] != c["identity_hash"]:
            raise ValueError("Task cache identity mismatch")
        self.entries, self.checked = manifest["entries"], set()

    def entry(self, row, mode):
        return self.entries[task_key(row["sample_id"], mode)]

    def load(self, row, split, mode="full"):
        from .finite_runtime import unpack_cache
        e = self.entry(row, mode)
        path = self.run / "cache" / e["file"]
        if e["split"] != split or e["sample_id"] != row["sample_id"] or e["mode"] != mode or path.name != task_key(row["sample_id"], mode)+".pt":
            raise ValueError("Task cache crossed a sample/mode/split")
        if path.name not in self.checked:
            if file_hash(path) != e["sha256"]:
                raise ValueError("Corrupt task cache")
            self.checked.add(path.name)
        return unpack_cache(self.torch, self.torch.load(path, map_location="cpu", weights_only=True))


def merge_tasks(run, c):
    entries = {}
    for shard in (0, 1):
        part = read(Path(run) / f"cache/shard-{shard}.json")
        if part["identity_hash"] != c["identity_hash"]:
            raise ValueError("Shard belongs to another run")
        for entry in part["entries"]:
            key = task_key(entry["sample_id"], entry["mode"])
            if key in entries or file_hash(Path(run) / "cache" / entry["file"]) != entry["sha256"]:
                raise ValueError("Duplicate or corrupt task entry")
            entries[key] = entry
    expected = {task_key(r["sample_id"], mode): split for split in ("train", "val") for r in c[split]
                for mode in (["full", *r["view_targets"]] if split == "train" else ["full"])}
    if {key: e["split"] for key, e in entries.items()} != expected:
        raise ValueError("Task shards incomplete or crossed splits")
    value = {"schema": VERSION, "identity_hash": c["identity_hash"], "entries": entries}
    write_json(Path(run) / "cache/complete.json", value)
    return value


def save_checkpoint(run, policy, optimizer, scheduler, c, *, seed, step, old_block, pool, smoke=False):
    root = Path(run) / "checkpoints" / ("smoke" if smoke else f"seed-{seed}")
    root.mkdir(parents=True, exist_ok=True)
    dest = root / f"step-{step:06d}"
    if dest.exists():
        raise ValueError("Refusing to overwrite a completed grounded checkpoint")
    temp = root / (dest.name+".partial-"+uuid.uuid4().hex)
    temp.mkdir()
    policy.model.save_pretrained(temp / "adapters", selected_adapters=["policy", "old"], safe_serialization=True)
    state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "rng": capture_rng(policy.torch), "step": step, "seed": seed, "old_block": old_block, "pool": pool}
    policy.torch.save(state, temp / "training_state.pt")
    marker = {"schema": VERSION, "identity_hash": c["identity_hash"], "seed": seed, "step": step,
              "old_block": old_block, "pool_hash": fingerprint(pool), "smoke": smoke,
              "policy_digest": adapter_digest(policy, "policy"), "old_digest": adapter_digest(policy, "old"),
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
    required = {"training_state.pt", "adapters/policy/adapter_model.safetensors", "adapters/old/adapter_model.safetensors"}
    if marker.get("schema") != VERSION or marker.get("identity_hash") != c["identity_hash"] or not required <= set(marker.get("files", {})):
        raise ValueError("Wrong algorithm or incomplete grounded checkpoint")
    for name, expected in marker["files"].items():
        candidate = (path / name).resolve()
        if not candidate.is_relative_to(path) or file_hash(candidate) != expected:
            raise ValueError("Grounded checkpoint integrity failure")
    return marker


def load_checkpoint(policy, path, c, *, optimizer=None, scheduler=None):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    path = Path(path)
    marker = validate_checkpoint(path, c)
    for name in ("policy", "old"):
        weights = load_file(str(path / f"adapters/{name}/adapter_model.safetensors"))
        set_peft_model_state_dict(policy.model, weights, adapter_name=name)
        if adapter_digest(policy, name) != marker[f"{name}_digest"]:
            raise ValueError("Adapter did not restore exactly")
    policy._set_trainability()
    if adapter_digest(policy, "reference") != marker["reference_digest"]:
        raise ValueError("Frozen v4 changed")
    if optimizer is None:
        return marker
    state = policy.torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    if state["step"] != marker["step"] or state["seed"] != marker["seed"] or state["old_block"] != marker["old_block"] or fingerprint(state["pool"]) != marker["pool_hash"]:
        raise ValueError("Optimizer/block/sampler state mismatch")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng(policy.torch, state["rng"])
    return state
