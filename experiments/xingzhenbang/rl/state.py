"""Experiment identity, RNG persistence and atomic completion markers."""
from __future__ import annotations

import importlib.metadata
import json
import platform
import random
import subprocess
import uuid
from pathlib import Path

from .experiment import ROOT, file_hash, fingerprint, write_json


def software_versions():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "torchvision", "transformers", "trl", "peft", "accelerate",
                 "bitsandbytes", "av", "numpy", "librosa", "pillow"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def code_identity():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                            cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    # Source hashes also identify deployments made from a source snapshot without .git.
    sources = [*sorted((ROOT / "rl").glob("*.py")), ROOT / "sft_prompt.py"]
    return {"commit": commit, "dirty": dirty,
            # Universal-newline reading makes Windows/Linux source snapshots
            # identical without relaxing checks on actual code changes.
            "source_hash": fingerprint({p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8")
                                        for p in sources})}


def capture_rng(torch):
    import numpy as np
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(torch, state):
    import numpy as np
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def adapter_digest(policy, name):
    import hashlib
    digest = hashlib.sha256()
    found = False
    for key, value in sorted(policy.model.state_dict().items()):
        if f".{name}." in key:
            found = True
            digest.update(key.replace(f".{name}.", ".adapter.").encode())
            raw = value.detach().cpu().contiguous().view(policy.torch.uint8).numpy().tobytes()
            digest.update(raw)
    if not found:
        raise ValueError(f"No adapter tensors found: {name}")
    return digest.hexdigest()


def save_checkpoint(run_dir, policy, optimizer, scheduler, completed_steps, identity):
    run_dir = Path(run_dir)
    destination = run_dir / f"checkpoint-{completed_steps:06d}"
    if destination.exists():
        marker = validate_checkpoint(destination, identity)
        if (marker["completed_steps"] != completed_steps
                or marker["policy_digest"] != adapter_digest(policy, "policy")):
            raise ValueError("Conflicting checkpoint destination")
        return destination
    temporary = run_dir / (destination.name + ".partial-" + uuid.uuid4().hex)
    temporary.mkdir(parents=True)
    policy.model.save_pretrained(temporary / "adapter", selected_adapters=["policy"], safe_serialization=True)
    policy.processor.save_pretrained(temporary / "processor")
    policy.torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                       "rng": capture_rng(policy.torch), "completed_steps": completed_steps},
                      temporary / "training_state.pt")
    write_json(temporary / "complete.json", {
        "schema": "omni_checkpoint_v1", "identity": identity, "completed_steps": completed_steps,
        "reference_digest": adapter_digest(policy, "reference"),
        "policy_digest": adapter_digest(policy, "policy"),
        "files": {p.relative_to(temporary).as_posix(): file_hash(p)
                  for p in sorted(temporary.rglob("*")) if p.is_file()}})
    temporary.rename(destination)
    write_json(run_dir / "latest_checkpoint.json", {"path": destination.name, "completed_steps": completed_steps})
    return destination


def validate_checkpoint(path, identity):
    path = Path(path)
    marker = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    required = {"training_state.pt", "adapter/policy/adapter_model.safetensors", "adapter/policy/adapter_config.json"}
    if (marker.get("schema") != "omni_checkpoint_v1" or not required <= set(marker.get("files", {}))
            or type(marker.get("completed_steps")) is not int or marker["completed_steps"] < 0
            or not marker.get("policy_digest") or not marker.get("reference_digest")):
        raise ValueError("Incomplete checkpoint manifest")
    if marker["identity"] != identity:
        raise ValueError("Checkpoint experiment/data/model/source identity mismatch")
    for relative, expected in marker["files"].items():
        candidate = (path / relative).resolve()
        if not candidate.is_relative_to(path.resolve()) or file_hash(candidate) != expected:
            raise ValueError(f"Checkpoint file failed integrity check: {relative}")
    return marker


def validate_run_directory(run_dir, identity, resume_marker=None):
    """Prevent rewinds from reusing validation files for different policy weights."""
    run_dir = Path(run_dir)
    identity_path = run_dir / "experiment_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Run directory belongs to a different experiment")
    elif any(run_dir.iterdir()):
        raise ValueError("Nonempty run directory has no experiment identity")
    else:
        write_json(identity_path, identity)
    if resume_marker is not None:
        steps = []
        for marker_path in run_dir.glob("checkpoint-*/complete.json"):
            value = json.loads(marker_path.read_text(encoding="utf-8"))
            steps.append(value["completed_steps"])
        if steps and max(steps) > resume_marker["completed_steps"]:
            raise ValueError("Resume the latest completed checkpoint; rewinding would reuse stale evaluations")


def validate_f1_cache(cache, identity):
    if cache.get("schema") != "train_f1_cache_v1" or cache.get("split") != "train":
        raise ValueError("Only a train_f1_cache_v1 TRAIN cache can supply rewards")
    if cache.get("baseline_identity") != identity:
        raise ValueError("F1 cache uses different data, base, SFT adapter or input settings")
