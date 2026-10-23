"""Pure-Python experiment contracts, data validation and resumable sampling."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
import uuid
from pathlib import Path

from .reward import REWARD_VERSIONS, VALID_EMOTIONS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = {
    "reward_version": "correctness_v1", "seed": 42, "max_steps": 200,
    "prompts_per_update": 8, "num_generations": 4, "learning_rate": 5e-6,
    "warmup_steps": 10, "beta": .02, "max_grad_norm": 1.0,
    "max_completion_length": 128, "max_prompt_tokens": 6144,
    "video_fps": 2.0, "video_min_frames": 4, "video_max_frames": 64, "video_pixels": 50176,
    "audio_sampling_rate": 16000, "save_every": 20, "eval_every": 100,
    "loss_type": "dr_grpo", "scale_rewards": "none", "attn_implementation": "sdpa",
}
DEFAULT_RUNTIME = {"gpu": "auto", "micro_batch_size": 1, "max_runtime_hours": None,
                   "min_free_memory_gib": 12}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_hash(path, adapter=False):
    path = Path(path)
    files = sorted({p for pattern in ("*.safetensors", "*.json", "*.model", "*.jinja", "*.txt")
                    for p in path.glob(pattern) if p.is_file()})
    if not files or not any(p.suffix == ".safetensors" for p in files):
        raise ValueError(f"No safetensors artifact found at {path}")
    if adapter and not (path / "adapter_config.json").is_file():
        raise ValueError(f"Missing adapter_config.json at {path}")
    return fingerprint({p.name: file_hash(p) for p in files})


def read_config(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) - {"experiment", "runtime"}:
        raise ValueError("Unknown top-level config field")
    experiment, runtime = dict(DEFAULT_EXPERIMENT), dict(DEFAULT_RUNTIME)
    for section, target in (("experiment", experiment), ("runtime", runtime)):
        unknown = set(raw.get(section, {})) - set(target)
        if unknown:
            raise ValueError(f"Unknown {section} options: {sorted(unknown)}")
        target.update(raw.get(section, {}))
    validate_config(experiment, runtime)
    return experiment, runtime


def validate_config(experiment, runtime):
    for key in ("max_steps", "prompts_per_update", "num_generations", "max_completion_length",
                "max_prompt_tokens", "video_min_frames", "video_max_frames", "video_pixels", "audio_sampling_rate",
                "save_every", "eval_every"):
        if type(experiment[key]) is not int or experiment[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if experiment["num_generations"] < 2 or experiment["prompts_per_update"] % 2:
        raise ValueError("GRPO needs G>=2 and an even number of prompts per update")
    if (experiment["video_min_frames"] % 2 or experiment["video_max_frames"] % 2
            or experiment["video_min_frames"] > experiment["video_max_frames"]
            or experiment["video_pixels"] % (28 * 28)):
        raise ValueError("Qwen video budget requires even frames and pixels divisible by 784")
    if experiment["loss_type"] != "dr_grpo" or experiment["scale_rewards"] != "none":
        raise ValueError("v1 uses only the mean-centered Dr.GRPO objective")
    if experiment["reward_version"] not in REWARD_VERSIONS:
        raise ValueError("Unknown reward version")
    for key in ("learning_rate", "beta", "max_grad_norm", "video_fps"):
        if not math.isfinite(experiment[key]) or experiment[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not 0 <= experiment["warmup_steps"] < experiment["max_steps"]:
        raise ValueError("warmup_steps must be less than max_steps")
    if runtime["micro_batch_size"] not in (1, 2, 4):
        raise ValueError("micro_batch_size must be 1, 2 or 4")
    if experiment["num_generations"] % runtime["micro_batch_size"]:
        raise ValueError("micro_batch_size must divide num_generations")
    hours = runtime["max_runtime_hours"]
    if hours is not None and (not math.isfinite(hours) or hours <= 0):
        raise ValueError("max_runtime_hours must be positive or null")


def canonical_record(row, require_labels=True):
    sid, video, protocol = row.get("sample_id"), row.get("video"), row.get("protocol")
    if not isinstance(sid, str) or not sid or not isinstance(video, str) or not video:
        raise ValueError("Every sample needs nonempty sample_id and video strings")
    if protocol is None:
        # Official test_release.jsonl contains only sample_id/video/text.
        # Use the same input-only ID convention as the SFT evaluator.
        parts = sid.split(":")
        tag = parts[1] if len(parts) >= 2 else None
        protocol = {"VT": "VT", "VA": "VA", "accept_a_va": "VA",
                    "accept_a_svt": "VT", "accept_c_svt": "VT"}.get(tag)
    if protocol not in ("VT", "VA"):
        raise ValueError(f"Unsupported protocol for {sid}")
    gold = row.get("views", {}).get("M12", {}).get("label", row.get("gt_label"))
    sample_type = row.get("sample_type")
    if require_labels and (gold not in VALID_EMOTIONS or sample_type not in ("Conflict", "Aligned")):
        raise ValueError(f"Missing original reward labels for {sid}")
    source = row.get("source_dataset")
    if not source:
        source = next((name for name in ("ch_sims_v2", "ch_sims", "accept_a_svt", "accept_c_svt", "accept_a_va")
                       if sid.startswith(name)), "unknown")
    return {"sample_id": sid, "video": video, "protocol": protocol,
            "text_content": str(row.get("text_content", row.get("text", ""))),
            "gt_label": gold if require_labels else None,
            "sample_type": sample_type if require_labels else None,
            "source_dataset": source,
            "split_group_id": row.get("split_group_id") or video}


def load_records(path, require_labels=True):
    unique, duplicates = {}, 0
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = canonical_record(json.loads(line), require_labels)
            sid = row["sample_id"]
            if sid in unique:
                if unique[sid] != row:
                    raise ValueError(f"Conflicting records share ID {sid}")
                duplicates += 1
            unique[sid] = row
    if not unique:
        raise ValueError(f"Empty dataset: {path}")
    return list(unique.values()), duplicates


def check_disjoint(train, val):
    for key in ("sample_id", "video", "split_group_id"):
        if {r[key] for r in train} & {r[key] for r in val}:
            raise ValueError(f"Train/validation {key} overlap")


class BalancedSampler:
    """Step-indexed sampling; restoration needs only the completed update count.

    Uniform sampling without replacement within each update/type; independent
    between updates. No hidden oversampling of individual rare-label examples.
    """
    def __init__(self, records, seed, prompts_per_update):
        self.pools = {t: [r for r in records if r["sample_type"] == t]
                      for t in ("Conflict", "Aligned")}
        self.seed, self.count = seed, prompts_per_update // 2
        if prompts_per_update % 2 or self.count < 1:
            raise ValueError("An even positive prompt count is required")
        if any(len(pool) < self.count for pool in self.pools.values()):
            raise ValueError("Too few unique examples for balanced sampling")

    def batch(self, completed_steps):
        rng = random.Random(int(fingerprint([self.seed, completed_steps]), 16))
        rows = []
        for pool in self.pools.values():
            rows.extend(rng.sample(pool, self.count))
        rng.shuffle(rows)
        return rows


def group_advantages(rewards):
    if len(rewards) < 2 or any(not math.isfinite(r) for r in rewards):
        raise ValueError("A finite reward group with G>=2 is required")
    mean = sum(rewards) / len(rewards)
    return [r - mean for r in rewards]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for attempt in range(8):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                # Windows may briefly lock a destination during another atomic
                # replacement. POSIX writers do not need this sharing retry.
                if os.name != "nt" or attempt == 7:
                    raise
                time.sleep(.01*(attempt+1))
    finally:
        if temporary.exists():
            temporary.unlink()
