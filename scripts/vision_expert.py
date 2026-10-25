"""Frozen Qwen3-VL visual evidence extraction and independent TRAIN evaluation.

GPU: preflight (32 training videos), extract (all training videos).
CPU: evaluate delegates to vision_expert_cpu; no validation/test inference here.
The extractor file is frozen independently of the CPU evaluation module so
adding that evaluation cannot invalidate an already-running visual cache.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import uuid

import numpy as np

from rl.experiment import artifact_hash, file_hash, fingerprint, write_json
from rl.reward import VALID_EMOTIONS

VERSION = "qwen3vl8b_visual_expert_v1"
PREFIX = '{"pred_emotion":"'
SETTINGS = {"video_fps": 2., "video_min_frames": 4, "video_max_frames": 64,
    "video_pixels_per_frame": 50176, "max_prompt_tokens": 6144,
    "attn_implementation": "sdpa", "quantization": "nf4_double_quant_bfloat16",
    "sampling": "qwen3_native_count_uniform_rounded_indices_pyav_video_only",
    "pixel_budget": "per_frame_max; qwen3_total_budget_scaled_by_padded_frame_count",
    "seed": 42, "preflight_videos": 32, "storage": "float32_npz",
    "position": "last_token_before_emotion_value", "audio": False,
    "supplied_transcript": False, "physical_gpu": "2", "memory_limit_gib": 32.}
SYSTEM_PROMPT = "You are an emotion recognition assistant."
INSTRUCTION = ("Judge only the emotion visible in this silent video. No audio or "
    "transcript is provided. Return exactly one compact JSON object with "
    "pred_emotion equal to positive, negative, or neutral.")
SAMPLING_NOTE = ("Uses the same nominal fps, frame-count limits and per-frame pixel ceiling as "
    "the Omni experiment. Qwen3 has different patch geometry, frame-count rounding, "
    "video preprocessing and timestamp tokens; exact frame equivalence to Omni is "
    "not established. This is not a representation-only ablation.")


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bytes_once(path, content):
    """Atomic, non-replacing publication; identical retries are idempotent."""
    path = Path(path)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError("Refusing to overwrite different artifact: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            if os.name == "nt":
                temporary.rename(path)
            else:
                os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError("Concurrent artifact differs: " + str(path))
    finally:
        temporary.unlink(missing_ok=True)


def json_once(path, value):
    bytes_once(path, json_bytes(value))


def emit(root, phase, **values):
    value = {"phase": phase, "updated": time.time(), "pid": os.getpid(), **values}
    write_json(Path(root) / "extract.status.json", value)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


@contextmanager
def output_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".extract.lock"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError as exc:
        raise ValueError("Extraction directory is locked; inspect the previous process first") from exc
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def vision_messages(video_path):
    """Only static instructions and one video placeholder reach the tokenizer."""
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "video", "video": str(video_path)},
                {"type": "text", "text": INSTRUCTION}]}]


def video_tasks(c):
    """Deduplicate training videos, preserving source order and their media SHA."""
    tasks = {}
    ids = set()
    for row in c["train"]:
        sid, video = row["sample_id"], row["video"]
        if sid in ids:
            raise ValueError("Duplicate original training sample ID")
        ids.add(sid)
        media_sha = c["identity"]["media"].get(video)
        if (not isinstance(video, str) or not video or not isinstance(media_sha, str) or
                len(media_sha) != 64 or any(ch not in "0123456789abcdef" for ch in media_sha)):
            raise ValueError("Every training video needs a verified media SHA-256")
        key = fingerprint([VERSION, c["identity_hash"], "train", "vision", video, media_sha])
        tasks.setdefault(video, {"key": key, "video": video, "media_sha256": media_sha,
                                 "split": "train", "mode": "vision"})
    if not tasks:
        raise ValueError("No training videos")
    return list(tasks.values())


def sample_indices(total_frames, native_fps):
    """Match Qwen3's documented native frame-count rule, with strict limits."""
    if (type(total_frames) is not int or total_frames < SETTINGS["video_min_frames"] or
            not isinstance(native_fps, (int, float)) or not math.isfinite(native_fps) or native_fps <= 0):
        raise ValueError("Reliable video length/FPS and at least four frames are required")
    count = int(total_frames / native_fps * SETTINGS["video_fps"])
    count = min(total_frames, max(SETTINGS["video_min_frames"],
                                min(count, SETTINGS["video_max_frames"])))
    return np.linspace(0, total_frames - 1, count).round().astype(np.int64)


def probe_video(path):
    """Read visual stream metadata; count video-only if container count is absent."""
    import av
    with av.open(str(path), mode="r") as container:
        if not container.streams.video:
            raise ValueError("Media has no visual stream")
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate is not None else None
        total = int(stream.frames or 0)
        if total <= 0:
            total = sum(1 for _ in container.decode(video=0))
    sample_indices(total, fps)
    return total, fps


def decode_silent_video(path):
    """Decode only video stream 0; no audio stream is decoded or returned."""
    import av
    total, fps = probe_video(path)
    indices = sample_indices(total, fps)
    wanted = set(indices.tolist())
    selected, timestamps, actual_total = [], [], 0
    with av.open(str(path), mode="r") as container:
        for i, frame in enumerate(container.decode(video=0)):
            actual_total = i + 1
            if i in wanted:
                selected.append(frame.to_ndarray(format="rgb24"))
                timestamps.append(float(frame.time) if frame.time is not None else None)
    if actual_total != total or len(selected) != len(indices):
        raise ValueError("Decoded video length differs from its verified sampling metadata")
    frames = np.stack(selected)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB video frames")
    metadata = {"total_num_frames": total, "fps": float(fps), "frames_indices": indices.tolist()}
    audit = {"decode_stream": {"video": 0}, "audio_decoded": False,
        "source_frames": total, "source_fps": float(fps), "sampled_frames": len(indices),
        "sample_indices": indices.tolist(), "source_frame_timestamps": timestamps,
        "processor_timestamps": (indices / fps).tolist(),
        "source_frame_height": int(frames.shape[1]), "source_frame_width": int(frames.shape[2])}
    return frames, metadata, audit


def processor_video_kwargs(sampled_frames, patch_size=16, temporal_patch_size=2, merge_size=2):
    if (type(sampled_frames) is not int or not SETTINGS["video_min_frames"] <= sampled_frames <= SETTINGS["video_max_frames"] or
            (patch_size, temporal_patch_size, merge_size) != (16, 2, 2)):
        raise ValueError("Unexpected Qwen3 frame/patch contract")
    padded_frames = math.ceil(sampled_frames / temporal_patch_size) * temporal_patch_size
    # Qwen3 smart_resize's budget is padded-frame-count * H * W, unlike Omni's
    # per-frame size. A one-patch minimum avoids rounding-driven upscaling
    # beyond the fixed ceiling on tiny inputs; no frame exceeds 50176 pixels.
    return {"do_sample_frames": False, "return_metadata": True,
            "size": {"shortest_edge": (patch_size * merge_size) ** 2 * padded_frames,
                     "longest_edge": SETTINGS["video_pixels_per_frame"] * padded_frames}}


def validate_processed_geometry(grid, sampled_frames, token_count):
    grid = np.asarray(grid)
    if grid.shape != (1, 3) or not np.issubdtype(grid.dtype, np.integer) or (grid <= 0).any():
        raise ValueError("Expected exactly one positive integer video grid")
    t, h, w = map(int, grid[0])
    actual_frames, height, width = 2 * t, 16 * h, 16 * w
    if (t != math.ceil(sampled_frames / 2) or h % 2 or w % 2 or
            not SETTINGS["video_min_frames"] <= actual_frames <= SETTINGS["video_max_frames"] or
            height * width > SETTINGS["video_pixels_per_frame"] or
            type(token_count) is not int or not 0 < token_count <= SETTINGS["max_prompt_tokens"]):
        raise ValueError("Processor exceeded the fixed frame, pixel, or untruncated token budget")
    return {"video_grid_thw": grid.tolist(), "processed_frames_including_temporal_padding": actual_frames,
            "processed_frame_height": height, "processed_frame_width": width,
            "processed_pixels_per_frame": height * width, "input_tokens_including_prefix": token_count}


def validate_audit(audit):
    geometry = validate_processed_geometry(audit["video_grid_thw"], audit["sampled_frames"],
                                           audit["input_tokens_including_prefix"])
    indices = sample_indices(audit["source_frames"], audit["source_fps"])
    timestamps = np.asarray(audit["processor_timestamps"], dtype=np.float64)
    if (any(audit.get(k) != v for k, v in geometry.items()) or len(indices) != audit["sampled_frames"] or
            indices.tolist() != audit["sample_indices"] or timestamps.shape != indices.shape or
            not np.isfinite(timestamps).all() or
            not np.allclose(timestamps, indices / audit["source_fps"], rtol=0., atol=1e-12) or
            len(audit["source_frame_timestamps"]) != len(indices) or
            any(t is not None and not math.isfinite(t) for t in audit["source_frame_timestamps"]) or
            audit.get("audio_decoded") is not False or audit.get("decode_stream") != {"video": 0} or
            audit.get("supplied_text_in_prompt") is not False or audit.get("gold_or_id_in_prompt") is not False or
            audit.get("prompt_hash") != fingerprint([SYSTEM_PROMPT, INSTRUCTION, PREFIX]) or
            not isinstance(audit.get("elapsed_seconds"), (int, float)) or
            not math.isfinite(audit["elapsed_seconds"]) or audit["elapsed_seconds"] <= 0 or
            any(type(audit.get(k)) is not int or not 0 < audit[k] <= 32 * 1024 ** 3
                for k in ("peak_allocated_bytes", "peak_reserved_bytes"))):
        raise ValueError("Visual feature violates isolated input, sampling, timing, or budget provenance")


def preflight_plan(root, c, tasks, identity):
    path = Path(root) / "preflight-plan.json"
    if path.exists():
        value = read(path)
        if value.get("identity_hash") != identity["identity_hash"] or value.get("tasks_hash") != fingerprint(tasks):
            raise ValueError("Preflight plan identity changed")
        selected = [t for key in value["keys"] for t in tasks if t["key"] == key]
        if len(selected) != SETTINGS["preflight_videos"] or len({t["key"] for t in selected}) != len(selected):
            raise ValueError("Preflight plan is incomplete or duplicated")
        return selected, value
    probes = []
    data_root = Path(c["data_root"]).resolve()
    for i, task in enumerate(tasks):
        media_path = (data_root / task["video"]).resolve()
        if not media_path.is_relative_to(data_root) or not media_path.is_file():
            raise ValueError("Preflight video is missing or escaped")
        total, fps = probe_video(media_path)
        probes.append({"key": task["key"], "video": task["video"], "source_frames": total,
            "source_fps": fps, "sampled_frames": len(sample_indices(total, fps))})
        if (i + 1) % 100 == 0:
            emit(root, "probing_training_video_metadata", completed=i + 1, total=len(tasks))
    longest = max(range(len(tasks)), key=lambda i: (probes[i]["sampled_frames"],
                  probes[i]["source_frames"] / probes[i]["source_fps"], -i))
    selected = tasks[:SETTINGS["preflight_videos"] - 1]
    selected.append(tasks[longest] if tasks[longest] not in selected else tasks[SETTINGS["preflight_videos"] - 1])
    value = {"identity_hash": identity["identity_hash"], "tasks_hash": fingerprint(tasks),
        "keys": [t["key"] for t in selected], "selection": "first31_plus_longest_sampled_training_video",
        "maximum_sampled_frames_in_training": probes[longest]["sampled_frames"],
        "covers_training_maximum_frames": True, "covers_64_frame_cap": probes[longest]["sampled_frames"] == 64,
        "metadata_probes": probes, "audio_decoded": False, "labels_used_for_selection": False}
    json_once(path, value)
    return selected, value


def _software():
    result = {"python": platform.python_version()}
    for name in ("numpy", "torch", "transformers", "bitsandbytes", "av", "torchvision"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def checked_source(source_run):
    from rl.evidence_store import context_for
    from rl.grounded_uncertainty import load_training
    c = context_for(source_run)
    if c["settings"] != c["identity"]["settings"]:
        raise ValueError("Original evidence settings changed")
    train = Path(c["data_root"]) / "train.jsonl"
    if (file_hash(train) != c["identity"]["raw_split_hashes"]["train"] or load_training(train) != c["train"]):
        raise ValueError("Original training rows or labels changed")
    if len(c["train"]) != 3354 or len(video_tasks(c)) != 2104:
        raise ValueError("Expected all 3354 original rows and 2104 distinct training videos")
    return c


def checked_model_artifact(path):
    path = Path(path)
    config = read(path / "config.json")
    if config.get("model_type") != "qwen3_vl" or "Qwen3VLForConditionalGeneration" not in config.get("architectures", []):
        raise ValueError("The expert must be the explicit Qwen3-VL checkpoint")
    width = config.get("text_config", {}).get("hidden_size")
    if type(width) is not int or width <= 0:
        raise ValueError("Model text hidden width is missing")
    index = path / "model.safetensors.index.json"
    if index.exists():
        weight_map = read(index).get("weight_map", {})
        shards = set(weight_map.values())
        if not shards or any(Path(name).name != name or not (path / name).is_file() or
                             (path / name).stat().st_size == 0 for name in shards):
            raise ValueError("Model download is incomplete or contains an invalid shard map")
    elif not (path / "model.safetensors").is_file():
        raise ValueError("Model download lacks its safetensors weights")
    return artifact_hash(path), width


def initialize(args, c):
    model_hash, width = checked_model_artifact(args.model_path)
    tasks = video_tasks(c)
    identity = {"algorithm": VERSION, "source_producer_identity": c["identity_hash"],
        "source_context_sha256": file_hash(Path(args.source_run) / "context.json"),
        "extractor_source_hash": fingerprint(Path(__file__).read_text(encoding="utf-8")),
        "model_artifact_sha256": model_hash, "hidden_size": width,
        "settings": SETTINGS, "software": _software(),
        "training_tasks_hash": fingerprint(tasks), "training_rows_hash": fingerprint(c["train"]),
        "prompt_hash": fingerprint([SYSTEM_PROMPT, INSTRUCTION, PREFIX]),
        "emotions": list(VALID_EMOTIONS), "sampling_note": SAMPLING_NOTE}
    value = {"identity": identity, "identity_hash": fingerprint(identity),
             "model_path": str(Path(args.model_path).resolve()),
             "source_run": str(Path(args.source_run).resolve())}
    json_once(Path(args.output_dir) / "identity.json", value)
    json_once(Path(args.output_dir) / "training-tasks.json", {"identity_hash": value["identity_hash"], "tasks": tasks})
    return value


def feature_paths(root, task):
    path = Path(root) / "features" / "train" / (task["key"] + ".npz")
    return path, path.with_suffix(".json")


def _payload_hash(hidden, native):
    digest = hashlib.sha256()
    digest.update(hidden.tobytes(order="C"))
    digest.update(native.tobytes(order="C"))
    return digest.hexdigest()


def _feature_expected(identity, task, frozen_sha):
    contract = identity["identity"]
    return {"algorithm": VERSION, "identity_hash": identity["identity_hash"],
        "task_key": task["key"], "video": task["video"], "media_sha256": task["media_sha256"],
        "split": "train", "mode": "vision", "hidden_size": contract["hidden_size"],
        "model_artifact_sha256": contract["model_artifact_sha256"],
        "extractor_source_hash": contract["extractor_source_hash"],
        "settings_hash": fingerprint(contract["settings"]), "frozen_sha256": frozen_sha}


def load_feature(root, identity, task, frozen_sha, required=False):
    path, marker_path = feature_paths(root, task)
    if not path.exists() and not marker_path.exists() and not required:
        return None
    if not path.is_file() or not marker_path.is_file():
        raise ValueError("Incomplete visual feature; inspect the interrupted artifact: " + task["key"])
    marker = read(marker_path)
    expected = _feature_expected(identity, task, frozen_sha)
    if any(marker.get(k) != v for k, v in expected.items()) or marker.get("archive_sha256") != file_hash(path):
        raise ValueError("Visual feature identity or checksum changed")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"hidden", "native", "metadata"}:
            raise ValueError("Visual archive has unexpected arrays")
        h, native = data["hidden"].copy(), data["native"].copy()
        metadata = json.loads(str(data["metadata"].item()))
    if (h.shape != (expected["hidden_size"],) or native.shape != (3,) or
            h.dtype != np.float32 or native.dtype != np.float32 or
            not np.isfinite(h).all() or not np.isfinite(native).all() or
            any(metadata.get(k) != v for k, v in expected.items()) or
            metadata.get("payload_sha256") != _payload_hash(h, native) or
            marker.get("payload_sha256") != metadata.get("payload_sha256")):
        raise ValueError("Visual arrays or embedded provenance failed verification")
    validate_audit(metadata["diagnostics"])
    return marker, (h, native), metadata


def save_feature(root, identity, task, frozen_sha, hidden, native, diagnostics):
    h, logits = np.asarray(hidden, dtype=np.float32), np.asarray(native, dtype=np.float32)
    expected = _feature_expected(identity, task, frozen_sha)
    if h.shape != (expected["hidden_size"],) or logits.shape != (3,) or not np.isfinite(h).all() or not np.isfinite(logits).all():
        raise ValueError("Cannot save invalid expert feature geometry or nonfinite values")
    metadata = {**expected, "payload_sha256": _payload_hash(h, logits), "diagnostics": diagnostics}
    buffer = io.BytesIO()
    np.savez(buffer, hidden=h, native=logits,
             metadata=np.array(json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False)))
    content = buffer.getvalue()
    marker = {**expected, "payload_sha256": metadata["payload_sha256"],
              "archive_sha256": hashlib.sha256(content).hexdigest()}
    path, marker_path = feature_paths(root, task)
    # Validate the input contract before publishing either file.
    validate_audit(diagnostics)
    bytes_once(path, content)
    json_once(marker_path, marker)
    return load_feature(root, identity, task, frozen_sha, required=True)


class VisionBank:
    """Published training cache, independent of Omni's fixed 2048-wide bank."""
    def __init__(self, root, identity, tasks):
        self.split, self.values, self.tasks = "train", {}, {t["video"]: t for t in tasks}
        frozen = read(Path(root) / "model-frozen.json")
        if frozen["identity_hash"] != identity["identity_hash"]:
            raise ValueError("Wrong frozen-model identity")
        manifest = read(Path(root) / "features/train/complete.json")
        if manifest.get("identity_hash") != identity["identity_hash"] or manifest.get("keys") != [t["key"] for t in tasks]:
            raise ValueError("Visual cache publication coverage/order changed")
        entries = []
        for task in tasks:
            marker, values, _ = load_feature(root, identity, task, frozen["frozen_sha256"], required=True)
            entries.append(marker)
            self.values[task["video"]] = values
        if entries != manifest.get("entries"):
            raise ValueError("Visual cache changed after full publication")

    def get(self, rows, mode):
        if mode != "vision" or not rows:
            raise ValueError("Visual expert bank provides only nonempty visual batches")
        try:
            selected = [self.values[r["video"]] for r in rows]
        except KeyError as exc:
            raise ValueError("Only published training videos may reach the expert bank") from exc
        return np.stack([v[0] for v in selected]), np.stack([v[1] for v in selected])


class HybridBank:
    """Replace vision only; preserve Omni text/audio native features exactly."""
    def __init__(self, omni, vision):
        if omni.split != "train" or vision.split != "train":
            raise ValueError("Independent development requires TRAIN-only banks")
        self.split, self.omni, self.vision = "train", omni, vision

    def get(self, rows, mode):
        if mode not in ("vision", "text", "audio"):
            raise ValueError("Fused or unsupported features are forbidden")
        return (self.vision if mode == "vision" else self.omni).get(rows, mode)


class QwenVisualExtractor:
    """Direct base forward; no Omni perception hooks and no full-prompt logits."""
    def __init__(self, model_path, width):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
        self.torch, self.width = torch, width
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, local_files_only=True,
            torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}, low_cpu_mem_usage=True,
            attn_implementation=SETTINGS["attn_implementation"],
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
        if self.model.config.text_config.hidden_size != width:
            raise ValueError("Loaded model hidden width differs from its artifact config")
        if not getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("The expert did not load with the fixed NF4 policy")
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        tokenizer = self.processor.tokenizer
        values = [tokenizer.encode(v, add_special_tokens=False) for v in VALID_EMOTIONS]
        if any(len(v) != 1 for v in values):
            raise ValueError("Emotion values must be native single tokens")
        self.value_ids = [v[0] for v in values]
        self.prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
        for value, token in zip(VALID_EMOTIONS, self.value_ids):
            if tokenizer.encode(PREFIX + value, add_special_tokens=False) != self.prefix_ids + [token]:
                raise ValueError("Emotion continuation crosses the fixed prefix token boundary")
        self.initial_versions = self.check_frozen()

    def check_frozen(self):
        if self.model.training or any(module.training for module in self.model.modules()):
            raise RuntimeError("Visual expert left eval mode")
        if any(p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise RuntimeError("Visual expert acquired a trainable parameter or gradient")
        if any("lora_" in name or "adapter" in name for name, _ in self.model.named_parameters()):
            raise RuntimeError("An unauthorized adapter entered the frozen visual expert")
        versions = {name: int(p._version) for name, p in self.model.named_parameters()}
        if hasattr(self, "initial_versions") and versions != self.initial_versions:
            raise RuntimeError("Visual expert parameters changed during extraction")
        return versions

    def frozen_digest(self):
        """Hash real packed weights, buffers and serialized NF4 quant state."""
        torch, digest = self.torch, hashlib.sha256()
        self.check_frozen()
        for name, value in sorted(self.model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            if not torch.is_tensor(value):
                digest.update(json_bytes(value))
                continue
            digest.update(json_bytes({"dtype": str(value.dtype), "shape": list(value.shape)}))
            raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
            digest.update(memoryview(raw))
            del raw
        self.check_frozen()
        return digest.hexdigest()

    def extract(self, task, data_root):
        torch = self.torch
        self.check_frozen()
        root = Path(data_root).resolve()
        path = (root / task["video"]).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_hash(path) != task["media_sha256"]:
            raise ValueError("Visual task media is missing, escaped, or changed")
        started = time.monotonic()
        frames, metadata, audit = decode_silent_video(path)
        rendered = self.processor.apply_chat_template(vision_messages(path), tokenize=False, add_generation_prompt=True)
        if str(path) in rendered or task["video"] in rendered:
            raise ValueError("A video path leaked into tokenized instructions")
        vp = self.processor.video_processor
        kwargs = processor_video_kwargs(len(frames), vp.patch_size, vp.temporal_patch_size, vp.merge_size)
        kwargs["video_metadata"] = [metadata]
        inputs = self.processor(text=rendered, videos=[frames], padding=True, return_tensors="pt",
                                return_mm_token_type_ids=True, videos_kwargs=kwargs)
        inputs = {key: value for key, value in inputs.items() if value is not None}
        if any("audio" in key or key in ("input_features", "feature_attention_mask", "pixel_values", "image_grid_thw") for key in inputs):
            raise RuntimeError("The vision-only processor returned forbidden audio/image inputs")
        allowed = {"input_ids", "attention_mask", "pixel_values_videos", "video_grid_thw", "mm_token_type_ids"}
        if set(inputs) - allowed or not allowed <= set(inputs):
            raise ValueError("Unexpected processor model inputs: " + str(sorted(set(inputs) - allowed)))
        if inputs["mm_token_type_ids"].shape != inputs["input_ids"].shape:
            raise ValueError("Multimodal token-type IDs must align before the extraction prefix")
        video_mask = inputs["input_ids"] == self.model.config.video_token_id
        expected_video_tokens = int(inputs["video_grid_thw"].prod().item()) // (vp.merge_size ** 2)
        if (int(video_mask.sum().item()) != expected_video_tokens or expected_video_tokens == 0 or
                not bool((inputs["mm_token_type_ids"][video_mask] == 2).all()) or
                int((inputs["mm_token_type_ids"] == 2).sum().item()) != expected_video_tokens or
                not bool(((inputs["mm_token_type_ids"] == 0) | (inputs["mm_token_type_ids"] == 2)).all())):
            raise ValueError("Video placeholders and Qwen3 modality-type IDs do not match the visual grid")
        prefix = torch.tensor([self.prefix_ids], dtype=inputs["input_ids"].dtype)
        inputs["input_ids"] = torch.cat((inputs["input_ids"], prefix), dim=-1)
        inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(prefix)), dim=-1)
        if "mm_token_type_ids" in inputs:
            inputs["mm_token_type_ids"] = torch.cat((inputs["mm_token_type_ids"], torch.zeros_like(prefix)), dim=-1)
        if inputs["input_ids"].shape[0] != 1 or not bool(inputs["attention_mask"][0, -1]):
            raise ValueError("Expected one unpadded extraction position")
        geometry = validate_processed_geometry(inputs["video_grid_thw"].cpu().numpy(),
            len(frames), int(inputs["input_ids"].shape[-1]))
        inputs = {key: value.to(device="cuda:0", dtype=torch.bfloat16) if torch.is_tensor(value) and value.is_floating_point()
                  else value.to("cuda:0") if torch.is_tensor(value) else value for key, value in inputs.items()}
        self.model.model.rope_deltas = None
        torch.cuda.reset_peak_memory_stats(0)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.model.model(**inputs, use_cache=False, return_dict=True, output_hidden_states=False)
            last = output.last_hidden_state[:, -1:, :]
            if tuple(last.shape) != (1, 1, self.width):
                raise RuntimeError("Wrong Qwen3 last hidden-state geometry")
            native = self.model.lm_head(last)[0, 0, self.value_ids].detach().float().cpu().numpy()
            hidden = last[0, 0].detach().float().cpu().numpy()
        torch.cuda.synchronize(0)
        self.check_frozen()
        if not np.isfinite(hidden).all() or not np.isfinite(native).all():
            raise FloatingPointError("Nonfinite frozen visual evidence")
        audit.update(geometry, elapsed_seconds=time.monotonic() - started,
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved(0)),
            supplied_text_in_prompt=False, gold_or_id_in_prompt=False,
            prompt_hash=fingerprint([SYSTEM_PROMPT, INSTRUCTION, PREFIX]))
        del inputs, output, last, frames
        self.model.model.rope_deltas = None
        return hidden, native, audit


def setup_gpu(args):
    if str(args.gpu) != "2" or float(args.memory_gib) != SETTINGS["memory_limit_gib"]:
        raise ValueError("This expert is fixed to physical GPU 2 with a 32 GiB allocator ceiling")
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        raise RuntimeError("GPU visibility must be set before CUDA initialization")
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Physical GPU 2 is unavailable")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., 32 * 1024 ** 3 / total), 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SETTINGS["seed"])
    torch.cuda.manual_seed_all(SETTINGS["seed"])
    return torch


def extract_run(args):
    root, source = Path(args.output_dir).resolve(), Path(args.source_run).resolve()
    if root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Expert output must be separate from the immutable producer run")
    with output_lock(root):
        try:
            emit(root, "verifying_source_and_model", command=args.command)
            c = checked_source(source)
            identity = initialize(args, c)
            tasks = video_tasks(c)
            preflight, plan = preflight_plan(root, c, tasks, identity)
            torch = setup_gpu(args)
            started = time.monotonic()
            extractor = QwenVisualExtractor(args.model_path, identity["identity"]["hidden_size"])
            load_seconds = time.monotonic() - started
            emit(root, "hashing_frozen_quantized_model")
            frozen_sha = extractor.frozen_digest()
            frozen = {"algorithm": VERSION, "identity_hash": identity["identity_hash"],
                "frozen_sha256": frozen_sha, "hidden_size": extractor.width,
                "native_emotion_token_ids": extractor.value_ids, "prefix_token_ids": extractor.prefix_ids,
                "weights_frozen": True, "optimizer_created": False}
            json_once(root / "model-frozen.json", frozen)
            chosen = preflight if args.command == "preflight" else preflight + [t for t in tasks if t not in preflight]
            entries, elapsed, entries_by_key = [], [], {}
            for i, task in enumerate(chosen):
                cached = load_feature(root, identity, task, frozen_sha)
                reused = cached is not None
                if cached is None:
                    hidden, native, diagnostics = extractor.extract(task, c["data_root"])
                    cached = save_feature(root, identity, task, frozen_sha, hidden, native, diagnostics)
                marker, _, meta = cached
                entries.append(marker)
                entries_by_key[task["key"]] = marker
                elapsed.append(meta["diagnostics"]["elapsed_seconds"])
                emit(root, "caching_training_vision", completed=i + 1, total=len(chosen),
                    video=task["video"], cache_reused=reused,
                    seconds=elapsed[-1], peak_reserved_bytes=meta["diagnostics"]["peak_reserved_bytes"])
                if i + 1 == SETTINGS["preflight_videos"]:
                    if extractor.frozen_digest() != frozen_sha:
                        raise RuntimeError("Quantized model changed during the 32-video preflight")
                    report = {"algorithm": VERSION, "identity_hash": identity["identity_hash"],
                        "frozen_sha256": frozen_sha, "videos": 32,
                        "keys": [t["key"] for t in preflight],
                        "archive_sha256": [entry["archive_sha256"] for entry in entries],
                        "median_seconds_per_video": float(np.median(elapsed)),
                        "mean_seconds_per_video": float(np.mean(elapsed)),
                        "estimated_2104_video_seconds_excluding_load": float(np.mean(elapsed) * len(tasks)),
                        "all_audio_isolated": True, "all_prompts_label_and_transcript_free": True,
                        "all_media_budgets_passed": True, "model_checksum_unchanged": True,
                        "covers_training_maximum_frames": True,
                        "maximum_sampled_frames_in_training": plan["maximum_sampled_frames_in_training"],
                        "covers_64_frame_cap": plan["covers_64_frame_cap"],
                        "timing_estimate_note": "Diagnostic extrapolation from first31 plus maximum-frame clip; not an ETA or random sample.",
                        "sampling_note": SAMPLING_NOTE}
                    json_once(root / "preflight.json", report)
                    emit(root, "preflight_passed", mean_seconds_per_video=report["mean_seconds_per_video"],
                         model_load_seconds=load_seconds, estimated_cache_seconds=report["estimated_2104_video_seconds_excluding_load"])
            if extractor.frozen_digest() != frozen_sha:
                raise RuntimeError("Quantized visual model changed during extraction")
            if args.command == "extract":
                manifest = {"algorithm": VERSION, "identity_hash": identity["identity_hash"],
                    "frozen_sha256": frozen_sha, "keys": [t["key"] for t in tasks],
                    "entries": [entries_by_key[t["key"]] for t in tasks],
                    "videos": len(tasks), "split": "train", "mode": "vision"}
                json_once(root / "features/train/complete.json", manifest)
                VisionBank(root, identity, tasks)
            emit(root, "complete", command=args.command, videos=len(chosen), model_checksum_unchanged=True,
                 validation_inference=False, test_inference=False)
            return identity
        except Exception as exc:
            write_json(root / "extract.error.json", {"type": type(exc).__name__, "message": str(exc),
                                                      "traceback": traceback.format_exc()})
            emit(root, "failed", error=str(exc))
            raise


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("command", choices=("preflight", "extract", "evaluate"))
    value.add_argument("--source-run", type=Path, required=True)
    value.add_argument("--model-path", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--gpu", default="2")
    value.add_argument("--memory-gib", type=float, default=32.)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    args.source_run, args.model_path, args.output_dir = (p.resolve() for p in (args.source_run, args.model_path, args.output_dir))
    if args.command == "evaluate":
        from scripts.vision_expert_cpu import evaluate_run
        return evaluate_run(args)
    return extract_run(args)


if __name__ == "__main__":
    main()
