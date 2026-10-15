#!/usr/bin/env python3
"""Evaluate a LoRA or QLoRA SFT adapter selected by its training YAML."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import torch
import transformers
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from tqdm.auto import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen25vl_baseline import (  # noqa: E402
    append_jsonl,
    config_fingerprint,
    grouped_metrics,
    infer_protocol,
    json_dump,
    load_jsonl,
    metric_block,
    normalize_response,
    utc_now,
)
from sft_config import load_config_metadata  # noqa: E402


SYSTEM_PROMPT = (
    "You are a multimodal emotion analysis assistant. Judge the overall emotion "
    "from the provided modalities, explicitly represent uncertainty when the "
    "evidence is insufficient or conflicting, and follow the requested JSON schema."
)
TARGET_INSTRUCTION = (
    "If the evidence conflicts or is insufficient for a reliable judgment, set "
    "is_uncertain to yes and ask one neutral clarifying question addressed to the "
    "current user. Otherwise set is_uncertain to no and clarifying_question to "
    "none. Return exactly one JSON object. pred_emotion must be positive, negative, "
    "or neutral; is_uncertain must be yes or no; clarifying_question must be a "
    "single string. Do not add Markdown or explanatory text."
)

MODEL_SPECS = {
    "qwen25vl": {
        "model_class": Qwen2_5_VLForConditionalGeneration,
    },
    "qwen3vl": {
        "model_class": Qwen3VLForConditionalGeneration,
    },
    "qwen25omni": {
        "model_class": Qwen2_5OmniForConditionalGeneration,
    },
}

OMNI_DEFAULT_FPS = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-method", choices=("lora", "qlora"))
    parser.add_argument("--data-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/sft/evaluation")
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override YAML-based loading. By default QLoRA configs use NF4 4-bit "
            "and LoRA configs use BF16."
        ),
    )
    parser.add_argument(
        "--attn-implementation", choices=("sdpa", "flash_attention_2"), default="sdpa"
    )
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--min-frames", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--min-visual-tokens", type=int, default=None)
    parser.add_argument("--max-visual-tokens", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def dialogue_for(row: dict[str, Any]) -> str:
    return str(row.get("text_content", row.get("text", ""))).strip()


def prompt_for(row: dict[str, Any], protocol: str, model_key: str) -> str:
    if protocol == "VT":
        modality = "The input contains the video and supplied transcript text."
        supplied_text = (
            f"\nSupplied text: {json.dumps(dialogue_for(row), ensure_ascii=False)}"
        )
    else:
        if model_key == "qwen25omni":
            modality = (
                "The input contains video frames and the video's embedded audio. "
                "Supplied transcript text is unavailable."
            )
        else:
            modality = (
                "The input contains visual video frames only. Audio and supplied "
                "transcript text are unavailable."
            )
        supplied_text = ""

    return (
        f"{modality}\n"
        "Analyze the person's overall emotion. Treat visible expression, body "
        "language, spoken or supplied text, and vocal evidence as separate signals "
        f"when those signals are available.{supplied_text}\n"
        f"{TARGET_INSTRUCTION}"
    )


def validate_args(args: argparse.Namespace) -> None:
    for name in ("fps", "min_frames", "max_frames", "min_visual_tokens", "max_visual_tokens"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if (
        args.min_frames is not None
        and args.max_frames is not None
        and args.max_frames < args.min_frames
    ):
        raise ValueError("Frame limits are invalid")
    if any(value is not None and value % 2 for value in (args.min_frames, args.max_frames)):
        raise ValueError("Qwen video frame limits must be multiples of 2")
    if (
        args.min_visual_tokens is not None
        and args.max_visual_tokens is not None
        and args.max_visual_tokens < args.min_visual_tokens
    ):
        raise ValueError("Visual token limits are invalid")
    if args.max_new_tokens <= 0 or args.checkpoint_every <= 0:
        raise ValueError("Token and checkpoint limits must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    if not (args.adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(args.adapter_path / "adapter_config.json")
    if not (args.adapter_path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(args.adapter_path / "adapter_model.safetensors")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")


def build_model(args: argparse.Namespace) -> tuple[Any, Any]:
    spec = MODEL_SPECS[args.model_key]
    if args.model_key == "qwen25omni":
        processor = Qwen2_5OmniProcessor.from_pretrained(
            args.model_path, local_files_only=True
        )
    else:
        processor = AutoProcessor.from_pretrained(
            args.model_path, local_files_only=True, trust_remote_code=True
        )
    if args.model_key == "qwen3vl":
        if args.fps is not None:
            processor.video_processor.fps = args.fps
        if args.min_frames is not None:
            processor.video_processor.min_frames = args.min_frames
        if args.max_frames is not None:
            processor.video_processor.max_frames = args.max_frames

    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
        "device_map": {"": "cuda:0"},
        "attn_implementation": args.attn_implementation,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    base_model = spec["model_class"].from_pretrained(args.model_path, **model_kwargs)
    if args.model_key == "qwen25omni":
        base_model.disable_talker()
        # LLaMAFactory trains and saves the Omni Thinker as the PEFT base model.
        # Mounting that adapter on the outer Omni container changes the expected
        # key prefix and silently leaves every LoRA tensor unloaded.
        base_model.thinker = PeftModel.from_pretrained(
            base_model.thinker,
            args.adapter_path,
            is_trainable=False,
            local_files_only=True,
        )
        model = base_model.eval()
    else:
        model = PeftModel.from_pretrained(
            base_model, args.adapter_path, is_trainable=False, local_files_only=True
        ).eval()
    return processor, model


def prepare_inputs(
    args: argparse.Namespace,
    processor: Any,
    row: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    protocol = infer_protocol(row)
    video_path = args.data_root / str(row["video"])
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    prompt = prompt_for(row, protocol, args.model_key)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path)},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    if args.model_key == "qwen25omni":
        use_audio = protocol == "VA"
        videos_kwargs: dict[str, Any] = {
            "use_audio_in_video": use_audio,
            "do_sample_frames": True,
            "fps": args.fps if args.fps is not None else OMNI_DEFAULT_FPS,
        }
        if args.min_frames is not None:
            videos_kwargs["min_frames"] = args.min_frames
        if args.max_frames is not None:
            videos_kwargs["max_frames"] = args.max_frames
        if args.min_visual_tokens is not None or args.max_visual_tokens is not None:
            native_size = processor.video_processor.size
            videos_kwargs["size"] = {
                "shortest_edge": (
                    args.min_visual_tokens * 28 * 28
                    if args.min_visual_tokens is not None
                    else native_size.shortest_edge
                ),
                "longest_edge": (
                    args.max_visual_tokens * 28 * 28
                    if args.max_visual_tokens is not None
                    else native_size.longest_edge
                ),
            }
        rendered = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        audio = audio_from_video(video_path) if use_audio else None
        inputs = processor(
            text=rendered,
            videos=str(video_path),
            audio=audio,
            padding=True,
            return_tensors="pt",
            videos_kwargs=videos_kwargs,
        ).to("cuda:0").to(torch.bfloat16)
        inputs["use_audio_in_video"] = use_audio
        return inputs, prompt

    if args.model_key == "qwen25vl":
        video_content = messages[1]["content"][0]
        if args.fps is not None:
            video_content["fps"] = args.fps
        if args.min_frames is not None:
            video_content["min_frames"] = args.min_frames
        if args.max_frames is not None:
            video_content["max_frames"] = args.max_frames
        if args.min_visual_tokens is not None:
            video_content["min_pixels"] = args.min_visual_tokens * 28 * 28
        if args.max_visual_tokens is not None:
            video_content["max_pixels"] = args.max_visual_tokens * 28 * 28
        if args.max_frames is not None and args.max_visual_tokens is not None:
            video_content["total_pixels"] = (
                args.max_frames * args.max_visual_tokens * 28 * 28
            )

        rendered = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        if isinstance(video_kwargs.get("fps"), list) and len(video_kwargs["fps"]) == 1:
            video_kwargs["fps"] = video_kwargs["fps"][0]
        inputs = processor(
            text=[rendered],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
    else:
        processor_kwargs: dict[str, Any] = {}
        if args.fps is not None:
            processor_kwargs["fps"] = args.fps
        if args.min_visual_tokens is not None or args.max_visual_tokens is not None:
            native_size = processor.video_processor.size
            processor_kwargs["size"] = {
                "shortest_edge": (
                    args.min_visual_tokens * 32 * 32
                    if args.min_visual_tokens is not None
                    else native_size.shortest_edge
                ),
                "longest_edge": (
                    args.max_visual_tokens * 32 * 32
                    if args.max_visual_tokens is not None
                    else native_size.longest_edge
                ),
            }
        template_kwargs = (
            {"processor_kwargs": processor_kwargs} if processor_kwargs else {}
        )
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        )
    return inputs.to("cuda:0"), prompt


def audio_from_video(video_path: Path, sampling_rate: int = 16_000) -> np.ndarray:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sampling_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    completed = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    audio = np.frombuffer(completed.stdout, dtype="<f4").copy()
    if audio.size == 0:
        raise RuntimeError(f"No audio samples decoded from {video_path}")
    return audio


def main() -> None:
    args = parse_args()
    args.config = resolve_path(args.config).resolve()
    training_config, config_meta = load_config_metadata(
        args.config, REPO_ROOT, expected_method=args.expected_method
    )
    args.model_key = config_meta["model_key"]
    args.method = config_meta["method"]
    if args.load_in_4bit is None:
        args.load_in_4bit = args.method == "qlora"
    args.data_root = resolve_path(args.data_root)
    args.model_path = resolve_path(
        args.model_path or Path(config_meta["model_path"])
    )
    args.adapter_path = resolve_path(
        args.adapter_path or Path(config_meta["output_dir"])
    )
    args.output_dir = resolve_path(args.output_dir)
    args.run_name = args.run_name or (
        f"{args.model_key}_{config_meta['run_name']}_{args.split}"
    )
    validate_args(args)

    input_name = "val.jsonl" if args.split == "val" else "test_release.jsonl"
    input_path = args.data_root / input_name
    rows = load_jsonl(input_path)
    if args.sample_id is not None:
        rows = [row for row in rows if str(row["sample_id"]) == args.sample_id]
        if not rows:
            raise ValueError(f"Unknown sample ID: {args.sample_id}")
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"No rows loaded from {input_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / args.run_name
    details_path = prefix.with_name(prefix.name + "_details.jsonl")
    predictions_path = prefix.with_name(prefix.name + "_predictions.csv")
    metrics_path = prefix.with_name(prefix.name + "_metrics.json")
    config_path = prefix.with_name(prefix.name + "_run_config.json")

    adapter_weights = args.adapter_path / "adapter_model.safetensors"
    experiment = {
        "training_config": str(args.config),
        "training_config_sha256": sha256_file(args.config),
        "training_method": args.method,
        "training_quantization_bit": training_config.get("quantization_bit"),
        "model_key": args.model_key,
        "model_path": str(args.model_path.resolve()),
        "adapter_path": str(args.adapter_path.resolve()),
        "adapter_sha256": sha256_file(adapter_weights),
        "input_path": str(input_path.resolve()),
        "dataset_sha256": sha256_file(input_path),
        "script_sha256": sha256_file(Path(__file__)),
        "split": args.split,
        "limit": args.limit,
        "sample_id": args.sample_id,
        "input_profile": (
            "native_default"
            if all(
                value is None
                for value in (
                    args.fps,
                    args.min_frames,
                    args.max_frames,
                    args.min_visual_tokens,
                    args.max_visual_tokens,
                )
            )
            else "custom"
        ),
        "load_in_4bit": args.load_in_4bit,
        "quantization": "bnb_nf4_double_quant" if args.load_in_4bit else "bf16",
        "attn_implementation": args.attn_implementation,
        "fps": args.fps,
        "min_frames": args.min_frames,
        "max_frames": args.max_frames,
        "min_visual_tokens": args.min_visual_tokens,
        "max_visual_tokens": args.max_visual_tokens,
        "max_new_tokens": args.max_new_tokens,
        "system_prompt": SYSTEM_PROMPT,
        "target_instruction": TARGET_INSTRUCTION,
        "validation_input_policy": {
            "VT": "video frames plus supplied text",
            "VA": (
                "video frames plus embedded audio; supplied text withheld"
                if args.model_key == "qwen25omni"
                else "visual video frames only; audio and supplied text withheld"
            ),
        },
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    fingerprint = config_fingerprint(experiment)
    run_config = {
        **experiment,
        "fingerprint": fingerprint,
        "created_at_utc": utc_now(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_device_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "rows": len(rows),
    }

    if args.overwrite:
        for path in (details_path, predictions_path, metrics_path, config_path):
            path.unlink(missing_ok=True)
    elif config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"Existing run {args.run_name!r} has a different configuration. "
                "Use a new --run-name or pass --overwrite."
            )
    elif details_path.exists():
        raise RuntimeError(
            f"Found {details_path} without a matching run configuration."
        )
    json_dump(config_path, run_config)

    completed: dict[str, dict[str, Any]] = {}
    if details_path.exists():
        for record in load_jsonl(details_path):
            if record.get("experiment_fingerprint") == fingerprint:
                completed[str(record["sample_id"])] = record

    print(
        json.dumps(
            {
                "event": "run_start",
                "model": args.model_key,
                "rows": len(rows),
                "resumed": len(completed),
                "gpu": torch.cuda.get_device_name(0),
                "quantization": experiment["quantization"],
                "fingerprint": fingerprint,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    processor, model = build_model(args)

    @torch.inference_mode()
    def infer_one(row: dict[str, Any]) -> dict[str, Any]:
        inputs, prompt = prepare_inputs(args, processor, row)
        if args.model_key == "qwen25omni":
            use_audio = bool(inputs.pop("use_audio_in_video"))
            generated_ids = model.generate(
                **inputs,
                use_audio_in_video=use_audio,
                thinker_max_new_tokens=args.max_new_tokens,
                return_audio=False,
                do_sample=False,
                use_cache=True,
            )
            input_length = inputs["input_ids"].shape[1]
            if generated_ids.shape[1] >= input_length and torch.equal(
                generated_ids[:, :input_length], inputs["input_ids"]
            ):
                trimmed = generated_ids[:, input_length:]
            else:
                trimmed = generated_ids
        else:
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]
        raw = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        normalized = normalize_response(raw)
        return {
            **normalized,
            "raw_response": raw,
            "input_prompt": prompt,
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "generated_tokens": int(trimmed.shape[1]),
        }

    pending_buffer: list[dict[str, Any]] = []
    start = time.time()
    pending_rows = [row for row in rows if str(row["sample_id"]) not in completed]
    for row in tqdm(
        pending_rows, desc=f"{args.model_key} {args.method.upper()} SFT"
    ):
        sample_start = time.time()
        torch.cuda.reset_peak_memory_stats()
        protocol = infer_protocol(row)
        base_record: dict[str, Any] = {
            "sample_id": str(row["sample_id"]),
            "protocol": protocol,
            "source_dataset": row.get("source_dataset", "unknown"),
            "video": str(row["video"]),
            "experiment_fingerprint": fingerprint,
        }
        if args.split == "val":
            base_record.update(
                {
                    "true_emotion": row["views"]["M12"]["label"],
                    "true_sample_type": row["sample_type"],
                    "use_in_main": bool(row.get("use_in_main", True)),
                }
            )
        try:
            prediction = infer_one(row)
            error = None
        except Exception as exc:
            prediction = {
                "pred_emotion": "neutral",
                "is_uncertain": "yes",
                "clarifying_question": (
                    "Could you clarify what emotion you intended to express?"
                ),
                "parse_status": "inference_error",
                "raw_response": "",
                "input_prompt": prompt_for(row, protocol, args.model_key),
                "input_tokens": None,
                "generated_tokens": None,
            }
            error = f"{type(exc).__name__}: {exc}"
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()

        record = {
            **base_record,
            **prediction,
            "inference_error": error,
            "elapsed_seconds": round(time.time() - sample_start, 4),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "completed_at_utc": utc_now(),
        }
        completed[record["sample_id"]] = record
        pending_buffer.append(record)
        if len(pending_buffer) >= args.checkpoint_every:
            append_jsonl(details_path, pending_buffer)
            pending_buffer.clear()

    if pending_buffer:
        append_jsonl(details_path, pending_buffer)

    ordered = [completed[str(row["sample_id"])] for row in rows]
    pd.DataFrame(ordered)[
        ["sample_id", "pred_emotion", "is_uncertain", "clarifying_question"]
    ].to_csv(predictions_path, index=False)

    result: dict[str, Any] = {
        "run_name": args.run_name,
        "experiment_fingerprint": fingerprint,
        "split": args.split,
        "model": str(args.model_path),
        "adapter": str(args.adapter_path),
        "quantization": experiment["quantization"],
        "samples": len(ordered),
        "elapsed_seconds_this_invocation": round(time.time() - start, 2),
        "inference_errors": sum(bool(record["inference_error"]) for record in ordered),
        "parse_status_counts": dict(Counter(record["parse_status"] for record in ordered)),
        "max_peak_cuda_allocated_bytes": max(
            record["peak_cuda_allocated_bytes"] for record in ordered
        ),
        "max_peak_cuda_reserved_bytes": max(
            record["peak_cuda_reserved_bytes"] for record in ordered
        ),
        "protocol_counts": dict(Counter(record["protocol"] for record in ordered)),
        "files": {
            "predictions": str(predictions_path),
            "details": str(details_path),
            "run_config": str(config_path),
        },
    }
    if args.split == "val":
        overall = metric_block(ordered)
        result.update(
            {
                "Macro-F1": overall["macro_f1_three_classes"],
                "Conflict_accuracy": overall["conflict_identification_accuracy"],
                "Aligned_calibration": overall["aligned_calibration"],
                "Score": overall["competition_score"],
                "overall": overall,
                "by_protocol": grouped_metrics(ordered, "protocol"),
                "by_source_dataset": grouped_metrics(ordered, "source_dataset"),
                "by_use_in_main": grouped_metrics(ordered, "use_in_main"),
            }
        )
    json_dump(metrics_path, result)

    summary = {
        "event": "run_complete",
        "model": args.model_key,
        "samples": len(ordered),
        "inference_errors": result["inference_errors"],
        "predictions": str(predictions_path),
        "metrics": str(metrics_path),
    }
    if args.split == "val":
        summary.update(
            {
                "Macro-F1": result["Macro-F1"],
                "Conflict_accuracy": result["Conflict_accuracy"],
                "Aligned_calibration": result["Aligned_calibration"],
                "Score": result["Score"],
            }
        )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
