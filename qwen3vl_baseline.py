#!/usr/bin/env python3
"""Zero-shot Qwen3-VL baseline for the video emotion benchmark.

Qwen3-VL consumes video frames and text, but not an audio stream. VT samples
therefore use both supported modalities. VA samples are evaluated as a clearly
marked visual-only limitation of this model and are reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


EMOTIONS = ("positive", "negative", "neutral")
VALID_UNCERTAINTY = {"yes", "no"}
SYSTEM_PROMPT = (
    "You are a careful multimodal emotion classifier. Return exactly one JSON "
    "object with string keys pred_emotion, is_uncertain, and "
    "clarifying_question. pred_emotion must be positive, negative, or neutral. "
    "is_uncertain must be yes or no. Use none as clarifying_question when "
    "is_uncertain is no. When uncertain, ask one brief, neutral, non-leading "
    "question that would help resolve the emotion. Do not include markdown or "
    "any text outside the JSON object."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Dataset"))
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/Qwen3-VL-4B-Instruct")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--run-name", default="qwen3vl_baseline_val")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--min-frames", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--min-visual-tokens",
        type=int,
        default=None,
        help="Optional minimum total Qwen3-VL visual token budget.",
    )
    parser.add_argument(
        "--max-visual-tokens",
        type=int,
        default=None,
        help="Optional maximum total Qwen3-VL visual token budget.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N rows. Intended for smoke tests.",
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Run one exact sample ID. Intended for default-config stress tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start a fresh run even if progress files already exist.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def infer_protocol(row: dict[str, Any]) -> str:
    protocol = row.get("protocol")
    if protocol in {"VT", "VA"}:
        return protocol

    parts = str(row["sample_id"]).split(":")
    if len(parts) >= 2 and parts[1] in {"VT", "VA"}:
        return parts[1]
    if len(parts) >= 2 and parts[1] == "accept_a_va":
        return "VA"
    if len(parts) >= 2 and parts[1] in {"accept_a_svt", "accept_c_svt"}:
        return "VT"
    raise ValueError(f"Cannot infer protocol from sample_id={row['sample_id']!r}")


def dialogue_for(row: dict[str, Any]) -> str:
    return str(row.get("text_content", row.get("text", ""))).strip()


def prompt_for(row: dict[str, Any], protocol: str) -> str:
    common = (
        "Classify the person's overall emotion in this sample. Treat visible "
        "expression, body language, and the second modality as separate evidence. "
        "If the available evidence conflicts or is insufficient for a reliable "
        "judgment, answer is_uncertain=yes. Otherwise answer no."
    )
    if protocol == "VT":
        dialogue = dialogue_for(row)
        return (
            f"Protocol: VT (vision plus supplied text). {common}\n"
            f"Supplied text: {json.dumps(dialogue, ensure_ascii=False)}"
        )

    return (
        "Protocol: VA (vision plus audio). This Qwen3-VL baseline can inspect "
        "the visual frames but cannot consume the audio stream. Do not invent "
        "speech content, vocal tone, or other unheard evidence. Treat missing "
        "audio as a limitation when deciding whether the overall multimodal "
        f"judgment is uncertain. {common}"
    )


def first_json_object(raw: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw):
        try:
            value, _ = decoder.raw_decode(raw[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def normalize_response(raw: str) -> dict[str, Any]:
    obj = first_json_object(raw)
    parse_status = "json"

    if obj is None:
        parse_status = "fallback"
        lower = raw.lower()
        found = [label for label in EMOTIONS if re.search(rf"\b{label}\b", lower)]
        emotion = found[0] if len(found) == 1 else "neutral"

        uncertain_match = re.search(
            r"(?:is_uncertain|uncertain)\s*[:=]\s*[\"']?(yes|no|true|false)",
            lower,
        )
        uncertain_raw = uncertain_match.group(1) if uncertain_match else "yes"
        question = "Could you clarify what emotion you intended to express?"
    else:
        emotion = str(obj.get("pred_emotion", "")).lower().strip()
        uncertain_raw = str(obj.get("is_uncertain", "")).lower().strip()
        question = str(obj.get("clarifying_question", "")).strip()

    if emotion not in EMOTIONS:
        emotion = "neutral"
        parse_status += "+invalid_emotion"

    if uncertain_raw in {"yes", "y", "true", "1"}:
        uncertain = "yes"
    elif uncertain_raw in {"no", "n", "false", "0"}:
        uncertain = "no"
    else:
        uncertain = "yes"
        parse_status += "+invalid_uncertainty"

    if uncertain == "no":
        question = "none"
    elif question.lower() in {"", "none", "null", "n/a"}:
        question = "Could you clarify what emotion you intended to express?"
        parse_status += "+missing_question"

    return {
        "pred_emotion": emotion,
        "is_uncertain": uncertain,
        "clarifying_question": question.replace("\n", " ").strip(),
        "parse_status": parse_status,
    }


def metric_block(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"samples": 0}

    y_true = [record["true_emotion"] for record in records]
    y_pred = [record["pred_emotion"] for record in records]
    conflict = [record for record in records if record["true_sample_type"] == "Conflict"]
    aligned = [record for record in records if record["true_sample_type"] == "Aligned"]

    macro_f1 = f1_score(
        y_true, y_pred, labels=list(EMOTIONS), average="macro", zero_division=0
    )
    present_labels = [label for label in EMOTIONS if label in set(y_true)]
    present_macro_f1 = f1_score(
        y_true, y_pred, labels=present_labels, average="macro", zero_division=0
    )
    conflict_accuracy = (
        float(np.mean([record["is_uncertain"] == "yes" for record in conflict]))
        if conflict
        else None
    )
    aligned_calibration = (
        float(np.mean([record["is_uncertain"] == "no" for record in aligned]))
        if aligned
        else None
    )
    score = (
        0.5 * macro_f1 + 0.3 * conflict_accuracy + 0.2 * aligned_calibration
        if conflict_accuracy is not None and aligned_calibration is not None
        else None
    )

    return {
        "samples": len(records),
        "true_emotion_counts": dict(Counter(y_true)),
        "pred_emotion_counts": dict(Counter(y_pred)),
        "true_sample_type_counts": dict(
            Counter(record["true_sample_type"] for record in records)
        ),
        "uncertainty_counts": dict(
            Counter(record["is_uncertain"] for record in records)
        ),
        "macro_f1_three_classes": float(macro_f1),
        "macro_f1_present_classes": float(present_macro_f1),
        "conflict_identification_accuracy": conflict_accuracy,
        "aligned_calibration": aligned_calibration,
        "overjudgment_rate": (
            1.0 - aligned_calibration if aligned_calibration is not None else None
        ),
        "competition_score": score,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(EMOTIONS),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix_labels": list(EMOTIONS),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(EMOTIONS)
        ).tolist(),
    }


def grouped_metrics(
    records: list[dict[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    values = sorted({str(record[key]) for record in records})
    return {
        value: metric_block([record for record in records if str(record[key]) == value])
        for value in values
    }


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def config_fingerprint(config: dict[str, Any]) -> str:
    stable = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(stable).hexdigest()[:16]


def validate_args(args: argparse.Namespace) -> None:
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.min_frames is not None and args.min_frames <= 0:
        raise ValueError("--min-frames must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if (
        args.min_frames is not None
        and args.max_frames is not None
        and args.max_frames < args.min_frames
    ):
        raise ValueError("Frame limits are invalid")
    if any(value is not None and value % 2 for value in (args.min_frames, args.max_frames)):
        raise ValueError("Qwen video frame limits must be multiples of 2")
    if args.min_visual_tokens is not None and args.min_visual_tokens <= 0:
        raise ValueError("--min-visual-tokens must be positive")
    if args.max_visual_tokens is not None and args.max_visual_tokens <= 0:
        raise ValueError("--max-visual-tokens must be positive")
    if (
        args.min_visual_tokens is not None
        and args.max_visual_tokens is not None
        and args.max_visual_tokens < args.min_visual_tokens
    ):
        raise ValueError("Visual token limits are invalid")
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run with the LLaMAFac environment and expose the "
            "requested GPU, for example CUDA_VISIBLE_DEVICES=7."
        )


def main() -> None:
    args = parse_args()
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

    dataset_signature = hashlib.sha256(input_path.read_bytes()).hexdigest()
    script_signature = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    experiment = {
        "model_path": str(args.model_path.resolve()),
        "input_path": str(input_path.resolve()),
        "dataset_sha256": dataset_signature,
        "script_sha256": script_signature,
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
        "fps": args.fps,
        "min_frames": args.min_frames,
        "max_frames": args.max_frames,
        "min_visual_tokens": args.min_visual_tokens,
        "max_visual_tokens": args.max_visual_tokens,
        "max_new_tokens": args.max_new_tokens,
        "system_prompt": SYSTEM_PROMPT,
        "va_input_limitation": "visual frames only; Qwen3-VL receives no audio",
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
            f"Found {details_path} without matching run configuration. Use a new "
            "--run-name or pass --overwrite."
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
                "rows": len(rows),
                "resumed": len(completed),
                "gpu": torch.cuda.get_device_name(0),
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "fingerprint": fingerprint,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    if args.fps is not None:
        processor.video_processor.fps = args.fps
    if args.min_frames is not None:
        processor.video_processor.min_frames = args.min_frames
    if args.max_frames is not None:
        processor.video_processor.max_frames = args.max_frames
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
    ).eval()

    @torch.inference_mode()
    def infer_one(row: dict[str, Any]) -> dict[str, Any]:
        protocol = infer_protocol(row)
        video_path = args.data_root / str(row["video"])
        if not video_path.is_file():
            raise FileNotFoundError(video_path)

        prompt = prompt_for(row, protocol)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(video_path),
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]
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

        template_kwargs: dict[str, Any] = {}
        if processor_kwargs:
            template_kwargs["processor_kwargs"] = processor_kwargs
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        ).to("cuda:0")
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
    for row in tqdm(pending_rows, desc="Qwen3-VL baseline"):
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
        except Exception as exc:  # Keep the run resumable while exposing every failure.
            prediction = {
                "pred_emotion": "neutral",
                "is_uncertain": "yes",
                "clarifying_question": (
                    "Could you clarify what emotion you intended to express?"
                ),
                "parse_status": "inference_error",
                "raw_response": "",
                "input_prompt": prompt_for(row, protocol),
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
    submission_columns = [
        "sample_id",
        "pred_emotion",
        "is_uncertain",
        "clarifying_question",
    ]
    pd.DataFrame(ordered)[submission_columns].to_csv(predictions_path, index=False)

    result: dict[str, Any] = {
        "run_name": args.run_name,
        "experiment_fingerprint": fingerprint,
        "split": args.split,
        "model": str(args.model_path),
        "samples": len(ordered),
        "elapsed_seconds_this_invocation": round(time.time() - start, 2),
        "inference_errors": sum(bool(record["inference_error"]) for record in ordered),
        "parse_status_counts": dict(
            Counter(record["parse_status"] for record in ordered)
        ),
        "max_peak_cuda_allocated_bytes": max(
            record["peak_cuda_allocated_bytes"] for record in ordered
        ),
        "max_peak_cuda_reserved_bytes": max(
            record["peak_cuda_reserved_bytes"] for record in ordered
        ),
        "protocol_counts": dict(Counter(record["protocol"] for record in ordered)),
        "va_input_limitation": "visual frames only; no audio was consumed",
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
