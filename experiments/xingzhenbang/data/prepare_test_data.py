#!/usr/bin/env python3
"""Validate and normalize the released unlabeled test split for inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROTOCOLS = {"VT", "VA"}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "Dataset" / "test_release.jsonl",
        help="Released unlabeled test JSONL.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "Dataset",
        help="Root used to validate relative video paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "cleaned" / "test",
        help="Directory for model-specific test manifests and the report.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_for_report(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            record["_source_line"] = line_number
            records.append(record)
    return records


def infer_protocol(sample_id: str) -> str:
    parts = sample_id.split(":")
    if len(parts) >= 2 and parts[1] in PROTOCOLS:
        return parts[1]
    if len(parts) >= 2 and parts[1] == "accept_a_va":
        return "VA"
    if len(parts) >= 2 and parts[1] in {"accept_a_svt", "accept_c_svt"}:
        return "VT"
    raise ValueError(f"Cannot infer protocol from sample_id={sample_id!r}")


def source_metadata(sample_id: str, video: str) -> tuple[str, str, str]:
    parts = sample_id.split(":")
    if len(parts) >= 4 and parts[1] in PROTOCOLS:
        source_dataset = parts[0]
        source_id = Path(video).relative_to("videos").with_suffix("").as_posix()
    elif len(parts) >= 3 and parts[0] == "gen":
        source_dataset = parts[1]
        source_id = ":".join(parts[2:])
    else:
        raise ValueError(f"Cannot infer source metadata from sample_id={sample_id!r}")
    return source_dataset, source_id, f"{source_dataset}:{source_id}"


def validate_record(record: dict[str, Any], dataset_root: Path) -> list[str]:
    reasons: list[str] = []
    if set(record) != {"sample_id", "video", "text", "_source_line"}:
        reasons.append("unexpected_schema")

    sample_id = record.get("sample_id")
    video = record.get("video")
    text = record.get("text")
    if not isinstance(sample_id, str) or not sample_id:
        reasons.append("invalid_sample_id")
    else:
        try:
            infer_protocol(sample_id)
        except ValueError:
            reasons.append("unknown_protocol")
    if not isinstance(video, str) or not video:
        reasons.append("invalid_video_path")
    elif Path(video).is_absolute() or not (dataset_root / video).is_file():
        reasons.append("missing_or_absolute_video_path")
    if not isinstance(text, str) or not text.strip():
        reasons.append("empty_text")
    return reasons


def normalized_record(record: dict[str, Any], model_family: str) -> dict[str, Any]:
    sample_id = record["sample_id"]
    video = record["video"]
    protocol = infer_protocol(sample_id)
    source_dataset, source_id, split_group_id = source_metadata(sample_id, video)

    if protocol == "VT":
        training_protocol = "VT"
        input_modalities = ["vision", "text"]
        action = "kept_vt_with_text_for_test"
    elif model_family == "vl":
        training_protocol = "V"
        input_modalities = ["vision"]
        action = "kept_va_visual_only_for_test"
    else:
        training_protocol = "VA"
        input_modalities = ["vision", "audio"]
        action = "kept_va_with_audio_without_text_for_test"

    return {
        "sample_id": sample_id,
        "split_group_id": split_group_id,
        "source_dataset": source_dataset,
        "source_id": source_id,
        "source_protocol": protocol,
        "training_protocol": training_protocol,
        "input_modalities": input_modalities,
        "video": video,
        "text": record["text"].strip(),
        "audio": video if "audio" in input_modalities else None,
        "split": "test",
        "source_line": record["_source_line"],
        "cleaning_action": action,
    }


def distribution(records: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record.get(key)) for record in records).items()))


def modality_distribution(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(Counter("+".join(record["input_modalities"]) for record in records).items())
    )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)
    path.chmod(0o644)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)
    path.chmod(0o644)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_path = args.input.absolute()
    dataset_root = args.dataset_root.absolute()
    output_dir = args.output_dir.absolute()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)

    raw_records = read_jsonl(input_path)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    rejection_reasons: Counter[str] = Counter()
    for record in raw_records:
        reasons = validate_record(record, dataset_root)
        sample_id = record.get("sample_id")
        if isinstance(sample_id, str):
            if sample_id in seen_sample_ids:
                reasons.append("duplicate_sample_id")
            seen_sample_ids.add(sample_id)
        if reasons:
            unique_reasons = sorted(set(reasons))
            rejection_reasons.update(unique_reasons)
            rejected.append(
                {
                    "sample_id": sample_id,
                    "source_line": record["_source_line"],
                    "reasons": unique_reasons,
                }
            )
        else:
            accepted.append(record)

    if rejected:
        raise ValueError(
            f"Released test set contains {len(rejected)} invalid rows; "
            f"reasons: {dict(rejection_reasons)}"
        )

    vl_records = [normalized_record(record, "vl") for record in accepted]
    omni_records = [normalized_record(record, "omni") for record in accepted]
    if [row["sample_id"] for row in vl_records] != [row["sample_id"] for row in omni_records]:
        raise RuntimeError("VL and Omni test order diverged")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in vl_records:
        groups[record["split_group_id"]].append(record)
    invalid_pairs = []
    for group_id, records in groups.items():
        if len(records) == 1:
            continue
        if (
            len(records) != 2
            or {record["source_protocol"] for record in records} != PROTOCOLS
            or len({record["video"] for record in records}) != 1
            or len({record["text"] for record in records}) != 1
        ):
            invalid_pairs.append(group_id)
    if invalid_pairs:
        raise ValueError(f"Invalid paired test groups: {invalid_pairs[:5]}")

    vl_path = output_dir / "test_vl_clean.jsonl"
    omni_path = output_dir / "test_omni_clean.jsonl"
    rejected_path = output_dir / "test_rejected.jsonl"
    report_path = output_dir / "cleaning_report.json"
    write_jsonl(vl_path, vl_records)
    write_jsonl(omni_path, omni_records)
    write_jsonl(rejected_path, rejected)

    report = {
        "schema_version": 1,
        "split": "test",
        "target_mode": "unlabeled_inference",
        "input": {
            "path": path_for_report(input_path, repo_root),
            "sha256": sha256_file(input_path),
            "records": len(raw_records),
        },
        "validation": {
            "accepted_records": len(accepted),
            "rejected_records": len(rejected),
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
            "groups": len(groups),
            "paired_vt_va_groups": sum(len(records) == 2 for records in groups.values()),
            "source_order_preserved": True,
        },
        "vl": {
            "path": path_for_report(vl_path, repo_root),
            "records": len(vl_records),
            "sha256": sha256_file(vl_path),
            "source_protocol": distribution(vl_records, "source_protocol"),
            "input_modalities": modality_distribution(vl_records),
            "actions": distribution(vl_records, "cleaning_action"),
        },
        "omni": {
            "path": path_for_report(omni_path, repo_root),
            "records": len(omni_records),
            "sha256": sha256_file(omni_path),
            "source_protocol": distribution(omni_records, "source_protocol"),
            "input_modalities": modality_distribution(omni_records),
            "actions": distribution(omni_records, "cleaning_action"),
        },
        "notes": [
            "All released test rows are preserved in their original order; paired VT/VA rows are not collapsed.",
            "The test split is unlabeled, so no emotion, uncertainty, or clarifying-question target is added.",
            "VL test inputs are VT=vision+text and VA=vision only.",
            "Omni test inputs are VT=vision+text with embedded audio disabled and VA=vision+audio with supplied text withheld.",
            "The raw Dataset/test_release.jsonl remains the submission-order source used by inference scripts.",
        ],
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
