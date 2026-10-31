#!/usr/bin/env python3
"""Clean a released labeled split for the VL and Omni SFT pipelines."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EMOTIONS = {"positive", "negative", "neutral"}
SAMPLE_TYPES = {"Aligned", "Conflict"}
PROTOCOLS = {"VT", "VA"}
EXPECTED_MODALITIES = {
    "VT": {"M1": "vision", "M2": "text", "M12": "vision+text"},
    "VA": {"M1": "vision", "M2": "audio", "M12": "vision+audio"},
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Validate a released labeled split and create deterministic, "
            "model-specific JSONL files without known contradictory supervision."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "Dataset" / "train.jsonl",
        help="Released labeled JSONL to clean.",
    )
    parser.add_argument(
        "--expected-split",
        choices=("train", "val"),
        default="train",
        help="Required value of each input record's split field.",
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
        default=repo_root / "data" / "cleaned" / "use_in_main_true",
        help="Directory for split-prefixed cleaned JSONL files and the report.",
    )
    parser.add_argument(
        "--ignore-use-in-main",
        action="store_true",
        help="Keep structurally valid records regardless of use_in_main.",
    )
    parser.add_argument(
        "--baseline-aligned-validation",
        action="store_true",
        help=(
            "Preserve every validation record and match baseline inputs: VL uses "
            "VT=vision+text and VA=vision; Omni uses VT=vision+text and "
            "VA=vision+audio."
        ),
    )
    return parser.parse_args()


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


def validate_record(
    record: dict[str, Any], dataset_root: Path, expected_split: str = "train"
) -> list[str]:
    reasons: list[str] = []
    protocol = record.get("protocol")
    sample_id = record.get("sample_id")
    split_group_id = record.get("split_group_id")
    video = record.get("video")
    text = record.get("text_content")
    views = record.get("views")

    if not isinstance(sample_id, str) or not sample_id:
        reasons.append("invalid_sample_id")
    if not isinstance(split_group_id, str) or not split_group_id:
        reasons.append("invalid_split_group_id")
    if record.get("split") != expected_split:
        reasons.append(f"not_{expected_split}_split")
    if protocol not in PROTOCOLS:
        reasons.append("invalid_protocol")
    if record.get("sample_type") not in SAMPLE_TYPES:
        reasons.append("invalid_sample_type")
    if not isinstance(text, str) or not text.strip():
        reasons.append("empty_text_content")
    if not isinstance(video, str) or not video:
        reasons.append("invalid_video_path")
    elif Path(video).is_absolute() or not (dataset_root / video).is_file():
        reasons.append("missing_or_absolute_video_path")

    if not isinstance(views, dict):
        reasons.append("invalid_views")
        return reasons

    expected = EXPECTED_MODALITIES.get(protocol, {})
    for view_name in ("M1", "M2", "M12"):
        view = views.get(view_name)
        if not isinstance(view, dict):
            reasons.append(f"invalid_{view_name}_view")
            continue
        if view.get("label") not in EMOTIONS:
            reasons.append(f"invalid_{view_name}_label")
        if expected and view.get("modality") != expected[view_name]:
            reasons.append(f"invalid_{view_name}_modality")
    return reasons


def target(record: dict[str, Any]) -> dict[str, str]:
    return {
        "pred_emotion": record["views"]["M12"]["label"],
        "is_uncertain": "yes" if record["sample_type"] == "Conflict" else "no",
    }


def paired_targets(records: dict[str, dict[str, Any]]) -> dict[str, dict[str, str]]:
    return {protocol: target(record) for protocol, record in sorted(records.items())}


def common_output(
    record: dict[str, Any],
    group: dict[str, dict[str, Any]],
    *,
    training_protocol: str,
    input_modalities: list[str],
    effective_sample_type: str,
    cleaning_action: str,
) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "source_sample_ids": [group[key]["sample_id"] for key in sorted(group)],
        "split_group_id": record["split_group_id"],
        "source_dataset": record.get("source_dataset"),
        "source_protocol": record["protocol"],
        "training_protocol": training_protocol,
        "input_modalities": input_modalities,
        "video": record["video"],
        "text": record["text_content"].strip(),
        "audio": record["video"] if "audio" in input_modalities else None,
        "pred_emotion": record["views"]["M12"]["label"],
        "is_uncertain": "yes" if effective_sample_type == "Conflict" else "no",
        "sample_type": effective_sample_type,
        "original_sample_type": record["sample_type"],
        "dominant_modality": record.get("dominant_modality"),
        "quality_flags": record.get("quality_flags", []),
        "source_is_generated": bool(record.get("source_is_generated", False)),
        "use_in_main": bool(record.get("use_in_main", False)),
        "cleaning_action": cleaning_action,
        "source_targets": paired_targets(group),
    }


def build_vl_records(
    groups: dict[str, dict[str, dict[str, Any]]]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    outputs: list[dict[str, Any]] = []
    actions: Counter[str] = Counter()
    for group in sorted(groups.values(), key=lambda item: min(r["_source_line"] for r in item.values())):
        if "VT" in group:
            chosen = group["VT"]
            action = "kept_vt_over_paired_va" if "VA" in group else "kept_unpaired_vt"
        else:
            chosen = group["VA"]
            action = "converted_unpaired_va_to_vt"

        actions[action] += 1
        outputs.append(
            common_output(
                chosen,
                group,
                training_protocol="VT",
                input_modalities=["vision", "text"],
                effective_sample_type=chosen["sample_type"],
                cleaning_action=action,
            )
        )
    return outputs, actions


def build_vl_records_preserving_source(
    records: list[dict[str, Any]], groups: dict[str, dict[str, dict[str, Any]]]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    outputs: list[dict[str, Any]] = []
    actions: Counter[str] = Counter()
    for record in records:
        group = groups[record["split_group_id"]]
        if record["protocol"] == "VT":
            action = "kept_vt_source_record_for_baseline_eval"
            training_protocol = "VT"
            input_modalities = ["vision", "text"]
        else:
            action = "kept_va_source_record_for_visual_only_eval"
            training_protocol = "V"
            input_modalities = ["vision"]
        actions[action] += 1
        outputs.append(
            common_output(
                record,
                group,
                training_protocol=training_protocol,
                input_modalities=input_modalities,
                effective_sample_type=record["sample_type"],
                cleaning_action=action,
            )
        )
    return outputs, actions


def build_omni_records(
    records: list[dict[str, Any]],
    groups: dict[str, dict[str, dict[str, Any]]],
    *,
    baseline_aligned_validation: bool = False,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    outputs: list[dict[str, Any]] = []
    actions: Counter[str] = Counter()
    for record in records:
        group = groups[record["split_group_id"]]
        if record["protocol"] == "VT":
            effective_sample_type = record["sample_type"]
            action = "kept_vt"
            training_protocol = "VT"
            input_modalities = ["vision", "text"]
        else:
            paired_types = {item["sample_type"] for item in group.values()}
            effective_sample_type = "Conflict" if "Conflict" in paired_types else "Aligned"
            if baseline_aligned_validation:
                action = (
                    "relabelled_va_from_paired_conflict_union_for_baseline_eval"
                    if effective_sample_type != record["sample_type"]
                    else "kept_va_with_audio_without_text_for_baseline_eval"
                )
                training_protocol = "VA"
                input_modalities = ["vision", "audio"]
            else:
                action = (
                    "relabelled_va_from_paired_conflict_union"
                    if effective_sample_type != record["sample_type"]
                    else "kept_va_with_text_and_audio"
                )
                training_protocol = "VTA"
                input_modalities = ["vision", "text", "audio"]

        actions[action] += 1
        outputs.append(
            common_output(
                record,
                group,
                training_protocol=training_protocol,
                input_modalities=input_modalities,
                effective_sample_type=effective_sample_type,
                cleaning_action=action,
            )
        )
    return outputs, actions


def distribution(records: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record.get(key)) for record in records).items()))


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary_path, path)
    path.chmod(0o644)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary_path, path)
    path.chmod(0o644)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_path = args.input.absolute()
    dataset_root = args.dataset_root.absolute()
    output_dir = args.output_dir.absolute()

    if args.baseline_aligned_validation and args.expected_split != "val":
        raise ValueError("--baseline-aligned-validation requires --expected-split val")

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)

    raw_records = read_jsonl(input_path)
    seen_sample_ids: set[str] = set()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()

    for record in raw_records:
        reasons = validate_record(record, dataset_root, args.expected_split)
        sample_id = record.get("sample_id")
        if isinstance(sample_id, str):
            if sample_id in seen_sample_ids:
                reasons.append("duplicate_sample_id")
            seen_sample_ids.add(sample_id)
        if not args.ignore_use_in_main and record.get("use_in_main") is not True:
            reasons.append("use_in_main_false")

        if reasons:
            for reason in set(reasons):
                rejection_reasons[reason] += 1
            rejected.append(
                {
                    "sample_id": sample_id,
                    "split_group_id": record.get("split_group_id"),
                    "source_line": record.get("_source_line"),
                    "reasons": sorted(set(reasons)),
                }
            )
        else:
            accepted.append(record)

    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in accepted:
        group = groups[record["split_group_id"]]
        protocol = record["protocol"]
        if protocol in group:
            raise ValueError(
                f"Multiple {protocol} records in split group {record['split_group_id']}"
            )
        group[protocol] = record

    emotion_mismatch_groups = [
        group_id
        for group_id, group in groups.items()
        if len({item["views"]["M12"]["label"] for item in group.values()}) > 1
    ]
    if emotion_mismatch_groups:
        preview = ", ".join(emotion_mismatch_groups[:5])
        raise ValueError(f"Paired groups disagree on M12 emotion: {preview}")

    if args.baseline_aligned_validation:
        vl_records, vl_actions = build_vl_records_preserving_source(
            accepted, groups
        )
    else:
        vl_records, vl_actions = build_vl_records(groups)
    omni_records, omni_actions = build_omni_records(
        accepted,
        groups,
        baseline_aligned_validation=args.baseline_aligned_validation,
    )

    vl_path = output_dir / f"{args.expected_split}_vl_clean.jsonl"
    omni_path = output_dir / f"{args.expected_split}_omni_clean.jsonl"
    rejected_path = output_dir / f"{args.expected_split}_rejected.jsonl"
    report_path = output_dir / "cleaning_report.json"
    write_jsonl(vl_path, vl_records)
    write_jsonl(omni_path, omni_records)
    write_jsonl(rejected_path, rejected)

    report = {
        "schema_version": 1,
        "split": args.expected_split,
        "variant": "all" if args.ignore_use_in_main else "use_in_main_true",
        "input": {
            "path": path_for_report(input_path, repo_root),
            "records": len(raw_records),
        },
        "filtering": {
            "use_in_main_policy": (
                "ignored" if args.ignore_use_in_main else "require_true"
            ),
            "vl_source_record_policy": (
                "preserve_all_with_baseline_inputs"
                if args.baseline_aligned_validation
                else "collapse_equivalent_groups"
            ),
            "evaluation_input_policy": (
                "baseline_aligned"
                if args.baseline_aligned_validation
                else "training_default"
            ),
            "accepted_records": len(accepted),
            "rejected_records": len(rejected),
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
            "accepted_groups": len(groups),
            "paired_groups": sum(len(group) == 2 for group in groups.values()),
        },
        "vl": {
            "path": path_for_report(vl_path, repo_root),
            "records": len(vl_records),
            "actions": dict(sorted(vl_actions.items())),
            "source_protocol": distribution(vl_records, "source_protocol"),
            "sample_type": distribution(vl_records, "sample_type"),
            "pred_emotion": distribution(vl_records, "pred_emotion"),
            "is_uncertain": distribution(vl_records, "is_uncertain"),
            "use_in_main": distribution(vl_records, "use_in_main"),
        },
        "omni": {
            "path": path_for_report(omni_path, repo_root),
            "records": len(omni_records),
            "actions": dict(sorted(omni_actions.items())),
            "source_protocol": distribution(omni_records, "source_protocol"),
            "training_protocol": distribution(omni_records, "training_protocol"),
            "sample_type": distribution(omni_records, "sample_type"),
            "pred_emotion": distribution(omni_records, "pred_emotion"),
            "is_uncertain": distribution(omni_records, "is_uncertain"),
            "use_in_main": distribution(omni_records, "use_in_main"),
        },
        "notes": [
            "Conflict samples are preserved; conflict means the model should output is_uncertain=yes.",
            (
                "VL preserves every source record with baseline inputs: VT uses vision+text and VA uses vision only."
                if args.baseline_aligned_validation
                else "VL keeps VT when VT and VA share a split_group_id because both become vision+text inputs."
            ),
            (
                "Omni baseline validation inputs are VT=vision+text and VA=vision+audio; supplied text is withheld for VA."
                if args.baseline_aligned_validation
                else "Omni training inputs are VT=vision+text and VA=vision+text+audio."
            ),
            "Omni VA uncertainty is the union of paired VT and VA conflict labels.",
            "For unpaired VA records, no VT text-view label exists, so the original VA target is retained.",
        ],
    }
    write_json(report_path, report)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
