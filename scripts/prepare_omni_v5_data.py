#!/usr/bin/env python3
"""Build source-aware Omni training groups for the v5 sampling experiment."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

from prepare_sft_data import read_jsonl, validate_sft_records, write_json, write_jsonl


FACTORS = {"aligned": 1, "svt_conflict": 1, "ch_sims_conflict": 3, "va_conflict": 3}


def group_record(record):
    if record.get("split") != "train":
        raise ValueError(f"Non-training record: {record.get('sample_id')}")
    answer = json.loads(record["messages"][-1]["content"])
    uncertain = answer.get("is_uncertain")
    if uncertain == "no":
        return "aligned"
    if uncertain != "yes":
        raise ValueError(f"Invalid uncertainty label: {record.get('sample_id')}")
    source = record.get("source_dataset")
    protocol = record.get("source_protocol")
    if source == "ch_sims_v2" and protocol == "VT":
        return "ch_sims_conflict"
    if protocol == "VA":
        return "va_conflict"
    if source == "accept_a_svt" and protocol == "VT":
        return "svt_conflict"
    raise ValueError(f"Unclassified conflict record: {record.get('sample_id')}")


def prepare(root, source_dir, output_dir):
    source_dir, output_dir = source_dir.resolve(), output_dir.resolve()
    if source_dir == output_dir:
        raise ValueError("Source and output directories must differ")
    train = read_jsonl(source_dir / "affect_sft_omni_train.jsonl")
    val_path = source_dir / "affect_sft_omni_val.jsonl"
    val = read_jsonl(val_path)
    validate_sft_records(train, "omni", root, "full")
    validate_sft_records(val, "omni", root, "emotion_only")
    train_ids = [record["sample_id"] for record in train]
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("Duplicate training sample IDs")
    if {r["split_group_id"] for r in train} & {r["split_group_id"] for r in val}:
        raise ValueError("Train/validation group overlap")

    groups = {name: [] for name in FACTORS}
    for record in train:
        groups[group_record(record)].append(record)
    if any(not records for records in groups.values()):
        raise ValueError("All four training groups must be nonempty")

    original = json.loads((root / "data/llamafactory/dataset_info.json").read_text())
    entries = {}
    for group, records in groups.items():
        name = f"affect_sft_omni_v5_{group}"
        entry = deepcopy(original["affect_sft_omni_train"])
        entry["file_name"] = f"omni_{group}.jsonl"
        if FACTORS[group] > 1:
            entry["num_samples"] = len(records) * FACTORS[group]
        entries[name] = entry
        write_jsonl(output_dir / entry["file_name"], records)
    entries["affect_sft_omni_val"] = deepcopy(original["affect_sft_omni_val"])
    entries["affect_sft_omni_val"]["file_name"] = os.path.relpath(val_path, output_dir)
    summary = {
        "policy": "LLaMAFactory num_samples; one fixed draw with replacement before epochs",
        "source_train": len(train),
        "validation": len(val),
        "groups": {name: {"original": len(records), "factor": FACTORS[name],
                          "effective": len(records) * FACTORS[name]}
                   for name, records in groups.items()},
        "effective_train": sum(len(records) * FACTORS[name] for name, records in groups.items()),
    }
    write_json(output_dir / "dataset_info.json", entries)
    write_json(output_dir / "sampling_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/llamafactory/generated"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/llamafactory/generated/omni_v5_targeted"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = args.source_dir if args.source_dir.is_absolute() else root / args.source_dir
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    print(json.dumps(prepare(root, source, output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
