#!/usr/bin/env python3
"""Split existing SFT training data for native LLaMAFactory oversampling."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

from prepare_sft_data import read_jsonl, validate_sft_records, write_json, write_jsonl


def partition_training(records):
    """Keep full records, including the selected question and media references."""
    aligned, conflict = [], []
    seen = set()
    for record in records:
        if record.get("split") != "train":
            raise ValueError("Only training records may be resampled.")
        sample_id = record["sample_id"]
        if sample_id in seen:
            raise ValueError(f"Expected single-question data; duplicate ID: {sample_id}")
        seen.add(sample_id)
        answer = json.loads(record["messages"][-1]["content"])
        uncertain = answer.get("is_uncertain")
        if uncertain not in ("yes", "no"):
            raise ValueError(f"Invalid is_uncertain for {sample_id}: {uncertain}")
        (conflict if uncertain == "yes" else aligned).append(record)
    if not aligned or not conflict:
        raise ValueError("Both Aligned and Conflict training records are required.")
    return aligned, conflict


def prepare(root, source_dir, output_dir, factor):
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError("Conflict factor must be a positive integer.")
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if source_dir == output_dir:
        raise ValueError("Use a separate output directory for resampling metadata.")
    original = json.loads((root / "data/llamafactory/dataset_info.json").read_text())
    entries, summary = {}, {}
    for family in ("vl", "omni"):
        train = read_jsonl(source_dir / f"affect_sft_{family}_train.jsonl")
        val_path = source_dir / f"affect_sft_{family}_val.jsonl"
        val = read_jsonl(val_path)
        validate_sft_records(train, family, root, "full")
        validate_sft_records(val, family, root, "emotion_only")
        if {r["split_group_id"] for r in train} & {r["split_group_id"] for r in val}:
            raise ValueError(f"Train/validation group overlap: {family}")
        aligned, conflict = partition_training(train)
        for kind, records in (("aligned", aligned), ("conflict", conflict)):
            name = f"affect_sft_{family}_train_{kind}"
            entries[name] = deepcopy(original[f"affect_sft_{family}_train"])
            entries[name]["file_name"] = f"{family}_{kind}.jsonl"
            # LLaMAFactory includes every original row, then draws extra rows
            # with replacement. No custom Trainer/sampler is needed.
            if kind == "conflict":
                entries[name]["num_samples"] = len(records) * factor
            write_jsonl(output_dir / entries[name]["file_name"], records)
        val_name = f"affect_sft_{family}_val"
        entries[val_name] = deepcopy(original[val_name])
        entries[val_name]["file_name"] = os.path.relpath(val_path, output_dir)
        summary[family] = {
            "original_train": len(train), "aligned": len(aligned),
            "original_conflict": len(conflict), "sampled_conflict": len(conflict) * factor,
            "sampled_train": len(aligned) + len(conflict) * factor,
            "validation": len(val),
        }
    write_json(output_dir / "dataset_info.json", entries)
    write_json(output_dir / "resampling_summary.json", {
        "conflict_factor": factor, "sampler": "LLaMAFactory num_samples",
        "datasets": summary,
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/llamafactory/generated"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/llamafactory/generated/resampled_2x"))
    parser.add_argument("--conflict-factor", type=int, default=2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    summary = prepare(root, root / args.source_dir, root / args.output_dir, args.conflict_factor)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
