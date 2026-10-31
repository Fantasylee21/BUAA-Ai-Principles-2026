#!/usr/bin/env python3
"""Validate original train labels and write unique RL records (no duplication).

Sampling is performed by the resumable step-indexed sampler at training time.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "rl"

from .experiment import ROOT, check_disjoint, fingerprint, load_records, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "Dataset/train.jsonl")
    parser.add_argument("--val-file", type=Path, default=ROOT / "Dataset/val.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/rl-prepared/train.jsonl")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Prepared output exists; use a new output path")
    records, duplicates = load_records(args.input)
    validation, repeated_val = load_records(args.val_file)
    if repeated_val or len(validation) != 666:
        raise ValueError("Expected 666 unique official validation records")
    check_disjoint(records, validation)
    if set(r["sample_type"] for r in records) != {"Conflict", "Aligned"}:
        raise ValueError("Training requires both sample types")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    report = {"unique_samples": len(records), "ignored_exact_duplicates": duplicates,
              "types": dict(Counter(r["sample_type"] for r in records)),
              "emotions": dict(Counter(r["gt_label"] for r in records)), "data_hash": fingerprint(records)}
    write_json(args.output.with_suffix(".summary.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
