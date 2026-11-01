#!/usr/bin/env python3
"""Validate complete SFT test predictions and write a submission CSV."""

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile


COLUMNS = ["sample_id", "pred_emotion", "is_uncertain", "clarifying_question"]


def validate_and_copy(test_path, prediction_path, metrics_path, output_path):
    test_ids = [json.loads(line)["sample_id"] for line in test_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    if len(test_ids) != 734 or len(set(test_ids)) != 734:
        raise ValueError("Expected 734 unique released test IDs")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("split") != "test" or metrics.get("samples") != 734 or metrics.get("inference_errors") != 0:
        raise ValueError("Test inference is incomplete or contains errors")
    if metrics.get("parse_status_counts", {}).get("json") != 734:
        raise ValueError("Not every test response parsed as JSON")
    if Path(metrics.get("files", {}).get("predictions", "")).resolve() != prediction_path.resolve():
        raise ValueError("Metrics and prediction CSV belong to different runs")
    with prediction_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"Unexpected prediction columns: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != 734 or [row["sample_id"] for row in rows] != test_ids:
        raise ValueError("Prediction IDs do not match released test order")
    for index, row in enumerate(rows, 1):
        emotion, uncertain = row["pred_emotion"], row["is_uncertain"]
        question = row["clarifying_question"].strip()
        if emotion not in {"positive", "negative", "neutral"} or uncertain not in {"yes", "no"}:
            raise ValueError(f"Invalid emotion or uncertainty at row {index}")
        if (uncertain == "yes" and question.lower() in {"", "none"}) or (
            uncertain == "no" and question != "none"
        ):
            raise ValueError(f"Invalid clarifying question at row {index}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, prefix=output_path.name + ".",
                                     suffix=".tmp", delete=False) as handle:
        temp = Path(handle.name)
        try:
            handle.write(prediction_path.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temp, output_path)
        finally:
            temp.unlink(missing_ok=True)
    print(f"Wrote {output_path.resolve()} ({len(rows)} predictions)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate_and_copy(args.test, args.predictions, args.metrics, args.output)


if __name__ == "__main__":
    main()
