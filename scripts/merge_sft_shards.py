#!/usr/bin/env python3
"""Merge independent evaluate_sft.py shards in official input order."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwen25vl_baseline import grouped_metrics, json_dump, load_jsonl, metric_block  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-name", required=True)
    p.add_argument("--split", choices=("val", "test"), required=True)
    p.add_argument("--num-shards", type=int, default=2)
    p.add_argument("--data-root", type=Path, default=ROOT / "Dataset")
    p.add_argument("--output-dir", type=Path, default=ROOT / "output/sft/evaluation")
    a = p.parse_args()
    if a.num_shards < 2:
        p.error("--num-shards must be at least 2")

    source = a.data_root / ("val.jsonl" if a.split == "val" else "test_release.jsonl")
    expected = load_jsonl(source)
    expected_ids = [str(row["sample_id"]) for row in expected]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError(f"Duplicate source sample IDs in {source}")

    records = {}
    configs = []
    metrics = []
    for index in range(a.num_shards):
        prefix = a.output_dir / f"{a.base_name}_shard{index}"
        config = json.loads(prefix.with_name(prefix.name + "_run_config.json").read_text())
        metric = json.loads(prefix.with_name(prefix.name + "_metrics.json").read_text())
        details = load_jsonl(prefix.with_name(prefix.name + "_details.jsonl"))
        if (config.get("num_shards") != a.num_shards
                or config.get("shard_index") != index
                or config.get("split") != a.split
                or Path(config.get("input_path", "")).resolve() != source.resolve()):
            raise ValueError(f"Shard configuration mismatch: {prefix}")
        expected_shard = expected_ids[index::a.num_shards]
        by_id = {str(row["sample_id"]): row for row in details}
        if len(by_id) != len(details) or set(by_id) != set(expected_shard):
            raise ValueError(f"Shard coverage mismatch: {prefix}")
        if metric.get("samples") != len(expected_shard):
            raise ValueError(f"Shard metrics incomplete: {prefix}")
        if any(row.get("run_id") != f"{a.base_name}_shard{index}" for row in details):
            raise ValueError(f"Shard run ID mismatch: {prefix}")
        records.update(by_id)
        configs.append(config)
        metrics.append(metric)

    for key in ("adapter_path", "model_path", "quantization", "input_profile",
                "training_config", "load_in_4bit", "attn_implementation"):
        if len({json.dumps(c.get(key), sort_keys=True) for c in configs}) != 1:
            raise ValueError(f"Shards disagree on {key}")
    if set(records) != set(expected_ids):
        raise ValueError("Merged prediction coverage mismatch")

    ordered = [{**records[sid], "run_id": a.base_name} for sid in expected_ids]
    prefix = a.output_dir / a.base_name
    details_path = prefix.with_name(prefix.name + "_details.jsonl")
    preds_path = prefix.with_name(prefix.name + "_predictions.csv")
    metrics_path = prefix.with_name(prefix.name + "_metrics.json")
    config_path = prefix.with_name(prefix.name + "_run_config.json")
    with details_path.open("w", encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with preds_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "pred_emotion", "is_uncertain", "clarifying_question"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in ordered)

    config = {**configs[0], "run_id": a.base_name, "num_shards": a.num_shards,
              "shard_index": None, "shard_run_names": [f"{a.base_name}_shard{i}" for i in range(a.num_shards)],
              "rows": len(ordered)}
    json_dump(config_path, config)
    result = {
        "run_name": a.base_name, "run_id": a.base_name, "split": a.split,
        "model": metrics[0]["model"], "adapter": metrics[0]["adapter"],
        "quantization": metrics[0]["quantization"], "samples": len(ordered),
        "elapsed_seconds_this_invocation": max(m["elapsed_seconds_this_invocation"] for m in metrics),
        "inference_errors": sum(bool(row.get("inference_error")) for row in ordered),
        "parse_status_counts": dict(Counter(row["parse_status"] for row in ordered)),
        "max_peak_cuda_allocated_bytes": max(row["peak_cuda_allocated_bytes"] for row in ordered),
        "max_peak_cuda_reserved_bytes": max(row["peak_cuda_reserved_bytes"] for row in ordered),
        "protocol_counts": dict(Counter(row["protocol"] for row in ordered)),
        "files": {"predictions": str(preds_path), "details": str(details_path),
                  "run_config": str(config_path)},
    }
    if a.split == "val":
        overall = metric_block(ordered)
        result.update({
            "Macro-F1": overall["macro_f1_three_classes"],
            "Conflict_accuracy": overall["conflict_identification_accuracy"],
            "Aligned_calibration": overall["aligned_calibration"],
            "Score": overall["competition_score"], "overall": overall,
            "by_protocol": grouped_metrics(ordered, "protocol"),
            "by_source_dataset": grouped_metrics(ordered, "source_dataset"),
            "by_use_in_main": grouped_metrics(ordered, "use_in_main"),
        })
    json_dump(metrics_path, result)
    print(json.dumps({"event": "merged", "run_name": a.base_name,
                      "samples": len(ordered), "metrics": str(metrics_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
