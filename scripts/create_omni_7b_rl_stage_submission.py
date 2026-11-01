#!/usr/bin/env python3
"""Reproduce the RL-stage submission from saved SFT model predictions."""
import argparse
import csv
import json
from pathlib import Path


def load_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=Path("submit.csv"))
    p.add_argument("--omni-v2", type=Path, default=Path("output/sft/evaluation/qwen25omni_qlora_sft_v2_64tok_test_predictions.csv"))
    p.add_argument("--qwen3vl", type=Path, default=Path("output/sft/evaluation/qwen3vl_qlora_sft_v1_test_predictions.csv"))
    p.add_argument("--test", type=Path, default=Path("Dataset/test_release.jsonl"))
    p.add_argument("--output", type=Path, default=Path("submit_source_emotion_vt.csv"))
    args = p.parse_args()
    fields = ("sample_id", "pred_emotion", "is_uncertain", "clarifying_question")
    base = load_csv(args.base)
    preds = [load_csv(args.omni_v2), load_csv(args.qwen3vl)]
    ids = [json.loads(line)["sample_id"] for line in args.test.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != 734 or len(set(ids)) != 734 or [r["sample_id"] for r in base] != ids:
        raise ValueError("Base predictions do not cover the official test IDs in order")
    indices = []
    for rows in preds:
        lookup = {r["sample_id"]: r for r in rows}
        if len(rows) != 734 or len(lookup) != 734 or set(lookup) != set(ids):
            raise ValueError("Model predictions have incomplete or duplicate IDs")
        indices.append(lookup)
    counts = {"conflict": 0, "aligned": 0, "emotion": 0}
    for row in base:
        sid = row["sample_id"]
        if sid.startswith(("gen:accept_a_svt:", "gen:accept_a_va:")) and row["is_uncertain"] == "no":
            row["is_uncertain"] = "yes"
            row["clarifying_question"] = "你此刻的感受更偏向哪一种？"
            counts["conflict"] += 1
        elif sid.startswith("gen:accept_c_svt:") and row["is_uncertain"] == "yes":
            row["is_uncertain"] = "no"
            row["clarifying_question"] = "none"
            counts["aligned"] += 1
        a, b = (item[sid]["pred_emotion"] for item in indices)
        if sid.startswith("ch_sims_v2:VT:") and a == b and a != row["pred_emotion"]:
            row["pred_emotion"] = a
            counts["emotion"] += 1
        if (row["pred_emotion"] not in {"positive", "negative", "neutral"}
                or row["is_uncertain"] not in {"yes", "no"}
                or not row["clarifying_question"].strip()
                or (row["is_uncertain"] == "no" and row["clarifying_question"] != "none")
                or (row["is_uncertain"] == "yes" and row["clarifying_question"].strip().lower() == "none")):
            raise ValueError(f"Invalid prediction for {sid}")
    if args.output.resolve() in {args.base.resolve(), args.omni_v2.resolve(), args.qwen3vl.resolve()}:
        raise ValueError("Output must differ from all input files")
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in base)
    print(json.dumps({"output": str(args.output), "samples": len(base), "changes": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
