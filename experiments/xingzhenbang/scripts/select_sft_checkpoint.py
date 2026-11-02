#!/usr/bin/env python3
"""Evaluate epoch adapters and select by the official validation Score."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from sft_config import load_config_metadata


def select_checkpoint(config_path, output_dir=None):
    root = Path(__file__).resolve().parents[1]
    _, meta = load_config_metadata(Path(config_path).resolve(), root)
    output = Path(output_dir or root / "output/sft/evaluation")
    output.mkdir(parents=True, exist_ok=True)
    adapters = sorted(Path(meta["output_dir"]).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    adapters = [p for p in adapters if (p / "adapter_model.safetensors").is_file()]
    if not adapters:
        raise RuntimeError("No epoch adapter checkpoints found.")
    candidates = []
    for adapter in adapters:
        name = f"{meta['model_key']}_{meta['run_name']}_{adapter.name}_val"
        subprocess.run([sys.executable, str(root / "scripts/evaluate_sft.py"),
                        "--config", str(config_path), "--adapter-path", str(adapter),
                        "--output-dir", str(output), "--run-name", name], cwd=root, check=True)
        result = json.loads((output / f"{name}_metrics.json").read_text())
        if result["samples"] != 666 or result["inference_errors"]:
            raise RuntimeError(f"Incomplete validation for {adapter}: {result['samples']} samples, "
                               f"{result['inference_errors']} inference errors")
        candidates.append({"checkpoint": str(adapter), "metrics_file": str(output / f"{name}_metrics.json"),
                           **{k: result[k] for k in ("Macro-F1", "Conflict_accuracy", "Aligned_calibration", "Score")}})
    best = max(candidates, key=lambda item: item["Score"])
    selection = {**best, "selected_by": "official_validation_Score", "samples": 666, "candidates": candidates}
    destination = Path(meta["output_dir"]) / "best_checkpoint.json"
    destination.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)
    return selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    select_checkpoint(args.config, args.output_dir)
