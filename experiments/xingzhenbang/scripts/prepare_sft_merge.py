#!/usr/bin/env python3
"""Create a minimal LLaMAFactory config for merging one SFT adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from sft_config import load_config_metadata, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--export-device", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--export-size", type=int, default=5)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    training_config_path = args.training_config.expanduser().resolve()
    adapter_path = resolve_path(args.adapter_path, repo_root).resolve()
    export_dir = resolve_path(args.export_dir, repo_root).resolve()
    output_config = args.output_config.expanduser().resolve()

    training_config, metadata = load_config_metadata(
        training_config_path, repo_root
    )
    adapter_config_path = adapter_path / "adapter_config.json"
    adapter_weights = adapter_path / "adapter_model.safetensors"
    legacy_adapter_weights = adapter_path / "adapter_model.bin"
    if not adapter_config_path.is_file():
        raise FileNotFoundError(adapter_config_path)
    if not adapter_weights.is_file() and not legacy_adapter_weights.is_file():
        raise FileNotFoundError(
            f"No adapter_model.safetensors or adapter_model.bin in {adapter_path}"
        )
    if export_dir == adapter_path:
        raise ValueError("Export directory must be a child of the adapter directory")
    if not is_relative_to(export_dir, adapter_path):
        raise ValueError(
            f"Export directory must be inside the adapter directory: {adapter_path}"
        )
    if export_dir.exists() and any(export_dir.iterdir()):
        raise ValueError(f"Export directory is not empty: {export_dir}")
    if args.export_size <= 0:
        raise ValueError("--export-size must be positive")

    adapter_config = load_json(adapter_config_path)
    if str(adapter_config.get("peft_type", "")).upper() != "LORA":
        raise ValueError(f"Adapter is not LoRA PEFT: {adapter_config_path}")
    adapter_base = adapter_config.get("base_model_name_or_path")
    if adapter_base:
        resolved_adapter_base = resolve_path(str(adapter_base), repo_root).resolve()
        expected_base = Path(metadata["model_path"]).resolve()
        if resolved_adapter_base != expected_base:
            raise ValueError(
                "Adapter base model does not match the training YAML: "
                f"{resolved_adapter_base} != {expected_base}"
            )

    template = training_config.get("template")
    if not template:
        raise ValueError("Training YAML is missing template")
    if training_config.get("bf16"):
        infer_dtype = "bfloat16"
    elif training_config.get("fp16"):
        infer_dtype = "float16"
    else:
        infer_dtype = "auto"

    merge_config = {
        "model_name_or_path": metadata["model_path"],
        "adapter_name_or_path": str(adapter_path),
        "template": template,
        "trust_remote_code": bool(training_config.get("trust_remote_code", True)),
        "finetuning_type": "lora",
        "infer_dtype": infer_dtype,
        "export_dir": str(export_dir),
        "export_size": args.export_size,
        "export_device": args.export_device,
        "export_legacy_format": False,
    }
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(
        yaml.safe_dump(merge_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "model_key": metadata["model_key"],
                "method": metadata["method"],
                "model_path": metadata["model_path"],
                "adapter_path": str(adapter_path),
                "export_dir": str(export_dir),
                "merge_config": str(output_config),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
