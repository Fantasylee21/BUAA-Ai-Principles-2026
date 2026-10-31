#!/usr/bin/env python3
"""Inspect and validate a project LLaMAFactory SFT YAML configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


MODEL_KEYS = {
    "qwen2.5-vl": "qwen25vl",
    "qwen3-vl": "qwen3vl",
    "qwen2.5-omni": "qwen25omni",
}


def resolve_path(path: str | Path, repo_root: Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (repo_root / value).resolve()


def infer_model_key(model_name_or_path: str) -> str:
    normalized = model_name_or_path.lower().replace("_", "-")
    for marker, key in MODEL_KEYS.items():
        if marker in normalized:
            return key
    raise ValueError(
        "Unsupported model_name_or_path. Expected Qwen2.5-VL, Qwen3-VL, "
        f"or Qwen2.5-Omni, got: {model_name_or_path}"
    )


def method_from_config(config: dict[str, Any]) -> str:
    quantization_bit = config.get("quantization_bit")
    if quantization_bit in (None, 0, "0", "none", "None"):
        return "lora"
    if int(quantization_bit) == 4:
        return "qlora"
    raise ValueError(f"Unsupported quantization_bit: {quantization_bit}")


def load_config_metadata(
    config_path: Path,
    repo_root: Path,
    expected_method: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"YAML root must be a mapping: {config_path}")

    required = ("model_name_or_path", "output_dir", "stage", "finetuning_type")
    missing = [name for name in required if not config.get(name)]
    if missing:
        raise ValueError(f"Missing required YAML fields: {', '.join(missing)}")
    if config["stage"] != "sft":
        raise ValueError(f"Expected stage: sft, got: {config['stage']}")
    if config["finetuning_type"] != "lora":
        raise ValueError(
            f"Expected finetuning_type: lora, got: {config['finetuning_type']}"
        )

    method = method_from_config(config)
    if expected_method is not None and method != expected_method:
        raise ValueError(
            f"Configuration is {method}, but the selected launcher expects "
            f"{expected_method}: {config_path}"
        )

    model_name_or_path = str(config["model_name_or_path"])
    output_dir = resolve_path(str(config["output_dir"]), repo_root)
    metadata = {
        "config_path": str(config_path.resolve()),
        "model_key": infer_model_key(model_name_or_path),
        "model_path": str(resolve_path(model_name_or_path, repo_root)),
        "output_dir": str(output_dir),
        "run_name": output_dir.name,
        "method": method,
        "quantization_bit": config.get("quantization_bit"),
    }
    return config, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-method", choices=("lora", "qlora"))
    parser.add_argument("--format", choices=("json", "lines"), default="json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    try:
        _, metadata = load_config_metadata(
            config_path, repo_root, expected_method=args.expected_method
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise SystemExit(f"Invalid SFT configuration: {exc}") from exc
    if args.format == "json":
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return
    for field in ("model_key", "model_path", "output_dir", "run_name", "method"):
        print(metadata[field])


if __name__ == "__main__":
    main()
