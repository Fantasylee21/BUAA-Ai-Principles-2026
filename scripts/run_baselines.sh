#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:-${GPU_ID:-2}}"
CONDA_ENV="${CONDA_ENV:-}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/Dataset}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/baseline}"

require_directory() {
    if [[ ! -d "$1" ]]; then
        printf 'Required directory not found: %s\n' "$1" >&2
        exit 1
    fi
}

command -v ffmpeg >/dev/null 2>&1 || {
    printf 'ffmpeg is required by the Qwen2.5-Omni baseline.\n' >&2
    exit 1
}

if [[ -n "$CONDA_ENV" ]]; then
    command -v conda >/dev/null 2>&1 || {
        printf 'CONDA_ENV is set, but conda is not available in PATH.\n' >&2
        exit 1
    }
    PYTHON_RUNNER=(conda run -n "$CONDA_ENV" python)
else
    command -v python >/dev/null 2>&1 || {
        printf 'python is not available in PATH. Activate a compatible environment first.\n' >&2
        exit 1
    }
    PYTHON_RUNNER=(python)
fi

require_directory "$DATA_ROOT"
require_directory "${MODEL_ROOT}/Qwen2.5-VL-3B-Instruct"
require_directory "${MODEL_ROOT}/Qwen3-VL-4B-Instruct"
require_directory "${MODEL_ROOT}/Qwen2.5-Omni-3B"
mkdir -p "$OUTPUT_DIR"

run_baseline() {
    local script_name="$1"
    local model_path="$2"
    local run_name="$3"

    printf '\n[%s] Starting %s on physical GPU %s\n' \
        "$(date --iso-8601=seconds)" "$run_name" "$GPU_ID"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "${PYTHON_RUNNER[@]}" \
        "${REPO_ROOT}/${script_name}" \
        --data-root "$DATA_ROOT" \
        --model-path "$model_path" \
        --output-dir "$OUTPUT_DIR" \
        --split val \
        --run-name "$run_name" \
        --overwrite
}

run_baseline \
    qwen25vl_baseline.py \
    "${MODEL_ROOT}/Qwen2.5-VL-3B-Instruct" \
    qwen25vl_baseline_val

run_baseline \
    qwen3vl_baseline.py \
    "${MODEL_ROOT}/Qwen3-VL-4B-Instruct" \
    qwen3vl_baseline_val

run_baseline \
    qwen25omni_baseline.py \
    "${MODEL_ROOT}/Qwen2.5-Omni-3B" \
    qwen25omni_baseline_val

printf '\nAll baseline runs completed. Results: %s\n' "$OUTPUT_DIR"
