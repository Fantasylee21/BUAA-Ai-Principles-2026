#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_KEY="${1:-}"
GPU_ID="${2:-${GPU_ID:-0}}"
CONDA_ENV="${CONDA_ENV:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/sft}"
RUN_NAME="${RUN_NAME:-lora_sft_v1}"
RESUME="${RESUME:-auto}"
PREPARE_DATA="${PREPARE_DATA:-0}"
DRY_RUN="${DRY_RUN:-0}"
LOCK_FILE="${REPO_ROOT}/third_party/llamafactory.lock.yaml"
LLAMAFACTORY_DIR="${REPO_ROOT}/third_party/LlamaFactory"
DATASET_INFO="${REPO_ROOT}/data/llamafactory/dataset_info.json"

usage() {
    cat <<'EOF'
Usage: bash scripts/run_lora_sft.sh MODEL [GPU_ID]

MODEL:
  qwen25vl | qwen3vl | qwen25omni | all

Environment overrides:
  CONDA_ENV       Run with a named conda environment; otherwise use active Python.
  OUTPUT_ROOT     Root for adapters/checkpoints (default: output/sft).
  RUN_NAME        Output subdirectory name (default: lora_sft_v1).
  RESUME          auto, none, or an exact checkpoint path (default: auto).
  PREPARE_DATA    Set to 1 to rebuild generated SFT data before training.
  DRY_RUN         Set to 1 to validate and print the launch without training.
EOF
}

if [[ -z "$MODEL_KEY" ]]; then
    usage >&2
    exit 2
fi

case "$MODEL_KEY" in
    qwen25vl|qwen3vl|qwen25omni|all) ;;
    *)
        printf 'Unknown model key: %s\n' "$MODEL_KEY" >&2
        usage >&2
        exit 2
        ;;
esac

require_file() {
    if [[ ! -f "$1" ]]; then
        printf 'Required file not found: %s\n' "$1" >&2
        exit 1
    fi
}

require_directory() {
    if [[ ! -d "$1" ]]; then
        printf 'Required directory not found: %s\n' "$1" >&2
        exit 1
    fi
}

require_directory "$LLAMAFACTORY_DIR"
require_file "$LOCK_FILE"
require_file "$DATASET_INFO"

EXPECTED_COMMIT="$(sed -n 's/^  commit: //p' "$LOCK_FILE")"
ACTUAL_COMMIT="$(git -C "$LLAMAFACTORY_DIR" rev-parse HEAD)"
if [[ -z "$EXPECTED_COMMIT" || "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
    printf 'LLaMAFactory lock mismatch: expected %s, found %s\n' \
        "${EXPECTED_COMMIT:-missing}" "$ACTUAL_COMMIT" >&2
    exit 1
fi

if [[ -n "$CONDA_ENV" ]]; then
    command -v conda >/dev/null 2>&1 || {
        printf 'CONDA_ENV is set, but conda is unavailable.\n' >&2
        exit 1
    }
    PYTHON_RUNNER=(conda run --no-capture-output -n "$CONDA_ENV" python)
else
    command -v python >/dev/null 2>&1 || {
        printf 'python is unavailable. Activate a compatible environment first.\n' >&2
        exit 1
    }
    PYTHON_RUNNER=(python)
fi

if [[ "$PREPARE_DATA" == "1" ]]; then
    "${PYTHON_RUNNER[@]}" "${REPO_ROOT}/scripts/prepare_sft_data.py"
elif [[ "$PREPARE_DATA" != "0" ]]; then
    printf 'PREPARE_DATA must be 0 or 1.\n' >&2
    exit 2
fi

if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    printf 'DRY_RUN must be 0 or 1.\n' >&2
    exit 2
fi

require_file "${REPO_ROOT}/data/llamafactory/generated/affect_sft_vl_train.jsonl"
require_file "${REPO_ROOT}/data/llamafactory/generated/affect_sft_vl_val.jsonl"
require_file "${REPO_ROOT}/data/llamafactory/generated/affect_sft_omni_train.jsonl"
require_file "${REPO_ROOT}/data/llamafactory/generated/affect_sft_omni_val.jsonl"

latest_checkpoint() {
    local output_dir="$1"
    find "$output_dir" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' \
        -printf '%f\n' 2>/dev/null | sort -V | tail -n 1
}

run_one() {
    local key="$1"
    local config_path model_dir output_dir log_dir log_path checkpoint_name
    case "$key" in
        qwen25vl)
            config_path="${REPO_ROOT}/configs/sft/qwen25vl/lora_sft_v1.yaml"
            model_dir="${REPO_ROOT}/models/Qwen2.5-VL-3B-Instruct"
            ;;
        qwen3vl)
            config_path="${REPO_ROOT}/configs/sft/qwen3vl/lora_sft_v1.yaml"
            model_dir="${REPO_ROOT}/models/Qwen3-VL-4B-Instruct"
            ;;
        qwen25omni)
            config_path="${REPO_ROOT}/configs/sft/qwen25omni/lora_sft_v1.yaml"
            model_dir="${REPO_ROOT}/models/Qwen2.5-Omni-3B"
            ;;
    esac

    require_file "$config_path"
    require_directory "$model_dir"
    output_dir="${OUTPUT_ROOT}/${key}/${RUN_NAME}"
    log_dir="${REPO_ROOT}/runs/sft/${key}"
    mkdir -p "$output_dir" "$log_dir" "${REPO_ROOT}/tmp/hf_datasets"
    log_path="${log_dir}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

    TRAIN_OVERRIDES=("output_dir=${output_dir}")
    case "$RESUME" in
        auto)
            checkpoint_name="$(latest_checkpoint "$output_dir")"
            if [[ -n "$checkpoint_name" ]]; then
                TRAIN_OVERRIDES+=("resume_from_checkpoint=${output_dir}/${checkpoint_name}")
            fi
            ;;
        none) ;;
        *)
            require_directory "$RESUME"
            TRAIN_OVERRIDES+=("resume_from_checkpoint=${RESUME}")
            ;;
    esac

    printf '\n[%s] Starting %s LoRA SFT on physical GPU %s\n' \
        "$(date --iso-8601=seconds)" "$key" "$GPU_ID"
    printf 'Config: %s\nOutput: %s\nLog: %s\n' \
        "$config_path" "$output_dir" "$log_path"

    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'Dry run command: CUDA_VISIBLE_DEVICES=%q python -m llamafactory.cli train %q' \
            "$GPU_ID" "$config_path"
        printf ' %q' "${TRAIN_OVERRIDES[@]}"
        printf '\n'
        return
    fi

    (
        cd "$REPO_ROOT"
        CUDA_VISIBLE_DEVICES="$GPU_ID" \
        FORCE_TORCHRUN=1 \
        NPROC_PER_NODE=1 \
        PYTHONPATH="${LLAMAFACTORY_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
        HF_DATASETS_CACHE="${REPO_ROOT}/tmp/hf_datasets" \
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
        TOKENIZERS_PARALLELISM=false \
        "${PYTHON_RUNNER[@]}" -m llamafactory.cli train \
            "$config_path" "${TRAIN_OVERRIDES[@]}" 2>&1 | tee "$log_path"
    )
}

if [[ "$MODEL_KEY" == "all" ]]; then
    for key in qwen25vl qwen3vl qwen25omni; do
        run_one "$key"
    done
else
    run_one "$MODEL_KEY"
fi
