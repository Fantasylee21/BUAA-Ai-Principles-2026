#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_METHOD="${1:-}"
CONFIG_INPUT="${2:-}"
GPU_ID="${3:-${GPU_ID:-0}}"
CONDA_ENV="${CONDA_ENV:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
RESUME="${RESUME:-auto}"
PREPARE_DATA="${PREPARE_DATA:-0}"
DRY_RUN="${DRY_RUN:-0}"
LLAMAFACTORY_DIR="${REPO_ROOT}/third_party/LlamaFactory"
DATASET_INFO="${REPO_ROOT}/data/llamafactory/dataset_info.json"

usage() {
    cat <<'EOF'
Usage: bash scripts/run_sft_training.sh METHOD CONFIG_YAML [GPU_ID]

METHOD:
  lora | qlora

Environment overrides:
  CONDA_ENV       Run with a named conda environment; otherwise use active Python.
  OUTPUT_DIR      Override output_dir from the selected YAML.
  RESUME          auto, none, or an exact checkpoint path (default: auto).
  PREPARE_DATA    Set to 1 to rebuild generated SFT data before training.
  DRY_RUN         Set to 1 to validate and print the launch without training.
EOF
}

if [[ "$EXPECTED_METHOD" != "lora" && "$EXPECTED_METHOD" != "qlora" ]]; then
    usage >&2
    exit 2
fi
if [[ -z "$CONFIG_INPUT" ]]; then
    usage >&2
    exit 2
fi

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

resolve_input_path() {
    local value="$1"
    if [[ "$value" = /* ]]; then
        printf '%s\n' "$value"
    elif [[ -f "$value" ]]; then
        realpath "$value"
    else
        printf '%s/%s\n' "$REPO_ROOT" "$value"
    fi
}

CONFIG_PATH="$(resolve_input_path "$CONFIG_INPUT")"
require_file "$CONFIG_PATH"
require_directory "$LLAMAFACTORY_DIR"
require_file "$DATASET_INFO"

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

mapfile -t CONFIG_META < <(
    "${PYTHON_RUNNER[@]}" "${REPO_ROOT}/scripts/sft_config.py" \
        --config "$CONFIG_PATH" \
        --repo-root "$REPO_ROOT" \
        --expected-method "$EXPECTED_METHOD" \
        --format lines
)
if [[ "${#CONFIG_META[@]}" -ne 5 ]]; then
    printf 'Failed to read configuration metadata: %s\n' "$CONFIG_PATH" >&2
    exit 1
fi
MODEL_KEY="${CONFIG_META[0]}"
MODEL_DIR="${CONFIG_META[1]}"
CONFIG_OUTPUT_DIR="${CONFIG_META[2]}"
RUN_NAME="${CONFIG_META[3]}"

require_directory "$MODEL_DIR"

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

if [[ "$MODEL_KEY" == "qwen25omni" ]]; then
    require_file "${REPO_ROOT}/data/llamafactory/generated/affect_sft_omni_train.jsonl"
    require_file "${REPO_ROOT}/data/llamafactory/generated/affect_sft_omni_val.jsonl"
else
    require_file "${REPO_ROOT}/data/llamafactory/generated/affect_sft_vl_train.jsonl"
    require_file "${REPO_ROOT}/data/llamafactory/generated/affect_sft_vl_val.jsonl"
fi

if [[ -n "$OUTPUT_DIR" ]]; then
    if [[ "$OUTPUT_DIR" != /* ]]; then
        OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
    fi
else
    OUTPUT_DIR="$CONFIG_OUTPUT_DIR"
fi
RUN_NAME="$(basename "$OUTPUT_DIR")"
LOG_DIR="${REPO_ROOT}/runs/sft/${MODEL_KEY}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "${REPO_ROOT}/tmp/hf_datasets"
if [[ "$RUN_NAME" == "qlora_sft_v3" && "$DRY_RUN" == "0" ]]; then
    # A model may be launched on another GPU while the original queue is active.
    # Wait for that training, then let the queue proceed directly to evaluation.
    exec 9>"${OUTPUT_DIR}/training.lock"
    flock 9
    if [[ -f "${OUTPUT_DIR}/train_results.json" ]]; then
        printf 'Training already completed: %s\n' "$OUTPUT_DIR"
        exit 0
    fi
fi
LOG_PATH="${LOG_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

if [[ "$EXPECTED_METHOD" == "qlora" && "$DRY_RUN" == "0" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_ID" "${PYTHON_RUNNER[@]}" -c \
        'import bitsandbytes, torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("bitsandbytes:", bitsandbytes.__version__)'
fi

latest_checkpoint() {
    find "$1" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' \
        -printf '%f\n' 2>/dev/null | sort -V | tail -n 1
}

TRAIN_OVERRIDES=("output_dir=${OUTPUT_DIR}")
case "$RESUME" in
    auto)
        CHECKPOINT_NAME="$(latest_checkpoint "$OUTPUT_DIR")"
        if [[ -n "$CHECKPOINT_NAME" ]]; then
            TRAIN_OVERRIDES+=("resume_from_checkpoint=${OUTPUT_DIR}/${CHECKPOINT_NAME}")
        fi
        ;;
    none) ;;
    *)
        require_directory "$RESUME"
        TRAIN_OVERRIDES+=("resume_from_checkpoint=${RESUME}")
        ;;
esac

printf '\n[%s] Starting %s %s SFT on physical GPU %s\n' \
    "$(date --iso-8601=seconds)" "$MODEL_KEY" "$EXPECTED_METHOD" "$GPU_ID"
printf 'Config: %s\nOutput: %s\nLog: %s\n' \
    "$CONFIG_PATH" "$OUTPUT_DIR" "$LOG_PATH"

if [[ "$DRY_RUN" == "1" ]]; then
    printf 'Dry run command: CUDA_VISIBLE_DEVICES=%q python scripts/train_sft.py %q' \
        "$GPU_ID" "$CONFIG_PATH"
    printf ' %q' "${TRAIN_OVERRIDES[@]}"
    printf '\n'
    exit 0
fi

(
    cd "$REPO_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    FORCE_TORCHRUN=0 \
    NPROC_PER_NODE=1 \
    PYTHONPATH="${LLAMAFACTORY_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    HF_DATASETS_CACHE="${REPO_ROOT}/tmp/hf_datasets/${MODEL_KEY}" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    "${PYTHON_RUNNER[@]}" "${REPO_ROOT}/scripts/train_sft.py" \
        "$CONFIG_PATH" "${TRAIN_OVERRIDES[@]}" 2>&1 | tee "$LOG_PATH"
)
