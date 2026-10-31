#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_METHOD="${1:-}"
CONFIG_INPUT="${2:-}"
GPU_ID="${3:-${GPU_ID:-0}}"
CONDA_ENV="${CONDA_ENV:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/sft/evaluation}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
RUN_NAME="${RUN_NAME:-}"
LIMIT="${LIMIT:-}"
OVERWRITE="${OVERWRITE:-0}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-auto}"
SPLIT="${SPLIT:-val}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
    cat <<'EOF'
Usage: bash scripts/run_sft_inference.sh METHOD CONFIG_YAML [GPU_ID]

METHOD:
  lora | qlora

Environment overrides:
  CONDA_ENV       Run with a named conda environment; otherwise use active Python.
  OUTPUT_DIR      Evaluation output directory (default: output/sft/evaluation).
  ADAPTER_PATH    Override the adapter path from the selected YAML.
  RUN_NAME        Override the YAML-derived result prefix.
  LIMIT           Evaluate only the first N records.
  OVERWRITE       Set to 1 to replace an existing run.
  LOAD_IN_4BIT    auto, 0, or 1 (default: auto; follows the selected YAML).
  SPLIT            val or test (default: val).
  DRY_RUN         Set to 1 to validate and print the command without inference.
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

if [[ "$CONFIG_INPUT" = /* ]]; then
    CONFIG_PATH="$CONFIG_INPUT"
elif [[ -f "$CONFIG_INPUT" ]]; then
    CONFIG_PATH="$(realpath "$CONFIG_INPUT")"
else
    CONFIG_PATH="${REPO_ROOT}/${CONFIG_INPUT}"
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
    printf 'Configuration file not found: %s\n' "$CONFIG_PATH" >&2
    exit 1
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
CONFIG_RUN_NAME="${CONFIG_META[3]}"

case "$OVERWRITE" in 0|1) ;; *) printf 'OVERWRITE must be 0 or 1.\n' >&2; exit 2 ;; esac
case "$DRY_RUN" in 0|1) ;; *) printf 'DRY_RUN must be 0 or 1.\n' >&2; exit 2 ;; esac
case "$LOAD_IN_4BIT" in auto|0|1) ;;
    *) printf 'LOAD_IN_4BIT must be auto, 0, or 1.\n' >&2; exit 2 ;;
esac
case "$SPLIT" in val|test) ;;
    *) printf 'SPLIT must be val or test.\n' >&2; exit 2 ;;
esac

ARGS=(
    "${REPO_ROOT}/scripts/evaluate_sft.py"
    --config "$CONFIG_PATH"
    --expected-method "$EXPECTED_METHOD"
    --output-dir "$OUTPUT_DIR"
    --split "$SPLIT"
)
if [[ -n "$RUN_NAME" ]]; then
    ARGS+=(--run-name "$RUN_NAME")
fi
if [[ -n "$ADAPTER_PATH" ]]; then
    if [[ "$ADAPTER_PATH" != /* ]]; then
        ADAPTER_PATH="${REPO_ROOT}/${ADAPTER_PATH}"
    fi
    ARGS+=(--adapter-path "$ADAPTER_PATH")
fi
if [[ -n "$LIMIT" ]]; then
    ARGS+=(--limit "$LIMIT")
fi
if [[ "$OVERWRITE" == "1" ]]; then
    ARGS+=(--overwrite)
fi
if [[ "$LOAD_IN_4BIT" == "0" ]]; then
    ARGS+=(--no-load-in-4bit)
elif [[ "$LOAD_IN_4BIT" == "1" ]]; then
    ARGS+=(--load-in-4bit)
fi

LOG_DIR="${REPO_ROOT}/runs/sft/evaluation"
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
fi
LOG_PATH="${LOG_DIR}/${MODEL_KEY}_${CONFIG_RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

printf '\n[%s] Evaluating %s %s adapter on physical GPU %s\n' \
    "$(date --iso-8601=seconds)" "$MODEL_KEY" "$EXPECTED_METHOD" "$GPU_ID"
printf 'Config: %s\nOutput: %s\nLog: %s\n' \
    "$CONFIG_PATH" "$OUTPUT_DIR" "$LOG_PATH"

if [[ "$DRY_RUN" == "1" ]]; then
    printf 'Dry run command: CUDA_VISIBLE_DEVICES=%q' "$GPU_ID"
    printf ' %q' "${PYTHON_RUNNER[@]}" "${ARGS[@]}"
    printf '\n'
    exit 0
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
TOKENIZERS_PARALLELISM=false \
    "${PYTHON_RUNNER[@]}" "${ARGS[@]}" 2>&1 | tee "$LOG_PATH"
