#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_INPUT="${1:-}"
ADAPTER_INPUT="${2:-}"
GPU_ID="${3:-${GPU_ID:-0}}"
CONDA_ENV="${CONDA_ENV:-}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"
EXPORT_SIZE="${EXPORT_SIZE:-5}"
MERGED_SUBDIR="${MERGED_SUBDIR:-merged_model}"
DRY_RUN="${DRY_RUN:-0}"
LOCK_FILE="${REPO_ROOT}/third_party/llamafactory.lock.yaml"
LLAMAFACTORY_DIR="${REPO_ROOT}/third_party/LlamaFactory"

usage() {
    cat <<'EOF'
Usage: bash scripts/merge_sft_adapter.sh CONFIG_YAML [ADAPTER_PATH] [GPU_ID]

ADAPTER_PATH defaults to output_dir from CONFIG_YAML. It may point to the final
adapter directory or a checkpoint-* directory. The merged full model is saved
under ADAPTER_PATH/merged_model by default.

Environment overrides:
  CONDA_ENV       Run with a named conda environment; otherwise use active Python.
  EXPORT_DEVICE   cpu or auto (default: cpu). auto can use the selected GPU.
  EXPORT_SIZE     Maximum output shard size in GB (default: 5).
  MERGED_SUBDIR   Child directory name under ADAPTER_PATH (default: merged_model).
  DRY_RUN         Set to 1 to validate and print the export config without merging.
EOF
}

if [[ -z "$CONFIG_INPUT" ]]; then
    usage >&2
    exit 2
fi
case "$EXPORT_DEVICE" in cpu|auto) ;;
    *) printf 'EXPORT_DEVICE must be cpu or auto.\n' >&2; exit 2 ;;
esac
case "$DRY_RUN" in 0|1) ;;
    *) printf 'DRY_RUN must be 0 or 1.\n' >&2; exit 2 ;;
esac
if [[ ! "$EXPORT_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    printf 'EXPORT_SIZE must be a positive integer.\n' >&2
    exit 2
fi
if [[ -z "$MERGED_SUBDIR" || "$MERGED_SUBDIR" == "." || "$MERGED_SUBDIR" == ".." || "$MERGED_SUBDIR" == */* ]]; then
    printf 'MERGED_SUBDIR must be one child-directory name.\n' >&2
    exit 2
fi

resolve_input_path() {
    local value="$1"
    if [[ "$value" = /* ]]; then
        printf '%s\n' "$value"
    elif [[ -e "$value" ]]; then
        realpath "$value"
    else
        printf '%s/%s\n' "$REPO_ROOT" "$value"
    fi
}

CONFIG_PATH="$(resolve_input_path "$CONFIG_INPUT")"
if [[ ! -f "$CONFIG_PATH" ]]; then
    printf 'Configuration file not found: %s\n' "$CONFIG_PATH" >&2
    exit 1
fi
if [[ ! -d "$LLAMAFACTORY_DIR" || ! -f "$LOCK_FILE" ]]; then
    printf 'LLaMAFactory checkout or lock file is missing.\n' >&2
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

EXPECTED_COMMIT="$(sed -n 's/^  commit: //p' "$LOCK_FILE")"
ACTUAL_COMMIT="$(git -C "$LLAMAFACTORY_DIR" rev-parse HEAD)"
if [[ -z "$EXPECTED_COMMIT" || "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
    printf 'LLaMAFactory lock mismatch: expected %s, found %s\n' \
        "${EXPECTED_COMMIT:-missing}" "$ACTUAL_COMMIT" >&2
    exit 1
fi

if [[ -z "$ADAPTER_INPUT" ]]; then
    mapfile -t CONFIG_META < <(
        "${PYTHON_RUNNER[@]}" "${REPO_ROOT}/scripts/sft_config.py" \
            --config "$CONFIG_PATH" --repo-root "$REPO_ROOT" --format lines
    )
    if [[ "${#CONFIG_META[@]}" -ne 5 ]]; then
        printf 'Failed to read adapter path from YAML: %s\n' "$CONFIG_PATH" >&2
        exit 1
    fi
    ADAPTER_PATH="${CONFIG_META[2]}"
else
    ADAPTER_PATH="$(resolve_input_path "$ADAPTER_INPUT")"
fi
if [[ ! -d "$ADAPTER_PATH" ]]; then
    printf 'Adapter directory not found: %s\n' "$ADAPTER_PATH" >&2
    exit 1
fi

EXPORT_DIR="${ADAPTER_PATH}/${MERGED_SUBDIR}"
mkdir -p "${REPO_ROOT}/tmp"
MERGE_CONFIG="$(mktemp "${REPO_ROOT}/tmp/sft_merge_XXXXXX.yaml")"
trap 'rm -f "$MERGE_CONFIG"' EXIT

"${PYTHON_RUNNER[@]}" "${REPO_ROOT}/scripts/prepare_sft_merge.py" \
    --training-config "$CONFIG_PATH" \
    --adapter-path "$ADAPTER_PATH" \
    --export-dir "$EXPORT_DIR" \
    --output-config "$MERGE_CONFIG" \
    --repo-root "$REPO_ROOT" \
    --export-device "$EXPORT_DEVICE" \
    --export-size "$EXPORT_SIZE"

printf '\n[%s] Merging SFT adapter\n' "$(date --iso-8601=seconds)"
printf 'Training config: %s\nAdapter: %s\nExport: %s\nDevice: %s\n' \
    "$CONFIG_PATH" "$ADAPTER_PATH" "$EXPORT_DIR" "$EXPORT_DEVICE"

if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nGenerated LLaMAFactory export config:\n'
    sed -n '1,200p' "$MERGE_CONFIG"
    printf '\nDry run command: python -m llamafactory.cli export %q\n' "$MERGE_CONFIG"
    exit 0
fi

(
    cd "$REPO_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    PYTHONPATH="${LLAMAFACTORY_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    TOKENIZERS_PARALLELISM=false \
        "${PYTHON_RUNNER[@]}" -m llamafactory.cli export "$MERGE_CONFIG"
)

if [[ ! -f "$EXPORT_DIR/config.json" ]]; then
    printf 'Merge finished without config.json in %s\n' "$EXPORT_DIR" >&2
    exit 1
fi
if ! find "$EXPORT_DIR" -maxdepth 1 -type f \
    \( -name 'model*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit | grep -q .; then
    printf 'Merge finished without model weight shards in %s\n' "$EXPORT_DIR" >&2
    exit 1
fi

cp "$CONFIG_PATH" "$EXPORT_DIR/source_training_config.yaml"
cp "$MERGE_CONFIG" "$EXPORT_DIR/merge_export_config.yaml"
printf 'Merged model saved to: %s\n' "$EXPORT_DIR"
