#!/usr/bin/env bash
# Prepare, train, and select the Omni 7B v5 experiment.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="configs/sft/qwen25omni/qlora_sft_v5_targeted_7b.yaml"
STAGE="${STAGE:-all}" # prepare | train | evaluate | all
GPU_ID="${GPU_ID:-4,5}"
EVAL_GPU_ID="${EVAL_GPU_ID:-${GPU_ID%%,*}}"
SOURCE_DIR="${SOURCE_DIR:-data/llamafactory/generated}"
PYTHON_RUNNER=(python)
if [[ -n "${CONDA_ENV:-}" ]]; then
    PYTHON_RUNNER=(conda run --no-capture-output -n "$CONDA_ENV" python)
fi
cd "$ROOT"

case "$STAGE" in
    prepare|train|evaluate|all) ;;
    *) echo "STAGE must be prepare, train, evaluate, or all" >&2; exit 2 ;;
esac

if [[ "$STAGE" == prepare || "$STAGE" == all ]]; then
    "${PYTHON_RUNNER[@]}" scripts/prepare_omni_v5_data.py --source-dir "$SOURCE_DIR"
fi

if [[ "$STAGE" == train || "$STAGE" == all ]]; then
    [[ -s data/llamafactory/generated/omni_v5_targeted/dataset_info.json ]] || {
        echo 'Missing v5 data. Run STAGE=prepare first.' >&2; exit 1;
    }
    bash scripts/run_sft_training.sh qlora "$CONFIG" "$GPU_ID"
fi

if [[ "$STAGE" == evaluate || "$STAGE" == all ]]; then
    # Select only by the complete official validation score; no test labels are used.
    CUDA_VISIBLE_DEVICES="$EVAL_GPU_ID" "${PYTHON_RUNNER[@]}" scripts/select_sft_checkpoint.py \
        --config "$CONFIG"
fi
