#!/usr/bin/env bash
# Prepare, train, select, and generate a test submission for Omni 7B v5.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="configs/sft/qwen25omni/qlora_sft_v5_targeted_7b.yaml"
STAGE="${STAGE:-all}" # prepare | train | evaluate | submit | finish | all
GPU_ID="${GPU_ID:-4,5}"
EVAL_GPU_ID="${EVAL_GPU_ID:-${GPU_ID%%,*}}"
SOURCE_DIR="${SOURCE_DIR:-data/llamafactory/generated}"
RUN_NAME="qwen25omni_7b_v5_test"
SUBMIT_PATH="${SUBMIT_PATH:-$ROOT/submit.csv}"
TRAIN_OUTPUT="output/sft/qwen25omni/qlora_sft_v5_targeted_7b"
# PEFT otherwise probes a derived Hub ID while saving this local adapter.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON_RUNNER=(python)
if [[ -n "${CONDA_ENV:-}" ]]; then
    PYTHON_RUNNER=(conda run --no-capture-output -n "$CONDA_ENV" python)
fi
cd "$ROOT"

case "$STAGE" in
    prepare|train|evaluate|submit|finish|all) ;;
    *) echo "STAGE must be prepare, train, evaluate, submit, finish, or all" >&2; exit 2 ;;
esac

if [[ "$STAGE" == prepare || "$STAGE" == all ]]; then
    "${PYTHON_RUNNER[@]}" scripts/prepare_omni_v5_data.py --source-dir "$SOURCE_DIR"
fi

if [[ "$STAGE" == train || "$STAGE" == all ]]; then
    [[ -s data/llamafactory/generated/omni_v5_targeted/dataset_info.json ]] || {
        echo 'Missing v5 data. Run STAGE=prepare first.' >&2; exit 1;
    }
    FINISHED_CHECKPOINT=0
    for CHECKPOINT in "$TRAIN_OUTPUT"/checkpoint-*; do
        if [[ -s "$CHECKPOINT/adapter_model.safetensors" ]]; then
            FINISHED_CHECKPOINT=1
            break
        fi
    done
    if [[ -s "$TRAIN_OUTPUT/train_results.json" && "$FINISHED_CHECKPOINT" == 1 ]]; then
        echo "Training already completed in $TRAIN_OUTPUT; continuing to evaluation."
    else
        bash scripts/run_sft_training.sh qlora "$CONFIG" "$GPU_ID"
    fi
fi

if [[ "$STAGE" == evaluate || "$STAGE" == finish || "$STAGE" == all ]]; then
    # Select only by the complete official validation score; no test labels are used.
    CUDA_VISIBLE_DEVICES="$EVAL_GPU_ID" "${PYTHON_RUNNER[@]}" scripts/select_sft_checkpoint.py \
        --config "$CONFIG"
fi

if [[ "$STAGE" == submit || "$STAGE" == finish || "$STAGE" == all ]]; then
    SELECTION="$TRAIN_OUTPUT/best_checkpoint.json"
    [[ -s "$SELECTION" ]] || {
        echo "Missing $SELECTION. Run STAGE=evaluate first." >&2; exit 1;
    }
    [[ -s Dataset/test_release.jsonl ]] || {
        echo 'Missing Dataset/test_release.jsonl.' >&2; exit 1;
    }
    ADAPTER_PATH="$("${PYTHON_RUNNER[@]}" - "$SELECTION" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(json.loads(Path(sys.argv[1]).read_text())["checkpoint"])
if not (checkpoint / "adapter_model.safetensors").is_file():
    raise SystemExit(f"Missing selected adapter: {checkpoint}")
print(checkpoint.resolve())
PY
)"
    SPLIT=test ADAPTER_PATH="$ADAPTER_PATH" LOAD_IN_4BIT=1 RUN_NAME="$RUN_NAME" OVERWRITE=1 \
        bash scripts/run_sft_inference.sh qlora "$CONFIG" "$EVAL_GPU_ID"
    "${PYTHON_RUNNER[@]}" scripts/validate_sft_submission.py \
        --test Dataset/test_release.jsonl \
        --predictions "output/sft/evaluation/${RUN_NAME}_predictions.csv" \
        --metrics "output/sft/evaluation/${RUN_NAME}_metrics.json" \
        --output "$SUBMIT_PATH"
fi
