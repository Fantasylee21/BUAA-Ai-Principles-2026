#!/usr/bin/env bash
# Usage: bash scripts/run_omni_parallel_bf16.sh ADAPTER {val|test} RUN_NAME [--overwrite]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${RL_PYTHON:-/home/djy/anaconda3/envs/llama-factory/bin/python}"
MODEL="$ROOT/../ckpt/Qwen2.5-Omni-7B"
ADAPTER="${1:?adapter path required}"
SPLIT="${2:?val or test required}"
RUN_NAME="${3:?run name required}"
shift 3
if [[ "$SPLIT" != val && "$SPLIT" != test ]]; then
  echo "Expected split val or test" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"
mkdir -p output/sft/evaluation

pids=()
for idx in 0 1; do
  gpu=$((4 + idx))
  log="output/sft/evaluation/${RUN_NAME}_shard${idx}.log"
  echo "Starting $SPLIT shard $idx on physical GPU $gpu (log: $log)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/evaluate_sft.py \
    --config configs/sft/qwen25omni/qlora_sft_v5_targeted_7b.yaml \
    --model-path "$MODEL" --adapter-path "$ADAPTER" \
    --load-in-4bit --split "$SPLIT" \
    --num-shards 2 --shard-index "$idx" \
    --run-name "${RUN_NAME}_shard${idx}" "$@" >"$log" 2>&1 &
  pids+=("$!")
done

status=0
for idx in 0 1; do
  if ! wait "${pids[$idx]}"; then
    echo "Shard $idx failed; last log lines:" >&2
    tail -30 "output/sft/evaluation/${RUN_NAME}_shard${idx}.log" >&2
    status=1
  fi
done
if [[ "$status" != 0 ]]; then
  exit 1
fi

"$PYTHON_BIN" scripts/merge_sft_shards.py \
  --base-name "$RUN_NAME" --split "$SPLIT" --num-shards 2
