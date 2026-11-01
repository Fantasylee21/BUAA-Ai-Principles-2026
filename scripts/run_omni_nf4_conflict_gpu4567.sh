#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${RL_PYTHON:-/home/djy/anaconda3/envs/llama-factory/bin/python}"
MODEL="$ROOT/../ckpt/Qwen2.5-Omni-7B"
BEST="$ROOT/output/sft/qwen25omni/qlora_sft_v5_targeted_7b/best_checkpoint.json"
RUN="$ROOT/output/rl/omni_v5_nf4_conflict75_gpu4567"
ADAPTER="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$BEST")"
export CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
resume=()
if [[ "${RL_RESUME:-0}" == 1 ]]; then resume=(--resume); fi
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=4 rl/train_omni_v5_nf4_conflict.py --model "$MODEL" --adapter "$ADAPTER" --output "$RUN" --steps "${RL_STEPS:-40}" --generations "${RL_GENERATIONS:-4}" --lr "${RL_LR:-1e-6}" --save-every 20 --reward-scheme metric --sampling stratified "${resume[@]}"
bash scripts/run_omni_parallel_nf4_rl.sh "$RUN/latest" val omni_v5_nf4_conflict75_val --overwrite
bash scripts/run_omni_parallel_nf4_rl.sh "$RUN/latest" test omni_v5_nf4_conflict75_test --overwrite
"$PYTHON_BIN" rl/make_submission.py --input output/sft/evaluation/omni_v5_nf4_conflict75_test_predictions.csv --test-file Dataset/test_release.jsonl --output submit_rl_nf4_conflict75.csv --strict
echo "Validation: output/sft/evaluation/omni_v5_nf4_conflict75_val_metrics.json"
echo "Submission: submit_rl_nf4_conflict75.csv"
