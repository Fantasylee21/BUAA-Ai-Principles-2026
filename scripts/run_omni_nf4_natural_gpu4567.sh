#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${RL_PYTHON:-/home/djy/anaconda3/envs/llama-factory/bin/python}"
MODEL="$ROOT/../ckpt/Qwen2.5-Omni-7B"
BEST="$ROOT/output/sft/qwen25omni/qlora_sft_v5_targeted_7b/best_checkpoint.json"
RUN="$ROOT/output/rl/omni_nf4_natural_gpu4567"
ADAPTER="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$BEST")"
if [[ "$ADAPTER" != "$ROOT/output/sft/qwen25omni/qlora_sft_v5_targeted_7b/checkpoint-118" ]]; then
  echo "The saved SFT baseline is checkpoint-118, but the selected checkpoint changed: $ADAPTER" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
resume=()
if [[ "${RL_RESUME:-0}" == 1 ]]; then resume=(--resume); fi
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=4 rl/train_omni_v5_nf4_natural.py --model "$MODEL" --adapter "$ADAPTER" --output "$RUN" --steps 40 --generations 4 --lr 1e-6 --save-every 10 --reward-scheme metric --sampling stratified "${resume[@]}"
for step in 10 20 30 40; do
  printf -v checkpoint '%04d' "$step"
  bash scripts/run_omni_parallel_nf4_rl.sh "$RUN/checkpoint-$checkpoint" val "omni_nf4_natural_step${step}_val" --overwrite
done
"$PYTHON_BIN" rl/select_natural_rl.py
STEP="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selection"]["step"])' "$RUN/selection.json")"
if [[ "$STEP" == 0 ]]; then
  echo "SFT remains best on validation; selection saved to $RUN/selection.json"
  exit 0
fi
printf -v checkpoint '%04d' "$STEP"
bash scripts/run_omni_parallel_nf4_rl.sh "$RUN/checkpoint-$checkpoint" test omni_nf4_natural_selected_test --overwrite
"$PYTHON_BIN" - "$ROOT/submit.csv" "$ROOT/output/sft/evaluation/omni_nf4_natural_selected_test_predictions.csv" "$ROOT/submit_rl_nf4_natural.csv" <<'PY'
import csv,sys
with open(sys.argv[1], newline='', encoding='utf-8') as f: base=list(csv.DictReader(f))
with open(sys.argv[2], newline='', encoding='utf-8') as f: rl=list(csv.DictReader(f))
by_id={r['sample_id']:r for r in rl}
if len(base)!=734 or len(by_id)!=734 or set(by_id)!={r['sample_id'] for r in base}: raise ValueError('Submission coverage mismatch')
for r in base:
    other=by_id[r['sample_id']]
    r['is_uncertain']=other['is_uncertain']
    r['clarifying_question']=other['clarifying_question']
with open(sys.argv[3], 'w', newline='', encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=['sample_id','pred_emotion','is_uncertain','clarifying_question']); w.writeheader(); w.writerows(base)
PY
"$PYTHON_BIN" rl/make_submission.py --input "$ROOT/submit_rl_nf4_natural.csv" --test-file Dataset/test_release.jsonl --output "$ROOT/submit_rl_nf4_natural_checked.csv" --strict
echo "Submission: $ROOT/submit_rl_nf4_natural_checked.csv"
