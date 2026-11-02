#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${SFT_PYTHON:-/home/djy/anaconda3/envs/llama-factory/bin/python}"
CONFIG="configs/sft/qwen25omni/qlora_sft_v6_continue_7b.yaml"
RUN="$ROOT/output/sft/qwen25omni/qlora_sft_v6_continue_7b"
export PATH="$(dirname "$PYTHON_BIN"):$PATH" RL_PYTHON="$PYTHON_BIN"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
if [[ ! -s data/llamafactory/generated/omni_v5_targeted/dataset_info.json ]]; then
  echo 'Missing v5 training data: data/llamafactory/generated/omni_v5_targeted' >&2; exit 1
fi
if [[ ! -s output/sft/qwen25omni/qlora_sft_v5_targeted_7b/checkpoint-118/adapter_model.safetensors ]]; then
  echo 'Missing source SFT checkpoint-118' >&2; exit 1
fi
if [[ ! -f "$RUN/train_results.json" ]]; then
  RESUME="${RESUME:-auto}" bash scripts/run_sft_training.sh qlora "$CONFIG" 6,7
fi
for step in 30 60 90 120; do
  [[ -s "$RUN/checkpoint-$step/adapter_model.safetensors" ]] || { echo "Missing checkpoint-$step" >&2; exit 1; }
  bash scripts/run_omni_parallel_nf4_gpu67.sh "$RUN/checkpoint-$step" val "omni_sft_v6_step${step}_val" --overwrite
done
"$PYTHON_BIN" - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); output=root/'output/sft/evaluation'; candidates=[]
for step in (30,60,90,120):
    path=output/f'omni_sft_v6_step{step}_val_metrics.json'; m=json.loads(path.read_text())
    if m['samples'] != 666 or m['inference_errors'] or m['quantization'] != 'bnb_nf4_double_quant' or m['parse_status_counts'] != {'json':666}:
        raise SystemExit(f'Invalid validation: {path}')
    candidates.append({'step':step, 'checkpoint':str(root/f'output/sft/qwen25omni/qlora_sft_v6_continue_7b/checkpoint-{step}'), **{k:m[k] for k in ('Score','Macro-F1','Conflict_accuracy','Aligned_calibration')}})
best=max(candidates,key=lambda x:x['Score']); baseline=json.loads((root/'output/sft/qwen25omni/qlora_sft_v5_targeted_7b/best_checkpoint.json').read_text())
selection={'best':best,'candidates':candidates,'sft_v5_baseline':{k:baseline[k] for k in ('Score','Macro-F1','Conflict_accuracy','Aligned_calibration')}}
(root/'output/sft/qwen25omni/qlora_sft_v6_continue_7b/selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(selection,ensure_ascii=False,indent=2))
PY
ADAPTER="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["best"]["checkpoint"])' "$RUN/selection.json")"
bash scripts/run_omni_parallel_nf4_gpu67.sh "$ADAPTER" test omni_sft_v6_selected_test --overwrite
"$PYTHON_BIN" scripts/validate_sft_submission.py --test Dataset/test_release.jsonl --predictions output/sft/evaluation/omni_sft_v6_selected_test_predictions.csv --metrics output/sft/evaluation/omni_sft_v6_selected_test_metrics.json --output "$ROOT/submit_sft_v6.csv"
echo "Selection: $RUN/selection.json"
echo "Submission: $ROOT/submit_sft_v6.csv"
