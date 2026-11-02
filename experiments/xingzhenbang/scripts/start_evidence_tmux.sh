#!/usr/bin/env bash
set -euo pipefail
ROOT="${BUAA_RL_ROOT:-/home/nas/xingzhenbang/buaa-ai2026}"
SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?Usage: start_evidence_tmux.sh /absolute/run-dir [gpu1] [gpu2] [session]}"
GPU1="${2:-1}"; GPU2="${3:-2}"; SESSION="${4:-buaa-evidence}"
PYTHON="$ROOT/.venv-rl/bin/python"
export PATH="$ROOT/media-tools/bin:$PATH"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$ROOT/tmp"
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "Session exists; refusing duplicate experiment: $SESSION" >&2
  exit 1
fi
mkdir -p "$RUN/logs"
ARGS=(run --run-dir "$RUN" --gpus "$GPU1" "$GPU2" --memory-gib 32
  --config "$SOURCE/configs/rl/evidence_bottleneck_v1.json")
if [[ -f "$RUN/context.json" ]]; then
  ARGS+=(--resume)
else
  ARGS+=(--data-root "$ROOT/Dataset" --model-path "$ROOT/models/Qwen2.5-Omni-3B"
    --sft-adapter-path "$ROOT/models/qwen25omni-sft"
    --baseline-csv "$ROOT/inputs/baseline_090813.csv"
    --source-run "$ROOT/runs/modality-relation-v1-20261024-r1")
fi
printf -v DASH '%q ' "$PYTHON" -u "$SOURCE/scripts/evidence_dashboard.py" "$RUN" --gpus "$GPU1" "$GPU2"
printf -v PARAMETERS '%q ' "$PYTHON" -u "$SOURCE/scripts/evidence_dashboard.py" "$RUN" --gpus "$GPU1" "$GPU2" --parameters
printf -v COMMAND '%q ' "$PYTHON" -u -m rl.train_evidence_bottleneck "${ARGS[@]}"
printf -v LOG '%q' "$RUN/logs/pipeline.log"
tmux new-session -d -s "$SESSION" -n overview -c "$SOURCE" -e "PATH=$PATH" -e "TMPDIR=$TMPDIR" "$DASH"
tmux set-option -t "$SESSION" remain-on-exit on
tmux set-option -t "$SESSION" mouse on
tmux new-window -t "$SESSION:1" -n pipeline -c "$SOURCE" "set -o pipefail; $COMMAND 2>&1 | tee -a $LOG"
for INDEX in 0 1; do
  printf -v TAIL '%q ' tail -n 25 -F "$RUN/logs/gpu-$INDEX.log"
  tmux new-window -t "$SESSION:$((INDEX+2))" -n "features-$INDEX" -c "$SOURCE" "$TAIL"
done
printf -v FITLOG '%q ' tail -n 25 -F "$RUN/logs/fit-42.log" "$RUN/logs/fit-43.log"
tmux new-window -t "$SESSION:4" -n cpu-fits -c "$SOURCE" "$FITLOG"
tmux new-window -t "$SESSION:5" -n parameters -c "$SOURCE" "$PARAMETERS"
tmux select-window -t "$SESSION:0"
echo "Started $SESSION. Outside tmux: tmux attach -t $SESSION"
echo "Already inside tmux: tmux switch-client -t $SESSION"
