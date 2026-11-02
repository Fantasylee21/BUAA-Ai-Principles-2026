#!/usr/bin/env bash
# Start only this experiment's processes. Shared GPUs are explicitly assigned.
set -euo pipefail
ROOT="${BUAA_RL_ROOT:-/home/nas/xingzhenbang/buaa-ai2026}"
SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?Usage: start_finite_tmux.sh /absolute/run-dir [gpu1] [gpu2] [session]}"
GPU1="${2:-1}"
GPU2="${3:-2}"
SESSION="${4:-buaa-finite}"
PYTHON="$ROOT/.venv-rl/bin/python"
export PATH="$ROOT/media-tools/bin:$PATH"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR="$ROOT/tmp"
export PYTHONUNBUFFERED=1
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "Session already exists: $SESSION (nothing restarted)" >&2
  exit 1
fi
mkdir -p "$RUN/logs"
ARGS=(run --run-dir "$RUN" --gpus "$GPU1" "$GPU2" --memory-gib 32)
if [[ -f "$RUN/context.json" ]]; then
  ARGS+=(--resume)
else
  ARGS+=(--data-root "$ROOT/Dataset" --model-path "$ROOT/models/Qwen2.5-Omni-3B"
         --sft-adapter-path "$ROOT/models/qwen25omni-sft"
         --diagnostic-predictions "$ROOT/repo/runs/omni-f1-pipeline-42/cache/train_predictions.jsonl")
fi
printf -v DASH '%q ' "$PYTHON" -u "$SOURCE/scripts/finite_dashboard.py" "$RUN"
printf -v COMMAND '%q ' "$PYTHON" -u -m rl.train_finite_actions "${ARGS[@]}"
printf -v LOG '%q' "$RUN/logs/pipeline.log"
tmux new-session -d -s "$SESSION" -n dashboard -c "$SOURCE" \
  -e "PATH=$PATH" -e "HF_HUB_OFFLINE=1" -e "TMPDIR=$TMPDIR" "$DASH"
tmux set-option -t "$SESSION" remain-on-exit on
tmux new-window -t "$SESSION:1" -n pipeline -c "$SOURCE" "set -o pipefail; $COMMAND 2>&1 | tee -a $LOG"
tmux set-option -w -t "$SESSION:1" remain-on-exit on
for ITEM in '2 training train' '3 validation worker' '4 preflight preflight' '5 cache0 cache-0' '6 cache1 cache-1'; do
  read -r INDEX NAME FILE <<< "$ITEM"
  printf -v TAIL '%q ' tail -n 35 -F "$RUN/logs/$FILE.log"
  tmux new-window -t "$SESSION:$INDEX" -n "$NAME" -c "$SOURCE" "$TAIL"
done
tmux select-window -t "$SESSION:0"
echo "Started $SESSION. Observe with: tmux attach -t $SESSION"
