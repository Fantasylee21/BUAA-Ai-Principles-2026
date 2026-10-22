#!/usr/bin/env bash
set -euo pipefail
ROOT="${BUAA_RL_ROOT:-/home/nas/xingzhenbang/buaa-ai2026}"
SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?Usage: start_grounded_tmux.sh /absolute/run-dir [gpu1] [gpu2] [session]}"
GPU1="${2:-1}"; GPU2="${3:-2}"; SESSION="${4:-buaa-grounded}"
PYTHON="$ROOT/.venv-rl/bin/python"
export PATH="$ROOT/media-tools/bin:$PATH"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export TMPDIR="$ROOT/tmp"
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "Session exists; no duplicate job started: $SESSION" >&2
  exit 1
fi
mkdir -p "$RUN/logs"
ARGS=(run --run-dir "$RUN" --gpus "$GPU1" "$GPU2" --memory-gib 32)
if [[ -n "${BUAA_GROUNDED_STARTED:-}" ]]; then
  ARGS+=(--budget-start "$BUAA_GROUNDED_STARTED")
fi
if [[ -f "$RUN/context.json" ]]; then
  ARGS+=(--resume)
else
  ARGS+=(--data-root "$ROOT/Dataset" --model-path "$ROOT/models/Qwen2.5-Omni-3B"
    --sft-adapter-path "$ROOT/models/qwen25omni-sft"
    --baseline-csv "$ROOT/inputs/baseline_090813.csv"
    --legacy-run "$ROOT/runs/finite-v1-20261022-r3")
fi
printf -v DASH '%q ' "$PYTHON" -u "$SOURCE/scripts/grounded_dashboard.py" "$RUN" --gpus "$GPU1" "$GPU2"
printf -v PARAMETERS '%q ' "$PYTHON" -u "$SOURCE/scripts/grounded_dashboard.py" "$RUN" --view parameters --gpus "$GPU1" "$GPU2"
printf -v COMMAND '%q ' "$PYTHON" -u -m rl.train_grounded_uncertainty "${ARGS[@]}"
printf -v LOG '%q' "$RUN/logs/pipeline.log"
tmux new-session -d -s "$SESSION" -n dashboard -c "$SOURCE" \
  -e "PATH=$PATH" -e "HF_HUB_OFFLINE=1" -e "TMPDIR=$TMPDIR" "$DASH"
tmux set-option -t "$SESSION" remain-on-exit on
tmux new-window -t "$SESSION:1" -n pipeline -c "$SOURCE" "set -o pipefail; $COMMAND 2>&1 | tee -a $LOG"
tmux set-option -w -t "$SESSION:1" remain-on-exit on
for ITEM in '2 training train' '3 validation worker' '4 preflight probe' '5 cache0 cache-0' '6 cache1 cache-1' '7 learning-check smoke'; do
  read -r INDEX NAME FILE <<< "$ITEM"
  printf -v TAIL '%q ' tail -n 25 -F "$RUN/logs/$FILE.log"
  tmux new-window -t "$SESSION:$INDEX" -n "$NAME" -c "$SOURCE" "$TAIL"
done
tmux new-window -t "$SESSION:8" -n parameters -c "$SOURCE" "$PARAMETERS"
tmux select-window -t "$SESSION:0"
echo "Started $SESSION. Outside tmux: tmux attach -t $SESSION"
echo "Already inside tmux: tmux switch-client -t $SESSION"
