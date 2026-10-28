#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:-/home/nas/xingzhenbang/buaa-ai2026}
RUN=${2:-$ROOT/runs/conflict-reward-v1-20261028}
PREVIOUS=${3:-$ROOT/runs/joint-affect-v1-20261026}
BASELINE=${4:-$ROOT/incoming/submission09-confirmed.csv}
SESSION=buaa-conflict-rl
SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON="$ROOT/.venv-joint-affect/bin/python"
export PATH="$ROOT/media-tools/bin:$PATH"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$SOURCE"
test -x "$PYTHON"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Existing session $SESSION; refusing duplicate launch."
  exit 2
fi
mkdir -p "$RUN/logs"
"$PYTHON" -B -m scripts.conflict_reward_pipeline prepare --project-root "$ROOT" --run-dir "$RUN" --previous-run "$PREVIOUS" --baseline "$BASELINE" > "$RUN/logs/prepare.log" 2>&1
printf -v COMMAND '%q -B -m scripts.conflict_reward_pipeline run --run-dir %q 2>&1 | tee -a %q; echo "Controller exited: ${PIPESTATUS[0]}"' "$PYTHON" "$RUN" "$RUN/logs/controller.log"
tmux new-session -d -s "$SESSION" -n pipeline -c "$SOURCE" "bash -o pipefail -c $(printf '%q' "$COMMAND"); exec bash"
printf -v PROGRESS '%q -B -m scripts.conflict_reward_progress --run-dir %q' "$PYTHON" "$RUN"
tmux new-window -t "$SESSION" -n progress -c "$SOURCE" "$PROGRESS"
tmux select-window -t "$SESSION:progress"
echo "tmux attach -t $SESSION"
