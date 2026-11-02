#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:-/home/nas/xingzhenbang/buaa-ai2026}
RUN=${2:-$ROOT/runs/joint-affect-v1-20261026}
SESSION=${3:-buaa-joint-affect}
SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON="$ROOT/.venv-joint-affect/bin/python"
export PATH="$ROOT/media-tools/bin:$PATH"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$SOURCE"
test -x "$PYTHON"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Existing session: $SESSION. Attach to it; refusing duplicate controller."
  exit 2
fi
mkdir -p "$RUN/logs"
if test ! -e "$RUN/plan.json"; then
  "$PYTHON" -B -m scripts.joint_affect_pipeline prepare --project-root "$ROOT" --run-dir "$RUN" > "$RUN/logs/prepare.log" 2>&1
fi
printf -v COMMAND '%q -B -m scripts.joint_affect_pipeline run --project-root %q --run-dir %q 2>&1 | tee -a %q; echo "Batch controller exited: ${PIPESTATUS[0]}"' "$PYTHON" "$ROOT" "$RUN" "$RUN/logs/controller.log"
tmux new-session -d -s "$SESSION" -n pipeline -c "$SOURCE" "bash -o pipefail -c $(printf '%q' "$COMMAND"); exec bash"
printf -v PROGRESS '%q -B -m scripts.joint_affect_progress --run-dir %q' "$PYTHON" "$RUN"
tmux new-window -t "$SESSION" -n progress -c "$SOURCE" "$PROGRESS"
tmux select-window -t "$SESSION:progress"
echo "tmux attach -t $SESSION"
