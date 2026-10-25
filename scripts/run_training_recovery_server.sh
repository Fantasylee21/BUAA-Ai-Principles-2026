#!/usr/bin/env bash
# Continue the existing checkpoints in a new batch; preserve the failed r2 report.
set -euo pipefail
MODE=${1:?Missing start/protocol/collect/visual}
ROOT=${2:?Missing server project root}
SOURCE=$ROOT/releases/model-training-v1-20261025
BATCH=$ROOT/runs/model-training-batch-20261025-r3
PYTHON=$ROOT/.venv-sft-protocol/bin/python
SCRIPT=$SOURCE/scripts/run_training_recovery_server.sh
mkdir -p "$BATCH/logs"
test -s "$BATCH/IMPLEMENTATION_PLAN.md"
test -s "$BATCH/source-manifest.json"
if [[ "$MODE" == start ]]; then
 if tmux has-session -t buaa-training-recovery 2>/dev/null; then
  printf 'Existing recovery controller retained.\n'
  exit 0
 fi
 tmux new-session -d -s buaa-training-recovery -n protocol bash "$SCRIPT" protocol "$ROOT"
 tmux new-window -d -t buaa-training-recovery -n collect bash "$SCRIPT" collect "$ROOT"
 tmux new-window -d -t buaa-training-recovery -n visual-cpu bash "$SCRIPT" visual "$ROOT"
 exit 0
fi
[[ "$MODE" == protocol || "$MODE" == collect || "$MODE" == visual ]]
cd "$SOURCE"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PATH="$ROOT/media-tools/bin:$PATH"
"$PYTHON" - "$BATCH/source-manifest.json" "$SOURCE" <<'PY'
import hashlib,json,sys
from pathlib import Path
source=Path(sys.argv[2]).resolve()
for relative,expected in json.load(open(sys.argv[1]))['source_sha256'].items():
    path=(source/relative).resolve()
    if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Recovery source changed: '+relative)
PY
if [[ "$MODE" == visual ]]; then
 VISUAL_SOURCE=$ROOT/releases/visual-expert-v2-20261025
 VISUAL_RUN=$ROOT/runs/visual-expert-cpu-recovery-20261025
 mkdir -p "$VISUAL_RUN/logs"
 "$PYTHON" - "$BATCH/source-manifest.json" "$VISUAL_SOURCE" <<'PY'
import hashlib,json,sys
from pathlib import Path
source=Path(sys.argv[2]).resolve()
for relative,expected in json.load(open(sys.argv[1]))['visual_recovery_source_sha256'].items():
    path=(source/relative).resolve()
    if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Visual CPU recovery source changed: '+relative)
PY
 cd "$VISUAL_SOURCE"
 export CUDA_VISIBLE_DEVICES=
 exec "$ROOT/.venv-rl/bin/python" -m scripts.vision_expert_cpu_recovery \
  --source-run "$ROOT/runs/evidence-bottleneck-v1-20261024-r1" \
  --cache-root "$ROOT/runs/visual-expert-v2-20261025" --output-dir "$VISUAL_RUN" \
  >> "$VISUAL_RUN/logs/evaluate.log" 2>&1
fi
exec "$PYTHON" -m scripts.server_training_batch "$MODE" \
 --project-root "$ROOT" --batch-dir "$BATCH" \
 --seed42-run "$ROOT/runs/protocol-sft-ablation-seed42-20261025" \
 --seed43-run "$ROOT/runs/protocol-sft-ablation-seed43-20261025" \
 --final-run "$ROOT/runs/protocol-sft-final-20261025" \
 --vision-run "$ROOT/runs/visual-expert-cpu-recovery-20261025" \
 --audio-cache-dir "$ROOT/releases/evidence-calibrator-v1-20261025/data/llamafactory/generated/protocol_consistent" \
 --gpu 1 --max-wait-hours 72 --poll-seconds 20 >> "$BATCH/logs/$MODE-driver.log" 2>&1
