#!/usr/bin/env bash
# Start once; detached server windows own all continuation and delivery work.
set -euo pipefail
MODE=${1:?Usage: run_server_training_batch.sh start PROJECT_ROOT SOURCE_ROOT}
PROJECT_ROOT=${2:?Missing server project root}
SOURCE_ROOT=${3:?Missing frozen source directory}
PYTHON=$PROJECT_ROOT/.venv-sft-protocol/bin/python
BATCH=$PROJECT_ROOT/runs/model-training-batch-20261025
SCRIPT=$SOURCE_ROOT/scripts/run_server_training_batch.sh
test -x "$PYTHON"
test -f "$BATCH/IMPLEMENTATION_PLAN.md"
test -f "$BATCH/source-manifest.json"
mkdir -p "$BATCH/logs"
if [[ "$MODE" == start ]]; then
 if tmux has-session -t buaa-training-delivery 2>/dev/null; then
  printf 'Existing buaa-training-delivery session retained; no duplicate started.\n'
  exit 0
 fi
 tmux new-session -d -s buaa-training-delivery -n protocol bash "$SCRIPT" protocol "$PROJECT_ROOT" "$SOURCE_ROOT"
 tmux new-window -d -t buaa-training-delivery -n collect bash "$SCRIPT" collect "$PROJECT_ROOT" "$SOURCE_ROOT"
 printf 'Started server continuation and result collection in buaa-training-delivery.\n'
 exit 0
fi
if [[ "$MODE" != protocol && "$MODE" != collect ]]; then
 printf 'Unknown mode: %s\n' "$MODE" >&2
 exit 2
fi
cd "$SOURCE_ROOT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PATH="$PROJECT_ROOT/media-tools/bin:$PATH"
"$PYTHON" - "$BATCH/source-manifest.json" "$SOURCE_ROOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
manifest=json.load(open(sys.argv[1]));source=Path(sys.argv[2]).resolve()
for relative,expected in manifest['source_sha256'].items():
 path=(source/relative).resolve()
 if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
  raise RuntimeError('Uploaded source checksum changed: '+relative)
PY
exec "$PYTHON" -m scripts.server_training_batch "$MODE" \
 --project-root "$PROJECT_ROOT" --batch-dir "$BATCH" \
 --seed42-run "$PROJECT_ROOT/runs/protocol-sft-ablation-seed42-20261025" \
 --seed43-run "$PROJECT_ROOT/runs/protocol-sft-ablation-seed43-20261025" \
 --final-run "$PROJECT_ROOT/runs/protocol-sft-final-20261025" \
 --vision-run "$PROJECT_ROOT/runs/visual-expert-v1-20261025" \
 --audio-cache-dir "$PROJECT_ROOT/releases/evidence-calibrator-v1-20261025/data/llamafactory/generated/protocol_consistent" \
 --gpu 1 --max-wait-hours 72 --poll-seconds 20 >> "$BATCH/logs/$MODE-driver.log" 2>&1
