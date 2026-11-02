#!/usr/bin/env bash
# Visual v2 has its own source and cache identity; the original attempt is kept.
set -euo pipefail
ROOT=${1:?Missing server project root}
SOURCE=${2:?Missing visual v2 source directory}
RUN=$ROOT/runs/visual-expert-v2-20261025
DOWNLOAD=$ROOT/runs/visual-expert-v1-20261025/download-manifest.json
PYTHON=$ROOT/.venv-rl/bin/python
mkdir -p "$RUN/logs"
cd "$SOURCE"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PATH="$ROOT/media-tools/bin:$PATH"
status() {
 "$PYTHON" - "$RUN" "$1" "${2:-0}" <<'PY'
import json,sys,time
from pathlib import Path
v={'phase':sys.argv[2],'exit_code':int(sys.argv[3]),'updated_unix':time.time(),
   'host':'10.62.193.44','gpu':2,'official_validation_used':False,'test_inference':False}
p=Path(sys.argv[1])/'pipeline.status.json';temp=p.with_suffix('.tmp');temp.write_text(json.dumps(v,indent=2)+'\n');temp.replace(p)
print(json.dumps(v),flush=True)
PY
}
trap 'task_code=$?; if [[ $task_code -ne 0 ]]; then status failed "$task_code"; fi' EXIT
sha256sum "$0" > "$RUN/pipeline-launcher.sha256"
"$PYTHON" - "$DOWNLOAD" "$RUN" <<'PY'
import json,sys
from pathlib import Path
source=Path(sys.argv[1]);v=json.loads(source.read_text())
assert v['status']=='complete' and v['revision']=='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
p=Path(sys.argv[2])/'weights-provenance.json';content=source.read_bytes()
if p.exists():assert p.read_bytes()==content
else:
 with p.open('xb') as f:f.write(content)
PY
used=$(nvidia-smi --id=2 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
if [[ ! "$used" =~ ^[0-9]+$ ]] || (( used > 512 )); then exit 73; fi
args=(--source-run "$ROOT/runs/evidence-bottleneck-v1-20261024-r1" --model-path "$ROOT/models/Qwen3-VL-8B-Instruct" --output-dir "$RUN" --gpu 2 --memory-gib 32)
status actual_32_video_preflight
"$PYTHON" -m scripts.vision_expert preflight "${args[@]}" > "$RUN/logs/preflight.log" 2>&1
"$PYTHON" - "$RUN/preflight.json" <<'PY'
import json,sys
v=json.load(open(sys.argv[1]));assert v['estimated_2104_video_seconds_excluding_load']<=24*3600
PY
status extracting_training_only_vision
"$PYTHON" -m scripts.vision_expert extract "${args[@]}" > "$RUN/logs/extract.log" 2>&1
status grouped_training_only_comparison
"$PYTHON" -m scripts.vision_expert evaluate "${args[@]}" > "$RUN/logs/evaluate.log" 2>&1
status complete
