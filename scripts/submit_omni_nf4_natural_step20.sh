#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${RL_PYTHON:-/home/djy/anaconda3/envs/llama-factory/bin/python}"
ADAPTER="$ROOT/output/rl/omni_nf4_natural_gpu4567/checkpoint-0020"
RUN_NAME="omni_nf4_natural_step20_test"
bash scripts/run_omni_parallel_nf4_rl.sh "$ADAPTER" test "$RUN_NAME" --overwrite
"$PYTHON_BIN" - "$ROOT/submit.csv" "$ROOT/output/sft/evaluation/${RUN_NAME}_predictions.csv" "$ROOT/output/sft/evaluation/${RUN_NAME}_hybrid.csv" <<'PY'
import csv, sys
with open(sys.argv[1], newline='', encoding='utf-8') as f:
    base = list(csv.DictReader(f))
with open(sys.argv[2], newline='', encoding='utf-8') as f:
    candidate = list(csv.DictReader(f))
by_id = {r['sample_id']: r for r in candidate}
if len(base) != 734 or len(candidate) != 734 or len(by_id) != 734 or len({r['sample_id'] for r in base}) != 734 or set(by_id) != {r['sample_id'] for r in base}:
    raise ValueError('Submission coverage mismatch')
for row in base:
    other = by_id[row['sample_id']]
    row['is_uncertain'] = other['is_uncertain']
    row['clarifying_question'] = other['clarifying_question']
with open(sys.argv[3], 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['sample_id', 'pred_emotion', 'is_uncertain', 'clarifying_question'])
    writer.writeheader()
    writer.writerows(base)
PY
"$PYTHON_BIN" rl/make_submission.py --input "$ROOT/output/sft/evaluation/${RUN_NAME}_hybrid.csv" --test-file Dataset/test_release.jsonl --output "$ROOT/submit_rl_nf4_natural_step20.csv" --strict
echo "Submission: $ROOT/submit_rl_nf4_natural_step20.csv"
