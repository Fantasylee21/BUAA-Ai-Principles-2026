#!/usr/bin/env python3
"""Select a validation checkpoint with SFT emotion and RL uncertainty."""
import csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qwen25vl_baseline import metric_block

out = ROOT / 'output/sft/evaluation'
base = out / 'qwen25omni_qlora_sft_v5_targeted_7b_checkpoint-118_val_details.jsonl'
rows = [json.loads(x) for x in base.open(encoding='utf-8')]
if len(rows) != 666 or len({r['sample_id'] for r in rows}) != 666:
    raise ValueError('SFT validation baseline is incomplete')
def report(records):
    m = metric_block(records)
    return {k: m[k] for k in ('competition_score','macro_f1_three_classes','conflict_identification_accuracy','aligned_calibration')}
results = [{'step': 0, **report(rows)}]
for step in (10,20,30,40):
    path = out / f'omni_nf4_natural_step{step}_val_details.jsonl'
    candidate = [json.loads(x) for x in path.open(encoding='utf-8')]
    if any(r.get('inference_error') or r.get('parse_status') != 'json' for r in candidate):
        raise ValueError(f'Invalid candidate predictions: {path}')
    index = {r['sample_id']: r for r in candidate}
    if len(index) != len(rows) or set(index) != {r['sample_id'] for r in rows}:
        raise ValueError(f'Incomplete candidate: {path}')
    hybrid = [{**r, 'is_uncertain': index[r['sample_id']]['is_uncertain'],
               'clarifying_question': index[r['sample_id']]['clarifying_question']} for r in rows]
    results.append({'step':step, **report(hybrid)})
best = max(results, key=lambda r:r['competition_score'])
path = ROOT / 'output/rl/omni_nf4_natural_gpu4567/selection.json'
path.write_text(json.dumps({'selection':best, 'candidates':results, 'emotion_source':'SFT checkpoint-118'},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'selection':best,'candidates':results},ensure_ascii=False))
