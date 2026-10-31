# Fixed evidence supplement for a confirmed public baseline

`evidence_supplement.py` implements only the approved first-stage decision:

```text
baseline.is_uncertain == yes
OR (argmax(vision) != argmax(second) AND calibrated_probability > 0.5)
```

The algorithm identifier is `omni_evidence_supplement_v1`. It provides no
threshold tuning interface. Probabilities use `positive, negative, neutral`
column order. Exact ties select the first maximum, as in NumPy `argmax`.
All arithmetic runs on CPU; the caller supplies existing evidence predictions
and a question-generation callback. No fitting or Kaggle submission is done.

The currently selected baseline, supplied explicitly to the runner, is the user's
`Downloads/submission_09.csv`, reported Public Score **0.92313**, with SHA-256
`f2551b6a12fc916d804287671373bee5caa6e593315725a36ca2b1b0e20b8498`.
`CONFIRMED_BASELINES` admits only that artifact and the original confirmed v4
CSV at **0.90813**. There is no runtime option to accept an arbitrary baseline
hash. `BASELINE_SHA` remains the legacy v4 constant for compatibility. The new
target is **0.95**, which has not been reached. The approved C1 reviewed CSV
was submitted once and scored **0.90175**, below the **0.92313** baseline by
**0.02138**. Retain `submission_09.csv`. The later C2 feature calibrator failed
its internal gate and performed no test inference or Kaggle submission.

## Recorded outcomes

C1 preserved all 734 baseline emotions and 104 existing yes answers, and added
106 yes answers. Only those new questions were replaced with deterministic,
neutral VT/VA templates addressed to the current user. Emotion and uncertainty
decisions remained identical to the raw candidate. Wording changes are not
claimed to improve the competition score.

The actual approved artifact is
`runs/evidence-supplement-v1-20261025-reviewed/submission.csv`, SHA-256
`551e62dcea2fd46841e1ee3aefbe50851dd957e29728971033c7c2b3a9c620ad`.
The parent/root agent verified **0.90175** in the signed-in Kaggle interface.
Its separate `kaggle-result.json` records that later submission outcome without
changing the immutable pre-submission manifest. No submission ID or timestamp
has been invented. Keep the failed candidate as an experiment record; do not
promote it over the baseline or scan thresholds around the Public result.

C2 (`runs/evidence-calibrator-v1-20261025/final-report.json`) raised its primary
group-held-out uncertainty task component from 0.36791893 to 0.37064220, but
reduced natural-conflict detections from 80/143 to 71/143, failing the locked
gate. Parent-directory pressure testing fell from 0.34212222 to 0.33364237 and
from 75/143 to 67/143 natural-conflict detections. The candidate was not locked;
no official validation report, test inference or Kaggle submission followed.
These are branch-level diagnostics, not independent evaluation of the full
fusion system. A new visual-expert experiment remains unverified.

## Run the fixed candidate

Run from the repository root. The CPU audit reads the retrieved producer
artifacts without loading weights:

```bash
python -m scripts.run_evidence_supplement audit \
  --source-run runs/evidence-bottleneck-v1-20261024-r1/retrieved-20261025 \
  --output-dir runs/evidence-supplement-validation
```

On the original CUDA server, use its producer directory and an independent
output directory. Supply the confirmed current baseline explicitly:

```bash
python -m scripts.run_evidence_supplement export \
  --source-run /home/nas/xingzhenbang/buaa-ai2026/runs/evidence-bottleneck-v1-20261024-r1 \
  --output-dir /home/nas/xingzhenbang/buaa-ai2026/runs/evidence-supplement-v1-20261025 \
  --baseline-csv /home/nas/xingzhenbang/buaa-ai2026/runs/evidence-supplement-v1-20261025/inputs/submission_09.csv \
  --gpu 1 --memory-gib 32 --cpu-threads 4
```

The runner checks actual software, weights, raw data and completed evidence
fits before GPU use. It freezes the baseline, extracts all 734 test rows for
complete evidence auditing, releases the base model, then loads frozen v4 only
if added yes answers need questions. It performs no parameter updates. Resume
with the identical command; feature and question checksums are revalidated.
`status.json` records progress, `error.json` records a failed attempt, and
`supplement-review.json` identifies decisions that restore yes answers removed
between the legacy and current baseline. These are audit flags, not prediction
rules. Use different output directories for `audit` and `export`.

The module lives outside `rl/` because historical evidence cache identities hash
all Python files in that directory. Adding this workflow does not change that
historical source identity.

## API

- `supplement_decisions(rows, baseline_by_id, vision, second, scores)` returns
  aligned decision records. Inputs are checked for unique IDs, exact baseline
  coverage, finite normalized `N x 3` evidence and `N` calibrated probabilities.
- `build_submission(rows, baseline_rows, vision, second, scores, question_fn)`
  returns `(output_rows, changes)`. Baseline rows must have precisely the four
  official columns and match input order. It accepts smaller datasets for
  validation diagnostics and contract tests.
- `question_fn(row, decision)` returns a nonempty question string, never `none`.
  It is called only for new yes answers. Its row contains only
  `sample_id, protocol, video` and, for VT only, `text_content, text` when present.
  VA never receives supplied transcript fields. Gold labels and source metadata
  are excluded. Decision information contains predictions only.
  The callback must generate a neutral question addressed to the current user.
- `export_submission(output_dir, rows, baseline_csv, vision, second, scores,
  question_fn, provenance)` checks all 734 official IDs, the confirmed baseline
  SHA-256, all output invariants, and then creates the review artifacts below.
  Provenance is a JSON object identifying the frozen model, evidence candidate,
  source cache, and input files; record their actual hashes in the runner.

Every existing baseline row remains identical except for added yes answers.
All baseline emotions and existing yes questions are preserved byte-for-byte
as CSV values. When run against the latest artifact, this preserves the actual
0.92313 predictions rather than replacing them with the older v4 CSV.
Question validity follows the repository's existing `question_is_consistent`
contract; whether a question addresses the current user needs model prompt
control and review, in addition to that structural check.

## Review artifacts and publication safety

The export directory contains `submission.csv`, `submission-changes.json`,
`submission-evidence.json` and `submission-manifest.json`. Changes include before
and after answers and their evidence probabilities; the evidence audit covers
every row. The manifest records baseline/output/artifact hashes, confirmed
baseline name and Public Score, unchanged emotion and old-yes checks, changed IDs, provenance and
`submitted_to_kaggle=false`.

All serialization and acceptance checks run before writing. Existing artifacts
with different bytes are rejected as a group. Identical retries are idempotent;
individual files use exclusive creation. The manifest is written last; parse it
and verify every recorded artifact hash before treating the export as complete.
A process interrupted during writing can
leave partial artifacts; retrying succeeds only with the identical content.
Neither the original CSV nor any differing published candidate is overwritten.
Baseline rows, SHA-256 and confirmed metadata are derived from one byte
snapshot. The source hash is rechecked before publication, so changing to
another confirmed baseline during question generation also cancels the export.

The original fixed rule is still a hypothesis formed after inspecting
validation results against v4; those diagnostic results remain historical and
do not validate applying the rule to the newer submission. A higher local score
does not establish a Kaggle improvement. The confirmed public best is now
0.92313. C1's actual 0.90175 result did not improve it, and C2 was stopped by its
internal gate. No result in this workflow demonstrates a Public Score of 0.95.
