# Omni conflict cost v1

This version is cost-sensitive supervised correction with a real modality-pair
constraint, not pure RL. It replaces fixed relative-reward increments for this
experiment while preserving previous runs and algorithms. Kaggle Public >=0.92
is the external goal, not a promised or locally measured result.

## Objective and inputs

Original v4 initializes trainable language LoRA. The reference, frozen emotion
outputs and question generator remain original v4. Actual VT or VA inputs use
an explicit prompt about conflicting emotion between the two available
modalities. IDs, sources, labels and true per-view emotions never enter prompts.
Native next-token logits give `d = logit(yes)-logit(no)`.

The main objective is `0.6 mean_C softplus(-d) + 0.4 mean_A softplus(d)`.
Uniform complete-data traversal uses fixed full-training class proportions to
weight individual losses. Unlike a fixed relative increment, confidently wrong
decisions retain a corrective gradient. Each update also includes one real
same-video opposite-label VT/VA pair with `softplus(1-(d_C-d_A))`.
Total loss is `main + 0.1 pair + 0.02 mean(KL(policy || original_v4))`.
The pair does not replace either sample's own supervised target. No auxiliary
single-modality backpropagation or old-policy snapshot is used.

## Data, checks and bounded training

Traverse all 3,354 training records without replacement per deterministic
shuffled epoch. Main batch size is 16 (short final batch retained), 210 updates
per epoch, at most two epochs. Cycle the 30 real pairs in shuffled order and log
their repetitions separately from main-data coverage. Only training data can
construct sampling or learning-check cohorts.

AdamW: LR 5e-6, no weight decay or dropout, clip 1, warmup 10, cosine horizon
420. Validate at updates 0, 105, 210, 315 and 420. New prompt/reference scoring
is cached separately; old compatible perception features may be reused only
after identity checks. Old optimizer state is not accepted. Exact recovery
retains data/pair order, optimizer, scheduler, RNG and the original run clock.

Before formal training, verify cached/raw values and gradients, modality
isolation, frozen reference and deterministic restoration. Measure both
zero-update baselines: original free generation and new-prompt field decisions.
The higher local score is the common selection baseline; new decisions need
not equal old-prompt decisions. A disposable 32-example/40-update learning check
has eight missed conflicts, eight false positives, eight easy conflicts and
eight easy aligned cases, balanced by VT/VA. Each hard group must correct at
least four; each easy group may introduce at most one error. Report changes on
the remaining training data as training diagnostics, not independent validation.
Discard temporary weights before the formal seed-42 run.

At the end of epoch one, wait for complete validation. Stop if no legal candidate
with unchanged emotions gains >=0.003 above the higher zero-update baseline.
Seed 42 must gain >=0.015 before seed 43 independently restarts from original
v4 for exactly the selected candidate's update count. Replication requires
>=0.005. Natural/VA conflicts and false positives remain reported; there is no
false-positive-28 or subgroup-count veto. Reports include paired changes and
descriptive video-group bootstrap intervals; they do not correct for repeated
model selection.
Score deltas use the higher baseline. The dashboard's corrected/lost-conflict
and added/removed-false-positive counts are explicitly relative to original
free generation; result files also retain the new-prompt comparison.

Wall-clock limits have been removed at the user's request. Both `budget_hours`
and `training_cutoff_hours` are null; no 24-hour shutdown or 20-hour phase cutoff
applies. The update ceilings, learning checks and score gates still stop the
experiment. Two shared GPUs each have a 32 GiB **PyTorch allocator** ceiling,
which is not a hardware reservation or total CUDA-process-memory guarantee.
Only this controller's own child processes are managed.

## Output and observation

Only both seeds passing can generate one test CSV using the selected seed-42
model. Its 734 ordered IDs and emotion values must equal the original confirmed
0.90813 CSV (SHA256
`2446fcec5e105651157cd376d9a80ac5792489c41d8c2f2b70d710a93c4bdf23`).
`no` uses question `none`; frozen v4 supplies required new `yes` questions.
There is no automatic Kaggle submission.
When a qualified submission is ready, the selected model also gets a
training-only single-modality diagnostic report.
This adds no optimizer updates or selection criteria.

From the immutable server release:

```bash
bash scripts/start_conflict_tmux.sh /absolute/new/run 1 2 buaa-conflict
```

Default root is `/home/nas/xingzhenbang/buaa-ai2026`, overridable with
`BUAA_RL_ROOT`. Source artifacts default to `grounded-v1-20261023-r3` under
`runs/` and `releases/`; override with `BUAA_CONFLICT_SOURCE_RUN` and
`BUAA_CONFLICT_SOURCE_RELEASE`. Those artifacts supply provenance/compatible
inputs, not formal policy or optimizer initialization. The same source release
and run directory can resume after its prior controller exits; duplicate tmux
sessions are refused. Elapsed runtime remains visible without a deadline.

For a repaired release, `BUAA_CONFLICT_REUSE_RUN` and
`BUAA_CONFLICT_REUSE_RELEASE` may identify a failed cost-training cache producer.
The importer verifies its source, model, media, settings, actual prompt/scoring
code, per-row membership, probabilities and every blob checksum. Only completed
journal rows are hard-linked into the new run. It preserves old failure records
and never imports old optimizer or temporary training weights.

New clarification questions still come from frozen v4 on the original prompt.
Only the question value is extracted; the serializer controls the final JSON
and escapes literal quotation marks. A deterministic opening-quote retry handles
malformed value boundaries. Missing questions still fail explicitly; no fixed
question is fabricated and decision fields cannot be rewritten by generation.

Outside tmux use `tmux attach -t buaa-conflict`; inside an older session use
`tmux switch-client -t buaa-conflict`. Window 0 shows loss, main/pair losses, KL,
LR, real coverage, best/latest local scores, correction/false-positive changes,
phases and the stopping reason. Historical Public 0.90813 remains distinct.
Windows 1/2/3/4 are pipeline/training/validation/numerical logs; 5/6 cache,
7 learning check, 8 actual parameters. Mouse clicking is enabled. Press
`Ctrl+B`, release, then `D` to detach while the server continues.
