# Grounded uncertainty REBEL v1

This version replaces six-JSON-action EPG for the next experiment. Existing
GRPO and finite-action outputs are preserved. A Public score of 0.92 is the
external acceptance target, not an asserted result.

## Fixed outputs and objective

The frozen v4 generates validation emotions and supplies the uncertainty prefix.
Test emotions and IDs come **only** from the confirmed 0.90813 CSV, SHA256
`2446fcec5e105651157cd376d9a80ac5792489c41d8c2f2b70d710a93c4bdf23`.
The run stores a read-only copy. All 734 IDs and emotions are checked in order.
Questions for new `yes` decisions come from frozen v4; `no` has question `none`.

Full-input uncertainty compares native next-token logits for `yes`/`no` after
the frozen emotion. Auxiliary silent-video, text-only, and audio-only tasks
compare three native emotion-value logits. Labels, source names, and sample
IDs never enter prompts. Unclear view labels skip that auxiliary loss only.

The pair objective is
`((s[a]-s[b]) - (old[a]-old[b]) - 2*(reward[a]-reward[b]))**2`.
Correctness rewards are one/zero; three-way tasks average gold-vs-wrong pairs.
Logit differences equal log-probability differences for these shared contexts.
The loss is `0.6 mean(C) + 0.4 mean(A) + 0.1 mean(views) + 0.02 mean(KL_U)`.
This is a task-specific adaptation of relative reward regression from
[REBEL](https://github.com/ZhaolinGao/REBEL), with modality-label auxiliary
training inspired by [OPPO](https://github.com/ZhiyuanHan-Aaron/OPPO).
It does not claim to reproduce either repository's complete algorithm.

## Data and updates

Each update has 16 main samples: natural Conflict VT/VA three each, synthetic
Conflict VT/VA one each, and Aligned VT/VA four each. Natural conflicts use
one uniform and two missed cases; aligned samples use two uniform and two
false positives. Empty hard pools fall back to the same training stratum.
One of the 30 opposite-label VT/VA groups occupies matching uniform slots.
Every draw is logged separately, including duplicates. This deliberately
changes the sampling distribution; it is not an unbiased uniform objective.

AdamW, LR 5e-6, no decay/dropout, clip one, warmup ten, cosine 200, seed 42.
Only language-policy LoRA is trainable. Frozen `old` is copied every 40 updates;
original-v4 `reference` stays frozen. Every 40 updates a separate GPU evaluates
all 666 validation samples and refreshes hard pools using **only 3,354 training
samples**. Checkpoints also save every 20 steps for recovery. They include both
policy and old adapters, optimizer, scheduler, RNG, pool, and block index.
Legacy optimizer states cannot load into this algorithm.

## Bounded state machine

1. Check all data/model/code identities and protect the baseline CSV.
2. Verify raw/cached values and gradients on real VT and VA samples in all
   applicable input modes. Use verified legacy **features**, never old scores.
3. Build disjoint cache shards on two GPUs, then validate every checksum.
   A repaired run can instead import a completed cache after checking the original
   source identity, unchanged media/value-scoring code, models, data, settings,
   software, and every tensor checksum. Hard links avoid recomputing or copying
   features. The new run records the producer identity and checks all 666 frozen
   reference decisions under the repaired runtime before training.
4. Require all 666 zero-update uncertainty decisions to equal free generation.
5. Run a disposable 20-update check on 32 training examples: hard-C and hard-A
   median correct log-odds gains each >=0.25, main regression loss drop >=10%,
   frozen reference, and exact next-update recovery. Discard its weights.
   CUDA uses deterministic algorithms, cuBLAS workspace `:4096:8`, deterministic
   cuDNN, no cuDNN benchmarking and no TF32. Recovery checks keep exact equality;
   `recovery-check.json` records parameter/gradient/optimizer/task/loss differences
   for both paths, and `smoke-updates.jsonl` retains the per-task short-run trace.
6. Train seed 42 from original v4, at most 200 updates. At step 80 and later,
   stop if no evaluated candidate has gain >=0.003, natural >=8/25, FP <=28.
7. Primary gate: gain >=0.015, natural >=10/25, VA >=4/10, FP <=28/592,
   zero emotion changes, valid complete outputs.
8. Only after the primary gate, and sufficient remaining time, independently
   train seed 43 from original v4. Replication gate: gain >=0.005, natural
   >=8/25, FP <=28/592, same output constraints.
9. Only with both gates passed, select the best primary-qualified checkpoint
   and generate a single 734-row CSV plus per-row changes and manifests.

All score gains are against the same-environment frozen free-generation
baseline. Reports retain per-row predictions, corrections/regressions, Macro-F1,
natural/VA counts, false positives, Wilson intervals, and paired video-group
bootstrap intervals. These intervals are descriptive, not selection correction.

No new training after hour 20; terminate the pipeline at hour 24. Resume retains
the original clock, fixed settings, code/data identity, old snapshot and pools.
Only this controller's own child processes can be terminated. There is no
automatic Kaggle upload. Each GPU has a **32 GiB PyTorch allocator ceiling**;
this is not a hardware reservation and does not cap unrelated CUDA overhead.

## Start and observe

On the server, from the immutable source release:

```bash
bash scripts/start_grounded_tmux.sh /absolute/new/run 1 2 buaa-grounded
```

Use `tmux attach -t buaa-grounded` outside tmux, or
`tmux switch-client -t buaa-grounded` inside an old session. Window 0 shows the
dashboard, prioritizing loss/reward/KL, current learning rate, training progress,
best and latest valid local scores, baseline comparisons, conflict/false-positive
counts, cache progress and GPU telemetry. Missing results stay explicitly pending;
historical Kaggle Public and local validation scores are displayed separately.
Completed/waiting stages retain their latest persisted training metrics. Windows 1-7
show pipeline, training, validation, numerical checks, two cache shards, and
learning checks. Window 8 displays the actual run parameters, sampling quotas,
input budgets, and promotion/stopping gates. `Ctrl+B`, then `D` detaches safely.
Completion or a rejected gate leaves its reason and logs visible.

Restart the same command with the **same source release and run directory**
only after the prior controller has exited. The script adds `--resume` when
context exists and refuses to start a duplicate tmux session. Completed
checkpoints are immutable and readers require their completion marker.

The startup assumes the original data/models, offline environment, legacy run
`runs/finite-v1-20261022-r3`, and baseline CSV under `inputs/baseline_090813.csv`
are already present below `BUAA_RL_ROOT` (default server project root).

To reuse a compatible completed grounded cache on a newly authorized attempt,
set `BUAA_GROUNDED_REUSE_RUN` and `BUAA_GROUNDED_REUSE_RELEASE` to its absolute run
and source-release directories before starting. The new attempt keeps separate
checkpoints, optimizer state and a bounded clock. It starts policy training from
original v4. Old smoke weights and old optimizer state are used only for isolated
recovery diagnosis, never as the formal training initialization.
