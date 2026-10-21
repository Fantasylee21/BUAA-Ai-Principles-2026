# Omni S2 RL: fixed experiments, replaceable compute

Continue the S1 **Omni v4 / checkpoint-714 LoRA adapter**. The base is
`Qwen/Qwen2.5-Omni-3B`; use the team HF repository's original adapter revision
`3aadc63707148c564bd27409a0ee80cf31988a47`. Current `main` contains the merged
model at `6a1e7fab0339e8b130261d7384b47f1a476885a1`; do not apply the SFT adapter
to that already-merged model. The supplied local adapter is byte-identical to
the original HF adapter; see [verification](../reports/rl/v4_artifact_verification.md).
Download/locate the original course `train.jsonl`, `val.jsonl`,
`test_release.jsonl` and videos separately. Do not train again from the base.

## What this implementation does

- `correctness_v1` is the default verifiable reward. `f1_delta_v1` is an
  optional controlled experiment, not an already-demonstrated improvement.
- Every update uses 4 Conflict + 4 Aligned prompts and 4 sampled completions
  per prompt. Sampling is uniform within type and deterministic by seed/step.
  No per-neutral-example replication is used. Exact legacy duplicate rows are
  removed and counted; conflicting duplicate IDs fail.
- Reward: invalid schema **-2**; otherwise `0.5 * p(type)/0.5 * emotion_reward`
  plus `0.3/0.5` for a correct Conflict decision or `0.2/0.5` for a correct
  Aligned decision; inconsistent question usage costs `0.02`. Correct JSON
  alone receives no bonus. This accuracy reward is a proxy for Macro-F1.
- The F1 variant replaces only `emotion_reward` with
  `clip(N * (F1(replace_one_training_prediction) - F1(frozen_training_matrix)), -1, 1)`.
  It requires a cache whose training IDs, data hash, base/SFT weights, input
  settings and source hash match. Validation/test labels never supply rewards.
- One NF4 base is shared by a trainable policy adapter and a frozen SFT
  reference. Both adapters' loaded tensors are compared against their artifacts.
  Visual/audio modules and projectors remain frozen; LoRA dropout is disabled.
- **Omni-specific streamed training loop, using TRL `GRPOConfig`**: this is not
  a claim that the generic upstream `GRPOTrainer` handles Omni video/audio.
  Every rollout and both probability passes carry the same video/audio fields.
  The loss implements one-iteration on-policy Dr.GRPO: advantages are only
  mean-centered; no per-group std scaling or length-dependent denominator.
  Sampling is exactly temperature 1, top-p 1, top-k 0. No rollout reuse is allowed.
  Model generation defaults are isolated so inherited repetition penalties or
  token suppression cannot change that sampling distribution. Omni versions
  without `logits_to_keep` use a scoped output-head suffix projection, preserving
  full input context and suffix gradients without allocating full-prompt logits.

The design follows verifiable outcome rewards in
[Open-R1](https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py),
the normalization analysis in [Dr.GRPO](https://arxiv.org/abs/2503.20783), and
the configuration contract in [TRL](https://huggingface.co/docs/trl/grpo_trainer).
The custom F1 approximation is our experimental adaptation, not a result from
these references.

## Setup and fixed experimental settings

Use a dedicated Linux environment. A candidate package set is in
`rl/requirements.txt`; it is **not a GPU-validated lockfile yet**. For a
4090/5090-capable build, install the CUDA 12.8 torch wheel first:

```bash
python -m venv .venv-rl
source .venv-rl/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r rl/requirements.txt
command -v ffmpeg
command -v ffprobe
command -v tmux
```

After the CUDA smoke test, retain the actual package versions from the run
manifest and `python -m pip freeze` with the experiment artifacts. Keep the
software set identical between control and candidate runs.

Defaults: seed 42, 200 optimizer updates, LR 5e-6, 10 scheduler warmup steps
(not a format-reward curriculum), cosine LR decay, KL 0.02, norm clip 1,
128 output tokens, 6144 prompt-token guard. Input follows the SFT evaluation
prompt: 2 FPS, 4..64 frames, 50,176 pixels per frame, full 16 kHz VA audio,
VT transcript and no audio; VA transcript is withheld. Prompts that exceed
the limit fail instead of dropping modalities.

GPU model/index, free-memory checks, execution microbatch (1/2/4 completions)
and wall-clock limit are runtime settings. They do not change G, effective
batch, input budgets or precision. `auto` respects `CUDA_VISIBLE_DEVICES` and
selects an idle card with the most free memory. An explicit GPU must already
be allocated to you. No other process is stopped. All modes are single-GPU;
independent experiments can be assigned separate allocated cards.

## Commands

The following paths are examples; provide the actual accessible paths after
server login. Run from the repository root. Passwords are entered only in
your interactive SSH terminal (`ssh YOUR_USER@YOUR_HOST`), not in scripts.

```bash
export DATA_ROOT=/actual/course/Dataset
export BASE_MODEL=/actual/models/Qwen2.5-Omni-3B
export SFT_ADAPTER=/actual/models/qwen25omni-sft
```

Inspect data/config without importing torch or loading weights:

```bash
python -m rl.train_grpo --dry-run --data-root "$DATA_ROOT" \
  --model-path "$BASE_MODEL" --sft-adapter-path "$SFT_ADAPTER" \
  --run-dir runs/config-check
```

Run the real-media smoke test in tmux. It probes the longest training video
of each protocol, checks initial policy/reference equality, tests whether
zeroing video/audio changes token probabilities, then performs 10 updates,
saves a checkpoint, and rejects a run with no policy change:

```bash
bash scripts/run_rl.sh --mode smoke --data-root "$DATA_ROOT" \
  --model-path "$BASE_MODEL" --sft-adapter-path "$SFT_ADAPTER" \
  --run-dir runs/omni-smoke --gpu auto
```

Smoke weights must not initialize a formal experiment. Validate the smoke
checkpoint with `--resume-from-checkpoint runs/omni-smoke/checkpoint-000010`
and the same run-dir/mode; hash checks and reload checks happen before use.

Train the default reward from a fresh SFT copy. The runner evaluates the
frozen SFT reference on all 666 original validation rows first, saves every
20 steps and evaluates at steps 100 and 200:

```bash
bash scripts/run_rl.sh --data-root "$DATA_ROOT" \
  --model-path "$BASE_MODEL" --sft-adapter-path "$SFT_ADAPTER" \
  --run-dir runs/omni-correctness-42 --gpu auto
```

Resume an interrupted experiment (a different GPU is permitted):

```bash
bash scripts/run_rl.sh --data-root "$DATA_ROOT" \
  --model-path "$BASE_MODEL" --sft-adapter-path "$SFT_ADAPTER" \
  --run-dir runs/omni-correctness-42 --gpu 1 --micro-batch-size 1 \
  --resume-from-checkpoint runs/omni-correctness-42/checkpoint-000100
```

The optional F1 experiment is a separate command and starts again from SFT:

```bash
bash scripts/run_rl.sh --mode cache --data-root "$DATA_ROOT" \
  --model-path "$BASE_MODEL" --sft-adapter-path "$SFT_ADAPTER" \
  --run-dir runs/omni-train-cache --gpu auto
bash scripts/run_rl.sh --config configs/rl/f1_delta_v1.json --data-root "$DATA_ROOT" \
  --model-path "$BASE_MODEL" --sft-adapter-path "$SFT_ADAPTER" \
  --f1-cache runs/omni-train-cache/f1_cache.json --run-dir runs/omni-f1-42 --gpu auto
```

Do not start the second command until the cache job reports `complete`.
Compare candidates at common completed update counts. There is no promised
eight-hour runtime: timings and peak allocated/reserved memory are measured
per update. `--max-runtime-hours` is optional and saves at an update boundary.

## Recovery, evaluation and submission

Each checkpoint is written to a new partial directory and published only
after adapter, processor, optimizer, scheduler, Python/NumPy/Torch/CUDA RNG
states and file hashes are complete. Config/data/model/source/software
identity is checked before restoring. The step-indexed sampler resumes at
the correct next batch. Cross-GPU floating-point results are not promised
bit-for-bit identical. Interruptions during an update recover from the last
completed checkpoint, not from partial gradients.

Logs: `RUN_DIR.console.log`, `RUN_DIR/training.jsonl`, `rollouts.jsonl`,
`status.json`, per-launch manifests and per-checkpoint validation files.
`tmux attach -t SESSION` shows the running job. SSH disconnects do not stop
tmux; server restarts require explicit checkpoint resume.

Model selection uses the historical **fixed-three-class** Macro-F1. The
union-label score is separately named as a diagnostic because the official
missing-class policy is not independently verified. Empty-type subgroups
receive a null composite score rather than renormalized weights. Natural
CH-SIMS conflict recall and VT/VA/source breakdowns are reported separately.

An eligible checkpoint must gain at least 0.005 score over its same-input SFT
baseline; lose no more than 0.005 Macro-F1 or 0.01 Aligned accuracy; not regress
natural-conflict recall; and have zero schema/question violations. Best score
wins among eligible checkpoints (earlier checkpoint on ties). Otherwise SFT
remains selected. `best_checkpoint.json` records the actual adapter path.

Generate the candidate test file with that selected adapter:

```bash
bash scripts/run_rl.sh --mode evaluate --split test --data-root "$DATA_ROOT" \
  --model-path "$BASE_MODEL" --sft-adapter-path "$SFT_ADAPTER" \
  --policy-path runs/omni-correctness-42/checkpoint-000200/adapter/policy \
  --run-dir runs/omni-selected-test --gpu auto
```

Replace the example checkpoint with `best_checkpoint.json`'s adapter. Test
input must have exactly 734 unique IDs. Submission construction uses strict
validation; it fails rather than inventing labels or questions. No Kaggle
upload is performed. Checkpoints, data and logs belong under ignored runs/
or explicit external paths; don't add them to Git.

## Validation status

Offline tests cover reward ordering, expected-weight identities, bounded F1
changes, scoring convention, leakage guards, reproducible sampling, strict
CSV export, checkpoint integrity, and (with CPU torch) actual autograd and
media batching. A tiny real Transformers/PEFT CPU model also verifies adapter
weights, optimizer/RNG restoration and continued updates, generation defaults,
and output-head suffix probabilities/gradients. These are not a successful
real-Omni CUDA training run. Real weight loading, longest VA/VT forward/backward,
peak GPU memory and a real save/reload must pass the smoke test before any formal
training result is claimed.

```bash
python -m unittest rl.test_reward rl.test_experiment rl.test_omni_runtime rl.test_cli rl.test_checkpoint_runtime
python -m pytest -q rl/test_make_submission.py tests/test_checkpoint_selection.py
```
