# Omni 7B SFT v5: targeted conflict sampling

This is a proposed **new training run**, not a measured improvement. It uses the
local Qwen2.5-Omni-7B checkpoint and compares against the local 7B v4 run.
The published v4 report's 3B results are a separate experiment.

## Evidence and change

The local 7B v4 validation metrics at checkpoints 119, 238, and 357 show Scores
of 0.6849, 0.7110, and 0.7180. At checkpoint 357, VA Conflict accuracy is
4/10, VT Conflict accuracy is 48/64, and Aligned calibration is 0.9544.
The source breakdown shows high Conflict accuracy on `accept_a_svt` (43/44),
while the CH-SIMS subset reaches only 7/25. These are small validation groups;
the pattern motivates an experiment, not a guarantee.

v5 changes only the training sampling policy. It keeps the same 7B base,
NF4 QLoRA, task-balanced loss, seed, learning rate, 3 epochs, input budget,
and full 666-row validation set as the local v4 configuration. All original
training rows are retained. LLaMAFactory `num_samples` adds replacement draws
before training, then the dataset is shuffled across epochs.

| Training group | Original | v4 effective | v5 effective |
| --- | ---: | ---: | ---: |
| Aligned | 2,913 | 2,913 | 2,913 |
| `accept_a_svt` Conflict | 232 | 464 | 232 |
| CH-SIMS VT Conflict | 107 | 214 | 321 |
| VA Conflict | 102 | 204 | 306 |
| **Total** | **3,354** | **3,795** | **3,772** |

The aim is to shift training toward the two weaker Conflict groups while
keeping the total volume close to v4. Select the final epoch checkpoint by
the official validation Score. Compare Macro-F1, Conflict accuracy, Aligned
calibration, and VA/VT breakdown with the local v4 run. Keep the v4 checkpoint
if v5 does not improve the intended metric.

## Run

The checkout needs the existing `Dataset`, generated ShareGPT data and audio,
locked `third_party/LlamaFactory`, and `../ckpt/Qwen2.5-Omni-7B`. In an
isolated worktree, these can be linked to the prepared main checkout. From
the repository root with the compatible environment active:

```bash
CONDA_ENV=llama-factory STAGE=prepare bash scripts/run_omni7b_sft_v5.sh
CONDA_ENV=llama-factory STAGE=train GPU_ID=4,5 bash scripts/run_omni7b_sft_v5.sh
CONDA_ENV=llama-factory STAGE=evaluate EVAL_GPU_ID=4 bash scripts/run_omni7b_sft_v5.sh
CONDA_ENV=llama-factory STAGE=submit EVAL_GPU_ID=4 bash scripts/run_omni7b_sft_v5.sh
```

After training has already completed, run `STAGE=finish` to perform validation
selection and submission without touching training or prepared data:

```bash
CONDA_ENV=llama-factory STAGE=finish EVAL_GPU_ID=4 bash scripts/run_omni7b_sft_v5.sh
```

`STAGE=all` runs those stages in order and writes `submit.csv` at the repository
root. It chooses the highest scoring validation checkpoint, runs full test
inference, checks all 734 IDs and output fields, then writes the CSV. Set
`SUBMIT_PATH` to change the destination. Set `GPU_ID` to one GPU index if using
one card. `SOURCE_DIR` can point at a prepared ShareGPT directory in another
checkout. The selected adapter and metrics are saved under
`output/sft/qwen25omni/qlora_sft_v5_targeted_7b/` and
`output/sft/evaluation/`. `STAGE=submit` expects an existing
`best_checkpoint.json` from `STAGE=evaluate`.

The launcher sets Hugging Face, Datasets, and Transformers offline mode because
the base checkpoint and training data are local. PEFT may otherwise probe the
derived `Qwen2.5-Omni-7B/thinker` Hub ID while saving an adapter. If training
already has `train_results.json` and saved epoch adapters, rerunning
`STAGE=all` skips training and continues with validation and submission.
