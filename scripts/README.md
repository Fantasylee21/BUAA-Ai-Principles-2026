# Project Scripts

Scripts in this directory resolve paths from the Git repository root and must
not modify the external LLaMAFactory checkout.

## Reproduce The Three Baselines

`run_baselines.sh` sequentially evaluates the following local checkpoints on
the validation split using their native/default input configurations:

1. Qwen2.5-VL-3B-Instruct
2. Qwen3-VL-4B-Instruct
3. Qwen2.5-Omni-3B

Prerequisites:

- A Python environment containing the required inference dependencies is
  activated.
- `Dataset` points to the released course dataset.
- `models` contains the three checkpoint directories used above.
- `ffmpeg` is available for extracting the audio embedded in VA videos for
  Qwen2.5-Omni.
- The selected GPU has enough free memory. Qwen2.5-Omni reached approximately
  23.09 GiB reserved memory during the recorded run on an RTX 4090.

The recorded baseline results used the following software versions:

- Python 3.13.15
- PyTorch 2.6.0+cu124
- CUDA 12.4
- Transformers 5.8.0

These versions document the reference environment; they are not strict runtime
requirements. Other mutually compatible Python, PyTorch, CUDA, and Transformers
versions may also run correctly, but their outputs and memory use should be
recorded separately.

Run all three baselines on physical GPU 2:

```bash
bash scripts/run_baselines.sh 2
```

The GPU can also be selected through an environment variable:

```bash
GPU_ID=2 bash scripts/run_baselines.sh
```

Optional path and environment overrides:

```bash
CONDA_ENV=your_conda_environment \
DATA_ROOT=/path/to/Dataset \
MODEL_ROOT=/path/to/models \
OUTPUT_DIR=/path/to/output/baseline \
bash scripts/run_baselines.sh 2
```

When `CONDA_ENV` is omitted, the script uses `python` from the currently active
environment. When it is set, the script runs through `conda run -n`.

The script passes `--overwrite`, so files with the following run prefixes are
replaced:

- `qwen25vl_baseline_val_*`
- `qwen3vl_baseline_val_*`
- `qwen25omni_baseline_val_*`

By default, results are written to `output/baseline/`. Each run produces:

- `*_details.jsonl`: per-sample predictions, raw responses, timing, and memory.
- `*_predictions.csv`: competition-format predictions.
- `*_metrics.json`: Macro-F1, conflict accuracy, aligned calibration, and score.
- `*_run_config.json`: model, dataset, input profile, environment, and hashes.

For VT samples, all three models receive video plus supplied text. For VA
samples, Qwen2.5-VL and Qwen3-VL are visual-only because they do not accept an
audio stream; Qwen2.5-Omni receives both video frames and embedded audio.

## Prepare LlamaFactory SFT Data

`prepare_sft_data.py` converts the cleaned official training and validation
splits into LlamaFactory ShareGPT multimodal datasets for the VL and Omni
training paths. Run it from any directory; paths default to this Git
repository.

Prerequisites:

- `Dataset` points to the released course dataset.
- `ffmpeg` is available in `PATH` for extracting audio from VA videos.
- Python 3.10 or newer is recommended. The conversion script itself uses only
  the Python standard library.

Create the cleaned official validation split once before conversion:

```bash
python data/prepare_training_data.py \
  --input Dataset/val.jsonl \
  --expected-split val \
  --ignore-use-in-main \
  --baseline-aligned-validation \
  --output-dir data/cleaned/val
```

Generate the default datasets:

```bash
python scripts/prepare_sft_data.py
```

The default behavior is:

- Read `data/cleaned/all/train_vl_clean.jsonl` and
  `data/cleaned/all/train_omni_clean.jsonl` as training data.
- Read `data/cleaned/val/val_vl_clean.jsonl` and
  `data/cleaned/val/val_omni_clean.jsonl` as validation data. These are derived
  from the released `Dataset/val.jsonl`, not sampled from the training split.
- Preserve all 666 official validation records for both model paths, including
  paired VT/VA records, so validation sample coverage and weighting match the
  baseline evaluation.
- Match the baseline modality policy exactly during validation: VL uses
  video+text for VT and video only for VA; Omni uses video+text with audio
  disabled for VT and video+audio without supplied text for VA.
- Keep one SFT example per cleaned record. For a Conflict record, select one of
  its five generated questions deterministically from `sample_id`. This keeps
  the original Conflict/Aligned ratio instead of weighting every Conflict
  record five times.
- Train targets contain `pred_emotion`, `is_uncertain`, and
  `clarifying_question`. Official validation targets contain only
  `pred_emotion`, because the released validation set has no reference
  clarifying question and validation is restricted to emotion recognition.
- Extract each train and validation VA audio track as 16 kHz mono FLAC for the
  Omni path.
- Write generated JSONL, extracted audio, and generation manifests under
  `data/llamafactory/generated/`. This directory is intentionally ignored by
  Git because it can be reproduced and currently occupies about 252 MiB.
- Rewrite `data/llamafactory/dataset_info.json` with the four registered
  datasets.

Audio extraction is idempotent. After a successful first run, skip the ffmpeg
step while rebuilding metadata or JSONL files:

```bash
python scripts/prepare_sft_data.py --skip-audio-extraction
```

To create an experimental dataset with all five questions as separate targets
for each Conflict record, use:

```bash
python scripts/prepare_sft_data.py --question-policy expand
```

This option increases the relative weight of Conflict records and overwrites
the default generated JSONL files. Rerun without the option to restore the
single-question datasets.

Relevant paths can be overridden explicitly:

```bash
python scripts/prepare_sft_data.py \
  --train-clean-dir data/cleaned/all \
  --val-clean-dir data/cleaned/val \
  --questions-dir data/cleaned \
  --dataset-root Dataset \
  --output-dir data/llamafactory/generated \
  --audio-workers 4
```

Use the generated datasets from LlamaFactory training YAML files as follows:

```yaml
# VL models
dataset_dir: data/llamafactory
media_dir: .
dataset: affect_sft_vl_train
eval_dataset: affect_sft_vl_val
```

```yaml
# Qwen2.5-Omni
dataset_dir: data/llamafactory
media_dir: .
dataset: affect_sft_omni_train
eval_dataset: affect_sft_omni_val
use_audio_in_video: true
```

For Qwen2.5-Omni, `use_audio_in_video: true` matches the baseline's VA audio
handling and jointly positions the separately extracted audio with the video.
VT records have no audio placeholder or audio path, so their embedded video
audio remains unused.

Run LlamaFactory from the repository root so that `media_dir: .` resolves both
`Dataset/videos/...` and the extracted files under
`data/llamafactory/generated/audio/...`. Keep experiment-specific training
hyperparameters in `configs/` rather than adding them to the conversion script.

## Run LoRA SFT

`run_lora_sft.sh` launches one of the three versioned configurations and checks
the LLaMAFactory lock commit before training. Pass the physical GPU index as the
second argument:

```bash
bash scripts/run_lora_sft.sh qwen25vl 2
bash scripts/run_lora_sft.sh qwen3vl 2
bash scripts/run_lora_sft.sh qwen25omni 2
```

Run all three sequentially on the same GPU:

```bash
bash scripts/run_lora_sft.sh all 2
```

The script uses Python from the active compatible environment by default. A
named conda environment can be selected with `CONDA_ENV`. Outputs go to
`output/sft/<model>/lora_sft_v1`, and logs go to `runs/sft/<model>/`.

`RESUME=auto` is the default and resumes the numerically latest
`checkpoint-*` directory when present. Use `RESUME=none` for a new output
directory, or pass an exact checkpoint path. To preserve an existing run,
choose another output name instead of overwriting it:

```bash
RUN_NAME=lora_sft_v2 bash scripts/run_lora_sft.sh qwen25vl 2
```

Generated ShareGPT data must already exist. Set `PREPARE_DATA=1` to rebuild it
before launching training. `OUTPUT_ROOT` can redirect all checkpoints and
adapters to another storage location.

Use `DRY_RUN=1` to run path, lock, dataset, checkpoint, and command checks
without loading a model or starting training.
