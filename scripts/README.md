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

`run_sft_training.sh` accepts the method, an explicit YAML configuration, and
the physical GPU index. It checks the method, model path, output directory,
generated data, and the LLaMAFactory lock commit before training:

```bash
bash scripts/run_sft_training.sh lora configs/sft/qwen25vl/lora_sft_v1.yaml 2
bash scripts/run_sft_training.sh lora configs/sft/qwen3vl/lora_sft_v1.yaml 2
bash scripts/run_sft_training.sh lora configs/sft/qwen25omni/lora_sft_v1.yaml 2
```

The script uses Python from the active compatible environment by default. A
named conda environment can be selected with `CONDA_ENV`. The adapter and
checkpoint directory comes from the YAML `output_dir`; logs go to
`runs/sft/<model>/`.

`RESUME=auto` is the default and resumes the numerically latest
`checkpoint-*` directory when present. Use `RESUME=none` for a new output
directory, or pass an exact checkpoint path. To preserve an existing run,
create a new versioned YAML with a different `output_dir`. A one-off output
override is also available:

```bash
OUTPUT_DIR=output/sft/qwen25vl/lora_sft_debug \
  bash scripts/run_sft_training.sh lora \
    configs/sft/qwen25vl/lora_sft_v1.yaml 2
```

When testing a one-off `OUTPUT_DIR` run, pass the same directory as
`ADAPTER_PATH` to the inference launcher. For reported experiments, prefer a
new YAML whose `output_dir` records the actual adapter location.

Generated ShareGPT data must already exist. Set `PREPARE_DATA=1` to rebuild it
before launching training. `OUTPUT_DIR` can temporarily override the YAML
checkpoint and adapter location.

Use `DRY_RUN=1` to run path, lock, dataset, checkpoint, and command checks
without loading a model or starting training.

## Run QLoRA SFT

Use the same `run_sft_training.sh` entry with method `qlora`. It requires the
selected YAML to contain a supported 4-bit QLoRA configuration and checks that
bitsandbytes imports successfully and CUDA is visible before a real run.

Run one model on a selected physical GPU:

```bash
bash scripts/run_sft_training.sh qlora configs/sft/qwen25vl/qlora_sft_v1.yaml 2
bash scripts/run_sft_training.sh qlora configs/sft/qwen3vl/qlora_sft_v1.yaml 2
bash scripts/run_sft_training.sh qlora configs/sft/qwen25omni/qlora_sft_v1.yaml 2
```

Validate paths, the LLaMAFactory lock, generated datasets, checkpoint resume,
and commands without loading a model:

```bash
DRY_RUN=1 bash scripts/run_sft_training.sh qlora \
  configs/sft/qwen3vl/qlora_sft_v1.yaml 2
```

The same `CONDA_ENV`, `OUTPUT_DIR`, `RESUME`, and `PREPARE_DATA` overrides are
supported. QLoRA requires a CUDA-compatible bitsandbytes installation; the
reference environment uses version 0.50.2.

## Evaluate LoRA And QLoRA SFT Adapters

`evaluate_sft.py` and `run_sft_inference.sh` use the selected training YAML as
the experiment identity. They read `model_name_or_path` and `output_dir` from
that YAML, infer the model family, attach the resulting PEFT adapter, and
evaluate the complete official validation split. QLoRA YAML files load the
base model with bitsandbytes NF4 4-bit by default; LoRA YAML files load it in
BF16. VT samples use video plus supplied text; VA samples use visual frames
only, matching the corresponding baseline input policy.

Evaluate a LoRA adapter:

```bash
bash scripts/run_sft_inference.sh lora \
  configs/sft/qwen25vl/lora_sft_v1.yaml 7
```

Evaluate a QLoRA adapter:

```bash
bash scripts/run_sft_inference.sh qlora \
  configs/sft/qwen3vl/qlora_sft_v1.yaml 7
```

Run a small smoke test before the full evaluation:

```bash
LIMIT=3 OVERWRITE=1 bash scripts/run_sft_inference.sh qlora \
  configs/sft/qwen3vl/qlora_sft_v1.yaml 7
```

The evaluator is resumable by default and writes per-sample details,
competition-format predictions, metrics, and an experiment fingerprint under
`output/sft/evaluation/`. The fingerprint includes the YAML path and SHA-256.
Set `OVERWRITE=1` to restart a run. `LOAD_IN_4BIT=auto` follows the YAML;
explicit `0` or `1` enables a separate loading ablation. Use another
`RUN_NAME` when changing this setting.

The evaluator also supports Qwen2.5-Omni. Its VT samples use video plus supplied
text with embedded audio disabled; VA samples use video plus embedded audio and
withhold supplied text, matching the Omni baseline policy. `ffmpeg` must be
available for decoding VA audio.

## Merge An SFT Adapter

`merge_sft_adapter.sh` merges either a LoRA or QLoRA PEFT adapter into the
original non-quantized base model. Pass the training YAML, an optional final
adapter or `checkpoint-*` directory, and the physical GPU index:

```bash
bash scripts/merge_sft_adapter.sh \
  configs/sft/qwen25vl/qlora_sft_v1.yaml \
  output/sft/qwen25vl/qlora_sft_v1/checkpoint-264 \
  2
```

When the adapter path is omitted, the script uses `output_dir` from the YAML.
The merged model is written to `<adapter_path>/merged_model/`. The output must
be a new or empty child directory so that full-model files cannot overwrite the
source adapter.

The generated LLaMAFactory export configuration deliberately omits
`quantization_bit`, `quantization_method`, and other QLoRA loading options.
QLoRA therefore reloads the original BF16/FP16 base checkpoint before merging,
which is required for a valid full model.

Validate paths, the adapter/base-model match, LLaMAFactory lock, and generated
export configuration without loading weights:

```bash
CONDA_ENV=your_environment DRY_RUN=1 \
  bash scripts/merge_sft_adapter.sh \
    configs/sft/qwen3vl/qlora_sft_v1.yaml \
    output/sft/qwen3vl/qlora_sft_v1/checkpoint-396 \
    2
```

Merging uses CPU by default. To let Transformers place the model on the
selected GPU, use `EXPORT_DEVICE=auto`. `EXPORT_SIZE` controls the maximum
weight shard size in GB, and `MERGED_SUBDIR` changes the child directory name:

```bash
EXPORT_DEVICE=auto EXPORT_SIZE=4 MERGED_SUBDIR=merged_bf16 \
  bash scripts/merge_sft_adapter.sh CONFIG.yaml ADAPTER_PATH 2
```

After export, the script verifies `config.json` and model weight shards, then
copies the source training YAML and generated export YAML into the merged model
directory for reproducibility.
