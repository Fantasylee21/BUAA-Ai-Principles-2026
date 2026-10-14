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

## Future SFT Scripts

Place deterministic dataset conversion, SFT launch, checkpoint evaluation, and
result summarization scripts here. Keep experiment-specific settings in
`configs/` instead of hard-coding them into launch scripts.
