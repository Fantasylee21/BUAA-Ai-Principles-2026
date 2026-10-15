# Training Configurations

Store immutable LLaMAFactory YAML configurations here. Create a new file for
each experiment instead of editing a configuration that has already produced a
reported result.

Training and evaluation launchers accept these YAML paths directly. The
launchers infer the model family from `model_name_or_path`, use `output_dir` as
the adapter/checkpoint location, and reject a LoRA/QLoRA launcher mismatch.
Adding a versioned YAML therefore does not require editing a shell script.

- `sft/qwen25vl/`: Qwen2.5-VL SFT configurations.
- `sft/qwen3vl/`: Qwen3-VL SFT configurations.
- `sft/qwen25omni/`: Qwen2.5-Omni SFT configurations.
- `inference/`: adapter inference and evaluation configurations.
- `export/`: adapter merge and model export configurations.

Initial LoRA SFT configurations:

- `sft/qwen25vl/lora_sft_v1.yaml`
- `sft/qwen3vl/lora_sft_v1.yaml`
- `sft/qwen25omni/lora_sft_v1.yaml`

Initial QLoRA SFT configurations:

- `sft/qwen25vl/qlora_sft_v1.yaml`
- `sft/qwen3vl/qlora_sft_v1.yaml`
- `sft/qwen25omni/qlora_sft_v1.yaml`

All six freeze the modality encoders and multimodal projector, then apply
rank-16 LoRA to the available language-model linear layers. They use batch size
1 with 16-step gradient accumulation, BF16, FlashAttention 2, gradient
checkpointing, three epochs, and official emotion-only validation after every
epoch.

The QLoRA configs keep the same datasets, modality budgets, LoRA parameters,
optimizer settings, effective batch size, and validation strategy as the LoRA
configs. This makes LoRA and QLoRA directly comparable. They load the frozen
base model with bitsandbytes 4-bit NF4 quantization, enable double
quantization, and upcast LayerNorm weights to FP32. The trainable adapters
remain higher precision; `bf16: true` controls mixed-precision computation.

QLoRA requires a CUDA-compatible bitsandbytes installation. The reference
environment currently uses bitsandbytes 0.50.2. LLaMAFactory requires at least
0.39.0 for 4-bit bitsandbytes loading. Do not set `quantization_bit` when
merging the trained adapter into a base model: adapter merging must load the
original non-quantized checkpoint.

Use `scripts/merge_sft_adapter.sh` for both LoRA and QLoRA merging. It derives a
minimal export configuration from the selected training YAML and removes the
training-time quantization settings automatically.

The configs sample at 2 FPS and cap videos at 64 frames. The released train and
validation videos require at most 38 frames at this rate, so the cap does not
drop frames. Per-frame pixels are limited to approximately 64-128 visual tokens
to keep initial single-GPU training within a practical memory range; this is a
recorded SFT training budget rather than the native baseline pixel budget.

Qwen2.5-Omni uses `use_audio_in_video: true`, 16 kHz audio, and the registered
Omni datasets. Its VT validation examples contain no audio input, while VA
validation examples contain video and audio but no supplied transcript.

`load_best_model_at_end` selects checkpoints by emotion-only validation loss.
This is the SFT training signal, not the competition Macro-F1. Run generation
and parse `pred_emotion` after training to compare against the baseline metrics.


本机 v3 为 `sft/<model>/qlora_sft_v3.yaml`。新增的 `task_balanced_loss` 由项目
`train_sft.py` 消费，须通过项目启动器运行；直接传给外部 LLaMAFactory CLI 不支持
该字段。训练后运行 `select_sft_checkpoint.py` 按完整官方验证集 Score 选择 adapter。
