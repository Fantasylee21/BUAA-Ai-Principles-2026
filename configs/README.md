# Training Configurations

Store immutable LLaMAFactory YAML configurations here. Create a new file for
each experiment instead of editing a configuration that has already produced a
reported result.

- `sft/qwen25vl/`: Qwen2.5-VL SFT configurations.
- `sft/qwen3vl/`: Qwen3-VL SFT configurations.
- `sft/qwen25omni/`: Qwen2.5-Omni SFT configurations.
- `inference/`: adapter inference and evaluation configurations.
- `export/`: adapter merge and model export configurations.

Initial LoRA SFT configurations:

- `sft/qwen25vl/lora_sft_v1.yaml`
- `sft/qwen3vl/lora_sft_v1.yaml`
- `sft/qwen25omni/lora_sft_v1.yaml`

All three freeze the modality encoders and multimodal projector, then apply
rank-16 LoRA to the available language-model linear layers. They use batch size
1 with 16-step gradient accumulation, BF16, FlashAttention 2, gradient
checkpointing, three epochs, and official emotion-only validation after every
epoch.

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
