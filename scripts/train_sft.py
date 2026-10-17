#!/usr/bin/env python3
"""Run LLaMAFactory with per-batch Omni audio/video positioning."""

from functools import wraps


def configure_omni_collator() -> None:
    from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq

    original = MultiModalDataCollatorForSeq2Seq.__call__

    @wraps(original)
    def collate(self, features):
        model_type = getattr(getattr(self.model, "config", None), "model_type", None)
        if model_type != "qwen2_5_omni_thinker":
            return original(self, features)

        # VT has no audio. With a global True, upstream RoPE fails to count
        # its video tokens and silently gives them ordinary text positions.
        # Decide before upstream injects masked dummy audio for unused towers.
        audio_flags = [bool(feature.get("audios")) for feature in features]
        if any(audio_flags) and not all(audio_flags):
            raise ValueError("Mixed Omni VT/VA batches require per_device batch size 1.")
        previous = self.processor.use_audio_in_video
        self.processor.use_audio_in_video = previous and any(audio_flags)
        try:
            return original(self, features)
        finally:
            self.processor.use_audio_in_video = previous

    MultiModalDataCollatorForSeq2Seq.__call__ = collate


if __name__ == "__main__":
    import json
    from pathlib import Path

    from llamafactory.hparams.parser import read_args

    args = read_args()
    loss_settings = args.pop("task_balanced_loss", None)
    configure_omni_collator()
    if loss_settings:
        from task_balanced_loss import configure_task_loss

        configure_task_loss(loss_settings)
        output = Path(args["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "loss_config.json").write_text(json.dumps(loss_settings, indent=2) + "\n")
        print("Task-balanced SFT loss:", loss_settings, flush=True)
    from llamafactory.train.tuner import run_exp

    run_exp(args=args)
