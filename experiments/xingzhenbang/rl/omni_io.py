"""Shared SFT/RL Omni inputs; gold labels never enter the rendered prompt."""
from __future__ import annotations

import subprocess
from pathlib import Path

from sft_prompt import SYSTEM_PROMPT, prompt_for


def messages_for(row, video_path):
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "video", "video": str(video_path)},
            {"type": "text", "text": prompt_for(row, row["protocol"], "qwen25omni")}]},
    ]


def audio_from_video(path, sampling_rate=16000):
    import numpy as np
    result = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
        "-ar", str(sampling_rate), "-f", "f32le", "pipe:1",
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if not audio.size:
        raise ValueError(f"No audio decoded from {path}")
    return audio


def prepare_omni_inputs(processor, row, data_root, settings, device="cuda:0"):
    import torch
    root = Path(data_root).resolve()
    path = (root / row["video"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing video or video outside data root: {path}")
    use_audio = row["protocol"] == "VA"
    messages = messages_for(row, path)
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    pixels = settings["video_pixels"]
    inputs = processor(
        text=rendered, videos=str(path),
        audio=audio_from_video(path, settings["audio_sampling_rate"]) if use_audio else None,
        padding=True, return_tensors="pt",
        videos_kwargs={"use_audio_in_video": use_audio, "do_sample_frames": True,
                       "fps": settings["video_fps"], "min_frames": settings["video_min_frames"],
                       "max_frames": settings["video_max_frames"],
                       "size": {"shortest_edge": pixels, "longest_edge": pixels}},
    )
    inputs = {k: v for k, v in inputs.items() if v is not None}
    inputs["use_audio_in_video"] = use_audio
    if inputs["input_ids"].shape[-1] > settings["max_prompt_tokens"]:
        raise ValueError(f"Prompt too long for {row['sample_id']}; media is never silently truncated")
    if "pixel_values_videos" not in inputs or "video_grid_thw" not in inputs:
        raise RuntimeError("Omni processor did not return video tensors")
    audio_keys = ("input_features", "feature_attention_mask")
    if use_audio and any(k not in inputs for k in audio_keys):
        raise RuntimeError("VA processor did not return audio tensors and mask")
    if not use_audio and any(k in inputs for k in audio_keys):
        raise RuntimeError("VT must not contain audio")
    return {k: v.to(device=device, dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point()
            else v.to(device=device) if torch.is_tensor(v) else v for k, v in inputs.items()}


def task_messages(row, video_path, mode):
    """Auxiliary prompts contain input evidence only, never annotation metadata."""
    if mode == "full":
        return messages_for(row, video_path)
    if mode not in ("vision", "text", "audio") or (mode == "text" and row["protocol"] != "VT") or (mode == "audio" and row["protocol"] != "VA"):
        raise ValueError("Unsupported isolated modality for this protocol")
    content = []
    if mode == "vision":
        content.append({"type": "video", "video": str(video_path)})
        instruction = "Judge only the emotion visible in this silent video. No audio or transcript is provided."
    elif mode == "audio":
        content.append({"type": "audio", "audio": str(video_path)})
        instruction = "Judge only the emotion conveyed by this audio. No video or transcript is provided."
    else:
        import json
        instruction = "Judge only the emotion conveyed by this text: " + json.dumps(row["text_content"], ensure_ascii=False)
    content.append({"type": "text", "text": instruction +
        ' Return exactly one compact JSON object with pred_emotion equal to positive, negative, or neutral.'})
    return [{"role": "system", "content": [{"type": "text", "text": "You are an emotion recognition assistant."}]},
            {"role": "user", "content": content}]


def prepare_task_inputs(processor, row, data_root, settings, mode, device="cuda:0"):
    if mode == "full":
        return prepare_omni_inputs(processor, row, data_root, settings, device)
    import torch
    root = Path(data_root).resolve()
    path = (root / row["video"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Missing or escaped task media")
    messages = task_messages(row, path, mode)
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    kwargs = {"text": rendered, "padding": True, "return_tensors": "pt"}
    if mode == "vision":
        pixels = settings["video_pixels"]
        kwargs.update(videos=str(path), videos_kwargs={"use_audio_in_video": False,
            "do_sample_frames": True, "fps": settings["video_fps"],
            "min_frames": settings["video_min_frames"], "max_frames": settings["video_max_frames"],
            "size": {"shortest_edge": pixels, "longest_edge": pixels}})
    elif mode == "audio":
        kwargs.update(audio=audio_from_video(path, settings["audio_sampling_rate"]),
                      audio_kwargs={"sampling_rate": settings["audio_sampling_rate"]})
    inputs = {k: v for k, v in processor(**kwargs).items() if v is not None}
    inputs["use_audio_in_video"] = False
    has_video = "pixel_values_videos" in inputs
    has_audio = "input_features" in inputs
    if has_video != (mode == "vision") or has_audio != (mode == "audio"):
        raise RuntimeError("Isolated modality input contains the wrong media")
    if inputs["input_ids"].shape[1] > settings["max_prompt_tokens"]:
        raise ValueError("Task input exceeds the untruncated token budget")
    return {k: v.to(device=device, dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point()
            else v.to(device=device) if torch.is_tensor(v) else v for k, v in inputs.items()}
