"""Shared prompts for baseline and SFT emotion evaluation."""

from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = (
    "You are a multimodal emotion analysis assistant. Judge the overall emotion "
    "from the provided modalities, explicitly represent uncertainty when the "
    "evidence is insufficient or conflicting, and follow the requested JSON schema."
)
TARGET_INSTRUCTION = (
    "If the evidence conflicts or is insufficient for a reliable judgment, set "
    "is_uncertain to yes and ask one neutral clarifying question addressed to the "
    "current user. Otherwise set is_uncertain to no and clarifying_question to "
    "none. Return exactly one JSON object. pred_emotion must be positive, negative, "
    "or neutral; is_uncertain must be yes or no; clarifying_question must be a "
    "single string. Do not add Markdown or explanatory text."
)


def prompt_for(row: dict[str, Any], protocol: str, model_key: str) -> str:
    """Build the exact user prompt used by SFT inference."""
    if protocol == "VT":
        modality = "The input contains the video and supplied transcript text."
        dialogue = str(row.get("text_content", row.get("text", ""))).strip()
        supplied_text = (
            f"\nSupplied text: {json.dumps(dialogue, ensure_ascii=False)}"
        )
    elif protocol == "VA":
        if model_key == "qwen25omni":
            modality = (
                "The input contains video frames and the video's embedded audio. "
                "Supplied transcript text is unavailable."
            )
        else:
            modality = (
                "The input contains visual video frames only. Audio and supplied "
                "transcript text are unavailable."
            )
        supplied_text = ""
    else:
        raise ValueError(f"Unsupported protocol: {protocol!r}")

    return (
        f"{modality}\n"
        "Analyze the person's overall emotion. Treat visible expression, body "
        "language, spoken or supplied text, and vocal evidence as separate signals "
        f"when those signals are available.{supplied_text}\n"
        f"{TARGET_INSTRUCTION}"
    )
