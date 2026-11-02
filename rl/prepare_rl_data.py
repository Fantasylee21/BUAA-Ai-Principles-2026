"""Prepare RL prompts for the GRPO stage.

Reads ``data/train.jsonl`` and writes a JSONL file (default
``rl/data/rl_prompts.jsonl``) with one row per RL prompt:

    {"sample_id", "protocol", "sample_type", "gt_label", "video",
     "text_content", "prompt": [{"role": "user", "content": [
        {"type": "video", "video": <path>},
        {"type": "text",  "text":  <instruction (+ transcript)>}]}]}

Class imbalance handling (facts about this dataset):
  * only 411 / 3354 (12.3%) train samples are Conflict -> --conflict-repeat
  * only 7 train samples are labeled "neutral"         -> --neutral-repeat
Repeated rows are exact copies; GRPO samples fresh generations per row, so
copies act as an oversampling weight. A deterministic shuffle (--seed) is
applied at the end.

Protocol note: VT = video + text (audio off), VA = video + audio (no text
provided). For VA rows the transcript is NOT included in the prompt by
default (--va-text-mode drop). Comment: VL models such as Qwen2.5-Omni
treat VA as VT — the video file carries its own audio track, so "drop"
simply withholds the transcript text while the model still sees/hears the
video; "keep" additionally appends the transcript (ablation only).
"""

import argparse
import json
import os
import random
from collections import Counter

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

OUTPUT_CONTRACT = (
    '{"pred_emotion": "positive"|"negative", '
    '"is_uncertain": "yes"|"no", '
    '"clarifying_question": "<string or \'none\'>"}'
)

# Base instruction enforcing the compact JSON output contract.
INSTRUCTION = (
    "You are an emotion recognition assistant. Watch the video and judge the "
    "speaker's true overall emotion.\n"
    "Answer with exactly one compact JSON object and nothing else (no "
    "reasoning, no extra text):\n"
    f"{OUTPUT_CONTRACT}\n"
    "Rules:\n"
    '- "pred_emotion" is "positive" or "negative".\n'
    '- Set "is_uncertain" to "yes" when the modalities seem to conflict or '
    "the evidence is ambiguous; otherwise set it to \"no\".\n"
    '- Provide a short "clarifying_question" ONLY when "is_uncertain" is '
    '"yes"; otherwise it must be exactly "none".\n'
    "Keep the answer short."
)

# Appended for VA samples in "drop" mode: no transcript is shown.
VA_AUDIO_HINT = "\nUse the audio from the video to judge the emotion."

# Template for the transcript block (VT samples, or VA with --va-text-mode keep).
TRANSCRIPT_TEMPLATE = "\nTranscript: {text}"


def build_prompt_text(protocol, text_content, va_text_mode):
    """Build the user-turn instruction text for one sample."""
    text = INSTRUCTION
    if protocol == "VA" and va_text_mode == "drop":
        # Project rule: VA = video + audio only, no transcript in the prompt.
        text += VA_AUDIO_HINT
    else:
        # VT samples always get the transcript; VA + "keep" is an ablation.
        text += TRANSCRIPT_TEMPLATE.format(text=text_content or "")
    return text


def build_row(record, va_text_mode):
    """Convert one raw train.jsonl record into an RL prompt row."""
    prompt = [
        {
            "role": "user",
            "content": [
                # Relative video path, as stored in the dataset
                # (e.g. "videos/video_0001/0001.mp4").
                {"type": "video", "video": record["video"]},
                {
                    "type": "text",
                    "text": build_prompt_text(
                        record["protocol"], record.get("text_content"), va_text_mode
                    ),
                },
            ],
        }
    ]
    return {
        "sample_id": record["sample_id"],
        "protocol": record["protocol"],
        "sample_type": record["sample_type"],
        "gt_label": record["views"]["M12"]["label"],
        "video": record["video"],
        "text_content": record.get("text_content", ""),
        "prompt": prompt,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare RL prompt JSONL for GRPO training."
    )
    parser.add_argument(
        "--input",
        default="data/train.jsonl",
        help="Path to the source JSONL (default: data/train.jsonl).",
    )
    parser.add_argument(
        "--output",
        default="rl/data/rl_prompts.jsonl",
        help="Output JSONL path (default: rl/data/rl_prompts.jsonl).",
    )
    parser.add_argument(
        "--va-text-mode",
        choices=["drop", "keep"],
        default="keep",
        help="For protocol==VA: keep the transcript in the prompt "
        "(default 'keep' prevents model deafness in RL where audio tracks "
        "are not loaded by TRL) or 'drop' (video-only ablation).",
    )
    parser.add_argument(
        "--conflict-repeat",
        type=int,
        default=4,
        help="Repeat factor for Conflict rows (default: 4).",
    )
    parser.add_argument(
        "--neutral-repeat",
        type=int,
        default=0,
        help="Repeat factor for rows whose gt_label is 'neutral' (default: 0 to drop neutrals).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the final deterministic shuffle (default: 42).",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    rows = []
    for record in records:
        if record["views"]["M12"]["label"] == "neutral" and args.neutral_repeat <= 0:
            continue  # Target competition test set only has positive and negative
        row = build_row(record, args.va_text_mode)
        repeat = 1
        if record["sample_type"] == "Conflict":
            repeat = max(repeat, args.conflict_repeat)
        if record["views"]["M12"]["label"] == "neutral":
            repeat = max(repeat, args.neutral_repeat)
        rows.extend([row] * repeat)

    random.Random(args.seed).shuffle(rows)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- summary --------------------------------------------------------
    src_types = Counter(r["sample_type"] for r in records)
    src_labels = Counter(r["views"]["M12"]["label"] for r in records)
    out_types = Counter(r["sample_type"] for r in rows)
    out_labels = Counter(r["gt_label"] for r in rows)
    out_protocols = Counter(r["protocol"] for r in rows)

    print(f"Input : {args.input} ({len(records)} samples)")
    print(f"  source sample_type: {dict(src_types)}")
    print(f"  source labels     : {dict(src_labels)}")
    print(f"Output: {args.output} ({len(rows)} prompts, seed={args.seed})")
    print(f"  sample_type: {dict(out_types)}")
    print(f"  gt_label   : {dict(out_labels)}")
    print(f"  protocol   : {dict(out_protocols)}")
    print(
        f"  oversampling: conflict x{args.conflict_repeat}, "
        f"neutral x{args.neutral_repeat}, va-text-mode={args.va_text_mode}"
    )


if __name__ == "__main__":
    main()
