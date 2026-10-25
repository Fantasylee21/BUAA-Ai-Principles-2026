import argparse
import json
import os
import subprocess
import sys
import time
import torch
import numpy as np
from peft import PeftModel
from transformers import (
    BitsAndBytesConfig,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_submission import resolve_test_file
import make_submission
from reward import parse_completion

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


def audio_from_video(video_path, sampling_rate=16000):
    try:
        command = [
            "ffmpeg", "-v", "error", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", str(sampling_rate),
            "-f", "f32le", "pipe:1"
        ]
        completed = subprocess.run(command, check=True, stdout=subprocess.PIPE)
        audio = np.frombuffer(completed.stdout, dtype="<f4").copy()
        if audio.size > 0:
            return audio
    except Exception:
        pass
    try:
        import av
        container = av.open(str(video_path))
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is not None:
            resampler = av.AudioResampler(format="flt", layout="mono", rate=sampling_rate)
            chunks = []
            for packet in container.demux(audio_stream):
                for frame in packet.decode():
                    for resampled in resampler.resample(frame):
                        chunks.append(resampled.to_ndarray())
            for resampled in resampler.resample(None):
                chunks.append(resampled.to_ndarray())
            if chunks:
                return np.concatenate(chunks, axis=1).squeeze(0)
    except Exception:
        pass
    return None


def infer_protocol(row):
    protocol = row.get("protocol")
    if protocol in {"VT", "VA"}:
        return protocol
    parts = str(row.get("sample_id", "")).split(":")
    if len(parts) >= 2 and parts[1] in {"VT", "VA"}:
        return parts[1]
    if len(parts) >= 2 and parts[1] == "accept_a_va":
        return "VA"
    if len(parts) >= 2 and parts[1] in {"accept_a_svt", "accept_c_svt"}:
        return "VT"
    return "VT"


def build_sft_prompt(row, protocol):
    text_val = str(row.get("text_content", row.get("text", ""))).strip()
    if protocol == "VT":
        modality = "The input contains the video and supplied transcript text."
        supplied_text = f"\nSupplied text: {json.dumps(text_val, ensure_ascii=False)}"
    else:
        modality = "The input contains video frames and the video's embedded audio. Supplied transcript text is unavailable."
        supplied_text = ""
    return (
        f"{modality}\n"
        "Analyze the person's overall emotion. Treat visible expression, body "
        "language, spoken or supplied text, and vocal evidence as separate signals "
        f"when those signals are available.{supplied_text}\n"
        f"{TARGET_INSTRUCTION}"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/root/autodl-tmp/models/qwen25omni-sft-adapter")
    p.add_argument("--adapter-path", default="output/rl/phase_grpo_v2/checkpoint-100")
    p.add_argument("--test-file", default=None)
    p.add_argument("--data-root", default="Dataset")
    p.add_argument("--output", default="output/rl/grpo_4bit/raw_predictions.jsonl")
    p.add_argument("--submission", default="output/rl/grpo_4bit/submission.csv")
    p.add_argument("--load-in-4bit", action="store_true", default=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def resolve_video(path, root):
    cand = os.path.join(root, path)
    return cand if os.path.isfile(cand) else path


def main():
    args = parse_args()
    test_file = resolve_test_file(args.test_file)
    print(f"[1/4] Test file: {test_file}")

    with open(test_file, "r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} test samples.")

    import av as _av
    if int(str(_av.__version__).split(".")[0]) >= 15:
        _orig_av_open = _av.open
        _av.open = lambda *a, **kw: _orig_av_open(*a, **{k: v for k, v in kw.items() if not k.startswith("metadata_")})

    print(f"[2/4] Loading processor & 4-bit base model from {args.model_path} ...")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    processor.video_processor.do_sample_frames = True
    processor.video_processor.fps = 2.0

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        quantization_config=quant_config,
        device_map={"": args.device},
        attn_implementation="sdpa",
    )
    model.disable_talker()

    if args.adapter_path:
        ad_dir = args.adapter_path
        if not os.path.isfile(os.path.join(ad_dir, "adapter_config.json")):
            for r, _, fs in os.walk(ad_dir):
                if "adapter_config.json" in fs:
                    ad_dir = r
                    break
        print(f"Mounting GRPO adapter on Thinker from: {ad_dir}")
        model.thinker = PeftModel.from_pretrained(model.thinker, ad_dir, is_trainable=False)

    model.eval()
    print(f"[3/4] Running 4-bit inference with native SFT prompt on {len(rows)} samples ...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    t_start = time.time()

    with open(args.output, "w", encoding="utf-8") as out_fh:
        for idx, row in enumerate(rows, 1):
            sid = row["sample_id"]
            video_path = resolve_video(row.get("video", ""), args.data_root)
            protocol = infer_protocol(row)
            prompt_text = build_sft_prompt(row, protocol)

            use_audio = (protocol == "VA")
            audio = audio_from_video(video_path) if use_audio else None
            actual_use_audio = use_audio and (audio is not None)

            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "video", "video": video_path}, {"type": "text", "text": prompt_text}]},
            ]

            t0 = time.time()
            try:
                videos_kwargs = {
                    "use_audio_in_video": actual_use_audio,
                    "do_sample_frames": True,
                    "fps": 2.0,
                }
                rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=rendered,
                    videos=str(video_path),
                    audio=audio if actual_use_audio else None,
                    padding=True,
                    return_tensors="pt",
                    videos_kwargs=videos_kwargs,
                )
                inputs = {k: v.to(args.device) if hasattr(v, "to") else v for k, v in inputs.items()}

                with torch.no_grad():
                    out_ids = model.generate(
                        **inputs,
                        use_audio_in_video=actual_use_audio,
                        return_audio=False,
                        thinker_max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
                in_len = inputs["input_ids"].shape[1]
                gen_ids = out_ids[0, in_len:] if out_ids.shape[1] >= in_len else out_ids[0]
                completion = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            except Exception as e:
                print(f"  [{idx}/{len(rows)}] Error on {sid}: {e}")
                completion = '{"pred_emotion": "negative", "is_uncertain": "no", "clarifying_question": "none"}'

            dt = time.time() - t0
            out_fh.write(json.dumps({"sample_id": sid, "completion": completion}, ensure_ascii=False) + "\n")
            out_fh.flush()

            parsed = parse_completion(completion)
            stat = f"emotion={parsed['pred_emotion']}, uncertain={parsed['is_uncertain']}" if parsed else "raw"
            print(f"  [{idx}/{len(rows)}] {sid} ({dt:.2f}s) -> {stat} | {completion[:70]}")

    elapsed = time.time() - t_start
    print(f"\nInference completed in {elapsed:.1f}s ({elapsed / len(rows):.2f}s/sample).")

    if args.submission:
        print(f"\n[4/4] Building Kaggle submission CSV: {args.submission} ...")
        make_submission.main([
            "--input", args.output,
            "--test-file", test_file,
            "--output", args.submission,
            "--fallback-emotion", "negative",
            "--binary-only",
        ])


if __name__ == "__main__":
    main()
