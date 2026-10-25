"""Run inference on the test set and generate Kaggle submission CSV."""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_submission import DEFAULT_TEST_FILE_CANDIDATES, resolve_test_file
import make_submission
from reward import parse_completion

OUTPUT_CONTRACT = (
    '{"pred_emotion": "positive"|"negative", '
    '"is_uncertain": "yes"|"no", '
    '"clarifying_question": "<string or \'none\'>"}'
)

INSTRUCTION = (
    "You are an emotion recognition assistant. Watch the video and judge the "
    "speaker's true overall emotion.\n"
    "Answer with exactly one compact JSON object and nothing else (no reasoning, no extra text):\n"
    f"{OUTPUT_CONTRACT}\n"
    "Rules:\n"
    '- "pred_emotion" is "positive" or "negative".\n'
    '- Set "is_uncertain" to "yes" when the modalities seem to conflict or the evidence is ambiguous; otherwise set it to "no".\n'
    '- Provide a short "clarifying_question" ONLY when "is_uncertain" is "yes"; otherwise it must be exactly "none".\n'
    "Keep the answer short."
)


def parse_args():
    p = argparse.ArgumentParser(description="Test inference and submission generation.")
    p.add_argument("--model-path", default="/root/autodl-tmp/models/qwen25omni-sft-adapter")
    p.add_argument("--adapter-path", default="output/rl/smoke")
    p.add_argument("--test-file", default=None)
    p.add_argument("--data-root", default="Dataset")
    p.add_argument("--output", default="output/rl/smoke/raw_predictions.jsonl")
    p.add_argument("--submission", default="output/rl/smoke/submission.csv")
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
    total_samples = len(rows)
    print(f"Loaded {total_samples} test samples.")

    if args.limit:
        rows = rows[: args.limit]
        print(f"Limiting inference to first {len(rows)} samples.")

    import torch
    from peft import PeftModel
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    import av as _av

    if int(str(_av.__version__).split(".")[0]) >= 15:
        _orig_av_open = _av.open
        _av.open = lambda *a, **kw: _orig_av_open(*a, **{k: v for k, v in kw.items() if not k.startswith("metadata_")})

    print(f"[2/4] Loading model & processor from {args.model_path} ...")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    processor.video_processor.do_sample_frames = True
    processor.video_processor.fps = 2.0

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path, dtype="bfloat16", attn_implementation="sdpa"
    )
    model.disable_talker()

    if args.adapter_path:
        ad_dir = args.adapter_path
        if not os.path.isfile(os.path.join(ad_dir, "adapter_config.json")):
            for r, _, fs in os.walk(ad_dir):
                if "adapter_config.json" in fs:
                    ad_dir = r
                    break
        print(f"Mounting adapter from: {ad_dir}")
        model.thinker = PeftModel.from_pretrained(model.thinker, ad_dir, is_trainable=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"[3/4] Running inference on {len(rows)} samples ...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    t_start = time.time()

    with open(args.output, "w", encoding="utf-8") as out_fh:
        for idx, row in enumerate(rows, 1):
            sid = row["sample_id"]
            video_path = resolve_video(row.get("video", ""), args.data_root)
            text_val = row.get("text") or row.get("text_content") or ""
            prompt_text = INSTRUCTION + (f"\nTranscript: {text_val.strip()}" if text_val.strip() else "\nUse the audio from the video to judge the emotion.")

            conv = [{"role": "user", "content": [{"type": "video", "video": video_path}, {"type": "text", "text": prompt_text}]}]
            t0 = time.time()
            try:
                inputs = processor.apply_chat_template(
                    conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
                )
                inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
                with torch.no_grad():
                    try:
                        out_ids = model.generate(
                            **inputs,
                            return_audio=False,
                            thinker_max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            use_cache=True,
                        )
                    except TypeError:
                        out_ids = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            use_cache=True,
                        )
                in_len = inputs["input_ids"].shape[1]
                if out_ids.shape[1] >= in_len and torch.equal(out_ids[:, :in_len], inputs["input_ids"]):
                    gen_ids = out_ids[0, in_len:]
                else:
                    gen_ids = out_ids[0]
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
    print(f"Raw predictions saved to: {args.output}")

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
