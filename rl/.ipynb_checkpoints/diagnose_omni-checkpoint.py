"""Diagnose the Qwen2.5-Omni processor + TRL apply_chat_template mismatch.

Run on the AutoDL server from the project root:

    cd /root/autodl-tmp/BUAA-Ai-Principles-2026
    python rl/diagnose_omni.py

Prints:
  1. Which chat template the merged SFT model ships and how it renders
     video/audio placeholders for our RL prompts.
  2. Whether the batched (TRL-style) apply_chat_template call crashes and
     whether per-sample calls work — this picks the fix.
"""
import json
import re
import traceback

from transformers import Qwen2_5OmniProcessor

MODEL = "/root/autodl-tmp/models/qwen25omni-sft-adapter"
DATA = "rl/data/rl_prompts.jsonl"

p = Qwen2_5OmniProcessor.from_pretrained(MODEL)
tok = p.tokenizer
print("video_token:", tok.video_token, "| audio_token:", tok.audio_token,
      "| image_token:", tok.image_token)

t = p.chat_template
print("\n=== chat template ===")
print("type:", type(t).__name__, "| len:", len(t) if isinstance(t, str) else "-")
if isinstance(t, str):
    for m in re.finditer(r"\{%-?\s*if[^%}]*(video|audio|image)[^%}]*%\}", t):
        print("TPL-BRANCH>", m.group(0)[:200])

rows = [json.loads(line) for line in open(DATA) if line.strip()][:2]
convs = [r["prompt"] for r in rows]

print("\n=== rendered text (tokenize=False) ===")
txts = p.apply_chat_template(conversation=convs, add_generation_prompt=True,
                             tokenize=False)
for i, txt in enumerate(txts):
    print(f"--- sample {i} ---")
    print(repr(txt[:250]))
    print("counts  video:", txt.count(tok.video_token),
          " audio:", txt.count(tok.audio_token),
          " image:", txt.count(tok.image_token))

print("\n=== video processor defaults ===")
vp = p.video_processor
print({k: getattr(vp, k, None) for k in
       ["do_sample_frames", "fps", "num_frames", "min_frames", "max_frames",
        "temporal_patch_size", "merge_size", "size"]})

print("\n=== batched tokenize=True (what TRL does) ===")
try:
    out = p.apply_chat_template(conversation=convs, add_generation_prompt=True,
                                tokenize=True, return_dict=True)
    print("OK:", {k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
                  for k, v in out.items()})
except Exception:
    traceback.print_exc()

print("\n=== per-sample tokenize=True (candidate workaround) ===")
for i, conv in enumerate(convs):
    try:
        out = p.apply_chat_template(conversation=[conv], add_generation_prompt=True,
                                    tokenize=True, return_dict=True)
        print(f"sample {i} OK:", {k: (tuple(v.shape) if hasattr(v, "shape")
                                     else type(v).__name__) for k, v in out.items()})
    except Exception as exc:
        print(f"sample {i} FAILED: {type(exc).__name__}: {exc}")
