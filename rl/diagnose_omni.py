"""Diagnose v2: Qwen2.5-Omni processor grid rows vs <|VIDEO|> tokens.

Run on the AutoDL server from the project root:

    cd /root/autodl-tmp/BUAA-Ai-Principles-2026
    python rl/diagnose_omni.py 2>&1 | tail -80

Includes the PyAV>=15 shim so we get PAST video decoding to the real issue.
"""
import json
import traceback

# --- PyAV >= 15 shim (same as in train_grpo.py) ----------------------------
import av as _av

if int(str(_av.__version__).split(".")[0]) >= 15:
    _orig = _av.open

    def _compat(*a, **kw):
        kw.pop("metadata_errors", None)
        kw.pop("metadata_encoding", None)
        return _orig(*a, **kw)

    _av.open = _compat
    print(f"[shim] av {_av.__version__} patched")
# -----------------------------------------------------------------------------

from transformers import Qwen2_5OmniProcessor

MODEL = "/root/autodl-tmp/models/qwen25omni-sft-adapter"
DATA = "rl/data/rl_prompts.jsonl"

p = Qwen2_5OmniProcessor.from_pretrained(MODEL)
tok = p.tokenizer

rows = [json.loads(line) for line in open(DATA) if line.strip()][:16]
convs = [r["prompt"] for r in rows]
vids = [r["video"] for r in rows]

print("\n=== 1. video_grid_thw rows: nested vs flat (2 videos) ===")
for label, arg in [("nested [[v1],[v2]]", [vids[:1], vids[1:2]]),
                   ("flat   [v1,v2]", vids[:2])]:
    try:
        vi = p.video_processor(videos=arg)
        print(f"{label}: grid_thw shape = {tuple(vi['video_grid_thw'].shape)}")
    except Exception as exc:
        print(f"{label}: FAILED {type(exc).__name__}: {exc}")

print("\n=== 2. rendered token counts vs videos, batch of 2 ===")
txts = p.apply_chat_template(conversation=convs[:2], add_generation_prompt=True,
                             tokenize=False)
total_tokens = sum(t.count(tok.video_token) for t in txts)
print(f"samples={len(txts)}, total <|VIDEO|> tokens={total_tokens}, videos=2")

print("\n=== 3. direct __call__ (text + nested videos), batch of 2 ===")
try:
    out = p(text=txts, videos=[vids[:1], vids[1:2]])
    print("OK:", {k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
                  for k, v in out.items()})
except Exception:
    traceback.print_exc()

print("\n=== 4. batched apply_chat_template tokenize=True (TRL path) ===")
for n in (1, 2, 4, 16):
    try:
        out = p.apply_chat_template(conversation=convs[:n],
                                    add_generation_prompt=True,
                                    tokenize=True, return_dict=True)
        grid = out.get("video_grid_thw")
        print(f"batch={n}: OK, grid_thw="
              f"{tuple(grid.shape) if grid is not None else None}")
    except Exception as exc:
        print(f"batch={n}: FAILED {type(exc).__name__}: {exc}")
        if n == 2:
            traceback.print_exc()
