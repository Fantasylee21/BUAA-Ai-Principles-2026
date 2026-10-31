#!/usr/bin/env python3
"""Rebuild a full Qwen2.5-Omni model from an LLaMAFactory thinker-only export.

LLaMAFactory exports Omni LoRA merges as the standalone thinker sub-model
(qwen2_5_omni_thinker). The full model wraps thinker + talker + token2wav
under `thinker.`/`talker.`/`token2wav.` prefixes. This script overlays the
merged thinker weights onto the original full model and saves a loadable
full-omni checkpoint.
"""
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

REPO = Path("/home/lijiaze/BUAA-Ai-Principles-2026")
ORIG = REPO / "models/Qwen2.5-Omni-3B"
MERGED = REPO / "output/sft/qwen25omni/lora_sft_v1_merged"
OUT = REPO / "output/sft/qwen25omni/lora_sft_v1_merged_full"

OUT.mkdir(parents=True, exist_ok=True)

# 1. copy config + non-weight files from original full model
for f in ORIG.iterdir():
    if f.suffix == ".safetensors" or f.name == "model.safetensors.index.json":
        continue
    if f.is_file():
        shutil.copy2(f, OUT / f.name)

# 2. load original full state dict
idx = json.load(open(ORIG / "model.safetensors.index.json"))
shards = sorted(set(idx["weight_map"].values()))
state = {}
for s in shards:
    state.update(load_file(ORIG / s))
print("original tensors:", len(state))

# 3. overlay merged thinker weights (prefix with thinker.)
midx = json.load(open(MERGED / "model.safetensors.index.json"))
mshards = sorted(set(midx["weight_map"].values()))
n = 0
for s in mshards:
    part = load_file(MERGED / s)
    for k, v in part.items():
        tk = "thinker." + k
        if tk not in state:
            raise KeyError(f"unexpected merged key {tk}")
        state[tk] = v
        n += 1
print("overlaid thinker tensors:", n)

# 4. save as single safetensors + minimal index removal
save_file(state, OUT / "model.safetensors", metadata={"format": "pt"})
print("saved to", OUT)
