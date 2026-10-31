"""Download one published emotion checkpoint and validate/merge it on CPU.

No upstream repository code, benchmark labels or pickled training states are
downloaded. The immutable source list and every downloaded file are hashed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--download-only", action="store_true")
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    from modelscope.hub.api import HubApi
    from modelscope.hub.snapshot_download import snapshot_download
    api = HubApi()
    existing = a.root / "bootstrap.status.json"
    previous = json.loads(existing.read_text()) if existing.exists() else {}
    started = previous.get("started_unix", time.time())
    state = {"phase": "download", "started_unix": started, "pid": os.getpid()}
    done = threading.Event()

    def progress():
        while not done.is_set():
            sizes = []
            for directory in (a.project / "models/Qwen2.5-Omni-7B", a.root / "published"):
                if directory.exists():
                    sizes.extend(f.stat().st_size for f in directory.rglob("*")
                                 if f.is_file() and not f.name.startswith("."))
            atomic(a.root / "bootstrap.status.json", {
                **state, "updated_unix": time.time(), "downloaded_bytes": sum(sizes),
                "elapsed_seconds": time.time() - started})
            done.wait(5)

    thread = threading.Thread(target=progress, daemon=True)
    thread.start()
    try:
        contracts = []
        specs = [
            ("Qwen/Qwen2.5-Omni-7B", a.project / "models/Qwen2.5-Omni-7B", ""),
            ("SmilingWeeping/EmotionReasoner", a.root / "published", "EmotionReasoner/checkpoint-195/"),
        ]
        for repo, destination, prefix in specs:
            files = api.get_model_files(repo, revision="master", recursive=True)
            selected = [f for f in files if str(f["Path"]).startswith(prefix)
                        and (not prefix or "/" not in str(f["Path"])[len(prefix):])
                        and (str(f["Path"]).endswith((".safetensors", ".json", ".jinja", ".txt"))
                             or (not prefix and f["Path"] == "spk_dict.pt")
                             or f["Path"] == "LICENSE")]
            if not selected or (prefix and not any(f["Path"].endswith("adapter_model.safetensors") for f in selected)):
                raise ValueError("Required safe model files are missing: " + repo)
            # Record the remote list before downloading; post-download SHA is
            # the definitive identity even when a publisher only exposes master.
            contract = {"repository": repo, "requested_revision": "master", "files": selected,
                        "destination": str(destination)}
            contracts.append(contract)
            state.update(repository=repo, expected_bytes=sum(f["Size"] for c in contracts for f in c["files"]))
            atomic(a.root / "download-sources.json", contracts)
            snapshot_download(repo, revision="master", local_dir=str(destination),
                              allow_file_pattern=[f["Path"] for f in selected], max_workers=4)
            for f in selected:
                path = destination / f["Path"]
                if not path.is_file() or path.stat().st_size != f["Size"]:
                    raise ValueError("Download size mismatch: " + str(path))
            contract["verified_files"] = [{"path": f["Path"], "bytes": f["Size"],
                                             "sha256": sha(destination / f["Path"])} for f in selected]
            atomic(a.root / "download-sources.json", contracts)
        state.update(phase="download_complete", completed_unix=time.time())
        if a.download_only:
            return
        merge(a, state)
        state.update(phase="complete", completed_unix=time.time())
    except BaseException as e:
        state.update(phase="failed", error=repr(e), completed_unix=time.time())
        raise
    finally:
        done.set()
        thread.join(timeout=8)
        atomic(a.root / "bootstrap.status.json", {**state, "updated_unix": time.time()})


def merge(a, state):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from peft import PeftConfig, PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    destination = a.root / "merged-emotion-base"
    marker = destination / "merge.complete.json"
    if marker.exists():
        state["merge"] = json.loads(marker.read_text())
        return
    state["phase"] = "merge_cpu"
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch.set_num_threads(8)
    base = a.project / "models/Qwen2.5-Omni-7B"
    adapter = a.root / "published/EmotionReasoner/checkpoint-195"
    # Transformers' full Omni loader always reads this official speaker tensor
    # asset, even for classification. Validate it with the restricted loader;
    # it is not a training/optimizer pickle and never executes remote code.
    speaker = torch.load(base / "spk_dict.pt", map_location="cpu", weights_only=True)
    if not isinstance(speaker, dict):
        raise ValueError("Expected an official speaker tensor dictionary")
    expected = load_file(str(adapter / "adapter_model.safetensors"))
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        str(base), torch_dtype=torch.float32, device_map={"": "cpu"},
        attn_implementation="sdpa", local_files_only=True)
    model.disable_talker()
    model.config.enable_audio_output = False
    # Publisher trained a full Omni PEFT model, including saved embeddings.
    # Never strip tensor prefixes or ignore incompatible saved modules.
    config = PeftConfig.from_pretrained(str(adapter), local_files_only=True)
    resolved = set()
    for key in expected:
        if not key.startswith("base_model.model.") or ".lora_" not in key:
            raise ValueError("Unexpected publisher tensor requires explicit handling: " + key)
        resolved.add(key[len("base_model.model."):].split(".lora_", 1)[0])
    native_names = dict(model.named_modules())
    if any(name not in native_names for name in resolved):
        raise ValueError("Published LoRA targets do not exist in native Omni")
    # Generic publisher targets would also instantiate unprovided adapters in
    # audio/talker modules. Bind precisely to modules actually in checkpoint.
    config.target_modules = sorted(resolved)
    # CPU merging only needs the generic PEFT wrapper. Omni's top-level
    # multimodal generator is not a standard causal-LM forward interface.
    config.task_type = None
    wrapped = PeftModel.from_pretrained(model, str(adapter), config=config,
        is_trainable=False, local_files_only=True)
    loaded = get_peft_model_state_dict(wrapped, save_embedding_layers=False)
    if set(expected) != set(loaded):
        raise ValueError("Published adapter tensor names did not load exactly: " +
                         str({"missing": sorted(set(expected)-set(loaded))[:8],
                              "extra": sorted(set(loaded)-set(expected))[:8]}))
    for key, value in expected.items():
        if value.shape != loaded[key].shape or not torch.equal(value.float(), loaded[key].detach().float().cpu()):
            raise ValueError("Published adapter was not loaded exactly: " + key)
    identity = {"adapter_sha256": sha(adapter / "adapter_model.safetensors"),
                "adapter_tensors": len(expected), "all_tensors_exact": True,
                "merge_dtype": "float32", "saved_dtype": "bfloat16",
                "resolved_targets": sorted(resolved), "lora_rank": config.r,
                "lora_alpha": config.lora_alpha, "scaling_unchanged": True}
    del expected, loaded
    merged = wrapped.merge_and_unload(safe_merge=True).to(dtype=torch.bfloat16)
    merged.save_pretrained(str(destination), safe_serialization=True, max_shard_size="4GB")
    processor = Qwen2_5OmniProcessor.from_pretrained(str(base), local_files_only=True)
    # A different publisher vocabulary must be checked before replacing it.
    from transformers import AutoTokenizer
    publisher_tok = AutoTokenizer.from_pretrained(str(adapter), local_files_only=True)
    if processor.tokenizer.get_vocab() != publisher_tok.get_vocab():
        raise ValueError("Publisher and base tokenizer vocabularies differ")
    processor.save_pretrained(str(destination))
    import shutil
    shutil.copyfile(base / "spk_dict.pt", destination / "spk_dict.pt")
    identity["files"] = {f.name: sha(f) for f in destination.iterdir() if f.is_file()}
    identity["completed_unix"] = time.time()
    atomic(marker, identity)
    state["merge"] = identity


if __name__ == "__main__":
    main()
