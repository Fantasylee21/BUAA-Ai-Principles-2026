"""Trainable isolated emotion evidence and a joint affect distribution.

Public API
----------
HFBackend(HFBackendConfig(...)) loads raw local HF weights only. Its
``encode_vision(row)`` and ``encode_second(row)`` return TokenBatch; no gold,
sample ID or annotation is rendered. ``JointAffectModel(backbone, 'C0'|'C1')``
accepts that backend or a CPU fake with the same two methods and hidden sizes.
``forward_row(row, second_row=None)`` preserves modality isolation for remixes.
``compute_objectives(outputs, make_targets(row), counts, phase=...)`` returns
separate main_loss/remix_loss/warmup_loss. **remix_loss already includes .25**;
the trainer must normalize real and remix accumulations separately.

Checkpoint directories contain trainable.pt and trainable.json, never base
weights or media. ``init_second_from`` initializes an 8B upgrade from the same
fold's earlier winner while leaving the new visual stream/pool untouched.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import uuid

import torch
from torch import nn
import torch.nn.functional as F

VERSION = "joint_affect_trainable_v1"
EMOTIONS = ("negative", "neutral", "positive")
EVIDENCE_LABELS = EMOTIONS + ("unclear",)
ALIGNED_INDICES = (0, 5, 10)
CONFLICT_INDICES = tuple(i for i in range(16) if i not in ALIGNED_INDICES)


@dataclass(frozen=True)
class ModelConfig:
    width: int = 256
    queries: int = 16
    layers: int = 2
    attention_heads: int = 4
    dropout: float = .05
    fusion_device: str = "cpu"


@dataclass(frozen=True)
class HFBackendConfig:
    vision_model_path: str
    second_model_path: str
    dataset_root: str
    audio_cache_dir: str | None = None
    vision_device: str = "cuda:0"
    second_device: str = "cuda:1"
    fps: float = 2.
    min_frames: int = 4
    max_frames: int = 64
    pixels_per_frame: int = 50176
    max_prompt_tokens: int = 6144
    audio_sampling_rate: int = 16000
    gradient_checkpointing: bool = True
    decoder_rank: int = 16
    decoder_alpha: int = 32
    perception_rank: int = 8
    perception_alpha: int = 16
    perception_top_blocks: int = 4
    lora_dropout: float = .05


@dataclass
class TokenBatch:
    tokens: torch.Tensor
    mask: torch.Tensor | None = None
    audit: dict | None = None

    def checked(self):
        if self.tokens.ndim != 3 or not self.tokens.shape[1] or not torch.isfinite(self.tokens).all():
            raise ValueError("Expected nonempty finite [batch,tokens,hidden] modality states")
        if self.mask is None:
            self.mask = torch.ones(self.tokens.shape[:2], dtype=torch.bool, device=self.tokens.device)
        if (self.mask.shape != self.tokens.shape[:2] or self.mask.dtype != torch.bool or
                not self.mask.any(-1).all()):
            raise ValueError("Every modality needs at least one valid token")
        return self


class QueryPool(nn.Module):
    def __init__(self, hidden_size, config):
        super().__init__()
        self.project = nn.Linear(hidden_size, config.width)
        self.queries = nn.Parameter(torch.randn(1, config.queries, config.width) * .02)
        self.attention = nn.MultiheadAttention(config.width, config.attention_heads,
            dropout=config.dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(config.width), nn.LayerNorm(config.width)
        self.ff = nn.Sequential(nn.Linear(config.width, 4 * config.width), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(4 * config.width, config.width))

    def forward(self, tokens, mask):
        values = self.project(tokens)
        queries = self.queries.expand(tokens.shape[0], -1, -1)
        pooled = queries + self.attention(queries, values, values,
            key_padding_mask=~mask, need_weights=False)[0]
        pooled = self.norm1(pooled)
        return self.norm2(pooled + self.ff(pooled))


class BidirectionalLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.v_to_m = nn.MultiheadAttention(config.width, config.attention_heads,
            dropout=config.dropout, batch_first=True)
        self.m_to_v = nn.MultiheadAttention(config.width, config.attention_heads,
            dropout=config.dropout, batch_first=True)
        self.v_norm1, self.m_norm1 = nn.LayerNorm(config.width), nn.LayerNorm(config.width)
        self.v_norm2, self.m_norm2 = nn.LayerNorm(config.width), nn.LayerNorm(config.width)
        self.v_ff = nn.Sequential(nn.Linear(config.width, 4 * config.width), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(4 * config.width, config.width))
        self.m_ff = nn.Sequential(nn.Linear(config.width, 4 * config.width), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(4 * config.width, config.width))

    def forward(self, visual, second):
        # Both directions use the preceding layer's streams, without detaching.
        v = self.v_norm1(visual + self.v_to_m(visual, second, second, need_weights=False)[0])
        m = self.m_norm1(second + self.m_to_v(second, visual, visual, need_weights=False)[0])
        return self.v_norm2(v + self.v_ff(v)), self.m_norm2(m + self.m_ff(m))


def joint_uncertainty(logits):
    if logits.shape[-1] == 16:
        flat = logits
    elif logits.shape[-2:] == (4, 4):
        flat = logits.flatten(-2)
    else:
        raise ValueError("Joint relation logits must have sixteen states")
    flat = flat.float()
    logit = torch.logsumexp(flat[..., list(CONFLICT_INDICES)], -1) - torch.logsumexp(
        flat[..., list(ALIGNED_INDICES)], -1)
    probabilities = flat.softmax(-1).reshape(*flat.shape[:-1], 4, 4)
    return probabilities, logit, probabilities.flatten(-2)[..., list(CONFLICT_INDICES)].sum(-1)


class JointAffectModel(nn.Module):
    def __init__(self, backbone, variant="C1", config=None):
        super().__init__()
        if variant not in ("C0", "C1"):
            raise ValueError("Only the fixed C0/C1 variants are supported")
        self.backbone, self.variant, self.config = backbone, variant, config or ModelConfig(
            fusion_device=getattr(backbone, "fusion_device", "cpu"))
        cfg = self.config
        if (cfg.width <= 0 or cfg.width % cfg.attention_heads or cfg.queries <= 0 or
                cfg.layers != 2 or cfg.attention_heads != 4):
            raise ValueError("Joint affect uses two bidirectional layers and four attention heads")
        # Place only new modules here. Calling .to() on the complete model can
        # move NF4 backbones off their independently assigned devices.
        device = torch.device(cfg.fusion_device)
        self.vision_pool = QueryPool(backbone.vision_hidden_size, cfg).to(device)
        self.second_pool = QueryPool(backbone.second_hidden_size, cfg).to(device)
        self.protocol_embedding = nn.Embedding(2, cfg.width).to(device)
        self.fusion = nn.ModuleList([BidirectionalLayer(cfg) for _ in range(cfg.layers)]).to(device)
        self.readout = nn.Sequential(nn.Linear(2 * cfg.width, cfg.width), nn.GELU(),
                                     nn.LayerNorm(cfg.width)).to(device)
        self.emotion_head = nn.Linear(cfg.width, 3).to(device)
        self.vision_head = nn.Linear(cfg.width, 4).to(device)
        self.second_head = nn.Linear(cfg.width, 4).to(device)
        if variant == "C1":
            self.joint_head = nn.Linear(cfg.width, 16).to(device)
        else:
            self.uncertainty_head = nn.Linear(cfg.width, 1).to(device)
        if getattr(backbone, "init_second_checkpoint", None):
            self.upgrade_initialization = self.init_second_from(backbone.init_second_checkpoint)

    def forward_tokens(self, visual, second, protocol="VT"):
        visual = visual if isinstance(visual, TokenBatch) else TokenBatch(visual)
        second = second if isinstance(second, TokenBatch) else TokenBatch(second)
        visual, second = visual.checked(), second.checked()
        if visual.tokens.shape[0] != second.tokens.shape[0] or protocol not in ("VT", "VA"):
            raise ValueError("The paired modality batch and protocol must agree")
        device = self.emotion_head.weight.device
        mixed = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with mixed:
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            v = self.vision_pool(visual.tokens.to(device=device, dtype=dtype), visual.mask.to(device))
            m = self.second_pool(second.tokens.to(device=device, dtype=dtype), second.mask.to(device))
            # Auxiliary heads see their own isolated stream before interaction.
            result = {"vision_logits": self.vision_head(v.mean(1)).float(),
                      "second_logits": self.second_head(m.mean(1)).float()}
            embedding = self.protocol_embedding.weight[int(protocol == "VA")][None, None]
            vf, mf = v + embedding, m + embedding
            for layer in self.fusion:
                vf, mf = layer(vf, mf)
            hidden = self.readout(torch.cat((vf.mean(1), mf.mean(1)), -1))
            result["emotion_logits"] = self.emotion_head(hidden).float()
            if self.variant == "C1":
                logits = self.joint_head(hidden).float()
                joint, ulogit, uncertainty = joint_uncertainty(logits)
                result.update(joint_logits=logits, joint_probs=joint,
                              uncertain_logit=ulogit, uncertain_prob=uncertainty)
            else:
                ulogit = self.uncertainty_head(hidden).float().squeeze(-1)
                result.update(uncertain_logit=ulogit, uncertain_prob=ulogit.sigmoid())
        result.update(emotion_probs=result["emotion_logits"].softmax(-1),
            vision_probs=result["vision_logits"].softmax(-1),
            second_probs=result["second_logits"].softmax(-1), variant=self.variant)
        return result

    def forward_row(self, row, second_row=None):
        second_row = row if second_row is None else second_row
        if row["protocol"] != second_row["protocol"]:
            raise ValueError("Remixes cannot cross VT/VA protocols")
        visual = self.backbone.encode_vision(row)
        second = self.backbone.encode_second(second_row)
        result = self.forward_tokens(visual, second, row["protocol"])
        result["input_audit"] = {"vision": visual.audit, "second": second.audit}
        return result

    def _group(self, name):
        if name.startswith("backbone."):
            return self.backbone.parameter_group(name[len("backbone."):]) if hasattr(self.backbone, "parameter_group") else "backbone"
        first = name.split(".", 1)[0]
        return "fusion" if first in ("protocol_embedding", "fusion", "readout") else first

    def trainable_audit(self):
        parameters, totals, frozen = [], {}, 0
        for name, p in self.named_parameters():
            if not p.requires_grad:
                frozen += p.numel()
                continue
            group = self._group(name)
            totals[group] = totals.get(group, 0) + p.numel()
            parameters.append({"name": name, "group": group, "shape": list(p.shape),
                "numel": p.numel(), "dtype": str(p.dtype), "device": str(p.device),
                "active_phases": ["evidence_warmup"] if self.variant == "C0" and group in ("vision_head", "second_head")
                    else ["main"] if group in ("emotion_head", "joint_head", "uncertainty_head", "fusion")
                    else ["evidence_warmup", "main"]})
        return {"variant": self.variant, "parameters": parameters, "groups": totals,
                "trainable_numel": sum(totals.values()), "frozen_numel": frozen,
                "connector_master_dtype": "torch.float32",
                "parameter_roles": {p["name"]: p["group"] for p in parameters},
                "backbone": getattr(self.backbone, "audit_config", {"kind": "CPU_fake_or_custom"})}

    def optimizer_groups(self, backbone_lr=2e-5, head_lr=1e-4, weight_decay=.01,
                         perception_lr=2e-5, connector_lr=2e-5):
        groups = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            category = self._group(name)
            lr = (connector_lr or backbone_lr) if "connector" in category else (
                (perception_lr or backbone_lr) if "perception" in category else
                backbone_lr if name.startswith("backbone.") else head_lr)
            decay = weight_decay if p.ndim > 1 and not name.endswith("bias") else 0.
            key = (category, lr, decay)
            groups.setdefault(key, []).append(p)
        return [{"name": k[0] + ("/decay" if k[2] else "/no_decay"), "params": p,
                 "lr": k[1], "weight_decay": k[2]} for k, p in groups.items()]

    def save_checkpoint(self, path, metadata=None):
        path = Path(path)
        state = {name: p.detach().cpu().clone() for name, p in self.named_parameters() if p.requires_grad}
        safe_metadata = json.loads(_json_bytes(metadata or {}))
        payload = {"format": VERSION, "variant": self.variant, "model_config": asdict(self.config),
            "hidden_sizes": [self.backbone.vision_hidden_size, self.backbone.second_hidden_size],
            "trainable": state, "metadata": safe_metadata}
        content = io.BytesIO()
        torch.save(payload, content)
        data = content.getvalue()
        _write_once(path / "trainable.pt", data)
        manifest = {k: v for k, v in payload.items() if k != "trainable"}
        manifest.update(sha256=hashlib.sha256(data).hexdigest(), audit=self.trainable_audit())
        _write_once(path / "trainable.json", _json_bytes(manifest))
        return manifest

    def load_checkpoint(self, path, strict=True):
        payload = _read_checkpoint(path)
        if payload["variant"] != self.variant or payload["hidden_sizes"] != [self.backbone.vision_hidden_size, self.backbone.second_hidden_size]:
            raise ValueError("Checkpoint architecture differs from this model")
        architecture = lambda c: {k: v for k, v in c.items() if k != "fusion_device"}
        if architecture(payload["model_config"]) != architecture(asdict(self.config)):
            raise ValueError("Checkpoint fusion configuration differs")
        return self._load_tensors(payload["trainable"], strict=strict)

    def init_second_from(self, path, load_compatible_heads=True):
        payload = _read_checkpoint(path)
        chosen = {}
        current = {n for n, p in self.named_parameters() if p.requires_grad}
        shared_checkpoint = not any(n.startswith("backbone.second_model.") for n in payload["trainable"])
        for name, value in payload["trainable"].items():
            if name.startswith("backbone.second_model.") or name.startswith("second_pool."):
                chosen[name] = value
            elif shared_checkpoint and name.startswith("backbone.vision_model."):
                # Native Omni shares one model, registered first as vision_model.
                # Its language/audio weights initialize only the upgrade's M
                # stream. Old visual adapters/connectors are never transferred.
                suffix = name[len("backbone.vision_model."):]
                if not any(part in suffix for part in (".visual.", ".vision_tower.", ".vision_model.")):
                    mapped = "backbone.second_model." + suffix
                    if mapped in current:
                        chosen[mapped] = value
            elif load_compatible_heads and not name.startswith(("backbone.", "vision_pool.")):
                chosen[name] = value
        required = {n for n, p in self.named_parameters() if p.requires_grad and n.startswith(("backbone.second_model.", "second_pool."))}
        if not required <= set(chosen):
            raise ValueError("Upgrade checkpoint is missing the complete second stream/pool")
        return self._load_tensors(chosen, strict=False)

    def _load_tensors(self, state, strict):
        current = {n: p for n, p in self.named_parameters() if p.requires_grad}
        if strict and set(state) != set(current):
            raise ValueError("Trainable checkpoint keys differ")
        loaded, skipped = [], []
        # Validate all shapes before applying any mutation.
        for name, value in state.items():
            if name not in current:
                if strict:
                    raise ValueError("Unknown checkpoint tensor: " + name)
                skipped.append(name)
                continue
            if value.shape != current[name].shape or not torch.isfinite(value).all():
                raise ValueError("Invalid checkpoint tensor shape/value: " + name)
        with torch.no_grad():
            for name, value in state.items():
                if name in current:
                    current[name].copy_(value.to(current[name].device, current[name].dtype))
                    loaded.append(name)
        return {"loaded": sorted(loaded), "skipped": sorted(skipped)}


def make_targets(row, second_row=None):
    second_row = row if second_row is None else second_row
    def state(record, key):
        value = record["views"][key]
        if value.get("is_clear") is not True:
            return 3
        if value["label"] not in EMOTIONS:
            raise ValueError("Invalid clear emotion target")
        return EMOTIONS.index(value["label"])
    visual, second = state(row, "M1"), state(second_row, "M2")
    remix = row.get("is_remix") is True or row.get("emotion_mask") == 0 or second_row is not row
    relation = int(not (visual == second and visual != 3))
    if not remix and relation != int(row["sample_type"] == "Conflict"):
        raise ValueError("Four-state relation target contradicts the original sample type")
    emotion = -100 if remix else EMOTIONS.index(row["views"]["M12"]["label"])
    natural = row.get("source_dataset") in ("ch_sims", "ch_sims_v2")
    return {"vision": visual, "second": second, "joint": 4 * visual + second,
        "emotion": emotion, "relation": relation, "emotion_mask": int(not remix),
        "is_remix": remix, "category": "aligned" if not relation else "natural_conflict" if natural else "synthetic_conflict",
        "video": row.get("video"), "sample_id": row.get("sample_id")}


def compute_objectives(outputs, targets, class_counts, phase="main", remix=None):
    """Per-row objectives; never normalize condition weights inside a batch.

    Counts: real_total/aligned/natural_conflict/synthetic_conflict,
    unique_videos/video_frequency. Optional remix_total/remix_aligned/
    remix_conflict preserve global balancing if the remix sampler is unequal.
    Returned remix_loss **already includes .25**. The caller separately divides
    real losses by real microbatch count and remix losses by remix count.
    """
    if phase not in ("main", "evidence_warmup"):
        raise ValueError("Unknown training phase")
    if outputs["emotion_logits"].shape[0] != 1:
        raise ValueError("The authorized trainer uses one logical row per microbatch")
    device = outputs["emotion_logits"].device
    zero = outputs["emotion_logits"].sum() * 0
    def ce(name, target):
        return F.cross_entropy(outputs[name], torch.tensor([target], device=device))
    vloss, mloss = ce("vision_logits", targets["vision"]), ce("second_logits", targets["second"])
    total = float(class_counts.get("real_total", class_counts.get("n_real", 0)))
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Global real training counts must be positive")
    frequency = class_counts.get("video_frequency", {})
    duplicates = frequency.get(targets.get("video"), 1)
    unique = class_counts.get("unique_videos", total)
    if duplicates <= 0 or unique <= 0:
        raise ValueError("Invalid global visual deduplication weights")
    visual_weight = float(targets.get("visual_weight", total / (float(unique) * duplicates)))
    vloss = visual_weight * vloss
    is_remix = targets.get("is_remix", not bool(targets.get("emotion_mask", 1))) if remix is None else remix
    joint = ce("joint_logits", targets.get("joint", 4 * targets["vision"] + targets["second"])) if "joint_logits" in outputs else zero
    binary = F.binary_cross_entropy_with_logits(outputs["uncertain_logit"],
        torch.tensor([float(targets["relation"])], device=device))
    main, remixed, warmup = zero, zero, zero
    if phase == "evidence_warmup":
        if is_remix:
            raise ValueError("Evidence warmup only uses original real examples")
        warmup = vloss + mloss
    elif is_remix:
        # No M12 target is read or fabricated for a remixed pair.
        group = "remix_conflict" if targets["relation"] else "remix_aligned"
        if "remix_total" in class_counts:
            count = float(class_counts.get(group, 0))
            if count <= 0:
                raise ValueError("Missing global remix class count")
            binary = binary * .5 * float(class_counts["remix_total"]) / count
        remixed = .25 * (binary + .5 * joint + .25 * vloss + .25 * mloss) if outputs["variant"] == "C1" else .25 * binary
    else:
        if targets.get("emotion_mask", 1) != 1 or targets["emotion"] not in range(3):
            raise ValueError("Original examples require the real M12 emotion target")
        category = targets.get("category")
        if category is None:
            category = "aligned" if not targets["relation"] else "natural_conflict" if targets.get("natural") else "synthetic_conflict"
        mass = {"aligned": .4, "natural_conflict": .3, "synthetic_conflict": .3}[category]
        count = float(class_counts.get(category, 0))
        if count <= 0:
            raise ValueError("Missing global class-conditional count: " + category)
        weighted_relation = binary * (total * mass / count)
        main = ce("emotion_logits", targets["emotion"]) + weighted_relation
        if outputs["variant"] == "C1":
            main = main + .5 * joint + .25 * vloss + .25 * mloss
    return {"loss": main + remixed + warmup, "main_loss": main, "remix_loss": remixed,
        "warmup_loss": warmup, "vision_loss": vloss, "second_loss": mloss,
        "joint_loss": joint, "relation_loss": binary, "visual_weight": visual_weight,
        "remix_weight_included": .25, "phase": phase}


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf8")


def _write_once(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("Refusing to overwrite a different model checkpoint: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "nt":
            temporary.rename(path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_checkpoint(path):
    path = Path(path)
    content = (path / "trainable.pt").read_bytes()
    manifest = json.loads((path / "trainable.json").read_text(encoding="utf8"))
    if manifest.get("sha256") != hashlib.sha256(content).hexdigest() or manifest.get("format") != VERSION:
        raise ValueError("Trainable checkpoint checksum or format changed")
    value = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    if value.get("format") != VERSION or value.get("variant") != manifest.get("variant"):
        raise ValueError("Trainable checkpoint manifest differs from its payload")
    return value


def isolated_messages(row, mode, media_path=None):
    """The only prompt builder; annotation keys are never inspected here."""
    content = []
    if mode == "vision":
        content.append({"type": "video", "video": str(media_path)})
        text = "Read only the visible emotion in this silent video. No audio or transcript is provided."
    elif mode == "audio":
        if row["protocol"] != "VA":
            raise ValueError("Only VA has isolated audio")
        content.append({"type": "audio", "audio": str(media_path)})
        text = "Read only the emotion conveyed by this audio. No video or supplied transcript is provided."
    elif mode == "text":
        if row["protocol"] != "VT" or not isinstance(row.get("text_content"), str) or not row["text_content"].strip():
            raise ValueError("VT requires a nonempty supplied utterance")
        text = "Read only the emotion in the supplied utterance.\n<utterance>\n" + row["text_content"] + "\n</utterance>"
    else:
        raise ValueError("Unknown isolated modality")
    content.append({"type": "text", "text": text})
    return [{"role": "system", "content": [{"type": "text", "text": "You are an emotion evidence encoder."}]},
            {"role": "user", "content": content}]


def module_plan(model, role, top_blocks=4):
    """Resolve exact HF names before quantization, including rank overrides."""
    if role not in ("both", "vision", "second"):
        raise ValueError("Unknown backbone role")
    linears = [n for n, module in model.named_modules() if isinstance(module, nn.Linear)]
    def sensor(name):
        return "vision" if ".visual." in "." + name or ".vision_tower." in "." + name else (
            "audio" if ".audio_tower." in "." + name or ".audio_encoder." in "." + name else None)
    allowed = {"vision", "audio"} if role == "both" else {"vision"} if role == "vision" else {"audio"}
    block_ids = {}
    for name in linears:
        kind = sensor(name)
        match = re.search(r"\.(?:blocks|layers|resblocks)\.(\d+)\.", "." + name)
        if kind in allowed and match:
            block_ids.setdefault(kind, set()).add(int(match.group(1)))
    if any(len(block_ids.get(kind, ())) < top_blocks for kind in allowed):
        raise ValueError("HF backbone lacks the required top perception blocks")
    selected = {k: set(sorted(v)[-top_blocks:]) for k, v in block_ids.items()}
    perception, decoder, connectors = [], [], set()
    for name in linears:
        kind = sensor(name)
        match = re.search(r"\.(?:blocks|layers|resblocks)\.(\d+)\.", "." + name)
        if kind in allowed and match and int(match.group(1)) in selected[kind]:
            perception.append(name)
        elif kind is None and ".layers." in "." + name and name.rsplit(".", 1)[-1] in (
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
            decoder.append(name)
        if kind in allowed:
            for pattern in (r"^(.*visual\.merger)(?:\.|$)",
                            r"^(.*visual\.deepstack_merger_list\.\d+)(?:\.|$)",
                            r"^(.*audio_tower\.proj)(?:\.|$)",
                            r"^(.*audio_encoder\.proj)(?:\.|$)"):
                found = re.match(pattern, name)
                if found:
                    connectors.add(found.group(1))
    if not decoder or not perception or not connectors:
        raise ValueError("Cannot identify decoder, perception adapters and dense connectors")
    return {"decoder": sorted(decoder), "perception": sorted(perception),
            "connectors": sorted(connectors), "top_block_ids": {k: sorted(v) for k, v in selected.items() if k in allowed}}


class HFBackend(nn.Module):
    """Local NF4 backbones with FP32-master LoRA and dense connectors.

    A native Omni run on one device shares the Thinker between two isolated
    forwards. An upgrade uses Qwen3VL on devices[0] and Omni on devices[1].
    No perception/last-token feature cache is reused because these features
    must acquire gradients. Optional existing audio PCM files are raw media.
    """
    def __init__(self, config):
        super().__init__()
        self.init_second_checkpoint = None
        if isinstance(config, dict):
            source = dict(config)
            requested = source.get("vision_model", "omni")
            devices = tuple(source.get("devices", ("cuda:0",)))
            if requested not in ("omni", "qwen3vl8b") or len(devices) != (2 if requested == "qwen3vl8b" else 1):
                raise ValueError("Native Omni requires one device; the 8B upgrade requires two")
            omni = source.get("omni_model_path", source.get("second_model_path"))
            if not omni:
                raise ValueError("The second stream needs the local Omni model path")
            vision_path = source.get("vision_model_path") if requested == "qwen3vl8b" else omni
            config = HFBackendConfig(vision_model_path=str(vision_path), second_model_path=str(omni),
                dataset_root=str(source["dataset_root"]),
                audio_cache_dir=source.get("audio_cache_dir", source.get("media_cache_dir")),
                vision_device=devices[0], second_device=devices[-1],
                max_frames=int(source.get("max_frames", 64)), fps=float(source.get("video_fps", 2.)),
                pixels_per_frame=int(source.get("video_pixels", 50176)))
            self.init_second_checkpoint = source.get("init_second_from")
        if not isinstance(config, HFBackendConfig):
            raise TypeError("HFBackend expects HFBackendConfig or the trainer's configuration dictionary")
        if (config.fps != 2. or config.min_frames != 4 or config.max_frames not in (32, 64) or
                config.pixels_per_frame != 50176 or config.max_prompt_tokens != 6144 or
                config.perception_top_blocks != 4 or config.decoder_rank != 16 or config.decoder_alpha != 32 or
                config.perception_rank != 8 or config.perception_alpha != 16):
            raise ValueError("HF backend differs from the fixed joint-affect contract")
        if not torch.cuda.is_available():
            raise RuntimeError("Real HF training requires the explicitly allocated CUDA devices")
        self.config, self.fusion_device, self.media_audit = config, config.second_device, []
        self.shared_omni = (Path(config.vision_model_path).resolve() == Path(config.second_model_path).resolve() and
                            config.vision_device == config.second_device)
        if self.shared_omni:
            model, processor, info = self._load_stream(config.second_model_path, config.second_device, "both")
            self.vision_model, self.second_model = model, model
            self.vision_processor, self.second_processor = processor, processor
            self.vision_info, self.second_info = info, info
        else:
            self.vision_model, self.vision_processor, self.vision_info = self._load_stream(
                config.vision_model_path, config.vision_device, "vision")
            self.second_model, self.second_processor, self.second_info = self._load_stream(
                config.second_model_path, config.second_device, "second")
        self.vision_hidden_size = self.vision_info["hidden_size"]
        self.second_hidden_size = self.second_info["hidden_size"]
        self.audit_config = {"settings": asdict(config), "shared_omni": self.shared_omni,
            "vision": self.vision_info, "second": self.second_info,
            "init_second_from": str(self.init_second_checkpoint) if self.init_second_checkpoint else None}
        self._roles = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if ".audio_tower." in name or ".audio_encoder." in name:
                category = "audio_perception" if "lora_" in name else "audio_connector"
            elif ".visual." in name or ".vision_tower." in name:
                category = "vision_perception" if "lora_" in name else "vision_connector"
            elif "lora_" in name:
                category = "language" if self.shared_omni else "vision_language" if name.startswith("vision_model.") else "second_language"
            else:
                raise RuntimeError("Unexpected trainable base parameter: " + name)
            self._roles[name] = category

    def parameter_group(self, name):
        if name not in self._roles:
            raise ValueError("Unclassified trainable HF parameter: " + name)
        return self._roles[name]

    def _load_stream(self, path, device, role):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (AutoConfig, AutoProcessor, BitsAndBytesConfig,
            Qwen2_5OmniThinkerForConditionalGeneration, Qwen3VLForConditionalGeneration)
        cfg = AutoConfig.from_pretrained(path, local_files_only=True)
        is_qwen3 = cfg.model_type == "qwen3_vl"
        if is_qwen3 and role != "vision":
            raise ValueError("Qwen3VL cannot supply the isolated audio stream")
        cls = Qwen3VLForConditionalGeneration if is_qwen3 else Qwen2_5OmniThinkerForConditionalGeneration
        model_cfg = cfg if is_qwen3 else getattr(cfg, "thinker_config", cfg)
        # Meta inventory never loads weights and gives exact full module names
        # before the quantizer decides which dense connectors to leave intact.
        with torch.device("meta"):
            skeleton = cls(model_cfg)
        plan = module_plan(skeleton, role, self.config.perception_top_blocks)
        del skeleton
        model, loading = cls.from_pretrained(path, config=model_cfg, local_files_only=True,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": device},
            output_loading_info=True,
            attn_implementation="sdpa", quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=plan["connectors"] + ["lm_head"]))
        if any(loading.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError("The raw HF backbone did not load completely: " + str(loading))
        if not getattr(model, "is_loaded_in_4bit", False):
            raise RuntimeError("HF backbone did not load with the fixed NF4 configuration")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=self.config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False})
        targets = plan["decoder"] + plan["perception"]
        model = get_peft_model(model, LoraConfig(r=self.config.decoder_rank,
            lora_alpha=self.config.decoder_alpha, lora_dropout=self.config.lora_dropout,
            target_modules=targets, rank_pattern={n: self.config.perception_rank for n in plan["perception"]},
            alpha_pattern={n: self.config.perception_alpha for n in plan["perception"]}, bias="none"))
        base = model.get_base_model()
        for name in targets:
            module = base.get_submodule(name)
            perception = name in plan["perception"]
            if (module.r.get("default") != (self.config.perception_rank if perception else self.config.decoder_rank) or
                    module.lora_alpha.get("default") != (self.config.perception_alpha if perception else self.config.decoder_alpha)):
                raise RuntimeError("PEFT did not honor the exact rank/alpha module policy: " + name)
        for name in plan["connectors"]:
            connector = base.get_submodule(name)
            if any("4bit" in type(m).__name__.lower() for m in connector.modules()):
                raise RuntimeError("Dense connector was accidentally quantized: " + name)
            # FP32 master parameters, BF16 operators under autocast: small
            # learning-rate updates must not disappear below a BF16 ULP.
            connector.to(dtype=torch.float32)
            for parameter in connector.parameters():
                parameter.requires_grad_(True)
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.data = parameter.data.float()
            if parameter.requires_grad and "lora_" not in name and not any(
                    ("." + module + ".") in ("." + name) for module in plan["connectors"]):
                raise RuntimeError("A base parameter outside the fixed connectors became trainable: " + name)
        base.config.use_cache = False
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        text_cfg = model_cfg.get_text_config()
        info = {"kind": "qwen3_vl" if is_qwen3 else "omni", "hidden_size": text_cfg.hidden_size,
            "plan": plan, "model_path": str(Path(path).resolve()), "device": str(device),
            "quantization": "nf4_double_quant_bfloat16", "connector_master_dtype": "float32",
            "gradient_checkpointing": self.config.gradient_checkpointing,
            "loading_missing_keys": [], "loading_mismatched_keys": [],
            "loading_unused_keys": len(loading.get("unexpected_keys", []))}
        return model, processor, info

    def _media_path(self, row):
        root = Path(self.config.dataset_root).resolve()
        path = (root / row["video"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Missing or escaped original media")
        return path

    def _decode_video(self, path):
        import av
        import numpy as np
        with av.open(str(path), mode="r") as container:
            stream = container.streams.video[0]
            fps, total = float(stream.average_rate), int(stream.frames or 0)
            if total <= 0:
                total = sum(1 for _ in container.decode(video=0))
        if not math.isfinite(fps) or fps <= 0 or total < self.config.min_frames:
            raise ValueError("Visual stream has insufficient reliable sampling metadata")
        count = min(total, max(self.config.min_frames,
            min(self.config.max_frames, int(total / fps * self.config.fps))))
        indices = np.linspace(0, total - 1, count).round().astype(np.int64)
        wanted, frames, timestamps = set(indices.tolist()), [], []
        decoded = 0
        with av.open(str(path), mode="r") as container:
            for i, frame in enumerate(container.decode(video=0)):
                decoded = i + 1
                if i in wanted:
                    frames.append(frame.to_ndarray(format="rgb24"))
                    timestamps.append(float(frame.time) if frame.time is not None else None)
        if decoded != total or len(frames) != len(indices):
            raise ValueError("Visual sampling metadata changed during decoding")
        return np.stack(frames), {"total_num_frames": total, "fps": fps, "frames_indices": indices.tolist()}, {
            "source_frames": total, "source_fps": fps, "sample_indices": indices.tolist(),
            "sampled_frames": count, "source_frame_timestamps": timestamps,
            "source_height": int(frames[0].shape[0]), "source_width": int(frames[0].shape[1]), "audio_decoded": False}

    def _audio(self, path, row):
        if self.config.audio_cache_dir:
            import soundfile as sf
            root = Path(self.config.audio_cache_dir).resolve()
            cached = (root / "audio" / Path(row["video"]).with_suffix(".flac")).resolve()
            if not cached.is_relative_to(root):
                raise ValueError("Escaped audio cache path")
            if cached.is_file():
                wave, rate = sf.read(cached, dtype="float32")
                if rate != self.config.audio_sampling_rate or wave.ndim != 1 or not wave.size:
                    raise ValueError("Existing audio cache violates mono 16kHz policy")
                return wave
        from rl.omni_io import audio_from_video
        return audio_from_video(path, self.config.audio_sampling_rate)

    def prepare_inputs(self, row, mode):
        processor = self.vision_processor if mode == "vision" else self.second_processor
        info = self.vision_info if mode == "vision" else self.second_info
        device = self.config.vision_device if mode == "vision" else self.config.second_device
        # Text-only inputs do not require or open the corresponding media.
        path = self._media_path(row) if mode != "text" else None
        messages = isolated_messages(row, mode, path)
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if path is not None and str(path) in rendered:
            raise ValueError("A media path leaked into the tokenized instructions")
        kwargs = {"text": rendered, "padding": True, "return_tensors": "pt",
                  "text_kwargs": {"add_special_tokens": False}}
        audit = {"mode": mode, "supplied_text": mode == "text", "gold_or_id_in_prompt": False}
        if mode == "vision":
            frames, metadata, geometry = self._decode_video(path)
            n = len(frames)
            vp = getattr(processor, "video_processor", None) or processor.image_processor
            patch, temporal, merge = int(vp.patch_size), int(vp.temporal_patch_size), int(vp.merge_size)
            if info["kind"] == "qwen3_vl":
                size = {"shortest_edge": (patch * merge) ** 2 * math.ceil(n / temporal) * temporal,
                        "longest_edge": n * self.config.pixels_per_frame}
                kwargs["return_mm_token_type_ids"] = True
            else:
                size = {"shortest_edge": self.config.pixels_per_frame, "longest_edge": self.config.pixels_per_frame}
            kwargs.update(videos=[frames], videos_kwargs={"do_sample_frames": False,
                "return_metadata": True, "video_metadata": [metadata], "size": size,
                "fps": self.config.fps})
            audit.update(geometry)
        elif mode == "audio":
            kwargs.update(audio=self._audio(path, row),
                          audio_kwargs={"sampling_rate": self.config.audio_sampling_rate})
        inputs = {k: v for k, v in processor(**kwargs).items() if v is not None}
        # Omni preserves the video processor's auxiliary VideoMetadata object;
        # Qwen3 consumes it while inserting timestamps. It is never a forward
        # argument. Check it against our own decoding before removing it.
        returned_metadata = inputs.pop("video_metadata", None)
        if returned_metadata is not None:
            if mode != "vision" or len(returned_metadata) != 1:
                raise ValueError("Unexpected isolated processor video metadata")
            returned = returned_metadata[0]
            returned_indices = returned["frames_indices"]
            if hasattr(returned_indices, "tolist"):
                returned_indices = returned_indices.tolist()
            if (returned["total_num_frames"] != metadata["total_num_frames"] or
                    not math.isclose(float(returned["fps"]), metadata["fps"], rel_tol=1e-9) or
                    list(returned_indices) != metadata["frames_indices"]):
                raise ValueError("Processor video metadata differs from our fixed frame selection")
            audit["processor_video_metadata"] = {"total_num_frames": int(returned["total_num_frames"]),
                "fps": float(returned["fps"]), "frames_indices": list(returned_indices)}
        allowed = {"input_ids", "attention_mask", "pixel_values_videos", "video_grid_thw",
            "video_second_per_grid", "mm_token_type_ids", "input_features", "feature_attention_mask",
            "audio_feature_lengths"}
        if set(inputs) - allowed:
            raise ValueError("Unrecognized isolated processor fields: " + str(sorted(set(inputs) - allowed)))
        # HF 5.8 Qwen3 may retain token lists when text_kwargs is explicit,
        # despite a top-level return_tensors='pt'. Normalize known integer
        # fields after removing metadata; never tensorize auxiliary objects.
        for key in ("input_ids", "attention_mask", "mm_token_type_ids", "video_grid_thw"):
            if key in inputs and not torch.is_tensor(inputs[key]):
                inputs[key] = torch.as_tensor(inputs[key], dtype=torch.long)
        tokens = int(inputs["input_ids"].shape[-1])
        if inputs["input_ids"].shape[0] != 1 or not 0 < tokens <= self.config.max_prompt_tokens:
            raise ValueError("Untruncated isolated input exceeds the fixed token budget")
        video_keys = {"pixel_values_videos", "video_grid_thw"}
        audio_keys = {"input_features", "feature_attention_mask"}
        if ((mode == "vision") != (video_keys <= set(inputs)) or
                (mode == "audio") != (audio_keys <= set(inputs)) or
                (mode != "vision" and bool(video_keys & set(inputs))) or
                (mode != "audio" and bool(audio_keys & set(inputs))) or
                "pixel_values" in inputs or "image_grid_thw" in inputs):
            raise ValueError("The isolated processor mixed input modalities")
        base = (self.vision_model if mode == "vision" else self.second_model).get_base_model()
        if mode == "vision":
            mask = inputs["input_ids"] == base.config.video_token_id
            grid = inputs["video_grid_thw"]
            t, h, w = map(int, grid[0].tolist())
            height, width = patch * h, patch * w
            if (grid.shape != (1, 3) or t != math.ceil(audit["sampled_frames"] / temporal) or
                    temporal * t > self.config.max_frames or height * width > self.config.pixels_per_frame):
                raise ValueError("Actual video geometry exceeds the fixed media budget")
            expected = t * h * w // (merge ** 2)
            if int(mask.sum()) != expected or expected <= 0:
                raise ValueError("Visual placeholders disagree with the actual grid")
            if info["kind"] == "qwen3_vl":
                mm = inputs.get("mm_token_type_ids")
                if mm is None or mm.shape != inputs["input_ids"].shape or not (mm[mask] == 2).all():
                    raise ValueError("Qwen3 multimodal token types disagree with the visual positions")
            audit.update(video_grid_thw=grid.tolist(), processed_height=height,
                processed_width=width, pixels_per_frame=height * width,
                processed_frames_including_padding=t * temporal)
        elif mode == "audio":
            mask = inputs["input_ids"] == base.config.audio_token_id
        else:
            utterance = row["text_content"]
            marker = "\n<utterance>\n"
            start = rendered.index(marker) + len(marker)
            if rendered[start:start + len(utterance)] != utterance:
                raise ValueError("Rendered text differs from the supplied utterance")
            encoded = processor.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
            if encoded["input_ids"] != inputs["input_ids"][0].tolist():
                raise ValueError("Text offsets do not align to the actual processed tokens")
            mask = torch.tensor([[a < start + len(utterance) and b > start for a, b in encoded["offset_mapping"]]], dtype=torch.bool)
        if not mask.any():
            raise ValueError("No isolated modality tokens were selected")
        if info["kind"] == "omni":
            inputs["use_audio_in_video"] = False
        audit.update(input_tokens=tokens, modal_tokens=int(mask.sum()), processor_keys=sorted(inputs))
        # Preserve the full preflight audit without retaining every training
        # row's decoded timestamp list for all later epochs.
        if len(self.media_audit) < 256:
            self.media_audit.append(audit)
        inputs = {k: v.to(device=device, dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point()
                  else v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        return inputs, mask.to(device), audit

    def _encode(self, row, mode):
        inputs, mask, audit = self.prepare_inputs(row, mode)
        model = self.vision_model if mode == "vision" else self.second_model
        info = self.vision_info if mode == "vision" else self.second_info
        base = model.get_base_model()
        if hasattr(base, "rope_deltas"):
            base.rope_deltas = None
        if hasattr(base, "model") and hasattr(base.model, "rope_deltas"):
            base.model.rope_deltas = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if info["kind"] == "qwen3_vl":
                # The base model returns final states directly, before lm_head.
                hidden = base.model(**inputs, use_cache=False, return_dict=True,
                                    output_hidden_states=False).last_hidden_state
            else:
                # Omni 5.8 merges media in its causal-LM wrapper. Capture the
                # final states WITH gradients and project zero positions: an
                # empty [B,0,vocab] result allocates no full vocabulary tensor.
                hidden = capture_omni_states(model, inputs)
        if hidden.shape[:2] != mask.shape or hidden.shape[0] != 1:
            raise RuntimeError("HF states and isolated token mask do not align")
        selected = hidden[:, mask[0], :]
        if self.training and not selected.requires_grad:
            raise RuntimeError("The trainable modality path lost its gradients")
        return TokenBatch(selected, audit=audit).checked()

    def encode_vision(self, row):
        return self._encode(row, "vision")

    def encode_second(self, row):
        if row["protocol"] not in ("VT", "VA"):
            raise ValueError("Unsupported original protocol")
        return self._encode(row, "text" if row["protocol"] == "VT" else "audio")


def capture_omni_states(model, inputs):
    """Keep final-token gradients while allocating zero vocabulary positions."""
    captured = []
    def capture(module, args):
        captured.append(args[0])
        return (args[0][:, :0, :], *args[1:])
    handle = model.get_base_model().get_output_embeddings().register_forward_pre_hook(capture)
    try:
        model(**inputs, use_cache=False, return_dict=True, output_hidden_states=False)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("Could not capture one differentiable Omni final-state sequence")
    return captured[0]
