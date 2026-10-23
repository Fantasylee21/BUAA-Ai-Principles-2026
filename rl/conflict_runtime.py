"""Conflict-specific prompts and native decision logits over frozen perception."""
from __future__ import annotations

import json
from pathlib import Path

from .finite_actions import omni_settings
from .finite_runtime import FinitePolicy
from .grounded_uncertainty import VALUES, main_prefix
from .omni_runtime import OmniPolicy
from .reward import parse_completion, question_is_consistent


SYSTEM_PROMPT = (
    "You are a multimodal emotion analysis assistant. Determine whether the two "
    "provided modalities express conflicting emotions. Follow the requested JSON schema."
)
INSTRUCTION = (
    "Compare the emotion expressed by each of the two provided modalities. "
    "Set is_uncertain to yes when their expressed emotions conflict, and to no "
    "when they agree. Limited detail by itself does not establish a conflict. "
    "Preserve the supplied pred_emotion field when completing the answer. "
    "Return exactly one JSON object with pred_emotion equal to positive, negative, "
    "or neutral; is_uncertain equal to yes or no; and clarifying_question as a "
    "single string. For no, clarifying_question must be none. For yes, ask one "
    "neutral clarifying question addressed to the current user. Do not add Markdown."
)


def conflict_messages(row, video_path):
    """Only actual evidence and the protocol enter the prompt, never metadata."""
    if row["protocol"] == "VT":
        modality = "The input contains silent video frames and supplied transcript text."
        text = str(row.get("text_content", row.get("text", ""))).strip()
        evidence = "\nSupplied text: " + json.dumps(text, ensure_ascii=False)
    elif row["protocol"] == "VA":
        modality = "The input contains video frames and the video's embedded audio. No transcript text is provided."
        evidence = ""
    else:
        raise ValueError("Conflict task requires the actual VT or VA protocol")
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "video", "video": str(video_path)},
            {"type": "text", "text": modality + evidence + "\n" + INSTRUCTION}]},
    ]


def prepare_conflict_inputs(processor, row, data_root, settings, device="cuda:0"):
    """Reprocess media so new text cannot inherit stale positions or placeholders."""
    import torch
    from .omni_io import audio_from_video
    root = Path(data_root).resolve()
    path = (root / row["video"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Missing or escaped conflict-task media")
    use_audio = row["protocol"] == "VA"
    messages = conflict_messages(row, path)
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
        raise ValueError("Conflict prompt exceeds the untruncated token budget")
    if not {"pixel_values_videos", "video_grid_thw"} <= set(inputs):
        raise RuntimeError("Conflict input must contain video")
    audio_keys = {"input_features", "feature_attention_mask"}
    if use_audio and not audio_keys <= set(inputs):
        raise RuntimeError("VA conflict input requires actual audio")
    if not use_audio and audio_keys & set(inputs):
        raise RuntimeError("VT conflict input must not contain audio")
    return {k: v.to(device=device, dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point()
            else v.to(device=device) if torch.is_tensor(v) else v for k, v in inputs.items()}


class ConflictPolicy(FinitePolicy):
    def __init__(self, model_path, adapter_path, settings, *, training):
        # Reuse perception/runtime mechanics without loading old-policy adapters
        # or tokenizing six obsolete JSON action sequences.
        OmniPolicy.__init__(self, model_path, adapter_path, omni_settings(settings), training=training)
        tokenizer = self.processor.tokenizer
        ids = {mode: [tokenizer.encode(v, add_special_tokens=False) for v in values]
               for mode, values in VALUES.items()}
        if any(len(tokens) != 1 for values in ids.values() for tokens in values):
            raise RuntimeError("Each decision or diagnostic value must be a single token")
        self.value_ids = {mode: [v[0] for v in values] for mode, values in ids.items()}
        for emotion in VALUES["vision"]:
            prefix = main_prefix(emotion)
            for value in VALUES["full"]:
                if tokenizer.encode(prefix+value, add_special_tokens=False) != tokenizer.encode(prefix, add_special_tokens=False)+tokenizer.encode(value, add_special_tokens=False):
                    raise RuntimeError("Decision field crosses a value token boundary")

    def prepare_conflict(self, row, data_root, device="cuda:0"):
        return prepare_conflict_inputs(self.processor, row, data_root, self.settings, device)

    def original_task(self, row, data_root, device="cuda:0"):
        from .omni_io import prepare_omni_inputs
        return prepare_omni_inputs(self.processor, row, data_root, self.settings, device)

    def prepare_task(self, row, data_root, mode="full", device="cuda:0"):
        if mode == "full":
            return self.prepare_conflict(row, data_root, device)
        from .omni_io import prepare_task_inputs
        return prepare_task_inputs(self.processor, row, data_root, self.settings, mode, device)

    def validate_perception_reuse(self, prefix, original, features):
        """Validate fresh media processing against the immutable original cache.

        The store separately verifies all media, model, processor and producer
        hashes. Raw pixel/audio tensors were omitted from old cache blobs, so
        grids, masks, media tokens and all non-text positional metadata must agree.
        """
        torch = self.torch
        omitted = {"input_ids", "attention_mask", "pixel_values_videos", "input_features"}
        if set(prefix) != set(original):
            raise ValueError("New conflict input has a different media-field contract")
        for key in set(prefix) - omitted:
            left, right = prefix[key], original[key]
            same = torch.equal(left, right) if torch.is_tensor(left) and torch.is_tensor(right) else left == right
            if not same:
                raise ValueError(f"Perception reuse changed media metadata: {key}")
        expected = {"video"} | ({"audio"} if prefix["use_audio_in_video"] else set())
        if set(features) != expected or any(not torch.isfinite(v).all() for v in features.values()):
            raise ValueError("Perception feature modalities or values are invalid")
        config = self.model.get_base_model().config
        for modality in expected:
            token = getattr(config, modality+"_token_id", None)
            if token is None:
                raise ValueError(f"Omni config lacks {modality} placeholder token identity")
            fresh_count = int((prefix["input_ids"] == token).sum())
            old_count = int((original["input_ids"] == token).sum())
            if fresh_count != old_count or fresh_count != features[modality].shape[0]:
                raise ValueError(f"Perception reuse changed {modality} token/feature count")
        return True

    def extend(self, prefix, text):
        torch = self.torch
        ids = torch.tensor(self.processor.tokenizer.encode(text, add_special_tokens=False),
                           device=prefix["input_ids"].device, dtype=torch.long).unsqueeze(0)
        result = dict(prefix)
        result["input_ids"] = torch.cat([prefix["input_ids"], ids], -1)
        result["attention_mask"] = torch.cat([prefix["attention_mask"], torch.ones_like(ids)], -1)
        return result

    def value_scores(self, prefix, features, mode="full", emotion=None, *, adapter="policy"):
        suffix = main_prefix(emotion) if mode == "full" else '{"pred_emotion":"'
        inputs = self.extend(prefix, suffix)
        self.training_mode()
        self.clear_positions()
        device_type = inputs["input_ids"].device.type
        with self.adapter(adapter), self.cached_perception(features), self.suffix_projection(1) as projection, \
                self.torch.autocast(device_type, dtype=self.torch.bfloat16, enabled=device_type == "cuda"):
            output = self.model(**inputs, **projection, use_cache=False, return_dict=True)
        return output.logits[0, -1, self.value_ids[mode]].float()

    def answer_frozen_emotion(self, prefix, features, baseline, *, adapter="policy", question_prefix=None):
        with self.torch.no_grad():
            scores = self.value_scores(prefix, features, "full", baseline["pred_emotion"], adapter=adapter)
        uncertain = VALUES["full"][int(scores.argmax())]
        if uncertain == "no":
            question = "none"
        elif baseline["is_uncertain"] == "yes":
            question = baseline["clarifying_question"]
        else:
            if question_prefix is None:
                raise ValueError("New yes decisions require the frozen v4 original question prompt")
            selected = main_prefix(baseline["pred_emotion"])+uncertain+'","clarifying_question":'
            extended = self.extend(question_prefix, selected)
            with self.cached_perception(features):
                _, tail = self.generate(extended, adapter="reference")
            parsed = parse_completion(selected+tail)
            if parsed is None or not question_is_consistent(parsed) or parsed["pred_emotion"] != baseline["pred_emotion"] or parsed["is_uncertain"] != uncertain:
                return selected+tail, scores.detach().double().softmax(-1).tolist()
            question = parsed["clarifying_question"]
        answer = {"pred_emotion": baseline["pred_emotion"], "is_uncertain": uncertain, "clarifying_question": question}
        return json.dumps(answer, ensure_ascii=False, separators=(",", ":")), scores.detach().double().softmax(-1).tolist()
