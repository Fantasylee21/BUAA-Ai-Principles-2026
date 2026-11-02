"""Omni finite-action probabilities, frozen perception cache and exact replay VJP."""
from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace

from .finite_actions import ACTIONS, PREFIXES, objective, omni_settings
from .omni_runtime import OmniPolicy, batch_for_completions
from .reward import parse_completion, question_is_consistent


class FinitePolicy(OmniPolicy):
    def __init__(self, model_path, sft_adapter_path, settings, **kwargs):
        super().__init__(model_path, sft_adapter_path, omni_settings(settings), **kwargs)
        tokenizer = self.processor.tokenizer
        self.action_ids = [self.torch.tensor(tokenizer.encode(p, add_special_tokens=False),
                                           device="cuda:0", dtype=self.torch.long) for p in PREFIXES]
        if any(tokenizer.decode(ids.tolist(), clean_up_tokenization_spaces=False) != p
               for ids, p in zip(self.action_ids, PREFIXES)):
            raise RuntimeError("Action tokenization must round-trip exactly")

    def perception(self, prefix):
        """Capture the exact public feature outputs consumed by Thinker.forward.

        Original masks/grids remain in the prefix: Omni still computes its own
        multimodal RoPE and placeholder checks. Only frozen encoder calls skip.
        """
        torch = self.torch
        self.model.eval()
        base = self.model.get_base_model()
        if any(p.requires_grad for name, p in self.model.named_parameters()
               if any(s in name.lower() for s in ("visual", "audio", "projector"))):
            raise RuntimeError("Cannot cache trainable perception modules")
        result = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if "pixel_values_videos" in prefix:
                result["video"] = base.get_video_features(
                    prefix["pixel_values_videos"], prefix["video_grid_thw"], return_dict=True).pooler_output.detach()
            if "input_features" in prefix:
                result["audio"] = base.get_audio_features(
                    prefix["input_features"], feature_attention_mask=prefix["feature_attention_mask"],
                    return_dict=True).last_hidden_state.detach()
        return result

    @contextlib.contextmanager
    def cached_perception(self, features):
        if features is None:
            yield
            return
        base = self.model.get_base_model()
        methods = {}
        try:
            if "video" in features:
                methods["get_video_features"] = base.get_video_features

                def video(pixel_values_videos, video_grid_thw, **kwargs):
                    # A canonical input has one video, repeated once per candidate.
                    count = video_grid_thw.shape[0]
                    return SimpleNamespace(pooler_output=features["video"].repeat(count, 1))

                base.get_video_features = video
            if "audio" in features:
                methods["get_audio_features"] = base.get_audio_features

                def audio(input_features, feature_attention_mask=None, **kwargs):
                    if feature_attention_mask is None:
                        raise RuntimeError("Cached audio requires the original mask")
                    count = feature_attention_mask.shape[0]
                    return SimpleNamespace(last_hidden_state=features["audio"].repeat(count, 1))

                base.get_audio_features = audio
            yield
        finally:
            for name, method in methods.items():
                setattr(base, name, method)

    def training_mode(self):
        self.model.train(self.training)
        for name, module in self.model.named_modules():
            if any(s in name.lower() for s in ("visual", "audio", "projector")):
                module.eval()

    def chunk_scores(self, prefix, indices):
        inputs, ids, mask = batch_for_completions(
            self.torch, prefix, [self.action_ids[i] for i in indices], self.pad_id)
        return (self.log_probabilities(inputs, ids) * mask).sum(-1)

    def scores(self, prefix, features=None, *, adapter="policy", micro_batch_size=None):
        # Larger chunks are diagnostic only. Production config enforces one:
        # actual Omni NF4/BF16 scores changed with candidate batch size.
        micro = micro_batch_size or self.settings["candidate_micro_batch_size"]
        if micro not in (1, 2, 3, 6):
            raise ValueError("Candidate microbatch must divide six")
        self.training_mode()
        with self.adapter(adapter), self.cached_perception(features):
            return self.torch.cat([self.chunk_scores(prefix, range(i, min(6, i + micro)))
                                   for i in range(0, 6, micro)])

    def backward_actions(self, prefix, features, reference_log_probs, rewards, *, divisor):
        """Chain rule VJP: one detached six-score pass, then exact graph replay.

        Never reuse policy scores across an optimizer update. No sampled-action
        importance ratio or forced-answer GRPO estimator is involved.
        """
        torch = self.torch
        if divisor < 1:
            raise ValueError("Batch divisor must be positive")
        with torch.no_grad():
            first = self.scores(prefix, features)
        leaf = first.detach().double().requires_grad_(True)
        ref = torch.as_tensor(reference_log_probs, device=first.device, dtype=torch.float64)
        reward = torch.as_tensor(rewards, device=first.device, dtype=torch.float64)
        loss, metrics = objective(leaf, ref, reward, self.settings["beta"])
        derivative, = torch.autograd.grad(loss / divisor, leaf)
        self.training_mode()
        micro = self.settings["candidate_micro_batch_size"]
        with self.adapter("policy"), self.cached_perception(features):
            for start in range(0, 6, micro):
                indices = range(start, min(6, start + micro))
                replay = self.chunk_scores(prefix, indices)
                if not torch.allclose(replay.detach(), first[start:start + micro], atol=2e-3, rtol=1e-4):
                    raise RuntimeError("Two-pass scores changed without an optimizer update")
                (replay.double() * derivative[start:start + micro].detach()).sum().backward()
        return {"loss": float(loss.detach()),
                **{k: float(v.detach()) for k, v in metrics.items() if k != "probabilities"},
                "action": int(first.argmax()), "score_gradient_norm": float(derivative.norm() * divisor)}

    def answer(self, prefix, features, *, action=None, adapter="policy"):
        """Question text always comes from the frozen v4, conditional on the action."""
        torch = self.torch
        probabilities = None
        if action is None:
            with torch.no_grad():
                values = self.scores(prefix, features, adapter=adapter)
                probabilities = values.double().softmax(-1).tolist()
                action = int(values.argmax())
        emotion, uncertain = ACTIONS[action]
        if uncertain == "no":
            parsed = {"pred_emotion": emotion, "is_uncertain": uncertain, "clarifying_question": "none"}
        else:
            selected = self.action_ids[action]
            extended = dict(prefix)
            extended["input_ids"] = torch.cat([prefix["input_ids"], selected.unsqueeze(0)], dim=1)
            extended["attention_mask"] = torch.cat(
                [prefix["attention_mask"], torch.ones_like(selected).unsqueeze(0)], dim=1)
            with self.cached_perception(features):
                _, tail = self.generate(extended, adapter="reference")
            parsed = parse_completion(PREFIXES[action] + tail)
            if (parsed is None or not question_is_consistent(parsed)
                    or (parsed["pred_emotion"], parsed["is_uncertain"]) != ACTIONS[action]):
                # Preserve a diagnostically invalid output. Evaluation rejects
                # promotion and strict export fails; never fabricate a question.
                return PREFIXES[action] + tail, probabilities
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":")), probabilities


def pack_cache(torch, prefix, features):
    """Drop expensive raw media but retain triggers and all positional metadata."""
    packed = {}
    for key, value in prefix.items():
        if key in ("pixel_values_videos", "input_features"):
            packed[key] = torch.empty(0, dtype=value.dtype)
        else:
            packed[key] = value.detach().cpu() if torch.is_tensor(value) else value
    return {"prefix": packed, "features": {k: v.detach().cpu() for k, v in features.items()}}


def unpack_cache(torch, packed, device="cuda:0"):
    prefix = {k: v.to(device) if torch.is_tensor(v) else v for k, v in packed["prefix"].items()}
    features = {k: v.to(device) for k, v in packed["features"].items()}
    return prefix, features
