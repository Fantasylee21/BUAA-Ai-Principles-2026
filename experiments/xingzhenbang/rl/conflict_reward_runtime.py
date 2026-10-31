"""Trainable language adapter; frozen reference and frozen perception only."""
from __future__ import annotations

from .finite_runtime import FinitePolicy
from .grounded_uncertainty import main_prefix


class ConflictRewardPolicy(FinitePolicy):
    def __init__(self, model_path, adapter_path, settings, **kwargs):
        super().__init__(model_path, adapter_path, settings, **kwargs)
        tokenizer = self.processor.tokenizer
        ids = [tokenizer.encode(v, add_special_tokens=False) for v in ("yes", "no")]
        if any(len(i) != 1 for i in ids):
            raise ValueError("yes/no must be single tokens")
        self.value_ids = [i[0] for i in ids]
        for emotion in ("positive", "negative", "neutral"):
            prefix = main_prefix(emotion)
            for value in ("yes", "no"):
                if tokenizer.encode(prefix+value, add_special_tokens=False) != tokenizer.encode(prefix, add_special_tokens=False)+tokenizer.encode(value, add_special_tokens=False):
                    raise ValueError("Action token crosses the canonical prefix boundary")

    def extend(self, prefix, text):
        torch = self.torch
        ids = torch.tensor(self.processor.tokenizer.encode(text, add_special_tokens=False),
                           device=prefix["input_ids"].device, dtype=torch.long)[None]
        inputs = dict(prefix)
        inputs["input_ids"] = torch.cat([prefix["input_ids"], ids], -1)
        inputs["attention_mask"] = torch.cat([prefix["attention_mask"], torch.ones_like(ids)], -1)
        return inputs

    def value_scores(self, prefix, features, emotion, *, adapter="policy"):
        inputs = self.extend(prefix, main_prefix(emotion))
        self.training_mode()
        self.clear_positions()
        with self.adapter(adapter), self.cached_perception(features), self.suffix_projection(1) as projection, self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            output = self.model(**inputs, **projection, use_cache=False, return_dict=True)
        return output.logits[0, -1, self.value_ids].float()

    def log_odds(self, prefix, features, emotion, *, adapter="policy"):
        values = self.value_scores(prefix, features, emotion, adapter=adapter)
        return values[0]-values[1]
