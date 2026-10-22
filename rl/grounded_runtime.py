"""Shared Omni base, frozen reference, block-frozen old policy, native value logits."""
from __future__ import annotations

import json

from .finite_runtime import FinitePolicy
from .grounded_uncertainty import VALUES, main_prefix, pair_loss, categorical_kl
from .reward import parse_completion, question_is_consistent
from .state import adapter_digest


class GroundedPolicy(FinitePolicy):
    def __init__(self, model_path, adapter_path, settings, *, training):
        super().__init__(model_path, adapter_path, settings, training=training)
        self.model.load_adapter(adapter_path, adapter_name="old", is_trainable=False, local_files_only=True)
        # Loading another adapter creates new Dropout modules after OmniPolicy's setup.
        for module in self.model.modules():
            if isinstance(module, self.torch.nn.Dropout):
                module.p = 0.
        self._set_trainability()
        tokenizer = self.processor.tokenizer
        self.value_ids = {mode: [tokenizer.encode(v, add_special_tokens=False) for v in values]
                          for mode, values in VALUES.items()}
        if any(len(v) != 1 for ids in self.value_ids.values() for v in ids):
            raise RuntimeError("Approved task values must each be exactly one token")
        self.value_ids = {k: [v[0] for v in ids] for k, ids in self.value_ids.items()}
        for emotion in VALUES["vision"]:
            prefix = main_prefix(emotion)
            for value in VALUES["full"]:
                if tokenizer.encode(prefix+value, add_special_tokens=False) != tokenizer.encode(prefix, add_special_tokens=False)+tokenizer.encode(value, add_special_tokens=False):
                    raise RuntimeError("Decision field crosses an action token boundary")

    def snapshot_old(self):
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        weights = {k: v.detach().clone() for k, v in get_peft_model_state_dict(self.model, adapter_name="policy").items()}
        set_peft_model_state_dict(self.model, weights, adapter_name="old")
        self._set_trainability()
        if adapter_digest(self, "old") != adapter_digest(self, "policy"):
            raise RuntimeError("Block old adapter did not exactly copy the policy")

    def prepare_task(self, row, data_root, mode):
        from .omni_io import prepare_task_inputs
        return prepare_task_inputs(self.processor, row, data_root, self.settings, mode)

    def extend(self, prefix, text):
        torch = self.torch
        ids = torch.tensor(self.processor.tokenizer.encode(text, add_special_tokens=False),
                           device=prefix["input_ids"].device, dtype=torch.long).unsqueeze(0)
        result = dict(prefix)
        result["input_ids"] = torch.cat([prefix["input_ids"], ids], -1)
        result["attention_mask"] = torch.cat([prefix["attention_mask"], torch.ones_like(ids)], -1)
        return result

    def value_scores(self, prefix, features, mode, emotion=None, *, adapter="policy"):
        suffix = main_prefix(emotion) if mode == "full" else '{"pred_emotion":"'
        inputs = self.extend(prefix, suffix)
        self.training_mode()
        self.clear_positions()
        device_type = inputs["input_ids"].device.type
        with self.adapter(adapter), self.cached_perception(features), self.suffix_projection(1) as projection, \
                self.torch.autocast(device_type, dtype=self.torch.bfloat16, enabled=device_type == "cuda"):
            output = self.model(**inputs, **projection, use_cache=False, return_dict=True)
        return output.logits[0, -1, self.value_ids[mode]].float()

    def backward_task(self, prefix, features, mode, emotion, target, *, old, reference=None,
                      weight=1., kl_weight=0.):
        torch = self.torch
        scores = self.value_scores(prefix, features, mode, emotion)
        old = torch.as_tensor(old, device=scores.device, dtype=torch.float32)
        rewards = torch.tensor([float(v == target) for v in VALUES[mode]], device=scores.device)
        pair = pair_loss(scores, old, rewards, self.settings["eta"])
        kl = categorical_kl(scores, torch.as_tensor(reference, device=scores.device)) if reference is not None else scores.sum()*0
        loss = weight*pair + kl_weight*kl
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite grounded objective")
        loss.backward()
        return {"pair_loss": float(pair.detach()), "loss": float(loss.detach()), "kl": float(kl.detach()),
                "expected_reward": float(scores.detach().double().softmax(-1)[VALUES[mode].index(target)]),
                "decision": VALUES[mode][int(scores.detach().argmax())],
                "scores": scores.detach().cpu().tolist()}

    def answer_frozen_emotion(self, prefix, features, baseline, *, adapter="policy"):
        torch = self.torch
        with torch.no_grad():
            scores = self.value_scores(prefix, features, "full", baseline["pred_emotion"], adapter=adapter)
        uncertain = VALUES["full"][int(scores.argmax())]
        if uncertain == "no":
            question = "none"
        elif baseline["is_uncertain"] == "yes":
            question = baseline["clarifying_question"]
        else:
            selected = main_prefix(baseline["pred_emotion"])+uncertain+'","clarifying_question":'
            extended = self.extend(prefix, selected)
            with self.cached_perception(features):
                _, tail = self.generate(extended, adapter="reference")
            parsed = parse_completion(selected+tail)
            if parsed is None or not question_is_consistent(parsed) or parsed["pred_emotion"] != baseline["pred_emotion"] or parsed["is_uncertain"] != uncertain:
                return selected+tail, scores.detach().double().softmax(-1).tolist()
            question = parsed["clarifying_question"]
        answer = {"pred_emotion": baseline["pred_emotion"], "is_uncertain": uncertain, "clarifying_question": question}
        return json.dumps(answer, ensure_ascii=False, separators=(",", ":")), scores.detach().double().softmax(-1).tolist()
