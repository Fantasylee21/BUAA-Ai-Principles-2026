import csv
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from rl.grounded_uncertainty import (RobustSampler, baseline_csv, categorical_kl, eligible,
                                     load_training, main_prefix, pair_loss, settings_from)
from rl.omni_io import task_messages


class GroundedContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_root = Path(os.environ.get("BUAA_DATA_ROOT", Path(__file__).resolve().parents[1] / "Dataset"))
        if not (data_root / "train.jsonl").is_file():
            raise unittest.SkipTest("Original training data required for grounded sampler contracts")
        cls.train = load_training(data_root / "train.jsonl")
        cls.predictions = {r["sample_id"]: "no" for r in cls.train}

    def test_sampler_quotas_pair_draw_ids_and_resume(self):
        from collections import Counter
        sampler = RobustSampler(self.train, self.predictions, 42)
        restarted = RobustSampler(self.train, self.predictions, 42)
        for step in (0, 1, 39, 40, 79, 199):
            batch = sampler.batch(step)
            self.assertEqual(batch, restarted.batch(step))
            self.assertEqual(len(batch), 16)
            self.assertEqual(len({s["draw_id"] for s in batch}), 16)
            self.assertEqual(Counter(s["row"]["sample_type"] for s in batch), {"Conflict": 8, "Aligned": 8})
            self.assertEqual(Counter(s["row"]["protocol"] for s in batch), {"VT": 8, "VA": 8})
            self.assertEqual(sum(s["row"]["sample_type"] == "Conflict" and s["row"]["natural"] for s in batch), 6)
            paired = [s["row"] for s in batch if s["selection"] == "paired"]
            # A hard draw can coincidentally repeat a paired ID; the forced slots still exist.
            self.assertGreaterEqual(len(paired), 2)
            groups = {r["split_group_id"] for r in paired}
            self.assertEqual(len(groups), 1)
            self.assertEqual({r["sample_type"] for r in paired}, {"Conflict", "Aligned"})

    def test_sampler_rejects_validation_membership(self):
        wrong = dict(self.predictions)
        wrong["not-a-training-id"] = "yes"
        with self.assertRaises(ValueError):
            RobustSampler(self.train, wrong, 42)

    def test_rewards_reverse_wrong_confident_binary_gradient(self):
        import torch
        scores = torch.tensor([-12., 0.], dtype=torch.float64, requires_grad=True)
        old = scores.detach().clone().requires_grad_(True)
        reward = torch.tensor([1., 0.], dtype=torch.float64)
        loss = pair_loss(scores, old, reward)
        loss.backward()
        self.assertAlmostEqual(float(loss.detach()), 4.)
        self.assertLess(float(scores.grad[0]), -1.)
        self.assertGreater(float(scores.grad[1]), 1.)
        self.assertIsNone(old.grad)

    def test_multiclass_gradient_only_compares_gold_against_wrong(self):
        import torch
        scores = torch.tensor([4., -8., 2.], dtype=torch.float64, requires_grad=True)
        old = scores.detach().clone()
        pair_loss(scores, old, torch.tensor([0., 1., 0.], dtype=torch.float64)).backward()
        self.assertLess(float(scores.grad[1]), 0)
        self.assertGreater(float(scores.grad[0]), 0)
        self.assertGreater(float(scores.grad[2]), 0)
        self.assertAlmostEqual(float(scores.grad.sum()), 0)
        self.assertAlmostEqual(float(categorical_kl(scores, scores.detach()).detach()), 0)

    def test_auxiliary_prompts_do_not_leak_labels_or_other_modalities(self):
        for protocol in ("VT", "VA"):
            row = next(r for r in self.train if r["protocol"] == protocol)
            for mode in ("vision", "text" if protocol == "VT" else "audio"):
                messages = task_messages(row, "/test/input.mp4", mode)
                content = messages[1]["content"]
                self.assertEqual([p["type"] for p in content if p["type"] != "text"],
                                 ["video"] if mode == "vision" else ["audio"] if mode == "audio" else [])
                rendered = json.dumps(messages, ensure_ascii=False)
                self.assertNotIn(row["sample_id"], rendered)
                self.assertNotIn("Conflict", rendered)
                self.assertNotIn("view_targets", rendered)
                if mode != "text":
                    self.assertNotIn(row["text_content"], rendered)

    def test_candidate_gates_require_exact_frozen_emotions(self):
        base = {"score": .7}
        candidate = {"score": .72, "counts": {"natural_conflict": {"correct": 10},
            "va_conflict": {"correct": 4}, "aligned_false_positive": {"correct": 28}},
            "emotion_changes": 0, "invalid_outputs": 0, "inconsistent_questions": 0}
        self.assertTrue(eligible(base, candidate)["eligible"])
        for change in ({"emotion_changes": 1}, {"invalid_outputs": 1}, {"score": .714}):
            self.assertFalse(eligible(base, {**candidate, **change})["eligible"])

    def test_fixed_configuration_and_single_decision_prefix(self):
        self.assertEqual(settings_from()["steps"], 200)
        self.assertEqual(main_prefix("negative"), '{"pred_emotion":"negative","is_uncertain":"')
        with self.assertRaises(ValueError):
            main_prefix("gold-label-not-predicted")

    def test_baseline_ids_and_emotions_are_checked(self):
        rows = [{"sample_id": str(i), "pred_emotion": "positive", "is_uncertain": "no", "clarifying_question": "none"} for i in range(734)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "baseline.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(baseline_csv(path, rows, check_sha=False), rows)
            with self.assertRaises(ValueError):
                baseline_csv(path, list(reversed(rows)), check_sha=False)


HAS_STACK = all(importlib.util.find_spec(n) for n in ("torch", "peft", "transformers"))


@unittest.skipUnless(HAS_STACK, "Real PEFT/torch stack required")
class GroundedRuntimeTests(unittest.TestCase):
    def make_policy(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import GPT2Config, GPT2LMHeadModel
        from rl.grounded_runtime import GroundedPolicy
        torch.manual_seed(19)
        base = GPT2LMHeadModel(GPT2Config(vocab_size=24, n_embd=16, n_layer=1, n_head=2,
                                        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
        config = LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], fan_in_fan_out=True,
                            task_type="CAUSAL_LM", lora_dropout=0)
        model = get_peft_model(base, config, adapter_name="policy")
        for name in ("reference", "old"):
            model.add_adapter(name, copy.deepcopy(config))
        policy = object.__new__(GroundedPolicy)
        policy.model, policy.torch, policy.training = model, torch, True
        policy.settings = settings_from()
        policy.value_ids = {"full": [7, 8], "vision": [9, 10, 11]}
        policy.processor = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *a, **kw: [2, 3]))
        model.set_adapter("policy")
        policy._set_trainability()
        policy.snapshot_old()
        return policy

    def test_one_position_logits_match_native_forward_and_gradient(self):
        import torch
        policy = self.make_policy()
        prefix = {"input_ids": torch.tensor([[1, 4, 5]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
        inputs = policy.extend(prefix, main_prefix("negative"))
        policy.training_mode()
        full = policy.model(**inputs, use_cache=False).logits[0, -1, [7, 8]].float()
        full.sum().backward()
        expected = {n: p.grad.clone() for n, p in policy.model.named_parameters() if p.requires_grad}
        policy.model.zero_grad(set_to_none=True)
        observed = []
        hook = policy.model.get_output_embeddings().register_forward_hook(lambda m, args, out: observed.append(out.shape[1]))
        try:
            actual = policy.value_scores(prefix, {}, "full", "negative")
        finally:
            hook.remove()
        self.assertEqual(observed, [1])
        self.assertTrue(torch.allclose(actual, full, atol=1e-7, rtol=1e-6))
        actual.sum().backward()
        for name, parameter in policy.model.named_parameters():
            if parameter.requires_grad:
                self.assertTrue(torch.allclose(parameter.grad, expected[name], atol=1e-7, rtol=1e-6), name)
            elif ".reference." in name or ".old." in name:
                self.assertIsNone(parameter.grad)

    def test_checkpoint_keeps_block_old_and_exact_next_optimizer_step(self):
        import torch
        from rl.grounded_store import save_checkpoint, load_checkpoint, validate_checkpoint
        from rl.state import adapter_digest
        policy = self.make_policy()
        params = [p for p in policy.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=.01, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        prefix = {"input_ids": torch.tensor([[1, 4, 5]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
        with torch.no_grad():
            old_scores = policy.value_scores(prefix, {}, "full", "negative", adapter="old")
        frozen_digest = adapter_digest(policy, "reference")
        def update():
            optimizer.zero_grad(set_to_none=True)
            score = policy.value_scores(prefix, {}, "full", "negative")
            loss = pair_loss(score, old_scores, torch.tensor([1., 0.]))
            loss.backward()
            optimizer.step()
            scheduler.step()
            return float(loss.detach())
        update()
        context = {"identity_hash": "grounded-real-state-test"}
        pool = {"train-only-id": "no"}
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            checkpoint = save_checkpoint(run, policy, optimizer, scheduler, context,
                                         seed=42, step=20, old_block=0, pool=pool)
            marker = validate_checkpoint(checkpoint, context)
            self.assertNotEqual(marker["old_digest"], marker["policy_digest"])
            expected_loss = update()
            expected_digest = adapter_digest(policy, "policy")
            policy.snapshot_old()  # A resume must restore the OLD block, not copy current policy.
            restored = load_checkpoint(policy, checkpoint, context, optimizer=optimizer, scheduler=scheduler)
            self.assertEqual(restored["pool"], pool)
            self.assertEqual(restored["old_block"], 0)
            self.assertEqual(adapter_digest(policy, "old"), marker["old_digest"])
            self.assertEqual(update(), expected_loss)
            self.assertEqual(adapter_digest(policy, "policy"), expected_digest)
            self.assertEqual(adapter_digest(policy, "reference"), frozen_digest)
            with self.assertRaises(ValueError):
                load_checkpoint(policy, checkpoint, {"identity_hash": "old-algorithm"})


if __name__ == "__main__":
    unittest.main()
