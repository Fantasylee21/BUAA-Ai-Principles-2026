"""Regression checks for prompt isolation, provenance and exact policy recovery."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rl.conflict_cost import VERSION, settings_from
from rl.conflict_runtime import ConflictPolicy, conflict_messages
from rl.conflict_store import (TaskCache, load_checkpoint, merge_tasks, save_checkpoint,
                               save_task, task_key, validate_checkpoint, verify_source)
from rl.experiment import file_hash, fingerprint, write_json
from rl.grounded_recovery import EXECUTION
from rl.grounded_uncertainty import main_prefix
from rl.state import adapter_digest


def paired_rows():
    return [{"sample_id": f"group-{i}-{protocol}", "split_group_id": f"group-{i}",
             "protocol": protocol, "video": f"video-{i}.mp4", "sample_type": kind}
            for i in range(30) for protocol, kind in (("VT", "Conflict"), ("VA", "Aligned"))]


class ConflictStoreTests(unittest.TestCase):
    def test_prompt_uses_only_available_evidence_and_conflict_semantics(self):
        for protocol in ("VT", "VA"):
            row = {"protocol": protocol, "text_content": "visible transcript sentinel",
                   "sample_id": "private-id", "sample_type": "Conflict", "source_dataset": "gold-source",
                   "gt_label": "gold-emotion", "view_targets": {"vision": {"label": "gold-view"}}}
            messages = conflict_messages(row, "/allowed/video.mp4")
            rendered = json.dumps(messages)
            for forbidden in ("private-id", "gold-source", "gold-emotion", "gold-view", "view_targets"):
                self.assertNotIn(forbidden, rendered)
            self.assertEqual("visible transcript sentinel" in rendered, protocol == "VT")
            self.assertNotIn("insufficient for a reliable judgment", rendered)
            self.assertIn("express conflicting emotions", rendered)
            mutated = {**row, "sample_type": "Aligned", "gt_label": "another-label"}
            self.assertEqual(messages, conflict_messages(mutated, "/allowed/video.mp4"))

    def cache_fixture(self, run):
        import torch
        (run / "cache").mkdir()
        c = {"identity_hash": "current", "identity": {"cache_parent": {"identity_hash": "original"}},
             "train": [{"sample_id": "train-a"}], "val": [{"sample_id": "val-a"}]}
        for shard, split in enumerate(("train", "val")):
            row = c[split][0]
            packed = {"prefix": {"input_ids": torch.tensor([[1, 2]]),
                                 "attention_mask": torch.tensor([[1, 1]])},
                      "features": {"video": torch.ones(2, 3)}}
            entry = save_task(run, row, split, "full", packed, [.2, -.4],
                {"pred_emotion": "negative", "is_uncertain": "no", "clarifying_question": "none"},
                torch, c["identity_hash"])
            write_json(run / f"cache/shard-{shard}.json", {"identity_hash": c["identity_hash"], "entries": [entry]})
        return c

    def test_new_cache_requires_full_membership_split_identity_and_blob_hash(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            c = self.cache_fixture(run)
            manifest = merge_tasks(run, c)
            cache = TaskCache(run, c, torch)
            prefix, features = cache.load(c["train"][0], "train", device="cpu")
            self.assertTrue(torch.equal(prefix["input_ids"], torch.tensor([[1, 2]])))
            self.assertEqual(features["video"].shape, (2, 3))
            with self.assertRaisesRegex(ValueError, "crossed"):
                cache.load(c["train"][0], "val", device="cpu")
            entry = manifest["entries"][task_key("train-a", "full")]
            (run / "cache" / entry["file"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Corrupt"):
                TaskCache(run, c, torch).load(c["train"][0], "train", device="cpu")
            entry["split"] = "val"
            write_json(run / "cache/complete.json", manifest)
            with self.assertRaisesRegex(ValueError, "boundary"):
                TaskCache(run, c, torch)

    def test_reference_cache_rejects_old_algorithm_and_obsolete_auxiliary_entries(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            c = self.cache_fixture(run)
            manifest = merge_tasks(run, c)
            manifest["schema"] = "omni_grounded_uncertainty_rebel_v1"
            write_json(run / "cache/complete.json", manifest)
            with self.assertRaisesRegex(ValueError, "algorithm"):
                TaskCache(run, c, torch)
            with self.assertRaisesRegex(ValueError, "full tasks only"):
                save_task(run, c["train"][0], "train", "vision", {}, [1., 2., 3.], {}, torch, "current")

    def test_unjournaled_cache_blob_reuses_only_exact_regenerated_tensors(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "cache").mkdir()
            packed = {"prefix": {"input_ids": torch.tensor([[3, 4]]), "flag": True},
                      "features": {"video": torch.ones(2, 3)}}
            first = save_task(run, {"sample_id": "a"}, "train", "full", packed, [1., 2.], {}, torch, "same")
            repeat = save_task(run, {"sample_id": "a"}, "train", "full", copy.deepcopy(packed), [1., 2.], {}, torch, "same")
            self.assertEqual(first, repeat)
            packed["features"]["video"][0, 0] += 1
            with self.assertRaisesRegex(ValueError, "conflicting"):
                save_task(run, {"sample_id": "a"}, "train", "full", packed, [1., 2.], {}, torch, "same")
            self.assertEqual(file_hash(run / "cache" / first["file"]), first["sha256"])

    def test_source_reuse_rejects_different_model_and_changed_split(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            (source / "cache").mkdir()
            identity = {"source": "producer-source", "settings": settings_from(), "execution": EXECUTION,
                "software": {"torch": "same"}, "base": "base-v4", "sft": "adapter-v4", "media": {},
                "split_hashes": {}, "raw_split_hashes": {}, "baseline_csv_sha256": "baseline"}
            rows = {"train": [{"sample_id": "train-a"}], "val": [{"sample_id": "val-a"}]}
            old_hash = fingerprint(identity)
            entries = {task_key(r["sample_id"], "full"): {"split": split, "sample_id": r["sample_id"],
                "mode": "full", "identity_hash": old_hash, "file": task_key(r["sample_id"], "full")+".pt"}
                for split, records in rows.items() for r in records}
            write_json(source / "context.json", {"identity": identity, "identity_hash": old_hash, **rows})
            write_json(source / "cache/complete.json", {"identity_hash": old_hash, "entries": entries})
            with patch("rl.conflict_store.source_hash_at", return_value="producer-source"), \
                    patch("rl.conflict_store.perception_contract", return_value={"media": "same"}):
                self.assertEqual(verify_source(source, source, identity)["identity_hash"], old_hash)
                with self.assertRaisesRegex(ValueError, "base"):
                    verify_source(source, source, {**identity, "base": "different-model"})
                entries[task_key("train-a", "full")]["split"] = "val"
                write_json(source / "cache/complete.json", {"identity_hash": old_hash, "entries": entries})
                with self.assertRaisesRegex(ValueError, "boundary"):
                    verify_source(source, source, identity)

    def test_perception_reuse_requires_media_tokens_masks_and_feature_lengths(self):
        import torch
        policy = object.__new__(ConflictPolicy)
        policy.torch = torch
        policy.model = SimpleNamespace(get_base_model=lambda: SimpleNamespace(
            config=SimpleNamespace(video_token_id=6, audio_token_id=7)))
        original = {"input_ids": torch.tensor([[1, 6, 6, 7]]),
                    "attention_mask": torch.ones(1, 4, dtype=torch.long),
                    "pixel_values_videos": torch.empty(0), "input_features": torch.empty(0),
                    "video_grid_thw": torch.tensor([[1, 2, 4]]),
                    "feature_attention_mask": torch.tensor([[1, 1, 1]]), "use_audio_in_video": True}
        fresh = {**original, "input_ids": torch.tensor([[2, 3, 6, 6, 7]]),
                 "attention_mask": torch.ones(1, 5, dtype=torch.long),
                 "pixel_values_videos": torch.ones(8, 4), "input_features": torch.ones(1, 2, 3)}
        features = {"video": torch.ones(2, 4), "audio": torch.ones(1, 4)}
        self.assertTrue(policy.validate_perception_reuse(fresh, original, features))
        with self.assertRaisesRegex(ValueError, "token/feature"):
            policy.validate_perception_reuse({**fresh, "input_ids": torch.tensor([[2, 6, 7]])}, original, features)
        with self.assertRaisesRegex(ValueError, "metadata"):
            policy.validate_perception_reuse({**fresh, "feature_attention_mask": torch.tensor([[1, 1, 0]])}, original, features)


class ConflictRuntimeRecoveryTests(unittest.TestCase):
    def make_policy(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import GPT2Config, GPT2LMHeadModel
        torch.manual_seed(17)
        base = GPT2LMHeadModel(GPT2Config(vocab_size=24, n_embd=16, n_layer=1, n_head=2,
                                        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
        config = LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], fan_in_fan_out=True,
                            task_type="CAUSAL_LM", lora_dropout=0)
        model = get_peft_model(base, config, adapter_name="policy")
        model.add_adapter("reference", copy.deepcopy(config))
        policy = object.__new__(ConflictPolicy)
        policy.model, policy.torch, policy.training = model, torch, True
        policy.settings = settings_from()
        policy.value_ids = {"full": [7, 8], "vision": [9, 10, 11]}
        policy.processor = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *a, **kw: [2, 3]))
        model.set_adapter("policy")
        policy._set_trainability()
        return policy

    def test_one_position_logits_and_gradient_match_full_native_projection(self):
        import torch
        policy = self.make_policy()
        prefix = {"input_ids": torch.tensor([[1, 4, 5]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
        inputs = policy.extend(prefix, main_prefix("negative"))
        policy.training_mode()
        expected_scores = policy.model(**inputs, use_cache=False).logits[0, -1, [7, 8]].float()
        expected_scores.sum().backward()
        expected_gradients = {name: value.grad.clone() for name, value in policy.model.named_parameters() if value.requires_grad}
        policy.model.zero_grad(set_to_none=True)
        actual = policy.value_scores(prefix, {}, "full", "negative")
        self.assertTrue(torch.allclose(actual, expected_scores, atol=1e-7, rtol=1e-6))
        actual.sum().backward()
        for name, parameter in policy.model.named_parameters():
            if parameter.requires_grad:
                self.assertTrue(torch.allclose(parameter.grad, expected_gradients[name], atol=1e-7, rtol=1e-6), name)
            elif ".reference." in name:
                self.assertIsNone(parameter.grad)

    def test_checkpoint_exact_next_update_and_rejection_of_old_or_changed_smoke(self):
        import torch
        policy = self.make_policy()
        optimizer = torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad], lr=.01, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        prefix = {"input_ids": torch.tensor([[1, 4, 5]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
        c = {"identity_hash": "new-cost-experiment", "train": paired_rows(), "settings": settings_from()}
        reference = adapter_digest(policy, "reference")
        def update():
            optimizer.zero_grad(set_to_none=True)
            logits = policy.value_scores(prefix, {}, "full", "negative")
            loss = torch.nn.functional.softplus(logits[1]-logits[0])
            loss.backward()
            optimizer.step()
            scheduler.step()
            return float(loss.detach())
        update()
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            checkpoint = save_checkpoint(run, policy, optimizer, scheduler, c, seed=42, step=1)
            expected_loss = update()
            expected_hash = adapter_digest(policy, "policy")
            restored = load_checkpoint(policy, checkpoint, c, optimizer=optimizer, scheduler=scheduler)
            self.assertEqual(restored["algorithm"], VERSION)
            self.assertNotIn("old_block", restored)
            self.assertEqual(update(), expected_loss)
            self.assertEqual(adapter_digest(policy, "policy"), expected_hash)
            self.assertEqual(adapter_digest(policy, "reference"), reference)
            write_json(run / "smoke-selection.json", {"samples": list(range(32))})
            smoke = save_checkpoint(run, policy, optimizer, scheduler, c, seed=42, step=10, smoke=True)
            self.assertIsNone(validate_checkpoint(smoke, c)["sampler"]["pair_sequence"])
            write_json(run / "smoke-selection.json", {"samples": list(reversed(range(32)))})
            with self.assertRaisesRegex(ValueError, "sampler"):
                validate_checkpoint(smoke, c)
            marker = json.loads((checkpoint / "complete.json").read_text())
            marker["schema"] = "omni_grounded_uncertainty_rebel_v1"
            write_json(checkpoint / "complete.json", marker)
            with self.assertRaisesRegex(ValueError, "algorithm"):
                validate_checkpoint(checkpoint, c)


if __name__ == "__main__":
    unittest.main()
