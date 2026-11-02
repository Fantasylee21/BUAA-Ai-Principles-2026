"""CPU tests of exact expectations, replay, rewards, resume and selection contracts."""
import contextlib
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from rl.experiment import fingerprint, write_json
from rl.finite_actions import ACTIONS, DEFAULTS, PREFIXES, epoch_batches, objective, reward_table, selection, validate_reward
from rl.finite_reports import report, comparison
from rl.finite_runtime import FinitePolicy, pack_cache, unpack_cache
from rl.finite_store import validate_checkpoint, merge_cache
from rl.metrics import macro_f1
from rl.omni_io import messages_for
from rl.train_finite_actions import evaluation_steps, parse_args


def records(n=19):
    return [{"sample_id": str(i), "video": f"{i}.mp4", "protocol": "VT", "text_content": "visible transcript",
             "source_dataset": "ch_sims", "split_group_id": str(i), "gt_label": ACTIONS[(i%3)*2][0],
             "sample_type": "Conflict" if i%4 == 0 else "Aligned"} for i in range(n)]


class FakePolicy(FinitePolicy):
    def __init__(self, micro=1):
        self.torch = torch
        self.settings = {**DEFAULTS, "candidate_micro_batch_size": micro}
        self.training = True
        self.model = torch.nn.Linear(3, 6, dtype=torch.float64)
        with torch.no_grad():
            self.model.weight.copy_(torch.arange(18, dtype=torch.float64).reshape(6,3)/30)
            self.model.bias.copy_(torch.arange(6, dtype=torch.float64)/10)

    def chunk_scores(self, prefix, indices):
        return self.model(prefix).tanh()[list(indices)]

    @contextlib.contextmanager
    def adapter(self, name):
        yield

    @contextlib.contextmanager
    def cached_perception(self, features):
        yield


class FiniteTests(unittest.TestCase):
    def test_all_actions_and_prefixes(self):
        self.assertEqual(len(set(ACTIONS)), 6)
        for (emotion, uncertain), prefix in zip(ACTIONS, PREFIXES):
            parsed = json.loads(prefix + '"none"}')
            self.assertEqual((parsed["pred_emotion"], parsed["is_uncertain"]), (emotion, uncertain))

    def test_rewards_equal_brute_force_full_training_f1(self):
        rows = records()
        predictions = {r["sample_id"]: ACTIONS[((i+1)%3)*2][0] for i, r in enumerate(rows)}
        table = reward_table(rows, predictions, epoch=0, parent_policy="v4", train_hash=fingerprint(rows))
        y = [r["gt_label"] for r in rows]
        p = [predictions[r["sample_id"]] for r in rows]
        base = macro_f1(y,p)
        for i, row in enumerate(rows):
            for a, (e,u) in enumerate(ACTIONS):
                changed = p.copy()
                changed[i] = e
                emotional = max(-1, min(1, len(rows)*(macro_f1(y, changed)-base)))
                conflict = row["sample_type"] == "Conflict"
                count = sum(r["sample_type"] == row["sample_type"] for r in rows)
                uncertainty = (.3 if conflict else .2)*len(rows)/count if u == ("yes" if conflict else "no") else 0
                self.assertAlmostEqual(table["rewards"][row["sample_id"]][a], .5*emotional+uncertainty)
        validate_reward(table, rows, 0, "v4")
        with self.assertRaises(ValueError):
            validate_reward(table, rows, 1, "v4")
        changed = copy.deepcopy(rows)
        changed[0]["gt_label"] = "negative"
        with self.assertRaises(ValueError):
            validate_reward(table, changed, 0, "v4")
        with self.assertRaises(ValueError):
            reward_table(rows, {**predictions, "val-only": "positive"}, epoch=0, parent_policy="v4", train_hash=fingerprint(rows))

    def test_population_uncertainty_weights_and_each_row_once(self):
        rows = records()
        table = reward_table(rows, {r["sample_id"]: r["gt_label"] for r in rows}, epoch=0, parent_policy="v4", train_hash=fingerprint(rows))
        # At the gold emotion, counterfactual emotion reward is exactly zero.
        totals = {"Conflict": 0., "Aligned": 0.}
        for r in rows:
            a = ACTIONS.index((r["gt_label"], "yes" if r["sample_type"] == "Conflict" else "no"))
            totals[r["sample_type"]] += table["rewards"][r["sample_id"]][a]/len(rows)
        self.assertAlmostEqual(totals["Conflict"], .3)
        self.assertAlmostEqual(totals["Aligned"], .2)
        b = epoch_batches(rows, 42, 0, 16)
        self.assertEqual([len(x) for x in b], [16,3])
        self.assertEqual(sorted(r["sample_id"] for batch in b for r in batch), sorted(r["sample_id"] for r in rows))
        self.assertEqual(b, epoch_batches(rows,42,0,16))
        self.assertNotEqual(b, epoch_batches(rows,42,1,16))

    def test_unsampled_correct_action_receives_improving_gradient(self):
        # The former four rollouts may all be action 0 (wrong). Enumeration
        # still creates a gradient toward action 5, which was never sampled.
        scores = torch.tensor([2.,0.,0.,0.,0.,-2.], dtype=torch.float64, requires_grad=True)
        ref = scores.detach().log_softmax(-1).requires_grad_(True)
        reward = torch.tensor([0.,0.,0.,0.,0.,1.], dtype=torch.float64)
        loss, metrics = objective(scores,ref,reward,.02)
        loss.backward()
        self.assertLess(scores.grad[5], 0)
        self.assertGreater(scores.grad[0], 0)
        self.assertIsNone(ref.grad)
        self.assertAlmostEqual(float(metrics["probabilities"].detach().sum()),1)
        self.assertAlmostEqual(float(metrics["kl"].detach()),0)
        self.assertGreater(float((scores.detach()-.1*scores.grad).softmax(-1)[5]),float(scores.detach().softmax(-1)[5]))

    def test_replay_and_all_microbatches_match_direct_gradient(self):
        x = torch.tensor([.2,-.4,.7],dtype=torch.float64)
        ref = torch.arange(6,dtype=torch.float64).log_softmax(-1)
        reward = torch.tensor([1.,0.,.3,-.5,0.,.7],dtype=torch.float64)
        direct = FakePolicy(6)
        loss, _ = objective(direct.scores(x),ref,reward,.02)
        (loss/3).backward()
        for micro in (1,2,3,6):
            policy = FakePolicy(micro)
            policy.backward_actions(x,None,ref,reward,divisor=3)
            for p,q in zip(policy.model.parameters(),direct.model.parameters()):
                self.assertTrue(torch.allclose(p.grad,q.grad,atol=1e-12,rtol=1e-10))

    def test_cache_roundtrip_preserves_positions_masks_and_features(self):
        prefix = {"input_ids": torch.tensor([[1,2,3]]), "attention_mask": torch.ones(1,3,dtype=torch.long),
                  "pixel_values_videos": torch.randn(3,9), "input_features": torch.randn(1,8,9),
                  "feature_attention_mask": torch.ones(1,9), "video_grid_thw": torch.tensor([[1,2,2]]),
                  "video_second_per_grid": [.5], "use_audio_in_video": True}
        features = {"video": torch.randn(2,7), "audio": torch.randn(3,7)}
        packed = pack_cache(torch,prefix,features)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"cache.pt"
            torch.save(packed,path)
            loaded, restored = unpack_cache(torch,torch.load(path,weights_only=True),device="cpu")
        self.assertEqual(loaded["pixel_values_videos"].numel(),0)
        self.assertEqual(loaded["input_features"].numel(),0)
        for k in ("input_ids","attention_mask","feature_attention_mask","video_grid_thw"):
            self.assertTrue(torch.equal(prefix[k],loaded[k]))
        self.assertEqual(loaded["video_second_per_grid"],[.5])
        for k in features:
            self.assertTrue(torch.equal(features[k],restored[k]))

    def test_resume_preserves_order_scheduler_optimizer_and_updates(self):
        rows = records(35)
        batches = epoch_batches(rows,42,0,16)
        ref = torch.zeros(6,dtype=torch.float64).log_softmax(-1)
        reward = torch.arange(6,dtype=torch.float64)/5
        def setup():
            p = FakePolicy()
            o = torch.optim.AdamW(p.model.parameters(),lr=5e-6,weight_decay=0)
            s = torch.optim.lr_scheduler.LambdaLR(o,lambda n: (n+1)/10 if n<10 else .5*(1+__import__('math').cos(n/420*__import__('math').pi)))
            return p,o,s
        def update(p,o,s,batch):
            o.zero_grad()
            for row in batch:
                x = torch.tensor([int(row["sample_id"])/50,.2,.8],dtype=torch.float64)
                p.backward_actions(x,None,ref,reward,divisor=len(batch))
            o.step(); s.step()
        p,o,s = setup()
        update(p,o,s,batches[0])
        snapshot = copy.deepcopy((p.model.state_dict(),o.state_dict(),s.state_dict()))
        for batch in batches[1:]:
            update(p,o,s,batch)
        restored,ro,rs = setup()
        restored.model.load_state_dict(snapshot[0]); ro.load_state_dict(snapshot[1]); rs.load_state_dict(snapshot[2])
        for batch in epoch_batches(rows,42,0,16)[1:]:
            update(restored,ro,rs,batch)
        self.assertEqual(s.state_dict(),rs.state_dict())
        for a,b in zip(p.model.parameters(),restored.model.parameters()):
            self.assertTrue(torch.equal(a,b))

    def test_gate_uses_both_baselines_and_contracts(self):
        baselines = {"free":{"score":.70},"finite":{"score":.74}}
        good = {"score":.745,"invalid_outputs":0,"inconsistent_questions":0}
        self.assertTrue(selection(baselines,good)["promote"])
        self.assertFalse(selection(baselines,{**good,"score":.7449})["promote"])
        self.assertFalse(selection(baselines,{**good,"inconsistent_questions":1})["promote"])
        self.assertFalse(selection(baselines,{**good,"invalid_outputs":1})["promote"])
        self.assertEqual(evaluation_steps(210,50),[50,100,150,210])

    def test_old_optimizer_and_incomplete_cache_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root/"complete.json",{"schema":"omni_checkpoint_v1"})
            with self.assertRaises(ValueError):
                validate_checkpoint(root,{"identity_hash":"new"})
            c = {"identity_hash":"new","train":records(2),"val":[]}
            for shard in (0,1):
                write_json(root/f"cache/shard-{shard}.json",{"identity_hash":"new","shard":shard,"entries":[]})
            with self.assertRaises(ValueError):
                merge_cache(root,c)

    def test_gold_labels_do_not_reach_prompt(self):
        row = records(1)[0]
        other = {**row,"gt_label":"negative","sample_type":"Aligned"}
        self.assertEqual(messages_for(row,Path("video.mp4")),messages_for(other,Path("video.mp4")))

    def test_paired_diagnostics_and_small_sample_intervals(self):
        rows = [{**r,"pred_emotion":r["gt_label"],"is_uncertain":"no","valid_json":True,"question_consistent":True} for r in records()]
        changed = copy.deepcopy(rows)
        changed[0]["is_uncertain"] = "yes"
        changed[1]["is_uncertain"] = "yes"
        diff = comparison(rows,changed,repeats=50)
        self.assertEqual(diff["conflicts_corrected"],1)
        self.assertEqual(diff["false_positives_added"],1)
        self.assertEqual(report(changed)["counts"]["natural_conflict"]["correct"],1)
        self.assertIsNotNone(diff["paired_group_bootstrap95_score_delta"])


if __name__ == "__main__":
    unittest.main()
