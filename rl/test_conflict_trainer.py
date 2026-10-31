"""Exercise the actual cost update and bounded orchestration without a GPU."""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rl import train_conflict_cost as entry
from rl.conflict_cost import EpochSampler, settings_from
from rl.experiment import write_json


def paired_training(size=60):
    rows = [{"sample_id": f"pair-{i:02d}-{protocol}", "split_group_id": f"pair-{i:02d}",
             "protocol": protocol, "video": f"video-{i}.mp4", "sample_type": kind,
             "source_dataset": "ch_sims"}
            for i in range(30) for protocol, kind in (("VT", "Conflict"), ("VA", "Aligned"))]
    rows.extend({"sample_id": f"extra-{i:04d}", "split_group_id": f"extra-{i:04d}",
                 "protocol": "VT", "video": f"extra-{i}.mp4", "sample_type": "Aligned",
                 "source_dataset": "ch_sims"} for i in range(size-len(rows)))
    return rows


class LinearPolicy:
    def __init__(self):
        import torch
        self.torch = torch
        self.model = torch.nn.Linear(3, 2, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.model.weight.copy_(torch.tensor([[.4, -.6, .1], [-.3, .8, .2]], dtype=torch.float64))
        self.reference = torch.tensor([[.3, .2, -.1], [-.7, .1, .6]], dtype=torch.float64)

    def value_scores(self, prefix, features, mode, emotion, *, adapter="policy"):
        return self.model(prefix["x"]) if adapter == "policy" else self.reference @ prefix["x"]


class LinearCache:
    def __init__(self, policy, rows):
        self.policy, self.rows = policy, rows
        self.inputs = {r["sample_id"]: policy.torch.tensor(
            [1. + i/7., (-1.)**i, .3+i/11.], dtype=policy.torch.float64) for i, r in enumerate(rows)}

    def load(self, row, split):
        assert split == "train"
        return {"x": self.inputs[row["sample_id"]]}, {}

    def entry(self, row, mode):
        assert mode == "full"
        return {"baseline": {"pred_emotion": "negative"},
                "reference_scores": (self.policy.reference @ self.inputs[row["sample_id"]]).tolist()}


class ActualUpdateTests(unittest.TestCase):
    def fixture(self):
        import torch
        rows = paired_training()[:5]
        policy = LinearPolicy()
        cache = LinearCache(policy, rows)
        settings = settings_from()
        c = {"train": rows, "settings": settings}
        slots = [{"row": row, "draw_id": f"test:{i}", "selection": "epoch_uniform"}
                 for i, row in enumerate(rows[:3])]
        pair = {"conflict": rows[0], "aligned": rows[1], "pair_id": "pair-00", "draw_id": "pair-draw"}
        optimizer = torch.optim.AdamW(policy.model.parameters(), lr=.01, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        return policy, cache, c, slots, pair, optimizer, scheduler

    def test_actual_update_matches_direct_joint_objective_and_clipped_adam(self):
        import torch
        import torch.nn.functional as F
        policy, cache, c, slots, pair, optimizer, scheduler = self.fixture()
        expected = LinearPolicy()
        expected.model.load_state_dict(policy.model.state_dict())
        expected_optimizer = torch.optim.AdamW(expected.model.parameters(), lr=.01, weight_decay=0)
        counts = Counter(r["sample_type"] for r in c["train"])
        loss = 0.
        for slot in slots:
            row = slot["row"]
            x = cache.inputs[row["sample_id"]]
            score = expected.model(x)
            d = score[0] - score[1]
            kind = row["sample_type"]
            sign, fraction = (-1., .6) if kind == "Conflict" else (1., .4)
            loss = loss + sum(counts.values()) * fraction / counts[kind] / len(slots) * F.softplus(sign*d)
            reference = expected.reference @ x
            lp, rp = score.log_softmax(-1), reference.log_softmax(-1)
            loss = loss + .02 * (lp.exp()*(lp-rp)).sum()/len(slots)
        sc = expected.model(cache.inputs[pair["conflict"]["sample_id"]])
        sa = expected.model(cache.inputs[pair["aligned"]["sample_id"]])
        loss = loss + .1*F.softplus(1.-((sc[0]-sc[1])-(sa[0]-sa[1])))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(expected.model.parameters(), 1.)
        expected_optimizer.step()
        actual = entry.update(policy, cache, slots, pair, optimizer, scheduler,
                              list(policy.model.parameters()), c)
        self.assertAlmostEqual(actual["loss"], float(loss.detach()), places=9)
        self.assertAlmostEqual(actual["grad_norm"], float(grad_norm), places=9)
        self.assertTrue(torch.allclose(policy.model.weight, expected.model.weight, atol=1e-12, rtol=0))
        self.assertEqual(actual["main_count"], 3)
        self.assertEqual(actual["pair_id"], pair["pair_id"])
        self.assertAlmostEqual(actual["pair_contribution"], .1*actual["pair_loss"])

    def test_actual_pair_backward_matches_joint_gradient(self):
        import torch
        from rl.conflict_cost import pair_loss
        policy, cache, c, slots, pair, _, _ = self.fixture()
        items = [(*cache.load(pair[key], "train"), "negative") for key in ("conflict", "aligned")]
        scores = [policy.value_scores(p, f, "full", emotion) for p, f, emotion in items]
        (.1*pair_loss(*scores)).backward()
        expected = policy.model.weight.grad.clone()
        policy.model.zero_grad(set_to_none=True)
        measured = entry.pair_backward(policy, items, c["settings"])
        self.assertTrue(torch.allclose(policy.model.weight.grad, expected, atol=1e-14, rtol=0))
        self.assertAlmostEqual(measured["pair_contribution"], .1*measured["pair_loss"])

    def test_actual_update_exact_adam_state_restore(self):
        import torch
        policy, cache, c, slots, pair, optimizer, scheduler = self.fixture()
        params = list(policy.model.parameters())
        entry.update(policy, cache, slots, pair, optimizer, scheduler, params, c)
        state = copy.deepcopy((policy.model.state_dict(), optimizer.state_dict(), scheduler.state_dict()))
        expected = entry.update(policy, cache, list(reversed(slots)), pair, optimizer, scheduler, params, c)
        expected_weight = policy.model.weight.detach().clone()
        expected_opt = copy.deepcopy(optimizer.state_dict())
        policy.model.load_state_dict(state[0])
        optimizer.load_state_dict(state[1])
        scheduler.load_state_dict(state[2])
        actual = entry.update(policy, cache, list(reversed(slots)), pair, optimizer, scheduler, params, c)
        self.assertEqual(actual, expected)
        self.assertTrue(torch.equal(policy.model.weight, expected_weight))
        for key, value in expected_opt["state"][0].items():
            restored = optimizer.state_dict()["state"][0][key]
            self.assertTrue(torch.equal(value, restored) if torch.is_tensor(value) else value == restored)

    def test_actual_update_full_peft_checkpoint_restores_next_pair_and_main_update(self):
        import torch
        from rl.test_conflict_store import ConflictRuntimeRecoveryTests
        from rl.conflict_store import save_checkpoint, load_checkpoint
        from rl.state import adapter_digest
        policy = ConflictRuntimeRecoveryTests().make_policy()
        rows = paired_training()
        c = {"identity_hash": "trainer-real-checkpoint", "train": rows, "settings": settings_from()}
        class Cache:
            def load(self, row, split):
                token = 4 if row["protocol"] == "VT" else 5
                return {"input_ids": torch.tensor([[1, token, 6]]),
                        "attention_mask": torch.ones(1, 3, dtype=torch.long)}, {}
            def entry(self, row, mode):
                prefix, features = self.load(row, "train")
                with torch.no_grad():
                    reference = policy.value_scores(prefix, features, "full", "negative", adapter="reference")
                return {"baseline": {"pred_emotion": "negative"}, "reference_scores": reference.tolist()}
        cache = Cache()
        sampler = EpochSampler(rows, 42)
        params = [p for p in policy.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=.001, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1/(step+1))
        entry.update(policy, cache, sampler.batch(0)[:2], sampler.pair(0), optimizer, scheduler, params, c)
        frozen = adapter_digest(policy, "reference")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_checkpoint(Path(directory), policy, optimizer, scheduler, c, seed=42, step=1)
            expected = entry.update(policy, cache, sampler.batch(1)[:2], sampler.pair(1), optimizer, scheduler, params, c)
            expected_digest = adapter_digest(policy, "policy")
            state = load_checkpoint(policy, checkpoint, c, optimizer=optimizer, scheduler=scheduler)
            actual = entry.update(policy, cache, sampler.batch(state["step"])[:2], sampler.pair(state["step"]),
                                  optimizer, scheduler, params, c)
            self.assertEqual(actual, expected)
            self.assertEqual(adapter_digest(policy, "policy"), expected_digest)
            self.assertEqual(adapter_digest(policy, "reference"), frozen)


class SamplingAndStageTests(unittest.TestCase):
    def test_unlimited_runtime_has_no_wall_clock_stop(self):
        c = {"deadline": None, "training_cutoff": None}
        with patch.object(entry.time, "time", return_value=10**12):
            self.assertEqual(entry.remaining(c), float("inf"))
            self.assertFalse(entry.past_training_cutoff(c))
            entry.check_budget(c)
        self.assertIsNone(settings_from()["budget_hours"])
        self.assertIsNone(settings_from()["training_cutoff_hours"])

    def test_smoke_resume_10_replays_check_and_resume_40_only_finishes_diagnostics(self):
        for completed_step in (10, 40):
            with self.subTest(completed_step=completed_step), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                selected, initial, final = [], [], []
                for cohort in ("missed_conflict", "false_positive", "easy_conflict", "easy_aligned"):
                    kind = "Conflict" if "conflict" in cohort else "Aligned"
                    for i in range(8):
                        sid = f"{cohort}:{i}"
                        row = {"sample_id": sid, "split_group_id": sid, "sample_type": kind,
                               "protocol": "VT" if i < 4 else "VA"}
                        selected.append({"row": row, "cohort": cohort})
                        was_correct = cohort.startswith("easy_")
                        initial.append({"sample_id": sid, "correct": was_correct})
                        final.append({"sample_id": sid, "correct": was_correct or i < 4})
                rest = {"sample_id": "remaining", "split_group_id": "remaining", "sample_type": "Aligned"}
                c = {"identity_hash": "smoke-test", "settings": settings_from(),
                     "train": [v["row"] for v in selected]+[rest], "deadline": time.time()+3600}
                checkpoint = run/f"checkpoints/smoke/step-{completed_step:06d}"
                write_json(checkpoint/"complete.json", {})
                if completed_step == 40:
                    write_json(run/"recovery-check.json", {"passed": True})
                (run/"smoke-updates.jsonl").write_text("".join(json.dumps({"step": step, "update_seconds": .5})+"\n"
                    for step in range(1, completed_step+1)), encoding="utf-8")
                policy = SimpleNamespace(torch=SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda: 0)))
                cache = SimpleNamespace(run=run, entry=lambda row, mode: {"reference_scores": [0., 1.]})
                args = SimpleNamespace(run_dir=run)
                with patch.object(entry, "make_policy", return_value=policy), patch.object(entry, "TaskCache", return_value=cache), \
                        patch.object(entry, "smoke_rows", return_value=selected), \
                        patch.object(entry, "optimizer_for", return_value=(None, None, [])), \
                        patch.object(entry, "adapter_digest", return_value="frozen"), \
                        patch.object(entry, "measure_decisions", side_effect=[initial, final,
                            [{"sample_id": "remaining", "target": "no", "correct": True}]]), \
                        patch.object(entry, "load_checkpoint") as load, patch.object(entry, "save_checkpoint") as save, \
                        patch.object(entry, "update", return_value={"loss": 1., "samples": []}) as update, \
                        patch.object(entry, "emit"), patch("rl.grounded_recovery.snapshot", return_value={}), \
                        patch("rl.grounded_recovery.compare_snapshots", return_value={"exact": True}):
                    entry.smoke(args, c)
                result = json.loads((run/"smoke.json").read_text())
                self.assertTrue(result["passed"])
                self.assertTrue(result["resume_exact"])
                self.assertEqual(result["remaining_train"]["records"], 1)
                self.assertEqual(update.call_count, 31 if completed_step == 10 else 0)
                self.assertEqual(load.call_count, 2 if completed_step == 10 else 1)
                self.assertEqual(save.call_count, 1 if completed_step == 10 else 0)
                for call in update.call_args_list:
                    self.assertIsNone(call.args[3])
                    self.assertEqual(call.kwargs["counts"], {"Conflict": 16, "Aligned": 16})
                    self.assertEqual(len(call.args[2]), 16)
                    self.assertTrue({slot["row"]["sample_id"] for slot in call.args[2]} <=
                                    {v["row"]["sample_id"] for v in selected})

    def test_smoke_selection_uses_32_train_rows_balanced_by_cohort_and_protocol(self):
        rows, by_id = [], {}
        for cohort in ("missed_conflict", "false_positive", "easy_conflict", "easy_aligned"):
            kind = "Conflict" if "conflict" in cohort else "Aligned"
            decision = "no" if cohort in ("missed_conflict", "easy_aligned") else "yes"
            for protocol in ("VT", "VA"):
                for i in range(5):
                    row = {"sample_id": f"{cohort}:{protocol}:{i}", "protocol": protocol, "sample_type": kind}
                    rows.append(row)
                    by_id[row["sample_id"]] = {"reference_scores": [1., 0.] if decision == "yes" else [0., 1.]}
        cache = SimpleNamespace(entry=lambda row, mode: by_id[row["sample_id"]])
        selected = entry.smoke_rows({"train": rows, "val": [{"sample_id": "never-select-val"}]}, cache)
        self.assertEqual(len({v["row"]["sample_id"] for v in selected}), 32)
        self.assertEqual(Counter((v["cohort"], v["row"]["protocol"]) for v in selected),
                         {(cohort, protocol): 4 for cohort in ("missed_conflict", "false_positive", "easy_conflict", "easy_aligned")
                          for protocol in ("VT", "VA")})
        self.assertEqual(selected, entry.smoke_rows({"train": list(reversed(rows))}, cache))
        without_hard_va = [r for r in rows if not r["sample_id"].startswith("missed_conflict:VA")]
        with self.assertRaisesRegex(ValueError, "cannot be fabricated"):
            entry.smoke_rows({"train": without_hard_va}, cache)

    def test_full_epoch_stops_at_210_when_score_gate_fails_and_retains_coverage(self):
        rows = paired_training(3354)
        c = {"train": rows, "settings": settings_from(), "deadline": time.time()+3600,
             "training_cutoff": time.time()+3000}
        observed = []
        def fake_update(policy, cache, slots, pair, optimizer, scheduler, parameters, context):
            observed.extend(slot["row"]["sample_id"] for slot in slots)
            return {"loss": 1., "samples": []}
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            def save(run, policy, optimizer, scheduler, context, *, seed, step):
                path = run/f"checkpoints/seed-{seed}/step-{step:06d}"
                write_json(path/"complete.json", {})
                return path
            args = SimpleNamespace(run_dir=run, seed=42, target_steps=420)
            with patch.object(entry, "make_policy", return_value=SimpleNamespace(torch=None)), \
                    patch.object(entry, "optimizer_for", return_value=(None, None, [])), \
                    patch.object(entry, "TaskCache"), patch.object(entry, "adapter_digest", return_value="frozen"), \
                    patch.object(entry, "update", side_effect=fake_update), patch.object(entry, "save_checkpoint", side_effect=save), \
                    patch.object(entry, "wait_result", return_value={}), patch.object(entry, "candidates", return_value=[]), \
                    patch.object(entry, "emit"):
                entry.train(args, c)
            done = json.loads((run/"train-42.done.json").read_text())
            self.assertEqual(done["step"], 210)
            self.assertEqual(done["reason"], "first_epoch_gate_not_met")
            self.assertEqual(done["main_draws"], 3354)
            self.assertEqual(done["unique_main_seen"], 3354)
            self.assertEqual(Counter(observed), Counter(r["sample_id"] for r in rows))
            self.assertEqual({p.name for p in (run/"requests").glob("*.json")},
                             {"seed-42-000105-eval.json", "seed-42-000210-eval.json"})

    def test_resume_at_first_epoch_boundary_checks_gate_before_new_update(self):
        c = {"train": paired_training(3354), "settings": settings_from(), "deadline": time.time()+3600,
             "training_cutoff": time.time()+3000}
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            write_json(run/"latest-42.json", {"checkpoint": "selected-saved-checkpoint"})
            write_json(run/"checkpoints/seed-42/step-000210/complete.json", {})
            args = SimpleNamespace(run_dir=run, seed=42, target_steps=420)
            with patch.object(entry, "make_policy", return_value=SimpleNamespace(torch=None)), \
                    patch.object(entry, "optimizer_for", return_value=(None, None, [])), \
                    patch.object(entry, "TaskCache"), patch.object(entry, "adapter_digest", return_value="frozen"), \
                    patch.object(entry, "load_checkpoint", return_value={"seed": 42, "step": 210}), \
                    patch.object(entry, "wait_result", return_value={}), patch.object(entry, "candidates", return_value=[]), \
                    patch.object(entry, "emit"), patch.object(entry, "update") as update:
                entry.train(args, c)
                update.assert_not_called()

    def controller_fixture(self, run):
        for folder in ("logs", "cache", "results"):
            (run/folder).mkdir()
        write_json(run/"probe.json", {"passed": True})
        write_json(run/"cache/complete.json", {})
        write_json(run/"smoke.json", {"passed": True})
        for seed in (42, 43):
            write_json(run/f"train-{seed}.done.json", {"duration_seconds": 100})
        args = SimpleNamespace(run_dir=run, gpus=["1", "2"], memory_gib=32, cpu_threads=4, resume=False)
        c = {"settings": settings_from(), "deadline": time.time()+1800, "training_cutoff": time.time()+900}
        return args, c

    def test_controller_requires_primary_then_matched_replication_before_export(self):
        for primary_pass, replication_pass in ((False, False), (True, False), (True, True)):
            with self.subTest(primary=primary_pass, replication=replication_pass), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                args, c = self.controller_fixture(run)
                primary = {"step": 105, "checkpoint": "seed42-step105", "primary": {"eligible": primary_pass}}
                replicate = {"step": 105, "checkpoint": "seed43-step105", "replication": {"eligible": replication_pass}}
                def completed(command, **kwargs):
                    return SimpleNamespace(poll=lambda: 0, returncode=0, wait=lambda **kwargs: 0)
                with patch.object(entry, "initialize", return_value=c), patch.object(entry, "zero_baselines"), \
                        patch.object(entry, "candidates", side_effect=lambda run, context, seed: [primary] if seed == 42 else [replicate]), \
                        patch.object(entry.subprocess, "Popen", side_effect=completed) as popen, patch.object(entry, "emit"):
                    entry.controller(args)
                roles = [call.args[0][4] for call in popen.call_args_list]
                self.assertEqual("export" in roles, primary_pass and replication_pass)
                worker_seeds = [call.args[0][call.args[0].index("--seed")+1] for call in popen.call_args_list if call.args[0][4] == "worker"]
                self.assertEqual(worker_seeds, ["42", "43"] if primary_pass else ["42"])
                if primary_pass:
                    selected = json.loads((run/"selection.json").read_text())
                    self.assertEqual(selected["best"]["checkpoint"], "seed42-step105")
                    self.assertEqual(selected["replication_passed"], replication_pass)

    def test_replication_rejects_a_different_update_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            write_json(run/"primary-selection.json", {"step": 105})
            args = SimpleNamespace(run_dir=run, seed=43, target_steps=210)
            c = {"train": paired_training(), "settings": settings_from()}
            with patch.object(entry, "make_policy", return_value=SimpleNamespace(torch=None)), \
                    patch.object(entry, "optimizer_for", return_value=(None, None, [])), patch.object(entry, "TaskCache"):
                with self.assertRaisesRegex(ValueError, "selected primary update count"):
                    entry.train(args, c)


if __name__ == "__main__":
    unittest.main()
