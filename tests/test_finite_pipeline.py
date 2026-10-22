"""Exercise bounded promotion flow and the real finite checkpoint format on CPU."""
import copy
import builtins
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from rl.experiment import fingerprint, write_json
from rl.finite_actions import DEFAULTS, reward_table
from rl.finite_store import load_policy, save_checkpoint, validate_checkpoint
from rl.omni_runtime import OmniPolicy
from rl.state import adapter_digest, restore_rng
from rl.test_finite_actions import FakePolicy, records
from rl.train_finite_actions import train


class PipelineTests(unittest.TestCase):
    def test_cache_repair_only_allows_the_known_import_move(self):
        from scripts.repair_finite_cuda_startup import OLD, NEW, verify_import_only_change
        before = {"rl/train_finite_actions.py": OLD + "    torch = policy.torch\n", "rl/finite_runtime.py":"unchanged"}
        after = {**before, "rl/train_finite_actions.py":NEW + "    torch = policy.torch\n"}
        verify_import_only_change(before, after)
        with self.assertRaises(ValueError):
            verify_import_only_change(before, {**after,"rl/finite_runtime.py":"different scoring"})
        with self.assertRaises(ValueError):
            verify_import_only_change(before, {**after,"rl/train_finite_actions.py":after["rl/train_finite_actions.py"]+"other change\n"})

    def test_gpu_binding_precedes_scheduler_dependency_import(self):
        # Emulate a scheduler dependency with an import-time CUDA side effect.
        # The old order fails before make_policy can select the allocated GPU.
        selected = False
        import_original = builtins.__import__
        class SchedulerImportReached(Exception):
            pass
        def select_first(*args, **kwargs):
            nonlocal selected
            selected = True
            return SimpleNamespace()
        def import_guard(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "transformers" and "get_cosine_schedule_with_warmup" in fromlist:
                self.assertTrue(selected, "CUDA-sensitive scheduler imported before GPU selection")
                raise SchedulerImportReached()
            return import_original(name, globals, locals, fromlist, level)
        with patch('rl.train_finite_actions.make_policy', side_effect=select_first), \
             patch('builtins.__import__', side_effect=import_guard):
            with self.assertRaises(SchedulerImportReached):
                train(SimpleNamespace(run_dir=Path("unused")), {"identity":{"settings":dict(DEFAULTS)}})

    def test_gate_stops_or_refreshes_once_and_never_exceeds_two_epochs(self):
        for promote in (False, True):
            with self.subTest(promote=promote), tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp)
                data = records(19)
                policy = FakePolicy()
                c = {"identity_hash":"test", "identity":{"settings":dict(DEFAULTS)},
                     "train":data,"diagnostic_ids":["0"]}
                background = {r["sample_id"]:r["gt_label"] for r in data}
                t0 = reward_table(data,background,epoch=0,parent_policy="reference",train_hash=fingerprint(data))
                write_json(run/"reward-epoch0.json",t0)
                requests, snapshots = [], []
                ref = torch.zeros(6,dtype=torch.float64).log_softmax(-1).tolist()
                cache = SimpleNamespace(entries={r["sample_id"]:{"reference_log_probs":ref} for r in data},
                    load=lambda row,split:(torch.tensor([int(row["sample_id"])/30,.1,.3],dtype=torch.float64),None))
                def request(root,name,kind,**kw):
                    requests.append((name,kind,kw))
                    if kind == "refresh":
                        t1=reward_table(data,background,epoch=1,parent_policy="end0",train_hash=fingerprint(data))
                        write_json(run/"reward-epoch1.json",t1)
                def save(root,p,opt,sched,step,ctx,table):
                    snapshots.append((step,copy.deepcopy(sched.state_dict()),table["epoch"]))
                    return run/f"checkpoints/step-{step:06d}"
                with patch('rl.train_finite_actions.make_policy',return_value=policy), \
                     patch('rl.train_finite_actions.Cache',return_value=cache), \
                     patch('rl.train_finite_actions.adapter_digest',return_value="reference"), \
                     patch('rl.train_finite_actions.save_checkpoint',side_effect=save), \
                     patch('rl.train_finite_actions.validate_checkpoint',return_value={"policy_digest":"end0"}), \
                     patch('rl.train_finite_actions.queue',side_effect=request), \
                     patch('rl.train_finite_actions.wait_result',return_value={}), \
                     patch('rl.train_finite_actions.leaderboard',return_value={"best":{"score":.8} if promote else None}), \
                     patch('rl.train_finite_actions.emit'):
                    train(SimpleNamespace(run_dir=run),c)
                self.assertEqual([s[0] for s in snapshots],[2,4] if promote else [2])
                self.assertEqual(sum(kind=="refresh" for _,kind,_ in requests),int(promote))
                self.assertTrue((run/"train.done.json").exists())
                self.assertEqual([s[2] for s in snapshots],[0,1] if promote else [0])

    def test_finite_checkpoint_real_peft_roundtrip_and_corruption_rejection(self):
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
        from transformers import GPT2Config, GPT2LMHeadModel, get_cosine_schedule_with_warmup
        torch.manual_seed(42)
        base = GPT2LMHeadModel(GPT2Config(vocab_size=16,n_embd=8,n_layer=1,n_head=2,
                                        resid_pdrop=0,embd_pdrop=0,attn_pdrop=0))
        config = LoraConfig(r=2,lora_alpha=4,target_modules=["c_attn"],fan_in_fan_out=True,task_type="CAUSAL_LM")
        model = get_peft_model(base,config,adapter_name="policy")
        initial = copy.deepcopy(get_peft_model_state_dict(model,adapter_name="policy"))
        model.add_adapter("reference",copy.deepcopy(config))
        set_peft_model_state_dict(model,initial,adapter_name="reference")
        model.set_adapter("policy")
        p = object.__new__(OmniPolicy)
        p.model,p.torch,p.training = model,torch,True
        p._set_trainability()
        class Processor:
            def save_pretrained(self,path):
                Path(path).mkdir()
                (Path(path)/"test.json").write_text("{}")
        p.processor = Processor()
        parameters=[t for t in model.parameters() if t.requires_grad]
        opt = torch.optim.AdamW(parameters,lr=5e-6,weight_decay=0)
        scheduler = get_cosine_schedule_with_warmup(opt,10,420)
        reference = adapter_digest(p,"reference")
        def update():
            opt.zero_grad()
            model(input_ids=torch.tensor([[1,2,3]]),use_cache=False).logits.square().mean().backward()
            opt.step(); scheduler.step()
        update()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"checkpoints").mkdir()
            context={"identity_hash":"unit-test"}
            table={"epoch":0,"train_hash":"train-only"}
            checkpoint=save_checkpoint(root,p,opt,scheduler,1,context,table)
            state=torch.load(checkpoint/"training_state.pt",map_location="cpu",weights_only=False)
            self.assertEqual(validate_checkpoint(checkpoint,context)["reward_hash"],fingerprint(table))
            update()
            expected=adapter_digest(p,"policy")
            marker=load_policy(p,checkpoint,context)
            self.assertEqual(adapter_digest(p,"policy"),marker["policy_digest"])
            opt.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
            restore_rng(torch,state["rng"])
            update()
            self.assertEqual(adapter_digest(p,"policy"),expected)
            self.assertEqual(adapter_digest(p,"reference"),reference)
            with (checkpoint/"training_state.pt").open("ab") as f:
                f.write(b"corruption")
            with self.assertRaises(ValueError):
                validate_checkpoint(checkpoint,context)


if __name__ == "__main__":
    unittest.main()
