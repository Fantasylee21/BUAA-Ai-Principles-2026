"""Numerical policy tests plus output and gated-pipeline regression tests."""
from __future__ import annotations

from collections import Counter
import csv
import importlib.util
import io
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from rl.conflict_reward import (reward_weight, anchored_logit, objective, batches,
    compare, choose, bootstrap, predicted)
from scripts.conflict_reward_export import render
from scripts.conflict_reward_pipeline import Pipeline
from scripts.joint_affect_pipeline import immutable
from rl.omni_io import messages_for


def row(i,conflict=False,protocol="VT"):
    return {"sample_id":str(i),"video":str(i)+".mp4","protocol":protocol,"text_content":"input text",
        "source_dataset":"ch_sims_v2","sample_type":"Conflict" if conflict else "Aligned",
        "views":{"M1":{"label":"positive","is_clear":True},
                 "M2":{"label":"negative" if conflict else "positive","is_clear":True},
                 "M12":{"label":"positive","is_clear":True}}}


def baseline(i,yes=False):
    return {"sample_id":str(i),"pred_emotion":"positive","is_uncertain":"yes" if yes else "no",
            "clarifying_question":"你希望如何理解这段表达？" if yes else "none"}


class Contracts(unittest.TestCase):
    def test_population_reward_is_exact_task_component(self):
        c={"Conflict":411,"Aligned":2943};n=sum(c.values())
        self.assertAlmostEqual(c["Conflict"]/n*reward_weight("Conflict",c)*.7+c["Aligned"]/n*reward_weight("Aligned",c)*.95,.3*.7+.2*.95)
        self.assertEqual(reward_weight("Conflict",c,"B"),2*reward_weight("Conflict",c,"A"))
        self.assertEqual(reward_weight("Aligned",c,"B"),reward_weight("Aligned",c,"A"))

    def test_all_rows_once_and_reproducible_epoch(self):
        rows=[row(i) for i in range(37)]
        first=batches(rows,42,0)
        self.assertEqual(Counter(r["sample_id"] for b in first for r in b),Counter(r["sample_id"] for r in rows))
        self.assertEqual([len(b) for b in first],[16,16,5])
        self.assertEqual(first,batches(rows,42,0));self.assertNotEqual(first,batches(rows,42,1))

    def test_zero_increment_keeps_either_anchor_even_extreme_native_logits(self):
        for value in (-1000.,-12.,0.,17.,1000.):
            self.assertAlmostEqual(anchored_logit(value,value,True),math.log(3))
            self.assertAlmostEqual(anchored_logit(value,value,False),-math.log(3))

    def test_metadata_and_va_transcript_do_not_enter_prompt(self):
        r=row(4,True,"VA");r["text_content"]="FORBIDDEN_TRANSCRIPT";r["sample_id"]="FORBIDDEN_ID"
        text=str(messages_for(r,Path("video.mp4")))
        self.assertNotIn("FORBIDDEN",text);self.assertNotIn("Conflict",text)
        r["protocol"]="VT"
        self.assertIn("FORBIDDEN_TRANSCRIPT",str(messages_for(r,Path("video.mp4"))))

    def test_reward_cannot_change_evaluation_weights(self):
        rows=[row(i,i<10) for i in range(20)]
        a=[predicted(baseline(i),.25) for i in range(20)]
        b=[predicted(baseline(i),.75) for i in range(20)]
        m=compare(rows,a,b,stage="pilot")
        self.assertAlmostEqual(m["gain"],.1)
        self.assertAlmostEqual(m["uncertainty_gain"],.1)

    def test_paired_bootstrap_preserves_group_pairing(self):
        rows=[row(i,i%2==0) for i in range(20)]
        a=[predicted(baseline(i),.25) for i in range(20)]
        b=[predicted(baseline(i),.75) for i in range(20)]
        result=bootstrap(rows,a,b,[str(i//2) for i in range(20)],repeats=100)
        for value in result["percentile_95"]:self.assertAlmostEqual(value,.1)

    def test_ties_prefer_simpler_passing_arm(self):
        reports={a:{"passed":True,"candidate":{"selection_score":s}} for a,s in zip(("A","B","C"),(.9,.90005,.90009))}
        self.assertEqual(choose(reports),"A")
        reports["A"]["passed"]=False;self.assertEqual(choose(reports),"B")
        reports["C"]["candidate"]["selection_score"]=.91;self.assertEqual(choose(reports),"C")

    def test_734_zero_update_csv_identity_and_bidirectional_changes(self):
        rows=[row(i) for i in range(734)]
        entries=[{"sample_id":str(i),"submission":{"baseline":baseline(i,i%7==0)}} for i in range(734)]
        predictions=[predicted(e["submission"]["baseline"],.75 if i%7==0 else .25) for i,e in enumerate(entries)]
        def unexpected(*args):raise AssertionError("No new question at zero update")
        text,changes=render(rows,entries,predictions,unexpected)
        saved=list(csv.DictReader(io.StringIO(text)))
        self.assertEqual(saved,[e["submission"]["baseline"] for e in entries]);self.assertEqual(changes,[])
        predictions[0]["uncertain_prob"]=.25;predictions[1]["uncertain_prob"]=.75
        text,changes=render(rows,entries,predictions,lambda *_:("你希望表达正面还是负面的感受？","test"))
        saved=list(csv.DictReader(io.StringIO(text)))
        self.assertEqual(saved[0]["clarifying_question"],"none")
        self.assertEqual(saved[1]["is_uncertain"],"yes");self.assertEqual(len(changes),2)
        self.assertTrue(all(x["pred_emotion"]=="positive" for x in saved))


@unittest.skipUnless(importlib.util.find_spec("torch"),"torch numerical checks run on server")
class PolicyMathematics(unittest.TestCase):
    def test_expected_group_gradient_matches_exact_expected_reward(self):
        import torch
        counts={"Conflict":2,"Aligned":8};g=8;p=.25
        for arm in ("A","B"):
            for gold in (False,True):
                expected=0.
                for k in range(g+1):
                    score=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
                    loss,_=objective(score,0.,False,gold,counts,arm,actions=[1]*k+[0]*(g-k))
                    derivative,=torch.autograd.grad(loss,score)
                    expected+=math.comb(g,k)*p**k*(1-p)**(g-k)*float(derivative)
                w=reward_weight("Conflict" if gold else "Aligned",counts,arm)
                exact=-w*p*(1-p)*(1 if gold else -1)
                self.assertAlmostEqual(expected,exact,places=12)

    def test_zero_advantage_repair_has_gradient_without_fake_rollout(self):
        import torch
        values=[]
        for arm in ("A","C"):
            s=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
            loss,m=objective(s,0.,False,True,{"Conflict":2,"Aligned":8},arm,actions=[0]*8)
            derivative,=torch.autograd.grad(loss,s);values.append(float(derivative))
            self.assertEqual(m["zero_advantage"],1)
        self.assertAlmostEqual(values[0],0.,places=12);self.assertLess(values[1],0.)

    def test_protection_only_applies_to_baseline_correct_aligned(self):
        import torch
        for b,y,should in ((False,False,True),(True,False,False),(False,True,False)):
            s=torch.tensor(.7,dtype=torch.float64,requires_grad=True)
            _,m=objective(s,0.,b,y,{"Conflict":2,"Aligned":8},"C",actions=[0,1]*4)
            self.assertEqual(m["keep_loss"]>0,should)

    def test_mean_not_selected_count_scales_repair(self):
        import torch
        counts={"Conflict":2,"Aligned":8}
        score=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
        loss,_=objective(score,0.,False,True,counts,"C",actions=[0]*8)
        full,=torch.autograd.grad(loss,score,retain_graph=True)
        half,=torch.autograd.grad(loss/2,score)
        self.assertAlmostEqual(float(half),float(full)/2)

    @unittest.skipUnless(importlib.util.find_spec("peft"),"PEFT checkpoint integration runs on server")
    def test_real_peft_optimizer_rng_round_trip(self):
        import torch
        from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
        from rl.state import save_checkpoint,adapter_digest
        from scripts.train_conflict_reward import restore
        torch.manual_seed(45)
        config=LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,target_modules=["0"])
        model=get_peft_model(torch.nn.Sequential(torch.nn.Linear(4,4)),config,adapter_name="policy")
        model.add_adapter("reference",config)
        set_peft_model_state_dict(model,get_peft_model_state_dict(model,adapter_name="policy"),adapter_name="reference")
        model.set_adapter("policy")
        def trainable():
            for n,p in model.named_parameters():p.requires_grad_(".policy." in n)
        trainable()
        policy=SimpleNamespace(model=model,torch=torch,_set_trainability=trainable,
            processor=SimpleNamespace(save_pretrained=lambda path:Path(path).mkdir(parents=True,exist_ok=True)))
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        def step():
            opt.zero_grad(set_to_none=True)
            x=torch.randn(3,4)
            model(x).square().mean().backward();opt.step();sch.step()
        step()
        with tempfile.TemporaryDirectory() as folder:
            cp=save_checkpoint(Path(folder),policy,opt,sch,1,{"test":"checkpoint"})
            step();expected=adapter_digest(policy,"policy")
            restore(policy,opt,sch,cp,{"test":"checkpoint"})
            step();self.assertEqual(adapter_digest(policy,"policy"),expected)


class FakePipeline(Pipeline):
    def __init__(self,root,fail=None):
        self.root=Path(root);self.plan={"identity_hash":"fake"};self.pilot=True;self.events=[];self.fail=fail
        immutable(self.root/"preflight/preflight.complete.json",{"passed":True,"resume_identical":True,"peak_reserved_gib":1})
    def context(self,grouping,fold):
        path=self.root/f"ref-{grouping}-{fold}.json";immutable(path,{"grouping":grouping,"fold":fold});return path
    def cache(self,context):pass
    def state(self,phase=None,**extra):self.events.append(("phase",phase))
    def parallel(self,tasks):return [fn(*args) for fn,args in tasks]
    def job(self,context,arm,seed):
        from scripts.joint_affect_pipeline import read
        c=read(context);self.events.append(("job",c["grouping"],c["fold"],arm,seed))
        path=self.root/f"jobs/{c['grouping']}/{arm}/{seed}/{c['fold']}"
        immutable(path/"train.complete.json",{"fake":True});return path
    def report(self,arm,seed=42,grouping="content",folds=(0,1,2),stage="primary"):
        self.events.append(("report",stage,arm))
        return {"passed":stage!=self.fail,"candidate":{"selection_score":.90+(.01 if arm=="B" else 0)}}
    def official(self,*args):self.events.append(("official",))


class Scheduling(unittest.TestCase):
    def run_case(self,failure):
        with tempfile.TemporaryDirectory() as root:
            p=FakePipeline(root,failure);result=p.cohort();return result,p.events
    def test_failed_pilot_never_touches_later_folds_or_official_data(self):
        result,events=self.run_case("pilot")
        self.assertEqual(result["reason"],"pilot_futility_gate_not_met")
        jobs=[e for e in events if e[0]=="job"]
        self.assertEqual(len(jobs),3);self.assertTrue(all(e[2]==0 for e in jobs))
        self.assertNotIn(("official",),events)
    def test_failed_primary_does_not_start_replication(self):
        result,events=self.run_case("primary")
        self.assertEqual(result["reason"],"primary_gate_not_met")
        self.assertFalse(any(e[0]=="job" and e[-1]==43 for e in events))
    def test_failed_replication_does_not_train_parent_reference(self):
        result,events=self.run_case("replication")
        self.assertEqual(result["reason"],"RL_seed43_gate_not_met")
        self.assertFalse(any(e[0]=="job" and e[1]=="parent" for e in events))
    def test_failed_pressure_does_not_train_full_models(self):
        result,events=self.run_case("pressure")
        self.assertEqual(result["reason"],"parent_pressure_gate_not_met")
        self.assertFalse(any(e[0]=="job" and e[2]=="full" for e in events))
    def test_delivery_requires_all_gates(self):
        result,events=self.run_case(None)
        self.assertTrue(result["candidate_created"]);self.assertEqual(result["winner"],"B")
        self.assertEqual(events[-1],("official",))
        self.assertEqual(sum(e[0]=="job" and e[2]=="full" for e in events),2)


if __name__=="__main__":unittest.main()
