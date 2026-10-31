"""Meaningful numerical and isolation tests for the locked experiment."""
from __future__ import annotations

import csv
import math
from pathlib import Path
import tempfile
import unittest

from rl.emotion_expert import (SETTINGS,admission,batches,class_weight,choose,export,
    gate,input_only,instruction,objective,targets,weights,evaluate)


def row(i,conflict=False,protocol="VT"):
    return {"sample_id":str(i),"video":str(i)+".mp4","protocol":protocol,"text_content":"DO_NOT_LEAK_VA",
        "sample_type":"Conflict" if conflict else "Aligned","source_dataset":"ch_sims_v2",
        "views":{"M1":{"label":"positive","is_clear":True},
                 "M2":{"label":"negative" if conflict else "positive","is_clear":True},
                 "M12":{"label":"negative","is_clear":True}}}


class Contracts(unittest.TestCase):
    def test_gpu2_only_and_serial_admission(self):
        self.assertEqual(SETTINGS["allowed_gpus"],[2])
        self.assertEqual(admission(1000,3000,{"full":2000,"supervised":1000,"frozen":500}),"frozen")

    def test_population_reward_matches_original_metric(self):
        c={"N":3354,"C":411,"A":2943}
        score=c["C"]/c["N"]*class_weight(1,c)*.72+c["A"]/c["N"]*class_weight(0,c)*.94
        self.assertAlmostEqual(score,.3*.72+.2*.94)
        self.assertAlmostEqual(class_weight(1,c,2),2*class_weight(1,c))
        self.assertEqual(class_weight(0,c,2),class_weight(0,c))

    def test_no_input_metadata_or_va_text_leakage(self):
        r=row(1,True,"VA");r["quality_flags"]=["GOLD_SECRET"]
        for task in ("E","U","V","M"):
            self.assertNotIn("DO_NOT_LEAK_VA",instruction(r,task))
            self.assertNotIn("GOLD_SECRET",instruction(r,task))
        self.assertNotIn("DO_NOT_LEAK_VA",instruction(row(2),"V"))
        self.assertEqual(set(input_only(r)),{"video","protocol","text_content"})

    def test_unclear_is_not_neutral(self):
        r=row(1);r["views"]["M1"].update(label="neutral",is_clear=False)
        self.assertEqual(targets(r)["V"],3)
        r["views"]["M1"]["is_clear"]=True
        self.assertEqual(targets(r)["V"],1)

    def test_each_original_row_once_last_partial_batch(self):
        rr=[row(i,i%5==0) for i in range(2013)];bb=batches(rr)
        self.assertEqual(len(bb),126);self.assertEqual(len(bb[-1]),13)
        self.assertEqual({r["sample_id"] for b in bb for r in b},{r["sample_id"] for r in rr})
        self.assertEqual(bb,batches(rr))

    def test_visual_duplicate_weights_are_normalized(self):
        rr=[row(i) for i in range(3)];rr[1]["video"]=rr[0]["video"]
        w=weights(rr)
        self.assertAlmostEqual(sum(w.values()),3)
        self.assertAlmostEqual(w["0"]+w["1"],w["2"])

    def test_selection_uses_dev_components_and_rl_gate(self):
        def metric(f1,u,va=2):return {"f1_pn":f1,"uncertainty_component":u,"groups":{"natural_VA":{"correct":va}}}
        dev={"E0":metric(.90,.4),"SE":metric(.92,.4),"U0":metric(.90,.4),"S":metric(.90,.405),
             "R1":metric(.90,.408),"R2":metric(.90,.42,1)}
        selected=choose(dev,["E0","SE"],["U0","S","R1","R2"])
        self.assertEqual(selected["emotion"],"SE");self.assertEqual(selected["uncertainty"],"R1")

    def test_gate_cannot_use_reward_lambda_to_change_metrics(self):
        base={"selection_score":.88,"f1_pn":.94,"groups":{"natural":{"correct":10},"natural_VA":{"correct":2}}}
        candidate={**base,"selection_score":.90,"groups":{"natural":{"correct":11},"natural_VA":{"correct":1}}}
        self.assertFalse(gate(candidate,base)["passed"])

    def test_export_keeps_order_and_enforces_question_consistency(self):
        rr=[row(i,i%7==0,"VA" if i%2 else "VT") for i in range(734)]
        e={r["sample_id"]:{"pred_emotion":"positive"} for r in rr}
        u={r["sample_id"]:{"uncertain_prob":.9 if i%7==0 else .1} for i,r in enumerate(rr)}
        scratch=Path(__file__).resolve().parents[1]/"tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            path=Path(tmp)/"submission.csv";report=export(rr,e,u,path)
            with path.open(encoding="utf-8",newline="") as f:out=list(csv.DictReader(f))
            self.assertEqual([r["sample_id"] for r in out],[r["sample_id"] for r in rr])
            self.assertTrue(all((r["clarifying_question"]=="none")== (r["is_uncertain"]=="no") for r in out))
            self.assertEqual(report["rows"],734)

    def test_metrics_and_paired_group_bootstrap(self):
        from scripts.emotion_expert_pipeline import bootstrap_interval
        rr=[row(i,i%3==0,"VA" if i%2 else "VT") for i in range(18)]
        e={r["sample_id"]:{"pred_emotion":"negative"} for r in rr}
        u={r["sample_id"]:{"uncertain_prob":float(r["sample_type"]=="Conflict"),
           "vision_probs":[0,0,1,0],"second_probs":[1,0,0,0] if r["sample_type"]=="Conflict" else [0,0,1,0]} for r in rr}
        result=evaluate(rr,e,u)
        self.assertEqual(result["tp"],6);self.assertEqual(result["fp"],0)
        self.assertEqual(result["evidence"]["both_correct"],18)
        interval=bootstrap_interval(rr,(e,u),(e,u),{r["video"]:str(i//2) for i,r in enumerate(rr)},repeats=20)
        self.assertEqual(interval["percentile_95"],[0.,0.])


class Numerical(unittest.TestCase):
    def test_frozen_quantized_bias_cast_is_explicit_and_idempotent(self):
        import torch
        from rl.emotion_expert_runtime import normalize_quantized_biases
        class Quantized(torch.nn.Linear):pass
        model=torch.nn.Sequential(Quantized(3,2),torch.nn.Linear(2,2))
        model.requires_grad_(False)
        expected=model[0].bias.detach().to(torch.bfloat16)
        untouched=model[1].bias.detach().clone()
        changes=normalize_quantized_biases(model,Quantized,torch.bfloat16)
        self.assertEqual(len(changes),1)
        self.assertTrue(torch.equal(model[0].bias,expected))
        self.assertTrue(torch.equal(model[1].bias,untouched))
        self.assertEqual(normalize_quantized_biases(model,Quantized,torch.bfloat16),[])

    def test_exact_expected_reward_gradient_and_dense_supervision(self):
        import torch
        c={"N":100,"C":10,"A":90}
        x=torch.tensor([0.,-8.],dtype=torch.float64,requires_grad=True)
        ref=x.detach().softmax(-1)
        s,_=objective(x,1,c,phase="S");gs=torch.autograd.grad(s,x)[0]
        r1,_=objective(x,1,c,phase="R1",reference=ref);g1=torch.autograd.grad(r1,x)[0]
        r2,_=objective(x,1,c,phase="R2",reference=ref);g2=torch.autograd.grad(r2,x)[0]
        p=x.detach().softmax(-1)[1]
        expected=class_weight(1,c)*p*(1-p)
        self.assertAlmostEqual(float(g1[1]-gs[1]),-float(expected),places=10)
        self.assertAlmostEqual(float(g2[1]-g1[1]),-float(expected),places=10)
        self.assertLess(float(gs[1]),-2.9)

    def test_aligned_rl_is_identical_in_both_lambda_arms(self):
        import torch
        x=torch.tensor([-.3,.8],dtype=torch.float64,requires_grad=True);c={"N":100,"C":10,"A":90}
        ref=[.6,.4]
        a,_=objective(x,0,c,phase="R1",reference=ref);b,_=objective(x,0,c,phase="R2",reference=ref)
        self.assertEqual(float(a.detach()),float(b.detach()))


if __name__=="__main__":unittest.main()
