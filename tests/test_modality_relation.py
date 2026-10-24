"""Decision contracts, real-data split safety, dense gradients and exact recovery."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import types
from unittest.mock import patch

from rl.experiment import ROOT, fingerprint, write_json
from rl.grounded_uncertainty import load_training
from rl.modality_relation import (DEFAULTS,METHODS,STATES,allowed_states,build_model,
    conflict_probability,epoch_order,gate,make_folds,row_losses,selection_score,training_weights)
from rl.relation_store import feature_entry,load_head,save_feature,save_head
from scripts.relation_dashboard import render


def row(v="negative",m="positive",cv=True,cm=True,protocol="VT"):
    return {"sample_id":"s","video":"videos/s.mp4","protocol":protocol,
        "gt_label":"negative",
        "sample_type":"Conflict" if v!=m else "Aligned","source_dataset":"ch_sims_v2",
        "split_group_id":"g","natural":True,"view_targets":{"vision":{"label":v,"clear":cv},
        "text" if protocol=="VT" else "audio":{"label":m,"clear":cm}}}


class Contracts(unittest.TestCase):
    def test_small_terminal_retains_loss_score_and_stop_reason_without_scrolling(self):
        with tempfile.TemporaryDirectory() as d,patch('scripts.relation_dashboard.gpu_summary',return_value='GPU TEST'):
            root=Path(d)
            write_json(root/'pipeline.status.json',{'phase':'complete','reason':'primary_gate_not_met'})
            write_json(root/'results/baseline.json',{'metrics':{'score':.712673}})
            terminal=types.SimpleNamespace(columns=150,lines=18)
            with patch('scripts.relation_dashboard.shutil.get_terminal_size',return_value=terminal):
                text=render(root)
            self.assertLessEqual(len(text.splitlines()),17)
            self.assertIn('LOSS:',text)
            self.assertIn('0.712673',text)
            self.assertIn('正常结束',text)

    def test_controller_stops_before_replication_or_test_when_primary_fails(self):
        import rl.train_modality_relation as app
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            rows=[{**row(),"sample_id":"c"},{**row(m="negative"),"sample_id":"a"}]
            c={"identity_hash":"unit","val":rows,"settings":DEFAULTS,"started":0,
               "baselines":{"val":{r["sample_id"]:{"pred_emotion":"negative","is_uncertain":"no","clarifying_question":"none"} for r in rows}}}
            score=app.extra_report(app.baseline_records(c))["score"]
            called=[]
            def pool(args,jobs):
                for command,role,extra in jobs:
                    called.append((command,role))
                    if command=="preflight":write_json(root/"preflight.json",{"passed":True,"identity_hash":"unit"})
                    if role.startswith("fit-"):
                        method=extra[extra.index("--method")+1]
                        write_json(root/"results"/f"{method}-seed42.json",{"method":method,"seed":42,"epochs":3,
                            "metrics":{"score":score+.001,"counts":{"aligned_false_positive":{"correct":0}}}})
            args=types.SimpleNamespace(run_dir=root)
            fake_fcntl=types.SimpleNamespace(flock=lambda *a:None,LOCK_EX=1,LOCK_NB=2)
            with patch.dict('sys.modules',{'fcntl':fake_fcntl}),patch.object(app,'initialize',return_value=c),\
                 patch.object(app,'run_pool',side_effect=pool),patch.object(app,'publish'),\
                 patch.object(app,'choose_epochs',return_value={m:{"epoch":3} for m in METHODS}):
                app.pipeline(args)
                count_before=len(called)
                app.pipeline(args)
                self.assertEqual(len(called),count_before)
            self.assertEqual(json.loads((root/'final-report.json').read_text())["reason"],"primary_gate_not_met")
            self.assertFalse(any(command in ('questions','export') or '43' in role for command,role in called))
            self.assertFalse((root/'submission.csv').exists())

    def test_oof_uses_each_row_once_and_ties_choose_earlier_epoch(self):
        from rl.train_modality_relation import choose_epochs
        rows=[]
        for i in range(6):
            r=row(m="positive" if i%2 else "negative")
            r.update(sample_id=str(i),split_group_id=str(i))
            rows.append(r)
        c={"identity_hash":"unit","train":rows,"baselines":{"train":{r["sample_id"]:{"pred_emotion":"negative"} for r in rows}}}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for method in METHODS:
                for fold in range(3):
                    ids=[2*fold,2*fold+1]
                    history=[{"epoch":e,"probabilities":[.1,.9] if e>=3 else [.1,.1]} for e in range(1,41)]
                    write_json(root/"heads"/f"{method}-fold{fold}-seed42"/"done.json",{"history":history,"held_indices":ids})
            choices=choose_epochs(root,c)
            self.assertTrue(all(v["epoch"]==3 for v in choices.values()))

    def test_clear_and_partial_state_constraints(self):
        self.assertEqual(sum(allowed_states(row())),1)
        r=row(m="neutral",cm=False)
        expected={STATES[i] for i,k in enumerate(allowed_states(r)) if k}
        self.assertEqual(expected,{("negative","positive"),("negative","neutral")})
        self.assertTrue(all(a!=b for a,b in expected))
        self.assertEqual(sum(allowed_states(row(cv=False,cm=False))),6)
        broken=row();broken["sample_type"]="Aligned"
        with self.assertRaises(ValueError): allowed_states(broken)

    def test_class_cost_weights_and_exact_selection(self):
        rows=[row()]+[row(m="negative") for _ in range(9)]
        weights=training_weights(rows)
        self.assertAlmostEqual(sum(weights[:1])/len(rows),.6)
        self.assertAlmostEqual(sum(weights[1:])/len(rows),.4)
        self.assertAlmostEqual(selection_score(rows,[1.]+[0.]*9),.5)
        self.assertAlmostEqual(selection_score(rows,[0.]*10),.2)
        # A score exactly at .5 is no, consistently in training and export.
        self.assertAlmostEqual(selection_score(rows,[.5]*10),.2)

    def test_gate_never_accepts_pending_or_invalid_outputs(self):
        base={"score":.71}
        good={"score":.725,"invalid_outputs":0,"inconsistent_questions":0,"emotion_changes":0}
        self.assertTrue(gate(base,good,.015)["passed"])
        for key in ("invalid_outputs","inconsistent_questions","emotion_changes"):
            self.assertFalse(gate(base,{**good,key:None},.015)["passed"])
            self.assertFalse(gate(base,{**good,key:1},.015)["passed"])
        self.assertFalse(gate(base,{"score":.99},.015)["passed"])
        self.assertFalse(gate(base,{**good,"score":.724},.015)["passed"])

    def test_epochs_are_complete_and_reproducible(self):
        self.assertEqual(epoch_order(3354,42,0),epoch_order(3354,42,0))
        self.assertNotEqual(epoch_order(3354,42,0),epoch_order(3354,42,1))
        self.assertEqual(sorted(epoch_order(3354,42,0)),list(range(3354)))

    def test_actual_dataset_group_folds_and_unclear_counts(self):
        path=ROOT/"Dataset/train.jsonl"
        if not path.exists(): self.skipTest("raw data unavailable in this checkout")
        rows=load_training(path)
        f=make_folds(rows)
        self.assertEqual(len(rows),3354)
        self.assertEqual(sum(r["sample_type"]=="Conflict" for r in rows),411)
        assignments={}
        for r,k in zip(rows,f):
            self.assertEqual(assignments.setdefault(r["split_group_id"],k),k)
            allowed_states(r)
        self.assertEqual(len(assignments),2104)
        rev=list(reversed(rows))
        self.assertEqual({r["sample_id"]:k for r,k in zip(rows,f)},
                         {r["sample_id"]:k for r,k in zip(rev,make_folds(rev))})
        self.assertEqual(sum(sum(allowed_states(r))>1 for r in rows),83)
        for fold in range(3):
            self.assertTrue(any(r["natural"] and r["protocol"]=="VA" and r["sample_type"]=="Conflict"
                                for r,k in zip(rows,f) if k==fold))

    def test_dashboard_keeps_development_and_formal_scores_distinct(self):
        with tempfile.TemporaryDirectory() as d,patch("scripts.relation_dashboard.gpu_summary",return_value="GPU TEST"):
            root=Path(d)
            write_json(root/"context.json",{"settings":DEFAULTS})
            write_json(root/"pipeline.status.json",{"phase":"complete","reason":"primary_gate_not_met"})
            write_json(root/"results/baseline.json",{"metrics":{"score":.712673}})
            write_json(root/"development.json",{"choices":{"joint_relation":{"epoch":8,"selection_score":.42}}})
            write_json(root/"fit.status.json",{"phase":"complete","updated":10,"loss":.4,"expected_reward":.8,
                "epoch":8,"epochs":8,"step":424,"total_steps":424,"epoch_seen":3354,"epoch_total":3354})
            with patch("scripts.relation_dashboard.shutil.get_terminal_size",return_value=type("S",(),{"columns":250})()):
                text=render(root)
            self.assertIn("LOSS: 0.400000",text)
            self.assertIn("开发任务分 0.420000",text)
            self.assertIn("本地原始 v4: 0.712673",text)
            self.assertIn("正常结束",text)
            self.assertIn("时间限制: 无",text)
            self.assertIn("未生成",text)
            self.assertNotIn("ERROR",text)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"),"torch/numpy required")
class Numerics(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch=torch
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)

    def test_allowed_log_loss_matches_explicit_sum_and_cross_entropy(self):
        t=self.torch
        logits=t.randn(3,9,requires_grad=True)
        rows=[row(),row(m="neutral",cm=False),row(m="negative")]
        masks=t.tensor([allowed_states(r) for r in rows])
        values=row_losses(logits,"joint_relation",masks,t.tensor([1,1,0]))
        explicit=-(logits.softmax(-1)*masks).sum(-1).log()
        t.testing.assert_close(values,explicit)
        target=t.tensor([STATES.index(("negative","positive"))])
        t.testing.assert_close(values[:1],t.nn.functional.cross_entropy(logits[:1],target,reduction="none"))
        weights=t.tensor(training_weights(rows))
        weighted=(values*weights).mean()
        manual=.6*values[:2].mean()+.4*values[2]
        t.testing.assert_close(weighted,manual)

    def test_extreme_wrong_action_has_correct_nonzero_gradient(self):
        t=self.torch
        logits=t.tensor([[100.,-100.,-100.,-100.,-100.,-100.,-100.,-100.,-100.]],requires_grad=True)
        mask=t.zeros(1,9,dtype=t.bool);mask[0,3]=True
        loss=row_losses(logits,"joint_relation",mask,t.tensor([1])).sum()
        loss.backward()
        self.assertTrue(t.isfinite(loss))
        self.assertLess(logits.grad[0,3].item(),-.99)
        self.assertGreater(logits.grad[0,0].item(),.99)
        # All four legacy sampled actions could be wrong; enumerated loss still corrects.
        binary=t.tensor([[100.,-100.]],requires_grad=True)
        row_losses(binary,"binary_control",mask,t.tensor([1])).sum().backward()
        self.assertLess(binary.grad[0,1].item(),-.99)

    def test_joint_probability_and_batched_predictions(self):
        t=self.torch
        from rl.relation_training import predict
        model=build_model("joint_relation")
        x=t.randn(5,3,2048)
        protocol=t.tensor([0,1,0,1,1])
        a,states=predict(model,x,protocol,"joint_relation",t,batch_size=1)
        b,_=predict(model,x,protocol,"joint_relation",t,batch_size=3)
        t.testing.assert_close(t.tensor(a),t.tensor(b),atol=1e-6,rtol=1e-6)
        for q,p in zip(a,states):
            self.assertAlmostEqual(sum(p),1.,places=6)
            self.assertAlmostEqual(q,sum(p[i] for i,(v,m) in enumerate(STATES) if v!=m),places=6)
        self.assertLess(sum(p.numel() for p in model.parameters()),1_000_000)

    def test_feature_corruption_and_checkpoint_algorithm_are_rejected(self):
        t=self.torch
        c={"identity_hash":"unit"}
        r=row()
        with tempfile.TemporaryDirectory() as d:
            save_feature(d,c,"train",r,t.randn(3,2048),{},t)
            entry,x=feature_entry(d,c,"train",r,t,required=True)
            from rl.relation_store import feature_file
            feature_file(d,"train",r["sample_id"]).write_bytes(b"corrupt")
            with self.assertRaises(ValueError):feature_entry(d,c,"train",r,t,required=True)
            payload={"method":"binary_control","epoch":1,"seed":42,"fold":0,"train_ids_hash":"ids"}
            path=Path(d)/"head.pt"
            save_head(path,payload,c,t)
            self.assertEqual(load_head(path,c,t),payload)
            marker=json.loads(Path(str(path)+".json").read_text())
            marker["algorithm"]="old_grpo"
            write_json(str(path)+".json",marker)
            with self.assertRaises(ValueError):load_head(path,c,t)

    def test_optimizer_rng_schedule_and_next_update_restore_exactly(self):
        t=self.torch
        from rl.relation_training import optimizer_for,seed_all,update
        from rl.state import capture_rng,restore_rng
        seed_all(t,42)
        model=build_model("joint_relation")
        opt,schedule=optimizer_for(model,DEFAULTS,3,t)
        x=t.randn(4,3,2048)
        p=t.tensor([0,1,0,1])
        targets=t.tensor([1,1,0,0])
        masks=t.tensor([allowed_states(row()),allowed_states(row(protocol="VA")),
                        allowed_states(row(m="negative")),allowed_states(row(m="negative",protocol="VA"))])
        weights=t.tensor(training_weights([row(),row(),row(m="negative"),row(m="negative")]))
        update(model,opt,schedule,x,p,masks,targets,weights,"joint_relation",t)
        saved=copy.deepcopy({"model":model.state_dict(),"optimizer":opt.state_dict(),
                             "scheduler":schedule.state_dict(),"rng":capture_rng(t)})
        first=update(model,opt,schedule,x,p,masks,targets,weights,"joint_relation",t)
        expected=copy.deepcopy(model.state_dict())
        expected_opt=copy.deepcopy(opt.state_dict())
        model.load_state_dict(saved["model"]);opt.load_state_dict(saved["optimizer"])
        schedule.load_state_dict(saved["scheduler"]);restore_rng(t,saved["rng"])
        second=update(model,opt,schedule,x,p,masks,targets,weights,"joint_relation",t)
        self.assertEqual(first,second)
        for k,v in model.state_dict().items(): self.assertTrue(t.equal(v,expected[k]),k)
        for k,v in opt.state_dict()["state"].items():
            for key,item in v.items():
                self.assertTrue(t.equal(item,expected_opt["state"][k][key]))


if __name__=="__main__":unittest.main()
