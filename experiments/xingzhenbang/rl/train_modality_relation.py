"""Bounded frozen-Omni relation/control experiment, from features to gated CSV."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .experiment import file_hash, fingerprint, write_json
from .finite_reports import binomial_count, comparison, report
from .modality_relation import METHODS, STATES, VERSION, gate, paired_diagnostics, selection_score
from .relation_store import context_for, feature_entry, feature_file, initialize, read
from .reward import parse_completion, question_is_consistent


def emit(run,role,phase,**values):
    value={"phase":phase,"pid":os.getpid(),"updated":time.time(),**values}
    write_json(Path(run)/(role+".status.json"),value)
    line=json.dumps({"role":role,**value},ensure_ascii=False)
    slot=os.environ.get("BUAA_RELATION_GPU_SLOT")
    if slot in ("0","1"):
        log=Path(run)/"logs"/f"gpu-{slot}.log"
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open("a",encoding="utf-8") as stream:
            stream.write(line+"\n")
    print(line,flush=True)


def torch_device(args,seed=42):
    # GPU selection must happen before any CUDA/RNG/memory initialization.
    from .grounded_recovery import prepare_environment, configure_torch
    from .omni_runtime import select_gpu
    prepare_environment()
    device=select_gpu(args.gpu,min_free_memory_gib=12 if args.command in ("preflight","cache","questions","export") else 2)
    import torch
    from .relation_training import seed_all
    torch.set_num_threads(args.cpu_threads)
    torch.cuda.set_per_process_memory_fraction(min(1.,args.memory_gib*1024/device["total_mib"]),0)
    configure_torch(torch)
    seed_all(torch,seed)
    print(json.dumps({"gpu":device,"allocator_limit_gib":args.memory_gib,"seed":seed}),flush=True)
    return torch


def extractor(args,c):
    torch=torch_device(args)
    from .relation_runtime import RelationExtractor
    from .grounded_store import TaskCache
    source_root=Path(c["source_run"])
    source_context=read(source_root/"context.json")
    if (source_context["identity_hash"]!=c["identity"]["source_identity"] or
            source_context["identity_hash"]!=fingerprint(source_context["identity"]) or
            file_hash(source_root/"cache/complete.json")!=c["identity"]["source_manifest_sha256"]):
        raise ValueError("Frozen source cache changed since preparation")
    policy=RelationExtractor(c["model"],c["adapter"],c["settings"])
    source=TaskCache(c["source_run"],source_context,torch)
    return policy,source


def frozen_state(policy):
    from .state import adapter_digest
    if any(p.requires_grad or p.grad is not None for p in policy.model.parameters()):
        raise RuntimeError("Omni parameter acquired a gradient")
    return {"versions":{n:int(p._version) for n,p in policy.model.named_parameters()},
            "reference":adapter_digest(policy,"reference")}


def preflight(args,c):
    from .omni_io import prepare_omni_inputs, prepare_task_inputs, task_messages, messages_for
    from .relation_store import save_feature
    policy,source=extractor(args,c)
    torch=policy.torch
    before=frozen_state(policy)
    rows=[]
    for protocol in ("VT","VA"):
        for kind in ("Conflict","Aligned"):
            rows.append(next(r for r in c["train"] if r["protocol"]==protocol and r["sample_type"]==kind))
    tasks=[("train",r) for r in rows]+[("val",next(r for r in c["val"] if r["protocol"]==p)) for p in ("VT","VA")]
    checks,times,probe_features=[],{"train":[],"val":[]},[]
    for i,(split,row) in enumerate(tasks):
        baseline=c["baselines"][split][row["sample_id"]]
        x,diagnostic=policy.extract(row,split,c,source,baseline)
        if split=="train": probe_features.append(x)
        times[split].append(diagnostic["seconds"])
        for mode in ("vision","text" if row["protocol"]=="VT" else "audio","full"):
            raw=(prepare_omni_inputs(policy.processor,row,c["data_root"],policy.settings) if mode=="full" else
                 prepare_task_inputs(policy.processor,row,c["data_root"],policy.settings,mode))
            h,logits=policy.hidden(raw,None,mode,baseline["pred_emotion"] if mode=="full" else None)
            slot=2 if mode=="full" else 0 if mode=="vision" else 1
            cached_logits=torch.tensor(diagnostic[mode]["logits"],device=logits.device)
            if not torch.allclose(h.cpu(),x[slot],atol=.01,rtol=1e-4) or not torch.allclose(logits,cached_logits,atol=.01,rtol=1e-4):
                raise RuntimeError("Raw and cached feature paths differ: "+mode)
            if mode=="full":
                reference=torch.tensor(source.entry(row,"full")["reference_scores"],device=logits.device)
                if not torch.allclose(reference,logits,atol=.01,rtol=1e-4):
                    raise RuntimeError("Original v4 native decision score was not reproduced")
            changed={**row,"source_dataset":"FORBIDDEN_SOURCE","sample_type":"FORBIDDEN_LABEL",
                     "gt_label":"FORBIDDEN_GOLD","view_targets":{},"natural":not row["natural"],"is_clear":False}
            path=Path(c["data_root"])/row["video"]
            builder=messages_for if mode=="full" else lambda r,p:task_messages(r,p,mode)
            if builder(row,path)!=builder(changed,path):
                raise RuntimeError("Labels or metadata leaked into the rendered input")
            checks.append({"sample_id":row["sample_id"],"split":split,"mode":mode,
                "hidden_max_abs":float((h.cpu()-x[slot]).abs().max()),
                "logit_max_abs":float((logits-cached_logits).abs().max())})
        save_feature(args.run_dir,c,split,row,x,diagnostic,torch)
        emit(args.run_dir,"preflight","raw_cached_and_modality_check",completed=i+1,total=len(tasks))
    head_check=check_head_recovery(args,c,rows,probe_features,torch)
    if before!=frozen_state(policy):
        raise RuntimeError("Feature extraction changed frozen Omni parameters")
    split_times={k:sum(v)/len(v) for k,v in times.items()}
    total_seconds=sum(split_times[k]*len(c[k]) for k in split_times)
    average=total_seconds/(len(c["train"])+len(c["val"]))
    value={"passed":True,"identity_hash":c["identity_hash"],"checks":checks,"head_recovery":head_check,"seconds_per_record_cached":average,
        "seconds_per_record_by_split":split_times,"estimated_feature_hours_two_gpus":total_seconds/2/3600,
        "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
        "frozen_reference":before["reference"],"note":"Estimate includes sampled validation view preprocessing; shared-GPU load, length distribution, initialization and final questions can change total time."}
    write_json(Path(args.run_dir)/"preflight.json",value)
    emit(args.run_dir,"preflight","complete",**value)


def check_head_recovery(args,c,rows,features,torch):
    import copy
    from .modality_relation import build_model,training_weights
    from .relation_store import save_head,load_head
    from .relation_training import optimizer_for,seed_all,tensors,update
    from .state import capture_rng,restore_rng
    seed_all(torch,42)
    model=build_model("joint_relation",c["settings"]).to("cuda:0")
    optimizer,scheduler=optimizer_for(model,c["settings"],1,torch)
    x=torch.stack(features).to("cuda:0")
    protocol,masks,targets=tensors(rows,torch,"cuda:0")
    weights=torch.tensor(training_weights(rows),device="cuda:0")
    update(model,optimizer,scheduler,x,protocol,masks,targets,weights,"joint_relation",torch)
    path=Path(args.run_dir)/"preflight-head.pt"
    payload={"method":"joint_relation","epoch":1,"seed":42,"fold":-2,
        "train_ids_hash":fingerprint([r["sample_id"] for r in rows]),"model":model.state_dict(),
        "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"rng":capture_rng(torch)}
    save_head(path,payload,c,torch)
    first=update(model,optimizer,scheduler,x,protocol,masks,targets,weights,"joint_relation",torch)
    expected=copy.deepcopy(model.state_dict())
    expected_optimizer=copy.deepcopy(optimizer.state_dict())
    expected_scheduler=copy.deepcopy(scheduler.state_dict())
    restored=load_head(path,c,torch)
    model.load_state_dict(restored["model"])
    optimizer.load_state_dict(restored["optimizer"])
    scheduler.load_state_dict(restored["scheduler"])
    restore_rng(torch,restored["rng"])
    second=update(model,optimizer,scheduler,x,protocol,masks,targets,weights,"joint_relation",torch)
    if first!=second or any(not torch.equal(v,expected[k]) for k,v in model.state_dict().items()) or scheduler.state_dict()!=expected_scheduler:
        raise RuntimeError("New-head CUDA checkpoint/RNG/next update did not restore exactly")
    for key,values in optimizer.state_dict()["state"].items():
        for name,value in values.items():
            if not torch.equal(value,expected_optimizer["state"][key][name]):
                raise RuntimeError("New-head Adam state changed after restoration")
    return {"passed":True,"exact_next_update":True,"first_loss":first["loss"],"restored_loss":second["loss"]}


def cache(args,c):
    from .relation_store import save_feature
    if args.split=="test":
        qualified_candidates(args.run_dir,c)
    policy,source=extractor(args,c)
    before=frozen_state(policy)
    splits=("test",) if args.split=="test" else ("train","val")
    tasks=[(split,row) for split in splits for row in c[split]][args.shard::2]
    for i,(split,row) in enumerate(tasks):
        if feature_entry(args.run_dir,c,split,row,policy.torch) is None:
            baseline=(c["baseline_test"][[r["sample_id"] for r in c["test"]].index(row["sample_id"])] if split=="test" else
                      c["baselines"][split][row["sample_id"]])
            x,diagnostic=policy.extract(row,split,c,source,baseline)
            save_feature(args.run_dir,c,split,row,x,diagnostic,policy.torch)
        emit(args.run_dir,args.role,"extracting_frozen_features",completed=i+1,total=len(tasks),sample_id=row["sample_id"])
    if before!=frozen_state(policy):
        raise RuntimeError("Omni changed while building features")
    emit(args.run_dir,args.role,"complete",completed=len(tasks),total=len(tasks))


def publish(run,c,splits):
    for split in splits:
        entries=[]
        for row in c[split]:
            path=feature_file(run,split,row["sample_id"])
            e=read(path.with_suffix(".json"))
            if (e["identity_hash"],e["sample_id"],e["split"],e["file"])!=(c["identity_hash"],row["sample_id"],split,path.name) or file_hash(path)!=e["sha256"]:
                raise ValueError("Incomplete or corrupt extracted feature shard")
            entries.append(e)
        write_json(Path(run)/"features"/split/"complete.json",{"identity_hash":c["identity_hash"],
            "entries":entries,"sample_order":[r["sample_id"] for r in c[split]]})


def baseline_records(c,split="val"):
    from .metrics import prediction_record
    rows=[]
    for r in c[split]:
        b=c["baselines"][split][r["sample_id"]]
        completion=json.dumps({k:b[k] for k in ("pred_emotion","is_uncertain","clarifying_question")},ensure_ascii=False)
        rows.append({**prediction_record(r,completion),"split_group_id":r["split_group_id"]})
    return rows


def decision_records(c,probabilities,split="val"):
    if len(probabilities)!=len(c[split]):
        raise ValueError("Decision count differs from official split")
    result=[]
    for r,p in zip(c[split],probabilities):
        b=c["baselines"][split][r["sample_id"]]
        result.append({**{k:r[k] for k in ("sample_id","gt_label","sample_type","protocol","source_dataset","split_group_id")},
            "pred_emotion":b["pred_emotion"],"is_uncertain":"yes" if float(p)>.5 else "no","conflict_score":float(p)})
    return result


def extra_report(rows):
    metrics=report(rows)
    metrics["counts"]["natural_va_conflict"]=binomial_count([r for r in rows if r["sample_type"]=="Conflict" and
        r["protocol"]=="VA" and r["source_dataset"] in ("ch_sims","ch_sims_v2")],"yes")
    return metrics


def summarize_fit(run,c,job):
    rows=decision_records(c,job["val_probabilities"])
    metrics=extra_report(rows)
    metrics.update(invalid_outputs=None,inconsistent_questions=None,emotion_changes=0,output_checks_pending=True)
    matrix=[[0]*9 for _ in range(9)]
    unclear=0
    if job["method"]=="joint_relation":
        for r,p in zip(c["val"],job["val_states"]):
            v=r["view_targets"]["vision"]
            m=r["view_targets"]["text" if r["protocol"]=="VT" else "audio"]
            if not(v["clear"] and m["clear"]):
                unclear+=1
                continue
            matrix[STATES.index((v["label"],m["label"]))][max(range(9),key=lambda i:p[i])]+=1
    value={"method":job["method"],"seed":job["seed"],"epochs":job["epochs"],"checkpoint":job["checkpoint"],
        "metrics":metrics,"comparison":comparison(baseline_records(c),rows),"paired_diagnostics":job["paired_diagnostics"],
        "joint_confusion_clear_only":matrix if job["method"]=="joint_relation" else None,
        "unclear_excluded_from_joint_confusion":unclear,"val_probabilities":job["val_probabilities"],
        "val_states":job["val_states"]}
    target=Path(run)/"results"/f"{job['method']}-seed{job['seed']}.json"
    write_json(target,value)
    write_json(target.with_name(target.stem+".decisions.json"),rows)
    return value


def fitting(args,c):
    from .relation_training import fit
    torch=torch_device(args,args.seed)
    result=fit(args.run_dir,c,args.method,args.fold,args.seed,args.target_epochs,torch,
               lambda phase,**v:emit(args.run_dir,args.role,phase,**v))
    if args.fold==-1:
        summarize_fit(args.run_dir,c,result)


def choose_epochs(run,c):
    choices={}
    curves={}
    for method in METHODS:
        jobs=[read(Path(run)/"heads"/f"{method}-fold{fold}-seed42"/"done.json") for fold in range(3)]
        curve=[]
        predictions=[]
        for epoch in range(1,41):
            q=[None]*len(c["train"])
            for job in jobs:
                record=job["history"][epoch-1]
                if record["epoch"]!=epoch:
                    raise ValueError("Development epoch history is incomplete")
                for i,p in zip(job["held_indices"],record["probabilities"]):
                    if q[i] is not None:
                        raise ValueError("Overlapping development predictions")
                    q[i]=p
            if any(p is None for p in q):
                raise ValueError("Missing development prediction")
            curve.append({"epoch":epoch,"selection_score":selection_score(c["train"],q)})
            predictions.append(q)
        best=max(curve,key=lambda v:(round(v["selection_score"],12),-v["epoch"]))
        choices[method]=best
        curves[method]=curve
        write_json(Path(run)/"results"/(method+".oof.json"),{"selected":best,"curve":curve,
            "records":decision_records(c,predictions[best["epoch"]-1],"train"),
            "note":"Development selection for the new head; original v4 previously saw this training set."})
    value={"identity_hash":c["identity_hash"],"choices":choices,"curves":curves}
    write_json(Path(run)/"development.json",value)
    return choices


def question_validation(args,c):
    from .metrics import prediction_record
    selected=read(args.candidate)
    if selected["seed"] not in (42,43) or len(selected["val_probabilities"])!=666:
        raise ValueError("Invalid candidate for output validation")
    policy,source=extractor(args,c)
    before=frozen_state(policy)
    rows=[]
    for i,(r,p) in enumerate(zip(c["val"],selected["val_probabilities"])):
        completion=policy.answer(r,"val",c,source,c["baselines"]["val"][r["sample_id"]],"yes" if p>.5 else "no")
        value={**prediction_record(r,completion),"split_group_id":r["split_group_id"],"conflict_score":p}
        if value["pred_emotion"]!=c["baselines"]["val"][r["sample_id"]]["pred_emotion"]:
            raise RuntimeError("Frozen emotion changed")
        rows.append(value)
        emit(args.run_dir,args.role,"checking_output_constraints",completed=i+1,total=666)
    if before!=frozen_state(policy):
        raise RuntimeError("Question validation changed Omni")
    metrics=extra_report(rows)
    if abs(metrics["score"]-selected["metrics"]["score"])>1e-12:
        raise RuntimeError("Question generation changed decision score")
    metrics.update(emotion_changes=0,output_checks_pending=False)
    selected.update(metrics=metrics,comparison=comparison(baseline_records(c),rows))
    write_json(args.candidate,selected)
    write_json(Path(args.candidate).with_suffix(".predictions.json"),rows)
    emit(args.run_dir,args.role,"complete",metrics=metrics)


def export(args,c):
    import io
    from .relation_store import load_head, load_matrix
    from .relation_training import build_model, predict, tensors
    primary,secondary=qualified_candidates(args.run_dir,c)
    policy,source=extractor(args,c)
    torch=policy.torch
    before=frozen_state(policy)
    payload=load_head(primary["checkpoint"],c,torch)
    if (payload["seed"],payload["fold"],payload["method"],payload["epoch"])!=(42,-1,primary["method"],primary["epochs"]):
        raise ValueError("Selected head is not the qualified seed42 final fit")
    model=build_model(primary["method"],c["settings"]).to("cuda:0")
    model.load_state_dict(payload["model"],strict=True)
    x=load_matrix(args.run_dir,c,"test",torch).to("cuda:0")
    protocol=torch.tensor([r["protocol"]=="VA" for r in c["test"]],device="cuda:0",dtype=torch.long)
    q,_=predict(model,x,protocol,primary["method"],torch)
    buffer=io.StringIO(newline="")
    fields=["sample_id","pred_emotion","is_uncertain","clarifying_question"]
    writer=csv.DictWriter(buffer,fieldnames=fields)
    writer.writeheader()
    changes=[]
    for i,(r,b,p) in enumerate(zip(c["test"],c["baseline_test"],q)):
        if r["sample_id"]!=b["sample_id"]:
            raise ValueError("Test sample order changed")
        answer=parse_completion(policy.answer(r,"test",c,source,b,"yes" if p>.5 else "no"))
        if answer is None or not question_is_consistent(answer) or answer["pred_emotion"]!=b["pred_emotion"]:
            raise ValueError("Illegal final test output")
        writer.writerow({"sample_id":r["sample_id"],**answer})
        if answer["is_uncertain"]!=b["is_uncertain"]:
            changes.append({"sample_id":r["sample_id"],"before":b["is_uncertain"],"after":answer["is_uncertain"],"conflict_score":p})
        emit(args.run_dir,args.role,"serializing_selected_submission",completed=i+1,total=734)
    if before!=frozen_state(policy):
        raise RuntimeError("Test export changed Omni")
    target=Path(args.run_dir)/"submission.csv"
    content=buffer.getvalue().encode("utf-8")
    if target.exists() and target.read_bytes()!=content:
        raise ValueError("Refusing to replace a different test CSV")
    temporary=target.with_suffix(".partial")
    temporary.write_bytes(content)
    temporary.replace(target)
    write_json(Path(args.run_dir)/"submission-manifest.json",{"identity_hash":c["identity_hash"],
        "file":str(target),"sha256":file_hash(target),"rows":734,"candidate":primary["checkpoint"],
        "baseline_sha256":c["identity"]["baseline_csv_sha256"],"changes":changes,"submitted_to_kaggle":False})
    emit(args.run_dir,args.role,"complete",rows=734,changes=len(changes),file=str(target))


def qualified_candidates(run,c):
    selected=read(Path(run)/"selected.json")
    primary=read(selected["candidate"])
    secondary=read(Path(run)/"results"/f"{primary['method']}-seed43.json")
    baseline=extra_report(baseline_records(c))
    if not gate(baseline,primary["metrics"],.015)["passed"] or not gate(baseline,secondary["metrics"],.005)["passed"]:
        raise ValueError("Both seeds must pass before test features or export")
    if primary["seed"]!=42 or secondary["seed"]!=43 or (primary["method"],primary["epochs"])!=(secondary["method"],secondary["epochs"]):
        raise ValueError("Replication method/epoch/seed changed")
    return primary,secondary


def run_pool(args,jobs):
    """At most one owned worker per assigned GPU; unrelated processes are untouched."""
    active=[]
    pending=list(jobs)
    logs=Path(args.run_dir)/"logs"
    logs.mkdir(parents=True,exist_ok=True)
    try:
        while pending or active:
            for gpu in args.gpus:
                if not pending or any(item["gpu"]==gpu for item in active):
                    continue
                command,role,extra=pending.pop(0)
                previous=Path(args.run_dir)/(role+".status.json")
                if previous.exists():
                    status=read(previous)
                    pid=status.get("pid")
                    proc=Path(f"/proc/{pid}/cmdline")
                    if status.get("phase") not in ("complete","failed") and proc.exists() and b"rl.train_modality_relation" in proc.read_bytes():
                        raise RuntimeError("An owned worker is still running; refusing duplicate GPU use: "+role)
                log=(logs/(role+".log")).open("a",encoding="utf-8")
                cmd=[sys.executable,"-u","-m","rl.train_modality_relation",command,"--run-dir",str(args.run_dir),
                     "--gpu",gpu,"--memory-gib",str(args.memory_gib),"--cpu-threads",str(args.cpu_threads),"--role",role,*extra]
                env={**os.environ,"BUAA_RELATION_GPU_SLOT":str(args.gpus.index(gpu))}
                process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
                active.append({"gpu":gpu,"role":role,"process":process,"log":log})
            for item in active[:]:
                result=item["process"].poll()
                if result is not None:
                    item["log"].close()
                    active.remove(item)
                    if result:
                        error=Path(args.run_dir)/(item["role"]+".error.json")
                        detail=read(error).get("error") if error.exists() else "see worker log"
                        raise RuntimeError(f"{item['role']} exited {result}: {detail}")
            if active:
                time.sleep(1)
    finally:
        for item in active:
            # Only subprocesses created by this controller are terminated.
            item["process"].terminate()
            try:
                item["process"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                item["process"].kill()
                item["process"].wait()
            item["log"].close()


def pipeline(args):
    import fcntl
    root=Path(args.run_dir)
    root.mkdir(parents=True,exist_ok=True)
    with (root/".controller.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c=initialize(args)
        if (root/"final-report.json").exists():
            previous=read(root/"final-report.json")
            if previous.get("identity_hash")!=c["identity_hash"]:
                raise ValueError("Completed run identity changed")
            emit(root,"pipeline","complete",reason=previous["reason"],already_completed=True)
            return
        base=extra_report(baseline_records(c))
        write_json(root/"results"/"baseline.json",{"metrics":base})
        if not (root/"preflight.json").exists():
            emit(root,"pipeline","preflight")
            run_pool(args,[("preflight","preflight",[])])
        preflight_result=read(root/"preflight.json")
        if preflight_result.get("passed") is not True or preflight_result.get("identity_hash")!=c["identity_hash"]:
            raise RuntimeError("Required preflight did not pass")
        emit(root,"pipeline","feature_extraction")
        run_pool(args,[("cache",f"cache-{i}",["--shard",str(i)]) for i in (0,1)])
        publish(root,c,("train","val"))
        emit(root,"pipeline","three_fold_development")
        jobs=[("fit",f"cv-{method}-{fold}",["--method",method,"--fold",str(fold),"--seed","42","--target-epochs","40"])
              for fold in range(3) for method in METHODS]
        run_pool(args,jobs)
        choices=choose_epochs(root,c)
        emit(root,"pipeline","full_training_seed42",choices=choices)
        run_pool(args,[("fit",f"fit-{method}-42",["--method",method,"--fold","-1","--seed","42",
                       "--target-epochs",str(choices[method]["epoch"])]) for method in METHODS])
        candidates=[read(root/"results"/f"{m}-seed42.json") for m in METHODS]
        selected=max(candidates,key=lambda v:(round(v["metrics"]["score"],12),
            -v["metrics"]["counts"]["aligned_false_positive"]["correct"],v["method"]=="binary_control"))
        candidate=root/"results"/f"{selected['method']}-seed42.json"
        write_json(root/"selected.json",{"candidate":str(candidate),"method":selected["method"],"epochs":selected["epochs"]})
        if selected["metrics"]["score"]-base["score"] < .015-1e-12:
            finish(root,c,"primary_gate_not_met",candidates,selected)
            return
        emit(root,"pipeline","primary_output_validation")
        run_pool(args,[("questions","questions-42",["--candidate",str(candidate)])])
        selected=read(candidate)
        if not gate(base,selected["metrics"],.015)["passed"]:
            finish(root,c,"primary_gate_not_met",candidates,selected)
            return
        emit(root,"pipeline","replication_seed43",method=selected["method"],epochs=selected["epochs"])
        run_pool(args,[("fit","fit-replication-43",["--method",selected["method"],"--fold","-1","--seed","43",
                       "--target-epochs",str(selected["epochs"])])])
        secondary=root/"results"/f"{selected['method']}-seed43.json"
        if read(secondary)["metrics"]["score"]-base["score"] < .005-1e-12:
            finish(root,c,"replication_gate_not_met",candidates,selected)
            return
        run_pool(args,[("questions","questions-43",["--candidate",str(secondary)])])
        if not gate(base,read(secondary)["metrics"],.005)["passed"]:
            finish(root,c,"replication_gate_not_met",candidates,selected)
            return
        emit(root,"pipeline","qualified_test_features")
        run_pool(args,[("cache",f"test-cache-{i}",["--shard",str(i),"--split","test"]) for i in (0,1)])
        publish(root,c,("test",))
        emit(root,"pipeline","selected_submission")
        run_pool(args,[("export","export",[])])
        finish(root,c,"qualified_submission_ready",candidates,selected)


def finish(root,c,reason,candidates,selected):
    # Reload candidates so constraint checks performed later are represented.
    candidates=[read(Path(root)/"results"/f"{m}-seed42.json") for m in METHODS]
    secondary=Path(root)/"results"/f"{selected['method']}-seed43.json"
    value={"identity_hash":c["identity_hash"],"reason":reason,"baseline":extra_report(baseline_records(c)),
        "candidates":candidates,"selected":read(Path(root)/"selected.json"),
        "replication":read(secondary) if secondary.exists() else None,
        "history_best_local_score":.71441920,"confirmed_kaggle_public":.90813,
        "new_kaggle_score":None,"wall_clock_hours":(time.time()-c["started"])/3600,
        "note":"Development folds isolate the new head only. No automatic Kaggle submission."}
    write_json(Path(root)/"final-report.json",value)
    emit(root,"pipeline","complete",reason=reason,score=selected["metrics"]["score"])


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("run","prepare","preflight","cache","fit","questions","export"))
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--config",type=Path)
    p.add_argument("--data-root",type=Path)
    p.add_argument("--model-path",type=Path)
    p.add_argument("--sft-adapter-path",type=Path)
    p.add_argument("--baseline-csv",type=Path)
    p.add_argument("--source-run",type=Path)
    p.add_argument("--source-release",type=Path)
    p.add_argument("--gpu",default="auto")
    p.add_argument("--gpus",nargs=2,default=["1","2"])
    p.add_argument("--memory-gib",type=float,default=32.)
    p.add_argument("--cpu-threads",type=int,default=4)
    p.add_argument("--role",default="worker")
    p.add_argument("--shard",type=int,choices=(0,1),default=0)
    p.add_argument("--split",choices=("trainval","test"),default="trainval")
    p.add_argument("--method",choices=METHODS)
    p.add_argument("--fold",type=int,default=-1)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--target-epochs",type=int,default=40)
    p.add_argument("--candidate",type=Path)
    return p


def main():
    args=parser().parse_args()
    role="pipeline" if args.command in ("run","prepare") else "preflight" if args.command=="preflight" else args.role
    try:
        if args.memory_gib!=32 or len(set(args.gpus))!=2:
            raise ValueError("Use two distinct allocated GPUs and the approved 32 GiB ceiling")
        if args.command=="run":
            pipeline(args)
        elif args.command=="prepare":
            c=initialize(args)
            emit(args.run_dir,"pipeline","prepared",identity_hash=c["identity_hash"])
        else:
            c=context_for(args.run_dir)
            if args.command=="preflight": preflight(args,c)
            elif args.command=="cache": cache(args,c)
            elif args.command=="fit": fitting(args,c)
            elif args.command=="questions": question_validation(args,c)
            elif args.command=="export": export(args,c)
    except Exception as error:
        value={"error":str(error),"type":type(error).__name__,"traceback":traceback.format_exc(),"time":time.time()}
        write_json(args.run_dir/(role+".error.json"),value)
        emit(args.run_dir,role,"failed",error=str(error))
        traceback.print_exc()
        return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
