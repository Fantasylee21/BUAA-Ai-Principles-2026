"""Deadline-aware, strictly serial GPU-2 experiment and fixed candidate export."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from rl.emotion_expert import (EMOTIONS,SETTINGS,VERSION,admission,atomic,choose,digest,
    evaluate,export,fingerprint,gate,input_only,jsonl,read,targets,csv_predictions,differences)

STOP=False
SOURCE=Path(__file__).resolve().parents[1]


def stop(signum,frame):
    global STOP
    STOP=True


def cutoffs():
    zone=ZoneInfo("Asia/Shanghai")
    return {"delivery":datetime(2026,10,29,23,0,tzinfo=zone).timestamp(),
            "hard_stop":datetime(2026,10,29,23,30,tzinfo=zone).timestamp(),
            "competition":datetime(2026,10,30,0,0,tzinfo=zone).timestamp()}


def status(run,phase,**values):
    value={"version":VERSION,"phase":phase,"gpu_policy":"GPU 2 only; serial",
           "updated_unix":time.time(),"pid":os.getpid(),**values}
    atomic(run/"pipeline.status.json",value)
    print(json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)


def prepare(project,run,frames=64):
    recipe=read(SOURCE/"configs/rl/emotion_expert_final_v1.json")
    if (recipe["allowed_gpus"]!=[2] or recipe["auto_submit"] is not False or
            any(recipe[k]!=SETTINGS[k] for k in ("seed","rank","alpha","dropout","batch_size",
                "sft_learning_rate","rl_learning_rate","sft_weight_decay","rl_weight_decay",
                "warmup_fraction","max_grad_norm")) or recipe["reference_kl"]!=SETTINGS["beta"]):
        raise ValueError("Locked experiment recipe changed")
    old=project/"runs/protocol-sft-ablation-seed42-20261025"
    manifest=read(old/"split-manifest.json")
    original=jsonl(project/"Dataset/train.jsonl")
    by_id={r["sample_id"]:r for r in original}
    partitions={name:jsonl(old/"raw"/(name+".jsonl")) for name in ("train","dev","holdout")}
    if [len(v) for v in partitions.values()]!=[2013,670,671] or len(by_id)!=3354:raise ValueError("Frozen partition sizes changed")
    seen=set();media={};owners=defaultdict(set);group_owners=defaultdict(set)
    for name,rows in partitions.items():
        for row in rows:
            sid=row["sample_id"]
            if sid in seen or row!=by_id.get(sid):raise ValueError("Original row/split identity changed")
            seen.add(sid);targets(row);safe=input_only(row)
            path=(project/"Dataset"/safe["video"]).resolve()
            if not path.is_relative_to((project/"Dataset").resolve()):raise ValueError("Media escapes data root")
            if safe["video"] not in media:media[safe["video"]]=digest(path)
            owners[media[safe["video"]]].add(name)
            group_owners[row["split_group_id"]].add(name)
    if seen!=set(by_id) or any(len(v)>1 for v in (*owners.values(),*group_owners.values())):raise ValueError("Content group leakage")
    if sum(r["sample_type"]=="Conflict" for r in original)!=411:raise ValueError("Original labels changed")
    settings={**SETTINGS,"video_max_frames":frames}
    model=run/"merged-emotion-base"
    splits={name:str(old/"raw"/(name+".jsonl")) for name in partitions}
    identity={"version":VERSION,"settings":settings,"split_sha256":{n:digest(p) for n,p in splits.items()},
        "original_train_sha256":digest(project/"Dataset/train.jsonl"),"old_split_sha256":digest(old/"split-manifest.json"),
        "model_merge_sha256":digest(model/"merge.complete.json"),"media_sha256":media,
        "sources":{name:digest(SOURCE/name) for name in SOURCE_FILES},"gpu_policy":[2]}
    context={"identity":identity,"settings":settings,"model":str(model),"data_root":str(project/"Dataset"),
        "run":str(run),"splits":splits,"media_sha256":media,"old_split":str(old/"split-manifest.json"),
        "reference_csv":{n:str(old/"experiment/evaluation"/("checkpoint-426_"+n+"_predictions.csv")) for n in ("dev","holdout")}}
    path=run/f"context-{frames}.json"
    if path.exists() and read(path)!=context:raise ValueError("Prepared context is immutable")
    atomic(path,context)
    atomic(run/"data-audit.json",{"partition_counts":{n:len(v) for n,v in partitions.items()},
        "content_isolation":True,"original_rows_exact":True,"media_files":len(media),
        "use_in_main_false_retained":sum(r.get("use_in_main") is False for r in original),
        "official_val_read":False,"official_test_read":False,"identity":fingerprint(identity)})
    return path,context


SOURCE_FILES=("rl/emotion_expert.py","rl/emotion_expert_runtime.py","rl/omni_runtime.py",
    "scripts/emotion_expert_bootstrap.py","scripts/train_emotion_expert.py",
    "scripts/emotion_expert_pipeline.py","scripts/emotion_expert_progress.py","configs/rl/emotion_expert_final_v1.json")


class Runner:
    def __init__(self,project,run):
        self.project=project;self.run=run;self.cut=cutoffs()
        self.child=None

    def check(self):
        if STOP or time.time()>=self.cut["hard_stop"]:raise InterruptedError("Batch hard stop")

    def gpu_idle(self):
        text=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.free","--format=csv,noheader,nounits"],text=True)
        uuid=None;free=0
        for line in text.splitlines():
            i,u,m=[v.strip() for v in line.split(",")]
            if i=="2":uuid=u;free=int(m)
        apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid","--format=csv,noheader"],text=True)
        return uuid is not None and uuid not in apps.splitlines() and free>=41*1024

    def job(self,context_path,name,mode,branch,*,variant="S",adapter=None,split="dev",evidence=False,deadline=None):
        self.check()
        out=self.run/"jobs"/name
        if (out/"complete.json").exists():return read(out/"complete.json")
        end=min(deadline or self.cut["hard_stop"],self.cut["hard_stop"])
        if time.time()>=end:raise InterruptedError("Phase deadline already expired")
        while not self.gpu_idle():
            self.check()
            if time.time()>=end:raise InterruptedError("GPU 2 unavailable before phase deadline")
            status(self.run,"waiting_gpu2",job=name,waiting_for="GPU 2 only; other GPUs will not be used")
            time.sleep(5)
        out.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,"-u","-m","scripts.train_emotion_expert","--context",str(context_path),
            "--output",str(out),"--mode",mode,"--branch",branch,"--variant",variant,
            "--split",split,"--deadline",str(end),"--gpu","2"]
        if adapter:cmd += ["--adapter",str(adapter)]
        if evidence:cmd.append("--evidence")
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="2",TOKENIZERS_PARALLELISM="false")
        log=self.run/"logs"/(name.replace("/","-")+".log")
        status(self.run,"running",job=name,job_status=str(out/"status.json"),command=cmd)
        with log.open("a",encoding="utf-8") as f:
            self.child=subprocess.Popen(cmd,cwd=SOURCE,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            atomic(out/"process.json",{"pid":self.child.pid,"command":cmd,"started_unix":time.time(),"gpu":2})
            while self.child.poll() is None:
                if STOP or time.time()>=end:
                    self.child.send_signal(signal.SIGTERM)
                    try:self.child.wait(timeout=50)
                    except subprocess.TimeoutExpired:
                        # This PID was created above and has not exited/reaped.
                        self.child.kill();self.child.wait()
                    raise InterruptedError("Owned job stopped at deadline")
                time.sleep(2)
            result=self.child.returncode;self.child=None
        if result or not (out/"complete.json").exists():
            details=read(out/"status.json") if (out/"status.json").exists() else {}
            raise RuntimeError("Worker failed: "+name+" "+str(details))
        return read(out/"complete.json")


def prediction(run,name):
    return {r["sample_id"]:r for r in jsonl(run/"jobs"/name/"predictions.jsonl")}


def estimates(e,u):
    # Training is serial; include both native/reference forwards for each RL
    # arm, full selected held evidence, official val and the 734-row test.
    es=e["p95_row_seconds"];us=u["p95_row_seconds"]
    ei=e["inference_mean_seconds"]["E"]
    ui=u["inference_mean_seconds"]["U"]
    vi=u["inference_mean_seconds"]["V"];mi=u["inference_mean_seconds"]["M"]
    init=100.
    delivery=(671+666)*(ei+ui+vi+mi)+734*(ei+ui)+5*init
    native_dev=670*(ei+ui)+2*init
    frozen=native_dev+delivery
    supervised=frozen+2013*(es+us)+670*(ei+ui)+4*init
    full=supervised+2*2013*(us+ui)+2*670*ui+671*(ui+vi+mi)+6*init
    return {"frozen":frozen,"supervised":supervised,"full":full,
            "basis":"serial GPU2, real preflight; admission adds 30% margin and 15 min export reserve"}


def bootstrap_interval(rows,reference,candidate,media,repeats=1000):
    import random
    grouped=defaultdict(list)
    for row in rows:grouped[media[row["video"]]].append(row)
    units=list(grouped.values());rng=random.Random(42);values=[]
    for _ in range(repeats):
        sample=[r for _ in units for r in rng.choice(units)]
        if not any(r["sample_type"]=="Conflict" for r in sample) or not any(r["sample_type"]=="Aligned" for r in sample):continue
        ids={r["sample_id"] for r in sample}
        subset=lambda pair:({sid:pair[0][sid] for sid in ids},{sid:pair[1][sid] for sid in ids})
        a=evaluate(sample,*subset(reference));b=evaluate(sample,*subset(candidate));values.append(b["selection_score"]-a["selection_score"])
    values.sort()
    return {"unit":"media_sha256","repeats":len(values),"seed":42,
            "percentile_95":[values[int(.025*(len(values)-1))],values[int(.975*(len(values)-1))]],
            "selection_adjusted":False,"independent_generalization_claim":False}


def pipeline(project,run):
    runner=Runner(project,run)
    # Hold a process-lifetime lock so two terminals cannot launch this batch.
    import fcntl
    run.mkdir(parents=True,exist_ok=True)
    lock=(run/".pipeline.lock").open("a+");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=read(run/"bootstrap.status.json")["started_unix"]
    budget_path=run/"startup-budget.json"
    startup_seconds=read(budget_path)["approved_seconds"] if budget_path.exists() else 3600
    if startup_seconds not in (3600,5400):raise ValueError("Unsupported initialization budget")
    startup_end=start+startup_seconds
    while not (run/"merged-emotion-base/merge.complete.json").exists():
        runner.check()
        b=read(run/"bootstrap.status.json")
        if b["phase"]=="failed":
            retry=run/"bootstrap.retry.json"
            if retry.exists() or time.time()>=startup_end:
                raise RuntimeError("Bootstrap failed: "+b.get("error","unknown"))
            # One engineering retry uses the same fixed checkpoint/download
            # directory and original startup deadline. Never reset its budget.
            atomic(run/"bootstrap-attempt1.status.json",b)
            with (run/"logs/bootstrap-retry.log").open("a") as log:
                child=subprocess.Popen([sys.executable,"-u",str(SOURCE/"scripts/emotion_expert_bootstrap.py"),
                    "--project",str(project),"--root",str(run)],cwd=SOURCE,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=""),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic(run/"bootstrap.status.json",{"phase":"retry_starting","pid":child.pid,
                "started_unix":start,"updated_unix":time.time()})
            atomic(retry,{"pid":child.pid,"original_started_unix":start,"retry_unix":time.time(),
                          "same_fixed_checkpoint":True,"source_sha256":digest(SOURCE/"scripts/emotion_expert_bootstrap.py")})
            time.sleep(3)
            continue
        if time.time()>=startup_end:
            pid=b.get("pid")
            if isinstance(pid,int) and pid>1:
                proc=Path("/proc")/str(pid)
                try:
                    cmd=(proc/"cmdline").read_bytes().replace(b"\0",b" ").decode()
                    if proc.stat().st_uid==os.getuid() and "emotion_expert_bootstrap.py" in cmd and str(run) in cmd:
                        os.kill(pid,signal.SIGTERM)
                except (OSError,ProcessLookupError):pass
            raise InterruptedError(f"{startup_seconds//60} minute download/initialization budget exceeded")
        status(run,"bootstrap",bootstrap=b);time.sleep(5)
    reports={}
    for frames in (64,32):
        status(run,"preparing_data",cohort_frames=frames)
        context_path,c=prepare(project,run,frames)
        try:
            for branch in ("E","U"):
                reports[branch]=runner.job(context_path,f"preflight-{frames}-{branch}","preflight",branch,deadline=startup_end)
            break
        except RuntimeError:
            out=run/"jobs"/f"preflight-{frames}-{branch}"/"status.json"
            if frames==64 and out.exists() and read(out).get("out_of_memory"):
                status(run,"uniform_media_fallback",from_frames=64,to_frames=32)
                continue
            raise
    durations=estimates(reports["E"],reports["U"])
    mode=admission(time.time(),runner.cut["delivery"],durations)
    atomic(run/"admission.json",{"mode":mode,"estimates_seconds":durations,"cutoffs":runner.cut,
        "gpu_policy":[2],"preflight":reports,"decided_unix":time.time()})
    if mode=="none":
        atomic(run/"final-report.json",{"status":"complete","reason":"no_complete_cohort_fits_deadline","candidate":None,"admission":durations})
        status(run,"complete",reason="no_complete_cohort_fits_deadline");return
    adapters={"E0":None,"U0":None};available_e=["E0"];available_u=["U0"]
    if mode in ("supervised","full"):
        for branch in ("E","U"):
            report=runner.job(context_path,"S-"+branch,"train",branch)
            adapters["SE" if branch=="E" else "S"]=report["adapter"]
        available_e.append("SE");available_u.append("S")
    if mode=="full":
        for variant in ("R1","R2"):
            report=runner.job(context_path,variant,"train","U",variant=variant,adapter=adapters["S"])
            adapters[variant]=report["adapter"]
        # Add both only after both one-epoch checkpoints passed completion.
        available_u.extend(("R1","R2"))
    for name in available_e:
        runner.job(context_path,"dev-"+name,"predict","E",adapter=adapters[name])
    for name in available_u:
        runner.job(context_path,"dev-"+name,"predict","U",adapter=adapters[name],variant=name if name in ("R1","R2") else "S")
    dev_rows=jsonl(c["splits"]["dev"])
    e0=prediction(run,"dev-E0");u0=prediction(run,"dev-U0")
    dev={name:evaluate(dev_rows,prediction(run,"dev-"+name),u0) for name in available_e}
    dev.update({name:evaluate(dev_rows,e0,prediction(run,"dev-"+name)) for name in available_u})
    chosen=choose(dev,available_e,available_u)
    atomic(run/"dev-results.json",dev);atomic(run/"selection.json",{**chosen,"adapters":adapters,"development_only":True})
    en=chosen["emotion"];un=chosen["uncertainty"];base=chosen["non_rl"]
    runner.job(context_path,"held-"+en,"predict","E",adapter=adapters[en],split="holdout")
    runner.job(context_path,"held-"+un,"predict","U",adapter=adapters[un],split="holdout",evidence=True,variant=un if un in ("R1","R2") else "S")
    held=jsonl(c["splits"]["holdout"]);ep=prediction(run,"held-"+en);up=prediction(run,"held-"+un)
    candidate=evaluate(held,ep,up);rl_check=None
    if un in ("R1","R2"):
        runner.job(context_path,"held-"+base,"predict","U",adapter=adapters[base],split="holdout",evidence=True)
        bup=prediction(run,"held-"+base);base_metrics=evaluate(held,ep,bup)
        rl_check={"gain":candidate["uncertainty_component"]-base_metrics["uncertainty_component"],"reference":base_metrics}
        if (rl_check["gain"]<.003-1e-12 or candidate["groups"].get("natural_VA",{}).get("correct",0)<base_metrics["groups"].get("natural_VA",{}).get("correct",0)):
            un=base;up=bup;candidate=base_metrics
    ref=csv_predictions(c["reference_csv"]["holdout"]);baseline=evaluate(held,*ref)
    decision=gate(candidate,baseline)
    report={"status":"complete","mode":mode,"selected_emotion":en,"selected_uncertainty":un,
        "held":candidate,"reference":baseline,"gate":decision,"rl_check":rl_check,
        "changes_from_reference":differences(held,ref,(ep,up)),
        "bootstrap":bootstrap_interval(held,ref,(ep,up),c["media_sha256"]),
        "candidate":None,"public_score":None,"single_seed_development_evidence":True,"auto_submit":False}
    atomic(run/"held-results.json",report)
    if not decision["passed"]:
        atomic(run/"final-report.json",report);status(run,"complete",reason="held_gate_not_met",report=report);return
    # Only the locked, accepted model may open official val/test inputs.
    delivery={**c,"splits":{**c["splits"],"val":str(project/"Dataset/val.jsonl"),"test":str(project/"Dataset/test_release.jsonl")}}
    atomic(run/"delivery-context.json",delivery);delivery_path=run/"delivery-context.json"
    for split in ("val","test"):
        for branch,name in (("E",en),("U",un)):
            runner.job(delivery_path,split+"-"+branch,"predict",branch,adapter=adapters[name],split=split,evidence=split=="val" and branch=="U",variant=name if name in ("R1","R2") else "S")
    val=jsonl(delivery["splits"]["val"])
    report["official_validation_diagnostic"]=evaluate(val,prediction(run,"val-E"),prediction(run,"val-U"))
    test=jsonl(delivery["splits"]["test"])
    baseline_path=run/"inputs/submission_09.csv"
    with baseline_path.open(encoding="utf-8-sig",newline="") as f:old={r["sample_id"]:r for r in csv.DictReader(f)}
    report["export"]=export(test,prediction(run,"test-E"),prediction(run,"test-U"),run/"submission.csv",old)
    atomic(run/"changes.json",report["export"].pop("changes"))
    report["candidate"]=str(run/"submission.csv")
    atomic(run/"final-report.json",report);status(run,"complete",reason="fixed_candidate_ready",candidate=report["candidate"])


def main():
    p=argparse.ArgumentParser();p.add_argument("--project",type=Path,required=True);p.add_argument("--run",type=Path,required=True)
    a=p.parse_args();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:pipeline(a.project,a.run)
    except BaseException as e:
        status(a.run,"stopped" if isinstance(e,InterruptedError) else "failed",error=repr(e))
        atomic(a.run/"failure.json",{"error":repr(e),"at_unix":time.time(),"candidate":None})
        raise


if __name__=="__main__":main()
