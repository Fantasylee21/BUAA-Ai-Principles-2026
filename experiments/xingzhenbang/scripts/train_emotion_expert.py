"""One serial GPU-2 worker: real preflight, one epoch, or fixed inference."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import time

from rl.emotion_expert import (EMOTIONS,VERSION,atomic,batches,class_weight,counts,digest,
    fingerprint,jsonl,objective,preflight_rows,read,targets,weights)

STOP=False


def stopped(signum,frame):
    global STOP
    STOP=True


def check(args):
    if STOP or time.time()>=args.deadline:raise InterruptedError("Owned worker stop/deadline requested")


def emit(args,phase,**values):
    value={"phase":phase,"branch":args.branch,"variant":args.variant,"gpu":2,
           "updated_unix":time.time(),"pid":os.getpid(),**values}
    atomic(args.output/"status.json",value)
    print(json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)


def setup(args,context,training,adapter=None,reference=None):
    if args.gpu!=2 or context["settings"]["allowed_gpus"]!=[2]:raise ValueError("Only physical GPU 2 is authorized")
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
    os.environ["HF_HUB_OFFLINE"]="1"
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    import random
    import numpy as np
    import torch
    if torch.cuda.device_count()!=1:raise RuntimeError("Exactly one CUDA device must be visible")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(min(1.,40*1024**3/torch.cuda.get_device_properties(0).total_memory))
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    from rl.emotion_expert_runtime import ExpertPolicy
    return ExpertPolicy(context,adapter,training=training,reference=reference)


def optimizer(policy,phase,steps):
    torch=policy.torch
    lr=1e-5 if phase=="S" else 5e-6
    opt=torch.optim.AdamW(policy.parameters(),lr=lr,weight_decay=.01 if phase=="S" else 0.,foreach=False)
    warmup=max(1,math.ceil(steps*.1))
    def scale(step):
        if step<warmup:return (step+1)/warmup
        return max(0.,(steps-step)/max(1,steps-warmup))
    sch=torch.optim.lr_scheduler.LambdaLR(opt,scale)
    return opt,sch


def save(args,policy,opt,sch,step,identity):
    import random
    import numpy as np
    checkpoint=args.output/f"checkpoint-{step:06d}"
    marker=checkpoint/"complete.json"
    if marker.exists():return checkpoint
    checkpoint.mkdir(parents=True,exist_ok=True)
    policy.save_adapter(checkpoint/"adapter")
    policy.torch.save({"optimizer":opt.state_dict(),"scheduler":sch.state_dict(),"step":step,
        "torch_rng":policy.torch.get_rng_state(),"cuda_rng":policy.torch.cuda.get_rng_state(),
        "python_rng":random.getstate(),"numpy_rng":np.random.get_state()},checkpoint/"state.pt")
    files={str(p.relative_to(checkpoint)):digest(p) for p in checkpoint.rglob("*") if p.is_file()}
    atomic(marker,{"identity":identity,"step":step,"files":files,"adapter":str(checkpoint/"adapter/policy")})
    atomic(args.output/"latest.json",{"checkpoint":str(checkpoint),"step":step,"identity":identity})
    return checkpoint


def restore(policy,opt,sch,path,identity):
    import random
    import numpy as np
    path=Path(path);m=read(path/"complete.json")
    if m["identity"]!=identity or any(digest(path/k)!=v for k,v in m["files"].items()):raise ValueError("Checkpoint identity/hash changed")
    state=policy.torch.load(path/"state.pt",map_location="cpu",weights_only=False)
    opt.load_state_dict(state["optimizer"]);sch.load_state_dict(state["scheduler"])
    for value in opt.state.values():
        for k,v in value.items():
            if policy.torch.is_tensor(v):value[k]=v.to("cuda:0") if k!="step" else v
    policy.torch.set_rng_state(state["torch_rng"]);policy.torch.cuda.set_rng_state(state["cuda_rng"])
    random.setstate(state["python_rng"]);np.random.set_state(state["numpy_rng"])
    return state["step"]


def one_row(policy,row,population,visual_weight,branch,phase,divisor,reference=None):
    import torch.nn.functional as F
    t=targets(row);torch=policy.torch;metrics={};timing={}
    if branch=="E":
        start=time.perf_counter();scores=policy.scores(row,"E")
        loss=F.cross_entropy(scores[None],scores.new_tensor([t["E"]],dtype=torch.long))
        (loss/divisor).backward()
        metrics.update(loss=float(loss.detach()),emotion_correct=int(int(scores.argmax())==t["E"]))
        timing["E"]=time.perf_counter()-start
    else:
        start=time.perf_counter()
        ref=reference
        if phase!="S" and ref is None:
            with torch.no_grad():ref=policy.scores(row,"U",adapter="reference").double().softmax(-1).tolist()
        scores=policy.scores(row,"U")
        loss,parts=objective(scores,t["U"],population,phase=phase,reference=ref)
        (loss/divisor).backward()
        metrics.update(parts,main_loss=float(loss.detach()),reference=ref)
        timing["U"]=time.perf_counter()-start
        total=float(loss.detach())
        for task,factor in (("V",.05*visual_weight),("M",.05)):
            start=time.perf_counter();scores=policy.scores(row,task)
            ce=F.cross_entropy(scores[None],scores.new_tensor([t[task]],dtype=torch.long))
            auxiliary=factor*ce;(auxiliary/divisor).backward();total+=float(auxiliary.detach())
            metrics[task+"_loss"]=float(ce.detach());metrics[task+"_correct"]=int(int(scores.argmax())==t[task])
            timing[task]=time.perf_counter()-start
        metrics["loss"]=total
    return metrics,timing


def train(args,c,rows,*,preflight=False):
    import numpy as np
    args.output.mkdir(parents=True,exist_ok=True)
    population=counts(c["train_rows"]);vw=weights(c["train_rows"])
    batch_list=batches(rows);steps=len(batch_list)
    identity={"version":VERSION,"context":fingerprint(c["identity"]),"branch":args.branch,
              "variant":args.variant,"training_ids":fingerprint([r["sample_id"] for r in rows]),
              "initial_adapter":digest(Path(args.adapter)/"adapter_model.safetensors") if args.adapter else None}
    latest=read(args.output/"latest.json") if (args.output/"latest.json").exists() and not preflight else None
    resume=Path(latest["checkpoint"]) if latest else None
    adapter=resume/"adapter/policy" if resume else args.adapter
    reference=args.adapter if args.variant!="S" else None
    emit(args,"loading",current_steps=0,total_steps=steps)
    policy=setup(args,c,True,adapter,reference)
    atomic(args.output/"trainable-parameters.json",policy.inventory())
    atomic(args.output/"quantized-bias-normalization.json",policy.bias_normalization)
    if not policy.inventory():raise RuntimeError("No actual trainable task LoRA")
    cfg=policy.model.peft_config["policy"]
    if cfg.r!=16 or cfg.lora_alpha!=32:raise RuntimeError("Task LoRA settings changed")
    opt,sch=optimizer(policy,args.variant,steps+1 if preflight else steps)
    start_step=restore(policy,opt,sch,resume,identity) if resume else 0
    initial_digest=policy.parameter_digest()
    frozen_digest=policy.parameter_digest("base") if preflight else None
    reference_digest=policy.parameter_digest("reference") if reference else None
    row_times=[];task_times={};observed=[];norms=[];committed=start_step*16
    totals={"seen":0,"emotion_correct":0,"natural_n":0,"natural_tp":0,"natural_va_n":0,
            "natural_va_tp":0,"aligned_n":0,"false_positives":0,"V_correct":0,"M_correct":0}
    if start_step and (args.output/"training.jsonl").exists():
        history=jsonl(args.output/"training.jsonl")
        if history:totals.update(history[-1].get("observation_counts",{}))
    previous_checkpoint=resume;continued_digest=None;continuous_optimizer=None
    try:
        for index,batch in enumerate(batch_list[start_step:],start_step):
            check(args);opt.zero_grad(set_to_none=True);items=[]
            for row in batch:
                check(args);started=time.perf_counter()
                metrics,timing=one_row(policy,row,population,vw[row["sample_id"]],args.branch,args.variant,len(batch))
                row_times.append(time.perf_counter()-started)
                for k,v in timing.items():task_times.setdefault(k,[]).append(v)
                items.append(metrics)
            norm=policy.torch.nn.utils.clip_grad_norm_(policy.parameters(),1.)
            if not math.isfinite(float(norm)) or float(norm)<=0:raise RuntimeError("Nonfinite/zero task gradient")
            norms.append(float(norm));opt.step();sch.step();committed+=len(batch)
            for row,item in zip(batch,items):
                totals["seen"]+=1
                if args.branch=="E":totals["emotion_correct"]+=item["emotion_correct"]
                else:
                    yes=item["prediction"]==1;gold=targets(row)["U"]
                    natural=bool(gold and row.get("source_dataset") in ("ch_sims","ch_sims_v2"))
                    totals["natural_n"]+=int(natural);totals["natural_tp"]+=int(natural and yes)
                    totals["natural_va_n"]+=int(natural and row["protocol"]=="VA")
                    totals["natural_va_tp"]+=int(natural and row["protocol"]=="VA" and yes)
                    totals["aligned_n"]+=int(not gold);totals["false_positives"]+=int(not gold and yes)
                    totals["V_correct"]+=item["V_correct"];totals["M_correct"]+=item["M_correct"]
            ratio=lambda x,n:totals[x]/totals[n] if totals[n] else None
            observation={"natural_recall":ratio("natural_tp","natural_n"),
                "natural_va_recall":ratio("natural_va_tp","natural_va_n"),
                "ordinary_false_positive_rate":ratio("false_positives","aligned_n"),
                "vision_accuracy":ratio("V_correct","seen") if args.branch=="U" else None,
                "second_accuracy":ratio("M_correct","seen") if args.branch=="U" else None,
                "emotion_accuracy":ratio("emotion_correct","seen") if args.branch=="E" else None}
            reduced={k:sum(float(x[k]) for x in items)/len(items) for k in items[0] if k not in ("reference",)}
            record={"step":index+1,"committed_rows":committed,"batch_rows":len(batch),
                    "sample_ids":[r["sample_id"] for r in batch],"metrics":reduced,
                    "grad_norm":float(norm),"learning_rate":opt.param_groups[0]["lr"],
                    "mean_row_seconds":sum(row_times)/len(row_times),"observation_counts":dict(totals)}
            observed.append(record)
            with (args.output/"training.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(record,ensure_ascii=False)+"\n")
            if preflight or (index+1)%10==0 or index+1==steps:
                previous_checkpoint=save(args,policy,opt,sch,index+1,identity)
            emit(args,"training",current_steps=index+1,total_steps=steps,epoch=1,
                 progress=(index+1)/steps,committed_rows=committed,metrics=reduced,
                 grad_norm=float(norm),allocated_gib=policy.torch.cuda.memory_allocated()/1024**3,
                 peak_gib=policy.torch.cuda.max_memory_allocated()/1024**3,
                 eta_seconds=sum(row_times)/len(row_times)*(len(rows)-committed),
                 mean_row_seconds=sum(row_times)/len(row_times),observations=observation)
        final_digest=policy.parameter_digest()
        if initial_digest==final_digest and start_step<steps:raise RuntimeError("Task LoRA did not actually change")
        if reference and policy.parameter_digest("reference")!=reference_digest:raise RuntimeError("Frozen S reference changed")
        peak=policy.torch.cuda.max_memory_allocated()/1024**3
        if peak>40:raise RuntimeError("40 GiB memory budget exceeded")
        if not row_times:
            history=jsonl(args.output/"training.jsonl")
            row_times=[r["mean_row_seconds"] for r in history];norms=[r["grad_norm"] for r in history]
        report={"complete":True,"identity":identity,"rows":len(rows),"updates":steps,
            "initial_digest":initial_digest,"final_digest":final_digest,"peak_gib":peak,
            "mean_row_seconds":float(np.mean(row_times)),"p95_row_seconds":float(np.quantile(row_times,.95)),
            "task_mean_seconds":{k:float(np.mean(v)) for k,v in task_times.items()},
            "min_grad_norm":min(norms),"all_updates_nonzero":True,
            "adapter":str(previous_checkpoint/"adapter/policy")}
        if preflight:
            if policy.parameter_digest("base")!=frozen_digest:raise RuntimeError("Frozen expert base changed")
            # Fresh NF4 reload, optimizer and RNG restoration must reproduce
            # the final update, rather than merely restore a counter.
            recovery_checkpoint=previous_checkpoint
            import hashlib
            def opt_hash(value):
                h=hashlib.sha256()
                for key,state in value.state.items():
                    for k,v in sorted(state.items()):
                        if policy.torch.is_tensor(v):h.update(v.detach().cpu().contiguous().reshape(-1).view(policy.torch.uint8).numpy().tobytes())
                return h.hexdigest()
            # The 32 coverage samples use effective batch16. One additional
            # TRAIN-row update tests fresh-reload replay without redundantly
            # rerunning a complete 16-row batch on a deadline-limited GPU.
            opt.zero_grad(set_to_none=True)
            one_row(policy,rows[0],population,vw[rows[0]["sample_id"]],args.branch,"S",1)
            policy.torch.nn.utils.clip_grad_norm_(policy.parameters(),1.);opt.step();sch.step()
            continuous_optimizer=opt_hash(opt);continued_digest=policy.parameter_digest()
            if continued_digest==final_digest:raise RuntimeError("Resume probe made no parameter update")
            del opt,sch,policy
            import gc;gc.collect()
            import torch;torch.cuda.empty_cache()
            policy=setup(args,c,True,recovery_checkpoint/"adapter/policy")
            opt,sch=optimizer(policy,"S",steps+1);restore(policy,opt,sch,recovery_checkpoint,identity)
            opt.zero_grad(set_to_none=True)
            one_row(policy,rows[0],population,vw[rows[0]["sample_id"]],args.branch,"S",1)
            policy.torch.nn.utils.clip_grad_norm_(policy.parameters(),1.);opt.step();sch.step()
            if policy.parameter_digest()!=continued_digest or opt_hash(opt)!=continuous_optimizer:
                raise RuntimeError("Fresh-reload adapter/Adam resume is not bitwise reproducible")
            report.update(frozen_base_unchanged=True,fresh_reload_resume_exact=True,resume_probe_changed_parameters=True)
            infer_times=[]
            policy.training=False
            with policy.torch.no_grad():
                for row in rows[:4]:
                    for task in (("E",) if args.branch=="E" else ("U","V","M")):
                        t0=time.perf_counter();policy.scores(row,task);infer_times.append((task,time.perf_counter()-t0))
            report["inference_mean_seconds"]={task:float(np.mean([v for k,v in infer_times if k==task])) for task in {k for k,v in infer_times}}
        atomic(args.output/"complete.json",report)
        emit(args,"complete",current_steps=steps,total_steps=steps,report=report)
    except InterruptedError:
        opt.zero_grad(set_to_none=True)
        save(args,policy,opt,sch,min(len(batch_list),committed//16),identity)
        emit(args,"stopped",committed_rows=committed,reason="deadline_or_signal")
        raise


def predict(args,c,rows):
    args.output.mkdir(parents=True,exist_ok=True)
    emit(args,"loading",current_steps=0,total_steps=len(rows))
    policy=setup(args,c,False,args.adapter)
    identity={"context":fingerprint(c["identity"]),"split":args.split,"branch":args.branch,
              "adapter_sha256":digest(Path(args.adapter)/"adapter_model.safetensors") if args.adapter else "frozen_expert",
              "evidence":args.evidence,"rows":fingerprint([r["sample_id"] for r in rows]),
              "split_sha256":digest(c["splits"][args.split])}
    contract=args.output/"identity.json"
    if contract.exists() and read(contract)!=identity:raise ValueError("Inference identity changed")
    atomic(contract,identity)
    path=args.output/"predictions.jsonl"
    previous={r["sample_id"]:r for r in jsonl(path)} if path.exists() else {}
    start=time.time()
    with path.open("a",encoding="utf-8") as f,policy.torch.no_grad():
        for index,row in enumerate(rows):
            check(args);sid=row["sample_id"]
            if sid not in previous:
                if args.branch=="E":
                    probs=policy.scores(row,"E").double().softmax(-1).tolist()
                    value={"sample_id":sid,"emotion_probs":probs,"pred_emotion":EMOTIONS[max(range(3),key=lambda i:probs[i])]}
                else:
                    probs=policy.scores(row,"U").double().softmax(-1).tolist()
                    value={"sample_id":sid,"uncertain_prob":probs[1],"action_probs":probs}
                    if args.evidence:
                        for task,name in (("V","vision_probs"),("M","second_probs")):
                            value[name]=policy.scores(row,task).double().softmax(-1).tolist()
                f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+"\n");f.flush()
                previous[sid]=value
            emit(args,"predicting",current_steps=index+1,total_steps=len(rows),split=args.split,
                 progress=(index+1)/len(rows),eta_seconds=(time.time()-start)/(index+1)*(len(rows)-index-1),
                 allocated_gib=policy.torch.cuda.memory_allocated()/1024**3)
    if set(previous)!={r["sample_id"] for r in rows}:raise ValueError("Prediction coverage mismatch")
    atomic(args.output/"complete.json",{"identity":identity,"rows":len(rows),"sha256":digest(path)})
    emit(args,"complete",current_steps=len(rows),total_steps=len(rows),split=args.split)


def main():
    p=argparse.ArgumentParser();p.add_argument("--context",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mode",choices=("train","preflight","predict"),required=True)
    p.add_argument("--branch",choices=("E","U"),required=True);p.add_argument("--variant",choices=("S","R1","R2"),default="S")
    p.add_argument("--adapter",type=Path);p.add_argument("--split",choices=("dev","holdout","val","test"),default="dev")
    p.add_argument("--gpu",type=int,choices=(2,),default=2);p.add_argument("--evidence",action="store_true")
    p.add_argument("--deadline",type=float,required=True)
    args=p.parse_args();signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    c=read(args.context)
    if c["identity"]["settings"]!=c["settings"] or c["settings"]["allowed_gpus"]!=[2]:raise ValueError("Frozen settings changed")
    for name in ("train","dev","holdout"):
        if digest(c["splits"][name])!=c["identity"]["split_sha256"][name]:raise ValueError("Frozen split changed")
    if digest(Path(c["model"])/"merge.complete.json")!=c["identity"]["model_merge_sha256"]:raise ValueError("Frozen emotion base changed")
    c["train_rows"]=jsonl(c["splits"]["train"])
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        check(args)
        if args.mode=="predict":predict(args,c,jsonl(c["splits"][args.split]))
        else:
            rows=c["train_rows"]
            if args.mode=="preflight":
                rows=preflight_rows(rows,c["data_root"])
                atomic(args.output/"sample-coverage.json",[{"sample_id":r["sample_id"],"protocol":r["protocol"],"targets":targets(r)} for r in rows])
            train(args,c,rows,preflight=args.mode=="preflight")
    except BaseException as e:
        emit(args,"failed" if not isinstance(e,InterruptedError) else "stopped",error=repr(e),
             out_of_memory="out of memory" in str(e).lower())
        raise


if __name__=="__main__":main()
