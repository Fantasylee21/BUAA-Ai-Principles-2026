"""Cache, audit, train and evaluate a single frozen-fold conflict RL job."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import signal
import time

from rl.conflict_reward import SETTINGS, VERSION, anchored_logit, batches, counts_for, objective, predicted
from rl.experiment import artifact_hash, fingerprint
from rl.reward import parse_completion, question_is_consistent
from scripts.joint_affect_pipeline import read, atomic, immutable, digest

STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def check_deadline(args):
    if STOP or time.time() >= args.deadline_unix:
        raise InterruptedError("Stop/deadline requested; preserve the last complete update")


def emit(args, phase, **values):
    value = {"algorithm": VERSION, "phase": phase, "arm": args.arm, "seed": args.seed,
             "updated_unix": time.time(), "pid": os.getpid(), **values}
    atomic(args.output / "status.json", value)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def context(args):
    c = read(args.context)
    if digest(c["manifest"]) != c["manifest_sha256"]:
        raise ValueError("Frozen original fold manifest changed")
    from scripts.joint_affect_data import validate_manifest
    manifest = read(c["manifest"])
    validate_manifest(manifest)
    if digest(manifest["train_jsonl"]) != manifest["identity"]["train_sha256"]:
        raise ValueError("Original TRAIN source changed")
    rows = manifest["train"]
    if c["fold"] == "full":
        train, held = list(range(len(rows))), []
    else:
        part = manifest["folds"][c["grouping"]]["folds"][int(c["fold"])]
        train, held = part["train"], part["held"]
    c.update(rows=rows, train=[rows[i] for i in train], held=[rows[i] for i in held], manifest_value=manifest,
             _deadline_unix=args.deadline_unix)
    args.output.mkdir(parents=True, exist_ok=True)
    return c


def setup(args, c, training=False, checkpoint=None):
    if args.gpu not in (1, 2):
        raise ValueError("Only physical GPU 1 or 2 is authorized")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    import random
    import numpy as np
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one requested CUDA device must be visible")
    torch.cuda.set_per_process_memory_fraction(min(1., 40 * 1024**3 / torch.cuda.get_device_properties(0).total_memory))
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32 = False
    # The real GPU preflight found bitwise resume drift without this. Keep
    # identical deterministic kernels for every arm, seed and inference job.
    torch.use_deterministic_algorithms(SETTINGS["deterministic_algorithms"])
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if artifact_hash(c["reference"], adapter=True) != c["reference_artifact_hash"]:
        raise ValueError("Frozen reference adapter identity changed")
    from rl.conflict_reward_runtime import ConflictRewardPolicy
    p = ConflictRewardPolicy(c["model"], c["reference"], SETTINGS, training=training,
                             **({"policy_path": str(Path(checkpoint)/"adapter/policy")} if checkpoint else {}))
    cfg = p.model.peft_config["policy"]
    if cfg.r != 16 or cfg.lora_alpha != 32:
        raise ValueError("Expected the original rank16 / alpha32 language adapter")
    if any(m.p != 0 for m in p.model.modules() if isinstance(m, torch.nn.Dropout)):
        raise ValueError("Runtime dropout must be zero")
    return p


class MediaCache:
    """Only immutable encoder outputs/processor inputs, never language states.

    The frozen encoders are shared by all folds/arms. Only creation is locked;
    completed, checksum-protected entries are immutable and allow concurrent
    readers. Policy/reference decoder forwards still run on every use.
    """
    def __init__(self, c, policy):
        self.c, self.policy = c, policy
        self.root = Path(c["batch_root"]) / "media-cache"
        self.checked = set()

    def key_for(self, row):
        media = self.c["manifest_value"]["media"].get(row["video"])
        if media is None:
            media = digest(Path(self.c["data_root"])/row["video"])
        safe = {k: row[k] for k in ("video", "protocol", "text_content")}
        # VA transcript is absent even from the cache identity/input payload.
        if safe["protocol"] == "VA":
            safe["text_content"] = ""
        key = fingerprint({"input": safe, "media_sha256": media, "base": self.c["base_identity"],
                           "settings": {k: SETTINGS[k] for k in SETTINGS if k.startswith(("video_", "audio_", "max_prompt"))}})
        return key,media,safe

    def check(self):
        if STOP or time.time()>=self.c.get("_deadline_unix",float("inf")):
            raise InterruptedError("Stop/deadline requested while waiting for media cache")

    def load(self, row):
        from rl.finite_runtime import pack_cache, unpack_cache
        from rl.cache_io import cache_writer_lock
        key,media,safe=self.key_for(row)
        root = self.root/key
        path = root/"media.pt"
        marker = root/"complete.json"
        if not marker.exists():
            with cache_writer_lock(root,self.check):
                # A different process may have published it while we waited.
                if not marker.exists():
                    if digest(Path(self.c["data_root"])/row["video"]) != media:
                        raise ValueError("Media content changed before decoding")
                    prefix = self.policy.prepare({"sample_id":row["sample_id"], **safe}, self.c["data_root"])
                    features = self.policy.perception(prefix)
                    packed = pack_cache(self.policy.torch, prefix, features)
                    temp = root/"media.partial"
                    self.policy.torch.save(packed, temp)
                    temp.replace(path)
                    # Publish metadata by rename, so readers cannot observe an
                    # empty/partial marker while the writer is flushing JSON.
                    atomic(marker, {"key": key, "sha256": digest(path), "frozen_perception_only": True})
                    del prefix, features, packed
        self.check()
        m = read(marker)
        if m["key"] != key or (key not in self.checked and digest(path) != m["sha256"]):
            raise ValueError("Frozen media cache failed integrity validation")
        packed = self.policy.torch.load(path, map_location="cpu", weights_only=True)
        self.checked.add(key)
        return unpack_cache(self.policy.torch, packed)


def baseline_csv(path):
    if not path or not Path(path).exists():
        return {}
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate baseline CSV IDs")
    for row in rows:
        value = {k:row[k] for k in ("pred_emotion", "is_uncertain", "clarifying_question")}
        if parse_completion(json.dumps(value)) is None or not question_is_consistent(value):
            raise ValueError("Illegal baseline fields")
    return {r["sample_id"]:r for r in rows}


def reference_entry(policy, media, row, existing=None):
    prefix, features = media.load(row)
    if existing is None:
        with policy.cached_perception(features):
            _, answer = policy.generate(prefix, adapter="reference")
        b = parse_completion(answer)
        if b is None or not question_is_consistent(b):
            raise ValueError("Frozen reference generated illegal JSON for " + row["sample_id"] + ": " + answer)
        b = {"sample_id": row["sample_id"], **b}
    else:
        b = dict(existing)
    with policy.torch.no_grad():
        lr = float(policy.log_odds(prefix, features, b["pred_emotion"], adapter="reference"))
    if not math.isfinite(lr):
        raise ValueError("Nonfinite reference log-odds")
    del prefix, features
    return {"sample_id": row["sample_id"], "baseline": b, "reference_log_odds": lr}


def select_preflight(rows):
    tests = [lambda r:r["protocol"]=="VT", lambda r:r["protocol"]=="VA",
        lambda r:r["sample_type"]=="Conflict" and r.get("source_dataset") in ("ch_sims", "ch_sims_v2"),
        lambda r:r["sample_type"]=="Conflict" and r.get("source_dataset") not in ("ch_sims", "ch_sims_v2"),
        lambda r:not r["views"]["M1"]["is_clear"], lambda r:not r["views"]["M2"]["is_clear"],
        lambda r:r["protocol"]=="VA" and r["sample_type"]=="Conflict" and r.get("source_dataset") in ("ch_sims", "ch_sims_v2")]
    selected=[]
    for test in tests:
        row=next((r for r in rows if r not in selected and test(r)), None)
        if row is None:
            raise ValueError("The preflight partition lacks a required real sample category")
        selected.append(row)
    selected.extend(r for r in rows if r not in selected)
    return selected[:32]


def cache(args, c):
    policy = setup(args, c)
    media = MediaCache(c, policy)
    if c.get("baseline_csv_sha256") and digest(c["baseline_csv"])!=c["baseline_csv_sha256"]:
        raise ValueError("Frozen SFT held baseline CSV changed")
    known = baseline_csv(c.get("baseline_csv"))
    cache_root = Path(c["cache_root"])
    cache_root.mkdir(parents=True, exist_ok=True)
    output = cache_root/f"shard-{args.shard}.json"
    journal = cache_root/f"shard-{args.shard}.jsonl"
    previous = {}
    if journal.exists():
        previous = {r["sample_id"]:r for r in (json.loads(s) for s in journal.read_text().splitlines() if s)}
    selected = [(split,row) for split in ("train", "held") for row in c[split]][args.shard::2]
    entries=[]
    for index,(split,row) in enumerate(selected,1):
        check_deadline(args)
        entry = previous.get(row["sample_id"])
        if entry is not None:
            if entry["context_sha256"] != digest(args.context) or entry["split"] != split:
                raise ValueError("Reference cache journal belongs to another fold")
        else:
            entry = {**reference_entry(policy, media, row, known.get(row["sample_id"]) if split == "held" else None),
                     "split":split, "context_sha256":digest(args.context)}
            with journal.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry,ensure_ascii=False,allow_nan=False)+"\n")
                f.flush()
        entries.append(entry)
        emit(args,"reference_cache", current_steps=index,total_steps=len(selected),split=split)
    atomic(output,{"context_sha256":digest(args.context),"entries":entries})
    emit(args,"complete",current_steps=len(selected),total_steps=len(selected))


def cached_entries(c):
    root=Path(c["cache_root"])
    marker=read(root/"complete.json")
    result={}
    for relative,sha in marker["files"].items():
        if digest(root/relative) != sha:
            raise ValueError("Frozen fold reference cache changed")
        for entry in read(root/relative)["entries"]:
            if entry["sample_id"] in result:
                raise ValueError("Duplicate cached reference sample")
            result[entry["sample_id"]]=entry
    expected={r["sample_id"] for r in c["train"]+c["held"]}
    if set(result)!=expected:
        raise ValueError("Reference cache does not exactly cover the original partition")
    return result


def identity(args, c):
    return {"algorithm":VERSION,"context_sha256":digest(args.context),"arm":args.arm,"seed":args.seed,
            "settings":SETTINGS,"batch_identity":c["batch_identity"]}


def optimizer_for(policy, steps):
    from transformers import get_cosine_schedule_with_warmup
    parameters=[p for p in policy.model.parameters() if p.requires_grad]
    opt=policy.torch.optim.AdamW(parameters,lr=SETTINGS["learning_rate"],weight_decay=0.)
    scheduler=get_cosine_schedule_with_warmup(opt,math.ceil(steps*.1),steps)
    return opt,scheduler


def restore(policy, optimizer, scheduler, path, experiment):
    from rl.state import validate_checkpoint, restore_rng
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    validate_checkpoint(path,experiment)
    weights=load_file(str(Path(path)/"adapter/policy/adapter_model.safetensors"),device="cpu")
    set_peft_model_state_dict(policy.model,weights,adapter_name="policy")
    state=policy.torch.load(Path(path)/"training_state.pt",map_location="cpu",weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng(policy.torch,state["rng"])
    policy._set_trainability()
    return state["completed_steps"]


def update(args, policy, media, selected, entries, counts, optimizer, scheduler):
    from rl.state import capture_rng, restore_rng
    before=capture_rng(policy.torch)
    optimizer.zero_grad(set_to_none=True)
    values=Counter()
    trace=[]
    try:
        for row in selected:
            check_deadline(args)
            entry=entries[row["sample_id"]]
            prefix,features=media.load(row)
            score=policy.log_odds(prefix,features,entry["baseline"]["pred_emotion"])
            loss,m=objective(score,entry["reference_log_odds"],entry["baseline"]["is_uncertain"]=="yes",
                             row["sample_type"]=="Conflict",counts,args.arm)
            (loss/len(selected)).backward()
            for key,value in m.items():
                if key not in ("actions",):
                    values[key]+=value/len(selected)
            is_c=row["sample_type"]=="Conflict"
            values["train_conflict_n" if is_c else "train_aligned_n"]+=1
            values["train_conflict_tp" if is_c else "train_aligned_fp"]+=int(m["probability"]>=.5)
            if row["protocol"]=="VA" and is_c and row.get("source_dataset") in ("ch_sims","ch_sims_v2"):
                values["train_natural_va_n"]+=1
                values["train_natural_va_tp"]+=int(m["probability"]>=.5)
            trace.append(m["actions"])
            del prefix,features,score,loss
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        restore_rng(policy.torch,before)
        raise
    grad=policy.torch.nn.utils.clip_grad_norm_([p for p in policy.model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    if not policy.torch.isfinite(grad):
        raise FloatingPointError("Nonfinite adapter gradient")
    optimizer.step();scheduler.step()
    return {**dict(values),"grad_norm":float(grad),"lr":scheduler.get_last_lr()[0]},trace


def preflight(args,c):
    from rl.state import adapter_digest, save_checkpoint
    policy=setup(args,c,training=True)
    media=MediaCache(c,policy)
    rows=select_preflight(c["train"])
    entries={}
    for i,row in enumerate(rows,1):
        check_deadline(args)
        entries[row["sample_id"]]=reference_entry(policy,media,row)
        emit(args,"preflight_reference",current_steps=i,total_steps=32)
    ref=adapter_digest(policy,"reference")
    initial=adapter_digest(policy,"policy")
    if initial!=ref:
        raise ValueError("Zero-update policy and reference adapters differ")
    audits=[]
    for row in rows:
        entry=entries[row["sample_id"]];prefix,features=media.load(row)
        with policy.torch.no_grad():
            score=float(policy.log_odds(prefix,features,entry["baseline"]["pred_emotion"]))
        s=anchored_logit(score,entry["reference_log_odds"],entry["baseline"]["is_uncertain"]=="yes")
        if (s>=0)!=(entry["baseline"]["is_uncertain"]=="yes") or abs(score-entry["reference_log_odds"])>.02:
            raise ValueError("Zero-update path changed the reference decision or native logits")
        # Confirm cache hooks reproduce the actual frozen encoder forward for VT/VA.
        if row["protocol"] not in {a["protocol"] for a in audits}:
            raw=policy.prepare(row,c["data_root"])
            with policy.torch.no_grad():
                direct=float(policy.log_odds(raw,None,entry["baseline"]["pred_emotion"]))
            if abs(direct-score)>.02:
                raise ValueError("Raw and frozen-perception-cache forward disagree")
            audits.append({"protocol":row["protocol"],"raw_cached_log_odds_difference":abs(direct-score),
                "has_audio": "input_features" in raw,"prompt_tokens":int(raw["input_ids"].shape[1])})
            if ("input_features" in raw)!=(row["protocol"]=="VA"):
                raise ValueError("Wrong media protocol")
            del raw
        del prefix,features
    params=[{"name":n,"shape":list(p.shape),"parameters":p.numel()} for n,p in policy.model.named_parameters() if p.requires_grad]
    if not params or any("lora_" not in p["name"] or ".policy." not in p["name"] or any(x in p["name"].lower() for x in ("visual","audio","projector")) for p in params):
        raise ValueError("Actual trainable parameters violate the language-only contract")
    immutable(args.output/"trainable-parameters.json",params)
    opt,sch=optimizer_for(policy,2)
    experiment=identity(args,c)
    start=time.monotonic()
    first,_=update(args,policy,media,rows[:16],entries,counts_for(c["train"]),opt,sch)
    cp=save_checkpoint(args.output,policy,opt,sch,1,experiment)
    second,trace=update(args,policy,media,rows[16:],entries,counts_for(c["train"]),opt,sch)
    final=adapter_digest(policy,"policy")
    peak_before=policy.torch.cuda.max_memory_reserved()/1024**3
    torch=policy.torch
    del policy,media,opt,sch
    import gc
    gc.collect();torch.cuda.empty_cache()
    emit(args,"preflight_reload_and_resume",current_steps=1,total_steps=2)
    policy=setup(args,c,training=True,checkpoint=cp)
    media=MediaCache(c,policy)
    opt,sch=optimizer_for(policy,2)
    restore(policy,opt,sch,cp,experiment)
    replay,trace2=update(args,policy,media,rows[16:],entries,counts_for(c["train"]),opt,sch)
    resumed=adapter_digest(policy,"policy")
    immutable(args.output/"resume-audit.json",{"fresh_model_reloaded":True,"action_trace_identical":trace==trace2,
        "uninterrupted_digest":final,"resumed_digest":resumed,"parameter_digest_identical":final==resumed})
    if trace!=trace2 or final!=resumed:
        raise ValueError("Interrupted/resumed update differs from uninterrupted training")
    if adapter_digest(policy,"reference")!=ref or final==initial or max(first["grad_norm"],second["grad_norm"])<=0:
        raise ValueError("Real adapter update/frozen reference audit failed")
    peak=max(peak_before,policy.torch.cuda.max_memory_reserved()/1024**3)
    if peak>40+1e-6:
        raise ValueError("40 GiB preflight memory budget exceeded")
    result={"passed":True,"samples":32,"zero_update_unchanged":True,"raw_media_checks":audits,
        "resume_identical":True,"fresh_model_reloaded":True,"reference_digest":ref,"updated_policy_digest":final,
        "trainable_parameters":sum(p["parameters"] for p in params),"peak_reserved_gib":peak,
        "seconds_per_update_with_warm_cache":(time.monotonic()-start)/3,
        "observations":[first,second],"weights_discarded":True}
    immutable(args.output/"preflight.complete.json",result)
    emit(args,"complete",**result)


def train(args,c):
    from rl.state import save_checkpoint, adapter_digest, validate_checkpoint
    experiment=identity(args,c)
    immutable(args.output/"identity.json",experiment)
    complete=args.output/"train.complete.json"
    if complete.exists():
        marker=read(complete);validate_checkpoint(marker["checkpoint"],experiment)
        return marker
    latest=args.output/"latest_checkpoint.json"
    cp=args.output/read(latest)["path"] if latest.exists() else None
    policy=setup(args,c,training=True,checkpoint=cp)
    entries=cached_entries(c);media=MediaCache(c,policy)
    schedule=[(epoch,batch) for epoch in range(2) for batch in batches(c["train"],args.seed,epoch)]
    opt,sch=optimizer_for(policy,len(schedule))
    step=restore(policy,opt,sch,cp,experiment) if cp else 0
    reference=adapter_digest(policy,"reference")
    totals=Counter();start=time.monotonic();start_step=step
    try:
        while step<len(schedule):
            epoch,selected=schedule[step]
            m,_=update(args,policy,media,selected,entries,counts_for(c["train"]),opt,sch)
            step+=1
            for k in ("train_conflict_n","train_conflict_tp","train_aligned_n","train_aligned_fp","train_natural_va_n","train_natural_va_tp"):
                totals[k]+=m.get(k,0)
            with (args.output/"training.jsonl").open("a",encoding="utf-8") as f:
                f.write(json.dumps({"step":step,"epoch":epoch,**m,"training_observations_only":True})+"\n")
            emit(args,"training",current_steps=step,total_steps=len(schedule),epoch=epoch+(step % math.ceil(len(c["train"])/16) or math.ceil(len(c["train"])/16))/math.ceil(len(c["train"])/16),
                 elapsed_seconds=time.monotonic()-start,eta_seconds=(time.monotonic()-start)/max(1,step-start_step)*(len(schedule)-step),
                 peak_reserved_gib=policy.torch.cuda.max_memory_reserved()/1024**3,**m,training_counts=dict(totals))
            if step%SETTINGS["save_every"]==0 or step==len(schedule):
                save_checkpoint(args.output,policy,opt,sch,step,experiment)
    except InterruptedError:
        save_checkpoint(args.output,policy,opt,sch,step,experiment)
        emit(args,"stopped_checkpoint_saved",current_steps=step,total_steps=len(schedule))
        raise
    if adapter_digest(policy,"reference")!=reference:
        raise ValueError("Frozen reference changed during RL")
    cp=save_checkpoint(args.output,policy,opt,sch,step,experiment)
    marker={"identity":experiment,"checkpoint":str(cp),"completed_steps":step,"epochs":2,
            "policy_digest":adapter_digest(policy,"policy"),"reference_digest":reference,
            "checkpoint_complete_sha256":digest(cp/"complete.json"),"official_inputs_used":False}
    immutable(complete,marker)
    emit(args,"complete",current_steps=step,total_steps=len(schedule))


def predict(args,c):
    from rl.state import validate_checkpoint
    marker=read(args.output/"train.complete.json")
    validate_checkpoint(marker["checkpoint"],identity(args,c))
    if marker["completed_steps"]!=2*math.ceil(len(c["train"])/16):
        raise ValueError("Only the fixed, final two-epoch checkpoint may be evaluated")
    entries=cached_entries(c)
    policy=setup(args,c,checkpoint=marker["checkpoint"])
    media=MediaCache(c,policy);results=[]
    for i,row in enumerate(c["held"],1):
        check_deadline(args)
        entry=entries[row["sample_id"]];b=entry["baseline"]
        prefix,features=media.load(row)
        with policy.torch.no_grad():
            score=float(policy.log_odds(prefix,features,b["pred_emotion"]))
        delta=score-entry["reference_log_odds"]
        s=anchored_logit(score,entry["reference_log_odds"],b["is_uncertain"]=="yes")
        q=1/(1+math.exp(-max(-700,min(700,s))))
        results.append(predicted(b,q,delta=delta))
        emit(args,"held_final_evaluation",current_steps=i,total_steps=len(c["held"]))
        del prefix,features
    path=args.output/"predictions.jsonl"
    temp=path.with_suffix(".partial")
    temp.write_text("".join(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n" for r in results),encoding="utf-8")
    temp.replace(path)
    immutable(args.output/"predict.complete.json",{"sha256":digest(path),"rows":len(results),
        "train_complete_sha256":digest(args.output/"train.complete.json"),"context_sha256":digest(args.context)})
    emit(args,"complete",current_steps=len(results),total_steps=len(results))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("cache","preflight","train","predict"))
    parser.add_argument("--context",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gpu",type=int,required=True)
    parser.add_argument("--arm",choices=("A","B","C"),default="A")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--shard",type=int,choices=(0,1),default=0)
    parser.add_argument("--deadline-unix",type=float,required=True)
    args=parser.parse_args(argv)
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    try:
        c=context(args)
        {"cache":cache,"preflight":preflight,"train":train,"predict":predict}[args.command](args,c)
    except BaseException as exc:
        emit(args,"stopped" if isinstance(exc,InterruptedError) else "failed",error=str(exc),error_type=type(exc).__name__)
        raise


if __name__=="__main__":
    main()
