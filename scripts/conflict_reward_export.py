"""Locked official inference and CSV export, reachable only after all gates."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import signal

from rl.conflict_reward import predicted
from rl.grounded_uncertainty import main_prefix
from rl.reward import parse_completion, question_is_consistent
from scripts.joint_affect_data import canonical_inference, question_for_protocol, metrics, FIELDS
from scripts.joint_affect_pipeline import read, digest, immutable, atomic, public_metrics, independent_audit
from scripts.train_conflict_reward import (context, setup, MediaCache, reference_entry, emit,
    check_deadline, request_stop, baseline_csv, identity)


def locked(args,c):
    root=Path(c["batch_root"]);lock=read(root/"final-model-lock.json")
    if not lock["all_gates_passed"] or lock["context_sha256"]!=digest(args.context) or lock["arm"]!=args.arm:
        raise ValueError("Official inference requires the fixed, passing model lock")
    for item in lock["models"]:
        path=Path(item["path"])/"train.complete.json"
        if digest(path)!=item["train_complete_sha256"]:
            raise ValueError("Final training receipt changed after model locking")
        from rl.state import validate_checkpoint
        marker=read(path)
        validate_checkpoint(marker["checkpoint"],marker["identity"])
    return root,lock


def official_rows(c,split):
    filename,count=("val.jsonl",666) if split=="val" else ("test_release.jsonl",734)
    path=Path(c["data_root"])/filename
    raw=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(raw)!=count or len({r["sample_id"] for r in raw})!=count:
        raise ValueError("Official input coverage changed")
    # Test labels/views/source metadata never enter the inference representation.
    safe=[]
    for r in raw:
        v=canonical_inference({k:r[k] for k in ("sample_id","video","protocol","text_content","text") if k in r})
        safe.append({k:v[k] for k in ("sample_id","video","protocol","text_content")})
    return path,safe


def cache(args,c):
    root,lock=locked(args,c);path,rows=official_rows(c,args.split)
    source=digest(path)
    base=baseline_csv(root/"submission09.csv") if args.split=="test" else {}
    if base and (digest(root/"submission09.csv")!=lock["baseline_sha256"] or set(base)!={r["sample_id"] for r in rows}):
        raise ValueError("Submission09 identity/coverage changed")
    policy=setup(args,c);media=MediaCache(c,policy)
    entries=[]
    journal=args.output/"entries.jsonl"
    existing={r["sample_id"]:r for r in (json.loads(s) for s in journal.read_text().splitlines())} if journal.exists() else {}
    for i,row in enumerate(rows,1):
        check_deadline(args)
        entry=existing.get(row["sample_id"])
        if entry is None:
            native=reference_entry(policy,media,row)
            comp=reference_entry(policy,media,row,base[row["sample_id"]]) if base else native
            entry={"sample_id":row["sample_id"],"native":native,"submission":comp,
                   "source_sha256":source,"lock_sha256":digest(root/"final-model-lock.json")}
            with journal.open("a",encoding="utf-8") as f:f.write(json.dumps(entry,ensure_ascii=False,allow_nan=False)+"\n")
        if entry["source_sha256"]!=source or entry["lock_sha256"]!=digest(root/"final-model-lock.json"):
            raise ValueError("Official reference journal identity changed")
        entries.append(entry)
        emit(args,"locked_official_reference",current_steps=i,total_steps=len(rows),split=args.split)
    if digest(path)!=source:raise ValueError("Official input changed during inference")
    atomic(args.output/"entries.json",entries)
    immutable(args.output/"complete.json",{"sha256":digest(args.output/"entries.json"),"rows":len(rows),
              "source_sha256":source,"lock_sha256":digest(root/"final-model-lock.json")})


def load_entries(root,split):
    folder=root/f"official/{split}/reference";m=read(folder/"complete.json")
    if digest(folder/"entries.json")!=m["sha256"] or digest(root/"final-model-lock.json")!=m["lock_sha256"]:
        raise ValueError("Locked official reference cache changed")
    return read(folder/"entries.json")


def predict(args,c):
    root,lock=locked(args,c)
    if str(args.output) not in {item["path"] for item in lock["models"]}:
        raise ValueError("Official worker is not one of the two locked models")
    marker=read(args.output/"train.complete.json")
    if marker["identity"]!=identity(args,c):raise ValueError("Seed/arm context mismatch")
    path,rows=official_rows(c,args.split);entries=load_entries(root,args.split)
    if [r["sample_id"] for r in rows]!=[r["sample_id"] for r in entries]:raise ValueError("Official IDs differ")
    policy=setup(args,c,checkpoint=marker["checkpoint"]);media=MediaCache(c,policy);results=[]
    for i,(row,entry) in enumerate(zip(rows,entries),1):
        check_deadline(args)
        prefix,features=media.load(row);scores={}
        for mode in ("native","submission"):
            e=entry[mode]["baseline"]["pred_emotion"]
            if e not in scores:
                with policy.torch.no_grad():scores[e]=float(policy.log_odds(prefix,features,e))
        results.append({"sample_id":row["sample_id"],**{mode+"_delta":scores[entry[mode]["baseline"]["pred_emotion"]]-entry[mode]["reference_log_odds"] for mode in ("native","submission")}})
        del prefix,features
        emit(args,"locked_official_prediction",current_steps=i,total_steps=len(rows),split=args.split)
    output=args.output/f"official-{args.split}.json"
    atomic(output,results)
    immutable(args.output/f"official-{args.split}.complete.json",{"sha256":digest(output),"source_sha256":digest(path),
        "lock_sha256":digest(root/"final-model-lock.json"),"rows":len(rows)})


def combined(lock,entries,split,mode):
    predictions=[]
    for item in lock["models"]:
        folder=Path(item["path"]);path=folder/f"official-{split}.json";m=read(folder/f"official-{split}.complete.json")
        if digest(path)!=m["sha256"]:raise ValueError("Final seed predictions changed")
        predictions.append(read(path))
    if any([p["sample_id"] for p in values]!=[e["sample_id"] for e in entries] for values in predictions):
        raise ValueError("Final ensemble order mismatch")
    result=[]
    for entry,a,b in zip(entries,*predictions):
        delta=(a[mode+"_delta"]+b[mode+"_delta"])/2
        baseline=entry[mode]["baseline"]
        s=math.log(3.)*(1 if baseline["is_uncertain"]=="yes" else -1)+delta
        p=1/(1+math.exp(-max(-700,min(700,s))))
        result.append(predicted(baseline,p,delta=delta))
    return result


def question(policy,media,row,b):
    prefix,features=media.load(row)
    start=main_prefix(b["pred_emotion"])+ 'yes","clarifying_question":'
    try:
        with policy.cached_perception(features):
            _,tail=policy.generate(policy.extend(prefix,start),adapter="reference")
        parsed=parse_completion(start+tail)
        if parsed and question_is_consistent(parsed) and parsed["pred_emotion"]==b["pred_emotion"] and parsed["is_uncertain"]=="yes":
            return parsed["clarifying_question"],"frozen_reference_generation"
    finally:del prefix,features
    return question_for_protocol(row["protocol"]),"protocol_template_after_invalid_generation"


def render(rows,entries,predictions,question_fn):
    if [r["sample_id"] for r in rows]!=[p["sample_id"] for p in predictions] or [r["sample_id"] for r in rows]!=[e["sample_id"] for e in entries]:
        raise ValueError("CSV rows must preserve exact ID order")
    output=[];changes=[]
    for row,entry,p in zip(rows,entries,predictions):
        b=entry["submission"]["baseline"]
        u="yes" if p["uncertain_prob"]>=.5 else "no"
        provenance="unchanged"
        if u=="no":q="none"
        elif b["is_uncertain"]=="yes":q=b["clarifying_question"]
        else:q,provenance=question_fn(row,b)
        value={"sample_id":row["sample_id"],"pred_emotion":b["pred_emotion"],"is_uncertain":u,"clarifying_question":q}
        if value["pred_emotion"]!=b["pred_emotion"] or not question_is_consistent(value):
            raise ValueError("CSV changed emotion or violated question consistency")
        output.append(value)
        if u!=b["is_uncertain"]:
            changes.append({"sample_id":row["sample_id"],"before":b,"after":value,"probability":p["uncertain_prob"],"mean_delta":p["delta"],"question_source":provenance})
    stream=io.StringIO(newline="");writer=csv.DictWriter(stream,fieldnames=FIELDS)
    writer.writeheader();writer.writerows(output)
    return stream.getvalue(),changes


def export(args,c):
    root,lock=locked(args,c);args.output.mkdir(parents=True,exist_ok=True)
    path,rows=official_rows(c,"val");entries=load_entries(root,"val")
    native=combined(lock,entries,"val","native")
    raw=[canonical_inference(json.loads(s)) for s in path.read_text(encoding="utf-8").splitlines() if s]
    score=metrics(raw,native);audit=independent_audit(raw,native,score)
    immutable(args.output/"independent-model-validation.json",{"metrics":public_metrics(score),"audit":audit,"used_for_selection":False})
    path,rows=official_rows(c,"test");entries=load_entries(root,"test")
    native=combined(lock,entries,"test","native");comp=combined(lock,entries,"test","submission")
    immutable(args.output/"independent-model-test.json",native)
    policy=setup(args,c);media=MediaCache(c,policy)
    text,changes=render(rows,entries,comp,lambda r,b:question(policy,media,r,b))
    out=args.output/"candidate.csv"
    if out.exists() and out.read_text(encoding="utf-8")!=text.replace("\r\n","\n"):
        raise ValueError("Refusing to replace a different fixed candidate")
    if not out.exists():out.write_bytes(text.encode("utf-8"))
    immutable(args.output/"changes.json",changes)
    immutable(args.output/"delivery-report.json",{"rows":734,"csv_sha256":digest(out),"emotion_changes":0,
        "uncertainty_changes":len(changes),"yes_to_no":sum(x["before"]["is_uncertain"]=="yes" for x in changes),
        "no_to_yes":sum(x["before"]["is_uncertain"]=="no" for x in changes),"mean_log_odds_increment_ensemble":True,
        "threshold":.5,"baseline_sha256":lock["baseline_sha256"],"submitted_to_kaggle":False,
        "submission09_fusion_has_no_independent_labelled_validation":True})
    emit(args,"complete",current_steps=734,total_steps=734)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("cache","predict","export"));p.add_argument("--context",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--arm",choices=("A","B","C"),required=True);p.add_argument("--seed",type=int,required=True)
    p.add_argument("--split",choices=("val","test"),required=True);p.add_argument("--deadline-unix",type=float,required=True)
    a=p.parse_args(argv);signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    c=context(a)
    {"cache":cache,"predict":predict,"export":export}[a.command](a,c)


if __name__=="__main__":main()
