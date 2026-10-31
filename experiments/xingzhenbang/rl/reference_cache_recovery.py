"""Attested import of immutable reference predictions after an I/O-only repair.

No policy checkpoints or optimizer state are imported. Scientific settings,
folds, reference weights, prompts and scoring code must remain identical.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

from rl.experiment import fingerprint
from scripts.joint_affect_pipeline import read,digest,immutable

REFERENCE_SOURCES=("rl/conflict_reward.py","rl/conflict_reward_runtime.py","rl/finite_runtime.py",
    "rl/finite_actions.py","rl/omni_runtime.py","rl/omni_io.py","sft_prompt.py")


def import_reference_cache(new_context,old_context):
    new_context,old_context=map(Path,(new_context,old_context))
    new,old=read(new_context),read(old_context)
    allowed={"batch_root","batch_identity","cache_root","manifest"}
    if any(new.get(k)!=old.get(k) for k in (set(new)|set(old))-allowed):
        raise ValueError("Reference context changed beyond output/provenance paths")
    plans=[]
    for c in (old,new):
        plan=read(Path(c["batch_root"])/"plan.json")
        if fingerprint(plan["identity"])!=plan["identity_hash"] or plan["identity_hash"]!=c["batch_identity"]:
            raise ValueError("Reference recovery needs authentic frozen batch identities")
        if digest(c["manifest"])!=c["manifest_sha256"]:
            raise ValueError("Reference manifest content changed")
        plans.append(plan["identity"])
    if plans[0]["settings"]!=plans[1]["settings"] or any(plans[0]["source_sha256"].get(p)!=plans[1]["source_sha256"].get(p) for p in REFERENCE_SOURCES):
        raise ValueError("Reference recovery cannot change scientific settings or inference sources")
    oldroot,newroot=Path(old["cache_root"]),Path(new["cache_root"])
    marker=read(oldroot/"complete.json")
    if marker["context_sha256"]!=digest(old_context):
        raise ValueError("Original reference completion marker uses another context")
    manifest=read(new["manifest"])
    rows=manifest["train"]
    if new["fold"]=="full":
        train,held=list(range(len(rows))),[]
    else:
        part=manifest["folds"][new["grouping"]]["folds"][int(new["fold"])];train,held=part["train"],part["held"]
    expected={rows[i]["sample_id"]:split for split,indices in (("train",train),("held",held)) for i in indices}
    seen=set();files={};origin=[]
    for name,sha in sorted(marker["files"].items()):
        path=(oldroot/name).resolve()
        if not path.is_relative_to(oldroot.resolve()) or digest(path)!=sha:
            raise ValueError("Original reference shard integrity failure")
        value=read(path)
        if value["context_sha256"]!=digest(old_context):
            raise ValueError("Original reference shard context differs")
        wrapped=[]
        for entry in value["entries"]:
            sid=entry["sample_id"]
            if sid in seen or sid not in expected or entry["split"]!=expected[sid] or entry["context_sha256"]!=digest(old_context):
                raise ValueError("Reference cache has duplicate IDs or crossed partitions")
            if entry["baseline"]["sample_id"]!=sid or not math.isfinite(entry["reference_log_odds"]):
                raise ValueError("Invalid reference prediction")
            seen.add(sid)
            item=deepcopy(entry)
            item.update(context_sha256=digest(new_context),recovered_from_context_sha256=digest(old_context),
                        original_reference_entry_hash=fingerprint(entry))
            if any(item[k]!=entry[k] for k in ("sample_id","split","baseline","reference_log_odds")):
                raise AssertionError("Reference values must remain exactly unchanged")
            wrapped.append(item)
        destination=newroot/Path(name).name
        immutable(destination,{"context_sha256":digest(new_context),"entries":wrapped})
        files[destination.name]=digest(destination);origin.append({"path":str(path),"sha256":sha})
    if seen!=set(expected):
        raise ValueError("Recovered reference cache lacks exact partition coverage")
    result={"files":files,"context_sha256":digest(new_context),"reference_values_unchanged":True,
        "rows":len(seen),"recovery_source":{"context":str(old_context),"context_sha256":digest(old_context),
            "completion_sha256":digest(oldroot/"complete.json"),"shards":origin},
        "policy_checkpoints_imported":False}
    immutable(newroot/"complete.json",result)
    return result
