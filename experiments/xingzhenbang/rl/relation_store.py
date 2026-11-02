"""Immutable relation experiment identities and atomically published features."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import uuid

from .experiment import ROOT, artifact_hash, check_disjoint, file_hash, fingerprint, load_records, write_json
from .grounded_store import cache_contract, source_hash_at, task_key
from .grounded_uncertainty import baseline_csv, load_training
from .modality_relation import VERSION, allowed_states, make_folds, settings_from
from .state import software_versions
from .grounded_recovery import EXECUTION


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_torch(path, value, torch):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+".partial-"+uuid.uuid4().hex)
    try:
        torch.save(value, temp)
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def context_for(run):
    c = read(Path(run)/"context.json")
    if c["identity_hash"] != fingerprint(c["identity"]) or c["identity"]["algorithm"] != VERSION:
        raise ValueError("Wrong algorithm or corrupt relation identity")
    if c["identity"]["source"] != source_hash_at(ROOT):
        raise ValueError("Running code differs from the immutable relation release")
    return c


def initialize(args):
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    if (run/"context.json").exists():
        if not args.resume:
            raise ValueError("Existing relation run requires --resume")
        return context_for(run)
    source, release = Path(args.source_run).resolve(), Path(args.source_release).resolve()
    old, manifest = read(source/"context.json"), read(source/"cache/complete.json")
    if old["identity_hash"] != fingerprint(old["identity"]) or manifest["identity_hash"] != old["identity_hash"]:
        raise ValueError("Source cache identity is corrupt")
    if source_hash_at(release) != old["identity"]["source"] or cache_contract(release) != cache_contract(ROOT):
        raise ValueError("Frozen source perception/prompt code changed")
    data_root = Path(args.data_root).resolve()
    train, val = load_training(data_root/"train.jsonl"), load_training(data_root/"val.jsonl")
    test, duplicates = load_records(data_root/"test_release.jsonl", require_labels=False)
    if (len(train), len(val), len(test)) != (3354, 666, 734) or duplicates:
        raise ValueError("Official split counts changed")
    for a, b in ((train,val), (train,test), (val,test)):
        check_disjoint(a,b)
    for row in train+val:
        allowed_states(row)
    settings = settings_from(args.config)
    for key in ("video_fps", "video_min_frames", "video_max_frames", "video_pixels",
                "audio_sampling_rate", "max_prompt_tokens", "attn_implementation"):
        if settings[key] != old["settings"][key]:
            raise ValueError("Source media precision/processing settings changed")
    frozen_test = baseline_csv(args.baseline_csv, test)
    media = {}
    for row in train+val+test:
        relative = row["video"]
        path = (data_root/relative).resolve()
        if not path.is_relative_to(data_root) or not path.is_file():
            raise ValueError("Missing media or escaped data root")
        if relative not in media:
            media[relative] = file_hash(path)
    model, adapter = str(Path(args.model_path).resolve()), str(Path(args.sft_adapter_path).resolve())
    identity = {"algorithm": VERSION, "source": source_hash_at(ROOT), "settings": settings,"execution":EXECUTION,
        "software": software_versions(), "base": artifact_hash(model), "sft": artifact_hash(adapter, adapter=True),
        "media": media, "raw_split_hashes": {s:file_hash(data_root/name) for s,name in
            (("train","train.jsonl"),("val","val.jsonl"),("test","test_release.jsonl"))},
        "baseline_csv_sha256": file_hash(args.baseline_csv), "source_identity": old["identity_hash"],
        "source_manifest_sha256": file_hash(source/"cache/complete.json"),
        "feature_contract": {"shape":[3,2048], "order":["vision","second","full"],
            "position":"last_effective_token_before_value_lm_head", "adapter":"original_v4_reference",
            "storage_dtype":"float32", "extraction_compute":"nf4_bfloat16"}}
    for key in ("base", "sft", "media", "raw_split_hashes", "baseline_csv_sha256", "software"):
        if identity[key] != old["identity"][key]:
            raise ValueError(f"Source model/data/runtime differs: {key}")
    baselines = {}
    for split, rows in (("train",train),("val",val)):
        old_rows = {r["sample_id"]:r for r in old[split]}
        baselines[split] = {}
        for row in rows:
            previous = old_rows.get(row["sample_id"], {})
            for key in ("sample_id","video","protocol","text_content","gt_label","sample_type","split_group_id"):
                if row[key] != previous.get(key):
                    raise ValueError("Source canonical inputs or original labels changed")
            e = manifest["entries"][task_key(row["sample_id"],"full")]
            if (e["split"],e["sample_id"],e["mode"]) != (split,row["sample_id"],"full"):
                raise ValueError("Source cache split/mode boundary changed")
            baselines[split][row["sample_id"]] = e["baseline"]
    folds = make_folds(train, settings["seed"], settings["folds"])
    identity["fold_hash"] = fingerprint(list(zip([r["sample_id"] for r in train],folds)))
    c = {"identity":identity, "identity_hash":fingerprint(identity), "settings":settings,
        "model":model,"adapter":adapter,"data_root":str(data_root),"source_run":str(source),
        "source_release":str(release),"train":train,"val":val,"test":test,
        "baselines":baselines,"baseline_test":frozen_test,"folds":folds,
        "started":time.time(),"deadline":None,"training_cutoff":None}
    import shutil
    frozen=run/"baseline_090813.csv"
    shutil.copyfile(args.baseline_csv,frozen)
    frozen.chmod(0o444)
    write_json(run/"context.json",c)
    write_json(run/"folds.json", {"identity_hash":c["identity_hash"],"assignments":dict(zip(
        [r["sample_id"] for r in train],folds))})
    return c


def feature_file(run, split, sid):
    return Path(run)/"features"/split/(task_key(sid,"three_vectors")+".pt")


def feature_entry(run, c, split, row, torch, *, required=False):
    path = feature_file(run,split,row["sample_id"])
    meta = path.with_suffix(".json")
    if not meta.exists():
        if required:
            raise ValueError("Feature completion marker missing")
        return None
    e = read(meta)
    if ((e.get("identity_hash"),e.get("sample_id"),e.get("split"),e.get("file")) !=
            (c["identity_hash"],row["sample_id"],split,path.name) or file_hash(path) != e["sha256"]):
        raise ValueError("Feature crossed an identity/sample/split or failed checksum")
    value = torch.load(path,map_location="cpu",weights_only=True)
    if value.shape != (3,2048) or value.dtype != torch.float32 or not torch.isfinite(value).all():
        raise ValueError("Feature shape/precision/finiteness invalid")
    return e,value


def save_feature(run,c,split,row,x,diagnostics,torch):
    path = feature_file(run,split,row["sample_id"])
    x=x.detach().cpu().float().contiguous()
    if x.shape != (3,2048) or not torch.isfinite(x).all():
        raise ValueError("Invalid extracted feature")
    atomic_torch(path,x,torch)
    write_json(path.with_suffix(".json"), {"identity_hash":c["identity_hash"],"split":split,
        "sample_id":row["sample_id"],"file":path.name,"sha256":file_hash(path),"diagnostics":diagnostics})
    loaded=feature_entry(run,c,split,row,torch,required=True)[1]
    if not torch.equal(x,loaded):
        raise ValueError("Feature disk round-trip changed values")


def publish_features(run,c,splits,torch):
    for split in splits:
        entries=[feature_entry(run,c,split,row,torch,required=True)[0] for row in c[split]]
        write_json(Path(run)/"features"/split/"complete.json", {"identity_hash":c["identity_hash"],
            "entries":entries,"sample_order":[r["sample_id"] for r in c[split]]})


def load_matrix(run,c,split,torch):
    manifest=read(Path(run)/"features"/split/"complete.json")
    rows=c[split]
    if manifest["identity_hash"] != c["identity_hash"] or manifest["sample_order"] != [r["sample_id"] for r in rows] or len(manifest["entries"])!=len(rows):
        raise ValueError("Feature matrix order/identity changed")
    values=[]
    for row,entry in zip(rows,manifest["entries"]):
        e,x=feature_entry(run,c,split,row,torch,required=True)
        if e != entry:
            raise ValueError("Published feature metadata changed")
        values.append(x)
    return torch.stack(values)


def save_head(path,payload,c,torch):
    atomic_torch(path,payload,torch)
    write_json(Path(str(path)+".json"),{"algorithm":VERSION,"identity_hash":c["identity_hash"],
        "sha256":file_hash(path),"method":payload["method"],"epoch":payload["epoch"],
        "seed":payload["seed"],"fold":payload["fold"],"train_ids_hash":payload["train_ids_hash"]})


def load_head(path,c,torch):
    marker=read(str(path)+".json")
    if marker.get("algorithm") != VERSION or marker.get("identity_hash") != c["identity_hash"] or marker["sha256"] != file_hash(path):
        raise ValueError("Wrong algorithm or corrupt relation checkpoint")
    payload=torch.load(path,map_location="cpu",weights_only=False)
    for key in ("method","epoch","seed","fold","train_ids_hash"):
        if payload[key] != marker[key]:
            raise ValueError("Checkpoint manifest mismatch")
    return payload
