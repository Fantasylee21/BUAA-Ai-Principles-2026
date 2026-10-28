"""Gated server-only A/B/C experiment; no Kaggle submission capability."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import traceback

from rl.conflict_reward import VERSION, SETTINGS, ARMS, predicted, compare, choose, bootstrap
from rl.experiment import artifact_hash, fingerprint
from scripts.joint_affect_pipeline import (SOURCE, read, atomic, immutable, digest,
    run_lock, Budget, process_token, verify_environment, runtime_identity)

BASELINE_SHA = "f2551b6a12fc916d804287671373bee5caa6e593315725a36ca2b1b0e20b8498"


def sources():
    names = {p for folder in ("rl", "scripts") for p in (SOURCE/folder).glob("*.py")}
    names.update(SOURCE/p for p in ("sft_prompt.py","qwen25vl_baseline.py",
        "configs/sft/qwen25omni/qlora_sft_v4_resample2x.yaml","scripts/run_conflict_reward_server.sh"))
    return {p.relative_to(SOURCE).as_posix():digest(p) for p in sorted(names)}


def prepare(project, root, previous, baseline):
    from scripts.joint_affect_data import validate_manifest
    from scripts.train_joint_affect import directory_hash
    from scripts.train_conflict_reward import baseline_csv
    project,root,previous=map(lambda p:Path(p).resolve(),(project,root,previous))
    if not root.is_relative_to(project/"runs") or root==previous:
        raise ValueError("A new, separate ignored run directory is required")
    if (root/"plan.json").exists():
        old=read(root/"plan.json")
        if old["identity"]["source_sha256"]!=sources():
            raise ValueError("An existing run cannot use changed source")
        return old
    old=read(previous/"plan.json")
    verify_environment(project,old["identity"])
    manifest=read(previous/"train-manifest.json");validate_manifest(manifest)
    refs=[]
    for fold in range(3):
        job=previous/f"cohort-64/jobs/content/omni/SFT-ref/seed42/fold{fold}"
        marker=read(job/"train.complete.json")
        if directory_hash(marker["checkpoint"])!=marker["checkpoint_sha256"]:
            raise ValueError("An original SFT reference checkpoint changed")
        csv=job/"reference-evaluation/held_predictions.csv"
        rows=baseline_csv(csv)
        expected={manifest["train"][i]["sample_id"] for i in manifest["folds"]["content"]["folds"][fold]["held"]}
        if set(rows)!=expected:
            raise ValueError("Original SFT held predictions do not match their fold")
        refs.append({"reference":marker["checkpoint"],"reference_artifact_hash":artifact_hash(marker["checkpoint"],adapter=True),
                     "baseline_csv":str(csv),"baseline_csv_sha256":digest(csv),"old_train_marker_sha256":digest(job/"train.complete.json")})
    if digest(baseline)!=BASELINE_SHA or len(baseline_csv(baseline))!=734:
        raise ValueError("Submission09 must be the confirmed, exact 0.92313 artifact")
    root.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(previous/"train-manifest.json",root/"train-manifest.json")
    shutil.copyfile(baseline,root/"submission09.csv")
    identity={"algorithm":VERSION,"project_root":str(project),"previous_run":str(previous),
        "source_sha256":sources(),"runtime":runtime_identity(),"settings":SETTINGS,
        "manifest_sha256":digest(root/"train-manifest.json"),"references":refs,
        "base_identity":fingerprint(old["identity"]["base_artifacts"]),
        "base_environment":{k:old["identity"][k] for k in ("base_artifacts","framework_artifacts")},
        "baseline_sha256":BASELINE_SHA,"pilot_gain":.003,"primary_gain":.005,
        "replication_gain":.003,"pressure_gain":.0025,"tie_tolerance":.0001,
        "grouping":"copied_original_content_and_parent_manifest", "submitted_to_kaggle":False}
    result={"identity":identity,"identity_hash":fingerprint(identity)}
    immutable(root/"plan.json",result)
    return result


class Pipeline:
    def __init__(self,root):
        self.root=Path(root).resolve();self.plan=read(self.root/"plan.json")
        self.project=Path(self.plan["identity"]["project_root"])
        self.manifest=read(self.root/"train-manifest.json");self.rows=self.manifest["train"]
        self.children={};self.mutex=threading.RLock();self.cancel=threading.Event()
        self.gpus=queue.Queue()
        for gpu in (1,2):self.gpus.put(gpu)
        self.budget=None;self.phase="initializing";self.pilot=not (self.root/"pilot-passed.json").exists()

    def state(self,phase=None,**extra):
        with self.mutex:
            if phase:self.phase=phase
            atomic(self.root/"status.json",{"algorithm":VERSION,"phase":self.phase,"updated_unix":time.time(),
                "active_jobs":dict(self.children),"submitted_to_kaggle":False,**extra})

    def deadline(self):
        d=self.budget.tick()
        if self.pilot:d=min(d,self.budget.started+24*3600)
        if self.cancel.is_set() or time.time()>=d:
            raise TimeoutError("The fixed pilot/batch deadline or cancellation was reached")
        return d

    @contextmanager
    def resource(self):
        while True:
            self.deadline()
            try:
                gpu=self.gpus.get(timeout=1);break
            except queue.Empty:pass
        try:
            # No killing/sharing an occupied card, even when memory looks ample.
            devices=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True)
            uuid=dict(line.replace(" ","").split(",") for line in devices.strip().splitlines())[str(gpu)]
            apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],text=True)
            if any(line.split(",")[0].strip()==uuid for line in apps.splitlines()):
                raise RuntimeError(f"GPU {gpu} is occupied; its processes were not touched")
            yield gpu
        finally:self.gpus.put(gpu)

    def child(self,command,output,gpu):
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy();env.update(OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4",
            HF_HUB_OFFLINE="1",HF_DATASETS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
        env["PATH"]=str(self.project/"media-tools/bin")+os.pathsep+env.get("PATH","")
        command=list(map(str,command))+["--deadline-unix",str(self.deadline())]
        with (output/"worker.log").open("a") as log:
            p=subprocess.Popen(command,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            key=str(output.relative_to(self.root))
            value={"pid":p.pid,"process_start_ticks":process_token(p.pid),"gpu":gpu,"command":command}
            atomic(output/"process.json",value)
            with self.mutex:self.children[key]=value
            try:
                while p.poll() is None:
                    self.deadline();self.state();time.sleep(2)
                if p.returncode:
                    tail=(output/"worker.log").read_text(errors="replace")[-5000:]
                    raise RuntimeError(f"Worker exited {p.returncode}: {key}\n{tail}")
            except BaseException:
                self.cancel.set()
                if p.poll() is None:
                    os.killpg(p.pid,signal.SIGTERM)
                    # Checkpoint handlers can need >30s on NAS. Never kill them
                    # merely because the controller is leaving its thread pool.
                    p.wait(timeout=180)
                raise
            finally:
                with self.mutex:self.children.pop(key,None)

    def worker(self,mode,context,output,arm="A",seed=42,shard=0):
        with self.resource() as gpu:
            command=[sys.executable,"-B","-m","scripts.train_conflict_reward",mode,"--context",context,
                     "--output",output,"--gpu",gpu,"--arm",arm,"--seed",seed,"--shard",shard]
            self.child(command,output,gpu)

    def parallel(self,tasks):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(fn,*args) for fn,args in tasks]
            try:
                results=[None]*len(futures)
                positions={future:i for i,future in enumerate(futures)}
                for future in as_completed(futures):results[positions[future]]=future.result()
                return results
            except BaseException:
                self.cancel.set()
                # Signal exact live child identities before executor waits.
                with self.mutex:
                    children=list(self.children.values())
                for c in children:
                    if process_token(c["pid"])==c["process_start_ticks"]:
                        os.killpg(c["pid"],signal.SIGTERM)
                raise

    def reference(self,grouping,fold):
        if grouping=="content" and fold!="full":
            return self.plan["identity"]["references"][int(fold)]
        job=self.root/f"references/{grouping}/fold{fold}/sft"
        marker=job/"train.complete.json"
        if not marker.exists():
            with self.resource() as gpu:
                command=[sys.executable,"-B","-m","scripts.train_joint_affect","train","--project-root",self.project,
                    "--run-dir",job,"--manifest",self.root/"train-manifest.json","--variant","SFT-ref",
                    "--fold",fold,"--grouping",grouping,"--seed",42,"--vision-model","omni","--max-frames",64,"--gpus",gpu]
                self.child(command,job,gpu)
        from scripts.train_joint_affect import directory_hash
        m=read(marker)
        if directory_hash(m["checkpoint"])!=m["checkpoint_sha256"]:
            raise ValueError("Reference checkpoint changed")
        return {"reference":m["checkpoint"],"reference_artifact_hash":artifact_hash(m["checkpoint"],adapter=True),
                "baseline_csv":None}

    def context(self,grouping,fold):
        root=self.root/f"references/{grouping}/fold{fold}"
        ref=self.reference(grouping,fold)
        c={"batch_root":str(self.root),"batch_identity":self.plan["identity_hash"],
            "manifest":str(self.root/"train-manifest.json"),"manifest_sha256":self.plan["identity"]["manifest_sha256"],
            "grouping":grouping,"fold":fold,"data_root":self.manifest["data_root"],
            "model":str(self.project/"models/Qwen2.5-Omni-3B"),"project_root":str(self.project),
            "base_identity":self.plan["identity"]["base_identity"],"cache_root":str(root/"cache"),**ref}
        immutable(root/"context.json",c)
        return root/"context.json"

    def cache(self,context):
        c=read(context);root=Path(c["cache_root"])
        if not (root/"complete.json").exists():
            self.parallel([(self.worker,("cache",context,root/f"worker-{i}","A",42,i)) for i in (0,1)])
            entries=[];files={}
            for i in (0,1):
                p=root/f"shard-{i}.json";value=read(p)
                if value["context_sha256"]!=digest(context):raise ValueError("Wrong reference-cache context")
                entries.extend(value["entries"]);files[p.name]=digest(p)
            if len(entries)!=len(self.rows) or len({r["sample_id"] for r in entries})!=len(self.rows):
                raise ValueError("Reference cache needs exact, unique original coverage")
            immutable(root/"complete.json",{"files":files,"context_sha256":digest(context)})
        return c

    def job(self,context,arm,seed):
        c=read(context);root=self.root/f"jobs/{c['grouping']}/{arm}/seed{seed}/fold{c['fold']}"
        if not (root/"train.complete.json").exists():self.worker("train",context,root,arm,seed)
        if c["fold"]!="full" and not (root/"predict.complete.json").exists():
            self.worker("predict",context,root,arm,seed)
        if c["fold"]!="full":
            marker=read(root/"predict.complete.json")
            if marker["sha256"]!=digest(root/"predictions.jsonl") or marker["train_complete_sha256"]!=digest(root/"train.complete.json"):
                raise ValueError("Prediction marker identity changed")
        return root

    def values(self,grouping,fold,arm,seed):
        c=read(self.root/f"references/{grouping}/fold{fold}/context.json")
        from scripts.train_conflict_reward import cached_entries
        # cached_entries expects selected rows, not the complete full TRAIN list.
        part=self.manifest["folds"][grouping]["folds"][fold]
        c.update(train=[self.rows[i] for i in part["train"]],held=[self.rows[i] for i in part["held"]])
        entries=cached_entries(c)
        base=[predicted(entries[r["sample_id"]]["baseline"],.75 if entries[r["sample_id"]]["baseline"]["is_uncertain"]=="yes" else .25) for r in c["held"]]
        path=self.root/f"jobs/{grouping}/{arm}/seed{seed}/fold{fold}/predictions.jsonl"
        new=[json.loads(s) for s in path.read_text().splitlines() if s]
        return part["held"],base,new

    def report(self,arm,seed=42,grouping="content",folds=(0,1,2),stage="primary"):
        indices=[];base=[];new=[]
        for fold in folds:
            ix,a,b=self.values(grouping,fold,arm,seed);indices.extend(ix);base.extend(a);new.extend(b)
        if len(folds)==3:
            ordered=sorted(zip(indices,base,new));base=[v[1] for v in ordered];new=[v[2] for v in ordered]
            indices=[v[0] for v in ordered]
        rows=[self.rows[i] for i in indices]
        design=self.manifest["folds"][grouping] if len(folds)==3 else None
        result=compare(rows,base,new,design,stage=stage)
        result["paired_group_bootstrap"]=bootstrap(rows,base,new,[self.manifest["folds"][grouping]["groups"][i] for i in indices])
        result.update(arm=arm,seed=seed,grouping=grouping,folds_included=list(folds),
            development_result=True,reference_source="frozen_fold_SFT",public_score=None)
        immutable(self.root/f"reports/{stage}-{arm}-seed{seed}.json",result)
        return result

    def cohort(self):
        context=self.context("content",0)
        preflight=self.root/"preflight"
        if not (preflight/"preflight.complete.json").exists():
            self.state("preflight_32_real_train_examples")
            self.worker("preflight",context,preflight,"C")
        pre=read(preflight/"preflight.complete.json")
        if not pre.get("passed") or not pre.get("resume_identical") or pre["peak_reserved_gib"]>40:
            raise ValueError("Preflight did not satisfy the real-update/resume/memory contract")
        self.state("reference_cache_fold0");self.cache(context)
        self.state("pilot_ABC_fold0")
        self.parallel([(self.job,(context,arm,42)) for arm in ARMS])
        pilot={arm:self.report(arm,folds=(0,),stage="pilot") for arm in ARMS}
        if not any(r["passed"] for r in pilot.values()):
            return {"status":"complete","reason":"pilot_futility_gate_not_met","pilot":pilot,"candidate_created":False}
        immutable(self.root/"pilot-passed.json",{"pilot_gain":.003,"passing_arms":[a for a in ARMS if pilot[a]["passed"]]})
        self.pilot=False
        for fold in (1,2):
            c=self.context("content",fold);self.state(f"reference_cache_fold{fold}");self.cache(c)
            self.state(f"ABC_fold{fold}");self.parallel([(self.job,(c,arm,42)) for arm in ARMS])
        reports={arm:self.report(arm) for arm in ARMS};winner=choose(reports)
        immutable(self.root/"primary-report.json",{"arms":reports,"winner":winner})
        if winner is None:return {"status":"complete","reason":"primary_gate_not_met","candidate_created":False}
        self.state("RL_seed43_replication",winner=winner)
        self.parallel([(self.job,(self.context("content",fold),winner,43)) for fold in (0,1,2)])
        replication=self.report(winner,seed=43,stage="replication")
        if not replication["passed"]:return {"status":"complete","reason":"RL_seed43_gate_not_met","candidate_created":False}
        self.state("parent_fold0_reference_and_pressure",winner=winner)
        parent=self.context("parent",0);self.cache(parent);self.job(parent,winner,42)
        pressure=self.report(winner,grouping="parent",folds=(0,),stage="pressure")
        if not pressure["passed"]:return {"status":"complete","reason":"parent_pressure_gate_not_met","candidate_created":False}
        self.state("full_SFT_reference",winner=winner)
        full=self.context("content","full");self.cache(full)
        self.state("full_RL_seed42_seed43",winner=winner)
        jobs=self.parallel([(self.job,(full,winner,seed)) for seed in (42,43)])
        lock={"algorithm":VERSION,"plan_identity":self.plan["identity_hash"],"arm":winner,"context":str(full),
              "context_sha256":digest(full),"models":[{"path":str(j),"train_complete_sha256":digest(j/"train.complete.json")} for j in jobs],
              "all_gates_passed":True,"official_inputs_used_for_selection":False,"baseline_sha256":BASELINE_SHA}
        immutable(self.root/"final-model-lock.json",lock)
        self.state("locked_official_validation_and_test",winner=winner)
        self.official(full,jobs,winner)
        return {"status":"complete","reason":"all_gates_and_locked_delivery_completed","winner":winner,"candidate_created":True}

    def official(self,context,jobs,arm):
        for split in ("val","test"):
            cacheout=self.root/f"official/{split}/reference"
            if not (cacheout/"complete.json").exists():
                self.delivery_worker("cache",context,cacheout,arm,42,split)
            self.parallel([(self.delivery_worker,("predict",context,j,arm,seed,split)) for j,seed in zip(jobs,(42,43))])
        self.delivery_worker("export",context,self.root/"delivery",arm,42,"test")

    def delivery_worker(self,mode,context,output,arm,seed,split):
        with self.resource() as gpu:
            self.child([sys.executable,"-B","-m","scripts.conflict_reward_export",mode,"--context",context,
                "--output",output,"--gpu",gpu,"--arm",arm,"--seed",seed,"--split",split],output,gpu)

    def archive(self):
        paths=[]
        for name in ("plan.json","status.json","clock.json","FINAL_REPORT.json","primary-report.json","final-model-lock.json","recovery.json"):
            if (self.root/name).is_file():paths.append(self.root/name)
        for folder in ("reports","preflight","delivery","jobs","official"):
            for p in (self.root/folder).rglob("*"):
                if p.is_file() and p.suffix in (".json",".jsonl",".csv") and not any(x.startswith("checkpoint-") for x in p.relative_to(self.root).parts):
                    paths.append(p)
        with tarfile.open(self.root/"results.tar.gz","w:gz") as tar:
            for p in sorted(set(paths)):tar.add(p,arcname=p.relative_to(self.root))
        atomic(self.root/"results-archive.json",{"sha256":digest(self.root/"results.tar.gz"),"files":len(set(paths))})

    def run(self):
        with run_lock(self.root):
            if (self.root/"FINAL_REPORT.json").exists():return read(self.root/"FINAL_REPORT.json")
            try:
                if self.plan["identity"]["source_sha256"]!=sources():raise ValueError("Frozen source changed")
                verify_environment(self.project,{**self.plan["identity"]["base_environment"],"runtime":self.plan["identity"]["runtime"]})
                self.budget=Budget(self.root,seconds=72*3600)
                # Do not duplicate children left by a killed controller.
                for path in self.root.glob("**/process.json"):
                    c=read(path)
                    if process_token(c["pid"])==c["process_start_ticks"]:
                        raise RuntimeError("A retained child still exists; inspect it before resuming: "+str(path))
                result=self.cohort()
            except BaseException as exc:
                self.cancel.set()
                result={"status":"stopped" if isinstance(exc,(TimeoutError,KeyboardInterrupt,InterruptedError)) else "failed",
                        "reason":str(exc),"traceback":traceback.format_exc(),"candidate_created":False}
            result.update(algorithm=VERSION,submitted_to_kaggle=False,new_public_score=None)
            immutable(self.root/"FINAL_REPORT.json",result)
            self.state(result["status"],reason=result["reason"]);self.archive()
            return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("prepare","run"));p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--project-root",type=Path);p.add_argument("--previous-run",type=Path);p.add_argument("--baseline",type=Path)
    a=p.parse_args(argv)
    if a.command=="prepare":value=prepare(a.project_root,a.run_dir,a.previous_run,a.baseline)
    else:value=Pipeline(a.run_dir).run()
    print(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2))


if __name__=="__main__":main()
