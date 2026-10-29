"""Read-only tmux progress panel. Nothing here starts or submits a job."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}


def bar(value,width=32):
    value=max(0.,min(1.,value));n=round(value*width)
    return "["+"#"*n+"-"*(width-n)+f"] {value:6.1%}"


def duration(value):
    value=max(0,int(value or 0));return f"{value//3600:02d}:{value//60%60:02d}:{value%60:02d}"


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--once",action="store_true")
    a=p.parse_args()
    while True:
        b=read(a.run/"bootstrap.status.json");s=read(a.run/"pipeline.status.json")
        job=read(s.get("job_status","/nonexistent"));ad=read(a.run/"admission.json")
        lines=["BUAA emotion expert + cost RL | GPU 2 ONLY | no Kaggle auto-submit",
               "="*78,"Pipeline: "+s.get("phase","bootstrap")+"   Bootstrap: "+b.get("phase","pending")]
        expected=b.get("expected_bytes",0);downloaded=b.get("downloaded_bytes",0)
        if expected and b.get("phase")=="download":
            lines += ["Download "+bar(downloaded/expected),f"{downloaded/1e9:.2f} / {expected/1e9:.2f} GB | "+b.get("repository","")]
        if job:
            current=job.get("current_steps",0);total=job.get("total_steps",0)
            lines += [f"Job: {s.get('job','')} | {job.get('phase','')} | epoch {job.get('epoch','-')}",
                      bar(current/total if total else 0),f"{current}/{total} | ETA {duration(job.get('eta_seconds'))}",
                      f"VRAM allocated {job.get('allocated_gib',0):.2f} GiB | peak {job.get('peak_gib',0):.2f} GiB | grad {job.get('grad_norm',0):.5f}"]
            m=job.get("metrics",{})
            lines += ["TRAIN OBSERVATION (not held evaluation): "+json.dumps(m,ensure_ascii=False),
                      "Rates: "+json.dumps(job.get("observations",{}),ensure_ascii=False),
                      "Updated "+duration(time.time()-job.get("updated_unix",time.time()))+" ago"]
        if ad:lines += ["Admitted cohort: "+ad.get("mode",""),"Estimates: "+str({k:duration(v) for k,v in ad.get("estimates_seconds",{}).items() if isinstance(v,(int,float))})]
        if s.get("error"):lines.append("ERROR: "+s["error"])
        if b.get("error"):lines.append("BOOTSTRAP ERROR: "+b["error"])
        if s.get("reason"):lines.append("Result: "+s["reason"])
        try:
            gpu=subprocess.check_output(["nvidia-smi","--id=2","--query-gpu=index,memory.used,memory.total,utilization.gpu","--format=csv,noheader"],text=True).strip()
            lines += ["GPU2 actual: "+gpu]
        except (OSError,subprocess.CalledProcessError):pass
        print(("" if a.once else "\033[2J\033[H")+"\n".join(lines),flush=True)
        if a.once:return
        time.sleep(2)


if __name__=="__main__":main()
