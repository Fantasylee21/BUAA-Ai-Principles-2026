"""Read-only compact tmux progress display; training counts are not evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import subprocess
import time


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}


def duration(seconds):
    seconds=max(0,int(seconds));return f"{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}"


def snapshot(root):
    root=Path(root);s=read(root/"status.json");clock=read(root/"clock.json")
    lines=["CONFLICT REWARD RL | "+datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S Beijing"),
        "Run: "+str(root),"Phase: "+s.get("phase","waiting for controller")]
    if clock:lines.append("Elapsed "+duration(time.time()-clock["started_unix"])+" | remaining budget "+duration(clock["deadline_unix"]-time.time())+" (not ETA)")
    try:lines.append(subprocess.check_output(["nvidia-smi","--id=1,2","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader"],text=True,timeout=5).strip())
    except (OSError,subprocess.SubprocessError):pass
    active=s.get("active_jobs",{})
    for name,process in active.items():
        v=read(root/name/"status.json");n=v.get("current_steps",0);total=v.get("total_steps",0);fraction=n/total if total else 0
        bar="#"*int(32*fraction)+"."*(32-int(32*fraction))
        lines.extend(["",f"GPU {process['gpu']} | {name}",v.get("phase","initializing model"),f"[{bar}] {n}/{total} ({100*fraction:.1f}%)"])
        if "eta_seconds" in v:lines.append("This job ETA: "+duration(v["eta_seconds"]))
        if "loss" in v:lines.append("loss={loss:.4f} | KL={kl:.5f} | zero-adv={zero_advantage:.1%} | gradient={grad_norm:.3f}".format(**v))
        c=v.get("training_counts",{})
        if c:lines.append("TRAIN observations only: conflict {}/{} | ordinary FP {}/{} | natural VA {}/{}".format(c.get("train_conflict_tp",0),c.get("train_conflict_n",0),c.get("train_aligned_fp",0),c.get("train_aligned_n",0),c.get("train_natural_va_tp",0),c.get("train_natural_va_n",0)))
    if not active:lines.append("No active worker. "+s.get("reason","Preparing or aggregating results."))
    lines.append("Read-only display. Ctrl-C stops this panel only. No Kaggle submission.")
    return "\n".join(lines)


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True);p.add_argument("--once",action="store_true");a=p.parse_args()
    while True:
        print(("" if a.once else "\033[2J\033[H")+snapshot(a.run_dir),flush=True)
        if a.once:return
        time.sleep(4)


if __name__=="__main__":main()
