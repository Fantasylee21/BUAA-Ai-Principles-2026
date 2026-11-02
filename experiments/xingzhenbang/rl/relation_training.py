"""FP32 head fitting with full-coverage epochs and exact epoch-boundary recovery."""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import time

from .experiment import fingerprint, write_json
from .modality_relation import (allowed_states, build_model, conflict_probability, epoch_order,
                               row_losses, selection_score, training_weights, paired_diagnostics)
from .relation_store import atomic_torch, load_head, load_matrix, read, save_head
from .state import capture_rng, restore_rng


def seed_all(torch,seed):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensors(rows,torch,device):
    return (torch.tensor([r["protocol"]=="VA" for r in rows],dtype=torch.long,device=device),
            torch.tensor([allowed_states(r) for r in rows],dtype=torch.bool,device=device),
            torch.tensor([r["sample_type"]=="Conflict" for r in rows],dtype=torch.long,device=device))


def optimizer_for(model,settings,steps_per_epoch,torch):
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings["learning_rate"],
                               weight_decay=settings["weight_decay"],foreach=False,fused=False)
    warm=settings["warmup_epochs"]*steps_per_epoch
    horizon=settings["epochs"]*steps_per_epoch
    def scale(step):
        if step<warm:
            return (step+1)/warm
        return .5*(1+math.cos(math.pi*min(1.,(step-warm)/max(1,horizon-warm))))
    return optimizer,torch.optim.lr_scheduler.LambdaLR(optimizer,scale)


def update(model,optimizer,scheduler,x,protocol,masks,targets,weights,method,torch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits=model(x,protocol)
    per_row=row_losses(logits,method,masks,targets)
    loss=(per_row*weights).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite head loss; update not applied")
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    lr=optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    q=conflict_probability(logits.detach(),method)
    return {"loss":float(loss.detach()),"unweighted_loss":float(per_row.detach().mean()),
        "expected_reward":float((weights*torch.where(targets.bool(),q,1-q)).mean()),
        "correct":int(((q>.5)==targets.bool()).sum()),"learning_rate":lr,"grad_norm":float(norm)}


def predict(model,x,protocol,method,torch,batch_size=64):
    model.eval()
    qs,states=[],[]
    with torch.no_grad():
        for i in range(0,len(x),batch_size):
            logits=model(x[i:i+batch_size],protocol[i:i+batch_size])
            qs.extend(conflict_probability(logits,method).cpu().tolist())
            states.extend(logits.softmax(-1).cpu().tolist())
    return qs,states


def fit(run,c,method,fold,seed,epochs,torch,emit):
    """CV runs all 40 epochs; final fits retain the same 40-epoch LR horizon."""
    s=c["settings"]
    if method not in ("binary_control","joint_relation") or fold not in (-1,0,1,2) or not 1<=epochs<=40:
        raise ValueError("Invalid bounded fitting job")
    job=f"{method}-fold{fold}-seed{seed}"
    root=Path(run)/"heads"/job
    root.mkdir(parents=True,exist_ok=True)
    done=root/"done.json"
    if done.exists():
        value=read(done)
        if value["identity_hash"]!=c["identity_hash"] or value["epochs"]!=epochs:
            raise ValueError("Completed job identity changed")
        load_head(value["checkpoint"],c,torch)
        return value
    ids=[i for i,k in enumerate(c["folds"]) if fold==-1 or k!=fold]
    held=[i for i,k in enumerate(c["folds"]) if k==fold] if fold>=0 else []
    rows=[c["train"][i] for i in ids]
    train_ids_hash=fingerprint([r["sample_id"] for r in rows])
    all_x=load_matrix(run,c,"train",torch).to("cuda:0")
    x=all_x[ids]
    protocol,masks,targets=tensors(rows,torch,"cuda:0")
    weights=torch.tensor(training_weights(rows),device="cuda:0",dtype=torch.float32)
    if held:
        held_rows=[c["train"][i] for i in held]
        held_x=all_x[held]
        held_protocol=tensors(held_rows,torch,"cuda:0")[0]
    seed_all(torch,seed)
    model=build_model(method,s).to("cuda:0").float()
    # Identical shuffle/dropout RNG across the two output-head geometries.
    seed_all(torch,seed)
    per_epoch=math.ceil(len(rows)/s["batch_size"])
    optimizer,scheduler=optimizer_for(model,s,per_epoch,torch)
    epoch0,history,step=0,[],0
    latest=root/"latest.json"
    checkpoint=None
    if latest.exists():
        checkpoint=Path(read(latest)["checkpoint"])
        payload=load_head(checkpoint,c,torch)
        if (payload["method"],payload["fold"],payload["seed"],payload["train_ids_hash"],payload["target_epochs"]) != (method,fold,seed,train_ids_hash,epochs):
            raise ValueError("Refusing mismatched head optimizer/data position")
        model.load_state_dict(payload["model"],strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        restore_rng(torch,payload["rng"])
        epoch0,history,step=payload["epoch"],payload["history"],payload["step"]
    started=time.monotonic()
    last={}
    for epoch in range(epoch0,epochs):
        order=epoch_order(len(rows),seed,epoch)
        seen=[]
        total_loss=total_reward=0.
        for offset in range(0,len(order),s["batch_size"]):
            ix=order[offset:offset+s["batch_size"]]
            last=update(model,optimizer,scheduler,x[ix],protocol[ix],masks[ix],targets[ix],weights[ix],method,torch)
            step+=1
            seen.extend(ix)
            total_loss+=last["loss"]*len(ix)
            total_reward+=last["expected_reward"]*len(ix)
            value={"method":method,"fold":fold,"seed":seed,"epoch":epoch+1,"epochs":epochs,
                "step":step,"total_steps":epochs*per_epoch,"epoch_seen":len(seen),"epoch_total":len(rows),
                "sample_ids":[rows[i]["sample_id"] for i in ix],**last}
            with (root/"updates.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(value,ensure_ascii=False)+"\n")
            emit("training",**{k:v for k,v in value.items() if k!="sample_ids"})
        if sorted(seen)!=list(range(len(rows))):
            raise RuntimeError("Training epoch did not cover each row exactly once")
        record={"epoch":epoch+1,"loss":total_loss/len(rows),"expected_reward":total_reward/len(rows)}
        if held:
            q,_=predict(model,held_x,held_protocol,method,torch)
            record.update(probabilities=q,selection_score=selection_score(held_rows,q))
        history.append(record)
        checkpoint=root/f"epoch-{epoch+1:03d}.pt"
        payload={"method":method,"fold":fold,"seed":seed,"epoch":epoch+1,"step":step,
            "target_epochs":epochs,"train_ids_hash":train_ids_hash,"model":model.state_dict(),
            "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
            "rng":capture_rng(torch),"history":history,"data_position":{"epoch":epoch+1,"offset":0}}
        save_head(checkpoint,payload,c,torch)
        write_json(latest,{"checkpoint":str(checkpoint)})
        emit("epoch_complete",**{**last,**{k:v for k,v in record.items() if k!="probabilities"},
             "method":method,"fold":fold,"seed":seed,"epochs":epochs,
             "epoch_total":len(rows),"epoch_seen":len(rows)})
    value={"identity_hash":c["identity_hash"],"method":method,"fold":fold,"seed":seed,
        "epochs":epochs,"checkpoint":str(checkpoint),"train_ids_hash":train_ids_hash,
        "held_indices":held,"history":history,"seconds":time.monotonic()-started,
        "parameters":sum(p.numel() for p in model.parameters())}
    if fold==-1:
        val=c["val"]
        val_x=load_matrix(run,c,"val",torch).to("cuda:0")
        p=tensors(val,torch,"cuda:0")[0]
        q,states=predict(model,val_x,p,method,torch)
        value.update(val_probabilities=q,val_states=states,paired_diagnostics=paired_diagnostics(val,q))
    write_json(done,value)
    emit("complete",method=method,fold=fold,seed=seed,epoch=epochs,epochs=epochs,
         step=step,total_steps=epochs*per_epoch,epoch_seen=len(rows),epoch_total=len(rows),**last)
    return value
