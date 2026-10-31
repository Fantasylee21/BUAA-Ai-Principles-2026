"""Fixed, testable contracts for the independent E/U emotion expert batch."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time

VERSION = "emotion_expert_costrl_v1"
EMOTIONS = ("negative", "neutral", "positive")
STATES = EMOTIONS + ("unclear",)
FIELDS = ("sample_id", "pred_emotion", "is_uncertain", "clarifying_question")
NATURAL = ("ch_sims", "ch_sims_v2")
QUESTIONS = {"VT": "你能补充说明，画面中的表现和这段话分别给你怎样的感受吗？",
             "VA": "你能补充说明，画面中的表现和声音分别给你怎样的感受吗？"}
SETTINGS = {"seed": 42, "gpu": 2, "batch_size": 16, "epochs": 1,
    "rank": 16, "alpha": 32, "dropout": 0., "sft_learning_rate": 1e-5,
    "rl_learning_rate": 5e-6, "sft_weight_decay": .01, "rl_weight_decay": 0.,
    "warmup_fraction": .1, "max_grad_norm": 1., "beta": .02,
    "video_fps": 2., "video_min_frames": 4, "video_max_frames": 64,
    "video_pixels": 50176, "face_count": 4, "face_size": 448, "face_padding": .25,
    "audio_sampling_rate": 16000, "max_prompt_tokens": 12288,
    "memory_gib": 40, "allowed_gpus": [2], "expected_split_sizes": [2013, 670, 671]}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def fingerprint(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def atomic(path, value):
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_bytes(encoded(value))
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def input_only(row):
    from pathlib import PurePosixPath
    video = row.get("video", "")
    if not isinstance(video, str) or not video or PurePosixPath(video).is_absolute() or ".." in PurePosixPath(video).parts:
        raise ValueError("A safe relative media path is required")
    protocol = row.get("protocol")
    if protocol not in ("VT", "VA"):
        tag = row.get("sample_id", "").split(":")[1]
        protocol = {"VT":"VT", "VA":"VA", "accept_a_svt":"VT", "accept_c_svt":"VT", "accept_a_va":"VA"}.get(tag)
    if protocol not in ("VT", "VA"):
        raise ValueError("Unknown input protocol")
    return {"video": video, "protocol": protocol,
            "text_content": str(row.get("text_content", row.get("text", ""))) if protocol == "VT" else ""}


def targets(row):
    views = row["views"]
    for name in ("M1", "M2", "M12"):
        if views[name]["label"] not in EMOTIONS or type(views[name]["is_clear"]) is not bool:
            raise ValueError("Original modality labels and clarity are required")
    if row["sample_type"] not in ("Aligned", "Conflict"):
        raise ValueError("Original relation label is required")
    return {"E": EMOTIONS.index(views["M12"]["label"]),
            "U": int(row["sample_type"] == "Conflict"),
            "V": STATES.index(views["M1"]["label"]) if views["M1"]["is_clear"] else 3,
            "M": STATES.index(views["M2"]["label"]) if views["M2"]["is_clear"] else 3}


def counts(rows):
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Nonempty unique original TRAIN rows required")
    values = Counter(r["sample_type"] for r in rows)
    if set(values) != {"Conflict", "Aligned"}:
        raise ValueError("Both original classes are required")
    return {"N":len(rows), "C":values["Conflict"], "A":values["Aligned"]}


def weights(rows):
    frequency = Counter(r["video"] for r in rows)
    return {r["sample_id"]: len(rows)/(len(frequency)*frequency[r["video"]]) for r in rows}


def class_weight(gold, population, lam=1.):
    if lam not in (1., 2.) or population["N"] != population["C"] + population["A"]:
        raise ValueError("Fixed reward or population contract changed")
    return (.3*lam*population["N"]/population["C"] if gold else .2*population["N"]/population["A"])


def objective(logits, gold, population, *, phase="S", reference=None, beta=.02):
    """Main U loss; auxiliary CE is independently backpropagated by the worker.

    Per-record population weights average to .3 E_C + .2 E_A under uniform
    traversal, including batches containing no Conflict records.
    """
    import torch
    import torch.nn.functional as F
    if logits.shape != (2,) or gold not in (0, 1) or phase not in ("S", "R1", "R2"):
        raise ValueError("Expected native two-action logits and an original binary target")
    lp = logits.double().log_softmax(-1)
    p = lp.exp()
    ce = -lp[gold] * class_weight(gold, population)
    reward = p[gold] * class_weight(gold, population, 2. if phase == "R2" else 1.)
    kl = lp.sum() * 0
    loss = ce
    if phase != "S":
        q = torch.as_tensor(reference, device=lp.device, dtype=torch.float64)
        if q.shape != (2,) or (q < 0).any() or not torch.isfinite(q).all() or abs(float(q.sum())-1) > 1e-6:
            raise ValueError("RL requires a frozen S policy distribution")
        kl = (p * (lp-q.clamp_min(1e-12).log())).sum()
        loss = loss-reward+beta*kl
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite task objective")
    return loss, {"ce":float(ce.detach()), "reward":float(reward.detach()),
                  "kl":float(kl.detach()), "prob_yes":float(p[1].detach()), "prediction":int(p.argmax())}


def batches(rows, seed=42):
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return [[rows[i] for i in indices[j:j+16]] for j in range(0,len(rows),16)]


def preflight_rows(rows, data_root):
    predicates=[lambda r:r["protocol"]=="VT",lambda r:r["protocol"]=="VA",
        lambda r:r["sample_type"]=="Conflict" and r.get("source_dataset") in NATURAL,
        lambda r:r["sample_type"]=="Conflict" and r.get("source_dataset") not in NATURAL,
        lambda r:not r["views"]["M1"]["is_clear"],lambda r:not r["views"]["M2"]["is_clear"],
        lambda r:r["protocol"]=="VA" and r["sample_type"]=="Conflict" and r.get("source_dataset") in NATURAL,
        lambda r:r["views"]["M12"]["label"]=="neutral"]
    selected=[]
    for predicate in predicates:
        found=next((r for r in rows if r not in selected and predicate(r)),None)
        if found is None:raise ValueError("TRAIN lacks a required preflight stratum")
        selected.append(found)
    for row in sorted(rows,key=lambda r:(Path(data_root)/r["video"]).stat().st_size,reverse=True)[:2]:
        if row not in selected:selected.append(row)
    selected.extend(r for r in rows if r not in selected)
    return selected[:32]


def instruction(row, task):
    safe = input_only(row)
    if task not in ("E", "U", "V", "M"):
        raise ValueError("Unknown prediction task")
    text = safe["text_content"]
    if task in ("E", "U"):
        evidence = ("输入包含静音视频和给定文本。给定文本："+json.dumps(text,ensure_ascii=False)
                    if safe["protocol"] == "VT" else "输入包含同一片段的视频和音频。没有提供转写文本。")
    elif task == "V":
        evidence = "仅根据静音画面判断可见的情感，不使用声音或给定文本。人脸裁剪仅补充画面细节。"
    else:
        evidence = ("仅根据这段文字判断情感："+json.dumps(text,ensure_ascii=False)
                    if safe["protocol"] == "VT" else "仅根据音频判断情感，不使用画面或给定转写。")
    if task == "U":
        query = ("判断两路情感证据是否冲突或无法判读。两路极性不同，或任一路证据无法判读，选B；"
                 "两路均可清晰判读且极性相同，选A。真实的平静中性不等于无法判读，模型自身犹豫也不等于冲突。"
                 "A=no（两路一致且清晰），B=yes（冲突或证据无法判读）。")
    else:
        query = ("判断综合情感极性。" if task == "E" else "判断当前这一路证据的情感状态。")
        query += "A=negative（负面），B=neutral（真实中性），C=positive（正面）"
        if task != "E":
            query += "，D=unclear（当前证据无法判读；不是平静中性）"
        query += "。"
    return evidence+"\n"+query+"仅输出一个选项字母，不输出解释。"


def f1(true, pred, labels):
    values=[]
    for label in labels:
        tp=sum(a==label and b==label for a,b in zip(true,pred))
        fp=sum(a!=label and b==label for a,b in zip(true,pred))
        fn=sum(a==label and b!=label for a,b in zip(true,pred))
        values.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return sum(values)/len(values)


def evaluate(rows, emotion, uncertainty):
    expected={r["sample_id"] for r in rows}
    if set(emotion)!=expected or set(uncertainty)!=expected:
        raise ValueError("Prediction coverage differs from the original split")
    true=[EMOTIONS[targets(r)["E"]] for r in rows]
    pred=[emotion[r["sample_id"]]["pred_emotion"] for r in rows]
    groups=defaultdict(lambda:{"n":0,"correct":0})
    evidence={"n":0,"vision_correct":0,"second_correct":0,"both_correct":0,
              "both_correct_missed_conflicts":0,"both_correct_false_positives":0,
              "false_positive_vision_wrong_second_correct":0}
    tp=fp=nc=na=0
    for row in rows:
        gold=targets(row)["U"]
        p=float(uncertainty[row["sample_id"]]["uncertain_prob"])
        if not math.isfinite(p) or not 0<=p<=1:
            raise ValueError("Invalid policy probability")
        yes=p>=.5
        nc+=gold;na+=1-gold;tp+=int(gold and yes);fp+=int(not gold and yes)
        category="aligned" if not gold else "natural" if row.get("source_dataset") in NATURAL else "synthetic"
        for key in (category,category+"_"+input_only(row)["protocol"]):
            groups[key]["n"]+=1;groups[key]["correct"]+=int(yes==bool(gold))
        item=uncertainty[row["sample_id"]]
        if "vision_probs" in item and "second_probs" in item:
            v=max(range(4),key=lambda i:item["vision_probs"][i])==targets(row)["V"]
            m=max(range(4),key=lambda i:item["second_probs"][i])==targets(row)["M"]
            evidence["n"]+=1;evidence["vision_correct"]+=int(v);evidence["second_correct"]+=int(m)
            evidence["both_correct"]+=int(v and m)
            evidence["both_correct_missed_conflicts"]+=int(v and m and gold and not yes)
            evidence["both_correct_false_positives"]+=int(v and m and not gold and yes)
            evidence["false_positive_vision_wrong_second_correct"]+=int(not gold and yes and not v and m)
    pn=f1(true,pred,("negative","positive"));three=f1(true,pred,EMOTIONS)
    component=.3*tp/nc+.2*(1-fp/na)
    protocols={}
    for protocol in ("VT","VA"):
        ix=[i for i,r in enumerate(rows) if input_only(r)["protocol"]==protocol]
        ct=sum(targets(rows[i])["U"] for i in ix);an=len(ix)-ct
        pt=sum(targets(rows[i])["U"] and uncertainty[rows[i]["sample_id"]]["uncertain_prob"]>=.5 for i in ix)
        pf=sum(not targets(rows[i])["U"] and uncertainty[rows[i]["sample_id"]]["uncertain_prob"]>=.5 for i in ix)
        protocols[protocol]={"n":len(ix),"tp":pt,"fp":pf,"n_conflict":ct,"n_aligned":an,
            "f1_pn":f1([true[i] for i in ix],[pred[i] for i in ix],("negative","positive")) if ix else None}
    return {"n":len(rows),"f1_pn":pn,"f1_fixed3":three,"selection_score":.5*pn+component,
            "score_fixed3":.5*three+component,"uncertainty_component":component,
            "n_conflict":nc,"n_aligned":na,"tp":tp,"fp":fp,"groups":dict(groups),
            "neutral_support":true.count("neutral"),
            "neutral_correct":sum(a==b=="neutral" for a,b in zip(true,pred)),"protocols":protocols,"evidence":evidence,
            "not_kaggle_score":True}


def differences(rows,reference,candidate):
    re,ru=reference;ce,cu=candidate
    out={"emotion_new_correct":0,"emotion_new_wrong":0,"new_correct_conflicts":0,
         "lost_correct_conflicts":0,"fixed_ordinary_false_positives":0,"new_ordinary_false_positives":0}
    for row in rows:
        sid=row["sample_id"];gold=targets(row)
        olde=re[sid]["pred_emotion"]==EMOTIONS[gold["E"]];newe=ce[sid]["pred_emotion"]==EMOTIONS[gold["E"]]
        old=ru[sid]["uncertain_prob"]>=.5;new=cu[sid]["uncertain_prob"]>=.5
        out["emotion_new_correct"]+=int(newe and not olde);out["emotion_new_wrong"]+=int(olde and not newe)
        if gold["U"]:
            out["new_correct_conflicts"]+=int(new and not old);out["lost_correct_conflicts"]+=int(old and not new)
        else:
            out["fixed_ordinary_false_positives"]+=int(old and not new);out["new_ordinary_false_positives"]+=int(new and not old)
    return out


def csv_predictions(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    return ({r["sample_id"]:{"pred_emotion":r["pred_emotion"]} for r in rows},
            {r["sample_id"]:{"uncertain_prob":float(r["is_uncertain"].lower()=="yes")} for r in rows})


def choose(dev, emotion_names, uncertainty_names):
    """Component-wise DEV selection; held labels are never accepted here."""
    e="E0"
    for name in emotion_names:
        if dev[name]["f1_pn"]>dev[e]["f1_pn"]+.0001:
            e=name
    base="U0"
    if "S" in uncertainty_names and dev["S"]["uncertainty_component"]>dev[base]["uncertainty_component"]+.0001:
        base="S"
    u=base
    for name in ("R1","R2"):
        if name not in uncertainty_names:
            continue
        va=lambda m:m["groups"].get("natural_VA",{}).get("correct",0)
        if (dev[name]["uncertainty_component"]>=dev[base]["uncertainty_component"]+.003-1e-12
                and va(dev[name])>=va(dev[base])
                and (u==base or dev[name]["uncertainty_component"]>dev[u]["uncertainty_component"]+.0001)):
            u=name
    return {"emotion":e,"uncertainty":u,"non_rl":base,"selection_split":"dev"}


def gate(candidate, baseline):
    reasons=[]
    if candidate["selection_score"]<baseline["selection_score"]+.01-1e-12:reasons.append("gain_below_0.01")
    if candidate["f1_pn"]<baseline["f1_pn"]-.005-1e-12:reasons.append("emotion_regression")
    for key in ("natural","natural_VA"):
        if candidate["groups"].get(key,{}).get("correct",0)<baseline["groups"].get(key,{}).get("correct",0):
            reasons.append(key+"_regression")
    return {"passed":not reasons,"reasons":reasons}


def admission(now, deadline, seconds):
    """Single GPU 2 only: complete groups, 30% margin plus export reserve."""
    for name in ("full","supervised","frozen"):
        if now+1.3*seconds[name]+900<=deadline:
            return name
    return "none"


def export(rows, emotion, uncertainty, path, baseline=None):
    if len(rows)!=734 or len({r["sample_id"] for r in rows})!=734:
        raise ValueError("Exactly 734 ordered unique test rows are required")
    ids={r["sample_id"] for r in rows}
    if set(emotion)!=ids or set(uncertainty)!=ids:raise ValueError("Test coverage mismatch")
    if baseline is not None and set(baseline)!=ids:raise ValueError("Best CSV comparison coverage mismatch")
    changes=[];temporary=Path(path).with_suffix(".csv.partial")
    with temporary.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
        for row in rows:
            sid=row["sample_id"];e=emotion[sid]["pred_emotion"];p=uncertainty[sid]["uncertain_prob"]
            if e not in EMOTIONS or not math.isfinite(p) or not 0<=p<=1:raise ValueError("Invalid final prediction")
            yes=p>=.5
            value={"sample_id":sid,"pred_emotion":e,"is_uncertain":"yes" if yes else "no",
                   "clarifying_question":QUESTIONS[input_only(row)["protocol"]] if yes else "none"}
            writer.writerow(value)
            if baseline and any(value[k]!=baseline[sid][k] for k in FIELDS[1:]):
                changes.append({"sample_id":sid,"before":baseline[sid],"after":value})
    with temporary.open(encoding="utf-8",newline="") as f:check=list(csv.DictReader(f))
    if [r["sample_id"] for r in check]!=[r["sample_id"] for r in rows]:raise ValueError("Export order changed")
    temporary.replace(path)
    return {"rows":734,"sha256":digest(path),"emotion_changes":sum(c["before"]["pred_emotion"]!=c["after"]["pred_emotion"] for c in changes),
            "changed_rows":len(changes),"changes":changes,"auto_submitted":False}
