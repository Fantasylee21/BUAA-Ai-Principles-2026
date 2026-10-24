"""Group-isolated nested evaluation, OOF calibration and one final candidate."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .evidence_bottleneck import (MODES, decision_scores, disagreement, evidence_probabilities,
                                  fit_calibration, fit_head, view_examples)
from .evidence_store import FeatureBank, read, saved_fit, save_fit
from .experiment import fingerprint, write_json
from .finite_reports import comparison
from .modality_relation import paired_diagnostics, selection_score
from .train_modality_relation import baseline_records, decision_records, extra_report


def train_only(c, rows):
    known = {r["sample_id"]: r for r in c["train"]}
    if not rows or any(known.get(r["sample_id"]) != r for r in rows):
        raise ValueError("Only original training rows can reach a fitted objective")


def fit_evidence(run, c, bank, rows, job, emit):
    train_only(c, rows)
    if bank.split != "train":
        raise ValueError("Evidence fitting requires TRAIN features")
    heads = {}
    for mode in MODES:
        examples = view_examples(rows, mode)
        selected, labels = zip(*examples)
        signature = fingerprint([c["identity_hash"], mode,
            [(r["sample_id"], y) for r, y in examples]])
        path = Path(run)/"heads"/job/(mode+".json")
        head = saved_fit(path, c, signature)
        if head is None:
            hidden, native = bank.get(selected, mode)
            emit("fitting_emotion", job=job, mode=mode, covered=len(examples), iteration=0)
            def progress(**v):
                emit("fitting_emotion", job=job, mode=mode, covered=len(examples), **v)
            head = fit_head(hidden, native, labels, c["settings"], progress)
            head.update(mode=mode, identity_hash=c["identity_hash"], signature=signature,
                examples=[r["sample_id"] for r in selected], labels=list(labels))
            save_fit(path, c, signature, head)
        heads[mode] = head
        emit("emotion_fit_complete", job=job, mode=mode, loss=head["solver"]["loss"],
             data_loss=head["solver"]["data_loss"], covered=len(examples),
             iteration=head["solver"]["iterations"], reused=path.exists())
    return heads


def predict_evidence(bank, rows, heads=None):
    hidden, native = bank.get(rows, "vision")
    v = evidence_probabilities(hidden, native, heads["vision"] if heads else None)
    m = np.empty((len(rows), 3), dtype=np.float64)
    for protocol, mode in (("VT", "text"), ("VA", "audio")):
        indices = [i for i, r in enumerate(rows) if r["protocol"] == protocol]
        if indices:
            hidden, native = bank.get([rows[i] for i in indices], mode)
            m[indices] = evidence_probabilities(hidden, native, heads[mode] if heads else None)
    return {"vision": v, "second": m, "q": disagreement(v, m)}


def calibrate(run, c, rows, q, job, emit):
    train_only(c, rows)
    signature = fingerprint([c["identity_hash"], [(r["sample_id"], r["sample_type"], r["protocol"]) for r in rows],
                             np.asarray(q).tolist()])
    path = Path(run)/"calibration"/(job+".json")
    result = saved_fit(path, c, signature)
    if result is None:
        result = fit_calibration(q, rows, c["settings"], lambda **v: emit("fitting_calibration", job=job, **v))
        result.update(identity_hash=c["identity_hash"], signature=signature,
                      examples=[r["sample_id"] for r in rows])
        save_fit(path, c, signature, result)
    emit("calibration_complete", job=job, loss=result["solver"]["loss"],
         data_loss=result["solver"]["data_loss"], offsets=result["offsets"])
    return result


def view_diagnostics(rows, probabilities):
    result = {}
    for mode in MODES:
        ids = [i for i, r in enumerate(rows) if mode in r["view_targets"] and r["view_targets"][mode]["clear"]]
        # Visual diagnostics count one unique video, consistent with training.
        if mode == "vision":
            seen = set()
            ids = [i for i in ids if not (rows[i]["video"] in seen or seen.add(rows[i]["video"]))]
        p = probabilities["vision"] if mode == "vision" else probabilities["second"]
        from .reward import VALID_EMOTIONS
        correct = sum(VALID_EMOTIONS[int(p[i].argmax())] == rows[i]["view_targets"][mode]["label"] for i in ids)
        result[mode] = {"correct": correct, "total": len(ids)}
    return result


def development(run, c, bank, emit):
    rows = c["train"]
    design = c["designs"]["42"]
    path = Path(run)/"development.json"
    if path.exists():
        value = read(path)
        if value["identity_hash"] != c["identity_hash"] or value["fold_hash"] != design["hash"]:
            raise ValueError("Development identity changed")
        return value
    scores = np.full(len(rows), np.nan)
    p_vision = np.full((len(rows), 3), np.nan)
    p_second = np.full_like(p_vision, np.nan)
    summaries = []
    for partition in design["nested"]:
        fold = partition["fold"]
        training, cal, held = ([rows[i] for i in partition[k]] for k in ("train", "calibration", "held"))
        job = f"nested42-fold{fold}"
        heads = fit_evidence(run, c, bank, training, job, emit)
        cal_q = predict_evidence(bank, cal, heads)["q"]
        bias = calibrate(run, c, cal, cal_q, job, emit)
        evidence = predict_evidence(bank, held, heads)
        ix = partition["held"]
        scores[ix] = decision_scores(evidence["q"], held, bias)
        p_vision[ix], p_second[ix] = evidence["vision"], evidence["second"]
        summaries.append({"fold": fold, "train": len(training), "calibration": len(cal), "held": len(held),
            "task_component": selection_score(held, scores[ix]), "offsets": bias["offsets"]})
        emit("outer_fold_complete", fold=fold, task_component=summaries[-1]["task_component"])
    if not np.isfinite(scores).all():
        raise RuntimeError("Missing outer-fold predictions")
    records = decision_records(c, scores.tolist(), "train")
    value = {"identity_hash": c["identity_hash"], "fold_hash": design["hash"], "partitions": summaries,
        "task_component": selection_score(rows, scores), "metrics": extra_report(records),
        "view_accuracy": view_diagnostics(rows, {"vision": p_vision, "second": p_second}),
        "records": records,
        "note": "Only the uncertainty branch is group-held-out. Fixed v4 emotion predictions saw training data; no development gating against resubstitution v4."}
    write_json(path, value)
    emit("development_complete", task_component=value["task_component"], metrics=value["metrics"])
    return value


def final_candidate(run, c, bank, seed, emit):
    if seed not in (42, 43):
        raise ValueError("Only approved partition seeds are allowed")
    target = Path(run)/"results"/f"evidence-seed{seed}.json"
    if target.exists():
        value = read(target)
        if value["identity_hash"] != c["identity_hash"]:
            raise ValueError("Candidate identity changed")
        return value
    rows = c["train"]
    design = c["designs"][str(seed)]
    q_oof = np.full(len(rows), np.nan)
    for fold in range(3):
        train_ix = [i for i, f in enumerate(design["outer"]) if f != fold]
        held_ix = [i for i, f in enumerate(design["outer"]) if f == fold]
        heads = fit_evidence(run, c, bank, [rows[i] for i in train_ix], f"oof{seed}-fold{fold}", emit)
        evidence = predict_evidence(bank, [rows[i] for i in held_ix], heads)
        q_oof[held_ix] = evidence["q"]
        emit("calibration_oof_fold_complete", seed=seed, fold=fold, covered=int(np.isfinite(q_oof).sum()))
    if not np.isfinite(q_oof).all():
        raise RuntimeError("Calibration needs exactly one held-out score per training row")
    write_json(Path(run)/f"calibration-oof-{seed}.json", {"identity_hash": c["identity_hash"],
        "fold_hash": design["hash"], "ids": [r["sample_id"] for r in rows], "q": q_oof.tolist()})
    bias = calibrate(run, c, rows, q_oof, f"final-{seed}", emit)
    heads = fit_evidence(run, c, bank, rows, f"final-{seed}", emit)
    val_bank = FeatureBank(run, c, "val")
    val = c["val"]
    evidence = predict_evidence(val_bank, val, heads)
    scores = decision_scores(evidence["q"], val, bias)
    records = decision_records(c, scores.tolist())
    metrics = extra_report(records)
    metrics.update(invalid_outputs=None, inconsistent_questions=None, emotion_changes=0, output_checks_pending=True)
    native = predict_evidence(val_bank, val)
    ablations = {}
    for name, q in (("native_uncalibrated", native["q"]), ("trained_uncalibrated", evidence["q"])):
        ablations[name] = extra_report(decision_records(c, decision_scores(q, val).tolist()))
    value = {"identity_hash": c["identity_hash"], "method": "evidence_bottleneck", "seed": seed,
        "heads": heads, "calibration": bias, "fold_hash": design["hash"],
        "metrics": metrics, "comparison": comparison(baseline_records(c), records),
        "paired_diagnostics": paired_diagnostics(val, scores), "ablations": ablations,
        "view_accuracy": view_diagnostics(val, evidence), "val_probabilities": scores.tolist(),
        "vision_probabilities": evidence["vision"].tolist(), "second_probabilities": evidence["second"].tolist(),
        "raw_disagreement": evidence["q"].tolist()}
    write_json(target.with_suffix(".decisions.json"), records)
    write_json(target, value)
    emit("candidate_complete", seed=seed, metrics=metrics, comparison=value["comparison"])
    return value
