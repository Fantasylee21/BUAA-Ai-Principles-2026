"""Historical fixed-three-class score, with separately named union diagnostics.

The official missing-class policy is not independently verified. Do not
compare fixed3 and union scores as if they measured an RL improvement.
"""
from __future__ import annotations

try:
    from .reward import VALID_EMOTIONS, parse_completion, question_is_consistent
except ImportError:
    from reward import VALID_EMOTIONS, parse_completion, question_is_consistent


def macro_f1(y_true, y_pred, labels=VALID_EMOTIONS):
    if len(y_true) != len(y_pred):
        raise ValueError("Gold and prediction lengths differ")
    if labels is None:
        labels = sorted(set(y_true) | {p for p in y_pred if p is not None})
    if not labels:
        return 0.0
    total = 0.0
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        denominator = sum(t == label for t in y_true) + sum(p == label for p in y_pred)
        total += 2 * tp / denominator if denominator else 0.0
    return total / len(labels)


def compute_competition_score(records):
    true = [r["gt_label"] for r in records]
    if any(label not in VALID_EMOTIONS for label in true):
        raise ValueError("Evaluation requires valid gold emotion labels")
    predicted = [r.get("pred_emotion") for r in records]
    conflict = [r for r in records if r["sample_type"] == "Conflict"]
    aligned = [r for r in records if r["sample_type"] == "Aligned"]
    if len(conflict) + len(aligned) != len(records):
        raise ValueError("Unknown gold sample type")
    f1 = macro_f1(true, predicted)
    union = macro_f1(true, predicted, labels=None)
    rc = sum(r.get("is_uncertain") == "yes" for r in conflict) / len(conflict) if conflict else None
    ac = sum(r.get("is_uncertain") == "no" for r in aligned) / len(aligned) if aligned else None
    complete = rc is not None and ac is not None
    return {"metric_version": "fixed3_v1", "macro_f1": f1,
            "macro_f1_union_diagnostic": union, "conflict_accuracy": rc,
            "aligned_calibration": ac, "score": .5*f1 + .3*rc + .2*ac if complete else None,
            "score_union_diagnostic": .5*union + .3*rc + .2*ac if complete else None,
            "n_records": len(records), "n_conflict": len(conflict), "n_aligned": len(aligned),
            "invalid_outputs": sum(not r.get("valid_json", r.get("pred_emotion") in VALID_EMOTIONS) for r in records),
            "inconsistent_questions": sum(r.get("question_consistent") is False for r in records),
            "label_counts": {label: true.count(label) for label in VALID_EMOTIONS}}


def parse_prediction(completion_text):
    parsed = parse_completion(completion_text)
    return {"pred_emotion": parsed["pred_emotion"] if parsed else None,
            "is_uncertain": parsed["is_uncertain"] if parsed else None}


def prediction_record(row, completion):
    parsed = parse_completion(completion)
    return {"sample_id": row["sample_id"], "gt_label": row.get("gt_label"),
            "sample_type": row.get("sample_type"), "protocol": row["protocol"],
            "source_dataset": row.get("source_dataset", "unknown"), "completion": completion,
            "valid_json": parsed is not None,
            "question_consistent": question_is_consistent(parsed) if parsed else False,
            "pred_emotion": parsed["pred_emotion"] if parsed else None,
            "is_uncertain": parsed["is_uncertain"] if parsed else None,
            "clarifying_question": parsed["clarifying_question"] if parsed else None}


def evaluation_report(records):
    result = compute_competition_score(records)
    for field in ("protocol", "source_dataset"):
        result["by_" + field] = {
            name: compute_competition_score([r for r in records if r[field] == name])
            for name in sorted({r[field] for r in records})}
    natural = [r for r in records if r["source_dataset"] in ("ch_sims", "ch_sims_v2")]
    result["natural_conflict_recall"] = compute_competition_score(natural)["conflict_accuracy"] if natural else None
    return result


def promotion_decision(baseline, candidate):
    reasons = []
    if candidate.get("score") is None or baseline.get("score") is None:
        return {"promote": False, "reasons": ["Both types are required for model selection"]}
    if candidate["score"] - baseline["score"] < .005 - 1e-12:
        reasons.append("score gain below 0.005")
    if candidate["macro_f1"] < baseline["macro_f1"] - .005 - 1e-12:
        reasons.append("Macro-F1 regression exceeds 0.005")
    if candidate["aligned_calibration"] < baseline["aligned_calibration"] - .01 - 1e-12:
        reasons.append("Aligned regression exceeds 0.01")
    natural = baseline.get("natural_conflict_recall")
    if natural is not None and (candidate.get("natural_conflict_recall") is None
                                or candidate["natural_conflict_recall"] < natural):
        reasons.append("natural-conflict recall regressed")
    if candidate.get("invalid_outputs", 0) or candidate.get("inconsistent_questions", 0):
        reasons.append("output contract violations")
    return {"promote": not reasons, "reasons": reasons}
