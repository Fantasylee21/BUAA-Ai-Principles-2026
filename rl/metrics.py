"""Competition metric computation for model selection / early stopping.

    Score = 0.5 * Macro-F1(emotion)
          + 0.3 * Conflict_accuracy
          + 0.2 * Aligned_calibration

  * Macro-F1: macro-averaged F1 over the union of labels present in
    y_true and y_pred. Implemented manually with ``collections`` —
    no sklearn dependency.
  * Conflict_accuracy: fraction of truly-conflicting samples
    (sample_type == "Conflict") predicted with is_uncertain == "yes".
  * Aligned_calibration: fraction of truly-aligned samples
    (sample_type == "Aligned") predicted with is_uncertain == "no".

Empty-subset handling
---------------------
If a subset is empty (e.g. no Conflict samples in the evaluated records),
its component is reported as None and the score is computed by
proportionally renormalizing the remaining weights (e.g. with no conflict
samples: score = (0.5 * f1 + 0.2 * cal) / 0.7). For the official validation
split this never triggers (val always contains conflicts), but we stay safe.

Invalid predictions
-------------------
Records with pred_emotion=None (unparseable model output) are mapped to the
sentinel label "<invalid>", which joins the F1 label union as an extra
class — so broken outputs are penalized in Macro-F1 rather than silently
dropped. Likewise, is_uncertain=None counts as a miss in both
conflict_accuracy and aligned_calibration (it stays in the denominator).
"""

from collections import Counter

# Component weights of the competition metric.
W_F1 = 0.5
W_CONFLICT = 0.3
W_ALIGNED = 0.2

# Sentinel used for unparseable / missing emotion predictions so that they
# are penalized inside Macro-F1 as an additional (always-wrong) class.
INVALID_LABEL = "<invalid>"


def macro_f1(y_true, y_pred):
    """Macro-averaged F1 over the union of labels in y_true and y_pred.

    Pure-python implementation (no sklearn). Labels with neither support nor
    predictions contribute nothing (they are not in the union by definition);
    a label with zero precision and zero recall contributes F1 = 0.
    Returns 0.0 for empty inputs.
    """
    labels = set(y_true) | set(y_pred)
    if not labels:
        return 0.0
    f1_sum = 0.0
    for label in labels:
        tp = fp = fn = 0
        for t, p in zip(y_true, y_pred):
            if p == label and t == label:
                tp += 1
            elif p == label and t != label:
                fp += 1
            elif p != label and t == label:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall > 0:
            f1_sum += 2.0 * precision * recall / (precision + recall)
    return f1_sum / len(labels)


def compute_competition_score(records):
    """Compute the competition score from per-sample records.

    Each record is a dict:
        {"gt_label": ..., "sample_type": ...,
         "pred_emotion": ... (or None if unparseable),
         "is_uncertain": ... (or None)}

    Returns a dict with keys:
        macro_f1, conflict_accuracy, aligned_calibration, score,
        n_records, n_conflict, n_aligned, label_counts

    See the module docstring for empty-subset renormalization and the
    invalid-prediction policy.
    """
    y_true, y_pred = [], []
    n_conflict = n_aligned = 0
    conflict_hits = aligned_hits = 0

    for rec in records:
        pred_emotion = rec.get("pred_emotion")
        y_true.append(rec.get("gt_label"))
        y_pred.append(pred_emotion if pred_emotion is not None else INVALID_LABEL)

        sample_type = str(rec.get("sample_type", "")).strip().lower()
        is_uncertain = rec.get("is_uncertain")
        if sample_type == "conflict":
            n_conflict += 1
            if is_uncertain == "yes":
                conflict_hits += 1
        elif sample_type == "aligned":
            n_aligned += 1
            if is_uncertain == "no":
                aligned_hits += 1

    f1 = macro_f1(y_true, y_pred)
    conflict_accuracy = (conflict_hits / n_conflict) if n_conflict else None
    aligned_calibration = (aligned_hits / n_aligned) if n_aligned else None

    # Weighted sum with renormalization over available components.
    weighted_sum = 0.0
    weight_sum = 0.0
    weighted_sum += W_F1 * f1
    weight_sum += W_F1
    if conflict_accuracy is not None:
        weighted_sum += W_CONFLICT * conflict_accuracy
        weight_sum += W_CONFLICT
    if aligned_calibration is not None:
        weighted_sum += W_ALIGNED * aligned_calibration
        weight_sum += W_ALIGNED
    score = weighted_sum / weight_sum if weight_sum else 0.0

    return {
        "macro_f1": f1,
        "conflict_accuracy": conflict_accuracy,
        "aligned_calibration": aligned_calibration,
        "score": score,
        "n_records": len(records),
        "n_conflict": n_conflict,
        "n_aligned": n_aligned,
        "label_counts": dict(Counter(y_true)),
    }


def parse_prediction(completion_text):
    """Parse a raw model completion into a metric record fragment.

    Reuses the validated parser from reward.py. Returns a dict with keys
    ``pred_emotion`` and ``is_uncertain`` (values may be None when the
    completion is not a valid prediction).
    """
    # Local import keeps this module usable as both a flat script neighbor
    # (``import metrics`` with rl/ on sys.path) and as part of a package.
    try:
        from .reward import parse_completion  # type: ignore  # package style
    except ImportError:
        from reward import parse_completion  # type: ignore  # script style

    parsed = parse_completion(completion_text)
    if parsed is None:
        return {"pred_emotion": None, "is_uncertain": None}
    return {
        "pred_emotion": parsed["pred_emotion"],
        "is_uncertain": parsed["is_uncertain"],
    }
