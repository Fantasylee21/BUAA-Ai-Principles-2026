"""Count-level attribution and uncertainty for a small held-out validation set."""
from __future__ import annotations

import math
import random

from .metrics import compute_competition_score, evaluation_report


def binomial_count(rows, target):
    n = len(rows)
    k = sum(r.get("is_uncertain") == target for r in rows)
    if not n:
        return {"correct": 0, "total": 0, "wilson95": None}
    z = 1.959963984540054
    center = (k / n + z*z / (2*n)) / (1 + z*z/n)
    half = z * math.sqrt(k/n * (1-k/n)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return {"correct": k, "total": n, "wilson95": [center-half, center+half]}


def report(rows):
    result = evaluation_report(rows)
    result["counts"] = {
        "natural_conflict": binomial_count([r for r in rows if r["sample_type"] == "Conflict"
            and r["source_dataset"] in ("ch_sims", "ch_sims_v2")], "yes"),
        "va_conflict": binomial_count([r for r in rows if r["sample_type"] == "Conflict" and r["protocol"] == "VA"], "yes"),
        "conflict": binomial_count([r for r in rows if r["sample_type"] == "Conflict"], "yes"),
        "aligned_false_positive": binomial_count([r for r in rows if r["sample_type"] == "Aligned"], "yes"),
        "emotion_correct": sum(r["gt_label"] == r["pred_emotion"] for r in rows)}
    return result


def comparison(baseline, candidate, repeats=1000):
    if [r["sample_id"] for r in baseline] != [r["sample_id"] for r in candidate]:
        raise ValueError("Paired evaluation needs identical record order")
    changes = {"conflicts_corrected": 0, "conflicts_lost": 0,
               "false_positives_added": 0, "false_positives_removed": 0,
               "emotion_corrected": 0, "emotion_lost": 0}
    groups = {}
    for i, (b, c) in enumerate(zip(baseline, candidate)):
        groups.setdefault(b["split_group_id"], []).append(i)
        before, after = b["is_uncertain"], c["is_uncertain"]
        if b["sample_type"] == "Conflict":
            changes["conflicts_corrected"] += before != "yes" and after == "yes"
            changes["conflicts_lost"] += before == "yes" and after != "yes"
        else:
            changes["false_positives_added"] += before != "yes" and after == "yes"
            changes["false_positives_removed"] += before == "yes" and after != "yes"
        bc, cc = b["pred_emotion"] == b["gt_label"], c["pred_emotion"] == c["gt_label"]
        changes["emotion_corrected"] += not bc and cc
        changes["emotion_lost"] += bc and not cc
    rng = random.Random(42)
    units = list(groups.values())
    differences = []
    for _ in range(repeats):
        ids = [i for unit in rng.choices(units, k=len(units)) for i in unit]
        b = compute_competition_score([baseline[i] for i in ids])["score"]
        c = compute_competition_score([candidate[i] for i in ids])["score"]
        if b is not None and c is not None:
            differences.append(c-b)
    differences.sort()
    changes["paired_group_bootstrap95_score_delta"] = ([differences[int(.025*len(differences))],
        differences[min(len(differences)-1, int(.975*len(differences)))]] if differences else None)
    changes["uncertainty_note"] = "Group bootstrap and Wilson intervals are descriptive, not a multiple-selection correction."
    return changes
