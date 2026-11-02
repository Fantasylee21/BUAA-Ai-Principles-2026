"""Validated TRAIN-only joint-affect data, folds, remixing, metrics and gates.

This module has no torch or optimization dependency. Label order is deliberately
independent of historical RL code: negative, neutral, positive, then unclear.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import random

from rl.experiment import file_hash, fingerprint

VERSION = "joint_affect_v1"
EMOTIONS = ("negative", "neutral", "positive")
EVIDENCE_LABELS = EMOTIONS + ("unclear",)
NATURAL = {"ch_sims", "ch_sims_v2"}
THRESHOLD = .5
BUDGET_HOURS = 120
QUESTIONS = {
    "VT": "你能补充说明，画面中的表现和这段话分别给你怎样的感受吗？",
    "VA": "你能补充说明，画面中的表现和声音分别给你怎样的感受吗？",
}
FIELDS = ("sample_id", "pred_emotion", "is_uncertain", "clarifying_question")


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_once(path, content):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("Refusing to replace a different joint-affect artifact: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def question_for_protocol(protocol):
    if protocol not in QUESTIONS:
        raise ValueError("Unknown protocol")
    return QUESTIONS[protocol]


def _natural(row):
    return row.get("source_dataset") in NATURAL


def _ids(rows):
    ids = [row.get("sample_id") for row in rows]
    if not rows or any(not isinstance(sid, str) or not sid or sid != sid.strip() for sid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Expected nonempty unique sample IDs")
    return ids


def targets(row):
    """Real M12 remains supervised even when either evidence view is unclear."""
    views = row.get("views", {})
    result = {}
    for name, key in (("M1", "vision"), ("M2", "second")):
        view = views.get(name, {})
        if view.get("label") not in EMOTIONS or type(view.get("is_clear")) is not bool:
            raise ValueError("Training evidence requires valid original labels and boolean is_clear")
        result[key] = EMOTIONS.index(view["label"]) if view["is_clear"] else 3
    if row.get("sample_type") not in ("Conflict", "Aligned"):
        raise ValueError("Original or remixed relation label is required")
    remix = bool(row.get("is_remix", False))
    if remix:
        if "M12" in views or row.get("emotion_mask") != 0:
            raise ValueError("Remix must have no fabricated M12 label and emotion_mask=0")
        emotion = -100
    else:
        emotion = views.get("M12", {}).get("label")
        if emotion not in EMOTIONS:
            raise ValueError("Real examples require their original M12 emotion")
        emotion = EMOTIONS.index(emotion)
    result.update(emotion=emotion, relation=int(row["sample_type"] == "Conflict"), emotion_mask=0 if remix else 1)
    return result


def canonical_inference(row):
    """Input-only helper: official test rows need not contain views/is_clear."""
    result = deepcopy(row)
    sid, video = result.get("sample_id"), result.get("video")
    if not isinstance(sid, str) or not sid or not isinstance(video, str) or not video:
        raise ValueError("Every input needs a sample ID and relative video")
    protocol = result.get("protocol")
    if protocol is None:
        tag = sid.split(":")[1] if ":" in sid else ""
        protocol = {"VT": "VT", "VA": "VA", "accept_a_svt": "VT", "accept_c_svt": "VT", "accept_a_va": "VA"}.get(tag)
    if protocol not in ("VT", "VA"):
        raise ValueError("Expected VT or VA")
    if Path(video).is_absolute() or ".." in PurePosixPath(video.replace("\\", "/")).parts:
        raise ValueError("Video path must stay within the data root")
    result.update(protocol=protocol, text_content=str(result.get("text_content", result.get("text", ""))))
    return result


def load_test(path, data_root=None, expected_rows=734):
    rows = [canonical_inference(json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    _ids(rows)
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError("Unexpected official test row count")
    if data_root is not None:
        root = Path(data_root).resolve()
        for row in rows:
            video = (root / row["video"]).resolve()
            if not video.is_relative_to(root) or not video.is_file():
                raise ValueError("Missing or escaped test media")
    return rows


def load_training(train_jsonl, data_root, expected_rows=3354):
    path, root = Path(train_jsonl).resolve(), Path(data_root).resolve()
    if path.name != "train.jsonl":
        raise ValueError("Only original train.jsonl is accepted for supervised preparation")
    rows = [canonical_inference(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _ids(rows)
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError("Original training row count changed")
    media = {}
    for row in rows:
        if row.get("split") != "train" or row.get("is_remix"):
            raise ValueError("Only original real TRAIN examples may define folds")
        modes = {"M1": "vision", "M2": "text" if row["protocol"] == "VT" else "audio"}
        for view, modality in modes.items():
            if row.get("views", {}).get(view, {}).get("modality") != modality:
                raise ValueError("Original view modality disagrees with its protocol")
        row["_targets"] = targets(row)
        row["emotion_mask"] = 1
        video = (root / row["video"]).resolve()
        if not video.is_relative_to(root) or not video.is_file():
            raise ValueError("Missing or escaped original training media")
        if row["video"] not in media:
            media[row["video"]] = file_hash(video)
    return rows, media


def content_groups(rows, media, pressure=False):
    """Same video, source pair and SHA form transitive content components."""
    _ids(rows)
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    seen = {}
    for i, row in enumerate(rows):
        video = row.get("video")
        sha = media.get(video)
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("Every video requires a verified SHA-256")
        keys = [("video", video), ("sha256", sha)]
        if row.get("split_group_id"):
            keys.append(("source_pair", row["split_group_id"]))
        if pressure:
            keys.append(("parent", str(PurePosixPath(video.replace("\\", "/")).parent)))
        for key in keys:
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    members = defaultdict(list)
    for i in range(len(rows)):
        members[find(i)].append(rows[i]["sample_id"])
    names = {i: fingerprint(sorted(ids)) for i, ids in members.items()}
    return [names[find(i)] for i in range(len(rows))]


def _balance(row):
    conflict, natural = row["sample_type"] == "Conflict", _natural(row)
    return [1, int(conflict), int(not conflict), int(natural and conflict),
            int(natural and conflict and row["protocol"] == "VA"),
            int(natural and conflict and row["protocol"] == "VT"),
            int(row["protocol"] == "VT"), int(row["protocol"] == "VA")]


def fold_manifest(rows, media, seed=42, pressure=False):
    if seed != 42:
        raise ValueError("The split is fixed at 42; seed 43 changes training, not partitions")
    groups = content_groups(rows, media, pressure)
    members = defaultdict(list)
    for i, group in enumerate(groups):
        members[group].append(i)
    if len(members) < 3:
        raise ValueError("Three independent groups are required")
    vectors = {g: [sum(_balance(rows[i])[k] for i in ids) for k in range(8)] for g, ids in members.items()}
    total = [max(1, sum(v[k] for v in vectors.values())) for k in range(8)]
    keys = sorted(members, key=lambda g: (-max(vectors[g][k] / total[k] for k in range(8)), -len(members[g]), fingerprint([seed, g])))
    loads, assignment = [[0.] * 8 for _ in range(3)], {}
    for n, group in enumerate(keys):
        def cost(fold):
            trial = [row[:] for row in loads]
            trial[fold] = [a + b for a, b in zip(trial[fold], vectors[group])]
            variance = sum(sum((trial[j][k] / total[k] - sum(trial[z][k] / total[k] for z in range(3)) / 3) ** 2 for j in range(3)) for k in range(8))
            return variance, loads[fold][0], fold
        fold = n if n < 3 else min(range(3), key=cost)
        assignment[group] = fold
        loads[fold] = [a + b for a, b in zip(loads[fold], vectors[group])]
    outer = [assignment[g] for g in groups]
    folds = []
    for fold in range(3):
        held = [i for i, f in enumerate(outer) if f == fold]
        train = [i for i, f in enumerate(outer) if f != fold]
        if {rows[i]["sample_type"] for i in held} != {"Conflict", "Aligned"}:
            raise ValueError("Every held fold requires original aligned and conflict support")
        if {groups[i] for i in held} & {groups[i] for i in train}:
            raise RuntimeError("An independent content group crossed the fold")
        folds.append({"fold": fold, "train": train, "held": held,
                      "support": {"natural_conflict": sum(_natural(rows[i]) and rows[i]["sample_type"] == "Conflict" for i in held),
                                  "natural_va_conflict": sum(_natural(rows[i]) and rows[i]["sample_type"] == "Conflict" and rows[i]["protocol"] == "VA" for i in held)}})
    result = {"split_seed": 42, "pressure": pressure, "groups": groups, "outer": outer,
              "folds": folds, "independent_groups": len(members)}
    result["hash"] = fingerprint(result)
    return result


def prepare_manifest(train_jsonl, data_root, expected_rows=3354):
    rows, media = load_training(train_jsonl, data_root, expected_rows)
    folds = {"content": fold_manifest(rows, media), "parent": fold_manifest(rows, media, pressure=True)}
    identity = {"algorithm": VERSION, "budget_hours": BUDGET_HOURS, "train_sha256": file_hash(train_jsonl),
                "rows_hash": fingerprint(rows), "media_sha256": media, "emotion_order": list(EMOTIONS),
                "evidence_order": list(EVIDENCE_LABELS), "threshold": THRESHOLD, "threshold_inclusive": True,
                "split_seed": 42, "fold_hashes": {k: v["hash"] for k, v in folds.items()}}
    return {"identity": identity, "identity_hash": fingerprint(identity), "algorithm": VERSION,
            "train_jsonl": str(Path(train_jsonl).resolve()), "data_root": str(Path(data_root).resolve()),
            "train": rows, "media": media, "folds": folds}


build_manifest = prepare_manifest


def validate_manifest(manifest):
    """Bind mutable-looking convenience fields back to the frozen identity."""
    identity, rows, media = manifest["identity"], manifest["train"], manifest["media"]
    if manifest["identity_hash"] != fingerprint(identity) or identity.get("algorithm") != VERSION:
        raise ValueError("Corrupt joint-affect manifest identity")
    if identity.get("emotion_order") != list(EMOTIONS) or identity.get("evidence_order") != list(EVIDENCE_LABELS):
        raise ValueError("Manifest uses a different label order")
    _ids(rows)
    if identity.get("rows_hash") != fingerprint(rows) or identity.get("media_sha256") != media:
        raise ValueError("Manifest rows or media are detached from their frozen identity")
    if set(media) != {row["video"] for row in rows} or any(row.get("split") != "train" or row.get("is_remix") for row in rows):
        raise ValueError("Only original training media and rows can define the manifest")
    for row in rows:
        if row.get("_targets") != targets(row):
            raise ValueError("Cached targets differ from original per-view labels")
    for name in ("content", "parent"):
        design = manifest["folds"][name]
        value = {key: item for key, item in design.items() if key != "hash"}
        if design["hash"] != fingerprint(value) or identity["fold_hashes"].get(name) != design["hash"]:
            raise ValueError("Manifest fold design changed: " + name)
        if len(design["groups"]) != len(rows) or len(design["outer"]) != len(rows) or {r["fold"] for r in design["folds"]} != {0, 1, 2}:
            raise ValueError("Manifest fold coverage changed")
        for fold in design["folds"]:
            train, held = fold["train"], fold["held"]
            if len(set(train)) != len(train) or len(set(held)) != len(held) or set(train) & set(held) or set(train) | set(held) != set(range(len(rows))):
                raise ValueError("Manifest training/held index isolation changed")
            if any(design["outer"][i] != fold["fold"] for i in held) or {design["groups"][i] for i in train} & {design["groups"][i] for i in held}:
                raise ValueError("Manifest content group leakage detected")
    return manifest


def class_counts(rows, indices=None):
    selected = [rows[i] for i in indices] if indices is not None else list(rows)
    if any(row.get("is_remix") for row in selected):
        raise ValueError("Class weights are computed from real training examples only")
    result = {"n_real": len(selected), "aligned": 0, "natural_conflict": 0, "synthetic_conflict": 0,
              "emotion": [0, 0, 0], "vision": [0, 0, 0, 0], "second": [0, 0, 0, 0], "relation": [0, 0]}
    for row in selected:
        t = targets(row)
        for name in ("emotion", "vision", "second", "relation"):
            result[name][t[name]] += 1
        category = "aligned" if row["sample_type"] == "Aligned" else "natural_conflict" if _natural(row) else "synthetic_conflict"
        result[category] += 1
    frequency = dict(Counter(row["video"] for row in selected))
    result.update(real_total=len(selected), video_frequency=frequency, unique_videos=len(frequency))
    return result


def visual_dup_count(rows, indices=None):
    selected = [rows[i] for i in indices] if indices is not None else list(rows)
    return dict(Counter(r["video"] for r in selected))


def visual_weights(rows, indices=None):
    selected = [rows[i] for i in indices] if indices is not None else list(rows)
    counts = visual_dup_count(selected)
    return [len(selected) / (len(counts) * counts[r["video"]]) for r in selected] if selected else []


def remix_training(rows, train_indices, media, seed, epoch=0):
    indices = list(train_indices)
    if len(indices) != len(set(indices)) or any(type(i) is not int or not 0 <= i < len(rows) for i in indices):
        raise ValueError("Remix needs unique real-training indices")
    if any(rows[i].get("split") != "train" or rows[i].get("is_remix") for i in indices):
        raise ValueError("Only original real training rows can be remixed")
    desired = len(indices) // 2
    if not desired:
        return []
    buckets = {"vision": defaultdict(list), "second": defaultdict(list)}
    for i in indices:
        row = rows[i]
        for view, name in (("M1", "vision"), ("M2", "second")):
            if row["views"][view].get("is_clear") is True:
                buckets[name][(row["protocol"], row["views"][view]["label"])].append(i)
    def valid(a, b):
        return rows[a]["video"] != rows[b]["video"] and media[rows[a]["video"]] != media[rows[b]["video"]]
    options = {True: [], False: []}
    for protocol in ("VT", "VA"):
        for left in EMOTIONS:
            for right in EMOTIONS:
                aa, bb = buckets["vision"][(protocol, left)], buckets["second"][(protocol, right)]
                if aa and bb and any(valid(a, b) for a in aa for b in bb):
                    options[left == right].append((protocol, left, right))
    if not options[True] or not options[False]:
        raise ValueError("Training partition lacks clear, different-media same/different remix support")
    rng = random.Random(int(fingerprint([VERSION, seed, epoch, sorted(rows[i]["sample_id"] for i in indices)]), 16))
    result = []
    for n in range(desired):
        same = n % 2 == 0
        protocol, left, right = rng.choice(options[same])
        aa, bb = buckets["vision"][(protocol, left)], buckets["second"][(protocol, right)]
        for _ in range(128):
            a, b = rng.choice(aa), rng.choice(bb)
            if valid(a, b):
                break
        else:
            a, b = next((a, b) for a in aa for b in bb if valid(a, b))
        vision, second = rows[a], rows[b]
        sid = "remix:" + fingerprint([seed, epoch, n, vision["sample_id"], second["sample_id"]])[:24]
        row = {"sample_id": sid, "split": "train", "protocol": protocol, "video": vision["video"],
               "text_content": second["text_content"] if protocol == "VT" else "", "_second_video": second["video"],
               "source_dataset": "train_remix", "sample_type": "Aligned" if same else "Conflict",
               "views": {"M1": deepcopy(vision["views"]["M1"]), "M2": deepcopy(second["views"]["M2"])},
               "is_remix": True, "emotion_mask": 0, "_vision_source_idx": a, "_second_source_idx": b,
               "_remix": {"vision_sample_id": vision["sample_id"], "second_sample_id": second["sample_id"],
                          "same_emotion": same, "seed": seed, "epoch": epoch,
                          "mix_ratio_note": "one per two real rows; same/different counts differ by at most one when odd"}}
        row["_targets"] = targets(row)
        result.append(row)
    return result


def second_source(row, all_rows):
    return all_rows[row["_second_source_idx"]] if row.get("is_remix") else row


def training_examples(rows, train_indices, media, seed, epoch=0):
    real = [deepcopy(rows[i]) for i in train_indices]
    for row in real:
        row["_targets"] = targets(row)
    return real + remix_training(rows, train_indices, media, seed, epoch)


def _probabilities(value, size):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError("Unexpected probability vector geometry")
    values = [float(v) for v in value]
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in values) or abs(sum(values) - 1.) > 1e-5:
        raise ValueError("Expected finite normalized probabilities")
    return values


def _predictions(rows, predictions):
    ids = _ids(rows)
    if isinstance(predictions, dict):
        if set(predictions) != set(ids):
            raise ValueError("Prediction mapping must cover every sample exactly")
        predictions = [{"sample_id": sid, **predictions[sid]} for sid in ids]
    if len(predictions) != len(rows) or [p.get("sample_id") for p in predictions] != ids:
        raise ValueError("Predictions must preserve exact sample IDs/order")
    result = []
    for row, prediction in zip(rows, predictions):
        p = _probabilities(prediction["emotion_probs"], 3)
        uncertain = float(prediction["uncertain_prob"])
        if not math.isfinite(uncertain) or not 0 <= uncertain <= 1:
            raise ValueError("Invalid uncertainty probability")
        result.append({**prediction, "pred_emotion": EMOTIONS[max(range(3), key=p.__getitem__)],
                       "is_uncertain": "yes" if uncertain >= THRESHOLD else "no"})
    return result


def _f1(records, labels):
    total = 0.
    for label in labels:
        tp = sum(r["gt_label"] == label and r["pred_emotion"] == label for r in records)
        denominator = sum(r["gt_label"] == label for r in records) + sum(r["pred_emotion"] == label for r in records)
        total += 2 * tp / denominator if denominator else 0.
    return total / len(labels)


def _metrics_records(records):
    _ids(records)
    conflict = [r for r in records if r["sample_type"] == "Conflict"]
    aligned = [r for r in records if r["sample_type"] == "Aligned"]
    natural = [r for r in conflict if r["source_dataset"] in NATURAL]
    synthetic = [r for r in conflict if r["source_dataset"] not in NATURAL]
    tp = lambda rr: sum(r["is_uncertain"] == "yes" for r in rr)
    recall, specificity = tp(conflict) / len(conflict) if conflict else None, 1 - tp(aligned) / len(aligned) if aligned else None
    fpn, f3 = _f1(records, ("negative", "positive")), _f1(records, EMOTIONS)
    component = .3 * recall + .2 * specificity if recall is not None and specificity is not None else None
    pairs = defaultdict(list)
    for r in records:
        pairs[r["video"]].append(r)
    opposite = [pair for pair in pairs.values() if len(pair) == 2 and {r["protocol"] for r in pair} == {"VT", "VA"} and {r["sample_type"] for r in pair} == {"Conflict", "Aligned"}]
    unc_correct = lambda r: r["is_uncertain"] == ("yes" if r["sample_type"] == "Conflict" else "no")
    evidence = [r for r in records if r.get("evidence_correct") is True]
    evidence_rows = [r for r in records if r.get("vision_correct") is not None]
    return {"rows": len(records), "selection_score": .5 * fpn + component if component is not None else None,
        "f1_pn": fpn, "macro_f1_fixed3": f3, "uncertainty_component": component,
        "competition_score_fixed3": .5 * f3 + component if component is not None else None,
        "conflict_recall": recall, "specificity": specificity, "n_conflict": len(conflict), "n_aligned": len(aligned),
        "true_positives": tp(conflict), "false_positives": tp(aligned), "natural_tp": tp(natural), "natural_n": len(natural),
        "natural_va_tp": tp([r for r in natural if r["protocol"] == "VA"]),
        "natural_va_n": sum(r["protocol"] == "VA" for r in natural),
        "natural_vt_tp": tp([r for r in natural if r["protocol"] == "VT"]),
        "natural_vt_n": sum(r["protocol"] == "VT" for r in natural),
        "synthetic_tp": tp(synthetic), "synthetic_n": len(synthetic),
        "vision_four_state_accuracy": sum(r["vision_correct"] for r in evidence_rows) / len(evidence_rows) if evidence_rows else None,
        "second_four_state_accuracy": sum(r["second_correct"] for r in evidence_rows) / len(evidence_rows) if evidence_rows else None,
        "neutral_support": sum(r["gt_label"] == "neutral" for r in records),
        "neutral_correct": sum(r["gt_label"] == r["pred_emotion"] == "neutral" for r in records),
        "joint_state_support": dict(sorted(Counter(str(r["joint_state"]) for r in records if "joint_state" in r).items())),
        "evidence_correct": len(evidence), "evidence_correct_but_rejected": sum(not unc_correct(r) for r in evidence),
        "aligned_evidence_correct_but_uncertain": sum(r["sample_type"] == "Aligned" and not unc_correct(r) for r in evidence),
        "opposite_pair_count": len(opposite), "opposite_pair_both_correct": sum(all(unc_correct(r) for r in pair) for pair in opposite),
        "opposite_pair_both_outputs_correct": sum(all(unc_correct(r) and r["gt_label"] == r["pred_emotion"] for r in pair) for pair in opposite),
        "_records": records}


def metrics(rows, predictions):
    if any(row.get("is_remix") for row in rows):
        raise ValueError("Evaluation uses original real examples, never remix targets")
    result = []
    for row, prediction in zip(rows, _predictions(rows, predictions)):
        t = targets(row)
        evidence_correct = vision_correct = second_correct = None
        if "vision_probs" in prediction and "second_probs" in prediction:
            v, m = _probabilities(prediction["vision_probs"], 4), _probabilities(prediction["second_probs"], 4)
            vision_correct = max(range(4), key=v.__getitem__) == t["vision"]
            second_correct = max(range(4), key=m.__getitem__) == t["second"]
            evidence_correct = vision_correct and second_correct
        result.append({"sample_id": row["sample_id"], "video": row["video"], "protocol": row["protocol"],
            "source_dataset": row.get("source_dataset", "unknown"), "sample_type": row["sample_type"],
            "gt_label": EMOTIONS[t["emotion"]], "pred_emotion": prediction["pred_emotion"],
            "is_uncertain": prediction["is_uncertain"], "evidence_correct": evidence_correct,
            "vision_correct": vision_correct, "second_correct": second_correct,
            "joint_state": 4 * t["vision"] + t["second"]})
    report = _metrics_records(result)
    report["by_protocol"] = {p: {k: v for k, v in _metrics_records([r for r in result if r["protocol"] == p]).items() if k != "_records"} for p in sorted({r["protocol"] for r in result})}
    return report


evaluate_predictions = metrics


def _paired(a, b):
    if {r["sample_id"] for r in a} != {r["sample_id"] for r in b}:
        raise ValueError("Paired predictions cover different holdout IDs")
    other = {r["sample_id"]: r for r in b}
    for r in a:
        s = other[r["sample_id"]]
        if any(r[k] != s[k] for k in ("gt_label", "sample_type", "protocol", "video", "source_dataset")):
            raise ValueError("Paired prediction labels or protocols differ")


def promotion_gate(fold_results, pressure_result=None, replication=None):
    if len(fold_results) != 3 or {r["fold"] for r in fold_results} != {0, 1, 2}:
        raise ValueError("The gate requires exactly three independent held folds")
    control_rows = [row for fold in fold_results for row in fold["control"]["_records"]]
    candidate_rows = [row for fold in fold_results for row in fold["candidate"]["_records"]]
    _paired(control_rows, candidate_rows)
    a, b = _metrics_records(control_rows), _metrics_records(candidate_rows)
    gains = []
    for fold in fold_results:
        _paired(fold["control"]["_records"], fold["candidate"]["_records"])
        fa, fb = _metrics_records(fold["control"]["_records"]), _metrics_records(fold["candidate"]["_records"])
        if fa["selection_score"] is None or fb["selection_score"] is None:
            raise ValueError("Every held fold needs both aligned and conflict support")
        gains.append(fb["selection_score"] - fa["selection_score"])
    if a["selection_score"] is None or b["selection_score"] is None:
        raise ValueError("Both aligned and conflict classes are needed for promotion")
    checks = {"score_gain_at_least_0_01": b["selection_score"] - a["selection_score"] >= .01 - 1e-12,
        "f1_pn_drop_at_most_0_005": b["f1_pn"] >= a["f1_pn"] - .005 - 1e-12,
        "natural_tp_gain_at_least_5": b["natural_tp"] - a["natural_tp"] >= 5,
        "natural_va_tp_gain_at_least_2": b["natural_va_tp"] - a["natural_va_tp"] >= 2,
        "false_positive_increase_at_most_1_percent_aligned": b["false_positives"] - a["false_positives"] <= .01 * a["n_aligned"] + 1e-12,
        "at_least_two_folds_do_not_regress": sum(gain >= -1e-12 for gain in gains) >= 2}
    primary_passed = all(checks.values())
    pressure_gain, pressure_passed = None, None
    if pressure_result is not None:
        pa, pb = pressure_result["control"], pressure_result["candidate"]
        _paired(pa["_records"], pb["_records"])
        _paired(control_rows, pa["_records"])
        _paired(candidate_rows, pb["_records"])
        pa, pb = _metrics_records(pa["_records"]), _metrics_records(pb["_records"])
        pressure_gain = pb["uncertainty_component"] - pa["uncertainty_component"]
        pressure_passed = pressure_gain >= .005 - 1e-12
    passed = primary_passed and pressure_passed is True
    replication_passed = replication.get("primary_passed", replication.get("passed", False)) if replication is not None else None
    return {"primary_passed": primary_passed, "pressure_passed": pressure_passed, "passed": passed,
        "checks": checks, "score_gain": b["selection_score"] - a["selection_score"], "fold_gains": gains,
        "pressure_gain": pressure_gain, "replication_passed": replication_passed,
        "ready_for_full_training": passed and replication_passed is True,
        "control": {k: v for k, v in a.items() if k != "_records"}, "candidate": {k: v for k, v in b.items() if k != "_records"}}


def export_csv(path, test_rows, predictions, expected_rows=734):
    if expected_rows is not None and len(test_rows) != expected_rows:
        raise ValueError("Official submission requires all 734 test rows")
    rows = []
    for original, prediction in zip(test_rows, _predictions(test_rows, predictions)):
        uncertain = prediction["is_uncertain"]
        rows.append({"sample_id": original["sample_id"], "pred_emotion": prediction["pred_emotion"],
                     "is_uncertain": uncertain,
                     "clarifying_question": question_for_protocol(original["protocol"]) if uncertain == "yes" else "none"})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    content = stream.getvalue().encode("utf-8")
    _write_once(path, content)
    return {"file": str(Path(path).resolve()), "rows": len(rows), "sha256": hashlib.sha256(content).hexdigest(),
            "question_policy": "neutral_modality_template_v1", "submitted_to_kaggle": False}
