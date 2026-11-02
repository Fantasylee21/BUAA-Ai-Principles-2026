#!/usr/bin/env python3
"""Audit a finished finite-action run on CPU; never update a training artifact."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl.experiment import check_disjoint, fingerprint, load_records
from rl.finite_actions import ACTIONS, validate_reward
from rl.finite_reports import report


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stats(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    def percentile(p):
        position = (len(values) - 1) * p
        lower = math.floor(position)
        upper = math.ceil(position)
        return values[lower] + (position - lower) * (values[upper] - values[lower])
    return {"n": len(values), "mean": statistics.mean(values), "min": values[0],
            "p10": percentile(.1), "median": percentile(.5), "p90": percentile(.9), "max": values[-1]}


def census(raw):
    groups = defaultdict(list)
    observable = defaultdict(list)
    for row in raw:
        groups[row["split_group_id"]].append(row)
        observable[(row["protocol"], row["video"], row.get("text_content", "") if row["protocol"] == "VT" else "")].append(row)
    cells = defaultdict(list)
    for row in raw:
        cells[f"{row['source_dataset']} / {row['protocol']} / {row['sample_type']}"].append(row)
    return {
        "n": len(raw), "groups": len(groups),
        "source_protocol_type": {key: len(value) for key, value in sorted(cells.items())},
        "emotions": dict(Counter(r["views"]["M12"]["label"] for r in raw)),
        "view_labels": {view: dict(Counter(r['views'][view]['label'] for r in raw)) for view in ('M1', 'M2', 'M12')},
        "use_in_main": dict(Counter(str(r.get("use_in_main")) for r in raw)),
        "annotation_count": dict(Counter(str(r.get("annotation_count")) for r in raw)),
        "label_basis": dict(Counter(str(r.get("label_basis")) for r in raw)),
        "generated": dict(Counter(str(r.get("source_is_generated")) for r in raw)),
        "type_vs_view_disagreement": dict(Counter(
            f"{r['sample_type']} / M1_ne_M2={r['views']['M1']['label'] != r['views']['M2']['label']}" for r in raw)),
        "use_in_main_by_type": dict(Counter(f"{r['sample_type']} / {r.get('use_in_main')}" for r in raw)),
        "same_group_different_type": sum(len({r["sample_type"] for r in group}) > 1 for group in groups.values()),
        "same_group_different_emotion": sum(len({r['views']['M12']['label'] for r in group}) > 1 for group in groups.values()),
        "same_group_different_visual_label": sum(len({r['views']['M1']['label'] for r in group}) > 1 for group in groups.values()),
        "same_group_different_video": sum(len({r['video'] for r in group}) > 1 for group in groups.values()),
        "combined_label_matches": dict(Counter(
            f"{r['sample_type']} / M12_eq_M1={r['views']['M12']['label'] == r['views']['M1']['label']} / M12_eq_M2={r['views']['M12']['label'] == r['views']['M2']['label']}" for r in raw)),
        "unclear_view_counts": {view: sum(not r['views'][view].get('is_clear', False) for r in raw) for view in ('M1', 'M2', 'M12')},
        "identical_observation_conflicting_targets": [
            {"sample_ids": [r["sample_id"] for r in group]}
            for group in observable.values()
            if len({(r['views']['M12']['label'], r['sample_type']) for r in group}) > 1],
        "vt_transcript_top": Counter(r.get("text_content", "") for r in raw if r["protocol"] == "VT").most_common(10),
    }


def gradient_rows(records, cache, table):
    output = []
    for row in records:
        entry = cache[row["sample_id"]]
        probs = [math.exp(v) for v in entry["reference_log_probs"]]
        assert abs(sum(probs) - 1) < 1e-6
        rewards = table["rewards"][row["sample_id"]]
        expected = sum(p*r for p, r in zip(probs, rewards))
        derivative = [p*(expected-r) for p, r in zip(probs, rewards)]
        best = max(range(6), key=probs.__getitem__)
        target = (row["gt_label"], "yes" if row["sample_type"] == "Conflict" else "no")
        gold = ACTIONS.index(target)
        output.append({
            "sample_id": row["sample_id"], "source": row["source_dataset"], "protocol": row["protocol"],
            "type": row["sample_type"], "joint_correct": best == gold,
            "emotion_correct": ACTIONS[best][0] == target[0],
            "uncertainty_correct": ACTIONS[best][1] == target[1],
            "p_gold_joint": probs[gold],
            "p_gold_emotion": sum(p for p, action in zip(probs, ACTIONS) if action[0] == target[0]),
            "p_gold_uncertainty": sum(p for p, action in zip(probs, ACTIONS) if action[1] == target[1]),
            "max_probability": max(probs), "task_score_gradient_norm": math.sqrt(sum(g*g for g in derivative)),
            "expected_reward": expected,
            "ce_score_gradient_norm_diagnostic": math.sqrt(sum((p-(i == gold))**2 for i, p in enumerate(probs))),
        })
    return output


def gradient_summary(items):
    return {"n": len(items), "joint_wrong": sum(not r["joint_correct"] for r in items),
            "emotion_wrong": sum(not r["emotion_correct"] for r in items),
            "uncertainty_wrong": sum(not r["uncertainty_correct"] for r in items),
            "max_probability": stats([r["max_probability"] for r in items]),
            "p_gold_uncertainty": stats([r["p_gold_uncertainty"] for r in items]),
            "p_gold_joint": stats([r["p_gold_joint"] for r in items]),
            "task_score_gradient_norm": stats([r["task_score_gradient_norm"] for r in items]),
            "gold_uncertainty_probability_below": {str(t): sum(r["p_gold_uncertainty"] < t for r in items) for t in (1e-2, 1e-3, 1e-6, 1e-9)},
            "task_gradient_below": {str(t): sum(r["task_score_gradient_norm"] < t for r in items) for t in (1e-3, 1e-6, 1e-9)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "Dataset")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir
    train, duplicates = load_records(args.data_root / "train.jsonl")
    val, val_duplicates = load_records(args.data_root / "val.jsonl")
    test, test_duplicates = load_records(args.data_root / "test_release.jsonl", require_labels=False)
    assert not (duplicates or val_duplicates or test_duplicates)
    check_disjoint(train, val)
    check_disjoint(train, test)
    check_disjoint(val, test)
    assert (len(train), len(val), len(test)) == (3354, 666, 734)
    context = read(run / "context.json")
    assert fingerprint(train) == context["identity"]["train_hash"]
    table = read(run / "reward-epoch0.json")
    validate_reward(table, train, 0, table["parent_policy"])
    manifest = read(run / "cache/complete.json")
    assert manifest["identity_hash"] == context["identity_hash"]
    entries = manifest["entries"]
    if isinstance(entries, list):
        entries = {r["sample_id"]: r for r in entries}
    before = gradient_rows(train, entries, table)
    masks = {
        "all": lambda r: True,
        "natural_conflict": lambda r: r["source"] == "ch_sims_v2" and r["type"] == "Conflict",
        "synthetic_conflict": lambda r: r["source"] != "ch_sims_v2" and r["type"] == "Conflict",
        "aligned": lambda r: r["type"] == "Aligned",
        "uncertainty_wrong": lambda r: not r["uncertainty_correct"],
        "joint_wrong": lambda r: not r["joint_correct"],
        "natural_conflict_missed": lambda r: r["source"] == "ch_sims_v2" and r["type"] == "Conflict" and not r["uncertainty_correct"],
        "natural_conflict_already_correct": lambda r: r["source"] == "ch_sims_v2" and r["type"] == "Conflict" and r["uncertainty_correct"],
        "aligned_false_positive": lambda r: r["type"] == "Aligned" and not r["uncertainty_correct"],
    }
    metrics = rows(run / "train-metrics.jsonl")
    observed = [r for update in metrics for r in update["samples"]]
    assert len(metrics) == 210 and len(observed) == len(train)
    assert Counter(r["sample_id"] for r in observed) == Counter(r["sample_id"] for r in train)
    evaluations = {}
    predictions = {}
    for path in sorted((run / "results").glob("*.predictions.jsonl")):
        name = path.name.removesuffix(".predictions.jsonl")
        predictions[name] = rows(path)
        assert [r["sample_id"] for r in predictions[name]] == [r["sample_id"] for r in val]
        assert all(all(p[field] == gold[field] for field in ("gt_label", "sample_type", "protocol", "source_dataset"))
                   for p, gold in zip(predictions[name], val)), name
        full = read(run / "results" / f"{name}.json")
        recomputed = report(predictions[name])
        assert recomputed == full["metrics"], name
        evaluations[name] = {"metrics": recomputed, "comparison": full.get("vs_baselines")}
    assert set(evaluations) == {"baseline-free", "baseline-finite", "eval-000050", "eval-000100", "eval-000150", "eval-000210"}
    base = {r["sample_id"]: r for r in predictions["baseline-free"]}
    changes = []
    for candidate in predictions["eval-000210"]:
        prior = base[candidate["sample_id"]]
        if (prior["pred_emotion"], prior["is_uncertain"]) != (candidate["pred_emotion"], candidate["is_uncertain"]):
            changes.append({"sample_id": candidate["sample_id"], "type": candidate["sample_type"],
                            "protocol": candidate["protocol"], "source": candidate["source_dataset"],
                            "before": [prior["pred_emotion"], prior["is_uncertain"]],
                            "after": [candidate["pred_emotion"], candidate["is_uncertain"]]})
    val_signals = {}
    best_rows = {r["sample_id"]: r for r in predictions["eval-000210"]}
    raw_val = {r["sample_id"]: r for r in rows(args.data_root / "val.jsonl")}
    for row in val:
        entry = entries[row["sample_id"]]
        pp = [math.exp(v) for v in entry["reference_log_probs"]]
        pred = best_rows[row["sample_id"]]
        val_signals[row["sample_id"]] = {
            "type": row["sample_type"], "source": row["source_dataset"], "protocol": row["protocol"],
            "use_in_main": raw_val[row["sample_id"]].get("use_in_main"),
            "reference_p_yes": sum(p for p, a in zip(pp, ACTIONS) if a[1] == "yes"),
            "trained_p_yes": sum(p for p, a in zip(pred["probabilities"], ACTIONS) if a[1] == "yes"),
            "reference_action": list(ACTIONS[max(range(6), key=pp.__getitem__)]),
            "trained_action": [pred["pred_emotion"], pred["is_uncertain"]],
        }
    clipping = {}
    for label in ("positive", "negative", "neutral"):
        selected = [r for r in train if r["gt_label"] == label]
        correct_rewards, wrong_rewards = [], []
        for row in selected:
            rr = table["rewards"][row["sample_id"]]
            # Remove the uncertainty reward and undo the 0.5 emotion weight.
            for i, action in enumerate(ACTIONS):
                if action[1] != "no":
                    continue
                u = .2 * len(train) / table["type_counts"]["Aligned"] if row["sample_type"] == "Aligned" else 0
                value = 2 * (rr[i] - u)
                (correct_rewards if action[0] == label else wrong_rewards).append(value)
        clipping[label] = {"correct": stats(correct_rewards), "wrong": stats(wrong_rewards),
                           "wrong_at_clip": sum(abs(abs(v)-1) < 1e-9 for v in wrong_rewards)}
    summary = {
        "run_dir": str(run.resolve()),
        "verification": {"training_rows_once": True, "all_metrics_recomputed": True,
                         "train_only_reward_reconstructed": True, "splits_disjoint": True},
        "dataset": {split: census(rows(args.data_root / f"{split}.jsonl")) for split in ("train", "val")},
        "test": {"n": len(test), "source_protocol": dict(Counter(f"{r['source_dataset']} / {r['protocol']}" for r in test))},
        "initial_training_signals": {name: gradient_summary([r for r in before if mask(r)]) for name, mask in masks.items()},
        "reward": {"type_counts": table["type_counts"], "emotion_by_class": clipping,
                   "uncertainty_gap": {kind: w*len(train)/n for kind, w in (("Conflict", .3), ("Aligned", .2)) for n in [table["type_counts"][kind]]}},
        "observed_training": {"updates": len(metrics), "samples": len(observed),
            "max_probability": stats([r["max_probability"] for r in observed]),
            "task_and_kl_score_gradient_norm": stats([r["score_gradient_norm"] for r in observed]),
            "joint_correct": sum(r["emotion_correct"] and r["uncertainty_correct"] for r in observed),
            "uncertainty_correct": sum(r["uncertainty_correct"] for r in observed),
            "emotion_correct": sum(r["emotion_correct"] for r in observed),
            "first50_mean_gradient_norm": statistics.mean(r["grad_norm"] for r in metrics[:50]),
            "last50_mean_gradient_norm": statistics.mean(r["grad_norm"] for r in metrics[-50:]),
            "gradient_clipped_updates": sum(r["grad_norm"] > 1 for r in metrics)},
        "evaluations": evaluations, "best_vs_free_changed_rows": changes,
        "initial_signals_by_sample": before,
        "validation_signals_by_sample": val_signals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("evaluations", "initial_signals_by_sample", "validation_signals_by_sample", "dataset", "best_vs_free_changed_rows")}, ensure_ascii=False, indent=2))
    print("CENSUS", json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("vt_transcript_top",)} for k, v in summary["dataset"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
