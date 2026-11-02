"""Pair TRAIN-derived SFT holdout predictions and apply the fixed promotion gate.

No model loading, training, official validation/test access, or remote operation.
Run: python -m scripts.protocol_sft_summary --run-dir runs/PROTOCOL_RUN
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path, PurePosixPath
import re

from rl.experiment import file_hash, fingerprint
from rl.metrics import compute_competition_score
from rl.reward import parse_completion, question_is_consistent
from scripts.evidence_supplement import FIELDS, write_immutable

VERSION = "omni_protocol_sft_ablation_v1"
NATURAL = {"ch_sims", "ch_sims_v2"}


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def checked_arm(root, arm, identity_hash, raw, hashes):
    selected_path, report_path = root / arm / "selected-checkpoint.json", root / arm / "holdout-report.json"
    selected, report = read(selected_path), read(report_path)
    hashes[selected_path], hashes[report_path] = file_hash(selected_path), file_hash(report_path)
    if (selected.get("algorithm"), selected.get("identity_hash"), selected.get("holdout_used_for_selection")) != (VERSION, identity_hash, False):
        raise ValueError("Invalid development-selected checkpoint identity: " + arm)
    if selected.get("selected_by") != "training-only development Score":
        raise ValueError("Checkpoint was not selected solely on training-derived development Score")
    if (report.get("algorithm"), report.get("identity_hash"), report.get("arm"), report.get("checkpoint"), report.get("used_for_selection")) != (
            VERSION, identity_hash, arm, selected["checkpoint"], False):
        raise ValueError("Holdout report does not belong to the selected checkpoint: " + arm)
    if report.get("official_val_read") is not False or report.get("official_test_read") is not False:
        raise ValueError("Official validation/test cannot be part of this holdout gate")
    checkpoint = PurePosixPath(str(selected["checkpoint"]).replace("\\", "/")).name
    if not re.fullmatch(r"checkpoint-\d+", checkpoint):
        raise ValueError("Invalid selected checkpoint name")
    prefix = root / arm / "evaluation" / (checkpoint + "_holdout")
    details_path = prefix.with_name(prefix.name + "_details.jsonl")
    csv_path = prefix.with_name(prefix.name + "_predictions.csv")
    metrics_path = prefix.with_name(prefix.name + "_metrics.json")
    details, metrics = read_rows(details_path), read(metrics_path)
    for path in (details_path, csv_path, metrics_path):
        if not path.resolve().is_relative_to(root):
            raise ValueError("Holdout artifact escapes the prepared run")
        hashes[path] = file_hash(path)
    if report["metrics"] != metrics:
        raise ValueError("Holdout metric file differs from the completed report")
    ids = [r["sample_id"] for r in raw]
    if [r.get("sample_id") for r in details] != ids or len(set(ids)) != len(ids):
        raise ValueError("Holdout details must cover the exact original IDs and order: " + arm)
    reader = csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8-sig"), newline=""))
    if reader.fieldnames != list(FIELDS):
        raise ValueError("Unexpected holdout prediction CSV schema")
    csv_predictions = list(reader)
    if [r.get("sample_id") for r in csv_predictions] != ids:
        raise ValueError("Holdout prediction CSV IDs/order changed")
    records = []
    for original, detail, csv_prediction in zip(raw, details, csv_predictions):
        label, kind = original["views"]["M12"]["label"], original["sample_type"]
        if (detail.get("true_emotion"), detail.get("true_sample_type"), detail.get("protocol"), detail.get("source_dataset", "unknown")) != (
                label, kind, original["protocol"], original.get("source_dataset", "unknown")):
            raise ValueError("Holdout gold labels/protocol/source differ from original training rows")
        if detail.get("inference_error"):
            raise ValueError("An inference error cannot enter the promotion gate")
        prediction = {key: detail.get(key) for key in FIELDS[1:]}
        if parse_completion(json.dumps(prediction)) is None or not question_is_consistent(prediction):
            raise ValueError("Invalid or inconsistent holdout prediction")
        if csv_prediction != {"sample_id": original["sample_id"], **prediction}:
            raise ValueError("CSV and detailed holdout predictions differ")
        records.append({"sample_id": original["sample_id"], "gt_label": label, "sample_type": kind,
                        "protocol": original["protocol"], "source_dataset": original.get("source_dataset", "unknown"), **prediction})
    measured = report_metrics(records)
    if metrics.get("samples") != len(raw) or metrics.get("inference_errors") != 0:
        raise ValueError("Incomplete selected-checkpoint holdout result")
    for source, target in (("Score", "score"), ("Macro-F1", "macro_f1"),
                           ("Conflict_accuracy", "conflict_accuracy"), ("Aligned_calibration", "aligned_calibration")):
        actual = metrics.get(source)
        expected = measured[target]
        if expected is None or not isinstance(actual, (int, float)) or not math.isfinite(actual) or abs(actual - expected) > 1e-9:
            raise ValueError("Reported holdout metric does not match its predictions: " + source)
    return records, measured, selected["checkpoint"]


def report_metrics(records):
    result = compute_competition_score(records)
    conflict = [r for r in records if r["sample_type"] == "Conflict"]
    aligned = [r for r in records if r["sample_type"] == "Aligned"]
    natural = [r for r in conflict if r["source_dataset"] in NATURAL]
    synthetic = [r for r in conflict if r["source_dataset"] not in NATURAL]
    def recall(rows):
        tp = sum(r["is_uncertain"] == "yes" for r in rows)
        return {"true_positives": tp, "total": len(rows), "recall": tp / len(rows) if rows else None}
    result.update(natural_conflict=recall(natural), synthetic_conflict=recall(synthetic),
                  false_positives=sum(r["is_uncertain"] == "yes" for r in aligned),
                  emotion_correct=sum(r["pred_emotion"] == r["gt_label"] for r in records))
    return result


def paired_report(control, experiment):
    keys = ("sample_id", "gt_label", "sample_type", "protocol", "source_dataset")
    if len(control) != len(experiment) or any(any(a[k] != b[k] for k in keys) for a, b in zip(control, experiment)):
        raise ValueError("Paired metrics require identical holdout IDs and original labels")
    before, after = report_metrics(control), report_metrics(experiment)
    changes = {key: 0 for key in ("emotion_corrected", "emotion_lost", "conflicts_corrected", "conflicts_lost",
                                 "false_positives_added", "false_positives_removed", "joint_corrected", "joint_lost")}
    changed = []
    for a, b in zip(control, experiment):
        ae, be = a["pred_emotion"] == a["gt_label"], b["pred_emotion"] == b["gt_label"]
        expected = "yes" if a["sample_type"] == "Conflict" else "no"
        au, bu = a["is_uncertain"] == expected, b["is_uncertain"] == expected
        changes["emotion_corrected"] += int(not ae and be)
        changes["emotion_lost"] += int(ae and not be)
        changes["joint_corrected"] += int(not (ae and au) and be and bu)
        changes["joint_lost"] += int(ae and au and not (be and bu))
        if expected == "yes":
            changes["conflicts_corrected"] += int(not au and bu)
            changes["conflicts_lost"] += int(au and not bu)
        else:
            changes["false_positives_added"] += int(au and not bu)
            changes["false_positives_removed"] += int(not au and bu)
        if (a["pred_emotion"], a["is_uncertain"]) != (b["pred_emotion"], b["is_uncertain"]):
            changed.append({"sample_id": a["sample_id"], "gt_label": a["gt_label"], "sample_type": a["sample_type"],
                            "control": {k: a[k] for k in ("pred_emotion", "is_uncertain")},
                            "experiment": {k: b[k] for k in ("pred_emotion", "is_uncertain")}})
    checks = {
        "score_gain_at_least_0_01": before["score"] is not None and after["score"] is not None and after["score"] - before["score"] >= .01 - 1e-12,
        "macro_f1_drop_at_most_0_005": after["macro_f1"] >= before["macro_f1"] - .005 - 1e-12,
        "natural_conflict_tp_does_not_regress": before["natural_conflict"]["total"] > 0 and after["natural_conflict"]["true_positives"] >= before["natural_conflict"]["true_positives"],
    }
    by_protocol = {p: {"control": report_metrics([r for r in control if r["protocol"] == p]),
                       "experiment": report_metrics([r for r in experiment if r["protocol"] == p])}
                   for p in sorted({r["protocol"] for r in control})}
    return {"control": before, "experiment": after, "by_protocol": by_protocol, "changes": changes,
            "changed_predictions": changed, "gate": {"passed": all(checks.values()), "checks": checks,
                "score_gain": after["score"] - before["score"] if before["score"] is not None and after["score"] is not None else None,
                "macro_f1_delta": after["macro_f1"] - before["macro_f1"]}}


def markdown(summary):
    lines = ["# Protocol SFT paired holdout summary", "", "Only original-training-derived holdout rows enter this comparison.", "",
             "| Arm | Score | Macro-F1 | Natural TP | Synthetic TP | Aligned FP |", "|---|---:|---:|---:|---:|---:|"]
    for arm in ("control", "experiment"):
        m = summary[arm]
        lines.append(f"| {arm} | {m['score']:.8f} | {m['macro_f1']:.8f} | {m['natural_conflict']['true_positives']}/{m['natural_conflict']['total']} | {m['synthetic_conflict']['true_positives']}/{m['synthetic_conflict']['total']} | {m['false_positives']} |")
    lines.extend(["", "Gate: **" + ("passed" if summary["gate"]["passed"] else "failed") + "**.",
                  "Next action: `" + summary["next_action"] + "`.", "",
                  "No official validation/test predictions or Kaggle scores are used by this gate.",
                  "Two successful seed comparisons are required before considering full-data training.", ""])
    return "\n".join(lines)


def summarize(run_dir, output_dir=None):
    root = Path(run_dir).resolve()
    output = Path(output_dir).resolve() if output_dir else root
    identity_path, manifest_path = root / "identity.json", root / "prepare-manifest.json"
    identity, manifest = read(identity_path), read(manifest_path)
    if identity["identity_hash"] != fingerprint(identity["identity"]) or manifest.get("identity_hash") != identity["identity_hash"]:
        raise ValueError("Prepared run identity fingerprint changed")
    if identity["identity"].get("algorithm") != VERSION or manifest.get("algorithm") != VERSION:
        raise ValueError("Not a protocol SFT ablation run")
    if manifest.get("official_val_read") is not False or manifest.get("official_test_read") is not False:
        raise ValueError("Only training-derived partitions can enter this gate")
    paths = (root / "split-manifest.json", root / "raw/holdout.jsonl")
    hashes = {identity_path: file_hash(identity_path), manifest_path: file_hash(manifest_path)}
    # Preparation writes native relative paths. Normalize their separators so
    # artifacts prepared on Windows remain readable after server deployment.
    artifacts = {}
    for relative, value in manifest["artifacts"].items():
        normalized = relative.replace("\\", "/")
        if normalized in artifacts and artifacts[normalized] != value:
            raise ValueError("Conflicting prepared artifact path aliases")
        artifacts[normalized] = value
    for path in paths:
        expected = artifacts.get(path.relative_to(root).as_posix())
        if expected != file_hash(path):
            raise ValueError("Prepared holdout artifact checksum changed: " + str(path))
        hashes[path] = expected
    split, raw = read(paths[0]), read_rows(paths[1])
    if fingerprint(split["design"]) != identity["identity"]["split_hash"] or split["identity_hash"] != identity["identity_hash"]:
        raise ValueError("Holdout split identity changed")
    if not raw or any(r.get("split") != "train" for r in raw) or len(raw) != split["partitions"]["holdout"]["rows"]:
        raise ValueError("Holdout must contain only the original train split")
    control, _, control_checkpoint = checked_arm(root, "control", identity["identity_hash"], raw, hashes)
    experiment, _, experiment_checkpoint = checked_arm(root, "experiment", identity["identity_hash"], raw, hashes)
    result = paired_report(control, experiment)
    seed = identity["identity"]["training_seed"]
    result.update(algorithm=VERSION, identity_hash=identity["identity_hash"], training_seed=seed,
        checkpoints={"control": control_checkpoint, "experiment": experiment_checkpoint},
        official_val_read=False, official_test_read=False, submitted_to_kaggle=False,
        advance_to_seed43=bool(seed == 42 and result["gate"]["passed"]),
        next_action=("run_seed43_replication" if seed == 42 else "require_both_seed_gates_before_full_training") if result["gate"]["passed"] else "stop_protocol_promotion",
        input_sha256={path.relative_to(root).as_posix(): value for path, value in hashes.items()})
    payloads = {output / "pair-summary.json": json_bytes(result), output / "pair-summary.md": markdown(result).encode("utf-8")}
    if any(file_hash(path) != value for path, value in hashes.items()):
        raise ValueError("Holdout inputs changed during paired summary")
    for path, content in payloads.items():
        if path.exists() and path.read_bytes() != content:
            raise ValueError("Refusing to overwrite a different paired summary")
    for path, content in payloads.items():
        write_immutable(path, content)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    result = summarize(args.run_dir, args.output_dir)
    print(json.dumps({"gate": result["gate"], "advance_to_seed43": result["advance_to_seed43"],
                      "next_action": result["next_action"]}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
