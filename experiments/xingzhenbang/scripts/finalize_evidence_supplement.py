"""Replace only newly added questions with neutral, user-facing templates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile

import numpy as np

from rl.experiment import file_hash, fingerprint, load_records
from rl.reward import VALID_EMOTIONS
from scripts.evidence_supplement import (FIELDS, VERSION, export_submission,
    read_baseline_snapshot, supplement_decisions, validate_submission, write_immutable)

POLICY = "neutral_modality_template_v1"
QUESTIONS = {
    "VT": "你能补充说明，画面中的表现和这段话分别给你怎样的感受吗？",
    "VA": "你能补充说明，画面中的表现和声音分别给你怎样的感受吗？",
}
ARTIFACTS = ("submission.csv", "submission-changes.json", "submission-evidence.json")


def read_json(content):
    return json.loads(content.decode("utf-8"))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def digest(content):
    return hashlib.sha256(content).hexdigest()


def csv_rows(content):
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
    if reader.fieldnames != list(FIELDS):
        raise ValueError("Raw CSV must use the four official columns")
    return list(reader)


def neutral_question(row, decision):
    if not decision["added_yes"]:
        raise ValueError("Only added yes questions may be replaced")
    return QUESTIONS[row["protocol"]]


def finalize(input_dir, baseline_csv, test_jsonl, output_dir):
    root, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    baseline_path, test_path = Path(baseline_csv).resolve(), Path(test_jsonl).resolve()
    if root.is_relative_to(output) or output.is_relative_to(root):
        raise ValueError("Reviewed output must be separate from the raw input directory and its ancestors")
    source_paths = [root / name for name in (*ARTIFACTS, "submission-manifest.json",
                    "run-identity.json", "baseline-frozen.csv")]
    source_paths.extend((baseline_path, test_path))
    # Resolve all protected files before choosing any publication paths.
    protected = {p.resolve() for p in source_paths}
    final_paths = {name: output / name for name in (*ARTIFACTS, "submission-manifest.json")}
    if any(p.resolve() in protected for p in final_paths.values()):
        raise ValueError("Reviewed artifacts may not alias a protected input")
    snapshots = {p: p.read_bytes() for p in source_paths}
    hashes = {p: digest(content) for p, content in snapshots.items()}

    def unchanged():
        if any(file_hash(path) != expected for path, expected in hashes.items()):
            raise ValueError("Raw source, baseline, or test data changed; finalization cancelled")

    manifest = read_json(snapshots[root / "submission-manifest.json"])
    if manifest.get("algorithm") != VERSION or manifest.get("threshold") != .5 or manifest.get("rows") != 734:
        raise ValueError("Raw manifest has an unsupported algorithm/threshold/row count")
    if set(manifest.get("artifact_sha256", {})) != set(ARTIFACTS):
        raise ValueError("Raw manifest must identify all three expected artifacts")
    for name in ARTIFACTS:
        if manifest["artifact_sha256"][name] != hashes[root / name]:
            raise ValueError("Raw artifact checksum failed: " + name)
    if manifest.get("sha256") != hashes[root / "submission.csv"]:
        raise ValueError("Raw submission SHA differs from its manifest")
    run = read_json(snapshots[root / "run-identity.json"])
    if run.get("identity_hash") != fingerprint(run["identity"]):
        raise ValueError("Raw run identity fingerprint is corrupt")
    identity, provenance = run["identity"], manifest.get("provenance", {})
    if (identity.get("algorithm"), identity.get("command"), identity.get("threshold"), identity.get("train_models")) != (VERSION, "export", .5, False):
        raise ValueError("Raw run is not a fixed inference-only supplement")
    if (provenance.get("run_identity") != run["identity_hash"] or
            provenance.get("producer_identity") != identity.get("producer_identity") or
            provenance.get("artifacts") != identity.get("artifacts")):
        raise ValueError("Raw manifest provenance is not linked to its run identity")
    expected_baseline = manifest.get("baseline_sha256")
    if any(value != expected_baseline for value in (
            identity.get("baseline_sha256"), provenance.get("expected_baseline_sha256"),
            hashes[root / "baseline-frozen.csv"], hashes[baseline_path])):
        raise ValueError("Raw manifest, run, frozen snapshot, and requested baseline disagree")
    rows, duplicates = load_records(test_path, require_labels=False)
    if duplicates or len(rows) != 734:
        raise ValueError("Official TEST must have 734 unique rows")
    baseline, baseline_hash, _ = read_baseline_snapshot(baseline_path, rows)
    if baseline_hash != expected_baseline:
        raise ValueError("Baseline changed while parsing its confirmed snapshot")
    evidence = read_json(snapshots[root / "submission-evidence.json"])
    if (evidence.get("algorithm"), evidence.get("threshold"), evidence.get("emotion_order")) != (VERSION, .5, list(VALID_EMOTIONS)):
        raise ValueError("Raw evidence contract changed")
    decisions = evidence.get("decisions", [])
    if len(decisions) != 734 or [d["sample_id"] for d in decisions] != [r["sample_id"] for r in rows]:
        raise ValueError("Raw evidence IDs/order differ from official TEST")
    vision = np.asarray([d["vision"] for d in decisions], dtype=np.float64)
    second = np.asarray([d["second"] for d in decisions], dtype=np.float64)
    scores = np.asarray([d["calibrated_probability"] for d in decisions], dtype=np.float64)
    recalculated = supplement_decisions(rows, {r["sample_id"]: r for r in baseline}, vision, second, scores)
    if recalculated != decisions:
        raise ValueError("Raw evidence decisions do not reproduce the fixed supplement")
    raw_rows = csv_rows(snapshots[root / "submission.csv"])
    validate_submission(rows, baseline, raw_rows, recalculated)
    added_ids = [d["sample_id"] for d in decisions if d["added_yes"]]
    if manifest.get("changed_sample_ids") != added_ids or manifest.get("added_yes") != len(added_ids):
        raise ValueError("Raw manifest added yes list differs from its decisions")
    changes = read_json(snapshots[root / "submission-changes.json"])
    if changes.get("algorithm") != VERSION or changes.get("baseline_sha256") != baseline_hash:
        raise ValueError("Raw change audit is linked to a different baseline")
    if [c["sample_id"] for c in changes.get("changes", [])] != added_ids:
        raise ValueError("Raw change audit does not cover the same added yes rows")
    new_provenance = {
        "question_policy": POLICY, "raw_manifest_sha256": hashes[root / "submission-manifest.json"],
        "raw_csv_sha256": hashes[root / "submission.csv"], "raw_run_identity": run["identity_hash"],
        "finalizer_source_sha256": file_hash(__file__), "test_sha256": hashes[test_path],
        "expected_baseline_sha256": baseline_hash, "raw_input_dir": str(root),
        "note": "Only questions on added yes rows are replaced. Emotion and uncertainty decisions match the raw candidate exactly. No score improvement is claimed for this wording change.",
    }
    # Stage using the existing exporter, then verify every invariant and source
    # hash before creating any of the four published reviewed artifacts.
    unchanged()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="supplement-review-", dir=output.parent) as tmp:
        staged = Path(tmp)
        def question(row, decision):
            unchanged()
            return neutral_question(row, decision)
        reviewed = export_submission(staged, rows, baseline_path, vision, second, scores,
                                     question, new_provenance)
        final_rows = csv_rows((staged / "submission.csv").read_bytes())
        validate_submission(rows, baseline, final_rows, recalculated)
        for raw, final, old, decision in zip(raw_rows, final_rows, baseline, decisions):
            if any(raw[key] != final[key] for key in ("sample_id", "pred_emotion", "is_uncertain")):
                raise ValueError("Finalization changed an emotion or uncertainty decision")
            if decision["added_yes"]:
                if final["clarifying_question"] != QUESTIONS[decision["protocol"]]:
                    raise ValueError("Finalized added yes does not use the neutral question policy")
            elif final != raw or final != old:
                raise ValueError("Finalization changed an existing baseline answer")
        payloads = {name: (staged / name).read_bytes() for name in ARTIFACTS}
        reviewed["file"] = str(final_paths["submission.csv"])
        payloads["submission-manifest.json"] = json_bytes(reviewed)
        unchanged()
        # Preflight the whole group before the first immutable write.
        for name, content in payloads.items():
            path = final_paths[name]
            if path.resolve() in protected:
                raise ValueError("Output aliases a protected raw input")
            if path.exists() and (not path.is_file() or path.read_bytes() != content):
                raise ValueError("Refusing to overwrite a different reviewed artifact: " + str(path))
        unchanged()
        for name, content in payloads.items():
            write_immutable(final_paths[name], content)
        for name, content in payloads.items():
            if final_paths[name].read_bytes() != content:
                raise RuntimeError("Published reviewed artifact differs from its validated bytes: " + name)
        if csv_rows(final_paths["submission.csv"].read_bytes()) != final_rows:
            raise RuntimeError("Published reviewed predictions differ from the validated rows")
    return reviewed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = finalize(args.input_dir, args.baseline_csv, args.test_jsonl, args.output_dir)
    print(json.dumps({"file": manifest["file"], "sha256": manifest["sha256"],
                      "rows": manifest["rows"], "questions_replaced": manifest["added_yes"],
                      "question_policy": POLICY, "submitted_to_kaggle": False}, ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    main()
