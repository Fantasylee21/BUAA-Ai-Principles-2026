"""Audit a frozen evidence candidate or export one fixed, resumable supplement.

The audit command requires only NumPy. GPU/model dependencies are imported only
by export. Neither command fits a model, changes the producer run, or submits to
Kaggle. Run as ``python -m scripts.run_evidence_supplement audit|export ...``.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from rl.experiment import ROOT, artifact_hash, file_hash, fingerprint, write_json
from rl.grounded_store import source_hash_at
from rl.metrics import evaluation_report
from rl.reward import VALID_EMOTIONS, parse_completion, question_is_consistent
from scripts.evidence_supplement import (VERSION, export_submission,
                                         supplement_decisions, write_immutable)

PRODUCER_VERSION = "omni_evidence_bottleneck_v1"
MODES = ("vision", "text", "audio")
POSTHOC_NOTE = (
    "This rule was chosen after examining validation results. This is a post-hoc "
    "diagnostic, not an independent estimate of generalization or Kaggle gain. "
    "Cached validation probabilities are checked algebraically; no model inference "
    "is repeated by this CPU audit. Fixed3 and union scores are separate diagnostics."
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def checked_context(source_run):
    source_run = Path(source_run)
    c = read(source_run / "context.json")
    if c["identity_hash"] != fingerprint(c["identity"]) or c["identity"]["algorithm"] != PRODUCER_VERSION:
        raise ValueError("Invalid frozen evidence context identity")
    if c["identity"]["source"] != source_hash_at(ROOT):
        raise ValueError("Producer rl source hash changed; old features cannot be reused")
    if c["identity"]["settings"] != c["settings"]:
        raise ValueError("Context settings differ from the frozen identity")
    if file_hash(source_run / "baseline_090813.csv") != c["identity"]["baseline_csv_sha256"]:
        raise ValueError("Original producer baseline hash changed")
    for split, expected in (("train", 3354), ("val", 666), ("test", 734)):
        if len(c[split]) != expected or len({r["sample_id"] for r in c[split]}) != expected:
            raise ValueError("Official context split count or IDs changed: " + split)
    return c


def view_examples(rows, mode):
    """Reconstruct the original producer signature without importing SciPy."""
    unique = {}
    for row in rows:
        if mode not in row["view_targets"]:
            continue
        target = row["view_targets"][mode]
        key = row["video"] if mode == "vision" else row["sample_id"]
        if key in unique and unique[key][1] != target:
            raise ValueError("Inconsistent visual supervision in frozen context")
        unique.setdefault(key, (row, target))
    return [(r, VALID_EMOTIONS.index(t["label"])) for r, t in unique.values() if t["clear"]]


def checked_fit(path, c, signature):
    path = Path(path)
    marker, value = read(str(path) + ".complete.json"), read(path)
    if (marker.get("algorithm"), marker.get("identity_hash"), marker.get("signature"), marker.get("sha256")) != (
            PRODUCER_VERSION, c["identity_hash"], signature, file_hash(path)):
        raise ValueError("Frozen fit marker/signature/checksum changed: " + str(path))
    if value.get("identity_hash") != c["identity_hash"] or value.get("signature") != signature or value["solver"].get("success") is not True:
        raise ValueError("Incomplete or mismatched frozen fit: " + str(path))
    return value


def calibrated_scores(q, rows, calibration):
    q = np.asarray(q, dtype=np.float64)
    offsets = np.asarray(calibration["offsets"], dtype=np.float64)
    if q.shape != (len(rows),) or not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
        raise ValueError("Invalid frozen disagreement probabilities")
    if offsets.shape != (2,) or not np.isfinite(offsets).all():
        raise ValueError("Invalid frozen calibration offsets")
    if any(r["protocol"] not in ("VT", "VA") for r in rows):
        raise ValueError("Invalid frozen protocol")
    clipped = np.clip(q, 1e-15, 1 - 1e-15)
    odds = np.log(clipped) - np.log1p(-clipped)
    odds += offsets[[int(r["protocol"] == "VA") for r in rows]]
    result = np.exp(-np.logaddexp(0., -odds))
    return np.where(q == 0, 0., np.where(q == 1, 1., result))


def checked_candidate(source_run, c):
    source_run = Path(source_run)
    candidate = read(source_run / "results/evidence-seed42.json")
    if (candidate.get("identity_hash"), candidate.get("seed"), candidate.get("method")) != (
            c["identity_hash"], 42, "evidence_bottleneck"):
        raise ValueError("Candidate is not the frozen seed 42 evidence model")
    fold_hash = c["designs"]["42"]["hash"]
    if candidate.get("fold_hash") != fold_hash or c["identity"]["fold_designs"]["42"] != fold_hash:
        raise ValueError("Candidate fold identity changed")
    hashes = {}
    for mode in MODES:
        examples = view_examples(c["train"], mode)
        signature = fingerprint([c["identity_hash"], mode, [(r["sample_id"], y) for r, y in examples]])
        path = source_run / "heads/final-42" / (mode + ".json")
        head = checked_fit(path, c, signature)
        if head != candidate["heads"][mode] or head.get("mode") != mode:
            raise ValueError("Candidate head differs from its completed artifact: " + mode)
        weights = np.asarray(head["parameters"], dtype=np.float64)
        if weights.shape != (3, 2049) or not np.isfinite(weights).all():
            raise ValueError("Invalid complete evidence parameters: " + mode)
        if head.get("examples") != [r["sample_id"] for r, _ in examples] or head.get("labels") != [y for _, y in examples]:
            raise ValueError("Evidence supervision list changed: " + mode)
        hashes["head_" + mode] = file_hash(path)
        hashes["head_" + mode + "_marker"] = file_hash(str(path) + ".complete.json")
    oof_path = source_run / "calibration-oof-42.json"
    oof = read(oof_path)
    ids = [r["sample_id"] for r in c["train"]]
    if (oof.get("identity_hash"), oof.get("fold_hash"), oof.get("ids")) != (c["identity_hash"], fold_hash, ids):
        raise ValueError("Frozen calibration OOF identity/order changed")
    calibrated_scores(oof["q"], c["train"], {"offsets": [0., 0.]})
    signature = fingerprint([c["identity_hash"],
        [(r["sample_id"], r["sample_type"], r["protocol"]) for r in c["train"]],
        np.asarray(oof["q"]).tolist()])
    path = source_run / "calibration/final-42.json"
    calibration = checked_fit(path, c, signature)
    if calibration != candidate["calibration"] or calibration.get("examples") != ids:
        raise ValueError("Candidate calibration differs from the completed fit")
    calibrated_scores(oof["q"], c["train"], calibration)
    hashes.update(candidate=file_hash(source_run / "results/evidence-seed42.json"),
                  calibration=file_hash(path), calibration_marker=file_hash(str(path) + ".complete.json"),
                  calibration_oof=file_hash(oof_path), context=file_hash(source_run / "context.json"))
    return candidate, hashes


def validation_audit(c, candidate, hashes):
    rows = c["val"]
    baseline = {sid: {key: prediction[key] for key in
                ("pred_emotion", "is_uncertain", "clarifying_question")}
                for sid, prediction in c["baselines"]["val"].items()}
    a = np.asarray(candidate["vision_probabilities"], dtype=np.float64)
    b = np.asarray(candidate["second_probabilities"], dtype=np.float64)
    decisions = supplement_decisions(rows, baseline, a, b, candidate["val_probabilities"])
    q = np.clip(1 - (a * b).sum(axis=1), 0., 1.)
    if not np.allclose(q, candidate["raw_disagreement"], atol=1e-12, rtol=0):
        raise ValueError("Cached validation disagreement does not match its evidence")
    scores = calibrated_scores(q, rows, candidate["calibration"])
    if not np.allclose(scores, candidate["val_probabilities"], atol=1e-12, rtol=0):
        raise ValueError("Cached validation calibration does not match its evidence")
    before, after, replacement = [], [], []
    changes = []
    for row, decision, score in zip(rows, decisions, scores):
        old = baseline[row["sample_id"]]
        record = {**row, **old, "sample_id": row["sample_id"]}
        before.append(record)
        after.append({**record, "is_uncertain": decision["is_uncertain"]})
        replacement.append({**record, "is_uncertain": "yes" if score > .5 else "no"})
        if decision["added_yes"]:
            changes.append({**decision, "sample_type": row["sample_type"],
                            "natural": row["source_dataset"] in ("ch_sims", "ch_sims_v2")})

    def report(records):
        value = evaluation_report(records)
        # This audit verifies only fixed decisions, without completing questions.
        value.update(invalid_outputs=None, inconsistent_questions=None, output_checks_pending=True)
        conflict = [r for r in records if r["sample_type"] == "Conflict"]
        natural = [r for r in conflict if r["source_dataset"] in ("ch_sims", "ch_sims_v2")]
        value["counts"] = {
            "conflict_tp": sum(r["is_uncertain"] == "yes" for r in conflict),
            "aligned_fp": sum(r["is_uncertain"] == "yes" for r in records if r["sample_type"] == "Aligned"),
            "natural_conflict_tp": sum(r["is_uncertain"] == "yes" for r in natural),
            "natural_conflicts": len(natural),
            "synthetic_conflict_tp": sum(r["is_uncertain"] == "yes" for r in conflict if r not in natural),
            "emotion_correct": sum(r["gt_label"] == r["pred_emotion"] for r in records)}
        return value

    base, new = report(before), report(after)
    return {"algorithm": VERSION, "producer_identity": c["identity_hash"], "verified_artifacts": hashes,
            "threshold": .5, "posthoc": True, "independent_validation": False, "note": POSTHOC_NOTE,
            "baseline": base, "evidence_replacement": report(replacement), "supplement": new,
            "score_delta": new["score"] - base["score"], "changes": changes,
            "emotion_changes": 0, "removed_yes": 0, "trained": False, "submitted_to_kaggle": False}


def verify_export_inputs(c):
    """Verify raw rows and actual model artifacts before any GPU allocation."""
    from rl.grounded_uncertainty import load_training
    from rl.experiment import load_records
    for split, name in (("train", "train.jsonl"), ("val", "val.jsonl"), ("test", "test_release.jsonl")):
        path = Path(c["data_root"]) / name
        if file_hash(path) != c["identity"]["raw_split_hashes"][split]:
            raise ValueError("Frozen raw split checksum changed: " + split)
        if split == "test":
            rows, duplicates = load_records(path, require_labels=False)
            if duplicates:
                raise ValueError("Duplicated raw TEST IDs")
        else:
            rows = load_training(path)
        if rows != c[split]:
            raise ValueError("Context rows differ from the original raw split: " + split)
    if artifact_hash(c["model"]) != c["identity"]["base"]:
        raise ValueError("Frozen base weights changed")
    if artifact_hash(c["adapter"], adapter=True) != c["identity"]["sft_for_output_only"]:
        raise ValueError("Frozen v4 adapter changed")
    parent = Path(c["baseline_context"])
    if file_hash(parent) != c["identity"]["baseline_context_sha256"]:
        raise ValueError("Producer baseline context changed")
    previous = read(parent)
    if previous["identity_hash"] != fingerprint(previous["identity"]) or previous["identity_hash"] != c["identity"]["baseline_identity"]:
        raise ValueError("Producer baseline context identity changed")
    return verify_question_source(c)


def verify_question_source(c):
    source = Path(c["source_run"])
    old = read(source / "context.json")
    if old["identity_hash"] != fingerprint(old["identity"]) or old["identity_hash"] != c["identity"]["question_source_identity"]:
        raise ValueError("Frozen v4 question source identity changed")
    manifest_path = source / "cache/complete.json"
    if file_hash(manifest_path) != c["identity"]["question_source_manifest_sha256"]:
        raise ValueError("Frozen v4 question cache manifest changed")
    if read(manifest_path).get("identity_hash") != old["identity_hash"]:
        raise ValueError("Question cache manifest belongs to another model")
    return {"context_sha256": file_hash(source / "context.json"), "manifest_sha256": file_hash(manifest_path),
            "identity_hash": old["identity_hash"], "test_cache_loaded": False}


def freeze_baseline(output_dir, source, rows, expected_digest):
    """Parse one confirmed snapshot and retain its exact bytes for long GPU work."""
    from scripts.evidence_supplement import read_baseline_snapshot
    baseline, digest, confirmed = read_baseline_snapshot(source, rows)
    content = Path(source).read_bytes()
    import hashlib
    if digest != expected_digest or hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("Source baseline changed before its immutable snapshot")
    target = Path(output_dir) / "baseline-frozen.csv"
    if target.resolve() == Path(source).resolve():
        raise ValueError("Source baseline cannot be the run's snapshot destination")
    write_immutable(target, content)
    return baseline, target, confirmed


@contextmanager
def run_lock(output_dir):
    path = Path(output_dir) / ".controller.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            if not stream.read(1):
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        # Closing releases either platform's advisory lock, including on errors.
        stream.close()


def emit(output_dir, phase, **values):
    state = {"phase": phase, "pid": os.getpid(), "updated": time.time(), **values}
    write_json(Path(output_dir) / "status.json", state)
    print(json.dumps(state, ensure_ascii=False, allow_nan=False), flush=True)


def run_identity(args, c, hashes):
    identity = {"algorithm": VERSION, "command": args.command,
                "producer_identity": c["identity_hash"], "artifacts": hashes,
                "baseline_sha256": file_hash(args.baseline_csv) if args.baseline_csv else None,
                "runner_sha256": file_hash(__file__),
                "supplement_sha256": file_hash(Path(__file__).with_name("evidence_supplement.py")),
                "threshold": .5, "train_models": False}
    value = {"identity": identity, "identity_hash": fingerprint(identity)}
    write_immutable(Path(args.output_dir) / "run-identity.json", json_bytes(value))
    return value


def extract_test(args, c, torch):
    from rl.evidence_runtime import BaseEvidenceExtractor
    from rl.evidence_bottleneck import tasks_for
    from rl.evidence_store import feature_entry, save_feature, publish
    from rl.train_evidence_bottleneck import verify_media
    tasks = tasks_for(c["test"], "test")
    checked = set()
    policy = None
    frozen = None
    try:
        for i, task in enumerate(tasks, 1):
            # Validate media even for reused features; a saved tensor is not proof
            # that the currently supplied file still has its original contents.
            verify_media(c, task["row"], task["mode"], checked)
            reused = feature_entry(args.output_dir, c, task) is not None
            if not reused:
                if policy is None:
                    policy = BaseEvidenceExtractor(c["model"], c["settings"])
                    frozen = policy.check_frozen()
                hidden, native, seconds = policy.extract(task["row"], c["data_root"], task["mode"])
                save_feature(args.output_dir, c, task, hidden, native, seconds)
            emit(args.output_dir, "extracting_test", completed=i, total=len(tasks),
                 sample_id=task["row"]["sample_id"], mode=task["mode"], reused=reused)
        if policy is not None and policy.check_frozen() != frozen:
            raise RuntimeError("Base model changed during read-only extraction")
        publish(args.output_dir, c, ("test",))
    finally:
        # Release the entire base model before a v4 question model can be loaded.
        del policy
        gc.collect()
        torch.cuda.empty_cache()
    emit(args.output_dir, "base_released", completed=len(tasks), total=len(tasks))


class QuestionCache:
    def __init__(self, output_dir, c, identity, baseline_rows, model_factory, emit_fn):
        self.root, self.c, self.identity = Path(output_dir), c, identity
        self.baseline = {r["sample_id"]: r for r in baseline_rows}
        self.factory, self.emit = model_factory, emit_fn
        self.policy, self.frozen, self.completed = None, None, 0

    @staticmethod
    def frozen_state(policy):
        return {n: (int(p._version), bool(p.requires_grad)) for n, p in policy.model.named_parameters()}

    def __call__(self, row, decision):
        allowed = ("sample_id", "protocol", "video", "text_content", "text")
        row = {key: row[key] for key in allowed if key in row}
        if row["protocol"] == "VA":
            row = {key: value for key, value in row.items() if key not in ("text_content", "text")}
        baseline = self.baseline[row["sample_id"]]
        signature = fingerprint([self.identity["identity_hash"], row, baseline, decision])
        path = self.root / "questions" / (fingerprint(row["sample_id"]) + ".json")
        if path.exists():
            saved = read(path)
            if saved.get("identity_hash") != self.identity["identity_hash"] or saved.get("signature") != signature:
                raise ValueError("Question resume cache belongs to different evidence/baseline")
            answer = saved.get("answer")
            reused = True
        else:
            if self.policy is None:
                self.policy = self.factory()
                self.frozen = self.frozen_state(self.policy)
                if any(trainable for _, trainable in self.frozen.values()):
                    raise ValueError("Question model unexpectedly has trainable parameters")
            answer = parse_completion(self.policy.answer(row, "test", self.c, None, baseline, "yes"))
            reused = False
        if not isinstance(answer, dict) or parse_completion(json.dumps(answer)) is None or not question_is_consistent(answer):
            raise ValueError("Invalid question output/cache")
        if answer["pred_emotion"] != baseline["pred_emotion"] or answer["is_uncertain"] != "yes":
            raise ValueError("Question generation changed a frozen field")
        if not reused:
            if self.frozen_state(self.policy) != self.frozen:
                raise RuntimeError("Frozen v4 changed during question generation")
            write_immutable(path, json_bytes({"identity_hash": self.identity["identity_hash"],
                                             "signature": signature, "sample_id": row["sample_id"], "answer": answer}))
        self.completed += 1
        self.emit("generating_questions", completed=self.completed, sample_id=row["sample_id"], reused=reused)
        return answer["clarifying_question"]

    def release(self, torch):
        self.policy = None
        gc.collect()
        torch.cuda.empty_cache()


def export(args, c, candidate, hashes, identity):
    # The old producer context is read verbatim and remains its own identity.
    from rl.evidence_store import context_for, FeatureBank
    from rl.evidence_training import predict_evidence
    from rl.evidence_bottleneck import decision_scores
    from rl.train_evidence_bottleneck import torch_device
    if context_for(args.source_run) != c:
        raise ValueError("Producer context changed during export initialization")
    from rl.state import software_versions
    import importlib.metadata
    software = {**software_versions(), "scipy": importlib.metadata.version("scipy")}
    if software != c["identity"]["software"]:
        raise ValueError("Actual software differs from the frozen feature producer; cannot reuse its identity: "
                         + json.dumps({"actual": software, "producer": c["identity"]["software"]}, sort_keys=True))
    question_source = verify_export_inputs(c)
    baseline, frozen_baseline, confirmed = freeze_baseline(args.output_dir, args.baseline_csv,
        c["test"], identity["identity"]["baseline_sha256"])
    emit(args.output_dir, "inputs_verified", rows=len(baseline), producer_identity=c["identity_hash"])
    torch = torch_device(args)
    extract_test(args, c, torch)
    evidence = predict_evidence(FeatureBank(args.output_dir, c, "test"), c["test"], candidate["heads"])
    scores = decision_scores(evidence["q"], c["test"], candidate["calibration"])
    decisions = supplement_decisions(c["test"], {r["sample_id"]: r for r in baseline},
                                      evidence["vision"], evidence["second"], scores)
    total_new = sum(d["added_yes"] for d in decisions)
    from scripts.evidence_supplement import read_baseline_csv
    original_v4 = {r["sample_id"]: r for r in read_baseline_csv(
        Path(args.source_run) / "baseline_090813.csv", c["test"])}
    changed_old_yes = [{**d, "old_v4_uncertain": "yes",
                        "restores_old_v4_yes": d["is_uncertain"] == "yes"}
                       for d in decisions if original_v4[d["sample_id"]]["is_uncertain"] == "yes"
                       and d["baseline_uncertain"] == "no"]
    review = {"run_identity": identity["identity_hash"], "baseline": confirmed,
              "baseline_sha256": identity["identity"]["baseline_sha256"],
              "old_v4_yes_removed_in_current_baseline": len(changed_old_yes),
              "restored_old_v4_yes": sum(d["restores_old_v4_yes"] for d in changed_old_yes),
              "old_v4_yes_review": changed_old_yes, "added_yes": total_new,
              "note": "The fixed supplement can restore older v4 yes decisions that the newer public baseline removed. Review these rows explicitly; this is not evidence that restoring them improves Kaggle."}
    write_immutable(Path(args.output_dir) / "supplement-review.json", json_bytes(review))
    emit(args.output_dir, "evidence_complete", rows=734, added_yes=total_new)

    def model_factory():
        from rl.relation_runtime import RelationExtractor
        verify_question_source(c)
        return RelationExtractor(c["model"], c["adapter"], c["settings"])

    questions = QuestionCache(args.output_dir, c, identity, baseline, model_factory,
        lambda phase, **v: emit(args.output_dir, phase, total=total_new, **v))

    def checked_question(row, decision):
        if file_hash(args.baseline_csv) != identity["identity"]["baseline_sha256"]:
            raise ValueError("Original baseline changed during inference; export cancelled")
        result = questions(row, decision)
        if file_hash(args.baseline_csv) != identity["identity"]["baseline_sha256"]:
            raise ValueError("Original baseline changed during question generation; export cancelled")
        return result

    try:
        provenance = {"run_identity": identity["identity_hash"], "producer_identity": c["identity_hash"],
                      "source_run": str(Path(args.source_run).resolve()), "artifacts": hashes,
                      "question_source": question_source, "posthoc": True, "note": POSTHOC_NOTE,
                      "original_baseline_file": str(Path(args.baseline_csv).resolve()),
                      "original_baseline": confirmed,
                      "expected_baseline_sha256": identity["identity"]["baseline_sha256"],
                      "actual_software": software, "producer_software": c["identity"]["software"],
                      "test_rows_extracted": 734, "training_performed": False,
                      "validation_audit_sha256": file_hash(Path(args.output_dir) / "validation-audit.json")}
        if file_hash(args.baseline_csv) != identity["identity"]["baseline_sha256"]:
            raise ValueError("Original baseline changed during extraction; export cancelled")
        manifest = export_submission(args.output_dir, c["test"], frozen_baseline,
            evidence["vision"], evidence["second"], scores, checked_question, provenance)
        if manifest["baseline_sha256"] != identity["identity"]["baseline_sha256"]:
            raise RuntimeError("Exported baseline digest differs from its frozen run identity")
    finally:
        questions.release(torch)
    emit(args.output_dir, "complete", rows=734, added_yes=manifest["added_yes"],
         submission=manifest["file"], sha256=manifest["sha256"], submitted_to_kaggle=False)
    return manifest


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("command", choices=("audit", "export"))
    value.add_argument("--source-run", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--baseline-csv", type=Path)
    value.add_argument("--gpu", default="1")
    value.add_argument("--memory-gib", type=float, default=32.)
    value.add_argument("--cpu-threads", type=int, default=4)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "export" and args.baseline_csv is None:
        raise ValueError("Export requires --baseline-csv")
    if not 0 < args.memory_gib <= 32 or args.cpu_threads < 1:
        raise ValueError("Memory limit must be positive and at most 32 GiB; CPU threads must be positive")
    args.source_run, args.output_dir = args.source_run.resolve(), args.output_dir.resolve()
    if args.output_dir.is_relative_to(args.source_run):
        raise ValueError("Output must be outside the immutable producer run")
    with run_lock(args.output_dir):
        try:
            emit(args.output_dir, "verifying_source", command=args.command)
            c = checked_context(args.source_run)
            candidate, hashes = checked_candidate(args.source_run, c)
            identity = run_identity(args, c, hashes)
            audit = validation_audit(c, candidate, hashes)
            write_immutable(args.output_dir / "validation-audit.json", json_bytes(audit))
            if args.command == "audit":
                emit(args.output_dir, "complete", command="audit", score=audit["supplement"]["score"],
                     counts=audit["supplement"]["counts"], posthoc=True)
                return audit
            return export(args, c, candidate, hashes, identity)
        except Exception as error:
            detail = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
            write_json(args.output_dir / "error.json", detail)
            emit(args.output_dir, "failed", error=detail["message"])
            raise


if __name__ == "__main__":
    main()
