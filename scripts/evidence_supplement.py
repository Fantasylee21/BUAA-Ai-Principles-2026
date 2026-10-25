"""Fixed, CPU-only evidence supplement for a confirmed public submission.

Keep this module outside ``rl``: historical evidence identities hash every
``rl/*.py`` file. No training, threshold search, or Kaggle upload occurs here.
Probability columns follow ``rl.reward.VALID_EMOTIONS`` in that exact order.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from rl.reward import VALID_EMOTIONS, VALID_UNCERTAIN, question_is_consistent

VERSION = "omni_evidence_supplement_v1"
THRESHOLD = .5
TEST_ROWS = 734
FIELDS = ("sample_id", "pred_emotion", "is_uncertain", "clarifying_question")
BASELINE_SHA = "2446fcec5e105651157cd376d9a80ac5792489c41d8c2f2b70d710a93c4bdf23"
CONFIRMED_BASELINES = {
    BASELINE_SHA: {
        "name": "qwen25omni_qlora_sft_v4_resample2x_test_predictions.csv",
        "public_score": .90813,
    },
    "f2551b6a12fc916d804287671373bee5caa6e593315725a36ca2b1b0e20b8498": {
        "name": "submission_09.csv", "public_score": .92313,
    },
}


def _hash(content):
    return hashlib.sha256(content).hexdigest()


def _confirmed_baseline(digest):
    if digest in CONFIRMED_BASELINES:
        return CONFIRMED_BASELINES[digest]
    # Retain the legacy constant as an injection point for existing CPU tests.
    # In production it is the original confirmed hash, not a caller option.
    if digest == BASELINE_SHA:
        return {"name": "qwen25omni_qlora_sft_v4_resample2x_test_predictions.csv",
                "public_score": .90813}
    raise ValueError("Baseline CSV is not one of the user-confirmed public artifacts")


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _ids(rows):
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("Expected a nonempty ordered row sequence")
    ids = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Every input row must be an object")
        sid = row.get("sample_id")
        if not isinstance(sid, str) or not sid or sid != sid.strip():
            raise ValueError("Every sample needs a nonempty, unpadded string ID")
        if row.get("protocol") not in ("VT", "VA"):
            raise ValueError("Every sample needs a VT or VA protocol")
        ids.append(sid)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs are not allowed")
    return ids


def _prediction(value, sid):
    if not isinstance(value, Mapping):
        raise ValueError("Baseline predictions must be objects")
    expected = set(FIELDS) if "sample_id" in value else set(FIELDS[1:])
    if set(value) != expected:
        raise ValueError("Baseline prediction must have exactly the official fields")
    if "sample_id" in value and value["sample_id"] != sid:
        raise ValueError("Baseline key and sample ID differ")
    if (value.get("pred_emotion") not in VALID_EMOTIONS or
            value.get("is_uncertain") not in VALID_UNCERTAIN or
            not isinstance(value.get("clarifying_question"), str) or
            not question_is_consistent(value)):
        raise ValueError("Illegal baseline emotion, uncertainty, or question")
    return {"sample_id": sid, **{key: value[key] for key in FIELDS[1:]}}


def _baseline_rows(rows, baseline_rows):
    ids = _ids(rows)
    if not isinstance(baseline_rows, (list, tuple)) or len(baseline_rows) != len(ids):
        raise ValueError("Baseline must cover every input row exactly once")
    output = []
    for sid, baseline in zip(ids, baseline_rows):
        if not isinstance(baseline, Mapping) or set(baseline) != set(FIELDS):
            raise ValueError("Baseline CSV must have exactly the four official columns")
        if baseline.get("sample_id") != sid:
            raise ValueError("Baseline and input IDs/order must match exactly")
        output.append(_prediction(baseline, sid))
    return output


def _evidence(n, vision, second, scores):
    try:
        a, b = np.asarray(vision, dtype=np.float64), np.asarray(second, dtype=np.float64)
        p = np.asarray(scores, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Evidence must contain numeric probabilities") from exc
    for label, value in (("vision", a), ("second", b)):
        if (value.shape != (n, 3) or not np.isfinite(value).all() or
                ((value < 0) | (value > 1)).any() or
                not np.allclose(value.sum(axis=1), 1., atol=1e-6, rtol=0.)):
            raise ValueError(f"{label} must be finite normalized (N, 3) probabilities")
    if p.shape != (n,) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Calibration scores must be a finite (N,) probability vector")
    return a, b, p


def supplement_decisions(rows, baseline_by_id, vision, second, scores):
    """Apply only ``v4_yes OR (argmax differs AND probability > .5)``.

    Arrays are aligned to ``rows``; no implicit sorting or broadcasting occurs.
    Exact argmax ties use NumPy's first maximum in VALID_EMOTIONS order.
    This function never reads ground-truth labels or modifies any input.
    """
    ids = _ids(rows)
    if not isinstance(baseline_by_id, Mapping) or set(baseline_by_id) != set(ids):
        raise ValueError("Baseline mapping must cover the unique input IDs exactly")
    baseline = [_prediction(baseline_by_id[sid], sid) for sid in ids]
    a, b, p = _evidence(len(ids), vision, second, scores)
    ai, bi = a.argmax(axis=1), b.argmax(axis=1)
    decisions = []
    for i, (row, old) in enumerate(zip(rows, baseline)):
        disagreement = bool(ai[i] != bi[i])
        added = old["is_uncertain"] == "no" and disagreement and bool(p[i] > THRESHOLD)
        uncertain = "yes" if old["is_uncertain"] == "yes" or added else "no"
        decisions.append({
            "sample_id": row["sample_id"], "protocol": row["protocol"],
            "baseline_pred_emotion": old["pred_emotion"],
            "baseline_uncertain": old["is_uncertain"],
            "baseline_question": old["clarifying_question"],
            "is_uncertain": uncertain, "added_yes": bool(added),
            "argmax_disagrees": disagreement,
            "vision_emotion": VALID_EMOTIONS[int(ai[i])],
            "second_emotion": VALID_EMOTIONS[int(bi[i])],
            "vision": a[i].tolist(), "second": b[i].tolist(),
            "calibrated_probability": float(p[i]),
        })
    return decisions


def _safe_question_row(row):
    # A callback receives evidence inputs, never training labels or metadata.
    keys = ("sample_id", "protocol", "video")
    if row["protocol"] == "VT":
        keys += ("text_content", "text")
    return {key: deepcopy(row[key]) for key in keys if key in row}


def build_submission(rows, baseline_rows, vision, second, scores, question_fn):
    """Return ``(four_column_rows, added_yes_audit)`` without filesystem writes.

    ``question_fn(safe_input_row, decision) -> str`` runs only for added yes.
    Its input row contains only sample_id/protocol/video, plus VT transcript
    text_content/text. VA never receives a supplied transcript.
    All existing v4 answers, including their original questions, are copied.
    """
    baseline = _baseline_rows(rows, baseline_rows)
    decisions = supplement_decisions(rows, {r["sample_id"]: r for r in baseline},
                                      vision, second, scores)
    if any(d["added_yes"] for d in decisions) and not callable(question_fn):
        raise ValueError("New yes decisions require a question generation callback")
    output, changes = [], []
    for row, old, decision in zip(rows, baseline, decisions):
        answer = dict(old)
        if decision["added_yes"]:
            question = question_fn(_safe_question_row(row), deepcopy(decision))
            if not isinstance(question, str):
                raise ValueError("Question callback must return a question string")
            answer.update(is_uncertain="yes", clarifying_question=question)
            _prediction(answer, row["sample_id"])
            changes.append({**decision, "before": dict(old), "after": dict(answer)})
        output.append(answer)
    validate_submission(rows, baseline, output, decisions)
    return output, changes


def validate_submission(rows, baseline_rows, output_rows, decisions=None):
    """Fail closed if output changes emotion, removes yes, or breaks schema."""
    baseline = _baseline_rows(rows, baseline_rows)
    output = _baseline_rows(rows, output_rows)
    if decisions is not None and (len(decisions) != len(rows) or
            [d["sample_id"] for d in decisions] != [r["sample_id"] for r in rows]):
        raise ValueError("Decision IDs/order must match the input")
    added = 0
    for i, (old, new) in enumerate(zip(baseline, output)):
        if old["pred_emotion"] != new["pred_emotion"]:
            raise ValueError("Supplement may not change the frozen v4 emotion")
        if old["is_uncertain"] == "yes" and new != old:
            raise ValueError("Supplement must preserve every existing v4 yes and question")
        if new["is_uncertain"] == old["is_uncertain"] and new != old:
            raise ValueError("Unchanged decisions must preserve the original answer")
        if old["is_uncertain"] == "no" and new["is_uncertain"] == "yes":
            added += 1
        if decisions is not None and new["is_uncertain"] != decisions[i]["is_uncertain"]:
            raise ValueError("Submission does not match the fixed supplement decisions")
    return {"rows": len(rows), "emotion_changes": 0, "removed_yes": 0,
            "invalid_outputs": 0, "inconsistent_questions": 0, "added_yes": added,
            "ids_and_order_preserved": True, "existing_questions_preserved": True}


def read_baseline_snapshot(path, rows):
    """Parse rows and identify their confirmed artifact from the same bytes."""
    ids = _ids(rows)
    if len(ids) != TEST_ROWS:
        raise ValueError("Official submission must contain exactly 734 rows")
    content = Path(path).read_bytes()
    digest = _hash(content)
    confirmed = dict(_confirmed_baseline(digest))
    try:
        stream = io.StringIO(content.decode("utf-8-sig"), newline="")
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Baseline must have the four official columns in order")
        baseline = list(reader)
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("Invalid baseline CSV") from exc
    return _baseline_rows(rows, baseline), digest, confirmed


def read_baseline_csv(path, rows):
    """Read a confirmed public artifact, checking hash, schema and 734 IDs."""
    baseline, _, _ = read_baseline_snapshot(path, rows)
    return baseline


def _csv_bytes(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _check_immutable(path, content):
    if path.exists() and (not path.is_file() or path.read_bytes() != content):
        raise ValueError(f"Refusing to overwrite a different artifact: {path}")


def write_immutable(path, content):
    """Create bytes once; identical retries are harmless, differing ones fail."""
    if not isinstance(content, bytes):
        raise TypeError("Immutable artifacts must be supplied as bytes")
    path = Path(path)
    _check_immutable(path, content)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Exclusive create also prevents a concurrent process from replacing a file.
        with path.open("xb") as stream:
            stream.write(content)
    except FileExistsError:
        _check_immutable(path, content)


def export_submission(output_dir, rows, baseline_csv, vision, second, scores,
                      question_fn, provenance):
    """Validate all 734 rows before writing four immutable review artifacts.

    The manifest is written last and marks a complete export. The source CSV
    cannot be the output path, even when proposed bytes happen to be identical.
    """
    root, source = Path(output_dir).resolve(), Path(baseline_csv).resolve()
    paths = {name: root / name for name in (
        "submission.csv", "submission-changes.json", "submission-evidence.json",
        "submission-manifest.json")}
    if source in paths.values():
        raise ValueError("Refusing to export over the original baseline CSV")
    if not isinstance(provenance, Mapping):
        raise ValueError("Provenance must be a JSON object")
    _json_bytes(dict(provenance))  # Fail before question generation or writing.
    baseline, baseline_hash, confirmed = read_baseline_snapshot(source, rows)
    output, changes = build_submission(rows, baseline, vision, second, scores, question_fn)
    decisions = supplement_decisions(rows, {r["sample_id"]: r for r in baseline},
                                      vision, second, scores)
    checks = validate_submission(rows, baseline, output, decisions)
    content = _csv_bytes(output)
    audit = {"algorithm": VERSION, "threshold": THRESHOLD,
             "baseline_sha256": baseline_hash, "baseline_name": confirmed["name"],
             "baseline_public_score": confirmed["public_score"], "changes": changes}
    evidence = {"algorithm": VERSION, "threshold": THRESHOLD,
                "emotion_order": list(VALID_EMOTIONS), "decisions": decisions}
    manifest = {
        "algorithm": VERSION, "threshold": THRESHOLD, "file": str(paths["submission.csv"]),
        "sha256": _hash(content), "rows": TEST_ROWS, "baseline_file": str(source),
        "baseline_sha256": baseline_hash, "baseline_name": confirmed["name"],
        "baseline_public_score": confirmed["public_score"], "added_yes": len(changes),
        "changed_sample_ids": [c["sample_id"] for c in changes], "checks": checks,
        "provenance": dict(provenance), "submitted_to_kaggle": False,
    }
    payloads = {"submission.csv": content, "submission-changes.json": _json_bytes(audit),
                "submission-evidence.json": _json_bytes(evidence)}
    manifest["artifact_sha256"] = {name: _hash(data) for name, data in payloads.items()}
    payloads["submission-manifest.json"] = _json_bytes(manifest)
    for name, data in payloads.items():
        if paths[name].resolve() == source:
            raise ValueError("An output path aliases the original baseline CSV")
        _check_immutable(paths[name], data)
    if _hash(source.read_bytes()) != baseline_hash:
        raise ValueError("Baseline changed during question generation; export cancelled")
    _confirmed_baseline(baseline_hash)
    for name, data in payloads.items():
        write_immutable(paths[name], data)
    return manifest
