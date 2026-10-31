"""Build a Kaggle-ready submission CSV from raw RL-model predictions.

Stdlib-only CLI tool; runs on a Mac without any ML stack. It validates raw
predictions on the test set against the official test file, applies safe
fix-ups (or fails in --strict mode), and writes the submission CSV in the
official test-file row order.

Submission contract (competition rules):
  * header exactly: sample_id,pred_emotion,is_uncertain,clarifying_question
  * one row per test sample, in the official test file order (734 rows)
  * pred_emotion in {positive, negative, neutral}
  * is_uncertain in {yes, no}
  * clarifying_question: the generated question when is_uncertain=yes,
    otherwise the literal string "none"
  * no empty field anywhere (Kaggle rejects the file otherwise)

Input formats (auto-detected by extension / content):
  a) JSONL: each line has at least ``sample_id`` and either a raw model
     completion under ``completion`` / ``output`` / ``response`` (parsed
     with the existing parser from rl/metrics.py ``parse_prediction``) or
     already-parsed fields ``pred_emotion`` / ``is_uncertain`` /
     ``clarifying_question``.
  b) CSV: a ``sample_id`` column plus either the parsed columns or a raw
     completion column (same key names).

Usage:
    python rl/make_submission.py --input rl/output/raw_predictions.jsonl
    python rl/make_submission.py --input preds.csv --strict \
        --test-file Dataset/test_release.jsonl --output rl/output/submission.csv
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter

# Make sibling modules importable regardless of the launch directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import parse_prediction  # noqa: E402  (stdlib-only parser)
from reward import parse_completion, question_is_consistent  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEADER = ["sample_id", "pred_emotion", "is_uncertain", "clarifying_question"]
VALID_EMOTIONS = {"positive", "negative", "neutral"}
VALID_UNCERTAIN = {"yes", "no"}

# Keys under which a raw model completion may appear in the input.
RAW_COMPLETION_KEYS = ("completion", "output", "response")

# Generic, bilingual-safe fallback question for is_uncertain=yes rows whose
# question is missing. Chinese matches the dataset language; English fallback
# (keep in sync if changed): "Which feeling best describes how you feel right
# now — more positive or more negative?"
DEFAULT_QUESTION = "你此刻的感受更偏向哪一种？"

# Default locations for the official test file, tried in order.
DEFAULT_TEST_FILE_CANDIDATES = ("Dataset/test_release.jsonl", "data/test_release.jsonl")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate raw RL predictions and build a Kaggle submission CSV."
    )
    parser.add_argument("--input", required=True,
                        help="Raw predictions: JSONL or CSV (auto-detected).")
    parser.add_argument("--test-file", default=None,
                        help="Official test JSONL (fields sample_id, video, text). "
                             "Default: tries Dataset/test_release.jsonl, then "
                             "data/test_release.jsonl.")
    parser.add_argument("--output", default="rl/output/submission.csv",
                        help="Output submission CSV (default: rl/output/submission.csv).")
    parser.add_argument("--strict", action="store_true",
                        help="Fail (non-zero exit) on any fix-up instead of "
                             "applying it. Default: fix + warn.")
    parser.add_argument("--fallback-emotion", default="negative",
                        choices=sorted(VALID_EMOTIONS),
                        help="Emotion used when a prediction is unparseable or "
                             "invalid (default: negative — the majority class; "
                             "neutral is dangerous: gold labels contain almost "
                             "none).")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def resolve_test_file(path_arg):
    """Resolve the test-file path from the CLI arg or the default candidates."""
    if path_arg:
        if not os.path.isfile(path_arg):
            raise SystemExit(f"ERROR: --test-file not found: {path_arg}")
        return path_arg
    for candidate in DEFAULT_TEST_FILE_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    raise SystemExit(
        "ERROR: no --test-file given and none of the default candidates exist: "
        + ", ".join(DEFAULT_TEST_FILE_CANDIDATES)
    )


def load_test_ids(test_file):
    """Load the official test JSONL and return sample_ids in file order."""
    ids = []
    with open(test_file, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"ERROR: bad JSON in test file line {lineno}: {exc}")
            sid = obj.get("sample_id")
            if not sid:
                raise SystemExit(f"ERROR: test file line {lineno} has no sample_id")
            ids.append(str(sid))
    if not ids:
        raise SystemExit(f"ERROR: test file {test_file} contains no rows")
    return ids


def load_predictions(path):
    """Load raw predictions as a list of row dicts (JSONL or CSV)."""
    if not os.path.isfile(path):
        raise SystemExit(f"ERROR: --input not found: {path}")
    if path.lower().endswith(".csv"):
        # utf-8-sig tolerates a BOM; DictReader handles quoting.
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or "sample_id" not in reader.fieldnames:
                raise SystemExit(
                    f"ERROR: CSV input {path} must have a 'sample_id' column; "
                    f"got header {reader.fieldnames}"
                )
            return [dict(row) for row in reader]
    # JSONL
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"ERROR: bad JSON in input line {lineno}: {exc}")
            if not isinstance(obj, dict):
                raise SystemExit(f"ERROR: input line {lineno} is not a JSON object")
            rows.append(obj)
    if not rows:
        raise SystemExit(f"ERROR: input {path} contains no prediction rows")
    return rows


def clean_field(value):
    """Normalize one output field: strip whitespace/quotes, flatten newlines.

    Returns the cleaned string ("" for None/missing). The csv module takes
    care of any remaining quoting needs on write.
    """
    if value is None:
        return ""
    s = str(value).replace("\r", " ").replace("\n", " ").strip()
    # Strip one layer of surrounding matching quotes (models sometimes wrap
    # values in quotes inside the JSON string itself).
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    return s


def extract_prediction(row):
    """Pull (pred_emotion, is_uncertain, clarifying_question) out of one row.

    Raw completions (under completion/output/response) are validated with the
    existing parser from rl/metrics.py (``parse_prediction``); the question
    text is recovered with reward.parse_completion, since parse_prediction
    deliberately only returns the two metric fields.
    Already-parsed rows are passed through as-is (cleaned later).
    """
    raw = None
    for key in RAW_COMPLETION_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            raw = value
            break
    if raw is not None:
        gate = parse_prediction(raw)
        if gate["pred_emotion"] is None and gate["is_uncertain"] is None:
            return {"pred_emotion": None, "is_uncertain": None,
                    "clarifying_question": None}
        full = parse_completion(raw)
        question = full["clarifying_question"] if full else None
        return {"pred_emotion": gate["pred_emotion"],
                "is_uncertain": gate["is_uncertain"],
                "clarifying_question": question}
    return {
        "pred_emotion": row.get("pred_emotion"),
        "is_uncertain": row.get("is_uncertain"),
        "clarifying_question": row.get("clarifying_question"),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    test_file = resolve_test_file(args.test_file)
    test_ids = load_test_ids(test_file)
    test_set = set(test_ids)
    rows = load_predictions(args.input)

    counters = Counter()   # fix-up kind -> count
    issues = []            # (kind, sample_id, human-readable detail)

    def record(kind, sample_id, detail):
        counters[kind] += 1
        issues.append((kind, sample_id, detail))

    # ---- index predictions by sample_id ----------------------------------
    by_id = {}
    input_order = []
    for row in rows:
        sid = clean_field(row.get("sample_id"))
        if not sid:
            record("row_without_sample_id", "<none>", "prediction row has no sample_id; skipped")
            continue
        if sid in by_id:
            record("duplicate_id", sid, "duplicate prediction; keeping the first occurrence")
            continue
        by_id[sid] = row
        input_order.append(sid)

    for sid in list(by_id):
        if sid not in test_set:
            record("extra_id", sid, "prediction for unknown sample_id; dropped")

    # Did the input order match the official order (ignoring extras)?
    filtered_input_order = [sid for sid in input_order if sid in test_set]
    if filtered_input_order != [sid for sid in test_ids if sid in set(filtered_input_order)]:
        counters["reordered"] += 1
        issues.append(("reordered", "<all>", "input row order differed from the "
                                             "official test file; reordered automatically"))

    # ---- per-sample validation + fix-ups, in official test order ---------
    out_rows = []
    for sid in test_ids:
        if sid not in by_id:
            record("missing_id", sid, "no prediction for test sample; filled with "
                                      "fallback defaults")
            out_rows.append([sid, args.fallback_emotion, "no", "none"])
            continue

        pred = extract_prediction(by_id[sid])
        if args.strict:
            strict_prediction = parse_completion(json.dumps(pred, ensure_ascii=False))
            if strict_prediction is None or not question_is_consistent(strict_prediction):
                record("output_contract_violation", sid,
                       "raw prediction does not satisfy the exact model output contract")
            else:
                # CSV quoting can preserve a question string exactly; strict export
                # must not quietly rewrite a valid model prediction.
                out_rows.append([sid, strict_prediction["pred_emotion"],
                                 strict_prediction["is_uncertain"],
                                 strict_prediction["clarifying_question"]])
                continue

        # -- field cleanup (whitespace / quotes / newlines) --
        cleaned = {}
        for key in ("pred_emotion", "is_uncertain", "clarifying_question"):
            raw_value = pred.get(key)
            value = clean_field(raw_value)
            cleaned[key] = value
            if raw_value is not None and value != str(raw_value):
                counters["field_cleaned"] += 1
                if args.strict:
                    record("field_normalization", sid, f"{key} would require normalization")

        # -- pred_emotion --
        emotion = cleaned["pred_emotion"].lower()
        if emotion not in VALID_EMOTIONS:
            record("emotion_fallback", sid,
                   f"unparseable/invalid pred_emotion "
                   f"{cleaned['pred_emotion']!r}; using {args.fallback_emotion!r}")
            emotion = args.fallback_emotion
        elif emotion != cleaned["pred_emotion"]:
            counters["field_cleaned"] += 1  # case-normalized

        # -- is_uncertain --
        uncertain = cleaned["is_uncertain"].lower()
        if uncertain not in VALID_UNCERTAIN:
            record("uncertain_fallback", sid,
                   f"invalid is_uncertain {cleaned['is_uncertain']!r}; using 'no'")
            uncertain = "no"

        # -- clarifying_question consistency with the output contract --
        question = cleaned["clarifying_question"]
        if uncertain == "no":
            if question == "":
                # Silent normalization (still counted for the report).
                counters["question_none_filled"] += 1
                question = "none"
            elif question.lower() != "none":
                record("question_dropped", sid,
                       "is_uncertain=no but a question was present; replaced with 'none'")
                question = "none"
            else:
                question = "none"  # normalize case variants like "None"
        else:  # uncertain == "yes"
            if question == "" or question.lower() == "none":
                record("question_default_inserted", sid,
                       "is_uncertain=yes but question missing/'none'; inserted "
                       "DEFAULT_QUESTION — CHECK THIS SAMPLE, a generic question "
                       "may hurt human evaluation")
                question = DEFAULT_QUESTION

        # Final guard: no empty field may reach the CSV (Kaggle rejects it).
        assert sid and emotion and uncertain and question, sid
        out_rows.append([sid, emotion, uncertain, question])

    # ---- strict mode: any fix-up is fatal ---------------------------------
    if args.strict and issues:
        print(f"STRICT MODE: {len(issues)} issue(s) found; refusing to write "
              f"the submission.", file=sys.stderr)
        for kind, sid, detail in issues[:20]:
            print(f"  [{kind}] {sid}: {detail}", file=sys.stderr)
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more.", file=sys.stderr)
        sys.exit(2)

    # ---- write the CSV -----------------------------------------------------
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        writer.writerows(out_rows)

    # ---- report -------------------------------------------------------------
    n = len(out_rows)
    emotion_dist = Counter(r[1] for r in out_rows)
    yes_count = sum(1 for r in out_rows if r[2] == "yes")
    yes_share = yes_count / n if n else 0.0

    print("==== submission report ====")
    print(f"input       : {args.input} ({len(rows)} prediction rows)")
    print(f"test file   : {test_file} ({len(test_ids)} samples)")
    print(f"output      : {args.output} ({n} rows + header)")
    print("fix-ups applied" + (" (none)" if not counters else ":"))
    for kind in sorted(counters):
        print(f"  {kind:<26} {counters[kind]}")
    if issues:
        shown = [f"{kind}:{sid}" for kind, sid, _ in issues[:5]]
        print(f"  first issues: {', '.join(shown)}"
              + (" ..." if len(issues) > 5 else ""))
    print(f"pred_emotion distribution: {dict(emotion_dist)}")
    print(f"is_uncertain=yes share   : {yes_share:.2%} ({yes_count}/{n})")
    if yes_share < 0.01 or yes_share > 0.50:
        print(f"WARNING: is_uncertain=yes share {yes_share:.2%} is outside "
              f"[1%, 50%] — suspicious, check your model (possible reward "
              f"hacking or a collapsed uncertainty head).")


if __name__ == "__main__":
    main()
