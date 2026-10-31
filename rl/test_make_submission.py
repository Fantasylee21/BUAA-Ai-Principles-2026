"""Stdlib-only test suite for rl/make_submission.py.

Run from the repo root:

    python rl/test_make_submission.py

Builds a tiny fake test file (5 samples) plus fake raw-prediction inputs in a
temp directory, and exercises every fix-up branch, the reorder case,
duplicate/missing/extra IDs, --strict failures, CSV input, and the
end-to-end happy path. Assertions cover output CSV contents and exit codes.
"""

import csv
import json
import os
import subprocess
import sys
import tempfile

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_submission.py")
SAMPLE_IDS = [f"test:s{i}" for i in range(1, 6)]  # 5 fake test samples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_test_file(path, ids=SAMPLE_IDS):
    with open(path, "w", encoding="utf-8") as fh:
        for sid in ids:
            fh.write(json.dumps({
                "sample_id": sid,
                "video": f"videos/{sid}.mp4",
                "text": "fake utterance",
            }) + "\n")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def parsed_row(sid, emotion="positive", uncertain="no", question="none"):
    return {"sample_id": sid, "pred_emotion": emotion,
            "is_uncertain": uncertain, "clarifying_question": question}


def completion_row(sid, emotion="positive", uncertain="no", question="none"):
    return {"sample_id": sid, "completion": json.dumps({
        "pred_emotion": emotion, "is_uncertain": uncertain,
        "clarifying_question": question})}


def clean_rows(ids=SAMPLE_IDS):
    return [parsed_row(sid) for sid in ids]


def run_cli(args):
    """Run make_submission.py in a subprocess; return (returncode, combined output)."""
    proc = subprocess.run(
        [sys.executable, SCRIPT, *args],
        capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.reader(fh))


class Fixture:
    """Temp dir with a fake test file and standard CLI prefix."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.test_file = os.path.join(self.dir, "test.jsonl")
        make_test_file(self.test_file)
        self.input = os.path.join(self.dir, "preds.jsonl")
        self.output = os.path.join(self.dir, "submission.csv")

    def base_args(self):
        return ["--input", self.input, "--test-file", self.test_file,
                "--output", self.output]

    def cleanup(self):
        self.tmp.cleanup()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_happy_path_jsonl():
    fx = Fixture()
    try:
        write_jsonl(fx.input, clean_rows())
        rc, out = run_cli(fx.base_args())
        assert rc == 0, out
        rows = read_csv(fx.output)
        assert rows[0] == ["sample_id", "pred_emotion", "is_uncertain",
                           "clarifying_question"], rows[0]
        assert len(rows) == 6, len(rows)
        for i, sid in enumerate(SAMPLE_IDS, 1):
            assert rows[i] == [sid, "positive", "no", "none"], rows[i]
        assert "fix-ups applied (none)" in out, out
        assert "is_uncertain=yes share   : 0.00%" in out, out
        # 0% yes-share must trigger the suspicious-model warning.
        assert "suspicious, check your model" in out, out
    finally:
        fx.cleanup()


def test_reorder_case():
    fx = Fixture()
    try:
        rows = clean_rows()
        # Reverse order and give each row a distinct emotion to verify mapping.
        for i, row in enumerate(rows):
            row["pred_emotion"] = ["positive", "negative", "positive",
                                   "negative", "positive"][i]
        write_jsonl(fx.input, list(reversed(rows)))
        rc, out = run_cli(fx.base_args())
        assert rc == 0, out
        assert "reordered" in out, out
        got = read_csv(fx.output)[1:]
        assert [r[0] for r in got] == SAMPLE_IDS, got  # official order restored
        assert got[0][1] == "positive" and got[1][1] == "negative", got
    finally:
        fx.cleanup()


def test_all_fixup_branches():
    fx = Fixture()
    try:
        rows = [
            # s1: unparseable raw completion -> fallback emotion, uncertain -> no
            {"sample_id": SAMPLE_IDS[0], "completion": "I think... oops, no json"},
            # s2: yes + empty question -> DEFAULT_QUESTION
            parsed_row(SAMPLE_IDS[1], "negative", "yes", ""),
            # s3: no + question present -> replaced with "none"
            parsed_row(SAMPLE_IDS[2], "positive", "no", "Are you happy?"),
            # s4: no + empty question -> silent "none"
            parsed_row(SAMPLE_IDS[3], "neutral", "no", ""),
            # s5: invalid is_uncertain -> "no"; messy whitespace/quotes cleaned
            parsed_row(SAMPLE_IDS[4], ' "Negative" ', "maybe", ' "none" '),
        ]
        write_jsonl(fx.input, rows)
        rc, out = run_cli(fx.base_args())
        assert rc == 0, out
        got = {r[0]: r for r in read_csv(fx.output)[1:]}
        assert got[SAMPLE_IDS[0]] == [SAMPLE_IDS[0], "negative", "no", "none"], got
        # s2: default question inserted (non-empty, not "none")
        assert got[SAMPLE_IDS[1]][2] == "yes", got
        assert got[SAMPLE_IDS[1]][3] not in ("", "none"), got
        assert got[SAMPLE_IDS[2]][3] == "none", got
        assert got[SAMPLE_IDS[3]] == [SAMPLE_IDS[3], "neutral", "no", "none"], got
        assert got[SAMPLE_IDS[4]] == [SAMPLE_IDS[4], "negative", "no", "none"], got
        # All fix-up counters appear in the report.
        for kind in ["emotion_fallback", "question_default_inserted",
                     "question_dropped", "question_none_filled",
                     "uncertain_fallback", "field_cleaned"]:
            assert kind in out, (kind, out)
    finally:
        fx.cleanup()


def test_raw_completion_parsing_used():
    fx = Fixture()
    try:
        write_jsonl(fx.input, [
            completion_row(SAMPLE_IDS[0], "negative", "yes", "Which modality?"),
            # JSON embedded in extra text must still parse.
            {"sample_id": SAMPLE_IDS[1], "response":
                'Answer: {"pred_emotion": "positive", "is_uncertain": "no", '
                '"clarifying_question": "none"} done'},
            *[parsed_row(sid) for sid in SAMPLE_IDS[2:]],
        ])
        rc, out = run_cli(fx.base_args())
        assert rc == 0, out
        got = {r[0]: r for r in read_csv(fx.output)[1:]}
        assert got[SAMPLE_IDS[0]] == [SAMPLE_IDS[0], "negative", "yes",
                                      "Which modality?"], got
        assert got[SAMPLE_IDS[1]] == [SAMPLE_IDS[1], "positive", "no", "none"], got
    finally:
        fx.cleanup()


def test_duplicate_missing_extra_ids():
    fx = Fixture()
    try:
        rows = [
            parsed_row(SAMPLE_IDS[0], "positive"),
            parsed_row(SAMPLE_IDS[0], "negative"),   # duplicate -> first wins
            parsed_row(SAMPLE_IDS[1]),
            # SAMPLE_IDS[2] missing -> fallback fill
            parsed_row(SAMPLE_IDS[3]),
            parsed_row(SAMPLE_IDS[4]),
            parsed_row("test:EXTRA", "negative"),     # extra -> dropped
        ]
        write_jsonl(fx.input, rows)
        rc, out = run_cli(fx.base_args())
        assert rc == 0, out
        for kind in ["duplicate_id", "missing_id", "extra_id"]:
            assert kind in out, (kind, out)
        got = {r[0]: r for r in read_csv(fx.output)[1:]}
        assert len(got) == 5 and "test:EXTRA" not in got, got
        assert got[SAMPLE_IDS[0]][1] == "positive", got   # first occurrence wins
        assert got[SAMPLE_IDS[2]] == [SAMPLE_IDS[2], "negative", "no", "none"], got
    finally:
        fx.cleanup()


def test_strict_mode_failures():
    fx = Fixture()
    try:
        # One broken row (unparseable completion) -> strict must fail.
        rows = clean_rows()
        rows[2] = {"sample_id": SAMPLE_IDS[2], "completion": "totally broken"}
        write_jsonl(fx.input, rows)
        rc, out = run_cli(fx.base_args() + ["--strict"])
        assert rc == 2, (rc, out)
        assert "STRICT MODE" in out, out
        assert SAMPLE_IDS[2] in out, out           # affected sample_id listed
        assert not os.path.exists(fx.output), "strict failure must not write CSV"

        # Strict + fully clean input -> success.
        write_jsonl(fx.input, clean_rows())
        rc, out = run_cli(fx.base_args() + ["--strict"])
        assert rc == 0, out
        assert os.path.exists(fx.output)
    finally:
        fx.cleanup()


def test_csv_input_format():
    fx = Fixture()
    try:
        csv_input = os.path.join(fx.dir, "preds.csv")
        with open(csv_input, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["sample_id", "pred_emotion", "is_uncertain",
                             "clarifying_question"])
            for sid in SAMPLE_IDS:
                writer.writerow([sid, "negative", "no", "none"])
        rc, out = run_cli(["--input", csv_input, "--test-file", fx.test_file,
                           "--output", fx.output])
        assert rc == 0, out
        got = read_csv(fx.output)[1:]
        assert all(r[1] == "negative" for r in got), got

        # CSV with a raw completion column instead of parsed fields.
        with open(csv_input, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["sample_id", "completion"])
            for sid in SAMPLE_IDS:
                writer.writerow([sid, json.dumps({
                    "pred_emotion": "positive", "is_uncertain": "no",
                    "clarifying_question": "none"})])
        rc, out = run_cli(["--input", csv_input, "--test-file", fx.test_file,
                           "--output", fx.output])
        assert rc == 0, out
        assert all(r[1] == "positive" for r in read_csv(fx.output)[1:])
    finally:
        fx.cleanup()


def test_fallback_emotion_flag():
    fx = Fixture()
    try:
        rows = clean_rows()
        rows[0] = {"sample_id": SAMPLE_IDS[0], "completion": "broken!"}
        write_jsonl(fx.input, rows)
        rc, out = run_cli(fx.base_args() + ["--fallback-emotion", "neutral"])
        assert rc == 0, out
        got = {r[0]: r for r in read_csv(fx.output)[1:]}
        assert got[SAMPLE_IDS[0]][1] == "neutral", got
    finally:
        fx.cleanup()


def test_no_empty_fields_guarantee():
    # Every deliberately broken/missing value must still yield a full row.
    fx = Fixture()
    try:
        write_jsonl(fx.input, [
            {"sample_id": SAMPLE_IDS[0]},                       # everything missing
            {"sample_id": SAMPLE_IDS[1], "pred_emotion": "",    # empty strings
             "is_uncertain": "", "clarifying_question": ""},
            parsed_row(SAMPLE_IDS[2], "positive", "yes", "  \n "),
            parsed_row(SAMPLE_IDS[3], "negative", "no", "none"),
            parsed_row(SAMPLE_IDS[4], "neutral", "no", "none"),
        ])
        rc, out = run_cli(fx.base_args())
        assert rc == 0, out
        for row in read_csv(fx.output)[1:]:
            assert len(row) == 4 and all(cell != "" for cell in row), row
    finally:
        fx.cleanup()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc!r}")
        except Exception as exc:  # noqa: BLE001 - report everything
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed.")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
