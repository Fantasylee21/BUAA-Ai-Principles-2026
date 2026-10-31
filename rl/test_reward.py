"""Stdlib-only test suite for rl/reward.py and rl/metrics.py.

Run from the workspace root:

    python rl/test_reward.py

No third-party dependencies; plain asserts with a small runner that reports
pass/fail per test.
"""

import json
import os
import sys

# Make sibling modules importable regardless of launch directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metrics  # noqa: E402
import reward  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_completion(pred_emotion="positive", is_uncertain="no",
                    clarifying_question="none"):
    return json.dumps({
        "pred_emotion": pred_emotion,
        "is_uncertain": is_uncertain,
        "clarifying_question": clarifying_question,
    })


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# reward.py tests
# ---------------------------------------------------------------------------

def test_valid_correct_answer_full_reward():
    res = reward.compute_sample_reward(
        make_completion("positive", "no", "none"), "positive", "Aligned"
    )
    assert res["valid_json"] is True
    assert res["components"]["reward_format"] == 1.0
    assert res["components"]["reward_emotion"] == 1.0
    assert res["components"]["reward_uncertainty"] == 0.4  # aligned correct
    assert res["components"]["reward_question_consistency"] == 0.0
    expected_total = 0.10 * 1.0 + 0.50 * 1.0 + 0.40 * 0.4 + 0.05 * 0.0
    assert approx(res["total"], expected_total), res["total"]


def test_wrong_emotion():
    res = reward.compute_sample_reward(
        make_completion("negative", "no", "none"), "positive", "Aligned"
    )
    assert res["components"]["reward_emotion"] == 0.0
    assert res["components"]["reward_format"] == 1.0  # format still valid


def test_uncertainty_signs_and_magnitudes():
    # Conflict: yes -> +0.6, no -> -0.3
    c_yes = reward.compute_sample_reward(
        make_completion(is_uncertain="yes", clarifying_question="Which modality?"),
        "positive", "Conflict",
    )["components"]["reward_uncertainty"]
    c_no = reward.compute_sample_reward(
        make_completion(is_uncertain="no"), "positive", "Conflict",
    )["components"]["reward_uncertainty"]
    # Aligned: no -> +0.4, yes -> -0.2
    a_no = reward.compute_sample_reward(
        make_completion(is_uncertain="no"), "positive", "Aligned",
    )["components"]["reward_uncertainty"]
    a_yes = reward.compute_sample_reward(
        make_completion(is_uncertain="yes", clarifying_question="Which modality?"),
        "positive", "Aligned",
    )["components"]["reward_uncertainty"]

    assert approx(c_yes, 0.6) and approx(c_no, -0.3)
    assert approx(a_no, 0.4) and approx(a_yes, -0.2)
    # Signs
    assert c_yes > 0 > c_no
    assert a_no > 0 > a_yes
    # Relative magnitudes: conflict branch outweighs aligned branch (0.6 vs 0.4),
    # mirroring the metric's 0.3 vs 0.2 asymmetry.
    assert c_yes > a_no
    assert abs(c_no) > abs(a_yes)


def test_malformed_json_format_zero():
    for bad in [
        "not json at all",
        '{"pred_emotion": "positive"',                      # unbalanced
        '{"pred_emotion": "happy", "is_uncertain": "no", "clarifying_question": "none"}',  # bad enum
        '{"pred_emotion": "positive", "is_uncertain": "no"}',  # missing field
        '{"pred_emotion": "positive", "is_uncertain": "maybe", "clarifying_question": "none"}',
        '[1, 2, 3]',
        "",
    ]:
        res = reward.compute_sample_reward(bad, "positive", "Aligned")
        assert res["valid_json"] is False, bad
        assert res["components"]["reward_format"] == 0.0, bad
        assert res["components"]["reward_emotion"] == 0.0, bad
        assert res["components"]["reward_uncertainty"] == 0.0, bad
        assert res["total"] == 0.0, bad


def test_missing_question_when_uncertain_penalty():
    for q in ["none", "", "   "]:
        res = reward.compute_sample_reward(
            make_completion(is_uncertain="yes", clarifying_question=q),
            "positive", "Conflict",
        )
        assert res["components"]["reward_question_consistency"] == -1.0, q


def test_question_present_when_not_uncertain_penalty():
    res = reward.compute_sample_reward(
        make_completion(is_uncertain="no", clarifying_question="Are you sure?"),
        "positive", "Aligned",
    )
    assert res["components"]["reward_question_consistency"] == -1.0


def test_none_question_when_not_uncertain_no_penalty():
    res = reward.compute_sample_reward(
        make_completion(is_uncertain="no", clarifying_question="none"),
        "positive", "Aligned",
    )
    assert res["components"]["reward_question_consistency"] == 0.0


def test_conversational_completion_parsing():
    payload = make_completion("negative", "no", "none")
    conv = [{"role": "assistant", "content": payload}]
    res = reward.compute_sample_reward(conv, "negative", "Aligned")
    assert res["valid_json"] is True
    assert res["components"]["reward_emotion"] == 1.0

    # Content as a list of typed parts also parses.
    conv_parts = [{"role": "assistant",
                   "content": [{"type": "text", "text": payload}]}]
    res = reward.compute_sample_reward(conv_parts, "negative", "Aligned")
    assert res["valid_json"] is True
    assert res["components"]["reward_emotion"] == 1.0


def test_json_embedded_in_extra_text():
    payload = make_completion("neutral", "yes", "Could you clarify?")
    noisy = f"Sure! Here is my answer:\n{payload}\nHope that helps."
    res = reward.compute_sample_reward(noisy, "neutral", "Conflict")
    assert res["valid_json"] is True
    assert res["components"]["reward_format"] == 1.0
    assert res["components"]["reward_emotion"] == 1.0
    assert approx(res["components"]["reward_uncertainty"], 0.6)


def test_trl_wrapper_signatures():
    prompts = ["p1", "p2"]
    completions = [
        make_completion("positive", "no", "none"),
        [{"role": "assistant", "content": make_completion("negative", "yes", "Q?")}],
    ]
    gt = ["positive", "negative"]
    st = ["Aligned", "Conflict"]
    fmt = reward.reward_format(prompts, completions, gt_label=gt, sample_type=st)
    emo = reward.reward_emotion(prompts, completions, gt_label=gt, sample_type=st)
    unc = reward.reward_uncertainty(prompts, completions, gt_label=gt, sample_type=st)
    qc = reward.reward_question_consistency(prompts, completions, gt_label=gt,
                                            sample_type=st)
    assert fmt == [1.0, 1.0]
    assert emo == [1.0, 1.0]
    assert approx(unc[0], 0.4) and approx(unc[1], 0.6)
    assert qc == [0.0, 0.0]


def test_build_reward_funcs_phases():
    warmup = reward.build_reward_funcs("warmup")
    full = reward.build_reward_funcs("full")
    assert [f.__name__ for f in warmup] == ["reward_format", "reward_emotion"]
    assert [f.__name__ for f in full] == [
        "reward_format", "reward_emotion",
        "reward_uncertainty", "reward_question_consistency",
    ]
    w_warmup = reward.reward_weights_for_phase("warmup")
    w_full = reward.reward_weights_for_phase("full")
    assert approx(sum(w_warmup), 0.10 + 0.50)
    assert approx(sum(w_full), 0.10 + 0.50 + 0.40 + 0.05)
    try:
        reward.build_reward_funcs("bogus")
        raise AssertionError("expected ValueError for unknown phase")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# metrics.py tests
# ---------------------------------------------------------------------------

def test_compute_competition_score_numeric():
    # Hand-computed example.
    records = [
        {"gt_label": "positive", "sample_type": "Aligned",
         "pred_emotion": "positive", "is_uncertain": "no"},
        {"gt_label": "negative", "sample_type": "Aligned",
         "pred_emotion": "negative", "is_uncertain": "no"},
        {"gt_label": "positive", "sample_type": "Conflict",
         "pred_emotion": "negative", "is_uncertain": "yes"},
        {"gt_label": "negative", "sample_type": "Conflict",
         "pred_emotion": "negative", "is_uncertain": "no"},
    ]
    # Macro-F1 over {positive, negative}:
    #   positive: tp=1, fp=0, fn=1 -> P=1.0, R=0.5, F1=2/3
    #   negative: tp=2, fp=1, fn=0 -> P=2/3, R=1.0, F1=0.8
    #   macro = (2/3 + 0.8) / 2 = 0.733333...
    f1 = (2.0 / 3.0 + 0.8) / 2.0
    ca = 1.0 / 2.0   # one of two conflict samples has is_uncertain=yes
    cal = 2.0 / 2.0  # both aligned samples have is_uncertain=no
    expected = 0.5 * f1 + 0.3 * ca + 0.2 * cal

    out = metrics.compute_competition_score(records)
    assert approx(out["macro_f1"], f1, 1e-12), out["macro_f1"]
    assert approx(out["conflict_accuracy"], ca, 1e-12)
    assert approx(out["aligned_calibration"], cal, 1e-12)
    assert approx(out["score"], expected, 1e-12), out["score"]
    assert out["n_conflict"] == 2 and out["n_aligned"] == 2


def test_compute_competition_score_invalid_predictions():
    # Unparseable prediction -> pred_emotion=None penalized in F1 and
    # is_uncertain=None penalized in calibration denominators.
    records = [
        {"gt_label": "positive", "sample_type": "Aligned",
         "pred_emotion": None, "is_uncertain": None},
        {"gt_label": "positive", "sample_type": "Conflict",
         "pred_emotion": "positive", "is_uncertain": "yes"},
    ]
    out = metrics.compute_competition_score(records)
    assert out["conflict_accuracy"] == 1.0
    assert out["aligned_calibration"] == 0.0  # None counts as a miss
    assert 0.0 <= out["macro_f1"] < 1.0
    assert 0.0 <= out["score"] <= 1.0


def test_compute_competition_score_empty_subsets():
    # No conflict samples -> conflict_accuracy None, weights renormalized.
    records = [
        {"gt_label": "positive", "sample_type": "Aligned",
         "pred_emotion": "positive", "is_uncertain": "no"},
    ]
    out = metrics.compute_competition_score(records)
    assert out["conflict_accuracy"] is None
    assert approx(out["aligned_calibration"], 1.0)
    assert approx(out["score"], (0.5 * 1.0 + 0.2 * 1.0) / 0.7)
    # Fully empty input does not crash.
    empty = metrics.compute_competition_score([])
    assert empty["score"] == 0.0


def test_parse_prediction():
    good = metrics.parse_prediction(
        make_completion("negative", "yes", "What is the tone?"))
    assert good == {"pred_emotion": "negative", "is_uncertain": "yes"}
    bad = metrics.parse_prediction("garbage, no json here")
    assert bad == {"pred_emotion": None, "is_uncertain": None}


def test_macro_f1_manual():
    # Union of labels includes a class predicted but never true.
    y_true = ["positive", "positive", "negative"]
    y_pred = ["positive", "neutral", "negative"]
    # positive: tp=1, fp=0, fn=1 -> F1 = 2/3
    # negative: tp=1, fp=0, fn=0 -> F1 = 1
    # neutral:  tp=0, fp=1, fn=0 -> F1 = 0
    expected = (2.0 / 3.0 + 1.0 + 0.0) / 3.0
    assert approx(metrics.macro_f1(y_true, y_pred), expected, 1e-12)
    assert metrics.macro_f1([], []) == 0.0


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
