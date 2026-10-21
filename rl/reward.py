"""Verifiable sampling-corrected rewards, independent of GPU libraries.

correctness_v1 uses accuracy as a proxy, NOT exact Macro-F1. f1_delta_v1
is an experimental bounded change to a frozen TRAIN-only confusion matrix.
"""
from __future__ import annotations

import json
import math
from collections import Counter

VALID_EMOTIONS = ("positive", "negative", "neutral")
VALID_UNCERTAIN = ("yes", "no")
REQUIRED_FIELDS = {"pred_emotion", "is_uncertain", "clarifying_question"}
REWARD_VERSIONS = ("correctness_v1", "f1_delta_v1")
INVALID_REWARD = -2.0
QUESTION_PENALTY = 0.02


def completion_to_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        completion = [completion]
    if isinstance(completion, (list, tuple)):
        for message in reversed(completion):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(p.get("text", "") for p in content
                               if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate JSON key: {key}")
        obj[key] = value
    return obj


def parse_completion(completion):
    """Accept exactly one JSON object; reject preambles and duplicate keys."""
    try:
        obj = json.loads(completion_to_text(completion).strip(), object_pairs_hook=_unique_object)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or set(obj) != REQUIRED_FIELDS:
        return None
    if obj["pred_emotion"] not in VALID_EMOTIONS or obj["is_uncertain"] not in VALID_UNCERTAIN:
        return None
    if not isinstance(obj["clarifying_question"], str):
        return None
    return obj


def question_is_consistent(prediction):
    question = prediction["clarifying_question"].strip()
    if prediction["is_uncertain"] == "no":
        return prediction["clarifying_question"] == "none"
    return bool(question) and question.lower() != "none"


def fixed_macro_f1(matrix):
    """Three true rows, with optional fourth prediction column for invalids."""
    total = 0.0
    for c in range(3):
        denominator = sum(matrix[c]) + sum(row[c] for row in matrix)
        total += 2 * matrix[c][c] / denominator if denominator else 0.0
    return total / 3


class FrozenF1Reference:
    def __init__(self, records, predictions):
        if set(predictions) != {r["sample_id"] for r in records}:
            raise ValueError("F1 cache IDs must match the unique training IDs exactly")
        self.matrix = [[0] * 4 for _ in range(3)]
        self.samples = {}
        for row in records:
            sid, label = row["sample_id"], row["gt_label"]
            if label not in VALID_EMOTIONS:
                raise ValueError(f"Invalid gold emotion for {sid}")
            pred = predictions[sid]
            y = VALID_EMOTIONS.index(label)
            b = VALID_EMOTIONS.index(pred) if pred in VALID_EMOTIONS else 3
            self.matrix[y][b] += 1
            self.samples[sid] = (y, b)
        self.base_f1 = fixed_macro_f1(self.matrix)
        self.n = len(records)

    def emotion_reward(self, sample_id, gt_label, prediction):
        y, original = self.samples[sample_id]
        if VALID_EMOTIONS[y] != gt_label:
            raise ValueError("F1 cache gold label mismatch")
        matrix = [row[:] for row in self.matrix]
        matrix[y][original] -= 1
        matrix[y][VALID_EMOTIONS.index(prediction)] += 1
        return max(-1.0, min(1.0, self.n * (fixed_macro_f1(matrix) - self.base_f1)))


class RewardEngine:
    def __init__(self, population_probabilities, sampling_probabilities=None,
                 version="correctness_v1", f1_reference=None):
        self.population = dict(population_probabilities)
        self.sampling = dict(sampling_probabilities or {"Conflict": .5, "Aligned": .5})
        for name, probabilities in (("population", self.population), ("sampling", self.sampling)):
            if set(probabilities) != {"Conflict", "Aligned"}:
                raise ValueError(f"{name}: both sample types are required")
            if any(not math.isfinite(p) or p <= 0 for p in probabilities.values()):
                raise ValueError(f"{name}: probabilities must be finite and positive")
            if not math.isclose(sum(probabilities.values()), 1.0):
                raise ValueError(f"{name}: probabilities must sum to one")
        # This bound makes the invalid penalty strictly lower than any valid reward.
        if self.sampling != {"Conflict": .5, "Aligned": .5}:
            raise ValueError("v1 rewards require the fixed 50:50 type sampler")
        if version not in REWARD_VERSIONS:
            raise ValueError(f"Unknown reward version: {version}")
        if version == "f1_delta_v1" and f1_reference is None:
            raise ValueError("f1_delta_v1 requires a validated TRAIN-only F1 cache")
        self.version, self.f1_reference = version, f1_reference

    @classmethod
    def from_records(cls, records, **kwargs):
        if not records:
            raise ValueError("Cannot derive reward prevalence from empty data")
        counts = Counter(row["sample_type"] for row in records)
        return cls({key: counts[key] / len(records) for key in ("Conflict", "Aligned")}, **kwargs)

    def score(self, completion, gt_label, sample_type, sample_id=None):
        if gt_label not in VALID_EMOTIONS or sample_type not in self.population:
            raise ValueError("Missing or invalid reward gold metadata")
        parsed = parse_completion(completion)
        result = {"total": INVALID_REWARD, "valid_json": parsed is not None,
                  "emotion": 0.0, "uncertainty": 0.0, "question_penalty": 0.0}
        if parsed is None:
            return result
        emotion = float(parsed["pred_emotion"] == gt_label)
        if self.version == "f1_delta_v1":
            emotion = self.f1_reference.emotion_reward(sample_id, gt_label, parsed["pred_emotion"])
        result["emotion"] = .5 * self.population[sample_type] / self.sampling[sample_type] * emotion
        target = "yes" if sample_type == "Conflict" else "no"
        weight = .3 if sample_type == "Conflict" else .2
        result["uncertainty"] = weight / self.sampling[sample_type] * (parsed["is_uncertain"] == target)
        result["question_penalty"] = 0.0 if question_is_consistent(parsed) else -QUESTION_PENALTY
        result["total"] = result["emotion"] + result["uncertainty"] + result["question_penalty"]
        return result

    def __call__(self, prompts, completions, gt_label, sample_type, sample_id, **kwargs):
        """TRL-compatible reward callable; fail on misaligned metadata."""
        n = len(completions)
        if any(len(values) != n for values in (prompts, gt_label, sample_type, sample_id)):
            raise ValueError("Reward inputs have different lengths")
        return [self.score(c, g, t, sid)["total"]
                for c, g, t, sid in zip(completions, gt_label, sample_type, sample_id)]


def compute_sample_reward(completion, gt_label, sample_type, *, population_probabilities,
                          version="correctness_v1", f1_reference=None, sample_id=None):
    return RewardEngine(population_probabilities, version=version,
                        f1_reference=f1_reference).score(completion, gt_label, sample_type, sample_id)
