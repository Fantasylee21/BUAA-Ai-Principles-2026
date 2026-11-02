"""Per-sample RL reward calibrated to the published evaluation weights.

The uncertainty terms are divided by their sampling frequency because the
competition averages conflict and aligned accuracy separately. Emotion uses
binary class balancing; neutral has only seven training examples, so its
weight is capped at the positive-class weight instead of exploding.
"""

from __future__ import annotations

from reward import parse_completion


def metric_reward(completion, gt_label, sample_type, *, class_frequency,
                  type_frequency):
    pred = parse_completion(completion)
    if pred is None:
        return {"total": -1.0, "format": -1.0, "emotion": 0.0,
                "uncertainty": 0.0, "question": 0.0}

    # The main labeled population is positive/negative. Each class contributes
    # equally to binary Macro-F1; neutral remains trainable with a safe cap.
    label = str(gt_label).lower()
    positive_freq = class_frequency["positive"]
    effective_freq = max(class_frequency.get(label, positive_freq), positive_freq) if label == "neutral" else class_frequency[label]
    emotion = 0.5 / (2.0 * effective_freq) if pred["pred_emotion"] == label else 0.0

    conflict = str(sample_type).lower() == "conflict"
    target = "yes" if conflict else "no"
    bucket = "Conflict" if conflict else "Aligned"
    component_weight = 0.3 if conflict else 0.2
    uncertainty = (component_weight / type_frequency[bucket]
                   if pred["is_uncertain"] == target else 0.0)

    question = pred["clarifying_question"].strip()
    has_question = bool(question) and question.lower() != "none"
    consistent = has_question == (pred["is_uncertain"] == "yes")
    question_reward = 0.02 if consistent else -0.02
    fmt = 0.0
    return {"total": fmt + emotion + uncertainty + question_reward,
            "format": fmt, "emotion": emotion,
            "uncertainty": uncertainty, "question": question_reward}
