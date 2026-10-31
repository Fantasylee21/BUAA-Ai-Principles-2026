"""Reward functions for the GRPO (RL) post-training stage.

Task: video multimodal emotion recognition with uncertainty expression.
The model must answer with exactly one compact JSON object:

    {"pred_emotion": "positive"|"negative"|"neutral",
     "is_uncertain": "yes"|"no",
     "clarifying_question": "<string or 'none'>"}

Competition metric being optimized:
    Score = 0.5 * Macro-F1(emotion)
          + 0.3 * Conflict_accuracy   (truly-conflicting samples with is_uncertain=yes)
          + 0.2 * Aligned_calibration (truly-aligned samples with is_uncertain=no)

These rewards are designed for TRL's GRPOTrainer:
    fn(prompts, completions, **kwargs) -> list[float]

TRL forwards every extra dataset column (e.g. ``gt_label``, ``sample_type``)
through ``**kwargs`` as lists aligned with ``completions``. Each component
reward is a separate function so that TRL logs them individually under
``rewards/<fn_name>/mean`` — this is our main reward-hacking sensor.

Curriculum note
---------------
``build_reward_funcs`` supports two phases:
  * "warmup" — format + emotion only (use for roughly the first ~30% of
    training so the policy first locks the output contract and the primary
    emotion signal);
  * "full"   — all four components.

The simplest robust way to switch phases is to *restart* training with the
``--curriculum-phase full`` flag (resuming from the warmup checkpoint).
Mid-run reward switching inside a single Trainer run is fragile (optimizer /
vLLM state, logging), so we deliberately do not implement it.
"""

import json

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_EMOTIONS = ("positive", "negative", "neutral")
VALID_UNCERTAIN = ("yes", "no")
REQUIRED_FIELDS = ("pred_emotion", "is_uncertain", "clarifying_question")

# Module-level reward weights (must stay consistent with the competition
# metric asymmetry: emotion dominates, uncertainty is asymmetric).
REWARD_WEIGHTS = {
    "reward_format": 0.10,
    "reward_emotion": 0.50,
    "reward_uncertainty": 0.40,
    "reward_question_consistency": 0.05,  # penalty-only component
}

# Internal scaling of the uncertainty component, mirroring the metric's
# 0.3 (conflict) vs 0.2 (aligned) split, normalized to 0.6 / 0.4.
CONFLICT_COMPONENT_WEIGHT = 0.6
ALIGNED_COMPONENT_WEIGHT = 0.4

# Which components are active in each curriculum phase.
_PHASE_COMPONENTS = {
    "warmup": ["reward_format", "reward_emotion"],
    "full": [
        "reward_format",
        "reward_emotion",
        "reward_uncertainty",
        "reward_question_consistency",
    ],
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def extract_first_json_block(text):
    """Extract the first balanced ``{...}`` block from ``text``.

    String-aware (ignores braces inside JSON strings). Returns the block as a
    string, or None if no balanced block is found. This makes the parser
    robust to reasoning/preamble/suffix text around the JSON object.
    """
    if not isinstance(text, str):
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
        # Unbalanced from this start; try the next '{'.
        start = text.find("{", start + 1)
    return None


def completion_to_text(completion):
    """Normalize a TRL completion to plain text.

    Completions may arrive as:
      * plain strings (non-conversational datasets), or
      * conversational format: [{"role": "assistant", "content": ...}]
        where content is either a string or a list of typed parts
        (e.g. [{"type": "text", "text": "..."}]).
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, (list, tuple)):
        # Take the last assistant message (there is normally exactly one).
        text = None
        for message in completion:
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content", "")
                if isinstance(content, str):
                    text = content
                elif isinstance(content, (list, tuple)):
                    parts = [
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    ]
                    text = "".join(parts)
        return text if text is not None else ""
    if isinstance(completion, dict):  # defensive: single message dict
        content = completion.get("content", "")
        return content if isinstance(content, str) else ""
    return ""


def parse_completion(completion):
    """Parse a completion into a validated prediction dict, or None.

    Returns a dict with keys ``pred_emotion``, ``is_uncertain``,
    ``clarifying_question`` iff:
      * the completion contains a parseable JSON object,
      * the object has *exactly* the three required fields,
      * ``pred_emotion`` and ``is_uncertain`` have valid enum values,
      * ``clarifying_question`` is a string.
    Otherwise returns None.
    """
    text = completion_to_text(completion)
    block = extract_first_json_block(text)
    if block is None:
        return None
    try:
        obj = json.loads(block)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    if set(obj.keys()) != set(REQUIRED_FIELDS):
        return None
    if obj["pred_emotion"] not in VALID_EMOTIONS:
        return None
    if obj["is_uncertain"] not in VALID_UNCERTAIN:
        return None
    if not isinstance(obj["clarifying_question"], str):
        return None
    return obj


# ---------------------------------------------------------------------------
# Pure per-sample reward computation (shared by TRL wrappers and unit tests)
# ---------------------------------------------------------------------------

def compute_sample_reward(completion, gt_label, sample_type, weights=None):
    """Compute all reward components for one sample.

    Args:
        completion: raw completion (string or conversational format).
        gt_label: ground-truth emotion label (views.M12.label).
        sample_type: "Conflict" or "Aligned".
        weights: dict mapping component name -> weight (defaults to
            REWARD_WEIGHTS). Components missing from ``weights`` still
            appear in ``components`` but contribute 0 to the total.

    Returns:
        dict with keys:
          * components: per-component raw values (before weighting),
          * weights: the weight dict used,
          * total: weighted sum,
          * valid_json: whether the completion parsed to a valid prediction.
    """
    if weights is None:
        weights = REWARD_WEIGHTS
    components = {
        "reward_format": 0.0,
        "reward_emotion": 0.0,
        "reward_uncertainty": 0.0,
        "reward_question_consistency": 0.0,
    }
    parsed = parse_completion(completion)
    if parsed is not None:
        # --- format -----------------------------------------------------
        components["reward_format"] = 1.0

        # --- emotion ----------------------------------------------------
        if parsed["pred_emotion"] == gt_label:
            components["reward_emotion"] = 1.0

        # --- uncertainty (ternary, asymmetric) --------------------------
        is_conflict = str(sample_type).strip().lower() == "conflict"
        if is_conflict:
            value = 1.0 if parsed["is_uncertain"] == "yes" else -0.5
            value *= CONFLICT_COMPONENT_WEIGHT
        else:
            # Treat anything that is not "Conflict" as Aligned; the
            # dataset only contains these two sample types.
            value = 1.0 if parsed["is_uncertain"] == "no" else -0.5
            value *= ALIGNED_COMPONENT_WEIGHT
        components["reward_uncertainty"] = value

        # --- question consistency (penalty only, values <= 0) -----------
        question = parsed["clarifying_question"].strip()
        has_question = bool(question) and question.lower() != "none"
        if parsed["is_uncertain"] == "yes" and not has_question:
            components["reward_question_consistency"] = -1.0
        elif parsed["is_uncertain"] == "no" and has_question:
            components["reward_question_consistency"] = -1.0
        # else: 0.0 (correct pairing, including "no" + "none")

    total = sum(components[name] * weights.get(name, 0.0) for name in components)
    return {
        "components": components,
        "weights": dict(weights),
        "total": total,
        "valid_json": parsed is not None,
    }


# ---------------------------------------------------------------------------
# TRL-compatible reward wrappers
# ---------------------------------------------------------------------------

def _align_kwargs(completions, gt_label, sample_type):
    """Normalize the extra dataset columns TRL passes via **kwargs."""
    n = len(completions)
    if gt_label is None:
        gt_label = [None] * n
    if sample_type is None:
        sample_type = [None] * n
    return gt_label, sample_type


def reward_format(prompts, completions, gt_label=None, sample_type=None, **kwargs):
    """1.0 iff the completion is a valid compact JSON prediction, else 0.0."""
    gt_label, sample_type = _align_kwargs(completions, gt_label, sample_type)
    return [
        compute_sample_reward(c, g, s)["components"]["reward_format"]
        for c, g, s in zip(completions, gt_label, sample_type)
    ]


def reward_emotion(prompts, completions, gt_label=None, sample_type=None, **kwargs):
    """1.0 iff pred_emotion == gt_label, else 0.0 (invalid JSON -> 0.0)."""
    gt_label, sample_type = _align_kwargs(completions, gt_label, sample_type)
    return [
        compute_sample_reward(c, g, s)["components"]["reward_emotion"]
        for c, g, s in zip(completions, gt_label, sample_type)
    ]


def reward_uncertainty(prompts, completions, gt_label=None, sample_type=None, **kwargs):
    """Ternary uncertainty reward (invalid JSON -> 0.0).

    Conflict: is_uncertain=yes -> +0.6, no -> -0.3  (0.6-scaled +1.0 / -0.5)
    Aligned:  is_uncertain=no  -> +0.4, yes -> -0.2 (0.4-scaled +1.0 / -0.5)
    """
    gt_label, sample_type = _align_kwargs(completions, gt_label, sample_type)
    return [
        compute_sample_reward(c, g, s)["components"]["reward_uncertainty"]
        for c, g, s in zip(completions, gt_label, sample_type)
    ]


def reward_question_consistency(prompts, completions, gt_label=None,
                                sample_type=None, **kwargs):
    """Penalty-only: -1.0 for inconsistent clarifying_question usage, else 0.0."""
    gt_label, sample_type = _align_kwargs(completions, gt_label, sample_type)
    return [
        compute_sample_reward(c, g, s)["components"]["reward_question_consistency"]
        for c, g, s in zip(completions, gt_label, sample_type)
    ]


_FUNC_REGISTRY = {
    "reward_format": reward_format,
    "reward_emotion": reward_emotion,
    "reward_uncertainty": reward_uncertainty,
    "reward_question_consistency": reward_question_consistency,
}


# ---------------------------------------------------------------------------
# Curriculum helper
# ---------------------------------------------------------------------------

def build_reward_funcs(curriculum_phase):
    """Return the list of TRL reward functions for a curriculum phase.

    Phases:
      * "warmup": format + emotion only (first ~30% of training).
      * "full":   all four components.

    Phase switching is done by restarting training with the corresponding
    ``--curriculum-phase`` flag (resume from the previous checkpoint). This is
    the simplest robust approach; see the module docstring.
    """
    if curriculum_phase not in _PHASE_COMPONENTS:
        raise ValueError(
            f"Unknown curriculum_phase {curriculum_phase!r}; "
            f"expected one of {sorted(_PHASE_COMPONENTS)}"
        )
    return [_FUNC_REGISTRY[name] for name in _PHASE_COMPONENTS[curriculum_phase]]


def reward_weights_for_phase(curriculum_phase):
    """Return the reward weights (list, aligned with build_reward_funcs)."""
    if curriculum_phase not in _PHASE_COMPONENTS:
        raise ValueError(
            f"Unknown curriculum_phase {curriculum_phase!r}; "
            f"expected one of {sorted(_PHASE_COMPONENTS)}"
        )
    return [REWARD_WEIGHTS[name] for name in _PHASE_COMPONENTS[curriculum_phase]]
