"""Real VT/VA numerical and memory checks before any long finite-action run."""
from __future__ import annotations

import time

from .finite_actions import objective
from .finite_runtime import pack_cache, unpack_cache
from .state import adapter_digest


def gradient_vector(policy):
    torch = policy.torch
    return torch.cat([(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).float().cpu().flatten()
                      for p in policy.model.parameters() if p.requires_grad])


def close_gradient(torch, actual, expected, name):
    # Large float32 reductions can even report a cosine slightly above one.
    actual, expected = actual.double(), expected.double()
    relative = float((actual-expected).norm() / expected.norm().clamp(min=1e-12))
    cosine = float(torch.nn.functional.cosine_similarity(actual, expected, dim=0))
    # BF16 kernels permit rounding differences; this is deliberately a numeric
    # tolerance, not bitwise equality. CPU float64 tests use 1e-10 tolerances.
    if relative > .03 or cosine < .999:
        raise RuntimeError(f"{name} gradient mismatch: relative={relative}, cosine={cosine}")
    return {"relative_error": relative, "cosine": cosine}


def preflight(policy, context):
    torch = policy.torch
    settings = context["identity"]["settings"]
    reference = adapter_digest(policy, "reference")
    policy_hash = adapter_digest(policy, "policy")
    result = []
    for protocol in ("VT", "VA"):
        # Inspect several real records and use the shortest processed input for
        # six retained graphs. Training still uses the unmodified full budget.
        rows = [r for r in context["train"] if r["protocol"] == protocol][:4]
        shortest = None
        for row in rows:
            raw = policy.prepare(row, context["data_root"])
            length = raw["input_ids"].shape[1]
            if shortest is None or length < shortest[0]:
                shortest = (length, row)
            del raw
        row = shortest[1]
        prefix = policy.prepare(row, context["data_root"])
        features = policy.perception(prefix)
        packed = pack_cache(torch, prefix, features)
        cached, features = unpack_cache(torch, packed)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with torch.no_grad():
            raw_scores = policy.scores(prefix)
            cached_scores = policy.scores(cached, features)
            ref = policy.scores(cached, features, adapter="reference").double().log_softmax(-1)
            batched = policy.scores(cached, features, micro_batch_size=2)
        for name, score in (("cached", cached_scores),):
            if not torch.allclose(raw_scores, score, atol=.01, rtol=1e-4):
                raise RuntimeError(f"{protocol} {name} action score mismatch")
        if not torch.allclose(ref, cached_scores.double().log_softmax(-1), atol=1e-5, rtol=0):
            raise RuntimeError("Original policy and reference differ")
        _, original_completion = policy.generate(prefix, adapter="reference")
        with policy.cached_perception(features):
            _, cached_completion = policy.generate(cached, adapter="reference")
        if original_completion != cached_completion:
            raise RuntimeError("Free-generation baseline changed with cached perception")
        rewards = torch.tensor([-.3, .2, .5, -.2, .8, .1], device="cuda:0", dtype=torch.float64)
        policy.model.zero_grad(set_to_none=True)
        direct_scores = policy.scores(prefix)
        loss, _ = objective(direct_scores, ref, rewards, settings["beta"])
        loss.backward()
        direct = gradient_vector(policy)
        del loss, direct_scores
        policy.model.zero_grad(set_to_none=True)
        cached_scores = policy.scores(cached, features, micro_batch_size=1)
        loss, _ = objective(cached_scores, ref, rewards, settings["beta"])
        loss.backward()
        cached_gradient = gradient_vector(policy)
        cache_check = close_gradient(torch, cached_gradient, direct, "cache/candidate_microbatch1")
        del loss, cached_scores, cached_gradient
        policy.model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        replay_start = time.monotonic()
        policy.backward_actions(cached, features, ref, rewards, divisor=1)
        torch.cuda.synchronize()
        replay_seconds = time.monotonic() - replay_start
        replay_check = close_gradient(torch, gradient_vector(policy), direct, "two-pass")
        del direct
        policy.model.zero_grad(set_to_none=True)
        # Exercise the frozen question generator even if argmax says no.
        raw_answer, _ = policy.answer(cached, features, action=0)
        from .reward import parse_completion, question_is_consistent
        parsed_answer = parse_completion(raw_answer)
        if parsed_answer is None or not question_is_consistent(parsed_answer):
            raise RuntimeError("Frozen v4 cannot produce a valid conditional question")
        result.append({"protocol": protocol, "sample_id": row["sample_id"], "prompt_tokens": shortest[0],
                       "cache_gradient": cache_check, "replay_gradient": replay_check,
                       "unsupported_batch2_score_max_difference": float((raw_scores-batched).abs().max()),
                       "unsupported_batch2_probability_max_difference": float((raw_scores.double().softmax(-1)-batched.double().softmax(-1)).abs().max()),
                       "question_probe": raw_answer,
                       "seconds_per_input_backward": replay_seconds,
                       "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                       "elapsed_seconds": time.monotonic()-started})
        print("PREFLIGHT", result[-1], flush=True)
        del prefix, features, packed, cached
        torch.cuda.empty_cache()
    if adapter_digest(policy, "reference") != reference or adapter_digest(policy, "policy") != policy_hash:
        raise RuntimeError("Preflight must not change any adapter")
    weights = {p: sum(r["protocol"] == p for r in context["train"]) for p in ("VT", "VA")}
    seconds = sum(r["seconds_per_input_backward"]*weights[r["protocol"]] for r in result)
    return {"passed": True, "checks": result, "rough_train_epoch_hours": seconds/3600,
            "estimate_note": "Short VT/VA probes only; excludes caching/evaluation and is optimistic for longer clips. "
                             "Use running full-data throughput for the updated ETA.", "reference_digest": reference}
