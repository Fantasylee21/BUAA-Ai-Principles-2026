"""Deterministic CUDA setup and quantitative, exact recovery diagnostics."""
from __future__ import annotations

import hashlib
import json
import os

EXECUTION = {"deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
             "cudnn_deterministic": True, "cudnn_benchmark": False, "allow_tf32": False}


def prepare_environment():
    # cuBLAS reads this when its CUDA handle is initialized, before model loading.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = EXECUTION["cublas_workspace_config"]


def configure_torch(torch):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def snapshot(policy, optimizer, scheduler, *, gradients=False):
    torch = policy.torch
    parameters = {name: parameter.detach().cpu().clone() for name, parameter in policy.model.named_parameters()
                  if parameter.requires_grad}
    state = optimizer.state_dict()
    tensors = {f"{index}/{key}": value.detach().cpu().clone()
               for index, values in state["state"].items() for key, value in values.items() if torch.is_tensor(value)}
    non_tensors = {f"{index}/{key}": value
                  for index, values in state["state"].items() for key, value in values.items() if not torch.is_tensor(value)}
    result = {"parameters": parameters, "optimizer": tensors,
              "metadata": {"param_groups": state["param_groups"], "non_tensors": non_tensors,
                           "scheduler": scheduler.state_dict()}}
    if gradients:
        result["gradients"] = {name: parameter.grad.detach().cpu().clone()
            for name, parameter in policy.model.named_parameters() if parameter.requires_grad and parameter.grad is not None}
    return result


def tensor_difference(torch, left, right):
    if set(left) != set(right):
        return {"exact": False, "reason": "tensor membership changed"}
    maximum, squared, reference, exact, worst = 0., 0., 0., True, None
    for name in sorted(left):
        original, restored = left[name], right[name]
        if original.shape != restored.shape or original.dtype != restored.dtype:
            return {"exact": False, "reason": "tensor shape/dtype changed", "worst": name}
        a, b = original.double(), restored.double()
        delta = a-b
        error = float(delta.abs().max()) if delta.numel() else 0.
        if error > maximum:
            maximum, worst = error, name
        squared += float(delta.square().sum())
        reference += float(a.square().sum())
        exact = exact and torch.equal(original, restored)
    return {"exact": exact, "max_abs": maximum, "relative_l2": (squared/max(reference, 1e-30))**.5,
            "worst": worst}


def compare_snapshots(torch, left, right):
    result = {key: tensor_difference(torch, left[key], right[key]) for key in ("parameters", "optimizer")}
    if "gradients" in left or "gradients" in right:
        result["gradients"] = tensor_difference(torch, left.get("gradients", {}), right.get("gradients", {}))
    result["metadata_exact"] = left["metadata"] == right["metadata"]
    result["exact"] = result["metadata_exact"] and all(value["exact"] for value in result.values() if isinstance(value, dict))
    return result


def task_differences(original, restored):
    fields = ("sample_id", "draw_id", "mode", "target", "selection", "weight")
    membership = len(original) == len(restored) and all(
        all(a.get(key) == b.get(key) for key in fields) for a, b in zip(original, restored))
    maximum = 0.
    for a, b in zip(original, restored):
        if len(a["scores"]) != len(b["scores"]):
            membership = False
        maximum = max([maximum, *(abs(x-y) for x, y in zip(a["scores"], b["scores"]))])
    encoded = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return {"same_samples_modes_targets": membership, "maximum_score_difference": maximum,
            "exact": original == restored,
            "original_sha256": hashlib.sha256(encoded(original)).hexdigest(),
            "restored_sha256": hashlib.sha256(encoded(restored)).hexdigest()}
