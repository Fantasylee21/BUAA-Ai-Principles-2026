"""Train, preflight and predict isolated-modality joint affect models.

Long jobs run only through this explicit CLI. Importing this module is CPU-safe.
The frozen pipeline manifest controls all folds; official inference needs a lock.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid

from scripts import joint_affect_data as data

VERSION = "joint_affect_trainer_v1"
SOURCE = Path(__file__).resolve().parents[1]
STOP_REQUESTED = False
EFFECTIVE_BATCH = 16


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_bytes(encoded(value))
    temporary.replace(path)


def immutable(path, value):
    path = Path(path)
    payload = encoded(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("Immutable training artifact changed: " + str(path))
    else:
        with path.open("xb") as stream:
            stream.write(payload)


def directory_hash(root):
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
    if not files:
        raise ValueError("Empty checkpoint directory")
    return fingerprint({p.relative_to(root).as_posix(): digest(p) for p in files})


def emit(args, phase, **values):
    value = {"algorithm": VERSION, "phase": phase, "updated_unix": time.time(),
             "pid": os.getpid(), "variant": args.variant, "fold": args.fold,
             "grouping": args.grouping, "seed": args.seed, **values}
    atomic(args.run_dir / "status.json", value)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def stopped(args):
    return STOP_REQUESTED or time.time() >= args.deadline_unix


def job_contract(args):
    return {"variant": args.variant, "fold": args.fold, "grouping": args.grouping,
            "seed": args.seed, "vision_model": args.vision_model, "max_frames": args.max_frames,
            "manifest_sha256": digest(args.manifest)}


def load_context(args):
    args.project_root = args.project_root.resolve()
    args.run_dir = args.run_dir.resolve()
    if not args.run_dir.is_relative_to(args.project_root / "runs"):
        raise ValueError("Joint affect output must remain under the project runs directory")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = read(args.manifest)
    data.validate_manifest(manifest)
    if fingerprint(manifest["identity"]) != manifest["identity_hash"]:
        raise ValueError("Joint-affect manifest identity changed")
    if digest(manifest["train_jsonl"]) != manifest["identity"]["train_sha256"]:
        raise ValueError("Original training source changed")
    rows = manifest["train"]
    if args.fold == "full":
        train, held = list(range(len(rows))), []
    else:
        fold = next(x for x in manifest["folds"][args.grouping]["folds"] if x["fold"] == args.fold)
        train, held = fold["train"], fold["held"]
        if set(train) & set(held) or set(train) | set(held) != set(range(len(rows))):
            raise ValueError("Train/held coverage or isolation changed")
    selected = {rows[i]["video"] for i in set(train) | set(held)}
    dataset = Path(manifest["data_root"]).resolve()
    for video in sorted(selected):
        path = (dataset / video).resolve()
        if not path.is_relative_to(dataset) or digest(path) != manifest["media"][video]:
            raise ValueError("Frozen media changed: " + video)
    source_files = ["scripts/train_joint_affect.py", "scripts/joint_affect_model.py", "scripts/joint_affect_data.py"]
    identity = {"algorithm": VERSION, "job": job_contract(args),
                "source_sha256": {name: digest(SOURCE / name) for name in source_files},
                "effective_batch": EFFECTIVE_BATCH, "warmup_epochs": 1, "main_epochs": 2,
                "reference_epochs": 3, "threshold": .5, "threshold_inclusive": True,
                "optimizer": {"name": "AdamW", "language_lr": 1e-4, "perception_connector_lr": 2e-5,
                              "head_lr": 3e-4, "weight_decay": .01, "warmup_fraction": .1, "gradient_clip": 1.},
                "init_second_from": str(args.init_second_from.resolve()) if args.init_second_from else None,
                "init_second_sha256": directory_hash(args.init_second_from) if args.init_second_from else None}
    plan_file = args.manifest.parent / "plan.json"
    if plan_file.is_file():
        from scripts.joint_affect_pipeline import verify_environment, fingerprint as plan_fingerprint
        plan = read(plan_file)
        if plan_fingerprint(plan["identity"]) != plan["identity_hash"]:
            raise ValueError("Corrupt parent experiment identity")
        verify_environment(args.project_root, plan["identity"])
        for name, expected in plan["identity"]["source_sha256"].items():
            if digest(SOURCE / name) != expected:
                raise ValueError("Source differs from the frozen parent experiment: " + name)
        identity["parent_plan_sha256"] = digest(plan_file)
    immutable(args.run_dir / "identity.json", {"identity": identity, "identity_hash": fingerprint(identity)})
    return manifest, rows, train, held, identity


def devices(args):
    requested = args.gpus.split(",") if args.gpus else [str(args.gpu)]
    if not requested or len(set(requested)) != len(requested) or any(x not in ("1", "2") for x in requested):
        raise ValueError("Only the authorized physical GPUs 1 and 2 may be selected")
    if args.vision_model == "qwen3vl8b" and len(requested) != 2:
        raise ValueError("The visual upgrade uses one expert on each of two GPUs")
    if args.vision_model == "omni" and len(requested) != 1:
        raise ValueError("Native Omni jobs use one GPU, allowing independent jobs in parallel")
    return requested


def setup_torch(args):
    requested = devices(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(requested)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or torch.cuda.device_count() != len(requested):
        raise RuntimeError("Requested CUDA devices are unavailable")
    for i in range(len(requested)):
        torch.cuda.set_per_process_memory_fraction(min(1., 40 * 1024**3 / torch.cuda.get_device_properties(i).total_memory), i)
        torch.cuda.reset_peak_memory_stats(i)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    return torch


def make_model(args, manifest, torch):
    from scripts.joint_affect_model import HFBackend, JointAffectModel
    backend_config = {"omni_model_path": str(args.project_root / "models/Qwen2.5-Omni-3B"),
                      "vision_model_path": str(args.project_root / "models/Qwen3-VL-8B-Instruct"),
                      "dataset_root": manifest["data_root"], "vision_model": args.vision_model,
                      "devices": tuple("cuda:" + str(i) for i in range(len(devices(args)))),
                      "max_frames": args.max_frames, "video_fps": 2., "video_pixels": 50176,
                      "memory_gib": 40, "seed": args.seed,
                      "media_cache_dir": str(args.run_dir.parent / "media-cache"),
                      "init_second_from": str(args.init_second_from) if args.init_second_from else None}
    backend = HFBackend(backend_config)
    model = JointAffectModel(backend, variant=args.variant)
    return model


def state_rng(torch):
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(torch, state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def save_state(args, model, optimizer, scheduler, torch, position, rng=None):
    root = args.run_dir / "checkpoints"
    root.mkdir(exist_ok=True)
    name = "step-" + str(position["global_step"]).zfill(7) + "-" + uuid.uuid4().hex[:8]
    temporary = root / (name + ".tmp")
    temporary.mkdir()
    model.save_checkpoint(temporary, metadata={"job": job_contract(args), "position": position})
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "position": position, "rng": rng or state_rng(torch)}, temporary / "training-state.pt")
    final = root / name
    temporary.rename(final)
    value = {"checkpoint": str(final), "checkpoint_sha256": directory_hash(final), "position": position}
    atomic(args.run_dir / "latest.json", value)
    return value


def validate_position(position, n_real):
    if (set(position) != {"epoch", "offset", "global_step"} or
            any(type(value) is not int for value in position.values()) or n_real <= 0):
        raise ValueError("Invalid checkpoint training position")
    epoch, offset, step = (position[k] for k in ("epoch", "offset", "global_step"))
    if (not 0 <= epoch <= 3 or offset < 0 or offset >= n_real or offset % EFFECTIVE_BATCH or
            (epoch == 3 and offset != 0) or
            step != epoch * math.ceil(n_real / EFFECTIVE_BATCH) + offset // EFFECTIVE_BATCH):
        raise ValueError("Invalid checkpoint epoch/offset/step relationship")
    return position


def load_state(args, model, optimizer, scheduler, torch, n_real=None):
    path = args.run_dir / "latest.json"
    if not path.exists():
        return {"epoch": 0, "offset": 0, "global_step": 0}
    latest = read(path)
    if directory_hash(latest["checkpoint"]) != latest["checkpoint_sha256"]:
        raise ValueError("Saved training checkpoint changed")
    state = torch.load(Path(latest["checkpoint"]) / "training-state.pt", map_location="cpu", weights_only=False)
    metadata = read(Path(latest["checkpoint"]) / "trainable.json")["metadata"]
    if (metadata.get("job") != job_contract(args) or
            metadata.get("position") != state["position"] or latest.get("position") != state["position"]):
        raise ValueError("Checkpoint job or position disagrees with its completion pointer")
    if n_real is not None:
        validate_position(state["position"], n_real)
    model.load_checkpoint(latest["checkpoint"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng(torch, state["rng"])
    return state["position"]


def optimizer_for(model, torch, steps):
    groups = model.optimizer_groups(backbone_lr=1e-4, head_lr=3e-4, weight_decay=.01)
    optimizer = torch.optim.AdamW(groups)
    warmup = max(1, math.ceil(steps * .1))
    def scale(step):
        if step < warmup:
            return (step + 1) / warmup
        return .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, steps - warmup))))
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def sample_loss(model, row, rows, counts, phase):
    from scripts.joint_affect_model import compute_objectives, make_targets
    second = data.second_source(row, rows) if row.get("is_remix") else None
    output = model.forward_row(row, second_row=second)
    target = make_targets(row, second_row=second)
    losses = compute_objectives(output, target, counts, phase=phase, remix=bool(row.get("is_remix")))
    return losses, output


def train_new(args, manifest, rows, train, identity):
    torch = setup_torch(args)
    emit(args, "loading_backbones")
    model = make_model(args, manifest, torch)
    counts = data.class_counts(rows, train)
    steps_per_epoch = math.ceil(len(train) / EFFECTIVE_BATCH)
    total_steps = 3 * steps_per_epoch
    optimizer, scheduler = optimizer_for(model, torch, total_steps)
    position = load_state(args, model, optimizer, scheduler, torch, n_real=len(train))
    immutable(args.run_dir / "trainable-parameters.json", model.trainable_audit())
    roles = model.trainable_audit().get("parameter_roles", {})
    observed = Counter()
    started = time.monotonic()
    initial_step = position["global_step"]
    for epoch in range(position["epoch"], 3):
        phase = "evidence_warmup" if epoch == 0 else "main"
        order = list(train)
        random.Random(args.seed + 1009 * epoch).shuffle(order)
        remix = data.remix_training(rows, train, manifest["media"], args.seed, epoch=epoch) if epoch else []
        model.train()
        for offset in range(position["offset"] if epoch == position["epoch"] else 0, len(order), EFFECTIVE_BATCH):
            before_batch = state_rng(torch)
            selected = [rows[i] for i in order[offset:offset + EFFECTIVE_BATCH]]
            mixed = remix[offset // 2:(offset + len(selected)) // 2] if epoch else []
            optimizer.zero_grad(set_to_none=True)
            measured = {"real_loss": 0., "remix_loss": 0.}
            try:
                for row in selected:
                    if stopped(args):
                        raise InterruptedError("Training deadline or stop requested")
                    losses, outputs = sample_loss(model, row, rows, counts, phase)
                    loss = losses["warmup_loss" if epoch == 0 else "main_loss"]
                    if not torch.isfinite(loss).all():
                        raise FloatingPointError("Nonfinite real training objective")
                    (loss / len(selected)).backward()
                    measured["real_loss"] += float(loss.detach().cpu()) / len(selected)
                    if outputs is not None and "vision_probs" in outputs:
                        target = data.targets(row)
                        for branch, key in (("vision", "vision_probs"), ("second", "second_probs")):
                            observed[branch + "_n"] += 1
                            observed[branch + "_correct"] += int(int(outputs[key].detach().argmax(-1)) == target[branch])
                        if row["protocol"] == "VA" and row.get("source_dataset") in data.NATURAL and target["relation"]:
                            observed["natural_va_n"] += 1
                            observed["natural_va_correct"] += int(float(outputs["uncertain_prob"].detach()) >= .5)
                    for key in ("vision_loss", "second_loss", "relation_loss", "joint_loss"):
                        if key in losses:
                            measured[key] = measured.get(key, 0.) + float(losses[key].detach().cpu()) / len(selected)
                for row in mixed:
                    if stopped(args):
                        raise InterruptedError("Training deadline or stop requested")
                    losses, _ = sample_loss(model, row, rows, counts, phase)
                    loss = losses["remix_loss"]
                    if not torch.isfinite(loss).all():
                        raise FloatingPointError("Nonfinite remix training objective")
                    (loss / len(mixed)).backward()
                    measured["remix_loss"] += float(loss.detach().cpu()) / len(mixed)
            except InterruptedError:
                optimizer.zero_grad(set_to_none=True)
                position.update(epoch=epoch, offset=offset)
                save_state(args, model, optimizer, scheduler, torch, position, rng=before_batch)
                emit(args, "stopped_checkpoint_saved", position=position)
                raise
            module_gradients = Counter()
            if roles:
                for name, parameter in model.named_parameters():
                    if name in roles and parameter.grad is not None:
                        module_gradients[roles[name]] = max(module_gradients[roles[name]], float(parameter.grad.detach().float().norm().cpu()))
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            position = {"epoch": epoch, "offset": offset + len(selected), "global_step": position["global_step"] + 1}
            if position["offset"] == len(order):
                position.update(epoch=epoch + 1, offset=0)
            elapsed = time.monotonic() - started
            duration = elapsed / max(1, position["global_step"] - initial_step)
            value = {"stage": phase, "current_steps": position["global_step"], "total_steps": total_steps,
                     "epoch": epoch + min(1., (offset + len(selected)) / len(order)),
                     "real_loss": measured["real_loss"], "remix_loss": measured["remix_loss"],
                     "loss": measured["real_loss"] + measured["remix_loss"], "grad_norm": float(norm), "lr": scheduler.get_last_lr(),
                     "module_gradients": dict(module_gradients), "training_observations": dict(observed),
                     "elapsed_seconds": elapsed, "remaining_seconds_estimate": duration * (total_steps - position["global_step"]),
                     "peak_reserved_gib": max(torch.cuda.max_memory_reserved(i) / 1024**3 for i in range(len(devices(args))))}
            with (args.run_dir / "trainer_log.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            emit(args, "training", **value)
            if position["global_step"] % 25 == 0 or position["offset"] == 0:
                save_state(args, model, optimizer, scheduler, torch, position)
    latest = read(args.run_dir / "latest.json")
    if latest["position"]["epoch"] != 3:
        raise ValueError("Training did not complete all fixed epochs")
    marker = {"algorithm": VERSION, "job": job_contract(args), "identity_hash": fingerprint(identity),
              "checkpoint": latest["checkpoint"], "checkpoint_sha256": latest["checkpoint_sha256"],
              "epochs": {"evidence": 1, "joint": 2}, "official_validation_used": False, "test_used": False,
              "train_rows": len(train), "optimizer_steps": total_steps, "completed_unix": time.time()}
    immutable(args.run_dir / "train.complete.json", marker)
    emit(args, "training_complete", checkpoint=latest["checkpoint"])
    return marker


def probe_indices(rows, train, priority=()):
    selected, used = [], set()
    for i in priority:
        if i in train and i not in used:
            selected.append(i)
            used.add(i)
    predicates = [lambda r: r["views"]["M12"]["label"] == "neutral",
                  lambda r: not r["views"]["M1"]["is_clear"],
                  lambda r: not r["views"]["M2"]["is_clear"]]
    for protocol in ("VT", "VA"):
        for kind in ("Aligned", "Conflict"):
            predicates.append(lambda r, p=protocol, k=kind: r["protocol"] == p and r["sample_type"] == k)
        predicates.append(lambda r, p=protocol: r["protocol"] == p and r["sample_type"] == "Conflict" and r.get("source_dataset") in data.NATURAL)
    for predicate in predicates:
        ix = next((i for i in train if i not in used and predicate(rows[i])), None)
        if ix is not None:
            selected.append(ix)
            used.add(ix)
    for i in train:
        if i not in used:
            selected.append(i)
            used.add(i)
        if len(selected) == 32:
            break
    if len(selected) < 32:
        raise ValueError("Real preflight needs at least 32 training rows")
    return selected


def media_extremes(rows, train, dataset_root, max_frames):
    """Probe metadata only; all priorities come from the current train fold."""
    import av
    probes = {}
    for i in train:
        video = rows[i]["video"]
        if video in probes:
            continue
        with av.open(str(Path(dataset_root) / video)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 0.
            duration = float(stream.duration * stream.time_base) if stream.duration and stream.time_base else float(stream.frames) / fps if fps else 0.
            if not fps or not math.isfinite(duration) or duration <= 0:
                raise ValueError("Cannot audit video duration: " + video)
            sampled = min(max_frames, max(4, 2 * round(duration)))
            probes[video] = {"row": i, "duration": duration, "sampled": sampled,
                             "area": stream.width * stream.height,
                             "aspect": max(stream.width / stream.height, stream.height / stream.width)}
    return [max(probes.values(), key=lambda x: x[key])["row"] for key in ("duration", "sampled", "area", "aspect")]


def preflight(args, manifest, rows, train, identity):
    torch = setup_torch(args)
    model = make_model(args, manifest, torch)
    counts = data.class_counts(rows, train)
    optimizer, _ = optimizer_for(model, torch, 8)
    priorities = media_extremes(rows, train, manifest["data_root"], args.max_frames)
    selected = probe_indices(rows, train, priorities)
    initial = {name: p.detach().float().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
    audit = model.trainable_audit()
    immutable(args.run_dir / "trainable-parameters.json", audit)
    gradients = Counter()
    started = time.monotonic()
    losses = []
    def fixed_loss(label):
        model.eval()
        result = 0.
        with torch.no_grad():
            for completed, ix in enumerate(selected, start=1):
                objective, _ = sample_loss(model, rows[ix], rows, counts, "main")
                result += float(objective["main_loss"].detach().cpu()) / len(selected)
                emit(args, "preflight_diagnostic_" + label, completed=completed, total=len(selected))
        if not math.isfinite(result):
            raise FloatingPointError("Nonfinite fixed preflight diagnostic")
        return result
    loss_before = fixed_loss("before")
    for phase in ("evidence_warmup", "main"):
        model.train()
        for start in range(0, len(selected), EFFECTIVE_BATCH):
            optimizer.zero_grad(set_to_none=True)
            batch = selected[start:start + EFFECTIVE_BATCH]
            for ix in batch:
                if stopped(args):
                    raise InterruptedError("Deadline during real preflight")
                outputs, _ = sample_loss(model, rows[ix], rows, counts, phase)
                loss = outputs["warmup_loss" if phase == "evidence_warmup" else "main_loss"]
                if not torch.isfinite(loss).all():
                    raise FloatingPointError("Nonfinite preflight objective")
                (loss / len(batch)).backward()
                losses.append(float(loss.detach().cpu()))
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    value = float(p.grad.detach().float().norm().cpu())
                    if not math.isfinite(value):
                        raise FloatingPointError("Nonfinite module gradient")
                    gradients[name] = max(gradients[name], value)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            optimizer.step()
            emit(args, "real_preflight", stage=phase, completed=start + len(batch), total=len(selected))
    loss_after = fixed_loss("after")
    if loss_after >= loss_before:
        raise ValueError("Real training preflight did not lower the fixed 32-row main objective")
    changed = {name: float((p.detach().float().cpu() - initial[name]).abs().max())
               for name, p in model.named_parameters() if name in initial}
    if not any(gradients.values()) or not any(changed.values()):
        raise ValueError("Preflight has no actual gradient/parameter update")
    # Model supplies explicit adapter roles: coverage is checked by role, not
    # by assuming every LoRA-A tensor has a gradient on its first update.
    roles = audit.get("parameter_roles", {})
    if not roles:
        raise ValueError("Model audit must supply explicit named parameter roles")
    grouped = {}
    for name, role in roles.items():
        row = grouped.setdefault(role, {"max_gradient_norm": 0., "max_parameter_delta": 0.})
        row["max_gradient_norm"] = max(row["max_gradient_norm"], gradients.get(name, 0.))
        row["max_parameter_delta"] = max(row["max_parameter_delta"], changed.get(name, 0.))
    for role, value in grouped.items():
        if value["max_gradient_norm"] <= 0 or value["max_parameter_delta"] <= 0:
            raise ValueError("Requested module role did not really update: " + role)
    peak = max(torch.cuda.max_memory_reserved(i) / 1024**3 for i in range(len(devices(args))))
    if peak > 40:
        raise RuntimeError("CUDA out of memory budget: preflight reserved >40 GiB")
    elapsed = time.monotonic() - started
    marker = {"algorithm": VERSION, "job": job_contract(args), "samples": len(selected),
              "samples_sha256": fingerprint([rows[i]["sample_id"] for i in selected]),
              "gradient_and_update_audit": grouped, "peak_reserved_gib": peak,
              "maximum_frame_cap": args.max_frames, "preflight_seconds_excluding_load": elapsed,
              "estimated_epochs_seconds": elapsed / (2 * len(selected)) * len(train) * 4,
              "estimate_note": "Diagnostic extrapolation, not an ETA. Only training-fold rows; parameters discarded.",
              "actual_media_audit": getattr(model.backbone, "media_audit", []),
              "loss_before": loss_before, "loss_after": loss_after,
              "losses_finite": True, "official_validation_used": False, "test_used": False}
    immutable(args.run_dir / "preflight.complete.json", marker)
    emit(args, "preflight_complete", peak_reserved_gib=peak)
    return marker


def reference_configuration(args, manifest, rows, train):
    import yaml
    from data.prepare_training_data import build_omni_records
    from scripts import prepare_sft_data as sft
    from scripts.protocol_sft_ablation import source_groups
    selected = [rows[i] for i in train]
    cleaned = build_omni_records(selected, source_groups(selected), omni_input_policy="protocol_consistent")[0]
    output = args.run_dir / "reference-data"
    questions, samples = sft.load_questions(SOURCE / "data/cleaned")
    cache = args.project_root / "releases/evidence-calibrator-v1-20261025/data/llamafactory/generated/protocol_consistent"
    audio = {}
    for row in cleaned:
        if row["source_protocol"] == "VA" and row["video"] not in audio:
            path = sft.extracted_audio_path(cache, row["video"])
            if not path.is_file():
                path = sft.extracted_audio_path(args.run_dir / "media-cache", row["video"])
                sft.extract_one_audio(Path(manifest["data_root"]) / row["video"], path, False)
            audio[row["video"]] = str(path)
    records = sft.make_sft_records(cleaned, Path(manifest["data_root"]), SOURCE, "omni", "train",
                                   questions, samples, "single", sft.DATA_ORDER_SEED, audio, "full")
    sft.validate_sft_records(records, "omni", SOURCE, "full")
    aligned, conflict = [], []
    for row in records:
        (conflict if json.loads(row["messages"][-1]["content"])["is_uncertain"] == "yes" else aligned).append(row)
    entry = sft.dataset_info("")["affect_sft_omni_train"]
    for filename, value in (("aligned.jsonl", aligned), ("conflict.jsonl", conflict)):
        path = output / filename
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in value)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError("Reference records changed")
        path.write_text(content, encoding="utf-8")
    immutable(output / "dataset_info.json", {
        "joint_aligned": {**deepcopy(entry), "file_name": "aligned.jsonl"},
        "joint_conflict": {**deepcopy(entry), "file_name": "conflict.jsonl", "num_samples": 2 * len(conflict)}})
    config = yaml.safe_load((SOURCE / "configs/sft/qwen25omni/qlora_sft_v4_resample2x.yaml").read_text(encoding="utf-8"))
    config.update(model_name_or_path=str(args.project_root / "models/Qwen2.5-Omni-3B"),
                  dataset_dir=str(output), media_dir=str(SOURCE), dataset="joint_aligned,joint_conflict",
                  output_dir=str(args.run_dir / "reference-checkpoints"), seed=args.seed, data_seed=args.seed,
                  do_eval=False, eval_strategy="no", num_train_epochs=3., flash_attn="sdpa",
                  save_strategy="steps", save_steps=25, save_total_limit=3, load_best_model_at_end=False,
                  video_maxlen=args.max_frames, overwrite_output_dir=False)
    config.pop("eval_dataset", None)
    config_path = args.run_dir / "reference-config.yaml"
    content = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    if config_path.exists() and config_path.read_text(encoding="utf-8") != content:
        raise ValueError("Reference config changed")
    config_path.write_text(content, encoding="utf-8")
    return config_path


def child_environment(args):
    return {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(devices(args)), "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "FORCE_TORCHRUN": "0", "NPROC_PER_NODE": "1",
            "PYTHONPATH": str(SOURCE / "third_party/LlamaFactory/src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "PATH": str(args.project_root / "media-tools/bin") + os.pathsep + os.environ.get("PATH", "")}


def run_child(args, command, logname):
    log = args.run_dir / "logs" / logname
    log.parent.mkdir(exist_ok=True)
    with log.open("ab") as stream:
        process = subprocess.Popen(command, cwd=SOURCE, env=child_environment(args), stdout=stream, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if stopped(args):
                    process.terminate()
                    try:
                        process.wait(timeout=55)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise InterruptedError("Child stopped at the global deadline; preserved latest checkpoint")
                time.sleep(1)
            if process.returncode:
                raise RuntimeError("Reference child failed; inspect " + str(log))
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=55)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


def reference_final_checkpoint(root):
    root = Path(root)
    state = read(root / "trainer_state.json")
    if state.get("epoch", 0) < 3 - 1e-6:
        raise InterruptedError("Reference has not reached its fixed third epoch")
    # HF may save the final update only in the output root when save_steps does
    # not divide total updates. Never substitute an earlier periodic adapter.
    final = root / "fixed-final-adapter"
    final.mkdir(exist_ok=True)
    for name in ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json"):
        source, target = root / name, final / name
        if not source.is_file():
            raise ValueError("Missing fixed final reference artifact: " + name)
        if target.exists() and digest(source) != digest(target):
            raise ValueError("Fixed final reference adapter changed")
        if not target.exists():
            shutil.copyfile(source, target)
    return final


def train_reference(args, manifest, rows, train, identity):
    config = reference_configuration(args, manifest, rows, train)
    emit(args, "training_sft_reference")
    command = [sys.executable, "-m", "scripts.train_joint_affect", "reference-worker", "--config", str(config)]
    root = args.run_dir / "reference-checkpoints"
    checkpoints = sorted(root.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    state = read(root / "trainer_state.json") if (root / "trainer_state.json").exists() else {}
    if state.get("epoch", 0) < 3 - 1e-6:
        if checkpoints:
            command.extend(["--resume", str(checkpoints[-1])])
        run_child(args, command, "reference-train.log")
    checkpoints = sorted(root.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if not checkpoints or read(root / "trainer_state.json").get("epoch", 0) < 3 - 1e-6:
        raise InterruptedError("Reference has not reached its fixed third epoch")
    checkpoint = reference_final_checkpoint(root)
    marker = {"algorithm": VERSION, "job": job_contract(args), "identity_hash": fingerprint(identity),
              "checkpoint": str(checkpoint), "checkpoint_sha256": directory_hash(checkpoint), "epochs": 3,
              "official_validation_used": False, "test_used": False, "completed_unix": time.time()}
    immutable(args.run_dir / "train.complete.json", marker)
    emit(args, "training_complete", checkpoint=str(checkpoint))
    return marker


def reference_worker(args):
    import torch
    from transformers import TrainerCallback
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(min(1., 40 * 1024**3 / torch.cuda.get_device_properties(0).total_memory), 0)
    class SaveAtStop(TrainerCallback):
        def on_step_end(self, arguments, state, control, **kwargs):
            if STOP_REQUESTED:
                control.should_save = True
                control.should_training_stop = True
            return control
    original = CustomSeq2SeqTrainer.__init__
    def initialize(self, *a, **kw):
        original(self, *a, **kw)
        self.add_callback(SaveAtStop())
    CustomSeq2SeqTrainer.__init__ = initialize
    import runpy
    sys.path.insert(0, str(SOURCE / "scripts"))
    sys.argv = [str(SOURCE / "scripts/train_sft.py"), str(args.config)]
    if args.resume:
        sys.argv.append("resume_from_checkpoint=" + str(args.resume))
    runpy.run_path(str(SOURCE / "scripts/train_sft.py"), run_name="__main__")


def inference_rows(args, manifest, rows, held):
    if args.split == "held":
        if args.fold == "full":
            raise ValueError("Full models have no held-out training rows")
        return [rows[i] for i in held]
    if args.fold != "full" or not (args.run_dir / "final_model_lock.json").is_file():
        raise ValueError("Official inference requires a locked full-data model")
    lock = read(args.run_dir / "final_model_lock.json")
    # Pipeline publishes hashes of both complete training markers before this
    # first access to official files; require this job to be explicitly bound.
    marker_sha = digest(args.run_dir / "train.complete.json")
    if not any(Path(item.get("directory", "")).resolve() == args.run_dir.resolve() and
               item.get("complete_sha256") == marker_sha for item in lock.get("models", [])):
        raise ValueError("Final model lock does not bind this completed training marker")
    path = args.project_root / "Dataset" / ("val.jsonl" if args.split == "val" else "test_release.jsonl")
    return data.load_test(path, args.project_root / "Dataset", expected_rows=666 if args.split == "val" else 734)


def predict(args, manifest, rows, held, identity):
    marker = read(args.run_dir / "train.complete.json")
    if marker["job"] != job_contract(args) or marker["checkpoint_sha256"] != directory_hash(marker["checkpoint"]):
        raise ValueError("Prediction checkpoint or job identity changed")
    selected = inference_rows(args, manifest, rows, held)
    output = args.output or args.run_dir / ("predictions.jsonl" if args.split == "held" else "predictions-" + args.split + ".jsonl")
    output = output.resolve()
    if not output.is_relative_to(args.run_dir):
        raise ValueError("Predictions must remain in this immutable job directory")
    if args.variant == "SFT-ref":
        input_dir = args.run_dir / "reference-inference" / args.split
        evaluation_rows = [{**row, "video": str((Path(manifest["data_root"]) / row["video"]).resolve())} for row in selected]
        path = input_dir / ("val.jsonl" if args.split != "test" else "test_release.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in evaluation_rows)
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError("Reference inference source changed")
        path.write_text(content, encoding="utf-8")
        from scripts.protocol_sft_ablation import limited_command
        command = limited_command(SOURCE / "scripts/evaluate_sft.py", ["--config", str(args.run_dir / "reference-config.yaml"),
            "--adapter-path", marker["checkpoint"], "--data-root", str(input_dir), "--output-dir", str(args.run_dir / "reference-evaluation"),
            "--run-name", args.split, "--split", "test" if args.split == "test" else "val", "--load-in-4bit", "--fps", "2",
            "--min-frames", "4", "--max-frames", str(args.max_frames), "--min-visual-tokens", "64", "--max-visual-tokens", "64"])
        run_child(args, command, "reference-predict-" + args.split + ".log")
        evaluated = read(args.run_dir / "reference-evaluation" / (args.split + "_metrics.json"))
        if evaluated.get("inference_errors") != 0 or evaluated.get("samples") != len(selected):
            raise ValueError("Reference inference contains errors or incomplete coverage")
        import csv
        with (args.run_dir / "reference-evaluation" / (args.split + "_predictions.csv")).open(encoding="utf-8-sig", newline="") as stream:
            converted = [{"sample_id": r["sample_id"], "emotion_probs": [float(r["pred_emotion"] == e) for e in data.EMOTIONS],
                          "uncertain_prob": float(r["is_uncertain"] == "yes"), "reference_onehot": True} for r in csv.DictReader(stream)]
        if [r["sample_id"] for r in converted] != [r["sample_id"] for r in selected]:
            raise ValueError("Reference generated IDs/order changed")
        with output.open("w", encoding="utf-8") as stream:
            for row in converted:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        torch = setup_torch(args)
        model = make_model(args, manifest, torch)
        model.load_checkpoint(marker["checkpoint"])
        model.eval()
        existing = []
        if output.exists():
            for line in output.read_text(encoding="utf-8").splitlines():
                existing.append(json.loads(line))
        if [r.get("sample_id") for r in existing] != [r["sample_id"] for r in selected[:len(existing)]]:
            raise ValueError("Resumed prediction IDs/order changed")
        with output.open("a", encoding="utf-8") as stream, torch.inference_mode():
            for index, row in enumerate(selected[len(existing):], start=len(existing)):
                if stopped(args):
                    raise InterruptedError("Prediction interrupted; persisted prefix retained")
                value = model.forward_row(row)
                record = {"sample_id": row["sample_id"],
                          "emotion_probs": value["emotion_probs"].detach().float().reshape(-1).cpu().tolist(),
                          "uncertain_prob": float(value["uncertain_prob"].detach().float().cpu())}
                for source, target in (("vision_probs", "vision_probs"), ("second_probs", "second_probs")):
                    if source in value:
                        record[target] = value[source].detach().float().reshape(-1).cpu().tolist()
                record["clarifying_question"] = data.question_for_protocol(row["protocol"])
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                emit(args, "predicting_" + args.split, completed=index + 1, total=len(selected))
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    if [r["sample_id"] for r in records] != [r["sample_id"] for r in selected]:
        raise ValueError("Incomplete predictions")
    immutable(Path(str(output) + ".complete.json"), {"job": job_contract(args), "split": args.split,
              "rows": len(records), "sha256": digest(output), "checkpoint_sha256": marker["checkpoint_sha256"]})
    emit(args, "predictions_complete", split=args.split, rows=len(records))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("preflight", "train", "predict", "reference-worker"))
    p.add_argument("--project-root", type=Path)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--variant", choices=("C0", "C1", "SFT-ref"), default="C0")
    p.add_argument("--fold", default="0")
    p.add_argument("--grouping", choices=("content", "parent"), default="content")
    p.add_argument("--seed", type=int, choices=(42, 43), default=42)
    p.add_argument("--vision-model", choices=("omni", "qwen3vl8b"), default="omni")
    p.add_argument("--gpu", default="1")
    p.add_argument("--gpus")
    p.add_argument("--max-frames", type=int, choices=(32, 64), default=64)
    p.add_argument("--deadline-unix", type=float, default=float("inf"))
    p.add_argument("--init-second-from", type=Path)
    p.add_argument("--split", choices=("held", "val", "test"), default="held")
    p.add_argument("--output", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--resume", type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if args.command == "reference-worker":
        return reference_worker(args)
    if any(x is None for x in (args.project_root, args.run_dir, args.manifest)):
        raise ValueError("project-root, run-dir and manifest are required")
    if args.fold != "full":
        args.fold = int(args.fold)
        if args.fold not in (0, 1, 2):
            raise ValueError("Fold must be 0,1,2 or full")
    devices(args)
    args.manifest = args.manifest.resolve()
    manifest, rows, train, held, identity = load_context(args)
    marker = args.run_dir / ("preflight.complete.json" if args.command == "preflight" else "train.complete.json")
    if args.command in ("train", "preflight") and marker.exists():
        value = read(marker)
        if value.get("job") != job_contract(args):
            raise ValueError("Completed job has a different identity")
        if args.command == "train" and directory_hash(value["checkpoint"]) != value["checkpoint_sha256"]:
            raise ValueError("Completed training checkpoint changed")
        return value
    try:
        if args.command == "predict":
            return predict(args, manifest, rows, held, identity)
        if args.command == "preflight":
            if args.variant == "SFT-ref":
                raise ValueError("Preflight applies to the two new trainable systems")
            return preflight(args, manifest, rows, train, identity)
        if args.variant == "SFT-ref":
            return train_reference(args, manifest, rows, train, identity)
        return train_new(args, manifest, rows, train, identity)
    except Exception as exc:
        emit(args, "stopped" if isinstance(exc, InterruptedError) else "failed", error=str(exc))
        atomic(args.run_dir / "error.json", {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
