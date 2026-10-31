"""Train one full-data protocol-consistent SFT model after two seed gates pass.

Epochs are fixed before final training as ceil(mean(two development-selected
epochs)). Official validation is a single locked-model diagnostic; test output
is a review candidate and is never submitted to Kaggle by this module.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import traceback

import yaml

from data.prepare_training_data import build_omni_records
from rl.experiment import ROOT, artifact_hash, file_hash, fingerprint, write_json
from rl.reward import parse_completion, question_is_consistent
from scripts import protocol_sft_ablation as ablation
from scripts import protocol_sft_summary as paired
from scripts.evidence_supplement import FIELDS, write_immutable
from scripts.run_evidence_supplement import run_lock

VERSION = "omni_protocol_sft_full_v1"
TRAIN_ROWS, VAL_ROWS, TEST_ROWS = 3354, 666, 734
FINAL_SEED = 42
EPOCH_RULE = "ceil(mean(seed42_and_seed43_experiment_development_selected_epochs))"
SOURCE_FILES = tuple(dict.fromkeys((*ablation.SOURCE_FILES,
    "scripts/protocol_sft_final.py", "scripts/protocol_sft_summary.py",
    "scripts/sft_config.py", "qwen25vl_baseline.py", "third_party/llamafactory.lock.yaml")))
PREPARE_CONFIG_FIELDS = {"model_name_or_path", "dataset_dir", "media_dir", "dataset",
    "eval_dataset", "output_dir", "seed", "data_seed", "num_train_epochs",
    "resume_from_checkpoint", "flash_attn"}


def read(path):
    return ablation.read(path)


def epoch_at(checkpoint):
    value = read(Path(checkpoint) / "trainer_state.json").get("epoch")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError("Selected checkpoint needs a finite trainer_state epoch")
    epoch = round(value)
    if not 1 <= epoch <= 3 or abs(value - epoch) > .001:
        raise ValueError("Checkpoint epoch must be approximately an integer from 1 through 3")
    return int(epoch)


def final_epochs(first, second):
    if any(type(epoch) is not int or not 1 <= epoch <= 3 for epoch in (first, second)):
        raise ValueError("Both development-selected epochs must be integers in 1..3")
    return math.ceil((first + second) / 2)


def _config_invariants(config):
    return {k: v for k, v in config.items() if k not in PREPARE_CONFIG_FIELDS}


def checked_seed(root, expected_seed):
    root = Path(root).resolve()
    identity, manifest = ablation.checked_run(root)
    raw_identity = identity["identity"]
    if raw_identity["training_seed"] != expected_seed:
        raise ValueError("Seed run was passed in the wrong position")
    summary = paired.summarize(root)
    if (summary.get("training_seed"), summary.get("identity_hash"), summary.get("gate", {}).get("passed")) != (
            expected_seed, identity["identity_hash"], True):
        raise ValueError("Both genuine paired seed promotion gates must pass before full training")
    selected_path = root / "experiment/selected-checkpoint.json"
    selected = read(selected_path)
    checkpoint = Path(selected["checkpoint"]).resolve()
    if not checkpoint.is_relative_to(root / "experiment/checkpoints"):
        raise ValueError("Development-selected experiment adapter escapes its seed run")
    if (selected.get("identity_hash") != identity["identity_hash"] or
            selected.get("selected_by") != "training-only development Score" or
            selected.get("holdout_used_for_selection") is not False or
            selected.get("adapter_sha256") != artifact_hash(checkpoint, adapter=True)):
        raise ValueError("Development selection or selected adapter changed")
    config = yaml.safe_load((root / "experiment/config.yaml").read_text(encoding="utf-8"))
    if (config.get("flash_attn") != "sdpa" or raw_identity.get("attention", {}).get("shared_flash_attn") != "sdpa" or
            config.get("adapter_name_or_path") or config.get("num_train_epochs") != 3. or
            config.get("seed") != expected_seed or config.get("data_seed") != expected_seed or
            Path(config["model_name_or_path"]).resolve() != Path(manifest["base_model"]).resolve()):
        raise ValueError("Both experiments must use common SDPA, the raw base and at most three planned epochs")
    sources = {str(root / name): file_hash(root / name) for name in (
        "identity.json", "prepare-manifest.json", "pair-summary.json", "experiment/selected-checkpoint.json")}
    sources[str(checkpoint / "trainer_state.json")] = file_hash(checkpoint / "trainer_state.json")
    for relative, digest in summary["input_sha256"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or file_hash(path) != digest:
            raise ValueError("Paired seed evidence changed during qualification")
        sources[str(path)] = digest
    return {"root": root, "identity": identity, "manifest": manifest,
            "summary": summary, "config": config, "epoch": epoch_at(checkpoint), "source_hashes": sources}


def prepare_final(seed42_run, seed43_run, output_dir, dataset_root, model_path,
                  audio_cache_dir, questions_dir=None):
    root, dataset, model = (Path(p).resolve() for p in (output_dir, dataset_root, model_path))
    if not root.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("Full-training artifacts must remain under the Git-ignored runs directory")
    a, b = checked_seed(seed42_run, 42), checked_seed(seed43_run, 43)
    if any(root.is_relative_to(value["root"]) or value["root"].is_relative_to(root) for value in (a, b)):
        raise ValueError("Final output must be separate from both immutable seed runs")
    left, right = a["identity"]["identity"], b["identity"]["identity"]
    for key in ("base_sha256", "train_sha256", "media_sha256", "base_config_sha256",
                "source_sha256", "questions_sha256", "split_hash", "attention", "conflict_factor", "max_epochs"):
        if left.get(key) != right.get(key):
            raise ValueError("Seed runs differ in their original model/data/protocol contract: " + key)
    base_config = yaml.safe_load(ablation.BASE_CONFIG.read_text(encoding="utf-8"))
    if (file_hash(ablation.BASE_CONFIG) != left["base_config_sha256"] or
            _config_invariants(a["config"]) != _config_invariants(b["config"]) or
            _config_invariants(a["config"]) != _config_invariants(base_config)):
        raise ValueError("Seed model settings differ from one another or from the original v4 hyperparameters")
    if artifact_hash(model) != left["base_sha256"]:
        raise ValueError("Full training must use the same original base weights")
    raw, media = ablation.verified_training(dataset / "train.jsonl", dataset, TRAIN_ROWS)
    if file_hash(dataset / "train.jsonl") != left["train_sha256"] or media != left["media_sha256"]:
        raise ValueError("Full training data differs from the two verified seed experiments")
    epochs = final_epochs(a["epoch"], b["epoch"])
    questions = Path(questions_dir).resolve() if questions_dir else ROOT / "data/cleaned"
    question_hashes = {name: file_hash(questions / name) for name in left["questions_sha256"]}
    if question_hashes != left["questions_sha256"]:
        raise ValueError("Final clarification supervision differs from the seed experiments")
    question_groups, question_samples = ablation.sft.load_questions(questions)
    cleaned = build_omni_records(raw, ablation.source_groups(raw), omni_input_policy="protocol_consistent")[0]
    if len(cleaned) != TRAIN_ROWS or (TRAIN_ROWS == 3354 and sum(r["sample_type"] == "Conflict" for r in cleaned) != 411):
        raise ValueError("Expected all original protocol-consistent training rows")
    audio_root, audio_map, audio_hashes = Path(audio_cache_dir).resolve(), {}, {}
    for row in cleaned:
        if row["source_protocol"] != "VA":
            continue
        path = ablation.sft.extracted_audio_path(audio_root, row["video"])
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("Missing existing training audio: " + str(path))
        digest = file_hash(path)
        suffix = "/audio/" + Path(row["video"]).with_suffix(".flac").as_posix()
        for seed in (a, b):
            expected = {value for name, value in seed["manifest"]["audio_sha256"].items()
                        if name.replace("\\", "/").endswith(suffix)}
            if expected != {digest}:
                raise ValueError("Final audio differs from the original seed media: " + row["video"])
        audio_map[row["video"]], audio_hashes[str(path)] = str(path), digest
    records = ablation.sft.make_sft_records(cleaned, dataset, ROOT, "omni", "train",
        question_groups, question_samples, "single", ablation.sft.DATA_ORDER_SEED, audio_map, "full")
    ablation.sft.validate_sft_records(records, "omni", ROOT, "full")
    aligned, conflict = [], []
    for record in records:
        target = json.loads(record["messages"][-1]["content"])
        (conflict if target["is_uncertain"] == "yes" else aligned).append(record)
    if len({r["sample_id"] for r in records}) != TRAIN_ROWS or not aligned or not conflict:
        raise ValueError("Final training must cover each original sample exactly once before class resampling")
    entry = ablation.sft.dataset_info("")["affect_sft_omni_train"]
    info = {"final_train_aligned": {**deepcopy(entry), "file_name": "aligned.jsonl"},
            "final_train_conflict": {**deepcopy(entry), "file_name": "conflict.jsonl", "num_samples": 2 * len(conflict)}}
    config = deepcopy(a["config"])
    config.update(model_name_or_path=str(model), dataset_dir=str(root / "dataset"), media_dir=str(ROOT),
        dataset="final_train_aligned,final_train_conflict", output_dir=str(root / "checkpoints"),
        seed=FINAL_SEED, data_seed=FINAL_SEED, num_train_epochs=float(epochs),
        resume_from_checkpoint=None, do_eval=False, eval_strategy="no", load_best_model_at_end=False,
        flash_attn="sdpa")
    config.pop("eval_dataset", None)
    config.pop("adapter_name_or_path", None)
    for key in ("metric_for_best_model", "greater_is_better"):
        config.pop(key, None)
    payloads = {root / "dataset/aligned.jsonl": ablation.jsonl_bytes(aligned),
        root / "dataset/conflict.jsonl": ablation.jsonl_bytes(conflict),
        root / "dataset/dataset_info.json": ablation.json_bytes(info),
        root / "config.yaml": yaml.safe_dump(config, allow_unicode=True, sort_keys=False).encode("utf-8")}
    sources = {**a["source_hashes"], **b["source_hashes"]}
    identity = {"algorithm": VERSION, "seed_identities": {"42": a["identity"]["identity_hash"], "43": b["identity"]["identity_hash"]},
        "seed_source_sha256": sources, "development_selected_epochs": {"42": a["epoch"], "43": b["epoch"]},
        "epoch_rule": EPOCH_RULE, "final_epochs": epochs, "final_seed": FINAL_SEED,
        "base_sha256": left["base_sha256"], "train_sha256": left["train_sha256"], "media_sha256": media,
        "questions_sha256": question_hashes, "audio_sha256": audio_hashes,
        "prepared_artifacts_sha256": {str(path.relative_to(root)): hashlib.sha256(content).hexdigest()
                                       for path, content in payloads.items()},
        "source_sha256": {name: file_hash(ROOT / name) for name in SOURCE_FILES},
        "official_val_read_during_prepare": False, "official_test_read_during_prepare": False}
    frozen = {"identity": identity, "identity_hash": fingerprint(identity)}
    with run_lock(root):
        write_immutable(root / "identity.json", ablation.json_bytes(frozen))
        for path, content in payloads.items():
            if path.exists() and path.read_bytes() != content:
                raise ValueError("Refusing to replace different prepared final-training artifacts")
        if any(file_hash(path) != digest for path, digest in sources.items()):
            raise ValueError("Seed evidence changed during final preparation")
        for path, content in payloads.items():
            write_immutable(path, content)
        manifest = {"algorithm": VERSION, "identity_hash": frozen["identity_hash"], "base_model": str(model),
            "dataset_root": str(dataset), "train_jsonl": str(dataset / "train.jsonl"), "rows": TRAIN_ROWS,
            "aligned": len(aligned), "conflicts": len(conflict), "sampled_training_rows": len(aligned) + 2 * len(conflict),
            "epochs": epochs, "epoch_rule": EPOCH_RULE, "from_original_base": True,
            "has_eval_dataset": False, "official_val_read": False, "official_test_read": False,
            "artifacts": {str(path.relative_to(root)): file_hash(path) for path in payloads}}
        write_immutable(root / "prepare-manifest.json", ablation.json_bytes(manifest))
    return manifest


def checked_final(root):
    root = Path(root).resolve()
    frozen, manifest = read(root / "identity.json"), read(root / "prepare-manifest.json")
    identity = frozen["identity"]
    if frozen["identity_hash"] != fingerprint(identity) or manifest.get("identity_hash") != frozen["identity_hash"]:
        raise ValueError("Final-training identity changed")
    if identity.get("algorithm") != VERSION or manifest.get("algorithm") != VERSION:
        raise ValueError("Wrong final-training algorithm")
    if (manifest.get("epochs") != identity["final_epochs"] or manifest.get("epoch_rule") != EPOCH_RULE or
            manifest.get("artifacts") != identity["prepared_artifacts_sha256"]):
        raise ValueError("Final epochs or prepared artifacts differ from their immutable identity")
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    if (config.get("do_eval") is not False or config.get("eval_strategy") != "no" or "eval_dataset" in config or
            config.get("adapter_name_or_path") or config.get("num_train_epochs") != identity["final_epochs"]):
        raise ValueError("Final fitting must have fixed epochs, raw-base initialization and no validation dataset")
    for name, digest in identity["source_sha256"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError("Final-training source changed: " + name)
    for name, digest in identity["seed_source_sha256"].items():
        if file_hash(name) != digest:
            raise ValueError("Verified seed evidence changed: " + name)
    for name, digest in manifest["artifacts"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or file_hash(path) != digest:
            raise ValueError("Final prepared artifact changed: " + name)
    if artifact_hash(manifest["base_model"]) != identity["base_sha256"] or file_hash(manifest["train_jsonl"]) != identity["train_sha256"]:
        raise ValueError("Original full-training base or source changed")
    for video, digest in identity["media_sha256"].items():
        if file_hash(Path(manifest["dataset_root"]) / video) != digest:
            raise ValueError("Original final-training video changed: " + video)
    for path, digest in identity["audio_sha256"].items():
        if file_hash(path) != digest:
            raise ValueError("Prepared full-training audio changed: " + path)
    return frozen, manifest


def checked_csv(path, expected_ids):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Final candidate must have precisely the four official columns")
        rows = list(reader)
    if (any(not isinstance(sid, str) or not sid for sid in expected_ids) or
            len(expected_ids) != len(set(expected_ids)) or [r.get("sample_id") for r in rows] != expected_ids):
        raise ValueError("Final prediction IDs/coverage/order differ from the official split")
    for row in rows:
        value = {k: row[k] for k in FIELDS[1:]}
        if set(row) != set(FIELDS) or parse_completion(json.dumps(value)) is None or not question_is_consistent(value):
            raise ValueError("Final CSV has an invalid emotion, uncertainty, or question")
    return rows


def evaluation_command(root, adapter, dataset_root, split):
    return ablation.limited_command(ROOT / "scripts/evaluate_sft.py", ["--config", str(root / "config.yaml"),
        "--adapter-path", str(adapter), "--data-root", str(dataset_root), "--output-dir", str(root / "evaluation"),
        "--run-name", "locked_final_" + split, "--split", split, "--load-in-4bit", "--fps", "2",
        "--min-frames", "4", "--max-frames", "64", "--min-visual-tokens", "64", "--max-visual-tokens", "64"])


def train_final(output_dir, gpu="1"):
    root = Path(output_dir).resolve()
    frozen, manifest = checked_final(root)
    identity = frozen["identity"]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false", "FORCE_TORCHRUN": "0", "NPROC_PER_NODE": "1",
        "PYTHONPATH": str(ROOT / "third_party/LlamaFactory/src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "HF_DATASETS_CACHE": str(root / "cache"), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    with run_lock(root / "execution"):
        try:
            checkpoints = root / "checkpoints"
            existing = sorted([p for p in checkpoints.glob("checkpoint-*") if re.fullmatch(r"checkpoint-\d+", p.name)],
                              key=lambda p: int(p.name.split("-")[-1]))
            if not (checkpoints / "train_results.json").exists():
                command = ablation.limited_command(ROOT / "scripts/train_sft.py", [str(root / "config.yaml")])
                if existing:
                    command.append("resume_from_checkpoint=" + str(existing[-1]))
                write_json(root / "status.json", {"phase": "full_training", "epochs": manifest["epochs"], "gpu": str(gpu)})
                subprocess.run(command, cwd=ROOT, env=environment, check=True)
            selected_path = root / "selected-checkpoint.json"
            if not selected_path.exists():
                adapters = [p for p in checkpoints.glob("checkpoint-*") if re.fullmatch(r"checkpoint-\d+", p.name)
                            and (p / "adapter_model.safetensors").is_file() and epoch_at(p) == manifest["epochs"]]
                if len(adapters) != 1:
                    raise ValueError("Expected exactly one adapter at the predetermined final epoch")
                adapter = adapters[0].resolve()
                write_immutable(selected_path, ablation.json_bytes({"algorithm": VERSION,
                    "identity_hash": frozen["identity_hash"], "checkpoint": str(adapter),
                    "adapter_sha256": artifact_hash(adapter, adapter=True), "epoch": manifest["epochs"],
                    "epoch_rule": EPOCH_RULE, "selected_by": "predetermined_final_epoch",
                    "official_val_used_for_selection": False, "official_test_used_for_selection": False}))
            selected = read(selected_path)
            adapter = Path(selected["checkpoint"]).resolve()
            if (selected.get("identity_hash") != frozen["identity_hash"] or selected.get("epoch") != manifest["epochs"] or
                    selected.get("selected_by") != "predetermined_final_epoch" or not adapter.is_relative_to(checkpoints) or
                    selected.get("official_val_used_for_selection") is not False or
                    selected.get("official_test_used_for_selection") is not False or
                    selected.get("adapter_sha256") != artifact_hash(adapter, adapter=True) or epoch_at(adapter) != manifest["epochs"]):
                raise ValueError("Locked final checkpoint changed")
            reports = {}
            dataset = Path(manifest["dataset_root"])
            for split, filename, expected_count in (("val", "val.jsonl", VAL_ROWS), ("test", "test_release.jsonl", TEST_ROWS)):
                # The first official split access happens only after checkpoint locking.
                raw = ablation.sft.read_jsonl(dataset / filename)
                ids = [row["sample_id"] for row in raw]  # Never access test labels.
                if len(ids) != expected_count or len(set(ids)) != expected_count:
                    raise ValueError("Official split count or unique IDs changed: " + split)
                source_digest = file_hash(dataset / filename)
                report_path = root / ("official-" + split + "-report.json")
                csv_path = root / "evaluation" / ("locked_final_" + split + "_predictions.csv")
                metric_path = root / "evaluation" / ("locked_final_" + split + "_metrics.json")
                if not report_path.exists():
                    write_json(root / "status.json", {"phase": "locked_official_" + split, "checkpoint": str(adapter)})
                    subprocess.run(evaluation_command(root, adapter, dataset, split), cwd=ROOT, env=environment, check=True)
                    metrics = read(metric_path)
                    if metrics.get("samples") != expected_count or metrics.get("inference_errors") != 0:
                        raise ValueError("Incomplete locked-model official generation: " + split)
                    checked_csv(csv_path, ids)
                    if source_digest != file_hash(dataset / filename):
                        raise ValueError("Official input changed during locked inference")
                    report = {"algorithm": VERSION, "identity_hash": frozen["identity_hash"],
                        "adapter_sha256": selected["adapter_sha256"], "split": split,
                        "source_sha256": source_digest, "csv_sha256": file_hash(csv_path),
                        "metrics_sha256": file_hash(metric_path), "metrics": metrics,
                        "used_for_selection": False, "submitted_to_kaggle": False}
                    write_immutable(report_path, ablation.json_bytes(report))
                report = read(report_path)
                if (report.get("identity_hash") != frozen["identity_hash"] or report.get("source_sha256") != source_digest or
                        report.get("adapter_sha256") != selected["adapter_sha256"] or report.get("used_for_selection") is not False or
                        report.get("csv_sha256") != file_hash(csv_path) or report.get("metrics_sha256") != file_hash(metric_path)):
                    raise ValueError("Completed official inference report changed: " + split)
                checked_csv(csv_path, ids)
                reports[split] = report
            candidate = root / "candidate.csv"
            write_immutable(candidate, (root / "evaluation/locked_final_test_predictions.csv").read_bytes())
            result = {"algorithm": VERSION, "identity_hash": frozen["identity_hash"], "status": "complete",
                "epochs": manifest["epochs"], "epoch_rule": EPOCH_RULE, "checkpoint": str(adapter),
                "adapter_sha256": selected["adapter_sha256"], "trained_from_original_base": True,
                "official_val_used_for_selection": False, "official_val": reports["val"]["metrics"],
                "official_test_rows": TEST_ROWS, "candidate": str(candidate), "candidate_sha256": file_hash(candidate),
                "submitted_to_kaggle": False}
            write_immutable(root / "final-report.json", ablation.json_bytes(result))
            write_json(root / "status.json", {"phase": "complete", "candidate": str(candidate), "submitted_to_kaggle": False})
            return result
        except Exception as exc:
            write_json(root / "status.json", {"phase": "failed", "error": str(exc)})
            write_json(root / "error.json", {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
            raise


def run_final(args):
    prepare_final(args.seed42_run, args.seed43_run, args.output_dir, args.dataset_root,
                  args.model_path, args.audio_cache_dir, getattr(args, "questions_dir", None))
    return train_final(args.output_dir, args.gpu)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    for name in ("seed42-run", "seed43-run", "output-dir", "dataset-root", "model-path", "audio-cache-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--questions-dir", type=Path)
    parser.add_argument("--gpu", default="1")
    result = run_final(parser.parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
