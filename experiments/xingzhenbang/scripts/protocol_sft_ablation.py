"""Prepare and launch a TRAIN-only, grouped legacy/protocol SFT comparison.

Preparation never opens official val/test files. GPU work is performed only by
the explicit ``train`` command; ``train --dry-run`` prints its launch contract.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from data.prepare_training_data import build_omni_records, validate_record
from rl.experiment import ROOT, artifact_hash, file_hash, fingerprint, load_records
from scripts import prepare_sft_data as sft
from scripts.evidence_supplement import write_immutable
from scripts.run_evidence_supplement import run_lock

VERSION = "omni_protocol_sft_ablation_v1"
ARMS = {"control": "legacy", "experiment": "protocol_consistent"}
BASE_CONFIG = ROOT / "configs/sft/qwen25omni/qlora_sft_v4_resample2x.yaml"
SOURCE_FILES = ("scripts/protocol_sft_ablation.py", "data/prepare_training_data.py",
                "scripts/prepare_sft_data.py", "scripts/train_evidence_calibrator.py",
                "scripts/train_sft.py", "scripts/task_balanced_loss.py", "scripts/evaluate_sft.py")


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def jsonl_bytes(rows):
    return ("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
                    for r in rows)).encode("utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def grouped_split(rows, media):
    # Reuse the exact content grouping and balancing implementation from C2;
    # only its five equal folds are mapped to train/train/train/dev/holdout.
    from scripts.train_evidence_calibrator import content_groups, _assign_folds
    groups = content_groups(rows, media)
    folds = _assign_folds(rows, groups, 42, 5)
    names = ["train" if fold < 3 else "dev" if fold == 3 else "holdout" for fold in folds]
    parts = {name: [i for i, part in enumerate(names) if part == name] for name in ("train", "dev", "holdout")}
    group_sets = [{groups[i] for i in ids} for ids in parts.values()]
    if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i)):
        raise RuntimeError("Content groups leaked between SFT partitions")
    return {"split_seed": 42, "requested_ratios": [.6, .2, .2], "groups": groups,
            "five_folds": folds, "partitions": parts, "assignment": names,
            "note": "Ratios are approximate because a content group is indivisible."}


def source_groups(rows):
    result = defaultdict(dict)
    for row in rows:
        group = result[row["split_group_id"]]
        if row["protocol"] in group:
            raise ValueError("Multiple original rows for the same video/protocol group")
        group[row["protocol"]] = row
    return result


def verified_training(train_jsonl, dataset_root, expected_rows):
    train_jsonl, dataset_root = Path(train_jsonl).resolve(), Path(dataset_root).resolve()
    if train_jsonl.name != "train.jsonl":
        raise ValueError("This comparison accepts only the original train.jsonl, never official val/test")
    rows = sft.read_jsonl(train_jsonl)
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError("Original training row count changed")
    if len({r.get("sample_id") for r in rows}) != len(rows):
        raise ValueError("Duplicate original training IDs")
    media = {}
    for row in rows:
        errors = validate_record(row, dataset_root, expected_split="train")
        if errors:
            raise ValueError("Invalid original training record: " + row.get("sample_id", "?") + " " + str(errors))
        path = (dataset_root / row["video"]).resolve()
        if not path.is_relative_to(dataset_root):
            raise ValueError("Training video escapes the data root")
        if row["video"] not in media:
            media[row["video"]] = file_hash(path)
    return rows, media


def summary(rows):
    return {"rows": len(rows), "conflicts": sum(r["sample_type"] == "Conflict" for r in rows),
            "use_in_main_false": sum(r.get("use_in_main") is not True for r in rows),
            "protocols": dict(Counter(r["protocol"] for r in rows))}


def prepare(output_dir, train_jsonl, dataset_root, model_path, questions_dir,
            audio_cache_dir=None, training_seed=42, audio_workers=4, flash_attn="sdpa", expected_rows=3354,
            split_fn=grouped_split):
    output, dataset_root, model_path = Path(output_dir).resolve(), Path(dataset_root).resolve(), Path(model_path).resolve()
    if not output.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("Ablation artifacts must stay under the Git-ignored repository runs directory")
    if training_seed not in (42, 43):
        raise ValueError("Only the planned primary/replication training seeds 42 and 43 are allowed")
    if flash_attn not in ("sdpa", "fa2"):
        raise ValueError("Choose one shared attention implementation: sdpa or fa2")
    base_config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    raw, media = verified_training(train_jsonl, dataset_root, expected_rows)
    canonical, duplicates = load_records(train_jsonl)
    if duplicates:
        raise ValueError("Original training IDs must be unique")
    design = split_fn(canonical, media)
    groups = source_groups(raw)
    full_clean = {arm: build_omni_records(raw, groups, omni_input_policy=policy)[0]
                  for arm, policy in ARMS.items()}
    if expected_rows == 3354 and [sum(r["sample_type"] == "Conflict" for r in full_clean[a]) for a in ARMS] != [441, 411]:
        raise ValueError("Expected full-source legacy 441 / original-protocol 411 conflicts changed")
    source_hashes = {name: file_hash(ROOT / name) for name in SOURCE_FILES}
    question_files = [Path(questions_dir) / name for name in
        ("clarifying_questions_by_group.jsonl", "clarifying_questions_by_sample_id.jsonl")]
    identity = {"algorithm": VERSION, "train_jsonl": str(Path(train_jsonl).resolve()),
                "train_sha256": file_hash(train_jsonl), "media_sha256": media,
                "base_sha256": artifact_hash(model_path), "base_config_sha256": file_hash(BASE_CONFIG),
                "source_sha256": source_hashes, "questions_sha256": {p.name: file_hash(p) for p in question_files},
                "split_hash": fingerprint(design), "training_seed": training_seed,
                "attention": {"shared_flash_attn": flash_attn,
                    "historical_v4_flash_attn": base_config["flash_attn"],
                    "differs_from_historical_v4": flash_attn != base_config["flash_attn"],
                    "note": "Both arms use the same explicit attention setting; this comparison isolates protocol within that setting, not an exact historical v4 runtime reproduction."},
                "max_epochs": 3, "conflict_factor": 2, "official_val_read": False, "official_test_read": False}
    frozen = {"identity": identity, "identity_hash": fingerprint(identity)}
    with run_lock(output):
        write_immutable(output / "identity.json", json_bytes(frozen))
        question_groups, question_samples = sft.load_questions(Path(questions_dir))
        if audio_cache_dir:
            audio_map = {}
            for row in full_clean["control"]:
                if row["source_protocol"] == "VA":
                    path = sft.extracted_audio_path(Path(audio_cache_dir).resolve(), row["video"])
                    if not path.is_file() or path.stat().st_size == 0:
                        raise FileNotFoundError("Missing cached training audio: " + str(path))
                    audio_map[row["video"]] = str(path)
        else:
            audio_map, _ = sft.prepare_audio(ROOT, dataset_root, output / "shared-media",
                full_clean["control"], audio_workers, False, False)
        partition_ids = {name: {raw[i]["sample_id"] for i in ix}
                         for name, ix in design["partitions"].items()}
        partition_raw = {name: [r for r in raw if r["sample_id"] in ids] for name, ids in partition_ids.items()}
        split_manifest = {"algorithm": VERSION, "identity_hash": frozen["identity_hash"], "design": design,
                          "full_source": summary(raw), "partitions": {name: summary(rr) for name, rr in partition_raw.items()},
                          "full_source_conflicts": {arm: sum(r["sample_type"] == "Conflict" for r in rr)
                                                    for arm, rr in full_clean.items()},
                          "retains_all_use_in_main_false": True, "official_val_read": False, "official_test_read": False}
        payloads = {output / "split-manifest.json": json_bytes(split_manifest)}
        # Dev and holdout score against original raw labels and baseline inputs.
        # Absolute video references keep evaluation isolated from official val.
        for name, rr in partition_raw.items():
            payloads[output / "raw" / (name + ".jsonl")] = jsonl_bytes(rr)
            if name != "train":
                eval_rows = [{**r, "video": str((dataset_root / r["video"]).resolve())} for r in rr]
                payloads[output / "evaluation-inputs" / name / "val.jsonl"] = jsonl_bytes(eval_rows)
        dev_clean = [r for r in full_clean["experiment"] if r["sample_id"] in partition_ids["dev"]]
        dev_sft = sft.make_sft_records(dev_clean, dataset_root, ROOT, "omni", "val", {}, {},
            "single", sft.DATA_ORDER_SEED, audio_map, "emotion_only")
        sft.validate_sft_records(dev_sft, "omni", ROOT, "emotion_only")
        arm_summaries = {}
        original_by_id = {r["sample_id"]: r for r in raw}
        for arm, policy in ARMS.items():
            arm_root = output / arm
            cleaned = [r for r in full_clean[arm] if r["sample_id"] in partition_ids["train"]]
            records = sft.make_sft_records(cleaned, dataset_root, ROOT, "omni", "train",
                question_groups, question_samples, "single", sft.DATA_ORDER_SEED, audio_map, "full")
            for record in records:
                original = original_by_id[record["sample_id"]]
                record["use_in_main"] = deepcopy(original.get("use_in_main", False))
                record.setdefault("views", deepcopy(original["views"]))
            sft.validate_sft_records(records, "omni", ROOT, "full")
            aligned, conflict = [], []
            for record in records:
                (conflict if json.loads(record["messages"][-1]["content"])["is_uncertain"] == "yes" else aligned).append(record)
            if not aligned or not conflict:
                raise ValueError("Both training classes are required in both arms")
            entry = sft.dataset_info("")["affect_sft_omni_train"]
            info = {"protocol_train_aligned": {**deepcopy(entry), "file_name": "aligned.jsonl"},
                    "protocol_train_conflict": {**deepcopy(entry), "file_name": "conflict.jsonl", "num_samples": 2 * len(conflict)},
                    "protocol_dev": {**deepcopy(entry), "file_name": "dev.jsonl"}}
            payloads.update({arm_root / "dataset" / "aligned.jsonl": jsonl_bytes(aligned),
                             arm_root / "dataset" / "conflict.jsonl": jsonl_bytes(conflict),
                             arm_root / "dataset" / "dev.jsonl": jsonl_bytes(dev_sft),
                             arm_root / "dataset" / "dataset_info.json": json_bytes(info)})
            config = deepcopy(base_config)
            config.update(model_name_or_path=str(model_path), dataset_dir=str(arm_root / "dataset"), media_dir=str(ROOT),
                dataset="protocol_train_aligned,protocol_train_conflict", eval_dataset="protocol_dev",
                output_dir=str(arm_root / "checkpoints"), seed=training_seed, data_seed=training_seed,
                num_train_epochs=3.0, resume_from_checkpoint=None, flash_attn=flash_attn)
            if config.get("adapter_name_or_path"):
                raise ValueError("Both arms must start from the same raw base, never an existing adapter")
            payloads[arm_root / "config.yaml"] = yaml.safe_dump(config, allow_unicode=True, sort_keys=False).encode("utf-8")
            arm_summaries[arm] = {"input_policy": policy, "training_rows": len(records),
                "training_aligned": len(aligned), "training_conflicts": len(conflict),
                "sampled_training_rows": len(aligned) + 2 * len(conflict), "dev_rows": len(dev_sft),
                "from_raw_base": True, "trained": False}
        if file_hash(train_jsonl) != identity["train_sha256"]:
            raise ValueError("Raw training source changed during preparation")
        for path, content in payloads.items():
            if path.exists() and path.read_bytes() != content:
                raise ValueError("Prepared artifact differs from the frozen run: " + str(path))
        for path, content in payloads.items():
            write_immutable(path, content)
        manifest = {"algorithm": VERSION, "identity_hash": frozen["identity_hash"], "base_model": str(model_path),
                    "dataset_root": str(dataset_root), "arms": arm_summaries,
                    "audio_sha256": {str((ROOT / path).resolve()): file_hash(ROOT / path) for path in sorted(set(audio_map.values()))},
                    "artifacts": {str(path.relative_to(output)): file_hash(path) for path in payloads},
                    "selection": "highest development Score; ties select earliest epoch checkpoint",
                    "holdout": "evaluate only the selected checkpoint once; never used for selection",
                    "trained": False, "official_val_read": False, "official_test_read": False}
        write_immutable(output / "prepare-manifest.json", json_bytes(manifest))
        return manifest


def checked_run(output):
    output = Path(output).resolve()
    identity, manifest = read(output / "identity.json"), read(output / "prepare-manifest.json")
    if identity["identity_hash"] != fingerprint(identity["identity"]) or manifest["identity_hash"] != identity["identity_hash"]:
        raise ValueError("Prepared ablation identity changed")
    for relative, digest in manifest["artifacts"].items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output) or file_hash(path) != digest:
            raise ValueError("Prepared ablation artifact changed: " + relative)
    for name, digest in identity["identity"]["source_sha256"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError("Ablation source changed after preparation: " + name)
    if artifact_hash(manifest["base_model"]) != identity["identity"]["base_sha256"]:
        raise ValueError("Raw base model changed after preparation")
    if file_hash(identity["identity"]["train_jsonl"]) != identity["identity"]["train_sha256"]:
        raise ValueError("Raw training file changed after preparation")
    for name, digest in identity["identity"]["media_sha256"].items():
        if file_hash(Path(manifest["dataset_root"]) / name) != digest:
            raise ValueError("Original training media changed: " + name)
    for path, digest in manifest["audio_sha256"].items():
        if file_hash(path) != digest:
            raise ValueError("Prepared training audio changed: " + path)
    return identity, manifest


def evaluation_command(output, arm, adapter, partition):
    return limited_command(ROOT / "scripts/evaluate_sft.py", ["--config", str(output / arm / "config.yaml"),
        "--adapter-path", str(adapter), "--data-root", str(output / "evaluation-inputs" / partition),
        "--output-dir", str(output / arm / "evaluation"), "--run-name", adapter.name + "_" + partition,
        "--split", "val", "--load-in-4bit", "--fps", "2", "--min-frames", "4", "--max-frames", "64",
        "--min-visual-tokens", "64", "--max-visual-tokens", "64"])


def limited_command(script, arguments):
    # Bind and cap the selected visible device before importing the training or
    # inference entry point. The process exits fully between epoch evaluations.
    prelude = (
        "import runpy,sys,torch; torch.set_num_threads(4); torch.cuda.set_device(0); "
        "total=torch.cuda.get_device_properties(0).total_memory; "
        "torch.cuda.set_per_process_memory_fraction(min(1.,32*1024**3/total),0); "
        "sys.argv=sys.argv[1:]; "
        "from pathlib import Path; sys.path.insert(0,str(Path(sys.argv[0]).parent)); "
        "runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    return [sys.executable, "-c", prelude, str(script), *arguments]


def train(output_dir, arm, gpu="1", dry_run=False):
    if arm not in ARMS:
        raise ValueError("Choose control or experiment")
    output = Path(output_dir).resolve()
    identity, manifest = checked_run(output)
    arm_root = output / arm
    checkpoints = arm_root / "checkpoints"
    command = limited_command(ROOT / "scripts/train_sft.py", [str(arm_root / "config.yaml")])
    existing = sorted(checkpoints.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if existing and not (checkpoints / "train_results.json").exists():
        command.append("resume_from_checkpoint=" + str(existing[-1]))
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false", "FORCE_TORCHRUN": "0", "NPROC_PER_NODE": "1",
        "PYTHONPATH": str(ROOT / "third_party/LlamaFactory/src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "HF_DATASETS_CACHE": str(output / "cache" / arm), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    plan = {"training_command": command, "gpu": str(gpu), "arm": arm, "identity_hash": identity["identity_hash"],
            "pytorch_allocator_limit_gib": 32,
            "selection": manifest["selection"], "holdout": manifest["holdout"], "official_val_read": False,
            "official_test_read": False, "dry_run": dry_run}
    if dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return plan
    with run_lock(arm_root / "execution"):
        if not (checkpoints / "train_results.json").exists():
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
        adapters = sorted(checkpoints.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        adapters = [p for p in adapters if (p / "adapter_model.safetensors").is_file()]
        if not 1 <= len(adapters) <= 3:
            raise ValueError("Expected one epoch checkpoint per epoch, at most three")
        split = read(output / "split-manifest.json")
        counts = {name: value["rows"] for name, value in split["partitions"].items()}
        selected_path = arm_root / "selected-checkpoint.json"
        if not selected_path.exists():
            candidates = []
            for adapter in adapters:
                subprocess.run(evaluation_command(output, arm, adapter, "dev"), cwd=ROOT, env=environment, check=True)
                metric_path = arm_root / "evaluation" / (adapter.name + "_dev_metrics.json")
                metrics = read(metric_path)
                if metrics["samples"] != counts["dev"] or metrics["inference_errors"]:
                    raise ValueError("Incomplete development generation; checkpoint selection stopped")
                candidates.append({"checkpoint": str(adapter), "score": metrics["Score"], "metrics_sha256": file_hash(metric_path)})
            best = max(candidates, key=lambda candidate: candidate["score"])
            selection = {"algorithm": VERSION, "identity_hash": identity["identity_hash"], **best,
                         "adapter_sha256": artifact_hash(best["checkpoint"], adapter=True),
                         "selected_by": "training-only development Score", "candidates": candidates,
                         "holdout_used_for_selection": False}
            write_immutable(selected_path, json_bytes(selection))
        selected = read(selected_path)
        if selected["identity_hash"] != identity["identity_hash"] or selected["adapter_sha256"] != artifact_hash(selected["checkpoint"], adapter=True):
            raise ValueError("Selected checkpoint identity changed")
        report_path = arm_root / "holdout-report.json"
        if report_path.exists():
            previous = read(report_path)
            if (previous.get("identity_hash"), previous.get("arm"), previous.get("checkpoint"), previous.get("used_for_selection")) != (
                    identity["identity_hash"], arm, selected["checkpoint"], False):
                raise ValueError("Existing holdout report belongs to another selected checkpoint")
            return previous
        # This is the first point at which holdout generation can be launched.
        adapter = Path(selected["checkpoint"])
        subprocess.run(evaluation_command(output, arm, adapter, "holdout"), cwd=ROOT, env=environment, check=True)
        metrics = read(arm_root / "evaluation" / (adapter.name + "_holdout_metrics.json"))
        if metrics["samples"] != counts["holdout"] or metrics["inference_errors"]:
            raise ValueError("Incomplete selected-checkpoint holdout evaluation")
        result = {"algorithm": VERSION, "identity_hash": identity["identity_hash"], "arm": arm,
                  "checkpoint": str(adapter), "metrics": metrics, "used_for_selection": False,
                  "official_val_read": False, "official_test_read": False, "submitted_to_kaggle": False}
        write_immutable(report_path, json_bytes(result))
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, default=ROOT / "Dataset/train.jsonl")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "Dataset")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/Qwen2.5-Omni-3B")
    parser.add_argument("--questions-dir", type=Path, default=ROOT / "data/cleaned")
    parser.add_argument("--audio-cache-dir", type=Path)
    parser.add_argument("--training-seed", type=int, choices=(42, 43), default=42)
    parser.add_argument("--audio-workers", type=int, default=4)
    parser.add_argument("--flash-attn", choices=("sdpa", "fa2"), default="sdpa",
                        help="Shared by both arms; sdpa avoids a FlashAttention-2 installation requirement.")
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.output_dir, args.train_jsonl, args.dataset_root, args.model_path,
            args.questions_dir, args.audio_cache_dir, args.training_seed, args.audio_workers, args.flash_attn)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    if not args.arm:
        parser.error("train requires --arm control|experiment")
    return train(args.output_dir, args.arm, args.gpu, args.dry_run)


if __name__ == "__main__":
    main()
