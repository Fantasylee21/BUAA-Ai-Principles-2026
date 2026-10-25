#!/usr/bin/env python3
"""Convert cleaned affect data into LlamaFactory multimodal SFT datasets."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import random
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable


EMOTIONS = {"positive", "negative", "neutral"}
UNCERTAINTY = {"yes", "no"}
SYSTEM_PROMPT = (
    "You are a multimodal emotion analysis assistant. Judge the overall emotion "
    "from the provided modalities, explicitly represent uncertainty when the "
    "evidence is insufficient or conflicting, and follow the requested JSON schema."
)
DATA_ORDER_SEED = "buaa-affect-sft-v2-official-val"
EVAL_SYSTEM_PROMPT = (
    "You are a multimodal emotion classification assistant. Classify the overall "
    "emotion from the provided modalities and follow the requested JSON schema."
)
INPUT_POLICIES = {
    ("vl", "train", "VT"): {"vision", "text"},
    ("vl", "train", "VA"): {"vision", "text"},
    ("vl", "val", "VT"): {"vision", "text"},
    ("vl", "val", "VA"): {"vision"},
    ("omni", "train", "VT"): {"vision", "text"},
    ("omni", "train", "VA"): {"vision", "text", "audio"},
    ("omni", "val", "VT"): {"vision", "text"},
    ("omni", "val", "VA"): {"vision", "audio"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reproducible LlamaFactory SFT data for VL and Omni models."
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument(
        "--train-clean-dir", type=Path, default=None
    )
    parser.add_argument(
        "--val-clean-dir", type=Path, default=None
    )
    parser.add_argument(
        "--questions-dir", type=Path, default=Path("data/cleaned")
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("Dataset"))
    parser.add_argument(
        "--output-dir", type=Path, default=None
    )
    parser.add_argument(
        "--omni-input-policy", choices=("legacy", "protocol_consistent"), default="legacy",
        help="Use protocol-consistent cleaned inputs and a separate generated directory; legacy remains the default.",
    )
    parser.add_argument("--selection-seed", default=DATA_ORDER_SEED)
    parser.add_argument(
        "--question-policy",
        choices=("single", "expand"),
        default="single",
        help=(
            "single keeps one deterministic question per Conflict record; "
            "expand creates five examples per Conflict record"
        ),
    )
    parser.add_argument("--audio-workers", type=int, default=4)
    parser.add_argument("--overwrite-audio", action="store_true")
    parser.add_argument(
        "--skip-audio-extraction",
        action="store_true",
        help="Require previously extracted audio instead of invoking ffmpeg.",
    )
    return parser.parse_args()


def resolve_from_root(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
    path.chmod(0o644)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    os.replace(temporary, path)
    path.chmod(0o644)


def validate_source_rows(
    vl_rows: list[dict[str, Any]],
    omni_rows: list[dict[str, Any]],
    dataset_root: Path,
    *,
    require_unique_vl_groups: bool,
) -> None:
    vl_groups = [row["split_group_id"] for row in vl_rows]
    if require_unique_vl_groups and len(vl_groups) != len(set(vl_groups)):
        raise ValueError("VL cleaned data must contain one row per split_group_id.")

    known_groups = set(vl_groups)
    unknown_omni = {row["split_group_id"] for row in omni_rows} - known_groups
    if unknown_omni:
        raise ValueError(f"Omni rows contain unknown groups: {sorted(unknown_omni)[:5]}")

    seen_sample_ids: set[str] = set()
    for name, rows in (("VL", vl_rows), ("Omni", omni_rows)):
        for row in rows:
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{name} row has invalid sample_id: {row!r}")
            scoped_id = f"{name}:{sample_id}"
            if scoped_id in seen_sample_ids:
                raise ValueError(f"Duplicate {name} sample_id: {sample_id}")
            seen_sample_ids.add(scoped_id)
            if row.get("pred_emotion") not in EMOTIONS:
                raise ValueError(f"Invalid emotion in {sample_id}: {row.get('pred_emotion')}")
            if row.get("is_uncertain") not in UNCERTAINTY:
                raise ValueError(
                    f"Invalid uncertainty in {sample_id}: {row.get('is_uncertain')}"
                )
            video = dataset_root / row["video"]
            if not video.is_file():
                raise FileNotFoundError(video)


def validate_omni_input_policy(rows: list[dict[str, Any]], policy: str) -> None:
    """Never reinterpret already union-labelled legacy rows as original labels."""
    if policy not in ("legacy", "protocol_consistent"):
        raise ValueError("Unknown Omni input policy")
    for row in rows:
        if row.get("omni_input_policy", "legacy") != policy:
            raise ValueError("Cleaned Omni rows do not match --omni-input-policy")
        if policy == "protocol_consistent":
            if row["source_protocol"] not in ("VT", "VA"):
                raise ValueError("Protocol-consistent rows require VT or VA")
            expected = {"vision", "text" if row["source_protocol"] == "VT" else "audio"}
            views = row.get("views", {})
            if (set(row["input_modalities"]) != expected or
                    row["training_protocol"] != row["source_protocol"] or
                    row["sample_type"] != row["original_sample_type"] or
                    set(views) != {"M1", "M2", "M12"} or
                    row["pred_emotion"] != views["M12"]["label"] or
                    row["is_uncertain"] != ("yes" if row["sample_type"] == "Conflict" else "no")):
                raise ValueError("Protocol-consistent inputs must retain original protocol and labels")


def load_questions(
    questions_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    group_rows = read_jsonl(questions_dir / "clarifying_questions_by_group.jsonl")
    sample_rows = read_jsonl(
        questions_dir / "clarifying_questions_by_sample_id.jsonl"
    )
    by_group = {row["split_group_id"]: row for row in group_rows}
    by_sample = {row["sample_id"]: row for row in sample_rows}
    if len(by_group) != len(group_rows):
        raise ValueError("Duplicate split_group_id in group question labels.")
    if len(by_sample) != len(sample_rows):
        raise ValueError("Duplicate sample_id in sample question labels.")
    for row in [*group_rows, *sample_rows]:
        questions = row.get("questions")
        if (
            not isinstance(questions, list)
            or len(questions) != 5
            or len(set(questions)) != 5
            or not all(isinstance(question, str) and question.strip() for question in questions)
        ):
            raise ValueError(f"Invalid question labels in {row!r}")
    return by_group, by_sample


def select_questions(
    question_row: dict[str, Any], sample_id: str, policy: str, seed: str
) -> list[tuple[int, str]]:
    questions = question_row["questions"]
    if policy == "expand":
        return list(enumerate(questions))
    index = random.Random(f"{seed}:question:{sample_id}").randrange(len(questions))
    return [(index, questions[index])]


def build_user_prompt(
    text: str, include_audio: bool, include_text: bool, target_mode: str
) -> str:
    media_tokens = "<video><audio>" if include_audio else "<video>"
    if include_audio and include_text:
        modality_text = "The input contains the video, its audio, and supplied transcript text."
    elif include_audio:
        modality_text = (
            "The input contains the video and its audio. No supplied transcript "
            "text is available."
        )
    elif include_text:
        modality_text = "The input contains the video and supplied transcript text."
    else:
        modality_text = (
            "The input contains visual video frames only. Audio and supplied "
            "transcript text are unavailable."
        )
    common = (
        f"{media_tokens}\n"
        f"{modality_text}\n"
        "Analyze the person's overall emotion. Treat visible expression, body "
        "language, spoken or supplied text, and vocal evidence as separate signals "
        "when those signals are available.\n"
    )
    if include_text:
        common += f"Supplied text: {json.dumps(text, ensure_ascii=False)}\n"
    if target_mode == "emotion_only":
        return (
            common
            + "Return exactly one JSON object with only pred_emotion. Its value must "
            "be positive, negative, or neutral. Do not assess uncertainty, ask a "
            "clarifying question, or add Markdown or explanatory text."
        )
    return (
        common
        + "If the evidence conflicts or is insufficient for a reliable judgment, "
        "set is_uncertain to yes and ask one neutral clarifying question addressed "
        "to the current user. Otherwise set is_uncertain to no and "
        "clarifying_question to none. Return exactly one JSON object. pred_emotion "
        "must be positive, negative, or neutral; is_uncertain must be yes or no; "
        "clarifying_question must be a single string. Do not add Markdown or "
        "explanatory text."
    )


def assistant_target(
    row: dict[str, Any], question: str | None, target_mode: str
) -> str:
    if target_mode == "emotion_only":
        value = {"pred_emotion": row["pred_emotion"]}
    else:
        value = {
            "pred_emotion": row["pred_emotion"],
            "is_uncertain": row["is_uncertain"],
            "clarifying_question": question,
        }
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def relative_to_repo(path: Path, repo_root: Path) -> str:
    # Keep repository-facing paths stable even when Dataset is a symlink whose
    # target lives outside the checkout.
    try:
        return path.absolute().relative_to(repo_root.absolute()).as_posix()
    except ValueError:
        return path.absolute().as_posix()


def extracted_audio_path(output_dir: Path, video_rel: str) -> Path:
    return output_dir / "audio" / Path(video_rel).with_suffix(".flac")


def extract_one_audio(
    source: Path, destination: Path, overwrite: bool
) -> tuple[str, bool]:
    if destination.is_file() and destination.stat().st_size > 0 and not overwrite:
        return destination.as_posix(), False

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".tmp.flac", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        command = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "flac",
            str(temporary),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(
                f"ffmpeg failed for {source}: {completed.stderr.strip()}"
            )
        os.replace(temporary, destination)
        destination.chmod(0o644)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination.as_posix(), True


def prepare_audio(
    repo_root: Path,
    dataset_root: Path,
    output_dir: Path,
    omni_rows: list[dict[str, Any]],
    workers: int,
    overwrite: bool,
    skip_extraction: bool,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    if workers < 1:
        raise ValueError("audio_workers must be positive.")

    videos = sorted(
        {
            row["audio"]
            for row in omni_rows
            if row["source_protocol"] == "VA" and row.get("audio")
        }
    )
    paths: dict[str, tuple[Path, Path]] = {}
    for video_rel in videos:
        source = dataset_root / video_rel
        destination = extracted_audio_path(output_dir, video_rel)
        if not source.is_file():
            raise FileNotFoundError(source)
        paths[video_rel] = (source, destination)

    if skip_extraction:
        missing = [str(dst) for _, dst in paths.values() if not dst.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} extracted audio files, first: {missing[0]}"
            )
    else:
        completed_count = 0
        extracted_count = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(extract_one_audio, source, destination, overwrite): video_rel
                for video_rel, (source, destination) in paths.items()
            }
            for future in as_completed(futures):
                _, extracted = future.result()
                completed_count += 1
                extracted_count += int(extracted)
                if completed_count % 100 == 0 or completed_count == len(futures):
                    print(
                        json.dumps(
                            {
                                "event": "audio_progress",
                                "completed": completed_count,
                                "total": len(futures),
                                "newly_extracted": extracted_count,
                            }
                        ),
                        flush=True,
                    )

    audio_map = {
        video_rel: relative_to_repo(destination, repo_root)
        for video_rel, (_, destination) in paths.items()
    }
    manifest = []
    for video_rel in videos:
        source, destination = paths[video_rel]
        destination.chmod(0o644)
        manifest.append(
            {
                "source_video": relative_to_repo(source, repo_root),
                "audio": audio_map[video_rel],
                "audio_bytes": destination.stat().st_size,
                "sampling_rate": 16000,
                "channels": 1,
                "codec": "flac",
            }
        )
    return audio_map, manifest


def make_sft_records(
    rows: list[dict[str, Any]],
    dataset_root: Path,
    repo_root: Path,
    model_family: str,
    split_name: str,
    questions_by_group: dict[str, dict[str, Any]],
    questions_by_sample: dict[str, dict[str, Any]],
    question_policy: str,
    seed: str,
    audio_map: dict[str, str],
    target_mode: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        input_modalities = set(row["input_modalities"])
        include_audio = model_family == "omni" and "audio" in input_modalities
        include_text = "text" in input_modalities
        if include_audio:
            audio_rel = row.get("audio")
            if not audio_rel or audio_rel not in audio_map:
                raise ValueError(f"Missing extracted audio for {row['sample_id']}")
            audios = [audio_map[audio_rel]]
        else:
            audios = []

        if target_mode == "emotion_only":
            selected = [(None, None)]
        elif row["is_uncertain"] == "yes":
            question_row = (
                questions_by_group.get(row["split_group_id"])
                if model_family == "vl"
                else questions_by_sample.get(row["sample_id"])
            )
            if question_row is None:
                raise ValueError(f"Missing questions for {model_family}:{row['sample_id']}")
            selected = select_questions(
                question_row, row["sample_id"], question_policy, seed
            )
        else:
            selected = [(None, "none")]

        for question_index, question in selected:
            record = {
                "messages": [
                    {
                        "role": "user",
                        "content": build_user_prompt(
                            row["text"], include_audio, include_text, target_mode
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": assistant_target(row, question, target_mode),
                    },
                ],
                "system": (
                    EVAL_SYSTEM_PROMPT
                    if target_mode == "emotion_only"
                    else SYSTEM_PROMPT
                ),
                "videos": [relative_to_repo(dataset_root / row['video'], repo_root)],
                "sample_id": row["sample_id"],
                "split_group_id": row["split_group_id"],
                "source_dataset": row["source_dataset"],
                "source_protocol": row["source_protocol"],
                "training_protocol": row["training_protocol"],
                "input_modalities": sorted(input_modalities),
                "split": split_name,
                "target_mode": target_mode,
                "question_index": question_index,
            }
            if model_family == "omni":
                record["audios"] = audios
                if row.get("omni_input_policy") == "protocol_consistent":
                    record.update({key: deepcopy(row[key]) for key in (
                        "omni_input_policy", "views", "view_supervision", "use_in_main",
                        "sample_type", "original_sample_type")})
            records.append(record)

    # Deterministic mixing avoids source-order bias. It also ensures Omni VT
    # and VA rows coexist in Arrow's first conversion batch, so the optional
    # audio column is inferred as list<string> rather than null.
    random.Random(f"{seed}:order:{model_family}:{split_name}").shuffle(records)
    return records


def validate_sft_records(
    records: list[dict[str, Any]],
    model_family: str,
    repo_root: Path,
    target_mode: str,
) -> dict[str, Any]:
    sample_ids: set[str] = set()
    groups: set[str] = set()
    targets = Counter()
    protocols = Counter()
    for index, record in enumerate(records):
        messages = record.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or [message.get("role") for message in messages] != ["user", "assistant"]
        ):
            raise ValueError(f"Invalid messages at {model_family}:{index}")

        user_content = messages[0].get("content", "")
        videos = record.get("videos")
        audios = record.get("audios", [])
        protocol = record.get("source_protocol")
        split_name = record.get("split")
        modalities = set(record.get("input_modalities", []))
        expected_modalities = INPUT_POLICIES.get(
            (model_family, split_name, protocol)
        )
        policy = record.get("omni_input_policy", "legacy")
        if policy not in ("legacy", "protocol_consistent"):
            raise ValueError(f"Unknown Omni input policy at {model_family}:{index}")
        if model_family == "omni" and policy == "protocol_consistent":
            if protocol not in ("VT", "VA") or split_name not in ("train", "val"):
                raise ValueError(f"Invalid protocol/split at {model_family}:{index}")
            expected_modalities = {"vision", "text" if protocol == "VT" else "audio"}
        if modalities != expected_modalities:
            raise ValueError(
                f"Input policy mismatch at {model_family}:{index}: "
                f"got {sorted(modalities)}, expected {sorted(expected_modalities or [])}"
            )
        if not isinstance(videos, list) or len(videos) != 1:
            raise ValueError(f"Invalid videos at {model_family}:{index}")
        if user_content.count("<video>") != len(videos):
            raise ValueError(f"Video placeholder mismatch at {model_family}:{index}")
        if user_content.count("<audio>") != len(audios):
            raise ValueError(f"Audio placeholder mismatch at {model_family}:{index}")
        if ("Supplied text:" in user_content) != ("text" in modalities):
            raise ValueError(f"Text prompt mismatch at {model_family}:{index}")
        for media_path in [*videos, *audios]:
            if not (repo_root / media_path).is_file():
                raise FileNotFoundError(repo_root / media_path)

        target = json.loads(messages[1]["content"])
        if target_mode == "emotion_only":
            if set(target) != {"pred_emotion"}:
                raise ValueError(f"Invalid emotion-only target at {model_family}:{index}")
        else:
            if set(target) != {
                "pred_emotion",
                "is_uncertain",
                "clarifying_question",
            }:
                raise ValueError(f"Invalid target schema at {model_family}:{index}")
            if target["is_uncertain"] not in UNCERTAINTY:
                raise ValueError(f"Invalid uncertainty at {model_family}:{index}")
            if target["is_uncertain"] == "no" and target["clarifying_question"] != "none":
                raise ValueError(f"Aligned target has a question at {model_family}:{index}")
            if target["is_uncertain"] == "yes" and target["clarifying_question"] == "none":
                raise ValueError(f"Conflict target lacks a question at {model_family}:{index}")
        if target["pred_emotion"] not in EMOTIONS:
            raise ValueError(f"Invalid emotion at {model_family}:{index}")

        sample_ids.add(record["sample_id"])
        groups.add(record["split_group_id"])
        target_key = (
            target["pred_emotion"]
            if target_mode == "emotion_only"
            else f"{target['pred_emotion']}:{target['is_uncertain']}"
        )
        targets[target_key] += 1
        protocols[record["source_protocol"]] += 1

    return {
        "examples": len(records),
        "base_sample_ids": len(sample_ids),
        "groups": len(groups),
        "target_mode": target_mode,
        "target_distribution": dict(sorted(targets.items())),
        "source_protocol_distribution": dict(sorted(protocols.items())),
    }


def dataset_info(file_prefix: str = "generated") -> dict[str, Any]:
    common_tags = {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
    }
    entries: dict[str, Any] = {}
    for model_family in ("vl", "omni"):
        for split in ("train", "val"):
            columns = {
                "messages": "messages",
                "system": "system",
                "videos": "videos",
            }
            if model_family == "omni":
                columns["audios"] = "audios"
            entries[f"affect_sft_{model_family}_{split}"] = {
                "file_name": str(Path(file_prefix) / f"affect_sft_{model_family}_{split}.jsonl").replace("\\", "/"),
                "formatting": "sharegpt",
                "columns": columns,
                "tags": common_tags,
            }
    return entries


def main() -> None:
    args = parse_args()
    inferred_root = Path(__file__).resolve().parents[1]
    repo_root = (args.repo_root or inferred_root).resolve()
    consistent = args.omni_input_policy == "protocol_consistent"
    clean_root = Path("data/cleaned/protocol_consistent")
    train_clean_dir = resolve_from_root(repo_root, args.train_clean_dir or (clean_root / "train" if consistent else Path("data/cleaned/all")))
    val_clean_dir = resolve_from_root(repo_root, args.val_clean_dir or (clean_root / "val" if consistent else Path("data/cleaned/val")))
    questions_dir = resolve_from_root(repo_root, args.questions_dir)
    dataset_root = resolve_from_root(repo_root, args.dataset_root)
    output_dir = resolve_from_root(repo_root, args.output_dir or Path(
        "data/llamafactory/generated/protocol_consistent" if consistent else "data/llamafactory/generated"))
    if consistent and output_dir.resolve() == (repo_root / "data/llamafactory/generated").resolve():
        raise ValueError("Protocol-consistent SFT requires a separate generated output directory")

    source_paths = {
        "vl_train": train_clean_dir / "train_vl_clean.jsonl",
        "omni_train": train_clean_dir / "train_omni_clean.jsonl",
        "vl_val": val_clean_dir / "val_vl_clean.jsonl",
        "omni_val": val_clean_dir / "val_omni_clean.jsonl",
    }
    source_rows = {name: read_jsonl(path) for name, path in source_paths.items()}
    for split_name in ("train", "val"):
        validate_omni_input_policy(source_rows[f"omni_{split_name}"], args.omni_input_policy)
    validate_source_rows(
        source_rows["vl_train"],
        source_rows["omni_train"],
        dataset_root,
        require_unique_vl_groups=True,
    )
    validate_source_rows(
        source_rows["vl_val"],
        source_rows["omni_val"],
        dataset_root,
        require_unique_vl_groups=False,
    )

    train_groups = {row["split_group_id"] for row in source_rows["vl_train"]}
    val_groups = {row["split_group_id"] for row in source_rows["vl_val"]}
    if overlap := train_groups & val_groups:
        raise ValueError(f"Official train/val group overlap: {sorted(overlap)[:5]}")
    train_samples = {row["sample_id"] for row in source_rows["omni_train"]}
    val_samples = {row["sample_id"] for row in source_rows["omni_val"]}
    if overlap := train_samples & val_samples:
        raise ValueError(f"Official train/val sample overlap: {sorted(overlap)[:5]}")

    questions_by_group, questions_by_sample = load_questions(questions_dir)

    audio_map, audio_manifest = prepare_audio(
        repo_root=repo_root,
        dataset_root=dataset_root,
        output_dir=output_dir,
        omni_rows=[*source_rows["omni_train"], *source_rows["omni_val"]],
        workers=args.audio_workers,
        overwrite=args.overwrite_audio,
        skip_extraction=args.skip_audio_extraction,
    )
    write_jsonl(output_dir / "audio_manifest.jsonl", audio_manifest)

    all_stats: dict[str, Any] = {}
    for model_family in ("vl", "omni"):
        family_groups: dict[str, set[str]] = {}
        for split_name in ("train", "val"):
            rows = source_rows[f"{model_family}_{split_name}"]
            target_mode = "full" if split_name == "train" else "emotion_only"
            records = make_sft_records(
                rows=rows,
                dataset_root=dataset_root,
                repo_root=repo_root,
                model_family=model_family,
                split_name=split_name,
                questions_by_group=questions_by_group,
                questions_by_sample=questions_by_sample,
                question_policy=args.question_policy,
                seed=args.selection_seed,
                audio_map=audio_map,
                target_mode=target_mode,
            )
            output_path = output_dir / f"affect_sft_{model_family}_{split_name}.jsonl"
            write_jsonl(output_path, records)
            stats = validate_sft_records(
                records, model_family, repo_root, target_mode
            )
            stats["file"] = relative_to_repo(output_path, repo_root)
            all_stats[f"{model_family}_{split_name}"] = stats
            family_groups[split_name] = {
                record["split_group_id"] for record in records
            }
        if family_groups["train"] & family_groups["val"]:
            raise RuntimeError(f"Group leakage detected in {model_family} outputs.")

    generation_manifest = {
        "format": "LlamaFactory ShareGPT multimodal SFT",
        "omni_input_policy": args.omni_input_policy,
        "question_policy": args.question_policy,
        "selection_seed": args.selection_seed,
        "partitions": {
            "train": {
                "source": relative_to_repo(train_clean_dir, repo_root),
                "groups": len(train_groups),
                "target_mode": "full",
                "input_policy": {
                    "vl": {"VT": ["vision", "text"], "VA": ["vision", "text"]},
                    "omni": {
                        "VT": ["vision", "text"],
                        "VA": ["vision", "audio"] if consistent else ["vision", "text", "audio"],
                    },
                },
            },
            "val": {
                "source": relative_to_repo(val_clean_dir, repo_root),
                "groups": len(val_groups),
                "target_mode": "emotion_only",
                "input_policy": {
                    "vl": {"VT": ["vision", "text"], "VA": ["vision"]},
                    "omni": {
                        "VT": ["vision", "text"],
                        "VA": ["vision", "audio"],
                    },
                },
            },
        },
        "source_files": [
            relative_to_repo(path, repo_root)
            for path in [*source_paths.values(),
                         questions_dir / "clarifying_questions_by_group.jsonl",
                         questions_dir / "clarifying_questions_by_sample_id.jsonl"]
        ],
        "audio": {
            "files": len(audio_manifest),
            "sampling_rate": 16000,
            "channels": 1,
            "codec": "flac",
        },
        "datasets": all_stats,
        "llamafactory_runtime": {
            "dataset_dir": relative_to_repo(output_dir, repo_root) if consistent else "data/llamafactory",
            "media_dir": ".",
            "dataset_names": sorted(dataset_info()),
        },
    }
    write_json(output_dir / "generation_manifest.json", generation_manifest)
    if consistent:
        write_json(output_dir / "dataset_info.json", dataset_info(""))
    else:
        write_json(repo_root / "data/llamafactory/dataset_info.json", dataset_info())

    print(
        json.dumps(
            {
                "event": "finished",
                "question_policy": args.question_policy,
                "partitions": generation_manifest["partitions"],
                "audio_files": len(audio_manifest),
                "datasets": all_stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
