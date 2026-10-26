"""Durable, train-only development and locked delivery of joint affect models.

Only ``run`` starts child trainers. ``prepare`` hashes training data, and
``progress`` reads state. No command submits anything to Kaggle.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import csv
import hashlib
import io
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import threading
import time
import traceback

VERSION = "joint_affect_pipeline_v1"
SOURCE = Path(__file__).resolve().parents[1]
LABELS = ("negative", "neutral", "positive")
FIELDS = ("sample_id", "pred_emotion", "is_uncertain", "clarifying_question")
HOURS = 120
EPS = 1e-12
OFFICIAL_SPLITS = (("val", "val.jsonl", 666), ("test", "test_release.jsonl", 734))
SOURCE_FILES = ("scripts/joint_affect_pipeline.py", "scripts/train_joint_affect.py",
    "scripts/joint_affect_model.py", "scripts/joint_affect_data.py",
    "scripts/joint_affect_progress.py", "scripts/run_joint_affect_server.sh",
    "scripts/train_sft.py", "scripts/task_balanced_loss.py", "scripts/evaluate_sft.py",
    "scripts/protocol_sft_ablation.py", "scripts/protocol_sft_summary.py",
    "scripts/prepare_sft_data.py", "data/prepare_training_data.py",
    "qwen25vl_baseline.py", "sft_prompt.py", "third_party/llamafactory.lock.yaml",
    "configs/sft/qwen25omni/qlora_sft_v4_resample2x.yaml")
SOURCE_FILES += ("data/cleaned/clarifying_questions_by_group.jsonl", "data/cleaned/clarifying_questions_by_sample_id.jsonl")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + str(threading.get_ident()))
    temporary.write_bytes(encoded(value))
    temporary.replace(path)


def immutable(path, value):
    path = Path(path)
    content = encoded(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("Refusing to replace immutable artifact: " + str(path))
    else:
        with path.open("xb") as stream:
            stream.write(content)


@contextmanager
def run_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stream = (root / ".pipeline.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            if not stream.read(1):
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        stream.close()


def source_hashes():
    names = sorted(set(SOURCE_FILES) | {p.relative_to(SOURCE).as_posix() for p in (SOURCE / "rl").glob("*.py")})
    return {name: digest(SOURCE / name) for name in names}


def runtime_identity():
    packages = ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "tokenizers",
                "numpy", "scipy", "av", "soundfile", "qwen-vl-utils", "datasets", "llamafactory")
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "executable": str(Path(sys.executable).absolute()), "packages": versions}


def base_inventory(project, *, hash_files=True):
    result = {}
    for model in ("Qwen2.5-Omni-3B", "Qwen3-VL-8B-Instruct"):
        root = Path(project) / "models" / model
        if not (root / "config.json").is_file():
            raise ValueError("Missing local raw backbone: " + model)
        files = sorted(p for p in root.rglob("*") if p.is_file() and
                       not any(x.startswith(".") for x in p.relative_to(root).parts))
        if not any(p.suffix == ".safetensors" for p in files):
            raise ValueError("Missing raw backbone weight shards: " + model)
        for path in files:
            stat = path.stat()
            entry = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if hash_files:
                entry["sha256"] = digest(path)
            result[model + "/" + path.relative_to(root).as_posix()] = entry
    return result


def verify_environment(project, identity):
    if "base_artifacts" in identity:
        current = base_inventory(project, hash_files=False)
        expected = {k: {n: v[n] for n in ("bytes", "mtime_ns")} for k, v in identity["base_artifacts"].items()}
        if current != expected:
            raise ValueError("Raw backbone artifact inventory changed after SHA freezing")
    if "runtime" in identity and runtime_identity() != identity["runtime"]:
        raise ValueError("Training runtime changed after plan freezing")
    if "framework_artifacts" in identity and framework_inventory(project) != identity["framework_artifacts"]:
        raise ValueError("External training framework source changed after plan freezing")


def framework_inventory(project):
    root = Path(project) / "third_party/LlamaFactory/src"
    names = sorted(root.rglob("*.py"))
    if not names:
        raise ValueError("Missing pinned LlamaFactory source")
    return {p.relative_to(root).as_posix(): digest(p) for p in names}


def prepare(project_root, run_dir, expected_rows=3354):
    from scripts.joint_affect_data import prepare_manifest
    project, output = Path(project_root).resolve(), Path(run_dir).resolve()
    if not output.is_relative_to((project / "runs").resolve()):
        raise ValueError("Pipeline output must be inside the project's ignored runs directory")
    manifest = prepare_manifest(project / "Dataset/train.jsonl", project / "Dataset", expected_rows=expected_rows)
    immutable(output / "train-manifest.json", manifest)
    identity = {"algorithm": VERSION, "train_manifest_sha256": digest(output / "train-manifest.json"),
        "source_sha256": source_hashes(), "project_root": str(project), "budget_hours": HOURS,
        "base_artifacts": base_inventory(project), "runtime": runtime_identity(),
        "framework_artifacts": framework_inventory(project),
        "grouping_seed": 42, "training_seeds": [42, 43], "folds": 3,
        "sft_reference_epochs": 3, "candidate_schedule": {"warmup_epochs": 1, "main_epochs": 2},
        "initial_max_frames": 64, "oom_fallback_max_frames": 32, "fallbacks": 1,
        "allowed_gpus": [1, 2], "memory_gib_per_card": 40,
        "label_order": list(LABELS), "uncertainty_threshold": .5, "threshold_rule": ">=",
        "selection": "0.5*F1_PN+0.3*conflict_recall+0.2*aligned_specificity",
        "primary_gate": {"pooled_gain": .01, "nondecreasing_folds": 2, "f1_pn_drop_max": .005,
            "natural_tp_gain": 5, "natural_va_tp_gain": 2, "fp_increase_fraction_max": .01},
        "pressure_uncertainty_gain_min": .005, "architecture_tie": "C0", "vision_tie": "omni",
        "final_probability_ensemble": "arithmetic_mean_seed42_seed43",
        "official_splits_for_selection": False, "submitted_to_kaggle": False}
    plan = {"identity": identity, "identity_hash": fingerprint(identity)}
    immutable(output / "plan.json", plan)
    return plan


class Budget:
    """A persisted wall deadline plus monotonic charging within each process."""
    def __init__(self, root, seconds=HOURS * 3600, wall=time.time, mono=time.monotonic):
        self.path, self.wall, self.mono = Path(root) / "clock.json", wall, mono
        self.mutex = threading.Lock()
        now = wall()
        if self.path.exists():
            old = read(self.path)
            if now < old["last_wall_unix"] - 2:
                raise RuntimeError("Wall clock moved backwards; refusing to extend the budget")
            self.started, self.deadline = old["started_unix"], old["deadline_unix"]
            self.remaining = min(old["remaining_seconds"], self.deadline - now)
        else:
            self.started, self.deadline, self.remaining = now, now + seconds, seconds
        self.last_mono = mono()
        self.tick()

    def tick(self):
        with self.mutex:
            now, current = self.wall(), self.mono()
            self.remaining = min(self.remaining - max(0., current - self.last_mono), self.deadline - now)
            self.last_mono = current
            value = {"started_unix": self.started, "deadline_unix": self.deadline,
                "remaining_seconds": max(0., self.remaining), "last_wall_unix": now}
            atomic(self.path, value)
            if self.remaining <= 0:
                raise TimeoutError("The original 120-hour batch budget is exhausted")
            return min(self.deadline, now + self.remaining)


class ChildFailure(RuntimeError):
    def __init__(self, message, oom=False):
        super().__init__(message)
        self.oom = oom


def is_cuda_oom(text):
    lowered = text.lower()
    return any(value in lowered for value in ("cuda out of memory", "cuda error: out of memory",
                                               "torch.cuda.outofmemoryerror"))


def new_log_tail(path, initial_bytes=0, limit=65536):
    """Read only this attempt's suffix; historical OOMs cannot trigger fallback."""
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        return ""
    size = path.stat().st_size
    # A child may intentionally replace its own attempt log.
    offset = initial_bytes if size >= initial_bytes else 0
    with path.open("rb") as stream:
        stream.seek(max(offset, size - limit))
        return stream.read(limit).decode("utf-8", errors="replace")


def process_token(pid):
    """Linux start ticks distinguish a retained child from a recycled PID."""
    try:
        value = Path("/proc") / str(pid) / "stat"
        text = value.read_text(encoding="utf-8")
        return text[text.rfind(")") + 2:].split()[19]
    except (OSError, IndexError):
        return None


def checked_predictions(path, rows):
    result = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids) or [r.get("sample_id") for r in result] != ids:
        raise ValueError("Predictions must cover exact split IDs once, in original order")
    for value in result:
        probabilities, uncertainty = value.get("emotion_probs"), value.get("uncertain_prob")
        if (not isinstance(probabilities, list) or len(probabilities) != 3 or
                any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities) or
                abs(sum(probabilities) - 1.) > 1e-5 or type(uncertainty) not in (int, float) or
                not math.isfinite(uncertainty) or not 0 <= uncertainty <= 1):
            raise ValueError("Invalid or nonfinite classification probabilities")
    return result


def public_metrics(value):
    if isinstance(value, dict):
        return {k: public_metrics(v) for k, v in value.items() if not k.startswith("_")}
    if isinstance(value, list):
        return [public_metrics(v) for v in value]
    return value


def compare(rows, reference, candidate, design):
    from scripts.joint_affect_data import metrics
    base, new = metrics(rows, reference), metrics(rows, candidate)
    folds = []
    for partition in design["folds"]:
        ix = partition["held"]
        a = metrics([rows[i] for i in ix], [reference[i] for i in ix])
        b = metrics([rows[i] for i in ix], [candidate[i] for i in ix])
        from scripts.joint_affect_data import targets
        state = lambda row: 4 * targets(row)["vision"] + targets(row)["second"]
        train_indices = partition.get("train", [i for i in range(len(rows)) if i not in set(ix)])
        seen = {state(rows[i]) for i in train_indices}
        unseen = [i for i in ix if state(rows[i]) not in seen]
        folds.append({"fold": partition["fold"], "control": public_metrics(a), "candidate": public_metrics(b),
                      "gain": b["selection_score"] - a["selection_score"], "seen_joint_states": sorted(seen),
                      "unseen_joint_states": sorted({state(rows[i]) for i in unseen}), "unseen_joint_rows": len(unseen),
                      "unseen_joint_candidate": public_metrics(metrics([rows[i] for i in unseen], [candidate[i] for i in unseen])) if unseen else None})
    gain = new["selection_score"] - base["selection_score"]
    checks = {"pooled_gain_at_least_0_01": gain >= .01 - EPS,
        "at_least_two_folds_do_not_regress": sum(v["gain"] >= -EPS for v in folds) >= 2,
        "f1_pn_drop_at_most_0_005": new["f1_pn"] >= base["f1_pn"] - .005 - EPS,
        "natural_tp_gain_at_least_5": new["natural_tp"] - base["natural_tp"] >= 5,
        "natural_va_tp_gain_at_least_2": new["natural_va_tp"] - base["natural_va_tp"] >= 2,
        "fp_increase_within_one_percent": new["false_positives"] - base["false_positives"] <=
            math.floor(.01 * base["n_aligned"] + EPS)}
    return {"control": public_metrics(base), "candidate": public_metrics(new), "folds": folds,
            "gain": gain, "checks": checks, "passed": all(checks.values())}


def independent_audit(rows, predictions, report):
    """Recompute from a confusion matrix and raw flags, outside data.metrics."""
    confusion, counts = Counter(), Counter()
    for row, value in zip(rows, predictions):
        predicted = LABELS[max(range(3), key=lambda i: value["emotion_probs"][i])]
        confusion[row["views"]["M12"]["label"], predicted] += 1
        group = "conflict" if row["sample_type"] == "Conflict" else "aligned"
        counts[group] += 1
        counts[group + "_yes"] += int(value["uncertain_prob"] >= .5)
    scores = []
    for label in LABELS:
        denominator = sum(v for (a, b), v in confusion.items() if a == label) + sum(v for (a, b), v in confusion.items() if b == label)
        scores.append(2 * confusion[label, label] / denominator if denominator else 0.)
    recall = counts["conflict_yes"] / counts["conflict"] if counts["conflict"] else None
    specificity = 1 - counts["aligned_yes"] / counts["aligned"] if counts["aligned"] else None
    audited = {"f1_pn": (scores[0] + scores[2]) / 2, "macro_f1_fixed3": sum(scores) / 3,
               "true_positives": counts["conflict_yes"], "false_positives": counts["aligned_yes"],
               "conflict_recall": recall, "specificity": specificity}
    for key, value in audited.items():
        if value is None and report[key] is None:
            continue
        if value is None or report[key] is None or abs(value - report[key]) > 1e-10:
            raise ValueError("Independent confusion-matrix audit disagrees: " + key)
    return {"passed": True, "recomputed": audited,
            "confusion_matrix": [[confusion[a, b] for b in LABELS] for a in LABELS]}


def ensemble(first, second):
    if [r["sample_id"] for r in first] != [r["sample_id"] for r in second]:
        raise ValueError("Final ensemble IDs differ")
    return [{"sample_id": a["sample_id"],
             "emotion_probs": [(p + q) / 2 for p, q in zip(a["emotion_probs"], b["emotion_probs"])],
             "uncertain_prob": (a["uncertain_prob"] + b["uncertain_prob"]) / 2}
            for a, b in zip(first, second)]


def candidate_csv(rows, predictions):
    from scripts.joint_affect_data import question_for_protocol
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    if [r["sample_id"] for r in rows] != [p["sample_id"] for p in predictions]:
        raise ValueError("Candidate order differs from the locked official input")
    for row, prediction in zip(rows, predictions):
        uncertain = prediction["uncertain_prob"] >= .5
        writer.writerow({"sample_id": row["sample_id"],
            "pred_emotion": LABELS[max(range(3), key=lambda i: prediction["emotion_probs"][i])],
            "is_uncertain": "yes" if uncertain else "no",
            "clarifying_question": question_for_protocol(row["protocol"]) if uncertain else "none"})
    content = stream.getvalue().encode("utf-8")
    check = csv.DictReader(io.StringIO(content.decode("utf-8"), newline=""))
    saved = list(check)
    if tuple(check.fieldnames) != FIELDS or len(saved) != len(rows):
        raise ValueError("Invalid four-column candidate schema")
    if any((r["is_uncertain"] == "no") != (r["clarifying_question"] == "none") or
           not r["clarifying_question"].strip() for r in saved):
        raise ValueError("Invalid candidate question contract")
    return content


class Pipeline:
    def __init__(self, project_root, run_dir, *, resume_failed=False, runner=None):
        self.project, self.root = Path(project_root).resolve(), Path(run_dir).resolve()
        if not self.root.is_relative_to(self.project / "runs"):
            raise ValueError("Pipeline output must remain under the project's runs directory")
        self.plan = read(self.root / "plan.json")
        self.identity = self.plan["identity"]
        if self.plan["identity_hash"] != fingerprint(self.identity) or self.identity["algorithm"] != VERSION:
            raise ValueError("Corrupt pipeline identity")
        if str(self.project) != self.identity["project_root"]:
            raise ValueError("Project root differs from frozen plan")
        if digest(self.root / "train-manifest.json") != self.identity["train_manifest_sha256"]:
            raise ValueError("Frozen training manifest changed")
        self.manifest = read(self.root / "train-manifest.json")
        self.rows, self.designs = self.manifest["train"], self.manifest["folds"]
        self.budget = None
        self.resume_failed, self.runner = resume_failed, runner
        self.mutex, self.children, self.active = threading.Lock(), {}, {}
        self.cancel = threading.Event()
        self.resource_condition = threading.Condition()
        self.available_gpus = {1, 2}
        self.phase, self.frames = "initializing", 64

    def check_sources(self):
        for name, expected in self.identity["source_sha256"].items():
            if digest(SOURCE / name) != expected:
                raise ValueError("Source changed after plan freezing: " + name)
        verify_environment(self.project, self.identity)

    def state(self, phase=None, **extra):
        with self.mutex:
            if phase is not None:
                self.phase = phase
            atomic(self.root / "status.json", {"algorithm": VERSION, "phase": self.phase,
                "max_frames": self.frames, "active_jobs": dict(self.active),
                "updated_unix": time.time(), "submitted_to_kaggle": False, **extra})

    def spec(self, variant, fold, grouping, seed, vision):
        return {"variant": variant, "fold": fold, "grouping": grouping, "seed": seed,
                "vision_model": vision, "max_frames": self.frames,
                "manifest_sha256": self.identity["train_manifest_sha256"]}

    def job_path(self, job, preflight=False):
        return self.root / ("cohort-" + str(self.frames)) / ("preflight" if preflight else "jobs") / job["grouping"] / job["vision_model"] / job["variant"] / ("seed" + str(job["seed"])) / ("fold" + str(job["fold"]))

    def validate_marker(self, path, job, kind):
        value = read(path)
        actual = value.get("job", value.get("identity", value))
        if any(actual.get(k) != v for k, v in job.items()):
            raise ValueError("Child result crossed job identity: " + str(path))
        if kind == "preflight":
            peak = value.get("peak_reserved_gib")
            if (value.get("samples") != 32 or type(peak) not in (int, float) or
                    not math.isfinite(peak) or not 0 <= peak <= 40 + EPS or
                    not value.get("actual_media_audit") or not value.get("gradient_and_update_audit") or
                    value.get("losses_finite") is not True):
                raise ValueError("Actual 32-example / 40-GiB preflight is missing")
        elif not isinstance(value.get("checkpoint_sha256"), str) or not value["checkpoint_sha256"]:
            raise ValueError("Training completion lacks a locked checkpoint digest")
        return value

    def stop_children(self):
        with self.mutex:
            children = list(self.children.values())
        for process in children:
            try:
                if process.poll() is None:
                    if os.name == "nt":
                        process.terminate()
                    elif os.getpgid(process.pid) == process.pid:
                        os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def validate_prediction_marker(self, output, job, split, trained, count):
        # The trainer's completion is independent of the pipeline's validated
        # receipt. A complete-looking prediction prefix is never sufficient.
        marker = Path(str(output) + ".complete.json")
        value = read(marker)
        if (value.get("job") != job or value.get("split") != split or value.get("rows") != count or
                value.get("sha256") != digest(output) or
                value.get("checkpoint_sha256") != trained["checkpoint_sha256"]):
            raise ValueError("Prediction completion or checkpoint identity changed")
        return value

    @contextmanager
    def resources(self, job):
        # Allocate two cards atomically: two waiting 8B jobs must never each
        # retain one card while waiting for the other.
        dual = job["vision_model"] == "qwen3vl8b"
        with self.resource_condition:
            while len(self.available_gpus) < (2 if dual else 1):
                if self.cancel.is_set():
                    raise ChildFailure("Cohort cancelled while waiting for GPU resources")
                self.resource_condition.wait(timeout=.5)
            if self.cancel.is_set():
                raise ChildFailure("Cohort cancelled before GPU allocation")
            allocated = (1, 2) if dual else (min(self.available_gpus),)
            self.available_gpus.difference_update(allocated)
        try:
            yield allocated
        finally:
            with self.resource_condition:
                self.available_gpus.update(allocated)
                self.resource_condition.notify_all()

    def child(self, command, job_dir, output_marker=None):
        self.check_sources()
        with self.mutex:
            if self.budget is None:
                self.budget = Budget(self.root)
        deadline = self.budget.tick()
        command += ["--deadline-unix", str(deadline)]
        if self.runner:
            return self.runner(command, job_dir, output_marker)
        if self.cancel.is_set():
            raise ChildFailure("Cohort cancelled after another job failed")
        environment = {**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1", "JOINT_AFFECT_MEMORY_GIB": "40"}
        if "--gpus" in command:
            gpu = command[command.index("--gpus") + 1]
            if gpu != "1,2":
                raise ValueError("The visual 8B expert requires physical GPUs 1 and 2")
        else:
            gpu = command[command.index("--gpu") + 1]
            if gpu not in ("1", "2"):
                raise ValueError("Only physical GPU 1 or 2 is authorized")
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        for attempt in range(6):
            used = subprocess.check_output(["nvidia-smi", "--id=" + gpu,
                "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip()
            cards = used.splitlines()
            if len(cards) == len(gpu.split(",")) and all(v.strip().isdigit() and int(v.strip()) <= 512 for v in cards):
                break
            self.budget.tick()
            if attempt == 5:
                raise ChildFailure("GPU " + gpu + " is occupied; other processes were not touched")
            time.sleep(2)
        log_path = job_dir / (command[3] + ".log")
        old_log_sizes = {str(path): path.stat().st_size for path in job_dir.rglob("*.log") if path.is_file()}
        with log_path.open("ab") as log:
            process = subprocess.Popen(command, cwd=SOURCE, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            with self.mutex:
                self.children[str(job_dir)] = process
            atomic(job_dir / "process.json", {"pid": process.pid, "command": command,
                "process_start_ticks": process_token(process.pid),
                "started_unix": time.time(), "job_dir": str(job_dir)})
            try:
                while process.poll() is None:
                    if self.cancel.is_set():
                        raise ChildFailure("Cohort cancelled after another job failed")
                    self.budget.tick()
                    self.state()
                    time.sleep(3)
                code = process.returncode
            except BaseException:
                self.cancel.set()
                self.stop_children()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        process.kill()
                    elif os.getpgid(process.pid) == process.pid:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
            finally:
                with self.mutex:
                    self.children.pop(str(job_dir), None)
        if code:
            # LlamaFactory errors are written to logs/reference-train.log;
            # its parent traceback only reports a failed child. Inspect new
            # log bytes, without reviving OOMs from an earlier failed attempt.
            text = "\n".join(new_log_tail(path, old_log_sizes.get(str(path), 0))
                             for path in job_dir.rglob("*.log"))
            raise ChildFailure("Child exited " + str(code) + "; see " + str(log_path), oom=is_cuda_oom(text))

    def second_initialization(self, job):
        if job["vision_model"] != "qwen3vl8b":
            return None
        original = {**job, "vision_model": "omni"}
        marker = self.job_path(original) / "train.complete.json"
        trained = self.validate_marker(marker, original, "train")
        checkpoint = Path(trained.get("checkpoint", ""))
        if not checkpoint.is_absolute() or not checkpoint.is_dir():
            raise ValueError("8B initialization requires the corresponding native Omni checkpoint")
        return {"checkpoint": str(checkpoint), "checkpoint_sha256": trained["checkpoint_sha256"],
                "train_complete_sha256": digest(marker)}

    def command(self, action, job, directory, gpus):
        result = [sys.executable, "-m", "scripts.train_joint_affect", action,
            "--project-root", str(self.project), "--run-dir", str(directory),
            "--manifest", str(self.root / "train-manifest.json"), "--variant", job["variant"],
            "--fold", str(job["fold"]), "--grouping", job["grouping"], "--seed", str(job["seed"]),
            "--vision-model", job["vision_model"], "--max-frames", str(self.frames)]
        result += ["--gpus", "1,2"] if len(gpus) == 2 else ["--gpu", str(gpus[0])]
        initialization = self.second_initialization(job)
        if initialization:
            result += ["--init-second-from", initialization["checkpoint"]]
        return result

    def job(self, job, preflight=False, split="held"):
        directory = self.job_path(job, preflight)
        with self.mutex:
            directory.mkdir(parents=True, exist_ok=True)
            immutable(directory / "launch.json", {"pipeline_identity": self.plan["identity_hash"], "job": job,
                "second_initialization": self.second_initialization(job)})
        failure = directory / "pipeline-error.json"
        if failure.exists() and not self.resume_failed:
            previous = read(failure)
            raise ChildFailure("Preserved job failure requires explicit --resume-failed: " + str(directory), previous.get("cuda_oom", False))
        if (directory / "error.json").exists() and not self.resume_failed:
            raise ChildFailure("Child failure requires explicit --resume-failed: " + str(directory))
        with self.resources(job) as gpus:
            return self.allocated_job(job, directory, failure, gpus, preflight, split)

    def allocated_job(self, job, directory, failure, gpus, preflight, split):
        with self.mutex:
            self.active[str(directory.relative_to(self.root))] = {"gpus": list(gpus), "split": split, "preflight": preflight}
        try:
            action = "preflight" if preflight else "train"
            marker = directory / (action + ".complete.json")
            if not marker.exists():
                self.child(self.command(action, job, directory, gpus), directory, marker)
            trained = self.validate_marker(marker, job, action)
            if preflight or job["fold"] == "full":
                return {"directory": str(directory), "completion": trained, "complete_sha256": digest(marker)}
            output = directory / "predictions.jsonl"
            prediction_marker = directory / "predictions.complete.json"
            if not prediction_marker.exists():
                self.child(self.command("predict", job, directory, gpus) + ["--split", split, "--output", str(output)], directory, output)
            held = next(p["held"] for p in self.designs[job["grouping"]]["folds"] if p["fold"] == job["fold"])
            predictions = checked_predictions(output, [self.rows[i] for i in held])
            self.validate_prediction_marker(output, job, split, trained, len(predictions))
            immutable(prediction_marker, {"job": job, "train_complete_sha256": digest(marker),
                "predictions_sha256": digest(output), "rows": len(predictions)})
            return predictions
        except BaseException as exc:
            atomic(directory / ("pipeline-error-" + str(time.time_ns()) + ".json"),
                {"type": type(exc).__name__, "message": str(exc), "cuda_oom": getattr(exc, "oom", False)})
            atomic(failure, {"type": type(exc).__name__, "message": str(exc), "cuda_oom": getattr(exc, "oom", False)})
            raise
        finally:
            with self.mutex:
                self.active.pop(str(directory.relative_to(self.root)), None)

    def folds(self, variant, vision="omni", seed=42, grouping="content"):
        jobs = [self.spec(variant, part["fold"], grouping, seed, vision) for part in self.designs[grouping]["folds"]]
        predictions = [None] * len(self.rows)
        failure = None
        with ThreadPoolExecutor(max_workers=2) as executor:
            pending = {executor.submit(self.job, job): job for job in jobs}
            for future in as_completed(pending):
                job = pending[future]
                try:
                    values = future.result()
                    ix = next(p["held"] for p in self.designs[grouping]["folds"] if p["fold"] == job["fold"])
                    for i, value in zip(ix, values):
                        if predictions[i] is not None:
                            raise ValueError("Duplicate outer-fold prediction")
                        predictions[i] = value
                except BaseException as exc:
                    if failure is None or getattr(exc, "oom", False):
                        failure = exc
                    self.cancel.set()
                    self.stop_children()
            if failure:
                raise failure
        if any(p is None for p in predictions):
            raise ValueError("Missing outer-fold prediction")
        from scripts.joint_affect_data import metrics
        pooled = public_metrics(metrics(self.rows, predictions))
        report = {"variant": variant, "vision_model": vision, "seed": seed, "grouping": grouping,
                  "pooled": pooled, "independent_metric_audit": independent_audit(self.rows, predictions, pooled),
                  "folds": [{"fold": part["fold"], "metrics": public_metrics(metrics(
                      [self.rows[i] for i in part["held"]], [predictions[i] for i in part["held"]]))}
                      for part in self.designs[grouping]["folds"]]}
        immutable(self.root / ("cohort-" + str(self.frames)) / "metrics" / grouping / vision / variant / ("seed" + str(seed)) / "report.json", report)
        return predictions

    def cohort(self):
        from scripts.joint_affect_data import metrics
        root = self.root / ("cohort-" + str(self.frames))
        self.state("actual_32_example_preflights")
        for variant in ("C0", "C1"):
            self.job(self.spec(variant, 0, "content", 42, "omni"), preflight=True)
        self.state("content_sft_reference")
        reference = self.folds("SFT-ref")
        architectures = {}
        for variant in ("C0", "C1"):
            self.state("content_" + variant)
            architectures[variant] = self.folds(variant)
        # C0 is deliberately first; exact ties therefore select it.
        winner = "C1" if metrics(self.rows, architectures["C1"])["selection_score"] > metrics(self.rows, architectures["C0"])["selection_score"] + EPS else "C0"
        vision, chosen = "omni", architectures[winner]
        self.state("preflight_visual_8b", variant=winner)
        self.job(self.spec(winner, 0, "content", 42, "qwen3vl8b"), preflight=True)
        self.state("content_visual_8b", variant=winner)
        bigger = self.folds(winner, "qwen3vl8b")
        old_score, big_score = (metrics(self.rows, p)["selection_score"] for p in (chosen, bigger))
        if big_score > old_score + EPS:
            vision, chosen = "qwen3vl8b", bigger
        selection = {"variant": winner, "vision_model": vision, "max_frames": self.frames,
            "omni_score": old_score, "qwen3vl8b_score": big_score, "official_inputs_used": False}
        immutable(root / "winner.json", selection)
        primary = compare(self.rows, reference, chosen, self.designs["content"])
        immutable(root / "primary-report.json", primary)
        if not primary["passed"]:
            return {"status": "complete", "reason": "primary_gate_not_met", "winner": selection, "primary": primary, "final_model_trained": False}
        self.state("parent_pressure_reference")
        pressure_ref = self.folds("SFT-ref", grouping="parent")
        if vision == "qwen3vl8b":
            self.state("parent_pressure_native_second_initialization", **selection)
            self.folds(winner, "omni", grouping="parent")
        self.state("parent_pressure_winner", **selection)
        pressure_new = self.folds(winner, vision, grouping="parent")
        pressure = compare(self.rows, pressure_ref, pressure_new, self.designs["parent"])
        pressure["uncertainty_gain"] = pressure["candidate"]["uncertainty_component"] - pressure["control"]["uncertainty_component"]
        pressure["pressure_passed"] = pressure["uncertainty_gain"] >= .005 - EPS
        immutable(root / "pressure-report.json", pressure)
        if not pressure["pressure_passed"]:
            return {"status": "complete", "reason": "parent_pressure_gate_not_met", "winner": selection,
                    "primary": primary, "pressure": pressure, "final_model_trained": False}
        self.state("seed43_reference")
        second_ref = self.folds("SFT-ref", seed=43)
        if vision == "qwen3vl8b":
            self.state("seed43_native_second_initialization", **selection)
            self.folds(winner, "omni", seed=43)
        self.state("seed43_winner", **selection)
        second_new = self.folds(winner, vision, seed=43)
        replication = compare(self.rows, second_ref, second_new, self.designs["content"])
        immutable(root / "replication-report.json", replication)
        if not replication["passed"]:
            return {"status": "complete", "reason": "replication_gate_not_met", "winner": selection,
                    "primary": primary, "pressure": pressure, "replication": replication, "final_model_trained": False}
        self.state("full_training_two_seeds", **selection)
        if vision == "qwen3vl8b":
            for seed in (42, 43):
                self.job(self.spec(winner, "full", "content", seed, "omni"))
        full = [self.job(self.spec(winner, "full", "content", seed, vision)) for seed in (42, 43)]
        lock = {"algorithm": VERSION, "pipeline_identity": self.plan["identity_hash"], "winner": selection,
                "models": full, "primary_passed": True, "pressure_passed": True,
                "replication_passed": True, "official_inputs_used_for_selection": False}
        immutable(root / "final-model-lock.json", lock)
        for item in full:
            immutable(Path(item["directory"]) / "final_model_lock.json", lock)
        reports = self.official(full, selection, root)
        return {"status": "complete", "reason": "all_gates_and_locked_final_inference_completed", "winner": selection,
                "primary": primary, "pressure": pressure, "replication": replication,
                "final_model_trained": True, "official": reports}

    def official(self, full, selection, root):
        from scripts.joint_affect_data import canonical_inference, metrics
        if not (root / "final-model-lock.json").is_file():
            raise ValueError("Official inference requires the locked final ensemble")
        result = {}
        for split, filename, count in OFFICIAL_SPLITS:
            self.state("locked_official_" + split, **selection)
            path = self.project / "Dataset" / filename
            content = path.read_bytes()
            source_digest = hashlib.sha256(content).hexdigest()
            raw = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
            if len(raw) != count or len({r["sample_id"] for r in raw}) != count:
                raise ValueError("Official split coverage changed")
            # Test views/labels are never accessed, even if such fields exist.
            canonical = [canonical_inference(r) for r in raw]
            safe = [{"sample_id": r["sample_id"], "protocol": r["protocol"]} for r in canonical]
            immutable(root / ("official-" + split + "-input.json"), {"sha256": source_digest, "ids": safe})
            predictions = []
            for seed, item in zip((42, 43), full):
                directory = Path(item["directory"])
                output = directory / ("predictions-" + split + ".jsonl")
                prediction_marker = output.with_suffix(".complete.json")
                job = self.spec(selection["variant"], "full", "content", seed, selection["vision_model"])
                if not prediction_marker.exists():
                    with self.resources(job) as gpus:
                        self.child(self.command("predict", job, directory, gpus) + ["--split", split, "--output", str(output)], directory, output)
                verified = checked_predictions(output, safe)
                trained = self.validate_marker(directory / "train.complete.json", job, "train")
                if digest(directory / "train.complete.json") != item["complete_sha256"]:
                    raise ValueError("Training marker changed after final ensemble lock")
                self.validate_prediction_marker(output, job, split, trained, count)
                immutable(prediction_marker, {"split": split, "source_sha256": source_digest,
                    "final_model_lock_sha256": digest(root / "final-model-lock.json"),
                    "train_complete_sha256": item["complete_sha256"],
                    "predictions_sha256": digest(output), "rows": count})
                predictions.append(verified)
            if digest(path) != source_digest:
                raise ValueError("Official input changed during locked inference")
            mean = ensemble(*predictions)
            output = root / ("locked-ensemble-" + split + ".json")
            immutable(output, mean)
            report = {"rows": count, "input_sha256": source_digest, "ensemble_sha256": digest(output),
                      "used_for_selection": False, "submitted_to_kaggle": False}
            if split == "val":
                report["metrics"] = public_metrics(metrics(canonical, mean))
            else:
                candidate = root / "candidate.csv"
                content = candidate_csv(safe, mean)
                if candidate.exists() and candidate.read_bytes() != content:
                    raise ValueError("Locked candidate changed")
                if not candidate.exists():
                    with candidate.open("xb") as stream:
                        stream.write(content)
                report.update(candidate=str(candidate), candidate_sha256=digest(candidate))
            immutable(root / ("official-" + split + "-report.json"), report)
            result[split] = report
        return result

    def run(self):
        with run_lock(self.root):
            if (self.root / "COMPLETE.json").exists():
                complete = read(self.root / "COMPLETE.json")
                if digest(self.root / "FINAL_REPORT.json") != complete["report_sha256"] or digest(self.root / "review-results.tar.gz") != complete["archive_sha256"]:
                    raise ValueError("Completed delivery checksum changed")
                return read(self.root / "FINAL_REPORT.json")
            failures = self.root / "cohort-failures"
            if (failures / "frames64.json").exists():
                previous = read(failures / "frames64.json")
                if not previous.get("cuda_oom"):
                    raise RuntimeError("Preserved non-OOM cohort failure requires a new reviewed recovery")
                self.frames = 32
            while True:
                try:
                    self.cancel.clear()
                    self.check_sources()
                    if self.budget is None:
                        self.budget = Budget(self.root)
                    # A detached child may have outlived a killed controller.
                    # Wait for its exact PID/start identity; never launch a duplicate.
                    for path in self.root.glob("cohort-*/**/process.json"):
                        previous = read(path)
                        token = previous.get("process_start_ticks")
                        while token is not None and process_token(previous["pid"]) == token:
                            self.state("waiting_for_retained_child", child=str(path.parent))
                            self.budget.tick()
                            time.sleep(3)
                    result = self.cohort()
                    break
                except BaseException as exc:
                    failed = {"algorithm": VERSION, "max_frames": self.frames, "type": type(exc).__name__,
                        "message": str(exc), "cuda_oom": getattr(exc, "oom", False), "traceback": traceback.format_exc()}
                    immutable(failures / ("frames" + str(self.frames) + ".json"), failed)
                    if self.frames == 64 and getattr(exc, "oom", False):
                        self.frames = 32
                        self.state("whole_cohort_restart_after_cuda_oom", prior_max_frames=64)
                        continue
                    result = {"status": "failed", "reason": str(exc), "failure": failed, "final_model_trained": False}
                    break
            result.update(algorithm=VERSION, pipeline_identity=self.plan["identity_hash"], max_frames=self.frames,
                          submitted_to_kaggle=False, new_public_score=None)
            immutable(self.root / "FINAL_REPORT.json", result)
            self.state(result["status"], reason=result["reason"])
            self.archive()
            return result

    def archive(self):
        files = []
        # Deliberate allowlist: no tensors, media, neural features, or raw data.
        for pattern in ("plan.json", "clock.json", "status.json", "FINAL_REPORT.json", "cohort-failures/*.json",
            "cohort-*/*report.json", "cohort-*/winner.json", "cohort-*/final-model-lock.json",
            "cohort-*/candidate.csv", "cohort-*/official-*-report.json", "cohort-*/locked-ensemble-*.json",
            "cohort-*/**/launch.json", "cohort-*/**/train.complete.json", "cohort-*/**/preflight.complete.json",
            "cohort-*/**/pipeline-error*.json", "cohort-*/**/predictions*.complete.json",
            "cohort-*/**/trainable-parameters.json", "cohort-*/**/identity.json",
            "cohort-*/metrics/**/report.json", "cohort-*/**/predictions*.jsonl"):
            files.extend(p for p in self.root.glob(pattern) if p.is_file() and not p.is_symlink())
        files = sorted(set(files))
        listing = {p.relative_to(self.root).as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size} for p in files}
        immutable(self.root / "delivery-manifest.json", listing)
        archive = self.root / "review-results.tar.gz"
        if archive.exists():
            raise ValueError("Unpublished archive already exists; inspect before recovery")
        partial = archive.with_name(archive.name + ".partial")
        with tarfile.open(partial, "x:gz") as tar:
            tar.add(self.root / "delivery-manifest.json", arcname="delivery-manifest.json", recursive=False)
            for path in files:
                tar.add(path, arcname=path.relative_to(self.root).as_posix(), recursive=False)
        partial.rename(archive)
        immutable(self.root / "COMPLETE.json", {"algorithm": VERSION, "delivery_complete": True,
            "outcome": read(self.root / "FINAL_REPORT.json")["status"], "report_sha256": digest(self.root / "FINAL_REPORT.json"),
            "archive_sha256": digest(archive), "model_weights_retained_on_server": True, "submitted_to_kaggle": False})


def progress_snapshot(root):
    """Read-only state; the time budget is not a forecast of task completion."""
    root = Path(root).resolve()
    status = read(root / "status.json") if (root / "status.json").exists() else {}
    clock = read(root / "clock.json") if (root / "clock.json").exists() else {}
    children = {}
    for name in status.get("active_jobs", {}):
        directory = (root / name).resolve()
        if not directory.is_relative_to(root):
            raise ValueError("Progress job path escaped the batch directory")
        path = directory / "status.json"
        if path.exists():
            children[name] = read(path)
    if clock:
        clock = {**clock, "budget_seconds_available_now": max(0., min(clock["remaining_seconds"], clock["deadline_unix"] - time.time())),
                 "elapsed_wall_seconds": max(0., time.time() - clock["started_unix"]),
                 "remaining_is_budget_not_training_eta": True}
    return {"status": status, "clock": clock, "children": children,
            "complete": read(root / "COMPLETE.json") if (root / "COMPLETE.json").exists() else {}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "progress"))
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume-failed", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        value = prepare(args.project_root, args.run_dir)
    elif args.command == "progress":
        value = progress_snapshot(args.run_dir)
    else:
        value = Pipeline(args.project_root, args.run_dir, resume_failed=args.resume_failed).run()
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    return value


if __name__ == "__main__":
    main()
