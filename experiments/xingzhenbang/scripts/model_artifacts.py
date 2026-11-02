"""List, download and verify the versioned experiment models in reports/models/registry.json.

Downloading never executes code from an archive or starts GPU inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "reports/models/registry.json"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_registry(path=DEFAULT_REGISTRY):
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise ValueError("Unsupported model registry version")
    seen = set()
    for entry in registry["artifacts"]:
        mid = entry["id"]
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", mid) or mid in seen:
            raise ValueError("Invalid or duplicate artifact ID")
        seen.add(mid)
        if entry["archive"]["filename"] != mid + ".zip":
            raise ValueError("Archive filename differs from artifact ID")
        if not re.fullmatch(r"[0-9a-f]{64}", entry["archive"]["sha256"]):
            raise ValueError("Missing archive checksum")
    for entry in registry["artifacts"]:
        if not set(entry.get("dependencies", [])).issubset(seen):
            raise ValueError("Unknown model dependency")
    return registry


def verify_directory(directory, entry):
    directory = Path(directory).resolve()
    for name, expected in entry["files"].items():
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(directory) or not path.is_file():
            raise ValueError("Missing or escaped model file: " + name)
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError("Model file checksum differs: " + name)
    metadata = json.loads((directory / "artifact.json").read_text(encoding="utf-8"))
    if metadata["id"] != entry["id"]:
        raise ValueError("Model metadata ID differs")
    return directory


def extract_verified(archive, destination, entry):
    """Extract only the explicitly listed files into a new, atomically published directory."""
    archive, destination = Path(archive), Path(destination).resolve()
    if archive.stat().st_size != entry["archive"]["bytes"] or digest(archive) != entry["archive"]["sha256"]:
        raise ValueError("Archive checksum differs; nothing extracted")
    if destination.exists():
        return verify_directory(destination, entry)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        names = []
        for info in bundle.infolist():
            name = info.filename
            parts = PurePosixPath(name)
            if (not name or "\\" in name or ":" in name or parts.is_absolute()
                    or any(part in ("..", ".") for part in parts.parts)
                    or name.endswith("/") or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError("Unsafe archive member: " + name)
            if name not in entry["files"] or info.file_size != entry["files"][name]["bytes"]:
                raise ValueError("Unexpected archive member: " + name)
            names.append(name)
        if len(names) != len(set(names)) or set(names) != set(entry["files"]):
            raise ValueError("Archive file list differs from registry")
        stage = Path(tempfile.mkdtemp(prefix=destination.name + ".partial-", dir=destination.parent))
        # A failed stage remains for inspection. Existing model files are never deleted.
        for info in bundle.infolist():
            path = stage / info.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, path.open("xb") as target:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    target.write(block)
    verify_directory(stage, entry)
    stage.rename(destination)
    return destination


def dependency_order(entries, selected, visiting=None, done=None):
    visiting = set() if visiting is None else visiting
    done = [] if done is None else done
    if selected in done:
        return done
    if selected in visiting:
        raise ValueError("Cyclic model dependencies")
    visiting.add(selected)
    for dep in entries[selected].get("dependencies", []):
        dependency_order(entries, dep, visiting, done)
    visiting.remove(selected)
    done.append(selected)
    return done


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    download = sub.add_parser("download")
    download.add_argument("model_id")
    download.add_argument("--output-dir", type=Path, default=ROOT / "models/experiments")
    download.add_argument("--revision", help="Optional explicit HF commit SHA; defaults to the pinned registry revision")
    download.add_argument("--archive-dir", type=Path, help="Use locally downloaded ZIPs; no network or login needed")
    verify = sub.add_parser("verify")
    verify.add_argument("model_id")
    verify.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    registry = read_registry(args.registry)
    entries = {entry["id"]: entry for entry in registry["artifacts"]}
    if args.command == "list":
        for entry in entries.values():
            print(json.dumps({key: entry[key] for key in ("id", "method", "kind", "evaluation", "promotion")}, ensure_ascii=False))
        return 0
    if args.model_id not in entries:
        parser.error("Unknown model ID; run list to see available versions")
    if args.command == "verify":
        print(verify_directory(args.directory, entries[args.model_id]))
        return 0
    revision = args.revision or registry.get("revision")
    if not args.archive_dir and not revision:
        parser.error("This registry has no published HF revision yet")
    if revision and not re.fullmatch(r"[0-9a-f]{40}", revision):
        parser.error("Use a full HF commit SHA to keep downloads reproducible")
    for mid in dependency_order(entries, args.model_id):
        entry = entries[mid]
        target = args.output_dir / mid
        if target.exists():
            path = verify_directory(target, entry)
        else:
            if args.archive_dir:
                archive = args.archive_dir / entry["archive"]["filename"]
            else:
                from huggingface_hub import hf_hub_download
                archive = Path(hf_hub_download(repo_id=registry["repo_id"], filename=entry["archive"]["filename"], revision=revision))
            path = extract_verified(archive, target, entry)
        print(json.dumps({"id": mid, "verified_directory": str(path.resolve()), "kind": entry["kind"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
