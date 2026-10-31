import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.model_artifacts import dependency_order, digest, extract_verified, verify_directory


class ModelArtifactsTests(unittest.TestCase):
    def package(self, root, names=None):
        content = {"artifact.json": json.dumps({"id": "test-model"}).encode(), "adapter/weights.bin": b"weights"}
        if names:
            content.update(names)
        archive = root / "test-model.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            for name, value in content.items():
                bundle.writestr(name, value)
        entry = {"id": "test-model", "archive": {"bytes": archive.stat().st_size, "sha256": digest(archive)},
                 "files": {name: {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()} for name, value in content.items()}}
        return archive, entry

    def test_round_trip_and_existing_model_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, entry = self.package(root)
            target = root / "models/test-model"
            self.assertEqual(extract_verified(archive, target, entry), target)
            self.assertEqual(extract_verified(archive, target, entry), target)
            (target / "adapter/weights.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_directory(target, entry)

    def test_archive_hash_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, entry = self.package(root)
            entry["archive"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "Archive checksum"):
                extract_verified(archive, root / "model", entry)
            self.assertFalse((root / "model").exists())

    def test_traversal_is_rejected_even_if_listed_in_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, entry = self.package(root, {"../escape.bin": b"bad"})
            with self.assertRaisesRegex(ValueError, "Unsafe archive"):
                extract_verified(archive, root / "model", entry)
            self.assertFalse((root / "escape.bin").exists())

    def test_unexpected_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, entry = self.package(root)
            del entry["files"]["adapter/weights.bin"]
            with self.assertRaisesRegex(ValueError, "Unexpected archive"):
                extract_verified(archive, root / "model", entry)

    def test_dependencies_are_ordered_and_cycles_rejected(self):
        entries = {"base": {"dependencies": []}, "policy": {"dependencies": ["base"]}}
        self.assertEqual(dependency_order(entries, "policy"), ["base", "policy"])
        entries["base"]["dependencies"] = ["policy"]
        with self.assertRaisesRegex(ValueError, "Cyclic"):
            dependency_order(entries, "policy")


if __name__ == "__main__":
    unittest.main()
