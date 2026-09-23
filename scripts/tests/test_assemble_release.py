"""Publication rejects tampered wheels and unrelated or incomplete evidence."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.assemble_release import assemble


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.artifacts = self.root / "artifacts"
        self.output = self.root / "publication"
        for system, machine, platform in (
            ("Linux", "x86_64", "manylinux_2_34_x86_64"),
            ("Darwin", "arm64", "macosx_11_0_arm64"),
        ):
            directory = self.artifacts / system
            core = self.wheel(directory / "build-1", f"qwen_mm-0.1.0-cp311-abi3-{platform}.whl")
            plugin = self.wheel(directory / "plugin-build-1", "qwen_mm_vllm-0.1.0-py3-none-any.whl")
            data = {
                "schema": "qwen-mm-release-v1",
                "host": {"system": system, "machine": machine},
                "status": "passed",
                "source": {"commit": "expected"},
                "version": "0.1.0",
                "steps": [{"exit_code": 0}],
                "artifact": core,
                "plugin_artifact": plugin,
            }
            (directory / "manifest.json").write_text(json.dumps(data))
            if system == "Linux":
                check = dict(data)
                del check["schema"]
                check["core_artifact"] = check.pop("artifact")
                check_dir = self.artifacts / "vllm"
                check_dir.mkdir()
                (check_dir / "manifest.json").write_text(json.dumps(check))

    @staticmethod
    def wheel(directory, name):
        directory.mkdir(parents=True)
        contents = name.encode()
        (directory / name).write_bytes(contents)
        return {
            "name": name,
            "sha256": hashlib.sha256(contents).hexdigest(),
            "bytes": len(contents),
        }

    def run_assembly(self):
        assemble(self.artifacts, self.output, commit="expected", version="0.1.0", tag="v0.1.0")

    def change_manifest(self, target, key, value):
        path = self.artifacts / target / "manifest.json"
        data = json.loads(path.read_text())
        data[key] = value
        path.write_text(json.dumps(data))

    def test_selects_two_native_wheels_and_one_plugin(self):
        self.run_assembly()
        self.assertEqual(len(list((self.output / "dist").glob("*.whl"))), 3)
        self.assertEqual(len((self.output / "SHA256SUMS").read_text().splitlines()), 3)

    def test_rejects_tampered_wheel(self):
        next((self.artifacts / "Linux" / "build-1").glob("*.whl")).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.run_assembly()

    def test_rejects_stale_source(self):
        self.change_manifest("Darwin", "source", {"commit": "old"})
        with self.assertRaisesRegex(ValueError, "stale"):
            self.run_assembly()

    def test_rejects_missing_platform(self):
        (self.artifacts / "Darwin" / "manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "both native"):
            self.run_assembly()

    def test_rejects_unrelated_vllm_check(self):
        self.change_manifest("vllm", "core_artifact", {"name": "another.whl"})
        with self.assertRaisesRegex(ValueError, "not bound"):
            self.run_assembly()

    def test_rejects_failed_verification(self):
        self.change_manifest("Linux", "steps", [{"exit_code": 1}])
        with self.assertRaisesRegex(ValueError, "failed or missing"):
            self.run_assembly()
