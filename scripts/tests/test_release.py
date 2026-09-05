"""Release provenance and wheel metadata must fail closed."""

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts import release


class ReleaseTests(unittest.TestCase):
    def test_source_refuses_untracked_files(self):
        with (
            patch.object(release.subprocess, "run"),
            patch.object(release, "capture", return_value="uncommitted.py"),
            self.assertRaisesRegex(RuntimeError, "uncommitted.py"),
        ):
            release.source_identity()

    def test_source_requires_clean_shipping_tree(self):
        with (
            patch.object(release.subprocess, "run") as run,
            patch.object(release, "capture", side_effect=["", "commit", "tree", "123"]),
        ):
            self.assertEqual(release.source_identity()["commit"], "commit")
            self.assertIn(":!.kd", run.call_args.args[0])
            self.assertTrue(run.call_args.kwargs["check"])

    def test_wheel_metadata_and_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "qwen_mm-0.1.0-cp311-abi3-manylinux_2_34_x86_64.whl"
            self.write_wheel(path)
            self.assertEqual(release.inspect_wheel(path)["sha256"], release.sha256(path))
            self.write_wheel(path, version="0.2.0")
            with self.assertRaises(AssertionError):
                release.inspect_wheel(path)

    def test_unpublishable_linux_tag_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "qwen_mm-0.1.0-cp311-abi3-linux_x86_64.whl"
            self.write_wheel(path)
            with self.assertRaises(AssertionError):
                release.inspect_wheel(path)

    @staticmethod
    def write_wheel(path, version="0.1.0"):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                "qwen_mm-0.1.0.dist-info/METADATA",
                f"Name: qwen-mm\nVersion: {version}\nRequires-Python: >=3.11,<3.12\n"
                "License-Expression: Apache-2.0\n",
            )
            archive.writestr("qwen_mm-0.1.0.dist-info/licenses/LICENSE", "license")
            for source in (release.ROOT / "crates/qwen-mm-python/python/qwen_mm").glob("*.py"):
                archive.write(source, f"qwen_mm/{source.name}")
