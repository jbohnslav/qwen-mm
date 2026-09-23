"""Release provenance and wheel metadata must fail closed."""

import subprocess
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

    def test_incompatible_old_numpy_pin_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "qwen_mm-0.1.0-cp311-abi3-manylinux_2_34_x86_64.whl"
            self.write_wheel(path, numpy="numpy==2.4.6")
            with self.assertRaises(AssertionError):
                release.inspect_wheel(path)

    def test_publish_rehearsal_works_without_an_index_server(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "qwen_mm-0.1.0-cp311-abi3-manylinux_2_34_x86_64.whl"
            self.write_wheel(path)
            result = subprocess.run(
                release.publish_dry_run_command(path), capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_plugin_wheel_requires_reviewed_entry_points_and_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "qwen_mm_vllm-0.1.0-py3-none-any.whl"
            for entry in ("register_native", "wrong_entry"):
                with zipfile.ZipFile(path, "w") as archive:
                    prefix = "qwen_mm_vllm-0.1.0.dist-info/"
                    archive.writestr(
                        prefix + "METADATA",
                        "Metadata-Version: 2.4\nName: qwen-mm-vllm\nVersion: 0.1.0\n"
                        "Requires-Python: >=3.11,<3.12\nLicense-Expression: Apache-2.0\n"
                        "Requires-Dist: qwen-mm==0.1.0\nRequires-Dist: vllm==0.23.0\n"
                        "Requires-Dist: transformers==5.14.1\n",
                    )
                    archive.write(release.ROOT / "LICENSE", prefix + "licenses/LICENSE")
                    archive.writestr(
                        prefix + "entry_points.txt",
                        "[vllm.general_plugins]\n"
                        f"qwen_mm_native_images = qwen_mm_vllm.native:{entry}\n"
                        "qwen_mm_serving_audit = qwen_mm_vllm.audit:register_audit\n",
                    )
                    for source in (release.ROOT / "integrations/vllm/qwen_mm_vllm").glob("*.py"):
                        archive.write(source, f"qwen_mm_vllm/{source.name}")
                if entry == "register_native":
                    self.assertEqual(
                        release.inspect_plugin_wheel(path)["sha256"], release.sha256(path)
                    )
                else:
                    with self.assertRaises(AssertionError):
                        release.inspect_plugin_wheel(path)

    @staticmethod
    def write_wheel(path, version="0.1.0", numpy="numpy>=2.3.5,<3"):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                "qwen_mm-0.1.0.dist-info/METADATA",
                f"Metadata-Version: 2.4\nName: qwen-mm\nVersion: {version}\nRequires-Python: >=3.11, <3.12\n"
                "License-Expression: Apache-2.0\n"
                f"Requires-Dist: {numpy}\nRequires-Dist: huggingface-hub==1.26.0\n",
            )
            for document in ("LICENSE", "NOTICE", "THIRD_PARTY_LICENSES.txt"):
                archive.write(
                    release.ROOT / document, f"qwen_mm-0.1.0.dist-info/licenses/{document}"
                )
            for source in (release.ROOT / "crates/qwen-mm-python/python/qwen_mm").glob("*.py"):
                archive.write(source, f"qwen_mm/{source.name}")
