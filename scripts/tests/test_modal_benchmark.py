from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

SUPPORT_PATH = Path(__file__).resolve().parents[1] / "modal_benchmark_support.py"
SPEC = importlib.util.spec_from_file_location("modal_benchmark_support", SUPPORT_PATH)
assert SPEC is not None and SPEC.loader is not None
support = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = support
SPEC.loader.exec_module(support)


def _benchmark() -> dict[str, object]:
    return {
        "schema_id": support.BENCHMARK_SCHEMA_ID,
        "schema_version": support.BENCHMARK_SCHEMA_VERSION,
        "architecture_family": "x86_64",
        "protocol": {"self_test_only": True},
        "release_eligibility": {"releasable": False},
    }


def _provenance() -> dict[str, object]:
    return {
        "base_image_reference": "python@sha256:" + "a" * 64,
        "base_image_tag": "python:test",
        "build_system_packages": ["build-essential=test"],
        "cargo_version": "cargo test",
        "cgroup_limits": {},
        "commands": ["test"],
        "completed_at": "2026-08-03T00:00:01+00:00",
        "schema_id": support.PROVENANCE_SCHEMA_ID,
        "schema_version": support.PROVENANCE_SCHEMA_VERSION,
        "execution": "modal_native_linux_x86_diagnostic",
        "execution_environment": {"UV_DEFAULT_INDEX": "https://pypi.org/simple"},
        "dedicated_host": False,
        "performance_claim": False,
        "system": "Linux",
        "machine": "x86_64",
        "cpu_description": "Intel Xeon test CPU",
        "logical_cpu_count": 4,
        "lscpu": {},
        "modal_client_version": "test",
        "modal_environment": {},
        "native_module_file": "ELF 64-bit shared object, x86-64",
        "platform": "Linux-test",
        "proc_meminfo_total_bytes": 1024,
        "python_version": "3.11.15",
        "python_packages": ["numpy==2.4.6"],
        "requested_cpu_physical_cores": 4,
        "requested_memory_mib": 8192,
        "rustc_version": "rustc test",
        "source_bytes": 1,
        "source_digest": "b" * 64,
        "source_dirty_state": "",
        "source_file_count": 1,
        "source_revision": "test",
        "started_at": "2026-08-03T00:00:00+00:00",
        "uname": "Linux test x86_64",
        "uv_version": "uv test",
        "visible_cpu_affinity": [0, 1, 2, 3],
        "wheel_name": "qwen_mm-test.whl",
        "wheel_sha256": support.sha256_bytes(b"wheel"),
    }


def _files() -> dict[str, bytes]:
    return {
        "benchmark/report.md": b"# qwen-mm paired benchmark v2\n",
        "benchmark/result.json": (json.dumps(_benchmark()) + "\n").encode(),
        "logs/benchmark.log": b"benchmark passed\n",
        "logs/build.log": b"build passed\n",
        "logs/validation.log": b"validation passed\n",
        "provenance.json": (json.dumps(_provenance()) + "\n").encode(),
        "build/qwen_mm-test.whl": b"wheel",
    }


class ModalBenchmarkSupportTests(unittest.TestCase):
    def test_source_digest_is_stable_and_ignores_build_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src/lib.rs").write_text("one", encoding="utf-8")
            first = support.source_tree_digest(root)
            (root / "target").mkdir()
            (root / "target/noise").write_text("ignored", encoding="utf-8")
            native = root / "package/_native.abi3.so"
            native.parent.mkdir()
            native.write_bytes(b"generated extension")
            (root / "reference/.venv").mkdir(parents=True)
            (root / "reference/.venv/secret").write_text("ignored", encoding="utf-8")
            (root / "nested/.ruff_cache").mkdir(parents=True)
            (root / "nested/.ruff_cache/cache").write_text("ignored", encoding="utf-8")
            self.assertEqual(first, support.source_tree_digest(root))
            (root / "src/lib.rs").write_text("two", encoding="utf-8")
            self.assertNotEqual(first, support.source_tree_digest(root))

    def test_committed_source_digest_ignores_gitignored_worktree_noise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
            (root / ".gitignore").write_text(
                ".DS_Store\nreference/results/conformance-local/\n", encoding="utf-8"
            )
            (root / "src").mkdir()
            (root / "src/lib.rs").write_text("recorded\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Digest Test",
                    "-c",
                    "user.email=digest@example.invalid",
                    "commit",
                    "--quiet",
                    "-m",
                    "recorded",
                ],
                cwd=root,
                check=True,
            )
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            recorded = support.committed_source_tree_digest(root, revision)
            self.assertEqual(recorded, support.source_tree_digest(root))

            (root / ".DS_Store").write_bytes(b"ignored metadata")
            generated = root / "reference/results/conformance-local/result.json"
            generated.parent.mkdir(parents=True)
            generated.write_text("{}\n", encoding="utf-8")
            status = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            self.assertEqual(status, "")
            self.assertEqual(support.committed_source_tree_digest(root, revision), recorded)
            self.assertNotEqual(support.source_tree_digest(root), recorded)
            with self.assertRaisesRegex(support.ModalBenchmarkArtifactError, "missing"):
                support.committed_source_tree_digest(root, "0" * 40)

    def test_native_architecture_validation_rejects_wrong_or_emulated_hosts(self) -> None:
        support.assert_native_linux_x86(
            system="Linux", machine="x86_64", cpu_description="AMD EPYC"
        )
        with self.assertRaisesRegex(support.ModalBenchmarkArtifactError, "Linux"):
            support.assert_native_linux_x86(
                system="Darwin", machine="x86_64", cpu_description="Intel"
            )
        with self.assertRaisesRegex(support.ModalBenchmarkArtifactError, "x86_64"):
            support.assert_native_linux_x86(
                system="Linux", machine="aarch64", cpu_description="Neoverse"
            )
        with self.assertRaisesRegex(support.ModalBenchmarkArtifactError, "emulation"):
            support.assert_native_linux_x86(
                system="Linux", machine="x86_64", cpu_description="QEMU Virtual CPU"
            )

    def test_integrity_checked_artifact_round_trip_and_atomic_write(self) -> None:
        artifact = support.create_artifact(_files())
        validated = support.validate_artifact(artifact)
        self.assertEqual(validated["provenance"]["machine"], "x86_64")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact.zip"
            support.write_artifact_atomically(artifact, output)
            self.assertEqual(output.read_bytes(), artifact)

    def test_tampered_or_unsafe_artifacts_fail_closed(self) -> None:
        artifact = support.create_artifact(_files())
        with zipfile.ZipFile(BytesIO(artifact)) as source:
            members = {name: source.read(name) for name in source.namelist()}
        members["benchmark/result.json"] = b"{}\n"
        tampered = BytesIO()
        with zipfile.ZipFile(tampered, "w") as archive:
            for name, value in members.items():
                archive.writestr(name, value)
        with self.assertRaisesRegex(support.ModalBenchmarkArtifactError, "mismatch"):
            support.validate_artifact(tampered.getvalue())

        unsafe = BytesIO()
        with zipfile.ZipFile(unsafe, "w") as archive:
            archive.writestr("../escape", b"bad")
        with self.assertRaisesRegex(support.ModalBenchmarkArtifactError, "unsafe"):
            support.validate_artifact(unsafe.getvalue())


if __name__ == "__main__":
    unittest.main()
