from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

SUPPORT_PATH = Path(__file__).resolve().parents[1] / "profile_capture_support.py"
SPEC = importlib.util.spec_from_file_location("profile_capture_support", SUPPORT_PATH)
assert SPEC is not None and SPEC.loader is not None
support = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = support
SPEC.loader.exec_module(support)

LOCAL_RUNNER_PATH = Path(__file__).resolve().parents[1] / "local_profile.py"
LOCAL_SPEC = importlib.util.spec_from_file_location("local_profile_test_module", LOCAL_RUNNER_PATH)
assert LOCAL_SPEC is not None and LOCAL_SPEC.loader is not None
local_runner = importlib.util.module_from_spec(LOCAL_SPEC)
LOCAL_SPEC.loader.exec_module(local_runner)


def _required_files() -> dict[str, bytes]:
    return {name: f"evidence:{name}\n".encode() for name in support.REQUIRED_ARTIFACT_FILES}


def _provenance() -> dict[str, object]:
    return {
        "schema_id": support.PROVENANCE_SCHEMA_ID,
        "schema_version": support.PROVENANCE_SCHEMA_VERSION,
        "execution": "modal_native_linux_x86_profile_v1",
        "source": {
            "revision": "a" * 40,
            "tree_sha256": "b" * 64,
            "clean": True,
            "git_status": "",
        },
        "host": {
            "system": "Linux",
            "machine": "x86_64",
            "cpu_description": "AMD EPYC test CPU",
        },
        "assets": {"tree_sha256": "c" * 64, "logical_bytes": 34_000_000},
        "protocol": {
            "profiles": list(support.PROFILE_ALIASES),
            "cases": list(support.PROFILE_CASES),
            "thread_budgets": list(support.THREAD_BUDGETS),
            "build_label": support.PROFILE_BUILD_LABEL,
            "observed_coordinates": 24,
            "observation_repetitions": 3,
            "sampler": {
                "name": "py-spy",
                "version": support.PY_SPY_VERSION,
                "native": True,
                "rate_hz": support.PY_SPY_RATE_HZ,
                "duration_seconds": support.PY_SPY_DURATION_SECONDS,
            },
        },
        "commands": ["maturin build --release --locked"],
    }


class PathAndDirectoryIdentityTests(unittest.TestCase):
    def test_local_runner_uses_clean_committed_identity_and_ignores_gitignored_noise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
            (root / ".gitignore").write_text("target/\n.DS_Store\n", encoding="utf-8")
            tracked = root / "source.txt"
            tracked.write_text("committed\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Profile Test",
                    "-c",
                    "user.email=profile@example.invalid",
                    "commit",
                    "--quiet",
                    "-m",
                    "source",
                ],
                cwd=root,
                check=True,
            )
            first = local_runner.clean_committed_source_identity(root)
            (root / "target").mkdir()
            (root / "target/noise").write_text("ignored\n", encoding="utf-8")
            (root / ".DS_Store").write_text("ignored\n", encoding="utf-8")
            self.assertEqual(local_runner.clean_committed_source_identity(root), first)
            tracked.write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "clean implementation commit"):
                local_runner.clean_committed_source_identity(root)

    def test_safe_relative_path_rejects_archive_escape_and_non_posix_paths(self) -> None:
        self.assertEqual(
            support.safe_relative_path("target/profile/result.json").as_posix(),
            "target/profile/result.json",
        )
        for value in ("", "/absolute", "../escape", "a/../../escape", "windows\\path"):
            with self.subTest(value=value), self.assertRaises(support.ProfileCaptureArtifactError):
                support.safe_relative_path(value)

    def test_directory_identity_accepts_content_equivalent_symlink_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "blobs").mkdir()
            blob = root / "blobs/pinned"
            blob.write_bytes(b"hash-pinned")
            (root / "snapshot").mkdir()
            (root / "snapshot/config.json").symlink_to("../blobs/pinned")
            first = support.directory_identity(root)
            self.assertEqual(first["entry_count"], 2)
            self.assertEqual(first["logical_bytes"], 2 * len(b"hash-pinned"))
            support.assert_directory_identity(root, first)
            (root / "snapshot/config.json").unlink()
            (root / "snapshot/config.json").write_bytes(b"hash-pinned")
            self.assertEqual(support.directory_identity(root), first)
            blob.write_bytes(b"changed")
            with self.assertRaisesRegex(
                support.ProfileCaptureArtifactError, "hash-pinned manifest"
            ):
                support.assert_directory_identity(root, first)


class CommandConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.python = Path("/runtime/bin/python")

    def test_profile_build_is_exact_release_with_debug_symbols_and_frame_pointers(self) -> None:
        command = support.profile_build_command(
            python=self.python, wheel_directory=Path("target/wheels")
        )
        required = {
            "CARGO_PROFILE_RELEASE_DEBUG=1",
            "CARGO_PROFILE_RELEASE_OPT_LEVEL=3",
            "CARGO_PROFILE_RELEASE_STRIP=none",
            "RUSTFLAGS=-Cforce-frame-pointers=yes",
            "maturin",
            "build",
            "--release",
            "--locked",
        }
        self.assertLessEqual(required, set(command))
        self.assertEqual(support.command_value(command, "--interpreter"), str(self.python))

    def test_real_benchmark_command_freezes_both_profiles_six_cases_and_equal_budgets(self) -> None:
        command = support.benchmark_command(
            python=self.python,
            workload=Path("benchmarks/workloads-v2.json"),
            assets_root=Path("assets"),
            phase_c_report=Path("phase-c.json"),
            output=Path("benchmark.json"),
            report=Path("benchmark.md"),
            phase_c_publish_report=Path(
                "benchmarks/profile-evidence-v1/x86_64/phase-c/report.json"
            ),
        )
        self.assertEqual(
            support.command_value(command, "--candidate-adapter"),
            "qwen_mm.benchmark:create_adapter",
        )
        self.assertEqual(
            support.command_value(command, "--profiles"), ",".join(support.PROFILE_ALIASES)
        )
        self.assertEqual(support.command_value(command, "--cases"), ",".join(support.PROFILE_CASES))
        self.assertEqual(support.command_value(command, "--thread-regimes"), "one,production")
        self.assertEqual(support.command_value(command, "--production-thread-budget"), "4")
        self.assertEqual(
            support.command_value(command, "--build-labels"), support.PROFILE_BUILD_LABEL
        )
        self.assertNotIn("synthetic", command)
        self.assertEqual(
            support.command_value(command, "--phase-c-publish-report"),
            "benchmarks/profile-evidence-v1/x86_64/phase-c/report.json",
        )

    def test_profile_capture_command_freezes_24_native_x86_coordinates(self) -> None:
        command = support.profile_capture_command(
            python=self.python,
            workload=Path("workload.json"),
            benchmark_result=Path("benchmark.json"),
            build_command="env RUSTFLAGS=-Cforce-frame-pointers=yes maturin build --release --locked",
            build_log=Path("build.log"),
            wheel=Path("wheel.whl"),
            phase_c_report=Path("phase-c.json"),
            artifact_directory=Path("target/artifacts"),
            artifact_publish_directory=Path("target/published"),
            source_revision="a" * 40,
            source_digest="b" * 64,
            output=Path("profile.json"),
            benchmark_phase_c_source_report=Path("scratch/phase-c.json"),
        )
        self.assertEqual(len(support.matrix_coordinates()), 24)
        self.assertEqual(support.command_value(command, "--thread-budgets"), "1,4")
        self.assertEqual(support.command_value(command, "--repetitions"), "3")
        self.assertEqual(support.command_value(command, "--py-spy"), "py-spy")
        self.assertEqual(
            support.command_value(command, "--artifact-publish-directory"), "target/published"
        )
        self.assertEqual(support.command_value(command, "--sampler-rate-hz"), "99")
        self.assertIn("--source-clean", command)
        self.assertEqual(
            support.command_value(command, "--benchmark-phase-c-source-report"),
            "scratch/phase-c.json",
        )
        self.assertTrue(
            support.command_value(command, "--artifact-publish-directory").startswith(
                "target/published"
            )
        )

    def test_final_host_paths_are_committable_and_architecture_separated(self) -> None:
        arm = support.host_paths("arm64")
        x86 = support.host_paths("x86_64")
        self.assertEqual(str(arm["root"]), "benchmarks/profile-evidence-v1/arm64")
        self.assertEqual(str(x86["root"]), "benchmarks/profile-evidence-v1/x86_64")
        self.assertNotIn("target", str(arm["profile_bundle"]))
        self.assertNotIn("target", str(x86["profile_bundle"]))

    def test_capture_validation_is_live_but_installed_validation_is_portable(self) -> None:
        live = support.benchmark_validation_command(
            python=Path("/runtime/python"),
            result=Path("scratch/result.json"),
            phase_c_source_report=Path("scratch/phase-c.json"),
        )
        self.assertIn("validate", live)
        self.assertNotIn("validate-portable", live)
        self.assertEqual(
            support.command_value(live, "--phase-c-source-report"), "scratch/phase-c.json"
        )
        portable = support.benchmark_portable_validation_command(
            python=Path("/runtime/python"),
            result=Path("published/result.json"),
            runtime_authentication=Path("published/runtime.json"),
        )
        self.assertIn("validate-portable", portable)
        self.assertEqual(
            support.command_value(portable, "--runtime-authentication"),
            "published/runtime.json",
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(support, "_run_canonical") as run_canonical,
        ):
            root = Path(directory)
            for architecture in ("arm64", "x86_64"):
                support.canonical_validate_installed_host(
                    repository_root=root,
                    python=Path("/runtime/python"),
                    assets_root=root / "assets",
                    architecture=architecture,
                )
            benchmark_commands = [
                call.args[0]
                for call in run_canonical.call_args_list
                if "qwen_mm_reference.benchmark_v2" in call.args[0]
            ]
            self.assertEqual(len(benchmark_commands), 2)
            self.assertTrue(all("validate-portable" in command for command in benchmark_commands))


class ProvenanceAndArchiveTests(unittest.TestCase):
    def test_capture_provenance_rejects_dirty_source_and_emulation(self) -> None:
        support.validate_capture_provenance(_provenance())
        dirty = _provenance()
        dirty["source"]["git_status"] = " M source.rs"  # type: ignore[index]
        with self.assertRaisesRegex(support.ProfileCaptureArtifactError, "uncommitted"):
            support.validate_capture_provenance(dirty)
        emulated = _provenance()
        emulated["host"]["cpu_description"] = "QEMU Virtual CPU"  # type: ignore[index]
        with self.assertRaisesRegex(support.ProfileCaptureArtifactError, "emulation"):
            support.validate_capture_provenance(emulated)

    def test_capture_provenance_accepts_frozen_native_macos_arm_lane(self) -> None:
        value = _provenance()
        value["execution"] = "local_native_macos_arm_profile_v1"
        value["host"] = {
            "system": "Darwin",
            "machine": "arm64",
            "cpu_description": "Apple M4 Max",
        }
        value["protocol"]["sampler"] = {  # type: ignore[index]
            "name": "sample",
            "native": True,
            "interval_ms": 1,
            "duration_seconds": support.PY_SPY_DURATION_SECONDS,
        }
        support.validate_capture_provenance(value, architecture="arm64")

    def test_runtime_authentication_requires_one_wheel_runtime_everywhere(self) -> None:
        runtime = {
            "package": "qwen_mm",
            "version": "0.1.0",
            "package_artifact_sha256": "d" * 64,
            "native_module": "qwen_mm._native",
            "native_artifact_sha256": "e" * 64,
        }
        value = {
            "wheel_contents": {
                "package_artifact_sha256": "d" * 64,
                "native_artifact_sha256": "e" * 64,
            },
            "installed_candidate": {"resolved": True, "runtime_identity": runtime},
            "phase_c_runtime_identity": runtime,
            "benchmark_runtime_identity": runtime,
            "profile_runtime_identity": runtime,
            "native_module_file": "ELF 64-bit shared object, x86-64",
            "all_runtime_identities_equal": True,
        }
        support.validate_runtime_authentication(value)
        value["profile_runtime_identity"] = {**runtime, "native_artifact_sha256": "f" * 64}
        with self.assertRaisesRegex(support.ProfileCaptureArtifactError, "different runtimes"):
            support.validate_runtime_authentication(value)

    def test_integrity_archive_is_deterministic_and_detects_tampering(self) -> None:
        files = _required_files()
        first = support.create_integrity_archive(files)
        second = support.create_integrity_archive(dict(reversed(list(files.items()))))
        self.assertEqual(first, second)
        self.assertEqual(
            set(support.read_integrity_archive(first)) - {"artifact-manifest.json"}, set(files)
        )

        with zipfile.ZipFile(io.BytesIO(first)) as source:
            unpacked = {name: source.read(name) for name in source.namelist()}
        changed = next(iter(support.REQUIRED_ARTIFACT_FILES))
        unpacked[changed] += b"tampered"
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            for name, data in unpacked.items():
                archive.writestr(name, data)
        with self.assertRaisesRegex(support.ProfileCaptureArtifactError, "mismatch"):
            support.read_integrity_archive(output.getvalue())

    def test_integrity_archive_rejects_missing_required_member(self) -> None:
        files = _required_files()
        files.pop(next(iter(support.REQUIRED_ARTIFACT_FILES)))
        with self.assertRaisesRegex(support.ProfileCaptureArtifactError, "missing required"):
            support.create_integrity_archive(files)

    def test_archive_caps_apply_before_compression_or_member_read(self) -> None:
        self.assertEqual(support.MAX_ARCHIVE_COMPRESSED_BYTES, 90 * 1024 * 1024)
        self.assertEqual(support.MAX_ARCHIVE_UNCOMPRESSED_BYTES, 512 * 1024 * 1024)
        self.assertEqual(support.MAX_ARCHIVE_MEMBER_BYTES, 128 * 1024 * 1024)
        files = _required_files()
        original = support.MAX_ARCHIVE_MEMBER_BYTES
        support.MAX_ARCHIVE_MEMBER_BYTES = 8
        try:
            with self.assertRaisesRegex(support.ProfileCaptureArtifactError, "size cap"):
                support.create_integrity_archive(files)
        finally:
            support.MAX_ARCHIVE_MEMBER_BYTES = original
        original_compressed = support.MAX_ARCHIVE_COMPRESSED_BYTES
        support.MAX_ARCHIVE_COMPRESSED_BYTES = 3
        try:
            with self.assertRaisesRegex(support.ProfileCaptureArtifactError, "compressed size cap"):
                support.read_integrity_archive(b"four")
        finally:
            support.MAX_ARCHIVE_COMPRESSED_BYTES = original_compressed

    def test_regenerated_manifest_does_not_make_semantic_tampering_valid(self) -> None:
        files = _required_files()
        files[str(support.PROFILE_BUNDLE)] = json.dumps(
            {"protocol": {"profiles": ["attacker-controlled"]}}
        ).encode()
        archive = support.create_integrity_archive(files)
        # The regenerated integrity manifest is internally consistent, but the
        # host semantic validator still rejects the invented profile protocol.
        with self.assertRaises(support.ProfileCaptureArtifactError):
            support.validate_profile_artifact(archive)

    def test_manifest_is_machine_readable(self) -> None:
        archive = support.create_integrity_archive(_required_files())
        with zipfile.ZipFile(io.BytesIO(archive)) as value:
            manifest = json.loads(value.read("artifact-manifest.json"))
        self.assertEqual(manifest["schema_id"], support.ARTIFACT_SCHEMA_ID)
        self.assertEqual(set(manifest["files"]), support.REQUIRED_ARTIFACT_FILES)


class IngestAndMergeTests(unittest.TestCase):
    @staticmethod
    def _validated_host() -> dict[str, object]:
        return {"provenance": {"source": {"revision": "a" * 40, "tree_sha256": "b" * 64}}}

    def test_atomic_ingest_writes_only_the_host_root_and_runs_canonical_validation(self) -> None:
        member = "benchmarks/profile-evidence-v1/x86_64/profile/bundle.json"
        files = {"artifact-manifest.json": b"manifest", member: b"bundle"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "assets"
            assets.mkdir()
            with (
                mock.patch.object(support, "validate_profile_artifact", return_value={"ok": True}),
                mock.patch.object(support, "read_integrity_archive", return_value=files),
                mock.patch.object(support, "canonical_validate_installed_host") as canonical,
            ):
                result = support.ingest_profile_artifact(
                    b"archive",
                    repository_root=root,
                    python=Path("/runtime/python"),
                    assets_root=assets,
                    architecture="x86_64",
                )
            self.assertEqual(result, {"ok": True})
            self.assertEqual((root / member).read_bytes(), b"bundle")
            canonical.assert_called_once()
            self.assertFalse(
                any(path.name.startswith(".profile-ingest-") for path in root.rglob("*"))
            )

    def test_atomic_ingest_rolls_back_when_canonical_validation_fails(self) -> None:
        member = "benchmarks/profile-evidence-v1/arm64/profile/bundle.json"
        files = {"artifact-manifest.json": b"manifest", member: b"bundle"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(support, "validate_profile_artifact", return_value={}),
                mock.patch.object(support, "read_integrity_archive", return_value=files),
                mock.patch.object(
                    support,
                    "canonical_validate_installed_host",
                    side_effect=support.ProfileCaptureArtifactError("semantic failure"),
                ),
                self.assertRaisesRegex(support.ProfileCaptureArtifactError, "semantic failure"),
            ):
                support.ingest_profile_artifact(
                    b"archive",
                    repository_root=root,
                    python=Path("/runtime/python"),
                    assets_root=root / "assets",
                    architecture="arm64",
                )
            self.assertFalse((root / "benchmarks/profile-evidence-v1/arm64").exists())

    def test_dual_ingest_rolls_back_both_hosts_when_second_canonical_check_fails(self) -> None:
        files = {
            architecture: {
                "artifact-manifest.json": b"manifest",
                f"benchmarks/profile-evidence-v1/{architecture}/profile/bundle.json": b"bundle",
            }
            for architecture in ("arm64", "x86_64")
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(
                    support, "validate_profile_artifact", return_value=self._validated_host()
                ),
                mock.patch.object(
                    support,
                    "read_integrity_archive",
                    side_effect=lambda _value, *, architecture: files[architecture],
                ),
                mock.patch.object(
                    support,
                    "canonical_validate_installed_host",
                    side_effect=[None, support.ProfileCaptureArtifactError("x86 semantic failure")],
                ),
                self.assertRaisesRegex(support.ProfileCaptureArtifactError, "x86 semantic failure"),
            ):
                support.ingest_profile_artifacts(
                    {"arm64": b"arm", "x86_64": b"x86"},
                    repository_root=root,
                    python=Path("/runtime/python"),
                    assets_root=root / "assets",
                )
            for architecture in ("arm64", "x86_64"):
                self.assertFalse((root / f"benchmarks/profile-evidence-v1/{architecture}").exists())

    def test_merge_commands_require_both_host_bundles_and_final_validation(self) -> None:
        commands = support.merge_profile_evidence_commands(
            repository_root=Path("/repo"), python=Path("/runtime/python")
        )
        merge, validate = commands
        self.assertIn("/repo/benchmarks/profile-evidence-v1/arm64/profile/bundle.json", merge)
        self.assertIn("/repo/benchmarks/profile-evidence-v1/x86_64/profile/bundle.json", merge)
        self.assertEqual(merge[-2:], ["--report", "/repo/benchmarks/profile-evidence-v1/report.md"])
        self.assertEqual(validate[-1], "/repo/benchmarks/profile-evidence-v1/bundle.json")


if __name__ == "__main__":
    unittest.main()
