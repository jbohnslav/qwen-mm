"""Pure-stdlib orchestration support for Phase D1 host profile capture.

The command builders in this module are deliberately separate from Modal so
the ARM and x86 host lanes can execute and review the exact same protocol.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

ARTIFACT_SCHEMA_ID = "qwen-mm-modal-profile-artifact-v1"
ARTIFACT_SCHEMA_VERSION = 1
PROVENANCE_SCHEMA_ID = "qwen-mm-modal-profile-provenance-v1"
PROVENANCE_SCHEMA_VERSION = 1

PROFILE_ALIASES = ("qwen3-vl-8b", "qwen3.5-9b")
PROFILE_CASES = ("text_short", "text_long", "image1", "image24", "ragged24", "rgb24")
THREAD_BUDGETS = (1, 4)
THREAD_REGIMES = ("one", "production")
PROFILE_BUILD_LABEL = "profiled-release"
PROFILE_REPETITIONS = 3
PROFILE_EVENT_CAPACITY = 4096
PY_SPY_VERSION = "0.4.1"
PY_SPY_DEPENDENCY = "py-spy==0.4.1"
PY_SPY_BINARY_PATH = "/workspace/qwen-mm/.venv/bin/py-spy"
PY_SPY_RATE_HZ = 99
PY_SPY_DURATION_SECONDS = 2

EVIDENCE_RELATIVE_ROOT = PurePosixPath("benchmarks/profile-evidence-v1")
X86_EVIDENCE_ROOT = EVIDENCE_RELATIVE_ROOT / "x86_64"
ARM_EVIDENCE_ROOT = EVIDENCE_RELATIVE_ROOT / "arm64"
BUILD_LOG = X86_EVIDENCE_ROOT / "build/build.log"
PROFILE_PUBLISH = X86_EVIDENCE_ROOT / "artifacts"
WHEEL_DIRECTORY = PROFILE_PUBLISH
ASSET_MANIFEST = X86_EVIDENCE_ROOT / "assets-manifest.json"
RUNTIME_AUTHENTICATION = X86_EVIDENCE_ROOT / "runtime-authentication.json"
PROVENANCE = X86_EVIDENCE_ROOT / "provenance.json"
PHASE_C_REPORT = X86_EVIDENCE_ROOT / "phase-c/report.json"
PHASE_C_SUMMARY = X86_EVIDENCE_ROOT / "phase-c/summary.md"
BENCHMARK_RESULT = X86_EVIDENCE_ROOT / "benchmark/result.json"
BENCHMARK_REPORT = X86_EVIDENCE_ROOT / "benchmark/report.md"
PROFILE_BUNDLE = X86_EVIDENCE_ROOT / "profile/bundle.json"
PROFILE_REPORT = X86_EVIDENCE_ROOT / "profile/report.md"
LOG_DIRECTORY = X86_EVIDENCE_ROOT / "logs"

FINAL_PROFILE_BUNDLE = EVIDENCE_RELATIVE_ROOT / "bundle.json"
FINAL_PROFILE_REPORT = EVIDENCE_RELATIVE_ROOT / "report.md"

MAX_ARCHIVE_COMPRESSED_BYTES = 90 * 1024 * 1024
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 512

REQUIRED_ARTIFACT_FILES = frozenset(
    str(path)
    for path in (
        ASSET_MANIFEST,
        BENCHMARK_REPORT,
        BENCHMARK_RESULT,
        BUILD_LOG,
        PHASE_C_REPORT,
        PHASE_C_SUMMARY,
        PROFILE_BUNDLE,
        PROFILE_REPORT,
        PROVENANCE,
        RUNTIME_AUTHENTICATION,
        LOG_DIRECTORY / "benchmark.log",
        LOG_DIRECTORY / "benchmark-validation.log",
        LOG_DIRECTORY / "install.log",
        LOG_DIRECTORY / "phase-c.log",
        LOG_DIRECTORY / "phase-c-validation.log",
        LOG_DIRECTORY / "profile.log",
        LOG_DIRECTORY / "profile-report.log",
        LOG_DIRECTORY / "profile-validation.log",
        LOG_DIRECTORY / "sampler-preflight.log",
    )
)


def host_paths(architecture: str) -> dict[str, PurePosixPath]:
    """Return the frozen final repository paths for one host capture."""

    if architecture not in {"arm64", "x86_64"}:
        raise ProfileCaptureArtifactError(f"unsupported profile architecture: {architecture}")
    root = EVIDENCE_RELATIVE_ROOT / architecture
    return {
        "root": root,
        "build_log": root / "build/build.log",
        "publish": root / "artifacts",
        "asset_manifest": root / "assets-manifest.json",
        "runtime_authentication": root / "runtime-authentication.json",
        "provenance": root / "provenance.json",
        "phase_c_report": root / "phase-c/report.json",
        "phase_c_summary": root / "phase-c/summary.md",
        "benchmark_result": root / "benchmark/result.json",
        "benchmark_report": root / "benchmark/report.md",
        "profile_bundle": root / "profile/bundle.json",
        "profile_report": root / "profile/report.md",
        "logs": root / "logs",
    }


def required_artifact_files(architecture: str) -> frozenset[str]:
    paths = host_paths(architecture)
    logs = paths["logs"]
    return frozenset(
        str(path)
        for path in (
            paths["asset_manifest"],
            paths["benchmark_report"],
            paths["benchmark_result"],
            paths["build_log"],
            paths["phase_c_report"],
            paths["phase_c_summary"],
            paths["profile_bundle"],
            paths["profile_report"],
            paths["provenance"],
            paths["runtime_authentication"],
            logs / "benchmark.log",
            logs / "benchmark-validation.log",
            logs / "install.log",
            logs / "phase-c.log",
            logs / "phase-c-validation.log",
            logs / "profile.log",
            logs / "profile-report.log",
            logs / "profile-validation.log",
            logs / "sampler-preflight.log",
        )
    )


class ProfileCaptureArtifactError(ValueError):
    """Raised when capture inputs or an evidence archive are not authentic."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def safe_relative_path(value: str | PurePosixPath) -> PurePosixPath:
    text = str(value)
    path = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != text
    ):
        raise ProfileCaptureArtifactError(f"unsafe evidence path: {text!r}")
    return path


def file_identity(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ProfileCaptureArtifactError(f"evidence path is not a regular file: {path}")
    identity: dict[str, Any] = {
        "sha256": sha256_path(path),
        "bytes": path.stat().st_size,
    }
    if relative_to is not None:
        try:
            relative = path.resolve().relative_to(relative_to.resolve()).as_posix()
        except ValueError as error:
            raise ProfileCaptureArtifactError(f"evidence path escapes root: {path}") from error
        safe_relative_path(relative)
        identity["path"] = relative
    return identity


def directory_identity(root: Path) -> dict[str, Any]:
    """Authenticate logical paths and bytes independent of symlink materialization.

    Modal 1.4 may upload a Hugging Face snapshot symlink as a regular file.  The
    manifest intentionally authenticates the bytes visible at every logical path,
    so that content-equivalent materialization is accepted while any byte/path
    change is rejected.
    """

    if not root.is_dir():
        raise ProfileCaptureArtifactError(f"asset directory is unavailable: {root}")
    records: list[dict[str, Any]] = []
    logical_bytes = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        safe_relative_path(relative)
        if path.is_symlink():
            if not path.is_file():
                raise ProfileCaptureArtifactError(f"asset symlink is dangling: {relative}")
            data = path.read_bytes()
            logical_bytes += len(data)
            records.append({"path": relative, "bytes": len(data), "sha256": sha256_bytes(data)})
        elif path.is_file():
            identity = file_identity(path)
            logical_bytes += identity["bytes"]
            records.append({"path": relative, **identity})
    if not records:
        raise ProfileCaptureArtifactError(f"asset directory is empty: {root}")
    return {
        "schema_id": "qwen-mm-logical-directory-identity-v2",
        "schema_version": 2,
        "tree_sha256": sha256_bytes(canonical_json(records)),
        "entry_count": len(records),
        "logical_bytes": logical_bytes,
        "entries": records,
    }


def assert_directory_identity(root: Path, expected: Mapping[str, Any]) -> None:
    actual = directory_identity(root)
    if actual != expected:
        raise ProfileCaptureArtifactError(
            "packaged asset directory differs from its local hash-pinned manifest"
        )


def profile_build_command(*, python: Path, wheel_directory: Path) -> list[str]:
    return [
        "env",
        "CARGO_PROFILE_RELEASE_DEBUG=1",
        "CARGO_PROFILE_RELEASE_OPT_LEVEL=3",
        "CARGO_PROFILE_RELEASE_STRIP=none",
        "RUSTFLAGS=-Cforce-frame-pointers=yes",
        "uv",
        "run",
        "--locked",
        "--no-sync",
        "maturin",
        "build",
        "--release",
        "--locked",
        "--interpreter",
        str(python),
        "--out",
        str(wheel_directory),
    ]


def install_wheel_command(*, python: Path, wheel: Path) -> list[str]:
    return [
        "uv",
        "pip",
        "install",
        "--python",
        str(python),
        "--reinstall",
        "--no-deps",
        str(wheel),
    ]


def phase_c_command(
    *,
    python: Path,
    wheel: Path,
    assets_root: Path,
    output: Path,
    report: Path,
    summary: Path,
) -> list[str]:
    return [
        str(python),
        "-m",
        "qwen_mm_reference.phase_c_conformance",
        "run",
        "--candidate-python",
        str(python),
        "--wheel",
        str(wheel),
        "--assets-root",
        str(assets_root),
        "--output",
        str(output),
        "--report",
        str(report),
        "--summary",
        str(summary),
    ]


def phase_c_validation_command(*, python: Path, assets_root: Path, report: Path) -> list[str]:
    return [
        str(python),
        "-m",
        "qwen_mm_reference.phase_c_conformance",
        "validate",
        "--assets-root",
        str(assets_root),
        "--report",
        str(report),
    ]


def benchmark_command(
    *,
    python: Path,
    workload: Path,
    assets_root: Path,
    phase_c_report: Path,
    output: Path,
    report: Path,
    phase_c_publish_report: Path | None = None,
) -> list[str]:
    command = [
        str(python),
        "-m",
        "qwen_mm_reference.benchmark_v2",
        "run",
        "--workload",
        str(workload),
        "--mode",
        "smoke",
        "--reference-adapter",
        "official",
        "--candidate-adapter",
        "qwen_mm.benchmark:create_adapter",
        "--profiles",
        ",".join(PROFILE_ALIASES),
        "--cases",
        ",".join(PROFILE_CASES),
        "--process-repetitions",
        "1",
        "--warmups",
        "1",
        "--minimum-samples",
        "3",
        "--minimum-seconds",
        "0",
        "--thread-regimes",
        ",".join(THREAD_REGIMES),
        "--build-labels",
        PROFILE_BUILD_LABEL,
        "--production-thread-budget",
        "4",
        "--phase-c-report",
        str(phase_c_report),
        "--phase-c-assets-root",
        str(assets_root),
        "--output",
        str(output),
        "--report",
        str(report),
    ]
    if phase_c_publish_report is not None:
        command.extend(("--phase-c-publish-report", str(phase_c_publish_report)))
    return command


def benchmark_validation_command(
    *, python: Path, result: Path, phase_c_source_report: Path | None = None
) -> list[str]:
    command = [
        str(python),
        "-m",
        "qwen_mm_reference.benchmark_v2",
        "validate",
        str(result),
    ]
    if phase_c_source_report is not None:
        command.extend(("--phase-c-source-report", str(phase_c_source_report)))
    return command


def benchmark_portable_validation_command(
    *, python: Path, result: Path, runtime_authentication: Path
) -> list[str]:
    return [
        str(python),
        "-m",
        "qwen_mm_reference.benchmark_v2",
        "validate-portable",
        str(result),
        "--runtime-authentication",
        str(runtime_authentication),
    ]


def profile_capture_command(
    *,
    python: Path,
    workload: Path,
    benchmark_result: Path,
    build_command: str,
    build_log: Path,
    wheel: Path,
    phase_c_report: Path,
    artifact_directory: Path,
    artifact_publish_directory: Path,
    source_revision: str,
    source_digest: str,
    output: Path,
    benchmark_phase_c_source_report: Path | None = None,
    py_spy: str = "py-spy",
) -> list[str]:
    command = [
        str(python),
        "-m",
        "qwen_mm_reference.profile_v1",
        "capture",
        "--workload",
        str(workload),
        "--benchmark-result",
        str(benchmark_result),
        "--profiles",
        ",".join(PROFILE_ALIASES),
        "--cases",
        ",".join(PROFILE_CASES),
        "--thread-budgets",
        ",".join(map(str, THREAD_BUDGETS)),
        "--repetitions",
        str(PROFILE_REPETITIONS),
        "--event-capacity",
        str(PROFILE_EVENT_CAPACITY),
        "--build-command",
        build_command,
        "--build-log",
        str(build_log),
        "--wheel",
        str(wheel),
        "--phase-c-report",
        str(phase_c_report),
        "--artifact-directory",
        str(artifact_directory),
        "--artifact-publish-directory",
        str(artifact_publish_directory),
        "--py-spy",
        py_spy,
        "--sampler-rate-hz",
        str(PY_SPY_RATE_HZ),
        "--sampler-duration-seconds",
        str(PY_SPY_DURATION_SECONDS),
        "--source-revision",
        source_revision,
        "--source-digest",
        source_digest,
        "--source-clean",
        "--output",
        str(output),
    ]
    if benchmark_phase_c_source_report is not None:
        command.extend(
            (
                "--benchmark-phase-c-source-report",
                str(benchmark_phase_c_source_report),
            )
        )
    return command


def profile_validation_command(*, python: Path, bundle: Path) -> list[str]:
    return [
        str(python),
        "-m",
        "qwen_mm_reference.profile_v1",
        "validate",
        "--allow-single-architecture",
        str(bundle),
    ]


def profile_report_command(*, python: Path, bundle: Path, output: Path) -> list[str]:
    return [
        str(python),
        "-m",
        "qwen_mm_reference.profile_v1",
        "report",
        str(bundle),
        "--output",
        str(output),
    ]


def wheel_contents_identity(wheel: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            package = [name for name in names if name == "qwen_mm/__init__.py"]
            native = [
                name
                for name in names
                if name.startswith("qwen_mm/_native.")
                and Path(name).suffix in {".so", ".dylib", ".pyd"}
            ]
            if len(package) != 1 or len(native) != 1:
                raise ProfileCaptureArtifactError(
                    "profile wheel must contain exactly one package and Linux native module"
                )
            package_bytes = archive.read(package[0])
            native_bytes = archive.read(native[0])
    except (OSError, zipfile.BadZipFile) as error:
        raise ProfileCaptureArtifactError(f"profile wheel is unreadable: {wheel}") from error
    return {
        "wheel": file_identity(wheel),
        "package_member": package[0],
        "package_artifact_sha256": sha256_bytes(package_bytes),
        "native_member": native[0],
        "native_artifact_sha256": sha256_bytes(native_bytes),
        "native_bytes": len(native_bytes),
    }


def validate_runtime_authentication(
    value: Mapping[str, Any], *, architecture: str = "x86_64"
) -> None:
    required = {
        "wheel_contents",
        "installed_candidate",
        "phase_c_runtime_identity",
        "benchmark_runtime_identity",
        "profile_runtime_identity",
        "native_module_file",
        "all_runtime_identities_equal",
    }
    if set(value) != required:
        raise ProfileCaptureArtifactError("runtime authentication fields are incomplete")
    installed = value["installed_candidate"]
    if not isinstance(installed, Mapping) or installed.get("resolved") is not True:
        raise ProfileCaptureArtifactError("installed candidate adapter did not resolve")
    runtime = installed.get("runtime_identity")
    if not isinstance(runtime, Mapping):
        raise ProfileCaptureArtifactError("installed candidate runtime identity is absent")
    compared = (
        value["phase_c_runtime_identity"],
        value["benchmark_runtime_identity"],
        value["profile_runtime_identity"],
    )
    if value["all_runtime_identities_equal"] is not True or any(
        candidate != runtime for candidate in compared
    ):
        raise ProfileCaptureArtifactError("Phase C, benchmark, and profile used different runtimes")
    wheel = value["wheel_contents"]
    if not isinstance(wheel, Mapping) or wheel.get("native_artifact_sha256") != runtime.get(
        "native_artifact_sha256"
    ):
        raise ProfileCaptureArtifactError("installed native module differs from the wheel")
    if wheel.get("package_artifact_sha256") != runtime.get("package_artifact_sha256"):
        raise ProfileCaptureArtifactError("installed package differs from the wheel")
    native_file = value["native_module_file"]
    markers = {
        "x86_64": ("ELF 64-bit", "x86-64"),
        "arm64": ("Mach-O 64-bit", "arm64"),
    }.get(architecture)
    if (
        markers is None
        or not isinstance(native_file, str)
        or not all(marker.lower() in native_file.lower() for marker in markers)
    ):
        raise ProfileCaptureArtifactError(
            f"profile runtime native extension does not match {architecture}"
        )


def validate_capture_provenance(value: Mapping[str, Any], *, architecture: str = "x86_64") -> None:
    if (
        value.get("schema_id") != PROVENANCE_SCHEMA_ID
        or value.get("schema_version") != PROVENANCE_SCHEMA_VERSION
    ):
        raise ProfileCaptureArtifactError("profile capture provenance schema mismatch")
    expected_execution = {
        "x86_64": "modal_native_linux_x86_profile_v1",
        "arm64": "local_native_macos_arm_profile_v1",
    }.get(architecture)
    if value.get("execution") != expected_execution:
        raise ProfileCaptureArtifactError("profile capture execution label mismatch")
    source = value.get("source")
    if not isinstance(source, Mapping) or source.get("clean") is not True:
        raise ProfileCaptureArtifactError("Modal profile evidence must use a clean source")
    for field in ("revision", "tree_sha256"):
        text = source.get(field)
        if not isinstance(text, str) or not text:
            raise ProfileCaptureArtifactError(f"source provenance is missing {field}")
    digest = source["tree_sha256"]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ProfileCaptureArtifactError("source tree digest is not SHA-256")
    if source.get("git_status") != "":
        raise ProfileCaptureArtifactError("Modal profile source has uncommitted files")
    host = value.get("host")
    expected_hosts = {
        "x86_64": ("Linux", {"x86_64", "amd64"}),
        "arm64": ("Darwin", {"arm64", "aarch64"}),
    }
    expected_system, expected_machines = expected_hosts[architecture]
    if (
        not isinstance(host, Mapping)
        or host.get("system") != expected_system
        or str(host.get("machine", "")).lower() not in expected_machines
    ):
        raise ProfileCaptureArtifactError(
            f"profile host is not native {expected_system} {architecture}"
        )
    cpu_description = str(host.get("cpu_description", "")).lower()
    if any(marker in cpu_description for marker in ("qemu", "tcg", "software emulation")):
        raise ProfileCaptureArtifactError("Modal profile host reports CPU emulation")
    assets = value.get("assets")
    if not isinstance(assets, Mapping) or len(str(assets.get("tree_sha256", ""))) != 64:
        raise ProfileCaptureArtifactError("profile asset provenance is incomplete")
    if assets.get("logical_bytes", 0) < 30 * 1024 * 1024:
        raise ProfileCaptureArtifactError("profile asset payload is unexpectedly small")
    protocol = value.get("protocol")
    expected = {
        "profiles": list(PROFILE_ALIASES),
        "cases": list(PROFILE_CASES),
        "thread_budgets": list(THREAD_BUDGETS),
        "build_label": PROFILE_BUILD_LABEL,
        "observed_coordinates": len(PROFILE_ALIASES) * len(PROFILE_CASES) * len(THREAD_BUDGETS),
        "observation_repetitions": PROFILE_REPETITIONS,
        "sampler": (
            {
                "name": "py-spy",
                "version": PY_SPY_VERSION,
                "native": False,
                "dependency": PY_SPY_DEPENDENCY,
                "binary_path": PY_SPY_BINARY_PATH,
                "rate_hz": PY_SPY_RATE_HZ,
                "duration_seconds": PY_SPY_DURATION_SECONDS,
            }
            if architecture == "x86_64"
            else {
                "name": "sample",
                "native": True,
                "interval_ms": 1,
                "duration_seconds": PY_SPY_DURATION_SECONDS,
            }
        ),
    }
    if protocol != expected:
        raise ProfileCaptureArtifactError("profile protocol provenance is not frozen D1")
    commands = value.get("commands")
    if (
        not isinstance(commands, list)
        or not commands
        or not all(isinstance(command, str) and command for command in commands)
    ):
        raise ProfileCaptureArtifactError("capture provenance has no command log")


def _assert_archive_limits(files: Mapping[str, bytes]) -> None:
    if len(files) + 1 > MAX_ARCHIVE_MEMBERS:
        raise ProfileCaptureArtifactError("capture archive has too many members")
    total = sum(len(data) for data in files.values())
    if total > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
        raise ProfileCaptureArtifactError("capture archive exceeds the uncompressed size cap")
    too_large = next(
        (name for name, data in files.items() if len(data) > MAX_ARCHIVE_MEMBER_BYTES), None
    )
    if too_large is not None:
        raise ProfileCaptureArtifactError(f"capture archive member exceeds size cap: {too_large}")


def create_integrity_archive(files: Mapping[str, bytes], *, architecture: str = "x86_64") -> bytes:
    required = required_artifact_files(architecture)
    missing = sorted(required - files.keys())
    if missing:
        raise ProfileCaptureArtifactError(
            f"capture archive is missing required files: {', '.join(missing)}"
        )
    if "artifact-manifest.json" in files:
        raise ProfileCaptureArtifactError("artifact manifest is generated")
    _assert_archive_limits(files)
    manifest_files: dict[str, dict[str, Any]] = {}
    for name, data in sorted(files.items()):
        safe_relative_path(name)
        if not isinstance(data, bytes):
            raise TypeError(f"archive member {name!r} must be bytes")
        manifest_files[name] = {"bytes": len(data), "sha256": sha256_bytes(data)}
    manifest = {
        "schema_id": ARTIFACT_SCHEMA_ID,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "files": manifest_files,
    }
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
        info = zipfile.ZipInfo("artifact-manifest.json", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, manifest_bytes)
    result = output.getvalue()
    if len(result) > MAX_ARCHIVE_COMPRESSED_BYTES:
        raise ProfileCaptureArtifactError("capture archive exceeds the compressed size cap")
    return result


def _json_member(files: Mapping[str, bytes], name: PurePosixPath) -> dict[str, Any]:
    try:
        value = json.loads(files[str(name)])
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProfileCaptureArtifactError(f"invalid JSON evidence member: {name}") from error
    if not isinstance(value, dict):
        raise ProfileCaptureArtifactError(f"JSON evidence member must be an object: {name}")
    return value


def read_integrity_archive(value: bytes, *, architecture: str = "x86_64") -> dict[str, bytes]:
    if len(value) > MAX_ARCHIVE_COMPRESSED_BYTES:
        raise ProfileCaptureArtifactError("capture archive exceeds the compressed size cap")
    try:
        with zipfile.ZipFile(io.BytesIO(value)) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_ARCHIVE_MEMBERS:
                raise ProfileCaptureArtifactError("capture archive has too many members")
            if sum(info.file_size for info in infos) > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                raise ProfileCaptureArtifactError(
                    "capture archive exceeds the uncompressed size cap"
                )
            oversized = next(
                (info.filename for info in infos if info.file_size > MAX_ARCHIVE_MEMBER_BYTES),
                None,
            )
            if oversized is not None:
                raise ProfileCaptureArtifactError(
                    f"capture archive member exceeds size cap: {oversized}"
                )
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ProfileCaptureArtifactError("capture archive has duplicate paths")
            for info in infos:
                safe_relative_path(info.filename)
                file_type = (info.external_attr >> 16) & 0o170000
                if file_type not in {0, 0o100000}:
                    raise ProfileCaptureArtifactError(
                        f"capture archive member is not a regular file: {info.filename}"
                    )
            files = {name: archive.read(name) for name in names}
    except (OSError, zipfile.BadZipFile) as error:
        raise ProfileCaptureArtifactError("capture artifact is not a readable ZIP") from error
    manifest = _json_member(files, PurePosixPath("artifact-manifest.json"))
    if (
        manifest.get("schema_id") != ARTIFACT_SCHEMA_ID
        or manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
    ):
        raise ProfileCaptureArtifactError("capture archive schema mismatch")
    declared = manifest.get("files")
    payload_names = set(files) - {"artifact-manifest.json"}
    if not isinstance(declared, Mapping) or set(declared) != payload_names:
        raise ProfileCaptureArtifactError("capture archive inventory mismatch")
    missing = sorted(required_artifact_files(architecture) - payload_names)
    if missing:
        raise ProfileCaptureArtifactError(
            f"capture archive is missing required files: {', '.join(missing)}"
        )
    for name in sorted(payload_names):
        metadata = declared[name]
        if not isinstance(metadata, Mapping) or metadata.get("bytes") != len(files[name]):
            raise ProfileCaptureArtifactError(f"capture archive size mismatch: {name}")
        if metadata.get("sha256") != sha256_bytes(files[name]):
            raise ProfileCaptureArtifactError(f"capture archive hash mismatch: {name}")
    _assert_archive_limits(
        {name: data for name, data in files.items() if name != "artifact-manifest.json"}
    )
    return files


def _published_identity_paths(bundle: Mapping[str, Any], provenance: Mapping[str, Any]) -> set[str]:
    captures = bundle.get("captures")
    if not isinstance(captures, list) or len(captures) != 1:
        raise ProfileCaptureArtifactError("host artifact must contain exactly one capture")
    capture = captures[0]
    identities: list[Any] = [
        capture.get("build", {}).get("log", {}),
        capture.get("build", {}).get("wheel", {}),
        capture.get("phase_c_report", {}),
        capture.get("paired_benchmark", {}),
        provenance.get("build", {}).get("log", {}),
        provenance.get("build", {}).get("wheel", {}),
    ]
    for item in capture.get("sampled_profiles", []):
        identities.extend(
            (
                item.get("worker", {}).get("result", {}),
                item.get("artifacts", {}).get("raw", {}),
                item.get("artifacts", {}).get("collapsed", {}),
            )
        )
    result: set[str] = set()
    for identity in identities:
        if not isinstance(identity, Mapping) or not isinstance(identity.get("path"), str):
            raise ProfileCaptureArtifactError("profile bundle has an invalid published identity")
        result.add(safe_relative_path(identity["path"]).as_posix())
    return result


def validate_profile_artifact(value: bytes, *, architecture: str = "x86_64") -> dict[str, Any]:
    paths = host_paths(architecture)
    files = read_integrity_archive(value, architecture=architecture)
    provenance = _json_member(files, paths["provenance"])
    validate_capture_provenance(provenance, architecture=architecture)
    assets = _json_member(files, paths["asset_manifest"])
    if assets != provenance["assets"]:
        raise ProfileCaptureArtifactError("asset manifest differs from provenance")
    runtime = _json_member(files, paths["runtime_authentication"])
    validate_runtime_authentication(runtime, architecture=architecture)

    phase_c = _json_member(files, paths["phase_c_report"])
    if phase_c.get("status") != "pass" or phase_c.get("passed") is not True:
        raise ProfileCaptureArtifactError("fresh Phase C gate did not pass")
    benchmark = _json_member(files, paths["benchmark_result"])
    benchmark_protocol = benchmark.get("protocol", {})
    if (
        benchmark.get("architecture_family") != architecture
        or benchmark_protocol.get("profiles") != list(PROFILE_ALIASES)
        or benchmark_protocol.get("cases") != list(PROFILE_CASES)
        or benchmark_protocol.get("thread_regimes") != list(THREAD_REGIMES)
        or benchmark_protocol.get("thread_budget_mapping") != {"one": 1, "production": 4}
        or benchmark_protocol.get("build_labels") != [PROFILE_BUILD_LABEL]
        or benchmark_protocol.get("self_test_only") is not False
    ):
        raise ProfileCaptureArtifactError("paired benchmark is not the real frozen D1 matrix")
    benchmark_phase_c = benchmark.get("release_eligibility", {}).get("phase_c", {})
    if (
        not isinstance(benchmark_phase_c, Mapping)
        or benchmark_phase_c.get("status") != "pass"
        or benchmark_phase_c.get("report_path") != str(paths["phase_c_report"])
        or benchmark_phase_c.get("assets_root") != "reference/.cache/huggingface"
        or benchmark_phase_c.get("report_sha256")
        != sha256_bytes(files[str(paths["phase_c_report"])])
        or benchmark_phase_c.get("evidence", {}).get("candidate_runtime_identity")
        != runtime["benchmark_runtime_identity"]
    ):
        raise ProfileCaptureArtifactError(
            "paired benchmark does not authenticate the final durable Phase C evidence"
        )

    bundle = _json_member(files, paths["profile_bundle"])
    protocol = bundle.get("protocol", {})
    if (
        protocol.get("profiles") != list(PROFILE_ALIASES)
        or protocol.get("cases") != list(PROFILE_CASES)
        or protocol.get("thread_budgets") != list(THREAD_BUDGETS)
        or protocol.get("repetitions") != PROFILE_REPETITIONS
    ):
        raise ProfileCaptureArtifactError("profile bundle protocol differs from frozen D1")
    captures = bundle.get("captures")
    if not isinstance(captures, list) or len(captures) != 1:
        raise ProfileCaptureArtifactError("host artifact must contain one profile capture")
    capture = captures[0]
    if capture.get("host", {}).get("architecture_family") != architecture:
        raise ProfileCaptureArtifactError("profile bundle capture architecture is incorrect")
    captured_source = capture.get("source", {})
    if (
        captured_source.get("revision") != provenance["source"]["revision"]
        or captured_source.get("tree_sha256") != provenance["source"]["tree_sha256"]
        or captured_source.get("dirty") is not False
    ):
        raise ProfileCaptureArtifactError("profile bundle source differs from clean provenance")

    def authenticate_published(identity: Mapping[str, Any]) -> None:
        name = identity.get("path")
        if not isinstance(name, str) or name not in files:
            raise ProfileCaptureArtifactError(
                "profile bundle references missing published evidence"
            )
        if len(files[name]) != identity.get("bytes") or sha256_bytes(files[name]) != identity.get(
            "sha256"
        ):
            raise ProfileCaptureArtifactError("published profile evidence identity mismatch")

    for identity in (
        capture.get("build", {}).get("log", {}),
        capture.get("build", {}).get("wheel", {}),
        capture.get("phase_c_report", {}),
        capture.get("paired_benchmark", {}),
    ):
        authenticate_published(identity)
    for identity in (
        provenance.get("build", {}).get("log", {}),
        provenance.get("build", {}).get("wheel", {}),
    ):
        authenticate_published(identity)
    if capture.get("build", {}).get("command") != provenance.get("build", {}).get("command"):
        raise ProfileCaptureArtifactError("profile bundle build command differs from provenance")
    expected_coordinates = {
        (profile, case, budget)
        for profile in PROFILE_ALIASES
        for case in PROFILE_CASES
        for budget in THREAD_BUDGETS
    }
    observations = capture.get("observations", [])
    observed = {
        (item.get("profile_alias"), item.get("case_id"), item.get("thread_budget"))
        for item in observations
    }
    if observed != expected_coordinates or len(observations) != len(expected_coordinates) * 3:
        raise ProfileCaptureArtifactError("observed profile matrix is incomplete")
    sampled = capture.get("sampled_profiles", [])
    sampled_coordinates = {
        (item.get("profile_alias"), item.get("case_id"), item.get("thread_budget"))
        for item in sampled
    }
    if sampled_coordinates != expected_coordinates or len(sampled) != len(expected_coordinates):
        raise ProfileCaptureArtifactError("sampled profile matrix is incomplete")
    for item in sampled:
        sampler = item.get("sampler", {})
        expected_sampler = "py-spy" if architecture == "x86_64" else "sample"
        expected_native = architecture == "arm64"
        if sampler.get("name") != expected_sampler or sampler.get("native") is not expected_native:
            raise ProfileCaptureArtifactError("sampled profile does not use the frozen sampler")
        for identity in (
            item.get("worker", {}).get("result", {}),
            item.get("artifacts", {}).get("raw", {}),
            item.get("artifacts", {}).get("collapsed", {}),
        ):
            name = identity.get("path")
            if not isinstance(name, str) or name not in files:
                raise ProfileCaptureArtifactError(
                    "profile bundle references a missing raw artifact"
                )
            if len(files[name]) != identity.get("bytes") or sha256_bytes(
                files[name]
            ) != identity.get("sha256"):
                raise ProfileCaptureArtifactError("sampled raw artifact identity mismatch")
    wheels = sorted(
        name for name in files if name.startswith(f"{paths['publish']}/") and name.endswith(".whl")
    )
    if len(wheels) != 1:
        raise ProfileCaptureArtifactError("capture archive must contain one profiled wheel")
    wheel_identity = runtime["wheel_contents"]["wheel"]
    if len(files[wheels[0]]) != wheel_identity.get("bytes") or sha256_bytes(
        files[wheels[0]]
    ) != wheel_identity.get("sha256"):
        raise ProfileCaptureArtifactError("archived profile wheel differs from authentication")
    expected_inventory = set(required_artifact_files(architecture)) | _published_identity_paths(
        bundle, provenance
    )
    payload_inventory = set(files) - {"artifact-manifest.json"}
    if payload_inventory != expected_inventory:
        raise ProfileCaptureArtifactError("capture archive contains non-allowlisted evidence")
    for name in required_artifact_files(architecture):
        if name.startswith(f"{paths['logs']}/") and not files[name].strip():
            raise ProfileCaptureArtifactError(f"capture log is empty: {name}")
    return {
        "provenance": provenance,
        "assets": assets,
        "runtime_authentication": runtime,
        "phase_c": phase_c,
        "benchmark": benchmark,
        "profile": bundle,
    }


def collect_evidence_files(
    repository_root: Path, *, architecture: str = "x86_64"
) -> dict[str, bytes]:
    """Collect only the fixed inventory and authenticated published identities."""

    root = repository_root.resolve()
    paths = host_paths(architecture)
    initial = set(required_artifact_files(architecture))
    initial_files: dict[str, bytes] = {}
    for name in sorted(initial):
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ProfileCaptureArtifactError(f"required durable evidence is missing: {name}")
        initial_files[name] = path.read_bytes()
    bundle = _json_member(initial_files, paths["profile_bundle"])
    provenance = _json_member(initial_files, paths["provenance"])
    inventory = initial | _published_identity_paths(bundle, provenance)
    files: dict[str, bytes] = {}
    prefix = f"{paths['root']}/"
    for name in sorted(inventory):
        if not name.startswith(prefix):
            raise ProfileCaptureArtifactError(
                f"published evidence is outside the {architecture} final root: {name}"
            )
        path = (root / name).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ProfileCaptureArtifactError(
                f"evidence path escapes repository: {name}"
            ) from error
        if not path.is_file() or path.is_symlink():
            raise ProfileCaptureArtifactError(f"allowlisted evidence is missing: {name}")
        files[name] = path.read_bytes()
    _assert_archive_limits(files)
    return files


def write_artifact_atomically(
    value: bytes, destination: Path, *, architecture: str = "x86_64"
) -> dict[str, Any]:
    validated = validate_profile_artifact(value, architecture=architecture)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(value)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return validated


def _run_canonical(command: Sequence[str], *, repository_root: Path) -> None:
    result = subprocess.run(
        list(command),
        cwd=repository_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise ProfileCaptureArtifactError(
            f"canonical evidence validation failed: {' '.join(command)}\n{result.stdout}"
        )


def canonical_validate_installed_host(
    *, repository_root: Path, python: Path, assets_root: Path, architecture: str
) -> None:
    """Run the authoritative Phase C, benchmark, and profile validators in place."""

    paths = host_paths(architecture)
    _run_canonical(
        phase_c_validation_command(
            python=python,
            assets_root=assets_root,
            report=repository_root / str(paths["phase_c_report"]),
        ),
        repository_root=repository_root,
    )
    benchmark_command = benchmark_portable_validation_command(
        python=python,
        result=repository_root / str(paths["benchmark_result"]),
        runtime_authentication=repository_root / str(paths["runtime_authentication"]),
    )
    _run_canonical(benchmark_command, repository_root=repository_root)
    _run_canonical(
        profile_validation_command(
            python=python, bundle=repository_root / str(paths["profile_bundle"])
        ),
        repository_root=repository_root,
    )


def ingest_profile_artifact(
    value: bytes,
    *,
    repository_root: Path,
    python: Path,
    assets_root: Path,
    architecture: str,
) -> dict[str, Any]:
    """Integrity-check, atomically install, then semantically validate one host archive."""

    validated = validate_profile_artifact(value, architecture=architecture)
    files = read_integrity_archive(value, architecture=architecture)
    root = repository_root.resolve()
    paths = host_paths(architecture)
    destination = root / str(paths["root"])
    if destination.exists():
        raise ProfileCaptureArtifactError(
            f"refusing to replace existing profile evidence: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".profile-ingest-", dir=destination.parent))
    staged_root = staging / architecture
    try:
        for name, data in sorted(files.items()):
            if name == "artifact-manifest.json":
                continue
            relative = safe_relative_path(name)
            expected_prefix = paths["root"].parts
            if relative.parts[: len(expected_prefix)] != expected_prefix:
                raise ProfileCaptureArtifactError(
                    f"archive member is outside final {architecture} root: {name}"
                )
            member = staged_root.joinpath(*relative.parts[len(expected_prefix) :])
            member.parent.mkdir(parents=True, exist_ok=True)
            member.write_bytes(data)
        os.replace(staged_root, destination)
        try:
            canonical_validate_installed_host(
                repository_root=root,
                python=python,
                assets_root=assets_root,
                architecture=architecture,
            )
        except Exception:
            os.replace(destination, staged_root)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return validated


def ingest_profile_artifacts(
    archives: Mapping[str, bytes],
    *,
    repository_root: Path,
    python: Path,
    assets_root: Path,
) -> dict[str, dict[str, Any]]:
    """Transactionally publish ARM and x86 archives, rolling both back on failure."""

    if set(archives) != {"arm64", "x86_64"}:
        raise ProfileCaptureArtifactError("dual-host ingest requires ARM64 and x86-64 archives")
    validated = {
        architecture: validate_profile_artifact(value, architecture=architecture)
        for architecture, value in archives.items()
    }
    revisions = {item["provenance"]["source"]["revision"] for item in validated.values()}
    digests = {item["provenance"]["source"]["tree_sha256"] for item in validated.values()}
    if len(revisions) != 1 or len(digests) != 1:
        raise ProfileCaptureArtifactError(
            "host archives do not attest the same clean implementation source"
        )
    unpacked = {
        architecture: read_integrity_archive(value, architecture=architecture)
        for architecture, value in archives.items()
    }
    root = repository_root.resolve()
    evidence_root = root / str(EVIDENCE_RELATIVE_ROOT)
    destinations = {
        architecture: root / str(host_paths(architecture)["root"]) for architecture in archives
    }
    existing = [str(path) for path in destinations.values() if path.exists()]
    if existing:
        raise ProfileCaptureArtifactError(
            f"refusing to replace existing profile evidence: {', '.join(existing)}"
        )
    evidence_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".profile-dual-ingest-", dir=evidence_root))
    installed: list[str] = []
    try:
        for architecture, files in unpacked.items():
            paths = host_paths(architecture)
            expected_prefix = paths["root"].parts
            staged_root = staging / architecture
            for name, data in sorted(files.items()):
                if name == "artifact-manifest.json":
                    continue
                relative = safe_relative_path(name)
                if relative.parts[: len(expected_prefix)] != expected_prefix:
                    raise ProfileCaptureArtifactError(
                        f"archive member is outside final {architecture} root: {name}"
                    )
                member = staged_root.joinpath(*relative.parts[len(expected_prefix) :])
                member.parent.mkdir(parents=True, exist_ok=True)
                member.write_bytes(data)
        for architecture in ("arm64", "x86_64"):
            os.replace(staging / architecture, destinations[architecture])
            installed.append(architecture)
        for architecture in ("arm64", "x86_64"):
            canonical_validate_installed_host(
                repository_root=root,
                python=python,
                assets_root=assets_root,
                architecture=architecture,
            )
    except Exception:
        for architecture in reversed(installed):
            destination = destinations[architecture]
            if destination.exists():
                os.replace(destination, staging / architecture)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return validated


def merge_profile_evidence_commands(*, repository_root: Path, python: Path) -> list[list[str]]:
    arm = repository_root / str(host_paths("arm64")["profile_bundle"])
    x86 = repository_root / str(host_paths("x86_64")["profile_bundle"])
    output = repository_root / str(FINAL_PROFILE_BUNDLE)
    report = repository_root / str(FINAL_PROFILE_REPORT)
    return [
        [
            str(python),
            "-m",
            "qwen_mm_reference.profile_v1",
            "merge",
            str(arm),
            str(x86),
            "--output",
            str(output),
            "--report",
            str(report),
        ],
        [str(python), "-m", "qwen_mm_reference.profile_v1", "validate", str(output)],
    ]


def merge_profile_evidence(*, repository_root: Path, python: Path, assets_root: Path) -> None:
    for architecture in ("arm64", "x86_64"):
        canonical_validate_installed_host(
            repository_root=repository_root,
            python=python,
            assets_root=assets_root,
            architecture=architecture,
        )
    for command in merge_profile_evidence_commands(repository_root=repository_root, python=python):
        _run_canonical(command, repository_root=repository_root)


def validate_final_profile_evidence(*, repository_root: Path, python: Path) -> None:
    commands = merge_profile_evidence_commands(repository_root=repository_root, python=python)
    _run_canonical(commands[-1], repository_root=repository_root)


def matrix_coordinates() -> set[tuple[str, str, int]]:
    return {
        (profile, case, budget)
        for profile in PROFILE_ALIASES
        for case in PROFILE_CASES
        for budget in THREAD_BUDGETS
    }


def command_value(command: Sequence[str], flag: str) -> str:
    """Small test/review helper for inspecting isolated CLI construction."""

    positions = [index for index, value in enumerate(command) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ProfileCaptureArtifactError(f"command does not contain one value for {flag}")
    return command[positions[0] + 1]
