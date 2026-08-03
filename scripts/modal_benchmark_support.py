"""Pure-stdlib support for the ephemeral Modal benchmark runner."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

ARTIFACT_SCHEMA_ID = "qwen-mm-modal-benchmark-artifact-v1"
ARTIFACT_SCHEMA_VERSION = 1
PROVENANCE_SCHEMA_ID = "qwen-mm-modal-benchmark-provenance-v1"
PROVENANCE_SCHEMA_VERSION = 1
BENCHMARK_SCHEMA_ID = "qwen-mm-benchmark-result-v2"
BENCHMARK_SCHEMA_VERSION = 2

REQUIRED_ARTIFACT_FILES = frozenset(
    {
        "benchmark/report.md",
        "benchmark/result.json",
        "logs/benchmark.log",
        "logs/build.log",
        "logs/validation.log",
        "provenance.json",
    }
)

_IGNORED_PARTS = frozenset(
    {
        ".git",
        ".kd",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "dist",
        "target",
    }
)


class ModalBenchmarkArtifactError(ValueError):
    """Raised when a Modal benchmark artifact is incomplete or invalid."""


def sha256_bytes(value: bytes) -> str:
    """Return a lowercase SHA-256 digest."""

    return hashlib.sha256(value).hexdigest()


def is_ignored_source_path(relative: Path) -> bool:
    """Return whether a generated path is omitted from upload and fingerprinting."""

    if any(part in _IGNORED_PARTS for part in relative.parts):
        return True
    if relative.name.startswith("_native") and relative.suffix in {".dylib", ".pyd", ".so"}:
        return True
    if relative.parts[:2] == ("benchmarks", "profile-evidence-v1"):
        return True
    return relative.parts[:2] == ("reference", ".cache")


def source_tree_digest(root: Path) -> tuple[str, int, int]:
    """Hash the exact source payload independently of mtimes and traversal order."""

    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if is_ignored_source_path(relative) or not path.is_file():
            continue
        data = path.read_bytes()
        encoded = relative.as_posix().encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
        file_count += 1
        total_bytes += len(data)
    return digest.hexdigest(), file_count, total_bytes


def committed_source_tree_digest(
    root: Path, revision: str, *, excluded: Iterable[str] = ()
) -> tuple[str, int, int]:
    """Hash the immutable files recorded by a canonical Git commit."""

    if len(revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ModalBenchmarkArtifactError("source revision is not a canonical Git object ID")
    try:
        resolved = subprocess.run(
            ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if resolved != revision:
            raise ModalBenchmarkArtifactError("source revision is not a canonical commit ID")
        archive_bytes = subprocess.run(
            ["git", "archive", "--format=tar", revision],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
            source_files: dict[str, bytes] = {}
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ModalBenchmarkArtifactError(
                        f"source tree entry cannot be read: {member.name}"
                    )
                source_files[member.name] = extracted.read()
    except ModalBenchmarkArtifactError:
        raise
    except (OSError, subprocess.CalledProcessError, tarfile.TarError) as error:
        raise ModalBenchmarkArtifactError(
            "source revision is missing or its Git tree cannot be read"
        ) from error

    excluded_paths = set(excluded)
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for name, data in sorted(source_files.items()):
        if name in excluded_paths or is_ignored_source_path(Path(name)):
            continue
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
        file_count += 1
        total_bytes += len(data)
    return digest.hexdigest(), file_count, total_bytes


def assert_native_linux_x86(*, system: str, machine: str, cpu_description: str) -> None:
    """Reject non-Linux, non-x86, and explicitly emulated CPU environments."""

    if system != "Linux":
        raise ModalBenchmarkArtifactError(f"Modal worker must report Linux, got {system!r}")
    if machine.lower() not in {"amd64", "x86_64"}:
        raise ModalBenchmarkArtifactError(
            f"Modal worker must report native x86_64, got {machine!r}"
        )
    normalized = cpu_description.lower()
    emulation_markers = ("qemu virtual cpu", "tcg cpu", "software emulation")
    marker = next((value for value in emulation_markers if value in normalized), None)
    if marker is not None:
        raise ModalBenchmarkArtifactError(
            f"Modal worker CPU description contains emulation marker {marker!r}"
        )


def _safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or path.as_posix() != name:
        raise ModalBenchmarkArtifactError(f"unsafe artifact member path: {name!r}")
    return path


def create_artifact(files: Mapping[str, bytes]) -> bytes:
    """Create a deterministic integrity-manifested ZIP from named payloads."""

    missing = sorted(REQUIRED_ARTIFACT_FILES - files.keys())
    if missing:
        raise ModalBenchmarkArtifactError(
            f"artifact payload is missing required files: {', '.join(missing)}"
        )
    if "artifact-manifest.json" in files:
        raise ModalBenchmarkArtifactError("artifact manifest is generated, not caller supplied")
    manifest_files: dict[str, dict[str, Any]] = {}
    for name, data in sorted(files.items()):
        _safe_member(name)
        if not isinstance(data, bytes):
            raise TypeError(f"artifact member {name!r} must be bytes")
        manifest_files[name] = {"bytes": len(data), "sha256": sha256_bytes(data)}
    manifest = {
        "schema_id": ARTIFACT_SCHEMA_ID,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "files": manifest_files,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
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
    return output.getvalue()


def _load_json_member(files: Mapping[str, bytes], name: str) -> dict[str, Any]:
    try:
        value = json.loads(files[name])
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ModalBenchmarkArtifactError(f"invalid JSON artifact member {name!r}") from error
    if not isinstance(value, dict):
        raise ModalBenchmarkArtifactError(f"artifact member {name!r} must contain an object")
    return value


def validate_artifact(value: bytes) -> dict[str, Any]:
    """Validate paths, manifest hashes, provenance, and benchmark identity."""

    try:
        with zipfile.ZipFile(io.BytesIO(value)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ModalBenchmarkArtifactError("artifact contains duplicate member names")
            for info in archive.infolist():
                _safe_member(info.filename)
                file_type = (info.external_attr >> 16) & 0o170000
                if file_type not in {0, 0o100000}:
                    raise ModalBenchmarkArtifactError(
                        f"artifact member is not a regular file: {info.filename!r}"
                    )
            files = {name: archive.read(name) for name in names}
    except (OSError, zipfile.BadZipFile) as error:
        raise ModalBenchmarkArtifactError("artifact is not a readable ZIP") from error
    manifest = _load_json_member(files, "artifact-manifest.json")
    if manifest.get("schema_id") != ARTIFACT_SCHEMA_ID:
        raise ModalBenchmarkArtifactError("artifact schema ID mismatch")
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ModalBenchmarkArtifactError("artifact schema version mismatch")
    declared = manifest.get("files")
    if not isinstance(declared, dict):
        raise ModalBenchmarkArtifactError("artifact manifest files must be an object")
    payload_names = set(files) - {"artifact-manifest.json"}
    if set(declared) != payload_names:
        raise ModalBenchmarkArtifactError("artifact manifest inventory mismatch")
    missing = sorted(REQUIRED_ARTIFACT_FILES - payload_names)
    if missing:
        raise ModalBenchmarkArtifactError(
            f"artifact is missing required files: {', '.join(missing)}"
        )
    for name in sorted(payload_names):
        metadata = declared.get(name)
        if not isinstance(metadata, dict):
            raise ModalBenchmarkArtifactError(f"invalid manifest metadata for {name!r}")
        if metadata.get("bytes") != len(files[name]):
            raise ModalBenchmarkArtifactError(f"artifact size mismatch for {name!r}")
        if metadata.get("sha256") != sha256_bytes(files[name]):
            raise ModalBenchmarkArtifactError(f"artifact digest mismatch for {name!r}")

    provenance = _load_json_member(files, "provenance.json")
    if provenance.get("schema_id") != PROVENANCE_SCHEMA_ID:
        raise ModalBenchmarkArtifactError("provenance schema ID mismatch")
    if provenance.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise ModalBenchmarkArtifactError("provenance schema version mismatch")
    assert_native_linux_x86(
        system=str(provenance.get("system", "")),
        machine=str(provenance.get("machine", "")),
        cpu_description=str(provenance.get("cpu_description", "")),
    )
    if provenance.get("execution") != "modal_native_linux_x86_diagnostic":
        raise ModalBenchmarkArtifactError("provenance execution label mismatch")
    if provenance.get("dedicated_host") is not False:
        raise ModalBenchmarkArtifactError("Modal diagnostic must not claim a dedicated host")
    if provenance.get("performance_claim") is not False:
        raise ModalBenchmarkArtifactError("Modal diagnostic must not make a performance claim")
    required_strings = (
        "base_image_reference",
        "base_image_tag",
        "cargo_version",
        "completed_at",
        "cpu_description",
        "modal_client_version",
        "native_module_file",
        "platform",
        "python_version",
        "rustc_version",
        "source_digest",
        "source_revision",
        "started_at",
        "uname",
        "uv_version",
        "wheel_name",
        "wheel_sha256",
    )
    for name in required_strings:
        if not isinstance(provenance.get(name), str) or not provenance[name].strip():
            raise ModalBenchmarkArtifactError(f"provenance field must be nonempty: {name}")
    if not isinstance(provenance.get("source_dirty_state"), str):
        raise ModalBenchmarkArtifactError("provenance must record source dirty state")
    for name in (
        "cgroup_limits",
        "execution_environment",
        "lscpu",
        "modal_environment",
    ):
        if not isinstance(provenance.get(name), dict):
            raise ModalBenchmarkArtifactError(f"provenance field must be an object: {name}")
    commands = provenance.get("commands")
    if (
        not isinstance(commands, list)
        or not commands
        or not all(isinstance(command, str) and command for command in commands)
    ):
        raise ModalBenchmarkArtifactError("provenance commands must be a nonempty string list")
    for name in ("build_system_packages", "python_packages"):
        packages = provenance.get(name)
        if (
            not isinstance(packages, list)
            or not packages
            or not all(isinstance(package, str) and package for package in packages)
        ):
            raise ModalBenchmarkArtifactError(
                f"provenance package inventory must be a nonempty string list: {name}"
            )
    for name in (
        "logical_cpu_count",
        "proc_meminfo_total_bytes",
        "requested_cpu_physical_cores",
        "requested_memory_mib",
        "source_bytes",
        "source_file_count",
    ):
        value = provenance.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ModalBenchmarkArtifactError(f"provenance field must be positive: {name}")
    affinity = provenance.get("visible_cpu_affinity")
    if (
        not isinstance(affinity, list)
        or not affinity
        or not all(isinstance(cpu, int) and not isinstance(cpu, bool) for cpu in affinity)
    ):
        raise ModalBenchmarkArtifactError("provenance CPU affinity must be a nonempty integer list")
    for name in ("source_digest", "wheel_sha256"):
        digest = provenance[name]
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ModalBenchmarkArtifactError(f"provenance field is not a SHA-256 digest: {name}")
    if (
        "ELF 64-bit" not in provenance["native_module_file"]
        or "x86-64" not in provenance["native_module_file"]
    ):
        raise ModalBenchmarkArtifactError("provenance native module is not x86-64 ELF")
    wheel_members = sorted(
        name for name in payload_names if name.startswith("build/") and name.endswith(".whl")
    )
    if wheel_members != [f"build/{provenance['wheel_name']}"]:
        raise ModalBenchmarkArtifactError("artifact wheel inventory does not match provenance")
    if sha256_bytes(files[wheel_members[0]]) != provenance["wheel_sha256"]:
        raise ModalBenchmarkArtifactError("artifact wheel digest does not match provenance")

    benchmark = _load_json_member(files, "benchmark/result.json")
    if benchmark.get("schema_id") != BENCHMARK_SCHEMA_ID:
        raise ModalBenchmarkArtifactError("benchmark schema ID mismatch")
    if benchmark.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise ModalBenchmarkArtifactError("benchmark schema version mismatch")
    if benchmark.get("architecture_family") != "x86_64":
        raise ModalBenchmarkArtifactError("benchmark architecture must be x86_64")
    protocol = benchmark.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("self_test_only") is not True:
        raise ModalBenchmarkArtifactError("D0 benchmark must remain self-test-only")
    eligibility = benchmark.get("release_eligibility")
    if not isinstance(eligibility, dict) or eligibility.get("releasable") is not False:
        raise ModalBenchmarkArtifactError("D0 benchmark must remain unreleasable")
    for log_name in ("logs/build.log", "logs/benchmark.log", "logs/validation.log"):
        if not files[log_name].strip():
            raise ModalBenchmarkArtifactError(f"artifact log is empty: {log_name}")
    if not files["benchmark/report.md"].startswith(b"# qwen-mm paired benchmark v2"):
        raise ModalBenchmarkArtifactError("benchmark report heading mismatch")
    return {"manifest": manifest, "provenance": provenance, "benchmark": benchmark}


def write_artifact_atomically(value: bytes, destination: Path) -> dict[str, Any]:
    """Validate before atomically replacing the requested local ZIP."""

    validated = validate_artifact(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
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


def read_files(root: Path, relative_paths: Iterable[str]) -> dict[str, bytes]:
    """Read named regular files below a trusted artifact staging directory."""

    files: dict[str, bytes] = {}
    resolved_root = root.resolve()
    for name in relative_paths:
        _safe_member(name)
        path = (resolved_root / name).resolve()
        if resolved_root not in path.parents or not path.is_file():
            raise ModalBenchmarkArtifactError(f"missing artifact staging file: {name!r}")
        files[name] = path.read_bytes()
    return files
