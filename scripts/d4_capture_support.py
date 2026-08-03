"""Strict build, affinity, noise, and artifact helpers for D4 capture runners.

This module deliberately produces raw capture bundles, not certification
decisions.  The certification evaluator owns the release schema and gates.
"""

from __future__ import annotations

import io
import json
import math
import os
import statistics
import tempfile
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from modal_benchmark_support import is_ignored_source_path, source_tree_digest
from modal_d3_support import (
    BASE_IMAGE,
    CGROUP_ATTESTATION_MODE,
    GVISOR_ATTESTATION_MODE,
    MAX_ARCHIVE_COMPRESSED_BYTES,
    MAX_ARCHIVE_MEMBER_BYTES,
    MAX_ARCHIVE_MEMBERS,
    MAX_ARCHIVE_UNCOMPRESSED_BYTES,
    MODAL_CPU,
    MODAL_MEMORY_MIB,
    RESOURCE_BINDING,
    RUST_VERSION,
    UV_VERSION,
    sha256_bytes,
    sha256_path,
)
from profile_capture_support import (
    assert_directory_identity,
    directory_identity,
    wheel_contents_identity,
)

ARTIFACT_SCHEMA_ID = "qwen-mm-d4-raw-capture-artifact-v1"
ARTIFACT_SCHEMA_VERSION = 1
BUILD_LABELS = ("shipping", "native")
THREAD_BUDGETS = (1, 2, 4, 8)
PROFILES = ("qwen3-vl-8b", "qwen3.5-9b")
PRODUCTION_THREAD_BUDGET = 8
NATIVE_RUSTFLAGS = "-C target-cpu=native"
NOISE_CV_MAX = 0.05
NOISE_MIN_OBSERVATIONS = 5
MODAL_NONPREEMPTIBLE = True
MODAL_SINGLE_USE_CONTAINER = True
PYPI_INDEX = "https://pypi.org/simple"
PYPI_OVERRIDE_ENVIRONMENT_NAMES = (
    "PIP_EXTRA_INDEX_URL",
    "PIP_INDEX_URL",
    "UV_EXTRA_INDEX_URL",
    "UV_INDEX",
    "UV_INDEX_URL",
)
BUILD_ENVIRONMENT_CAPTURE_NAMES = (
    "CARGO_HOME",
    "HF_HUB_OFFLINE",
    "HOME",
    "LC_ALL",
    "PATH",
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
    "RUSTUP_HOME",
    "SSL_CERT_FILE",
    "TMPDIR",
    "TRANSFORMERS_OFFLINE",
    "TZ",
    "UV_CACHE_DIR",
    "UV_DEFAULT_INDEX",
    "UV_PYTHON_DOWNLOADS",
)
CAPTURE_INPUT_PATHS = (
    "Cargo.lock",
    "benchmarks/performance-certification-schema-v1.json",
    "benchmarks/profile-schema-v1.json",
    "benchmarks/result-schema-v2.json",
    "benchmarks/workload-schema-v2.json",
    "benchmarks/workloads-v2.json",
    "reference/models.json",
    "reference/phase-c/v1/schema-v1.json",
    "rust-toolchain.toml",
    "uv.lock",
)

REQUIRED_BASE_MEMBERS = frozenset(
    {
        "capture-index.json",
        "provenance.json",
        "builds/native/build.json",
        "builds/shipping/build.json",
        "logs/native/build.log",
        "logs/native/matrix.log",
        "logs/native/sync.log",
        "logs/shipping/build.log",
        "logs/shipping/matrix.log",
        "logs/shipping/sync.log",
        "captures/native/repeat24_cached-unsupported.json",
        "captures/shipping/repeat24_cached-unsupported.json",
    }
)
REQUIRED_BASE_MEMBERS = frozenset(
    set(REQUIRED_BASE_MEMBERS)
    | {f"builds/{build_label}/capture.json" for build_label in BUILD_LABELS}
    | {
        f"captures/{build_label}/t{budget}/{name}"
        for build_label in BUILD_LABELS
        for budget in THREAD_BUDGETS
        for name in ("noise.json", "report.md", "result.json")
    }
    | {
        f"phase-c/{build_label}/{position}/{name}"
        for build_label in BUILD_LABELS
        for position in ("pre", "post")
        for name in ("report.json", "summary.md")
    }
)


class D4CaptureError(ValueError):
    """Raised when a D4 raw capture plan or artifact is unsafe/incomplete."""


def source_payload_identity(root: Path) -> dict[str, Any]:
    """Reuse the authenticated Modal source-payload policy from D0/D3."""

    digest, file_count, byte_count = source_tree_digest(root)
    return {"sha256": digest, "file_count": file_count, "bytes": byte_count}


def assets_identity(root: Path) -> dict[str, Any]:
    """Reuse the logical-directory identity used by D1/D3 assets."""

    return directory_identity(root)


def capture_input_identities(root: Path) -> dict[str, dict[str, Any]]:
    """Hash every repository contract consumed by capture/certification."""

    identities: dict[str, dict[str, Any]] = {}
    for name in CAPTURE_INPUT_PATHS:
        path = root / name
        if not path.is_file():
            raise D4CaptureError(f"capture contract is missing: {name}")
        identities[name] = {"bytes": path.stat().st_size, "sha256": sha256_path(path)}
    return identities


def assert_assets_identity(root: Path, expected: Mapping[str, Any]) -> None:
    """Fail closed when packaged processor assets differ from the local tree."""

    assert_directory_identity(root, expected)


def source_upload_ignored(relative: Path) -> bool:
    """Expose the existing source-upload exclusion policy to both runners."""

    return is_ignored_source_path(relative)


def build_environment(build_label: str) -> dict[str, str]:
    """Return the only build-label-specific environment allowed by D4."""

    if build_label == "shipping":
        return {}
    if build_label == "native":
        return {"RUSTFLAGS": NATIVE_RUSTFLAGS}
    raise D4CaptureError(f"unknown D4 build label: {build_label!r}")


def normalized_capture_environment(base: Mapping[str, str]) -> dict[str, str]:
    """Remove ambient build/index overrides and pin deterministic capture values."""

    environment = dict(base)
    for name in (
        "CARGO_ENCODED_RUSTFLAGS",
        "RUSTFLAGS",
        *PYPI_OVERRIDE_ENVIRONMENT_NAMES,
    ):
        environment.pop(name, None)
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "LC_ALL": "C",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TZ": "UTC",
            "UV_DEFAULT_INDEX": PYPI_INDEX,
        }
    )
    return environment


def build_environment_evidence(environment: Mapping[str, str]) -> dict[str, Any]:
    """Record the complete controlled build environment without host credentials."""

    forbidden = (
        "CARGO_ENCODED_RUSTFLAGS",
        "RUSTFLAGS",
        *PYPI_OVERRIDE_ENVIRONMENT_NAMES,
    )
    if any(name in environment for name in forbidden):
        raise D4CaptureError("build environment contains a forbidden ambient override")
    return {
        "captured_names": list(BUILD_ENVIRONMENT_CAPTURE_NAMES),
        "set": {
            name: environment[name]
            for name in BUILD_ENVIRONMENT_CAPTURE_NAMES
            if name in environment
        },
        "unset": list(forbidden),
        "uncaptured_ambient_policy": "credentials_and_unrelated_host_variables_excluded",
    }


def assert_build_invariants(values: Mapping[str, Mapping[str, Any]]) -> None:
    """Require both variants to observe one package/toolchain/host environment."""

    if set(values) != set(BUILD_LABELS):
        raise D4CaptureError("build invariant evidence must cover shipping and native")
    if values["shipping"] != values["native"]:
        raise D4CaptureError(
            "shipping/native build toolchain, packages, host, or environment changed"
        )


def normalize_build_artifact_paths(value: str, replacements: Mapping[Path, str]) -> str:
    """Normalize only declared variant paths, including macOS symlink resolution."""

    normalized = value
    for path, marker in replacements.items():
        candidates = {str(path), str(path.resolve())}
        for candidate in sorted(candidates, key=len, reverse=True):
            normalized = normalized.replace(candidate, marker)
    return normalized


def reference_sync_command(*, venv: Path) -> list[str]:
    """Sync only the locked reference workspace package into a clean active venv."""

    return [
        "env",
        f"VIRTUAL_ENV={venv}",
        "uv",
        "sync",
        "--locked",
        "--package",
        "qwen-mm-reference",
        "--active",
        "--inexact",
        "--default-index",
        PYPI_INDEX,
    ]


def reference_sync_evidence(
    *, venv: Path, environment: Mapping[str, str], log: Path
) -> dict[str, Any]:
    """Record the exact locked-sync command, relevant environment, and log identity."""

    if environment.get("UV_DEFAULT_INDEX") != PYPI_INDEX or any(
        name in environment for name in PYPI_OVERRIDE_ENVIRONMENT_NAMES
    ):
        raise D4CaptureError("reference sync environment did not normalize package indexes")
    if not log.is_file():
        raise D4CaptureError("reference sync log is missing")
    return {
        "command": reference_sync_command(venv=venv),
        "environment": {
            "command_overrides": {"VIRTUAL_ENV": str(venv)},
            "process": {
                name: environment[name]
                for name in (
                    "HF_HUB_OFFLINE",
                    "LC_ALL",
                    "PATH",
                    "PYTHONHASHSEED",
                    "PYTHONNOUSERSITE",
                    "TRANSFORMERS_OFFLINE",
                    "TZ",
                    "UV_DEFAULT_INDEX",
                )
            },
            "removed": list(PYPI_OVERRIDE_ENVIRONMENT_NAMES),
        },
        "log": {
            "bytes": log.stat().st_size,
            "sha256": sha256_path(log),
        },
    }


def wheel_build_command(
    *, python: Path, output: Path, build_label: str, cargo_target_dir: Path
) -> list[str]:
    """Create a normal locked release build, with one native-only override."""

    environment = {
        "CARGO_TARGET_DIR": str(cargo_target_dir),
        **build_environment(build_label),
    }
    command = [
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
        str(output),
    ]
    return ["env", *(f"{name}={value}" for name, value in environment.items()), *command]


def installed_build_identity(*, python: Path, wheel: Path) -> dict[str, Any]:
    """Bind a wheel archive identity to the extracted extension it contains."""

    contents = wheel_contents_identity(wheel)
    return {
        "python": str(python),
        "wheel": {
            "name": wheel.name,
            "bytes": wheel.stat().st_size,
            "sha256": sha256_path(wheel),
            "contents": contents,
        },
    }


def reconcile_installed_runtime(*, wheel: Path, runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Prove the loaded extension is byte-for-byte the extension retained in the wheel."""

    contents = wheel_contents_identity(wheel)
    expected_sha256 = contents["native_artifact_sha256"]
    expected_bytes = contents["native_bytes"]
    if (
        runtime.get("native_sha256") != expected_sha256
        or runtime.get("native_bytes") != expected_bytes
    ):
        raise D4CaptureError("installed native module differs from the retained wheel")
    return {
        "verified": True,
        "wheel_contents": contents,
        "installed_runtime": dict(runtime),
    }


def parse_cpu_list(value: str) -> set[int]:
    """Parse a Linux cpuset such as ``0-3,8,10-11``."""

    cpus: set[int] = set()
    if not value.strip():
        raise D4CaptureError("CPU list is empty")
    try:
        for group in value.split(","):
            start_text, separator, end_text = group.strip().partition("-")
            start = int(start_text)
            end = int(end_text) if separator else start
            if start < 0 or end < start:
                raise ValueError
            cpus.update(range(start, end + 1))
    except ValueError as error:
        raise D4CaptureError(f"invalid CPU list: {value!r}") from error
    return cpus


def physical_core_masks(
    lscpu_parse: str,
    *,
    allowed_cpus: Iterable[int],
    budgets: Sequence[int] = THREAD_BUDGETS,
) -> dict[int, tuple[int, ...]]:
    """Choose nested one-thread-per-physical-core masks from ``lscpu -p``.

    Socket and core IDs form the physical-core key.  Sibling logical CPUs are
    never selected into the same mask.
    """

    allowed = set(allowed_cpus)
    if not allowed:
        raise D4CaptureError("worker exposes no allowed CPUs")
    physical: dict[tuple[int, int], int] = {}
    for raw_line in lscpu_parse.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        columns = line.split(",")
        if len(columns) != 4:
            raise D4CaptureError("lscpu topology row must contain CPU,CORE,SOCKET,ONLINE")
        try:
            cpu, core, socket_id = (int(columns[index]) for index in range(3))
        except ValueError as error:
            raise D4CaptureError(f"invalid lscpu topology row: {line!r}") from error
        if columns[3].strip().lower() not in {"y", "yes", "1", "true"} or cpu not in allowed:
            continue
        key = (socket_id, core)
        physical[key] = min(cpu, physical.get(key, cpu))
    representatives = tuple(sorted(physical.values()))
    requested = tuple(budgets)
    if (
        not requested
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in requested
        )
        or tuple(sorted(set(requested))) != requested
    ):
        raise D4CaptureError("thread budgets must be unique increasing positive integers")
    if len(representatives) < requested[-1]:
        raise D4CaptureError(
            f"worker exposes {len(representatives)} physical cores; {requested[-1]} required"
        )
    return {budget: representatives[:budget] for budget in requested}


def local_affinity_provenance() -> dict[str, Any]:
    """State the macOS limitation explicitly; never invent a CPU mask."""

    return {
        "mode": "unavailable",
        "reason": "macOS has no supported taskset/sched_setaffinity equivalent",
        "masks": {f"t{budget}": None for budget in THREAD_BUDGETS},
    }


def modal_resource_attestation(
    *,
    cgroup_limits: Mapping[str, str],
    platform_text: str,
    uname: str,
    visible_affinity: Sequence[int],
    masks: Mapping[int, Sequence[int]],
) -> dict[str, Any]:
    """Apply D3's cgroup/gVisor resource modes to D4's complete CPU masks."""

    expected_visible = set(visible_affinity)
    if len(expected_visible) < int(MODAL_CPU):
        raise D4CaptureError("Modal affinity is smaller than the 16-CPU request")
    expected_masks = set(THREAD_BUDGETS)
    if set(masks) != expected_masks:
        raise D4CaptureError("Modal affinity masks do not cover t1/t2/t4/t8")
    for budget, mask in masks.items():
        if (
            len(mask) != budget
            or len(set(mask)) != budget
            or not set(mask).issubset(expected_visible)
        ):
            raise D4CaptureError(f"invalid Modal t{budget} physical-core mask")
    if cgroup_limits:
        required = {
            "/sys/fs/cgroup/cpu.max",
            "/sys/fs/cgroup/cpuset.cpus.effective",
            "/sys/fs/cgroup/memory.max",
        }
        if not required.issubset(cgroup_limits):
            raise D4CaptureError("partial cgroup v2 evidence is not sufficient")
        try:
            quota_text, period_text = cgroup_limits["/sys/fs/cgroup/cpu.max"].split()
            period = int(period_text)
            quota = None if quota_text == "max" else int(quota_text)
            memory_text = cgroup_limits["/sys/fs/cgroup/memory.max"]
            memory_limit = None if memory_text == "max" else int(memory_text)
        except (TypeError, ValueError) as error:
            raise D4CaptureError("cgroup CPU or memory limit is invalid") from error
        if period <= 0 or (quota is not None and quota / period < MODAL_CPU):
            raise D4CaptureError("cgroup CPU quota is smaller than the Modal request")
        if memory_limit is not None and memory_limit < MODAL_MEMORY_MIB * 1024 * 1024:
            raise D4CaptureError("cgroup memory limit is smaller than the Modal request")
        cpuset = parse_cpu_list(cgroup_limits["/sys/fs/cgroup/cpuset.cpus.effective"])
        if len(cpuset) < int(MODAL_CPU) or not expected_visible.issubset(cpuset):
            raise D4CaptureError("cgroup effective cpuset is smaller than visible affinity")
        mode = CGROUP_ATTESTATION_MODE
    else:
        combined = f"{platform_text} {uname}".lower()
        if "gvisor" not in combined:
            raise D4CaptureError("missing cgroups without an authenticated gVisor fallback")
        mode = GVISOR_ATTESTATION_MODE
    return {
        "mode": mode,
        "requested_resources_bound_by": RESOURCE_BINDING,
        "requested_physical_cores": MODAL_CPU,
        "requested_memory_mib": MODAL_MEMORY_MIB,
        "nonpreemptible": MODAL_NONPREEMPTIBLE,
        "single_use_container": MODAL_SINGLE_USE_CONTAINER,
        "visible_affinity": list(visible_affinity),
        "physical_core_masks": {f"t{budget}": list(masks[budget]) for budget in THREAD_BUDGETS},
        "cgroup_limits": dict(cgroup_limits),
    }


def noise_assessment(values: Sequence[float]) -> dict[str, Any]:
    """Evaluate the frozen no-pruning CV rule over all supplied observations."""

    if len(values) < NOISE_MIN_OBSERVATIONS:
        raise D4CaptureError(
            f"noise assessment requires at least {NOISE_MIN_OBSERVATIONS} observations"
        )
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
        for value in values
    ):
        raise D4CaptureError("noise observations must be finite positive numbers")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    cv = standard_deviation / mean
    return {
        "rule": "sample standard deviation / mean <= 0.05",
        "threshold": NOISE_CV_MAX,
        "observation_count": len(values),
        "retained_observation_count": len(values),
        "pruned_observation_count": 0,
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "cv": cv,
        "pass": cv <= NOISE_CV_MAX,
    }


def _safe_name(name: str) -> None:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or path.as_posix() != name:
        raise D4CaptureError(f"unsafe artifact path: {name!r}")


def _check_archive_limits(files: Mapping[str, bytes]) -> None:
    if len(files) + 1 > MAX_ARCHIVE_MEMBERS:
        raise D4CaptureError("raw capture artifact contains too many members")
    total = 0
    for name, value in files.items():
        _safe_name(name)
        if not isinstance(value, bytes):
            raise TypeError(f"artifact member must be bytes: {name}")
        if len(value) > MAX_ARCHIVE_MEMBER_BYTES:
            raise D4CaptureError(f"artifact member exceeds size limit: {name}")
        total += len(value)
    if total > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
        raise D4CaptureError("raw capture artifact exceeds uncompressed-size limit")


def create_capture_archive(files: Mapping[str, bytes]) -> bytes:
    """Create a bounded deterministic archive with D3-strength integrity data."""

    missing = sorted(REQUIRED_BASE_MEMBERS - files.keys())
    if missing:
        raise D4CaptureError(f"raw capture artifact is incomplete: {missing}")
    if "artifact-manifest.json" in files:
        raise D4CaptureError("artifact manifest is generated, not caller supplied")
    _check_archive_limits(files)
    manifest = {
        "schema_id": ARTIFACT_SCHEMA_ID,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "files": {
            name: {"bytes": len(value), "sha256": sha256_bytes(value)}
            for name, value in sorted(files.items())
        },
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, value in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, value)
        info = zipfile.ZipInfo("artifact-manifest.json", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    value = output.getvalue()
    if len(value) > MAX_ARCHIVE_COMPRESSED_BYTES:
        raise D4CaptureError("raw capture artifact exceeds compressed-size limit")
    return value


def read_capture_archive(value: bytes) -> dict[str, bytes]:
    """Validate archive bounds, paths, hashes, build labels, and raw inventory."""

    if len(value) > MAX_ARCHIVE_COMPRESSED_BYTES:
        raise D4CaptureError("raw capture artifact exceeds compressed-size limit")
    try:
        with zipfile.ZipFile(io.BytesIO(value)) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_ARCHIVE_MEMBERS or len({info.filename for info in infos}) != len(
                infos
            ):
                raise D4CaptureError("raw capture archive member count or uniqueness is invalid")
            total = 0
            for info in infos:
                _safe_name(info.filename)
                file_type = (info.external_attr >> 16) & 0o170000
                if file_type not in {0, 0o100000} or info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                    raise D4CaptureError(f"unsafe raw capture member: {info.filename}")
                total += info.file_size
            if total > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                raise D4CaptureError("raw capture artifact exceeds uncompressed-size limit")
            files = {info.filename: archive.read(info.filename) for info in infos}
    except (OSError, zipfile.BadZipFile) as error:
        raise D4CaptureError("raw capture artifact is not a readable ZIP") from error
    try:
        manifest = json.loads(files.pop("artifact-manifest.json"))
        index = json.loads(files["capture-index.json"])
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise D4CaptureError("raw capture manifest/index is missing or invalid") from error
    if (
        manifest.get("schema_id") != ARTIFACT_SCHEMA_ID
        or manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or set(manifest.get("files", {})) != set(files)
    ):
        raise D4CaptureError("raw capture manifest identity or inventory mismatch")
    for name, data in files.items():
        if manifest["files"].get(name) != {"bytes": len(data), "sha256": sha256_bytes(data)}:
            raise D4CaptureError(f"raw capture member identity mismatch: {name}")
    missing = sorted(REQUIRED_BASE_MEMBERS - files.keys())
    if missing:
        raise D4CaptureError(f"raw capture artifact is incomplete: {missing}")
    if index.get("build_labels") != list(BUILD_LABELS) or index.get("thread_budgets") != list(
        THREAD_BUDGETS
    ):
        raise D4CaptureError("raw capture index changed the frozen build/thread matrix")
    declared = index.get("files")
    if not isinstance(declared, list) or set(declared) != set(files) - {"capture-index.json"}:
        raise D4CaptureError("raw capture index inventory mismatch")
    if any(token in name.lower() for name in files for token in ("pruned", "filtered", "dropped")):
        raise D4CaptureError("raw capture artifact may not package pruned samples")
    for build_label in BUILD_LABELS:
        name = f"captures/{build_label}/repeat24_cached-unsupported.json"
        try:
            unsupported = json.loads(files[name])
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise D4CaptureError("cached unsupported record is invalid") from error
        expected_coordinates = {
            (profile, budget) for profile in PROFILES for budget in (1, PRODUCTION_THREAD_BUDGET)
        }
        coordinates = unsupported.get("coordinates")
        if (
            unsupported.get("schema_id") != "qwen-mm-d4-cached-unsupported-v1"
            or unsupported.get("build_label") != build_label
            or unsupported.get("case_id") != "repeat24_cached"
            or unsupported.get("cache_mode") != "enabled"
            or unsupported.get("support_status") != "unsupported"
            or unsupported.get("support_reason") != "adapter_cache_supported_false"
            or unsupported.get("timing_kind") != "unsupported"
            or unsupported.get("timed_pair_count") != 0
            or unsupported.get("timed_sample_count") != 0
            or not isinstance(coordinates, list)
            or {
                (coordinate.get("profile_alias"), coordinate.get("thread_budget"))
                for coordinate in coordinates
                if isinstance(coordinate, Mapping)
            }
            != expected_coordinates
            or any(
                not isinstance(coordinate, Mapping)
                or coordinate.get("reference_cache_supported") is not False
                or coordinate.get("candidate_cache_supported") is not False
                for coordinate in coordinates
            )
        ):
            raise D4CaptureError("cached row is not proven unsupported with zero timing")
        for budget in THREAD_BUDGETS:
            noise_name = f"captures/{build_label}/t{budget}/noise.json"
            result_name = f"captures/{build_label}/t{budget}/result.json"
            try:
                noise = json.loads(files[noise_name])
                result = json.loads(files[result_name])
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise D4CaptureError("raw benchmark/noise payload is invalid") from error
            assessments = noise.get("assessments")
            if (
                noise.get("sample_pruning") != "forbidden"
                or noise.get("pass") is not True
                or not isinstance(assessments, list)
                or not assessments
                or any(
                    not isinstance(assessment, Mapping)
                    or assessment.get("pass") is not True
                    or assessment.get("observation_count")
                    != assessment.get("retained_observation_count")
                    or assessment.get("pruned_observation_count") != 0
                    for assessment in assessments
                )
            ):
                raise D4CaptureError("raw capture failed the frozen no-pruning noise rule")
            pairs = result.get("pairs")
            if not isinstance(pairs, list) or any(
                isinstance(pair, Mapping) and pair.get("case_id") == "repeat24_cached"
                for pair in pairs
            ):
                raise D4CaptureError("repeat24_cached was timed in a raw benchmark result")
    return files


def write_capture_archive(
    value: bytes,
    destination: Path,
    *,
    expected_source: Mapping[str, Any] | None = None,
    expected_assets: Mapping[str, Any] | None = None,
) -> None:
    """Validate and atomically persist a raw capture archive."""

    files = read_capture_archive(value)
    if expected_source is not None or expected_assets is not None:
        try:
            provenance = json.loads(files["provenance.json"])
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise D4CaptureError("raw capture provenance is invalid") from error
        if expected_source is not None and provenance.get("source") != dict(expected_source):
            raise D4CaptureError("remote source identity differs from the local upload")
        if expected_assets is not None and provenance.get("assets") != dict(expected_assets):
            raise D4CaptureError("remote asset identity differs from the local upload")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def toolchain_pins() -> dict[str, Any]:
    """Return inherited D3 image/resource pins for provenance and tests."""

    return {
        "base_image": BASE_IMAGE,
        "rust": RUST_VERSION,
        "uv": UV_VERSION,
        "modal_cpu": MODAL_CPU,
        "modal_memory_mib": MODAL_MEMORY_MIB,
    }
