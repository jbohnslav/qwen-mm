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
from datetime import datetime
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
D4_TIMED_CASES = (
    "text_short",
    "text_long",
    "image1",
    "image24",
    "jpeg24_requests",
    "ragged24",
    "aligned24",
    "minmax_boundaries",
    "rgb24",
    "repeat24_uncached",
    "repeat24_separated",
    "images_1",
    "images_4",
    "images_16",
    "images_32",
    "images_64",
)
PRODUCTION_THREAD_BUDGET = 8
D4_RANDOM_SEED = 20260731
NATIVE_RUSTFLAGS = "-C target-cpu=native"
NOISE_CV_MAX = 0.05
NOISE_MIN_OBSERVATIONS = 5
LOCAL_LINUX_POWER_POLICY_SUFFIXES = {
    "scaling_governor": "/scaling_governor",
    "energy_performance_preference": "/energy_performance_preference",
    "scaling_min_freq": "/scaling_min_freq",
    "scaling_max_freq": "/scaling_max_freq",
    "intel_pstate_no_turbo": "/intel_pstate/no_turbo",
    "cpufreq_boost": "/cpufreq/boost",
}
LOCAL_LINUX_AFFINITY_PROBE_THREADS = (1, 2, 4)
LOCAL_LINUX_AFFINITY_PROBE_SECONDS = 1.0
LOCAL_LINUX_CPU_WALL_RATIO_MAX = 1.25
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


def wheel_benchmark_artifact_identity(wheel: Path) -> dict[str, Any]:
    """Hash the exact adapter wrapper executed from a retained D4 wheel."""

    try:
        with zipfile.ZipFile(wheel) as archive:
            members = [name for name in archive.namelist() if name == "qwen_mm/benchmark.py"]
            if len(members) != 1:
                raise D4CaptureError(
                    "D4 wheel must contain exactly one qwen_mm/benchmark.py adapter"
                )
            value = archive.read(members[0])
    except (OSError, zipfile.BadZipFile) as error:
        raise D4CaptureError("D4 retained wheel is unreadable") from error
    return {"member": members[0], "bytes": len(value), "sha256": sha256_bytes(value)}


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


def physical_core_representatives(
    lscpu_parse: str, *, allowed_cpus: Iterable[int]
) -> tuple[int, ...]:
    """Return one allowed logical CPU for each physical socket/core pair."""

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
    return tuple(sorted(physical.values()))


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

    representatives = physical_core_representatives(lscpu_parse, allowed_cpus=allowed_cpus)
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


def validate_local_linux_affinity_enforcement(
    value: Mapping[str, Any], *, expected_affinity: Sequence[int]
) -> dict[str, Any]:
    """Validate the frozen pre/post native-thread affinity enforcement probe."""

    fields = {
        "method",
        "expected_affinity",
        "probe_seconds",
        "cpu_wall_ratio_max",
        "pre",
        "post",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise D4CaptureError("local Linux affinity-enforcement record has an invalid shape")
    expected = list(expected_affinity)
    if (
        len(expected) != 1
        or any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in expected)
        or value["method"] != "taskset-t1-python-native-threads-v1"
        or value["expected_affinity"] != expected
        or value["probe_seconds"] != LOCAL_LINUX_AFFINITY_PROBE_SECONDS
        or value["cpu_wall_ratio_max"] != LOCAL_LINUX_CPU_WALL_RATIO_MAX
    ):
        raise D4CaptureError("local Linux affinity-enforcement controls changed")

    normalized_phases: dict[str, Any] = {}
    probe_fields = {
        "native_thread_count",
        "wall_seconds",
        "process_cpu_seconds",
        "process_cpu_wall_ratio",
        "worker_reports",
    }
    worker_fields = {
        "worker_index",
        "native_thread_id",
        "affinity",
        "observed_cpus",
        "iterations",
    }
    for phase in ("pre", "post"):
        raw_phase = value[phase]
        if not isinstance(raw_phase, Mapping) or set(raw_phase) != {"probes"}:
            raise D4CaptureError(f"local Linux affinity-enforcement {phase} shape is invalid")
        probes = raw_phase["probes"]
        if not isinstance(probes, list) or len(probes) != len(LOCAL_LINUX_AFFINITY_PROBE_THREADS):
            raise D4CaptureError(
                f"local Linux affinity-enforcement {phase} probe inventory is incomplete"
            )
        normalized_probes: list[dict[str, Any]] = []
        for raw_probe, thread_count in zip(probes, LOCAL_LINUX_AFFINITY_PROBE_THREADS, strict=True):
            if not isinstance(raw_probe, Mapping) or set(raw_probe) != probe_fields:
                raise D4CaptureError(
                    f"local Linux affinity-enforcement {phase} probe shape is invalid"
                )
            if raw_probe["native_thread_count"] != thread_count:
                raise D4CaptureError(
                    f"local Linux affinity-enforcement {phase} thread matrix changed"
                )
            timing: dict[str, float] = {}
            for name in (
                "wall_seconds",
                "process_cpu_seconds",
                "process_cpu_wall_ratio",
            ):
                raw_timing = raw_probe[name]
                if (
                    isinstance(raw_timing, bool)
                    or not isinstance(raw_timing, (int, float))
                    or not math.isfinite(raw_timing)
                    or raw_timing <= 0
                ):
                    raise D4CaptureError(
                        f"local Linux affinity-enforcement {phase} timing is invalid"
                    )
                timing[name] = float(raw_timing)
            if timing["wall_seconds"] < LOCAL_LINUX_AFFINITY_PROBE_SECONDS:
                raise D4CaptureError(f"local Linux affinity-enforcement {phase} probe is too short")
            recomputed_ratio = timing["process_cpu_seconds"] / timing["wall_seconds"]
            if not math.isclose(
                timing["process_cpu_wall_ratio"],
                recomputed_ratio,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise D4CaptureError(
                    f"local Linux affinity-enforcement {phase} CPU/wall ratio is stale"
                )
            if timing["process_cpu_wall_ratio"] > LOCAL_LINUX_CPU_WALL_RATIO_MAX:
                raise D4CaptureError(
                    f"local Linux affinity-enforcement {phase} exceeded CPU/wall limit"
                )
            reports = raw_probe["worker_reports"]
            if not isinstance(reports, list) or len(reports) != thread_count:
                raise D4CaptureError(
                    f"local Linux affinity-enforcement {phase} worker inventory is incomplete"
                )
            native_ids: set[int] = set()
            normalized_reports: list[dict[str, Any]] = []
            for worker_index, report in enumerate(reports):
                if not isinstance(report, Mapping) or set(report) != worker_fields:
                    raise D4CaptureError(
                        f"local Linux affinity-enforcement {phase} worker shape is invalid"
                    )
                native_id = report["native_thread_id"]
                iterations = report["iterations"]
                if (
                    report["worker_index"] != worker_index
                    or isinstance(native_id, bool)
                    or not isinstance(native_id, int)
                    or native_id <= 0
                    or native_id in native_ids
                    or isinstance(iterations, bool)
                    or not isinstance(iterations, int)
                    or iterations <= 0
                    or report["affinity"] != expected
                    or report["observed_cpus"] != expected
                ):
                    raise D4CaptureError(
                        f"local Linux affinity-enforcement {phase} worker report is invalid"
                    )
                native_ids.add(native_id)
                normalized_reports.append(dict(report))
            normalized_probes.append(
                {
                    "native_thread_count": thread_count,
                    **timing,
                    "worker_reports": normalized_reports,
                }
            )
        normalized_phases[phase] = {"probes": normalized_probes}
    return {
        "method": value["method"],
        "expected_affinity": expected,
        "probe_seconds": LOCAL_LINUX_AFFINITY_PROBE_SECONDS,
        "cpu_wall_ratio_max": LOCAL_LINUX_CPU_WALL_RATIO_MAX,
        **normalized_phases,
    }


def local_linux_resource_attestation(
    *,
    cgroup_limits: Mapping[str, str],
    allocated_physical_cores: int,
    allocated_memory_bytes: int,
    power_policy: Mapping[str, Mapping[str, str]],
    exclusive_physical_cores: bool,
    dedicated_capture: bool,
    stable: bool,
    visible_affinity: Sequence[int],
    masks: Mapping[int, Sequence[int]],
    affinity_enforcement: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a controlled, fixed local-Linux cgroup allocation.

    A Docker cpuset enforces this capture's affinity but does not prove host-
    level physical-core exclusivity, so that claim must remain false.
    ``dedicated_capture`` asserts no other benchmark work ran in the capture
    allocation, while ``stable`` asserts that its resource limits and the host
    power/performance policy remained unchanged for the complete capture.
    """

    if exclusive_physical_cores is not False:
        raise D4CaptureError("local Linux Docker capture cannot claim exclusive physical cores")
    if dedicated_capture is not True or stable is not True:
        raise D4CaptureError(
            "local Linux capture requires a dedicated capture and stable host controls"
        )
    visible = list(visible_affinity)
    if (
        not visible
        or len(visible) != len(set(visible))
        or any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in visible)
    ):
        raise D4CaptureError("local Linux visible affinity is invalid")
    if not isinstance(power_policy, Mapping) or set(power_policy) != {"paths", "unavailable"}:
        raise D4CaptureError("local Linux power-policy snapshot has an invalid shape")
    paths = power_policy["paths"]
    unavailable = power_policy["unavailable"]
    if not isinstance(paths, Mapping) or not isinstance(unavailable, Mapping):
        raise D4CaptureError("local Linux power-policy records must be objects")
    if set(unavailable) - set(LOCAL_LINUX_POWER_POLICY_SUFFIXES):
        raise D4CaptureError("local Linux power-policy snapshot has unknown endpoint groups")
    for path, value in paths.items():
        if (
            not isinstance(path, str)
            or not path.startswith("/sys/devices/system/cpu/")
            or PurePosixPath(path).as_posix() != path
            or ".." in PurePosixPath(path).parts
            or not any(
                path.endswith(suffix) for suffix in LOCAL_LINUX_POWER_POLICY_SUFFIXES.values()
            )
            or not isinstance(value, str)
            or not value.strip()
        ):
            raise D4CaptureError("local Linux power-policy path/value is invalid")
    for group, suffix in LOCAL_LINUX_POWER_POLICY_SUFFIXES.items():
        matching = [path for path in paths if path.endswith(suffix)]
        observed = bool(matching)
        missing = group in unavailable
        if observed == missing:
            raise D4CaptureError(
                f"local Linux power-policy group must be observed or unavailable: {group}"
            )
        if missing and (not isinstance(unavailable[group], str) or not unavailable[group].strip()):
            raise D4CaptureError(f"local Linux power-policy unavailable reason is invalid: {group}")
        if (
            group
            in {
                "scaling_governor",
                "energy_performance_preference",
                "scaling_min_freq",
                "scaling_max_freq",
            }
            and observed
        ):
            cpu_prefix = "/sys/devices/system/cpu/cpu"
            observed_cpus: set[int] = set()
            for path in matching:
                cpu_text, separator, endpoint = path[len(cpu_prefix) :].partition("/cpufreq/")
                if not separator or not cpu_text.isdigit() or endpoint != group:
                    raise D4CaptureError(
                        f"local Linux per-CPU power-policy path is invalid: {path}"
                    )
                observed_cpus.add(int(cpu_text))
            if observed_cpus != set(visible):
                raise D4CaptureError(
                    f"local Linux power-policy group does not cover visible CPUs: {group}"
                )
        if group == "intel_pstate_no_turbo" and matching not in (
            [],
            ["/sys/devices/system/cpu/intel_pstate/no_turbo"],
        ):
            raise D4CaptureError("local Linux no-turbo power-policy path is invalid")
        if group == "cpufreq_boost" and matching not in (
            [],
            ["/sys/devices/system/cpu/cpufreq/boost"],
        ):
            raise D4CaptureError("local Linux boost power-policy path is invalid")
    if (
        isinstance(allocated_physical_cores, bool)
        or not isinstance(allocated_physical_cores, int)
        or allocated_physical_cores < max(THREAD_BUDGETS)
    ):
        raise D4CaptureError("local Linux allocation exposes fewer than 8 physical cores")
    if (
        isinstance(allocated_memory_bytes, bool)
        or not isinstance(allocated_memory_bytes, int)
        or allocated_memory_bytes <= 0
    ):
        raise D4CaptureError("local Linux allocated memory must be a positive byte count")
    if set(masks) != set(THREAD_BUDGETS):
        raise D4CaptureError("local Linux affinity masks do not cover t1/t2/t4/t8")
    prior: list[int] = []
    for budget in THREAD_BUDGETS:
        mask = list(masks[budget])
        if (
            len(mask) != budget
            or len(mask) != len(set(mask))
            or any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in mask)
            or not set(mask).issubset(visible)
            or (prior and mask[: len(prior)] != prior)
        ):
            raise D4CaptureError(f"invalid local Linux t{budget} physical-core mask")
        prior = mask

    required = {
        "/sys/fs/cgroup/cpu.max",
        "/sys/fs/cgroup/cpuset.cpus.effective",
        "/sys/fs/cgroup/memory.max",
    }
    if set(cgroup_limits) != required:
        raise D4CaptureError("local Linux capture requires the complete cgroup v2 control set")
    try:
        quota_text, period_text = cgroup_limits["/sys/fs/cgroup/cpu.max"].split()
        period = int(period_text)
        quota = None if quota_text == "max" else int(quota_text)
        memory_text = cgroup_limits["/sys/fs/cgroup/memory.max"]
        memory_limit = None if memory_text == "max" else int(memory_text)
    except (TypeError, ValueError) as error:
        raise D4CaptureError("local Linux cgroup CPU or memory limit is invalid") from error
    if period <= 0 or quota is None or quota / period < allocated_physical_cores:
        raise D4CaptureError("local Linux CPU quota is smaller than its physical-core allocation")
    if memory_limit is None or memory_limit != allocated_memory_bytes:
        raise D4CaptureError("local Linux memory limit differs from its allocated memory")
    cpuset = parse_cpu_list(cgroup_limits["/sys/fs/cgroup/cpuset.cpus.effective"])
    if cpuset != set(visible):
        raise D4CaptureError("local Linux effective cpuset differs from visible affinity")
    result = {
        "mode": "cgroup_v2_cpuset",
        "allocated_physical_cores": allocated_physical_cores,
        "allocated_memory_bytes": allocated_memory_bytes,
        "power_policy": {
            "paths": dict(paths),
            "unavailable": dict(unavailable),
        },
        "exclusive_physical_cores": False,
        "dedicated_capture": True,
        "stable": True,
        "visible_affinity": visible,
        "physical_core_masks": {f"t{budget}": list(masks[budget]) for budget in THREAD_BUDGETS},
        "cgroup_limits": dict(cgroup_limits),
    }
    if affinity_enforcement is not None:
        result["affinity_enforcement"] = validate_local_linux_affinity_enforcement(
            affinity_enforcement,
            expected_affinity=masks[min(THREAD_BUDGETS)],
        )
    return result


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


def result_noise_assessment(result: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the complete no-pruning CV evidence from raw process medians."""

    groups: dict[tuple[str, str, str], list[float]] = {}
    pairs = result.get("pairs")
    if not isinstance(pairs, list):
        raise D4CaptureError("benchmark result has no raw pairs")
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise D4CaptureError("benchmark pair is invalid")
        for implementation in ("reference", "candidate"):
            try:
                value = float(pair["implementations"][implementation]["summary"]["wall_ms"]["p50"])
                key = (str(pair["profile_alias"]), str(pair["case_id"]), implementation)
            except (KeyError, TypeError, ValueError) as error:
                raise D4CaptureError("benchmark pair lacks a raw process median") from error
            groups.setdefault(key, []).append(value)
    assessments = []
    for (profile, case, implementation), values in sorted(groups.items()):
        assessments.append(
            {
                "profile_alias": profile,
                "case_id": case,
                "implementation": implementation,
                **noise_assessment(values),
            }
        )
    return {
        "rule_frozen_before_capture": True,
        "sample_pruning": "forbidden",
        "assessments": assessments,
        "pass": all(item["pass"] for item in assessments),
    }


def validate_d4_result_contract(
    result: Mapping[str, Any],
    *,
    architecture: str,
    build_label: str,
    budget: int,
    affinity_cpus: Sequence[int] | None,
) -> None:
    """Require one raw benchmark to match its complete frozen D4 coordinate."""

    protocol = result.get("protocol")
    expected_cases = ["image24"] if budget in {2, 4} else list(D4_TIMED_CASES)
    expected_affinity = None if affinity_cpus is None else list(affinity_cpus)
    if not isinstance(protocol, Mapping):
        raise D4CaptureError("raw benchmark protocol is missing")
    expected = {
        "reference_adapter": "official",
        "candidate_adapter": "qwen_mm.benchmark:create_adapter",
        "profiles": list(PROFILES),
        "cases": expected_cases,
        "process_repetitions": 5,
        "random_seed": D4_RANDOM_SEED,
        "order": "randomized AB/BA per process repetition",
        "warmups": 3,
        "minimum_samples": 30,
        "minimum_seconds": 5.0,
        "thread_regimes": [f"t{budget}"],
        "thread_budget_mapping": {f"t{budget}": budget},
        "affinity_cpu_mapping": {f"t{budget}": expected_affinity},
        "build_labels": [build_label],
        "timing_protocol": "instrumentation-free-v1",
        "timing_sample_fields": [
            "sequence",
            "wall_ms",
            "cpu_ms",
            "throughput_per_s",
            "core_utilization",
        ],
        "timing_floor_policy": (
            "minimum_samples one-operation latency samples; supplemental individually clocked "
            "exact-stability operations aggregate only to minimum_seconds and are excluded from "
            "latency distributions"
        ),
        "resource_census_position": "after_all_timed_samples",
        "production_thread_budget": PRODUCTION_THREAD_BUDGET,
    }
    if result.get("mode") != "dedicated" or any(
        protocol.get(field) != value for field, value in expected.items()
    ):
        raise D4CaptureError("raw benchmark changed the frozen D4 protocol or matrix")
    pairs = result.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != len(PROFILES) * len(expected_cases) * 5:
        raise D4CaptureError("raw benchmark pair inventory is incomplete")
    phase_c = result.get("release_eligibility", {}).get("phase_c", {})
    if not isinstance(phase_c, Mapping) or phase_c.get("status") != "pass":
        raise D4CaptureError("raw benchmark lacks a passing Phase C prerequisite")


def _provenance_affinity_masks(
    provenance: Mapping[str, Any], architecture: str
) -> dict[int, list[int] | None]:
    try:
        if architecture == "arm64":
            raw_masks = provenance["host"]["affinity"]["masks"]
        else:
            host = provenance["host"]
            attestation = host["resource_attestation"]
            raw_masks = attestation["physical_core_masks"]
    except (KeyError, TypeError) as error:
        raise D4CaptureError("raw capture lacks physical-core affinity provenance") from error
    if not isinstance(raw_masks, Mapping) or set(raw_masks) != {
        f"t{budget}" for budget in THREAD_BUDGETS
    }:
        raise D4CaptureError("raw capture physical-core mask inventory is incomplete")
    masks: dict[int, list[int] | None] = {}
    prior: list[int] = []
    for budget in THREAD_BUDGETS:
        raw = raw_masks[f"t{budget}"]
        if architecture == "arm64":
            if raw is not None:
                raise D4CaptureError("macOS capture invented unavailable CPU affinity")
            masks[budget] = None
            continue
        if (
            not isinstance(raw, list)
            or len(raw) != budget
            or len(raw) != len(set(raw))
            or any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in raw)
            or (prior and raw[: len(prior)] != prior)
        ):
            raise D4CaptureError("raw capture physical-core masks are invalid or non-nested")
        masks[budget] = raw
        prior = raw
    if architecture == "x86_64":
        try:
            visible = attestation["visible_affinity"]
            recomputed = physical_core_masks(
                host["lscpu_parse"], allowed_cpus=visible, budgets=THREAD_BUDGETS
            )
            provider = provenance["provider"]
            if provider == "modal":
                rebuilt_attestation = modal_resource_attestation(
                    cgroup_limits=attestation["cgroup_limits"],
                    platform_text=host["platform"],
                    uname=host["uname"],
                    visible_affinity=visible,
                    masks=recomputed,
                )
            elif provider == "local_linux":
                physical_count = len(
                    physical_core_representatives(host["lscpu_parse"], allowed_cpus=visible)
                )
                if attestation["allocated_physical_cores"] != physical_count:
                    raise D4CaptureError(
                        "local Linux physical-core allocation differs from CPU topology"
                    )
                rebuilt_attestation = local_linux_resource_attestation(
                    cgroup_limits=attestation["cgroup_limits"],
                    allocated_physical_cores=attestation["allocated_physical_cores"],
                    allocated_memory_bytes=attestation["allocated_memory_bytes"],
                    power_policy=attestation["power_policy"],
                    exclusive_physical_cores=attestation["exclusive_physical_cores"],
                    dedicated_capture=attestation["dedicated_capture"],
                    stable=attestation["stable"],
                    visible_affinity=visible,
                    masks=recomputed,
                    affinity_enforcement=attestation["affinity_enforcement"],
                )
            else:
                raise D4CaptureError(f"unknown x86 capture provider: {provider!r}")
        except (KeyError, TypeError) as error:
            raise D4CaptureError("raw capture CPU topology attestation is incomplete") from error
        if {budget: list(mask) for budget, mask in recomputed.items()} != {
            budget: masks[budget] for budget in THREAD_BUDGETS
        } or rebuilt_attestation != attestation:
            raise D4CaptureError("raw capture physical-core masks differ from CPU topology")
    return masks


def validate_archived_build(
    files: Mapping[str, bytes], provenance: Mapping[str, Any], build_label: str
) -> dict[str, Any]:
    """Re-hash the retained wheel and reconcile every recorded build identity."""

    prefix = f"builds/{build_label}/"
    wheel_names = sorted(
        name for name in files if name.startswith(prefix) and name.endswith(".whl")
    )
    if len(wheel_names) != 1:
        raise D4CaptureError(f"{build_label}: raw archive must retain exactly one wheel")
    try:
        build = json.loads(files[f"builds/{build_label}/build.json"])
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise D4CaptureError(f"{build_label}: archived build metadata is invalid") from error
    wheel_name = wheel_names[0]
    wheel_bytes = files[wheel_name]
    with tempfile.TemporaryDirectory(prefix="qwen-mm-d4-wheel-verify-") as temporary:
        wheel_path = Path(temporary) / Path(wheel_name).name
        wheel_path.write_bytes(wheel_bytes)
        contents = wheel_contents_identity(wheel_path)
        benchmark_artifact = wheel_benchmark_artifact_identity(wheel_path)
    try:
        identity = build["identity"]["wheel"]
        runtime = build["runtime"]
        reconciliation = build["runtime_reconciliation"]
        expected_wheel_sha = provenance["build_wheel_sha256"][build_label]
        expected_native_sha = provenance["build_native_sha256"][build_label]
    except (KeyError, TypeError) as error:
        raise D4CaptureError(f"{build_label}: build identity is incomplete") from error
    if (
        identity.get("name") != Path(wheel_name).name
        or identity.get("bytes") != len(wheel_bytes)
        or identity.get("sha256") != sha256_bytes(wheel_bytes)
        or identity.get("contents") != contents
        or reconciliation.get("verified") is not True
        or reconciliation.get("wheel_contents") != contents
        or runtime.get("native_sha256") != contents.get("native_artifact_sha256")
        or runtime.get("native_bytes") != contents.get("native_bytes")
        or expected_wheel_sha != contents.get("wheel", {}).get("sha256")
        or expected_native_sha != contents.get("native_artifact_sha256")
    ):
        raise D4CaptureError(f"{build_label}: retained wheel/build/runtime identities differ")
    return {**build, "_retained_benchmark_artifact": benchmark_artifact}


def validate_archived_phase_c(
    files: Mapping[str, bytes],
    provenance: Mapping[str, Any],
    build_label: str,
    build: Mapping[str, Any],
    *,
    assets_root: Path | None,
) -> dict[str, str]:
    """Authenticate both fresh Phase C envelopes and optionally their full assets."""

    def timestamp(value: Any, name: str) -> datetime:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as error:
            raise D4CaptureError(f"{name} is not an ISO-8601 timestamp") from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise D4CaptureError(f"{name} lacks a timezone")
        return parsed

    try:
        capture = json.loads(files[f"builds/{build_label}/capture.json"])
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as error:
        raise D4CaptureError(f"{build_label}: archived capture metadata is invalid") from error
    started = timestamp(provenance.get("started_at"), "provenance.started_at")
    completed = timestamp(provenance.get("completed_at"), "provenance.completed_at")
    capture_created = timestamp(capture.get("created_at"), f"{build_label}.capture.created_at")
    reports: dict[str, str] = {}
    report_times: dict[str, datetime] = {}
    for position in ("pre", "post"):
        name = f"phase-c/{build_label}/{position}/report.json"
        try:
            report = json.loads(files[name])
            runtime = report["candidate"]["runtime_identity"]
            scope = report["scope"]
            git = report["provenance"]["git"]
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as error:
            raise D4CaptureError(
                f"{build_label}: archived Phase C {position} is invalid"
            ) from error
        if (
            report.get("status") != "pass"
            or report.get("passed") is not True
            or scope.get("profiles") != list(PROFILES)
            or scope.get("skipped_case_ids") != []
            or scope.get("declared_case_ids") != scope.get("executed_case_ids")
            or git.get("revision") != provenance.get("source_revision")
            or git.get("gate_inputs_clean") is not True
            or git.get("gate_input_status") != []
            or runtime.get("native_artifact_sha256")
            != build.get("runtime", {}).get("native_sha256")
            or runtime.get("package_artifact_sha256")
            != build.get("runtime_reconciliation", {})
            .get("wheel_contents", {})
            .get("package_artifact_sha256")
        ):
            raise D4CaptureError(f"{build_label}: Phase C {position} provenance is stale")
        if assets_root is not None:
            try:
                from qwen_mm_reference.phase_c_conformance import validate_report

                validate_report(report, assets_root=assets_root)
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
                raise D4CaptureError(
                    f"{build_label}: Phase C {position} failed full validation"
                ) from error
        reports[position] = sha256_bytes(files[name])
        report_times[position] = timestamp(
            report.get("created_at"), f"{build_label}.phase_c.{position}.created_at"
        )
    if reports["pre"] == reports["post"]:
        raise D4CaptureError(f"{build_label}: Phase C pre/post reports were replayed")
    if not (started <= report_times["pre"] < report_times["post"] <= capture_created <= completed):
        raise D4CaptureError(f"{build_label}: Phase C/capture timestamps are stale or inverted")
    reports.update(
        pre_created_at=report_times["pre"].isoformat(),
        post_created_at=report_times["post"].isoformat(),
        capture_created_at=capture_created.isoformat(),
    )
    return reports


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


def read_capture_archive(
    value: bytes, *, phase_c_assets_root: Path | None = None
) -> dict[str, bytes]:
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
    try:
        provenance = json.loads(files["provenance.json"])
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise D4CaptureError("raw capture provenance is invalid") from error
    architecture = provenance.get("architecture_family")
    if index.get("architecture_family") != architecture or architecture not in {"arm64", "x86_64"}:
        raise D4CaptureError("raw capture architecture provenance is inconsistent")
    affinity_masks = _provenance_affinity_masks(provenance, architecture)
    for build_label in BUILD_LABELS:
        build = validate_archived_build(files, provenance, build_label)
        phase_c_reports = validate_archived_phase_c(
            files,
            provenance,
            build_label,
            build,
            assets_root=phase_c_assets_root,
        )
        pre_created = datetime.fromisoformat(phase_c_reports["pre_created_at"])
        post_created = datetime.fromisoformat(phase_c_reports["post_created_at"])
        previous_result_created: datetime | None = None
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
            try:
                result_created = datetime.fromisoformat(
                    str(result.get("created_at")).replace("Z", "+00:00")
                )
            except ValueError as error:
                raise D4CaptureError("raw benchmark creation timestamp is invalid") from error
            if (
                result_created.tzinfo is None
                or result_created.utcoffset() is None
                or not pre_created < result_created < post_created
                or (
                    previous_result_created is not None
                    and result_created <= previous_result_created
                )
            ):
                raise D4CaptureError(
                    "raw benchmark timestamps are stale, inverted, or outside fresh Phase C"
                )
            previous_result_created = result_created
            try:
                from qwen_mm_reference.benchmark_v2 import validate_result_portable

                validate_result_portable(result)
            except (ImportError, RuntimeError, TypeError, ValueError) as error:
                raise D4CaptureError("raw benchmark failed portable validation") from error
            if (
                result.get("architecture_family") != architecture
                or result.get("protocol", {}).get("build_labels") != [build_label]
                or result.get("protocol", {}).get("thread_regimes") != [f"t{budget}"]
                or result.get("protocol", {}).get("random_seed") != D4_RANDOM_SEED
            ):
                raise D4CaptureError("raw benchmark coordinate provenance is inconsistent")
            candidate_identity = result.get("protocol", {}).get("candidate_identity", {})
            if (
                not isinstance(candidate_identity, Mapping)
                or candidate_identity.get("artifact_sha256")
                != build["_retained_benchmark_artifact"]["sha256"]
            ):
                raise D4CaptureError(
                    "raw benchmark adapter differs from qwen_mm/benchmark.py in its retained wheel"
                )
            validate_d4_result_contract(
                result,
                architecture=architecture,
                build_label=build_label,
                budget=budget,
                affinity_cpus=affinity_masks[budget],
            )
            if (
                result.get("release_eligibility", {}).get("phase_c", {}).get("report_sha256")
                != phase_c_reports["pre"]
            ):
                raise D4CaptureError("raw benchmark is not bound to its archived pre-Phase C run")
            if noise != result_noise_assessment(result):
                raise D4CaptureError("raw capture noise assessment differs from raw medians")
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
    phase_c_assets_root: Path | None = None,
) -> None:
    """Validate and atomically persist a raw capture archive."""

    files = read_capture_archive(value, phase_c_assets_root=phase_c_assets_root)
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


def local_linux_toolchain_pins() -> dict[str, Any]:
    """Return build-tool pins without inventing Modal image/resource claims."""

    return {"rust": RUST_VERSION, "uv": UV_VERSION}
