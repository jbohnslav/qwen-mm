"""Versioned whole-operation observation capture for Phase D profiling."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .benchmark_protocol import (
    THREAD_ENVIRONMENT_NAMES,
    AdapterContext,
    BenchmarkProtocolError,
    architecture_family,
    configure_threads,
    load_adapter,
    load_workload,
    materialize_case,
    normalize_outputs,
    output_signature,
    select_cases,
    workload_provenance,
)
from .benchmark_v2 import (
    candidate_artifact_identity,
    validate_result,
    validate_result_portable,
)
from .fixtures import repository_root
from .phase_c_conformance import validate_report as validate_phase_c_report

PROFILE_SCHEMA_ID = "qwen-mm-profile-bundle-v1"
PROFILE_SCHEMA_VERSION = 1
OBSERVATION_SCHEMA_VERSION = "qwen-mm-observation-v1"
DEFAULT_CASES = ("text_short", "text_long", "image1", "image24", "ragged24", "rgb24")
DEFAULT_PROFILES = ("qwen3-vl-8b", "qwen3.5-9b")
DEFAULT_THREAD_BUDGETS = (1, 4)
DEFAULT_REPETITIONS = 3
DEFAULT_ADAPTER = "qwen_mm.benchmark:create_observed_adapter"
BASELINE_ADAPTER = "qwen_mm.benchmark:create_adapter"
PROFILE_BUILD_LABEL = "profiled-release"
SAMPLER_PROTOCOLS = {
    "arm64": {
        "name": "sample",
        "native": True,
        "duration_seconds": 2,
        "interval_ms": 1,
        "conversion_version": "macos-sample-callgraph-v1",
    },
    "x86_64": {
        "name": "py-spy",
        "native": False,
        "dependency": "py-spy==0.4.1",
        "binary_path": "/workspace/qwen-mm/.venv/bin/py-spy",
        "rate_hz": 99,
        "duration_seconds": 2,
    },
}
REQUIRED_COMMON_STAGES = {
    "binding.prepare_batch",
    "binding.parse_requests",
    "binding.destination.allocate",
    "native.batch.plan",
    "native.chat.render",
    "native.chat.tokenize",
    "native.destination.execute",
    "binding.numpy.materialize",
}
REQUIRED_MEDIA_STAGES = {
    "native.media.plan",
    "native.media.decode_color",
    "native.media.resize",
    "native.media.normalize_patchify_layout",
}


class ProfileArtifactError(ValueError):
    """Raised when a profile bundle is incomplete, inconsistent, or tampered."""


def _validate_profile_build_command(command: str) -> None:
    token_list = shlex.split(command)
    tokens = set(token_list)
    required_assignments = {
        "CARGO_PROFILE_RELEASE_DEBUG=1",
        "CARGO_PROFILE_RELEASE_OPT_LEVEL=3",
        "CARGO_PROFILE_RELEASE_STRIP=none",
        "RUSTFLAGS=-Cforce-frame-pointers=yes",
    }
    required = {
        *required_assignments,
        "maturin",
        "build",
        "--release",
        "--locked",
    }
    if not required <= tokens:
        raise ProfileArtifactError(
            "profile build must pin release opt-level=3, debuginfo=1, frame pointers, no strip, "
            "and locked Maturin release mode"
        )
    assignment_names = {value.split("=", 1)[0] for value in required_assignments}
    relevant_assignments = [
        value for value in token_list if value.partition("=")[0] in assignment_names
    ]
    if (
        len(relevant_assignments) != len(required_assignments)
        or set(relevant_assignments) != required_assignments
    ):
        raise ProfileArtifactError("profile build command has duplicate or contradictory flags")
    try:
        maturin_index = token_list.index("maturin")
    except ValueError as error:
        raise ProfileArtifactError("profile build command omits Maturin") from error
    if token_list[maturin_index : maturin_index + 2] != ["maturin", "build"]:
        raise ProfileArtifactError("profile build command must execute `maturin build`")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _schema() -> dict[str, Any]:
    path = repository_root() / "benchmarks" / "profile-schema-v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileArtifactError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ProfileArtifactError(f"JSON artifact {path} must contain an object")
    return value


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repository_root(),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _source_identity(
    *,
    revision: str | None = None,
    source_digest: str | None = None,
    declared_clean: bool = False,
) -> dict[str, Any]:
    observed_revision = _git_output("rev-parse", "HEAD")
    effective_revision = revision or os.environ.get("QWEN_MM_SOURCE_REVISION") or observed_revision
    if not effective_revision:
        raise ProfileArtifactError("source revision is required when .git is unavailable")
    declared_digest = source_digest or os.environ.get("QWEN_MM_SOURCE_DIGEST")
    effective_digest = declared_digest
    file_count: int | None = None
    total_bytes: int | None = None
    status = _git_output("status", "--porcelain")
    if observed_revision is not None:
        if revision is not None and revision != observed_revision:
            raise ProfileArtifactError("declared source revision differs from observed git HEAD")
        if status is None:
            raise ProfileArtifactError("git status is unavailable for the observed source revision")
        dirty = bool(status)
        if declared_clean and dirty:
            raise ProfileArtifactError("--source-clean conflicts with observed dirty git state")
        try:
            from scripts.modal_benchmark_support import (
                ModalBenchmarkArtifactError,
                committed_source_tree_digest,
                source_tree_digest,
            )
        except ImportError as error:
            raise ProfileArtifactError(
                "source digest is required when the repository fingerprint helper is unavailable"
            ) from error
        try:
            digest_source = source_tree_digest if dirty else committed_source_tree_digest
            if dirty:
                observed_digest, file_count, total_bytes = digest_source(repository_root())
            else:
                observed_digest, file_count, total_bytes = digest_source(
                    repository_root(), observed_revision
                )
        except (ModalBenchmarkArtifactError, OSError) as error:
            raise ProfileArtifactError(
                "source digest is required when the repository fingerprint helper is unavailable"
            ) from error
        if declared_digest is not None and declared_digest != observed_digest:
            raise ProfileArtifactError("declared source digest differs from observed source tree")
        effective_revision = observed_revision
        effective_digest = observed_digest
        clean_attestation = "git_status_porcelain"
    elif effective_digest is None:
        try:
            from scripts.modal_benchmark_support import source_tree_digest

            effective_digest, file_count, total_bytes = source_tree_digest(repository_root())
        except (ImportError, OSError) as error:
            raise ProfileArtifactError(
                "source digest is required when the repository fingerprint helper is unavailable"
            ) from error
    if len(effective_digest) != 64:
        raise ProfileArtifactError("source digest must be a SHA-256 hex digest")
    if observed_revision is None:
        if not declared_clean or revision is None or source_digest is None:
            raise ProfileArtifactError(
                "packaged source without .git requires explicit clean attestation, revision, and digest"
            )
        dirty = False
        clean_attestation = "packaged_source_from_clean_revision_and_digest"
    return {
        "revision": effective_revision,
        "tree_sha256": effective_digest,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "dirty": dirty,
        "clean_attestation": clean_attestation,
    }


def _source_tree_digest_excluding(excluded: set[str]) -> tuple[str, int, int]:
    from scripts.modal_benchmark_support import is_ignored_source_path

    root = repository_root()
    digest = hashlib.sha256()
    count = 0
    total = 0
    paths: list[Path] = []
    for current, directories, files in os.walk(root):
        current_path = Path(current)
        directories[:] = sorted(
            directory
            for directory in directories
            if not is_ignored_source_path((current_path / directory).relative_to(root))
        )
        paths.extend(current_path / name for name in files)
    for path in sorted(paths):
        relative = path.relative_to(root)
        if relative.as_posix() in excluded or is_ignored_source_path(relative):
            continue
        data = path.read_bytes()
        encoded = relative.as_posix().encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
        count += 1
        total += len(data)
    return digest.hexdigest(), count, total


def _source_tree_digest_at_revision(revision: str, excluded: set[str]) -> tuple[str, int, int]:
    """Recompute the upload fingerprint from an immutable recorded Git commit."""

    try:
        from scripts.modal_benchmark_support import (
            ModalBenchmarkArtifactError,
            committed_source_tree_digest,
        )
    except ImportError as error:
        raise ProfileArtifactError("recorded source digest helper is unavailable") from error
    try:
        return committed_source_tree_digest(repository_root(), revision, excluded=excluded)
    except ModalBenchmarkArtifactError as error:
        raise ProfileArtifactError(str(error)) from error


def _cpu_description() -> str:
    commands = (
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        ["lscpu"],
    )
    for command in commands:
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError):
            continue
        description = result.stdout.strip()
        if description:
            return description
    return platform.processor() or "unknown"


def _host_provenance() -> dict[str, Any]:
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = None
    cgroup: dict[str, str | None] = {}
    for name in ("cpu.max", "cpuset.cpus.effective"):
        path = Path("/sys/fs/cgroup") / name
        try:
            cgroup[name] = path.read_text(encoding="utf-8").strip()
        except OSError:
            cgroup[name] = None
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "architecture_family": architecture_family(),
        "hostname": platform.node(),
        "cpu_description": _cpu_description(),
        "logical_cpu_count": os.cpu_count(),
        "affinity_cpus": affinity,
        "effective_cpu_count": len(affinity) if affinity is not None else os.cpu_count(),
        "cgroup_v2": cgroup,
        "python": platform.python_version(),
    }


def _validate_thread_settings(settings: Any, budget: int, *, label: str) -> None:
    if not isinstance(settings, Mapping) or settings.get("budget") != budget:
        raise ProfileArtifactError(f"{label} thread settings differ from its budget")
    environment = settings.get("environment")
    expected = {name: str(budget) for name in THREAD_ENVIRONMENT_NAMES}
    if environment != expected:
        raise ProfileArtifactError(f"{label} thread environment is not exactly budget-pinned")


def _cpuset_count(value: str) -> int:
    count = 0
    for item in value.split(","):
        bounds = item.strip().split("-", 1)
        try:
            start = int(bounds[0])
            end = int(bounds[-1])
        except ValueError as error:
            raise ProfileArtifactError("invalid cgroup cpuset provenance") from error
        if start < 0 or end < start:
            raise ProfileArtifactError("invalid cgroup cpuset provenance")
        count += end - start + 1
    return count


def _validate_host_provenance(host: Mapping[str, Any]) -> None:
    architecture = host.get("architecture_family")
    system = host.get("system")
    machine = str(host.get("machine", "")).lower()
    expected = {
        "arm64": ("Darwin", {"arm64", "aarch64"}),
        "x86_64": ("Linux", {"x86_64", "amd64"}),
    }
    if architecture not in expected:
        raise ProfileArtifactError("profile host architecture is unsupported")
    expected_system, machines = expected[architecture]
    if system != expected_system or machine not in machines:
        raise ProfileArtifactError("profile host OS/machine does not match architecture lane")
    effective = host.get("effective_cpu_count")
    if not isinstance(effective, int) or isinstance(effective, bool) or effective < 4:
        raise ProfileArtifactError("profile host exposes fewer than four effective CPUs")
    if system != "Linux":
        return
    cgroup = host.get("cgroup_v2")
    if not isinstance(cgroup, Mapping):
        raise ProfileArtifactError("Linux capture lacks cgroup-v2 CPU provenance")
    cpu_max = cgroup.get("cpu.max")
    if isinstance(cpu_max, str) and cpu_max:
        values = cpu_max.split()
        if len(values) != 2:
            raise ProfileArtifactError("invalid cgroup cpu.max provenance")
        if values[0] != "max":
            try:
                quota, period = map(int, values)
            except ValueError as error:
                raise ProfileArtifactError("invalid cgroup cpu.max provenance") from error
            if quota <= 0 or period <= 0 or quota / period < 4:
                raise ProfileArtifactError("Linux cgroup CPU quota exposes fewer than four CPUs")
    cpuset = cgroup.get("cpuset.cpus.effective")
    if isinstance(cpuset, str) and cpuset and _cpuset_count(cpuset) < 4:
        raise ProfileArtifactError("Linux cgroup cpuset exposes fewer than four CPUs")


def _operation_coordinates(protocol: Mapping[str, Any]) -> set[tuple[str, str, int, int]]:
    return {
        (profile, case, budget, repetition)
        for profile in protocol["profiles"]
        for case in protocol["cases"]
        for budget in protocol["thread_budgets"]
        for repetition in range(protocol["repetitions"])
    }


def _expected_media_scopes(payload: Any) -> list[tuple[int, int, int, int, int]]:
    scopes: list[tuple[int, int, int, int, int]] = []
    media_index = 0
    for request_index, messages in enumerate(payload.messages):
        local_indices: dict[int, int] = {}
        for message_index, message in enumerate(messages):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for content_index, item in enumerate(content):
                if item.get("type") != "image":
                    continue
                global_index = item["buffer_index"]
                local_index = local_indices.setdefault(global_index, len(local_indices))
                scopes.append(
                    (
                        request_index,
                        message_index,
                        content_index,
                        media_index,
                        local_index,
                    )
                )
                media_index += 1
    return scopes


def _sample_coordinates(protocol: Mapping[str, Any]) -> set[tuple[str, str, int]]:
    return {
        (profile, case, budget)
        for profile in protocol["profiles"]
        for case in protocol["cases"]
        for budget in protocol["thread_budgets"]
    }


def _artifact_identity(path: Path, *, published_path: Path | None = None) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(repository_root().resolve()).as_posix()
    except ValueError as error:
        if published_path is None:
            raise ProfileArtifactError(
                "external evidence requires its future repository-relative published path"
            ) from error
        if published_path.is_absolute() or ".." in published_path.parts:
            raise ProfileArtifactError(
                "published artifact path must be safe and repository-relative"
            ) from error
        relative = published_path.as_posix()
    return {"path": relative, "sha256": _sha256_path(path), "bytes": path.stat().st_size}


def _read_authenticated_artifact(identity: Mapping[str, Any]) -> tuple[Path, bytes]:
    relative = Path(str(identity.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ProfileArtifactError("sampled artifact path must be safe and repository-relative")
    path = (repository_root() / relative).resolve()
    try:
        path.relative_to(repository_root().resolve())
    except ValueError as error:
        raise ProfileArtifactError("sampled artifact escapes the repository") from error
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ProfileArtifactError(f"cannot read sampled artifact {relative}") from error
    if len(data) != identity.get("bytes") or hashlib.sha256(data).hexdigest() != identity.get(
        "sha256"
    ):
        raise ProfileArtifactError(f"sampled artifact hash/size mismatch: {relative}")
    return path, data


def _validate_wheel_runtime(wheel: bytes, runtime: Mapping[str, Any]) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
            names = archive.namelist()
            native_members = [
                name
                for name in names
                if name.startswith("qwen_mm/_native")
                and Path(name).suffix in {".so", ".dylib", ".pyd"}
            ]
            if len(native_members) != 1 or "qwen_mm/__init__.py" not in names:
                raise ProfileArtifactError("profile wheel has an invalid qwen_mm payload")
            native_hash = hashlib.sha256(archive.read(native_members[0])).hexdigest()
            package_hash = hashlib.sha256(archive.read("qwen_mm/__init__.py")).hexdigest()
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise ProfileArtifactError("profile wheel is unreadable") from error
    if native_hash != runtime.get("native_artifact_sha256") or package_hash != runtime.get(
        "package_artifact_sha256"
    ):
        raise ProfileArtifactError("wheel members differ from the measured runtime identity")


def _parse_collapsed(raw: str, *, required_frame: str | None = None) -> tuple[str, int, int]:
    stacks: dict[str, int] = defaultdict(int)
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            stack, count_text = line.rsplit(" ", 1)
            count = int(count_text)
        except (ValueError, TypeError) as error:
            raise ProfileArtifactError("py-spy raw artifact is not collapsed-stack data") from error
        if not stack or count <= 0:
            raise ProfileArtifactError("py-spy collapsed stacks require positive sample counts")
        if required_frame is not None and not any(
            required_frame in frame for frame in stack.split(";")
        ):
            continue
        stacks[stack] += count
    if not stacks:
        qualifier = " containing the whole-operation boundary" if required_frame else ""
        raise ProfileArtifactError(f"py-spy produced no sampled stacks{qualifier}")
    collapsed = "".join(f"{stack} {stacks[stack]}\n" for stack in sorted(stacks))
    return collapsed, sum(stacks.values()), len(stacks)


def _parse_py_spy_summary(output: str) -> tuple[int, int]:
    matches = re.findall(r"\bSamples:\s*(\d+)\s+Errors:\s*(\d+)\b", output)
    if len(matches) != 1:
        raise ProfileArtifactError("py-spy output lacks one unambiguous success summary")
    samples, errors = (int(value) for value in matches[0])
    if samples <= 0:
        raise ProfileArtifactError("py-spy reported no samples")
    if errors != 0:
        raise ProfileArtifactError(f"py-spy reported {errors} sampling errors")
    return samples, errors


_DEMANGLED_CORE_FRAME = re.compile(r"(?<![A-Za-z0-9_])qwen_mm_core::")
_DEMANGLED_BINDING_FRAME = re.compile(r"(?<![A-Za-z0-9_])(?:qwen_mm_python|qwen_mm_native)::")
_RUST_V0_CORE_CRATE = re.compile(r"C(?:s[0-9A-Za-z]*_)?12qwen_mm_core")
_RUST_V0_BINDING_CRATE = re.compile(r"C(?:s[0-9A-Za-z]*_)?14qwen_mm_native")


def _is_qwen_core_frame(frame: str) -> bool:
    """Recognize both demangled and Rust-v0 qwen-mm core symbols."""

    return _DEMANGLED_CORE_FRAME.search(frame) is not None or (
        frame.startswith("_R") and _RUST_V0_CORE_CRATE.search(frame) is not None
    )


def _is_qwen_binding_frame(frame: str) -> bool:
    """Recognize the package name and the cdylib crate name used by real builds."""

    return _DEMANGLED_BINDING_FRAME.search(frame) is not None or (
        frame.startswith("_R") and _RUST_V0_BINDING_CRATE.search(frame) is not None
    )


def _is_qwen_native_frame(frame: str) -> bool:
    return _is_qwen_core_frame(frame) or _is_qwen_binding_frame(frame)


def _is_symbolized_qwen_native_frame(frame: str) -> bool:
    lowered = frame.lower()
    return (
        _is_qwen_native_frame(frame)
        and ("::" in frame or frame.startswith("_R"))
        and "unknown" not in lowered
        and "+0x" not in lowered
    )


def _parse_macos_sample(raw: str) -> tuple[str, int, int]:
    """Reduce the indented macOS `sample` call graph to canonical collapsed stacks."""

    try:
        call_graph = raw.split("Call graph:\n", 1)[1].split("\nTotal number in stack", 1)[0]
    except IndexError as error:
        raise ProfileArtifactError("macOS sample report lacks a complete call graph") from error
    nodes: list[dict[str, Any]] = []
    stack: list[int] = []
    for line in call_graph.splitlines():
        match = re.match(
            r"^(?P<prefix>[\s+!:|]*)(?P<count>\d+)\s+(?P<frame>.+?)\s*$",
            line,
        )
        if match is None:
            continue
        indent = len(match.group("prefix"))
        count = int(match.group("count"))
        frame = match.group("frame")
        # Debug-enabled Rust builds append a source location after the sampled
        # address. Remove it first so the address and instruction offset below
        # can be canonicalized instead of fragmenting one function by ASLR.
        frame = re.sub(r"\s+\S+\.rs:\d+(?::\d+)?\s*$", "", frame)
        frame = re.sub(r"\s+\[[^]]+\]\s*$", "", frame)
        frame = re.sub(
            r"\s+\+\s+(?:0x[0-9a-fA-F]+|\d+(?:,\d+)*(?:,\.\.\.)?)\s*$",
            "",
            frame,
        )
        while stack and nodes[stack[-1]]["indent"] >= indent:
            stack.pop()
        parent = stack[-1] if stack else None
        node = {
            "indent": indent,
            "count": count,
            "frame": frame,
            "parent": parent,
            "child_count": 0,
        }
        nodes.append(node)
        index = len(nodes) - 1
        if parent is not None:
            nodes[parent]["child_count"] += count
        stack.append(index)
    collapsed_counts: dict[str, int] = defaultdict(int)
    for index, node in enumerate(nodes):
        self_count = node["count"] - node["child_count"]
        if self_count <= 0 or str(node["frame"]).startswith("Thread_"):
            continue
        frames = []
        cursor: int | None = index
        while cursor is not None:
            frame = str(nodes[cursor]["frame"])
            if not frame.startswith("Thread_"):
                frames.append(frame.replace(";", ":"))
            cursor = nodes[cursor]["parent"]
        frames.reverse()
        collapsed_counts[";".join(frames)] += self_count
    if not collapsed_counts:
        raise ProfileArtifactError("macOS sample call graph contains no self samples")
    collapsed = "".join(
        f"{stack_text} {collapsed_counts[stack_text]}\n" for stack_text in sorted(collapsed_counts)
    )
    return collapsed, sum(collapsed_counts.values()), len(collapsed_counts)


def _stack_rankings(collapsed: str) -> dict[str, Any]:
    inclusive: dict[str, int] = defaultdict(int)
    self_samples: dict[str, int] = defaultdict(int)
    sample_count = 0
    native_sample_count = 0
    for line in collapsed.splitlines():
        stack, count_text = line.rsplit(" ", 1)
        count = int(count_text)
        frames = [frame for frame in stack.split(";") if frame]
        if not frames:
            raise ProfileArtifactError("sampled stack contains no frames")
        sample_count += count
        if any(_is_qwen_native_frame(frame) for frame in frames):
            native_sample_count += count
        for frame in set(frames):
            inclusive[frame] += count
        self_samples[frames[-1]] += count

    def ranked(values: Mapping[str, int]) -> list[dict[str, Any]]:
        return [
            {"frame": frame, "samples": samples, "share": samples / sample_count}
            for frame, samples in sorted(values.items(), key=lambda item: (-item[1], item[0]))
        ]

    return {
        "sample_count": sample_count,
        "native_sample_count": native_sample_count,
        "native_sample_share": native_sample_count / sample_count,
        "inclusive": ranked(inclusive),
        "self": ranked(self_samples),
    }


def qwen_mm_profile_iteration(adapter: Any, payload: Any) -> Mapping[str, Any]:
    """Stable whole-boundary stack marker: Python payload through materialized NumPy."""

    return adapter.run(payload)


def _sample_worker(args: argparse.Namespace) -> None:
    config = _load_json(args.config)
    workload = load_workload(Path(config["workload_path"]))
    case = select_cases(workload, case_ids=[config["case_id"]])[0]
    payload = materialize_case(case)
    thread_settings = configure_threads(config["thread_budget"], configure_torch=False)
    adapter = load_adapter(
        BASELINE_ADAPTER,
        AdapterContext(config["profile_alias"], PROFILE_BUILD_LABEL, config["thread_budget"]),
    )
    runtime_identity = candidate_artifact_identity(BASELINE_ADAPTER).get("runtime_identity")
    if not isinstance(runtime_identity, Mapping):
        raise ProfileArtifactError("sample worker cannot resolve its candidate runtime identity")
    # Setup and correctness checks deliberately bypass the stable stack marker.
    # On x86 py-spy launches this finite process and records until it exits; only
    # marked whole-operation samples are admitted to the canonical collapse.
    before = normalize_outputs(adapter.run(payload), payload)
    before_signature = output_signature(before)
    del before
    ready = {
        "pid": os.getpid(),
        "input_fingerprint": payload.input_fingerprint,
        "logical_input_fingerprint": payload.logical_input_fingerprint,
        "before_signature": before_signature,
    }
    Path(config["ready_path"]).write_text(json.dumps(ready), encoding="utf-8")
    iterations = 0
    duration_seconds = config.get("duration_seconds")
    measurement_window: dict[str, int] | None = None
    if duration_seconds is None:
        stop_path = Path(config["stop_path"])
        while not stop_path.exists():
            sampled = qwen_mm_profile_iteration(adapter, payload)
            iterations += 1
            del sampled
    else:
        if duration_seconds != SAMPLER_PROTOCOLS["x86_64"]["duration_seconds"]:
            raise ProfileArtifactError("sample worker duration is not the frozen x86 protocol")
        requested_duration_ns = duration_seconds * 1_000_000_000
        started_ns = time.monotonic_ns()
        deadline_ns = started_ns + requested_duration_ns
        final_iteration_started_ns = started_ns
        while True:
            iteration_started_ns = time.monotonic_ns()
            if iteration_started_ns >= deadline_ns:
                break
            final_iteration_started_ns = iteration_started_ns
            sampled = qwen_mm_profile_iteration(adapter, payload)
            iterations += 1
            del sampled
        completed_ns = time.monotonic_ns()
        measurement_window = {
            "requested_duration_ns": requested_duration_ns,
            "started_ns": started_ns,
            "deadline_ns": deadline_ns,
            "final_iteration_started_ns": final_iteration_started_ns,
            "completed_ns": completed_ns,
        }
    after = normalize_outputs(adapter.run(payload), payload)
    postcheck_completed_ns = time.monotonic_ns()
    result = {
        "pid": os.getpid(),
        "iterations": iterations,
        "input_fingerprint": payload.input_fingerprint,
        "logical_input_fingerprint": payload.logical_input_fingerprint,
        "before_signature": before_signature,
        "after_signature": output_signature(after),
        "profile_alias": config["profile_alias"],
        "case_id": config["case_id"],
        "thread_budget": config["thread_budget"],
        "thread_settings": thread_settings,
        "runtime_identity": dict(runtime_identity),
        **(
            {
                "measurement_window": {
                    **measurement_window,
                    "postcheck_completed_ns": postcheck_completed_ns,
                }
            }
            if measurement_window is not None
            else {}
        ),
    }
    Path(config["result_path"]).write_text(json.dumps(result), encoding="utf-8")
    if duration_seconds is not None:
        # py-spy 0.4.1 performs a final waitpid after sampling a child.  Keep the
        # finite worker alive after its authenticated result is durable so the
        # controller can stop py-spy cleanly; py-spy then terminates its child
        # instead of racing the host's process handling at child exit.
        while True:
            signal.pause()


def _wait_for_path(path: Path, process: subprocess.Popen[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise ProfileArtifactError(
                f"sample worker exited before readiness: stdout={stdout!r} stderr={stderr!r}"
            )
        time.sleep(0.05)
    process.terminate()
    raise ProfileArtifactError("sample worker did not become ready before timeout")


def _linux_process_command(pid: int) -> list[str] | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    return [item.decode(errors="surrogateescape") for item in raw.split(b"\0") if item]


def _recorded_worker_pids(paths: Sequence[Path | None]) -> set[int]:
    pids: set[int] = set()
    for path in paths:
        if path is None:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        pid = value.get("pid") if isinstance(value, Mapping) else None
        if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0:
            pids.add(pid)
    return pids


def _cleanup_authenticated_workers(pids: Sequence[int], worker_command: list[str]) -> int:
    cleaned = 0
    for pid in set(pids):
        if _linux_process_command(pid) != worker_command:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        else:
            cleaned += 1
    return cleaned


def _run_py_spy_child(
    sampler_command: list[str],
    worker_command: list[str],
    *,
    ready_path: Path | None = None,
    result_path: Path | None = None,
    timeout: float = 300.0,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a py-spy child capture with bounded, authenticated orphan cleanup."""

    sampler = subprocess.Popen(
        sampler_command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + timeout
    authenticated_worker_pid: int | None = None
    authentication_error: str | None = None
    try:
        if result_path is not None:
            while sampler.poll() is None:
                result_pids = _recorded_worker_pids((result_path,))
                if result_pids:
                    if len(result_pids) != 1:
                        authentication_error = "py-spy result records an ambiguous worker PID"
                    else:
                        authenticated_worker_pid = next(iter(result_pids))
                        if _linux_process_command(authenticated_worker_pid) != worker_command:
                            authentication_error = (
                                "py-spy result PID does not match the exact worker command"
                            )
                        elif ready_path is not None:
                            ready_pids = _recorded_worker_pids((ready_path,))
                            if ready_pids != result_pids:
                                authentication_error = (
                                    "py-spy readiness/result worker PIDs do not match"
                                )
                    # Signal only py-spy, never its process group.  Even if the
                    # result is malformed, py-spy owns and safely tears down its
                    # actual child before we reject the capture below.
                    try:
                        os.kill(sampler.pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass
                    break
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(sampler_command, timeout)
                time.sleep(0.02)
        remaining = max(0.001, deadline - time.monotonic())
        stdout, stderr = sampler.communicate(timeout=remaining)
    except subprocess.TimeoutExpired as error:
        try:
            children_text = Path(f"/proc/{sampler.pid}/task/{sampler.pid}/children").read_text(
                encoding="utf-8"
            )
            child_pids = [int(value) for value in children_text.split()]
        except (OSError, ValueError):
            child_pids = []
        recorded_pids = _recorded_worker_pids((ready_path, result_path))
        discovered_pids = [*child_pids, *recorded_pids]
        sampler.kill()
        cleaned = _cleanup_authenticated_workers(discovered_pids, worker_command)
        try:
            sampler.communicate(timeout=10.0)
        except subprocess.TimeoutExpired:
            # A surviving descendant can retain py-spy's PIPE descriptors even
            # after py-spy itself is dead. Close our read ends and reap only the
            # sampler process with a second bounded wait.
            for stream in (sampler.stdout, sampler.stderr):
                if stream is not None:
                    stream.close()
            try:
                sampler.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                pass
        cleanup = (
            f"; terminated {cleaned} authenticated child worker(s)"
            if discovered_pids
            else "; no live child worker was discoverable for cleanup"
        )
        raise ProfileArtifactError(
            f"py-spy child capture exceeded the {timeout:g} s timeout{cleanup}"
        ) from error
    if result_path is not None and authenticated_worker_pid is None:
        cleaned = _cleanup_authenticated_workers(
            list(_recorded_worker_pids((ready_path, result_path))), worker_command
        )
        cleanup = (
            f"; terminated {cleaned} authenticated child worker(s)"
            if cleaned
            else "; no live authenticated child worker was discoverable for cleanup"
        )
        raise ProfileArtifactError(
            "py-spy exited before an authenticated worker result was published" + cleanup
        )
    if authenticated_worker_pid is not None:
        surviving_command = _linux_process_command(authenticated_worker_pid)
        if surviving_command == worker_command:
            _cleanup_authenticated_workers((authenticated_worker_pid,), worker_command)
            raise ProfileArtifactError("py-spy exited while its authenticated child was still live")
    if authentication_error is not None:
        _cleanup_authenticated_workers(
            list(_recorded_worker_pids((ready_path, result_path))), worker_command
        )
        raise ProfileArtifactError(authentication_error)
    result = subprocess.CompletedProcess(
        sampler_command, sampler.returncode, stdout=stdout, stderr=stderr
    )
    if result.returncode != 0:
        _cleanup_authenticated_workers(
            list(_recorded_worker_pids((ready_path, result_path))), worker_command
        )
    return result


def capture_sampled_profile(
    *,
    profile: str,
    case_id: str,
    thread_budget: int,
    workload_path: Path,
    artifact_directory: Path,
    artifact_publish_directory: Path | None,
    py_spy: str,
    rate_hz: int,
    duration_seconds: int,
) -> dict[str, Any]:
    architecture = architecture_family()
    expected_sampler = SAMPLER_PROTOCOLS.get(architecture)
    if expected_sampler is None or duration_seconds != expected_sampler["duration_seconds"]:
        raise ProfileArtifactError("D1 sampler settings do not match the architecture protocol")
    if architecture == "x86_64" and rate_hz != expected_sampler["rate_hz"]:
        raise ProfileArtifactError("D1 x86 sampler is frozen to Python py-spy at 99 Hz for 2 s")
    binary = "/usr/bin/sample" if architecture == "arm64" else shutil.which(py_spy)
    if binary is None:
        raise ProfileArtifactError(f"cannot resolve sampler binary: {py_spy}")
    binary_path = Path(binary).resolve()
    if architecture == "x86_64" and str(binary_path) != expected_sampler["binary_path"]:
        raise ProfileArtifactError("D1 x86 sampler is not the locked workspace py-spy executable")
    binary_sha256 = _sha256_path(binary_path)
    binary_bytes = binary_path.stat().st_size
    if architecture == "arm64":
        version_output = subprocess.run(
            ["/usr/bin/what", str(binary_path)], check=True, capture_output=True, text=True
        ).stdout
        version = next(
            (line.strip() for line in version_output.splitlines() if "PROGRAM:sample" in line),
            "",
        )
        if not version:
            raise ProfileArtifactError("cannot identify the macOS sample tool version")
    else:
        version = subprocess.run(
            [str(binary_path), "--version"], check=True, capture_output=True, text=True
        ).stdout.strip()
        if version != "py-spy 0.4.1":
            raise ProfileArtifactError("D1 requires exactly py-spy 0.4.1")
    artifact_directory.mkdir(parents=True, exist_ok=True)
    slug = f"{architecture}-{profile}-{case_id}-t{thread_budget}"
    raw_path = artifact_directory / f"{slug}.sampler.raw"
    collapsed_path = artifact_directory / f"{slug}.collapsed"
    worker_result_path = artifact_directory / f"{slug}.worker-result.json"
    with tempfile.TemporaryDirectory(prefix="qwen-mm-profile-worker-") as temporary:
        temporary_path = Path(temporary)
        config_path = temporary_path / "config.json"
        ready_path = temporary_path / "ready.json"
        stop_path = temporary_path / "stop"
        result_path = temporary_path / "result.json"
        raw_temporary = temporary_path / "profile.raw"
        config = {
            "workload_path": str(workload_path.resolve()),
            "case_id": case_id,
            "profile_alias": profile,
            "thread_budget": thread_budget,
            "ready_path": str(ready_path),
            "stop_path": str(stop_path),
            "result_path": str(result_path),
            **({"duration_seconds": duration_seconds} if architecture == "x86_64" else {}),
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")
        worker_command = [
            sys.executable,
            "-m",
            "qwen_mm_reference.profile_v1",
            "_sample_worker",
            "--config",
            str(config_path),
        ]
        if architecture == "arm64":
            worker = subprocess.Popen(
                worker_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            try:
                _wait_for_path(ready_path, worker, timeout=300.0)
                ready = _load_json(ready_path)
                sampler_command = [
                    str(binary_path),
                    str(ready["pid"]),
                    str(duration_seconds),
                    str(expected_sampler["interval_ms"]),
                    "-mayDie",
                    "-file",
                    str(raw_temporary),
                ]
                sampler = subprocess.run(sampler_command, capture_output=True, text=True)
                stop_path.write_text("stop\n", encoding="utf-8")
                stdout, stderr = worker.communicate(timeout=300)
                if sampler.returncode != 0:
                    raise ProfileArtifactError(
                        f"{expected_sampler['name']} failed: {sampler.stderr.strip()}"
                    )
                if worker.returncode != 0:
                    raise ProfileArtifactError(
                        f"sample worker failed: stdout={stdout!r} stderr={stderr!r}"
                    )
                result = _load_json(result_path)
            finally:
                if worker.poll() is None:
                    worker.terminate()
                    worker.wait(timeout=10)
        else:
            # Linux ptrace policies commonly reject sibling attachment. Let
            # py-spy create the finite worker so the target is its child; the
            # worker's authenticated monotonic window controls the two-second
            # whole-operation measurement, publishes its post-check result, and
            # waits while the controller stops py-spy cleanly.
            sampler_command = [
                str(binary_path),
                "record",
                "--rate",
                str(rate_hz),
                "--format",
                "raw",
                "-o",
                str(raw_temporary),
                "--",
                *worker_command,
            ]
            sampler = _run_py_spy_child(
                sampler_command,
                worker_command,
                ready_path=ready_path,
                result_path=result_path,
            )
            if sampler.returncode != 0:
                raise ProfileArtifactError(
                    f"{expected_sampler['name']} failed: {sampler.stderr.strip()}"
                )
            reported_sample_count, sampler_errors = _parse_py_spy_summary(
                f"{sampler.stdout}\n{sampler.stderr}"
            )
            if not ready_path.is_file() or not result_path.is_file():
                raise ProfileArtifactError(
                    "py-spy child worker exited without readiness/result provenance: "
                    f"stdout={sampler.stdout!r} stderr={sampler.stderr!r}"
                )
            ready = _load_json(ready_path)
            result = _load_json(result_path)
            if ready.get("pid") != result.get("pid"):
                raise ProfileArtifactError("py-spy child worker PID changed during capture")
        raw = raw_temporary.read_text(encoding="utf-8")
    if architecture == "arm64":
        collapsed, sample_count, unique_stacks = _parse_macos_sample(raw)
        sampler_errors = 0
        reported_sample_count = None
    else:
        _, raw_sample_count, _ = _parse_collapsed(raw)
        if raw_sample_count != reported_sample_count:
            raise ProfileArtifactError(
                "py-spy raw sample total differs from its reported success summary"
            )
        collapsed, sample_count, unique_stacks = _parse_collapsed(
            raw, required_frame="qwen_mm_profile_iteration"
        )
    if architecture == "x86_64" and "qwen_mm_profile_iteration" not in collapsed:
        raise ProfileArtifactError("sampled stacks omit the whole-operation Python boundary marker")
    collapsed_frames = [
        frame for line in collapsed.splitlines() for frame in line.rsplit(" ", 1)[0].split(";")
    ]
    if architecture == "arm64" and (
        not any(marker in collapsed for marker in ("PyEval", "eval_frame"))
        or not any(_is_qwen_core_frame(frame) for frame in collapsed_frames)
        or not any(_is_qwen_binding_frame(frame) for frame in collapsed_frames)
    ):
        raise ProfileArtifactError("macOS samples omit CPython/binding/core boundary evidence")
    if result["before_signature"] != result["after_signature"]:
        raise ProfileArtifactError("sample worker outputs changed across measurement")
    raw_path.write_text(raw, encoding="utf-8")
    collapsed_path.write_text(collapsed, encoding="utf-8")
    worker_result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if _sha256_path(binary_path) != binary_sha256 or binary_path.stat().st_size != binary_bytes:
        raise ProfileArtifactError("sampler binary changed during capture")
    worker_source = Path(__file__).resolve()
    return {
        "profile_alias": profile,
        "architecture_family": architecture,
        "case_id": case_id,
        "thread_budget": thread_budget,
        "thread_regime": "one" if thread_budget == 1 else "production",
        "input_fingerprint": result["input_fingerprint"],
        "logical_input_fingerprint": result["logical_input_fingerprint"],
        "before_signature": result["before_signature"],
        "after_signature": result["after_signature"],
        "iterations": result["iterations"],
        "whole_boundary_marker": "qwen_mm_profile_iteration",
        "worker": {
            "source": _artifact_identity(worker_source),
            "result": _artifact_identity(
                worker_result_path,
                published_path=(
                    None
                    if artifact_publish_directory is None
                    else artifact_publish_directory / worker_result_path.name
                ),
            ),
            "command": shlex.join(worker_command),
            "target_pid": result["pid"],
            "runtime_identity": result["runtime_identity"],
        },
        "sampler": {
            "name": expected_sampler["name"],
            "version": version,
            "binary_path": str(binary_path),
            "binary_sha256": binary_sha256,
            "binary_bytes": binary_bytes,
            "command": shlex.join(sampler_command),
            "native": expected_sampler["native"],
            "duration_seconds": duration_seconds,
            "sample_count": sample_count,
            "unique_stacks": unique_stacks,
            "errors": sampler_errors,
            "install_provenance": (
                "macOS system /usr/bin/sample authenticated by executable hash"
                if architecture == "arm64"
                else "locked dev dependency py-spy==0.4.1 authenticated by version and executable hash"
            ),
            **(
                {
                    "interval_ms": expected_sampler["interval_ms"],
                    "conversion_version": expected_sampler["conversion_version"],
                }
                if architecture == "arm64"
                else {
                    "rate_hz": rate_hz,
                    "reported_sample_count": reported_sample_count,
                }
            ),
        },
        "artifacts": {
            "raw": _artifact_identity(
                raw_path,
                published_path=(
                    None
                    if artifact_publish_directory is None
                    else artifact_publish_directory / raw_path.name
                ),
            ),
            "collapsed": _artifact_identity(
                collapsed_path,
                published_path=(
                    None
                    if artifact_publish_directory is None
                    else artifact_publish_directory / collapsed_path.name
                ),
            ),
        },
        "rankings": _stack_rankings(collapsed),
    }


def validate_observation(
    report: Mapping[str, Any],
    *,
    media_count: int,
    request_count: int,
    retained_output_bytes: int,
    expected_media_scopes: Sequence[tuple[int, int, int, int, int]] = (),
    require_complete_stages: bool = True,
) -> None:
    if report.get("schema_version") != OBSERVATION_SCHEMA_VERSION:
        raise ProfileArtifactError("observation schema version mismatch")
    if report.get("outcome") != "success" or report.get("error_category") is not None:
        raise ProfileArtifactError("profile capture must contain a successful observation")
    if report.get("dropped_events") != 0:
        raise ProfileArtifactError("profile observation dropped bounded events")
    duration = report.get("duration_ns")
    spans = report.get("spans")
    buffers = report.get("buffers")
    copies = report.get("copies")
    if not isinstance(duration, int) or isinstance(duration, bool) or duration < 0:
        raise ProfileArtifactError("observation duration is invalid")
    if (
        not isinstance(spans, list)
        or not spans
        or not isinstance(buffers, list)
        or not isinstance(copies, list)
    ):
        raise ProfileArtifactError("observation spans, buffers, and copies must be arrays")
    sequences: dict[int, Mapping[str, Any]] = {}
    for span in spans:
        if not isinstance(span, Mapping):
            raise ProfileArtifactError("observation span must be an object")
        sequence = span.get("sequence")
        parent = span.get("parent_sequence")
        inclusive = span.get("duration_ns")
        exclusive = span.get("exclusive_duration_ns")
        if not isinstance(sequence, int) or sequence in sequences:
            raise ProfileArtifactError("observation span sequences must be unique integers")
        if parent is not None and (not isinstance(parent, int) or parent >= sequence):
            raise ProfileArtifactError("observation parent sequence must precede its child")
        if not isinstance(inclusive, int) or not isinstance(exclusive, int):
            raise ProfileArtifactError("observation span durations must be integers")
        if inclusive < 0 or exclusive < 0 or exclusive > inclusive:
            raise ProfileArtifactError("exclusive span duration must be within inclusive duration")
        if span.get("outcome") != "success" or span.get("error_category") is not None:
            raise ProfileArtifactError("successful profile contains a failed stage")
        sequences[sequence] = span
    if sorted(sequences) != list(range(len(spans))):
        raise ProfileArtifactError("observation span sequences must be contiguous")
    for span in spans:
        parent = span.get("parent_sequence")
        if parent is not None and parent not in sequences:
            raise ProfileArtifactError("observation span references a missing parent")
        if span["started_ns"] + span["duration_ns"] > duration:
            raise ProfileArtifactError("observation span extends beyond the operation")
        if parent is not None:
            parent_span = sequences[parent]
            if span["started_ns"] < parent_span["started_ns"] or (
                span["started_ns"] + span["duration_ns"]
                > parent_span["started_ns"] + parent_span["duration_ns"]
            ):
                raise ProfileArtifactError("observation child span escapes its parent interval")
    children: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for span in spans:
        if span["parent_sequence"] is not None:
            children[span["parent_sequence"]].append(span)
    for sequence, parent_span in sequences.items():
        direct = sorted(children.get(sequence, []), key=lambda child: child["started_ns"])
        for left, right in zip(direct, direct[1:], strict=False):
            if left["started_ns"] + left["duration_ns"] > right["started_ns"]:
                raise ProfileArtifactError("direct child spans overlap")
        expected_exclusive = parent_span["duration_ns"] - sum(
            child["duration_ns"] for child in direct
        )
        if parent_span["exclusive_duration_ns"] != expected_exclusive:
            raise ProfileArtifactError("exclusive duration does not reconcile with direct children")
    stage_names = {str(span.get("name")) for span in spans}
    required = set(REQUIRED_COMMON_STAGES)
    if media_count:
        required.update(REQUIRED_MEDIA_STAGES)
    if require_complete_stages and not required <= stage_names:
        missing = ", ".join(sorted(required - stage_names))
        raise ProfileArtifactError(f"observation is missing required stages: {missing}")
    counts: dict[str, int] = defaultdict(int)
    for span in spans:
        counts[span["name"]] += 1
    expected_multiplicity = {
        "binding.prepare_batch": 1,
        "binding.parse_requests": 1,
        "native.batch.plan": 1,
        "native.chat.render": request_count,
        "native.chat.tokenize": request_count,
        "binding.destination.allocate": 1,
        "native.destination.execute": 1,
        "binding.numpy.materialize": 1,
    }
    if media_count:
        expected_multiplicity.update(
            {
                "native.media.plan": media_count,
                "native.media.decode_color": media_count,
                "native.media.resize": media_count,
                "native.media.normalize_patchify_layout": media_count,
            }
        )
    if any(counts.get(name) != expected for name, expected in expected_multiplicity.items()):
        raise ProfileArtifactError(
            "observation stage multiplicity does not match request/media scope"
        )
    media_spans = [span for span in spans if span["name"].startswith("native.media.")]
    for span in media_spans:
        scope = span["scope"]
        if any(
            scope.get(name) is None
            for name in (
                "request_index",
                "message_index",
                "content_item_index",
                "media_index",
                "input_index",
            )
        ):
            raise ProfileArtifactError("media span lacks full request/occurrence correlation")
        if not span["shape"] or span["input_bytes"] <= 0:
            raise ProfileArtifactError("media span lacks byte/shape evidence")
    for name in REQUIRED_MEDIA_STAGES if media_count else ():
        actual_scopes = [
            (
                span["scope"]["request_index"],
                span["scope"]["message_index"],
                span["scope"]["content_item_index"],
                span["scope"]["media_index"],
                span["scope"]["input_index"],
            )
            for span in spans
            if span["name"] == name
        ]
        if actual_scopes != list(expected_media_scopes):
            raise ProfileArtifactError(f"{name} scopes differ from exact occurrence inventory")
    for name in ("native.chat.render", "native.chat.tokenize"):
        request_scopes = [span["scope"]["request_index"] for span in spans if span["name"] == name]
        if request_scopes != list(range(request_count)):
            raise ProfileArtifactError(f"{name} request scopes are incomplete or reordered")
    operation_spans = {
        span["name"]: span
        for span in spans
        if span["name"]
        in {
            "binding.prepare_batch",
            "binding.destination.allocate",
            "native.destination.execute",
            "binding.numpy.materialize",
        }
    }
    for span in operation_spans.values():
        if span["output_bytes"] != retained_output_bytes:
            raise ProfileArtifactError(
                "whole-boundary span output bytes differ from materialized arrays"
            )

    allocations = report.get("allocations")
    calls = report.get("calls")
    if not isinstance(allocations, Mapping) or not isinstance(calls, Mapping):
        raise ProfileArtifactError("observation counters must be objects")
    for key in (
        "allocation_count",
        "allocated_bytes",
        "copy_count",
        "copied_bytes",
        "transient_live_bytes",
        "peak_transient_live_bytes",
        "retained_final_output_bytes",
    ):
        value = allocations.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ProfileArtifactError(f"invalid allocation counter {key}")
    if allocations["transient_live_bytes"] != 0:
        raise ProfileArtifactError("successful operation retained transient buffers")
    expected_calls = {
        "public_python_calls": 1,
        "native_batch_calls": 1,
        "native_visual_calls": media_count,
        "python_callbacks": 0,
        "hugging_face_calls": 0,
        "qwen_vl_utils_calls": 0,
        "pillow_calls": 0,
        "torchvision_calls": 0,
    }
    if any(calls.get(name) != expected for name, expected in expected_calls.items()):
        raise ProfileArtifactError(
            "observation call counters do not prove one-call native execution"
        )
    buffer_sequences = []
    transient_events: list[tuple[int, int]] = []
    for buffer in buffers:
        if not isinstance(buffer, Mapping):
            raise ProfileArtifactError("observation buffer event must be an object")
        buffer_sequences.append(buffer.get("sequence"))
        allocated_at = buffer.get("allocated_at_ns")
        released_at = buffer.get("released_at_ns")
        if not isinstance(allocated_at, int) or not 0 <= allocated_at <= duration:
            raise ProfileArtifactError("buffer allocation timestamp is outside the operation")
        if released_at is not None and (
            not isinstance(released_at, int) or not allocated_at <= released_at <= duration
        ):
            raise ProfileArtifactError("buffer release timestamp is outside its lifetime")
        if buffer.get("class") == "transient":
            if released_at is None:
                raise ProfileArtifactError("successful operation has an unreleased transient")
            transient_events.extend(
                ((allocated_at, buffer["bytes"]), (released_at, -buffer["bytes"]))
            )
        elif buffer.get("class") == "retained_output" and released_at is not None:
            raise ProfileArtifactError("retained output was released before report capture")
        elif buffer.get("class") == "discarded_output":
            raise ProfileArtifactError("successful operation contains discarded output")
    if buffer_sequences != list(range(len(buffers))):
        raise ProfileArtifactError("buffer event sequences must be contiguous")
    if allocations["allocation_count"] != len(buffers) or allocations["allocated_bytes"] != sum(
        buffer["bytes"] for buffer in buffers
    ):
        raise ProfileArtifactError("allocation counters do not reconcile with buffer events")
    for sequence, copy_event in enumerate(copies):
        if not isinstance(copy_event, Mapping):
            raise ProfileArtifactError("observation copy event must be an object")
        if copy_event.get("sequence") != sequence:
            raise ProfileArtifactError("copy event sequences must be contiguous")
        if not isinstance(copy_event.get("name"), str) or not copy_event["name"]:
            raise ProfileArtifactError("copy event name is invalid")
        if (
            not isinstance(copy_event.get("bytes"), int)
            or isinstance(copy_event["bytes"], bool)
            or copy_event["bytes"] < 0
        ):
            raise ProfileArtifactError("copy event bytes are invalid")
        copy_scope = copy_event.get("scope")
        if not isinstance(copy_scope, Mapping) or set(copy_scope) != {
            "request_index",
            "message_index",
            "content_item_index",
            "media_index",
            "input_index",
        }:
            raise ProfileArtifactError("copy event scope is invalid")
    if allocations["copy_count"] != len(copies) or allocations["copied_bytes"] != sum(
        copy_event["bytes"] for copy_event in copies
    ):
        raise ProfileArtifactError("copy counters do not reconcile with copy events")
    grouped: dict[int, dict[str, int]] = defaultdict(lambda: {"allocate": 0, "release": 0})
    for timestamp, delta in transient_events:
        key = "allocate" if delta >= 0 else "release"
        grouped[timestamp][key] += abs(delta)
    live = 0
    minimum_peak = 0
    maximum_peak = 0
    for timestamp in sorted(grouped):
        values = grouped[timestamp]
        allocated = values["allocate"]
        released = values["release"]
        if released > live + allocated:
            raise ProfileArtifactError("transient buffer lifetime underflowed")
        maximum_peak = max(maximum_peak, live + allocated)
        live = live + allocated - released
        minimum_peak = max(minimum_peak, live)
    reported_peak = allocations["peak_transient_live_bytes"]
    if live != 0 or not minimum_peak <= reported_peak <= maximum_peak:
        raise ProfileArtifactError("transient live/peak counters fall outside tie-aware bounds")
    retained = sum(
        int(buffer["bytes"])
        for buffer in buffers
        if isinstance(buffer, Mapping) and buffer.get("class") == "retained_output"
    )
    if retained != allocations["retained_final_output_bytes"]:
        raise ProfileArtifactError("retained output buffer events do not reconcile with counters")
    if retained != retained_output_bytes:
        raise ProfileArtifactError("retained output bytes differ from actual array signatures")


def validate_sampled_profile(
    sampled: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    observed_signatures: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> None:
    coordinate = (
        sampled.get("profile_alias"),
        sampled.get("case_id"),
        sampled.get("thread_budget"),
    )
    expected_regime = "one" if coordinate[2] == 1 else "production"
    if sampled.get("thread_regime") != expected_regime:
        raise ProfileArtifactError("sampled profile thread regime is relabeled")
    if sampled.get("iterations", 0) <= 0:
        raise ProfileArtifactError("sampled worker completed no whole-operation iterations")
    if sampled.get("before_signature") != sampled.get("after_signature"):
        raise ProfileArtifactError("sampled pre/post output signatures differ")
    if sampled.get("before_signature") != observed_signatures.get(coordinate):
        raise ProfileArtifactError("sampled and observed outputs differ at one coordinate")
    if sampled.get("whole_boundary_marker") != "qwen_mm_profile_iteration":
        raise ProfileArtifactError("sampled profile lacks the frozen whole-boundary marker")
    sampler = sampled.get("sampler")
    if not isinstance(sampler, Mapping):
        raise ProfileArtifactError("sampled profile lacks sampler provenance")
    architecture = sampled.get("architecture_family")
    expected_sampler = protocol["samplers"].get(architecture)
    if expected_sampler != SAMPLER_PROTOCOLS.get(architecture):
        raise ProfileArtifactError("sampled profile architecture/sampler protocol is invalid")
    if (
        sampler.get("name") != expected_sampler["name"]
        or sampler.get("native") is not expected_sampler["native"]
        or sampler.get("duration_seconds") != expected_sampler["duration_seconds"]
        or sampler.get("errors") != 0
        or not isinstance(sampler.get("version"), str)
        or not sampler["version"]
        or not isinstance(sampler.get("install_provenance"), str)
        or not sampler["install_provenance"]
        or not isinstance(sampler.get("binary_sha256"), str)
        or len(sampler["binary_sha256"]) != 64
        or sampler.get("binary_bytes", 0) <= 0
    ):
        raise ProfileArtifactError("sampled profile does not prove its frozen sampler capture")
    command = shlex.split(str(sampler.get("command", "")))
    worker = sampled.get("worker")
    if not isinstance(worker, Mapping) or worker.get("target_pid", 0) <= 0:
        raise ProfileArtifactError("sample worker PID provenance is invalid")
    worker_command = shlex.split(str(worker.get("command", "")))
    if (
        len(worker_command) != 6
        or not worker_command[0]
        or worker_command[1:5]
        != ["-m", "qwen_mm_reference.profile_v1", "_sample_worker", "--config"]
        or not worker_command[5]
        or worker_command[5].startswith("-")
    ):
        raise ProfileArtifactError(
            "sample worker command does not have the frozen authenticated shape"
        )
    if architecture == "x86_64":
        if (
            sampler.get("version") != "py-spy 0.4.1"
            or sampler.get("binary_path") != expected_sampler["binary_path"]
            or sampler.get("rate_hz") != expected_sampler["rate_hz"]
            or "interval_ms" in sampler
            or "conversion_version" in sampler
        ):
            raise ProfileArtifactError("x86 sampled profile is not pinned py-spy evidence")
        expected_prefix = [
            str(sampler.get("binary_path")),
            "record",
            "--rate",
            str(expected_sampler["rate_hz"]),
            "--format",
            "raw",
            "-o",
        ]
        if (
            len(command) != len(expected_prefix) + 2 + len(worker_command)
            or command[: len(expected_prefix)] != expected_prefix
            or not command[len(expected_prefix)]
            or command[len(expected_prefix)].startswith("-")
            or command[len(expected_prefix) + 1] != "--"
            or command[len(expected_prefix) + 2 :] != worker_command
        ):
            raise ProfileArtifactError(
                "x86 sampler command does not exactly launch the authenticated child protocol"
            )
    else:
        if (
            "PROGRAM:sample" not in sampler["version"]
            or sampler.get("binary_path") != "/usr/bin/sample"
            or sampler.get("interval_ms") != expected_sampler["interval_ms"]
            or sampler.get("conversion_version") != expected_sampler["conversion_version"]
            or "rate_hz" in sampler
        ):
            raise ProfileArtifactError("ARM sampled profile is not native macOS sample evidence")
        required_command = {
            str(sampled["worker"]["target_pid"]),
            str(expected_sampler["duration_seconds"]),
            str(expected_sampler["interval_ms"]),
            "-mayDie",
            "-file",
        }
        if (
            not command
            or command[0] != sampler.get("binary_path")
            or not required_command <= set(command)
        ):
            raise ProfileArtifactError("sampler command does not bind native/raw/rate protocol")
    raw_path, raw_bytes = _read_authenticated_artifact(sampled["artifacts"]["raw"])
    collapsed_path, collapsed_bytes = _read_authenticated_artifact(
        sampled["artifacts"]["collapsed"]
    )
    if raw_path == collapsed_path:
        raise ProfileArtifactError(
            "raw and canonical collapsed profiles must be distinct artifacts"
        )
    try:
        raw = raw_bytes.decode()
        collapsed = collapsed_bytes.decode()
    except UnicodeDecodeError as error:
        raise ProfileArtifactError("sampled profile artifacts must be UTF-8") from error
    if architecture == "arm64":
        expected_collapsed, sample_count, unique_stacks = _parse_macos_sample(raw)
    else:
        _, raw_sample_count, _ = _parse_collapsed(raw)
        if sampler.get("reported_sample_count") != raw_sample_count:
            raise ProfileArtifactError(
                "py-spy raw sample total differs from its reported sample count"
            )
        expected_collapsed, sample_count, unique_stacks = _parse_collapsed(
            raw, required_frame="qwen_mm_profile_iteration"
        )
    if collapsed != expected_collapsed:
        raise ProfileArtifactError("collapsed profile is not the canonical reduction of raw stacks")
    collapsed_frames = [
        frame for line in collapsed.splitlines() for frame in line.rsplit(" ", 1)[0].split(";")
    ]
    expected_samples = (
        expected_sampler["duration_seconds"] * 1000 // expected_sampler["interval_ms"]
        if architecture == "arm64"
        else expected_sampler["rate_hz"] * expected_sampler["duration_seconds"]
    )
    plausible_minimum = max(1, expected_samples // 4)
    plausible_maximum = expected_samples * 2
    if not plausible_minimum <= sample_count <= plausible_maximum:
        raise ProfileArtifactError("sample count is implausible for configured rate/duration")
    if sampler.get("sample_count") != sample_count or sampler.get("unique_stacks") != unique_stacks:
        raise ProfileArtifactError("sampler counts do not match raw stacks")
    if architecture == "x86_64" and "qwen_mm_profile_iteration" not in collapsed:
        raise ProfileArtifactError("sampled stacks omit the whole Python boundary marker")
    if architecture == "arm64" and (
        not any(marker in collapsed for marker in ("PyEval", "eval_frame"))
        or not any(_is_qwen_core_frame(frame) for frame in collapsed_frames)
        or not any(_is_qwen_binding_frame(frame) for frame in collapsed_frames)
    ):
        raise ProfileArtifactError(
            "macOS samples omit CPython/binding/core whole-boundary evidence"
        )
    rankings = _stack_rankings(collapsed)
    if architecture == "arm64":
        native_frames = {frame for frame in collapsed_frames if _is_qwen_native_frame(frame)}
        symbolized = [frame for frame in native_frames if _is_symbolized_qwen_native_frame(frame)]
        if not symbolized:
            raise ProfileArtifactError(
                "native sampled frames are unsymbolized offsets/unknowns only"
            )
        if rankings["native_sample_share"] < 0.01:
            raise ProfileArtifactError("native sampled frames represent less than 1% of samples")
    if sampled.get("rankings") != rankings:
        raise ProfileArtifactError("sampled self/inclusive rankings do not match raw stacks")
    worker_path, _ = _read_authenticated_artifact(worker["source"])
    if worker_path.resolve() != Path(__file__).resolve():
        raise ProfileArtifactError("sample worker source is not the profile_v1 implementation")
    _, worker_result_bytes = _read_authenticated_artifact(worker.get("result", {}))
    try:
        worker_result = json.loads(worker_result_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProfileArtifactError("sample worker result is not valid JSON") from error
    expected_worker_fields = {
        "pid": worker["target_pid"],
        "iterations": sampled["iterations"],
        "input_fingerprint": sampled["input_fingerprint"],
        "logical_input_fingerprint": sampled["logical_input_fingerprint"],
        "before_signature": sampled["before_signature"],
        "after_signature": sampled["after_signature"],
        "profile_alias": sampled["profile_alias"],
        "case_id": sampled["case_id"],
        "thread_budget": sampled["thread_budget"],
        "runtime_identity": worker["runtime_identity"],
    }
    if not isinstance(worker_result, Mapping) or any(
        worker_result.get(name) != value for name, value in expected_worker_fields.items()
    ):
        raise ProfileArtifactError("sample worker result differs from bundled provenance")
    measurement_window = worker_result.get("measurement_window")
    if architecture == "x86_64":
        expected_window_fields = {
            "requested_duration_ns",
            "started_ns",
            "deadline_ns",
            "final_iteration_started_ns",
            "completed_ns",
            "postcheck_completed_ns",
        }
        if not isinstance(measurement_window, Mapping) or set(measurement_window) != (
            expected_window_fields
        ):
            raise ProfileArtifactError("x86 worker lacks its authenticated measurement window")
        if any(
            not isinstance(measurement_window[name], int)
            or isinstance(measurement_window[name], bool)
            for name in expected_window_fields
        ):
            raise ProfileArtifactError("x86 worker measurement timestamps must be integers")
        requested_ns = expected_sampler["duration_seconds"] * 1_000_000_000
        started_ns = measurement_window["started_ns"]
        deadline_ns = measurement_window["deadline_ns"]
        final_iteration_started_ns = measurement_window["final_iteration_started_ns"]
        completed_ns = measurement_window["completed_ns"]
        postcheck_completed_ns = measurement_window["postcheck_completed_ns"]
        if (
            measurement_window["requested_duration_ns"] != requested_ns
            or deadline_ns - started_ns != requested_ns
            or not started_ns <= final_iteration_started_ns < deadline_ns
            or completed_ns < deadline_ns
            or postcheck_completed_ns < completed_ns
        ):
            raise ProfileArtifactError("x86 worker measurement window violates the 2 s protocol")
    elif measurement_window is not None:
        raise ProfileArtifactError("ARM attach worker unexpectedly claims x86 duration control")
    _validate_thread_settings(
        worker_result.get("thread_settings"),
        sampled["thread_budget"],
        label="sample worker",
    )


def summarize_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    architectures: dict[str, Any] = {}
    for capture in bundle["captures"]:
        architecture = capture["host"]["architecture_family"]
        time_totals: dict[str, list[int]] = defaultdict(list)
        allocation_totals: dict[tuple[str, str], list[int]] = defaultdict(list)
        lifetime_totals: dict[tuple[str, str], list[int]] = defaultdict(list)
        copy_totals: dict[str, dict[str, int]] = defaultdict(
            lambda: {"copied_bytes": 0, "copy_count": 0}
        )
        coordinate_time: dict[tuple[str, str, int], dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for operation in capture["observations"]:
            observation = operation["observation"]
            coordinate = (
                operation["profile_alias"],
                operation["case_id"],
                operation["thread_budget"],
            )
            for span in observation["spans"]:
                time_totals[span["name"]].append(span["exclusive_duration_ns"])
                coordinate_time[coordinate][span["name"]] += span["exclusive_duration_ns"]
            for buffer in observation["buffers"]:
                key = (buffer["name"], buffer["class"])
                allocation_totals[key].append(buffer["bytes"])
                released = buffer["released_at_ns"]
                end = observation["duration_ns"] if released is None else released
                lifetime_totals[key].append(max(0, end - buffer["allocated_at_ns"]))
            for copy_event in observation["copies"]:
                copy_totals[copy_event["name"]]["copied_bytes"] += copy_event["bytes"]
                copy_totals[copy_event["name"]]["copy_count"] += 1
        total_exclusive = sum(sum(values) for values in time_totals.values())
        timing = [
            {
                "stage": name,
                "exclusive_duration_ns": sum(values),
                "mean_exclusive_duration_ns": sum(values) // len(values),
                "share_of_observed_stage_time": (
                    sum(values) / total_exclusive if total_exclusive else 0.0
                ),
                "sample_count": len(values),
            }
            for name, values in time_totals.items()
        ]
        allocations = [
            {
                "buffer": name,
                "class": class_name,
                "allocated_bytes": sum(values),
                "allocation_count": len(values),
            }
            for (name, class_name), values in allocation_totals.items()
        ]
        copies = [
            {
                "copy": name,
                **values,
            }
            for name, values in copy_totals.items()
        ]
        lifetimes = [
            {
                "buffer": name,
                "class": class_name,
                "total_lifetime_ns": sum(values),
                "mean_lifetime_ns": sum(values) // len(values),
                "sample_count": len(values),
            }
            for (name, class_name), values in lifetime_totals.items()
        ]
        timing.sort(key=lambda item: (-item["exclusive_duration_ns"], item["stage"]))
        allocations.sort(key=lambda item: (-item["allocated_bytes"], item["buffer"], item["class"]))
        copies.sort(
            key=lambda item: (
                -item["copied_bytes"],
                item["copy"],
            )
        )
        lifetimes.sort(key=lambda item: (-item["total_lifetime_ns"], item["buffer"], item["class"]))
        sampled_by_coordinate = {
            (
                sampled["profile_alias"],
                sampled["case_id"],
                sampled["thread_budget"],
            ): sampled
            for sampled in capture["sampled_profiles"]
        }
        coordinates = []
        for coordinate in sorted(coordinate_time):
            sampled = sampled_by_coordinate[coordinate]
            stage_values = coordinate_time[coordinate]
            stage_total = sum(stage_values.values())
            coordinates.append(
                {
                    "profile_alias": coordinate[0],
                    "case_id": coordinate[1],
                    "thread_budget": coordinate[2],
                    "observed_exclusive": [
                        {
                            "stage": stage,
                            "exclusive_duration_ns": duration,
                            "share": duration / stage_total if stage_total else 0.0,
                        }
                        for stage, duration in sorted(
                            stage_values.items(), key=lambda item: (-item[1], item[0])
                        )
                    ],
                    "sampled_inclusive": sampled["rankings"]["inclusive"],
                    "sampled_self": sampled["rankings"]["self"],
                }
            )
        architectures[architecture] = {
            "timing": timing,
            "allocations": allocations,
            "copies": copies,
            "buffer_lifetimes": lifetimes,
            "coordinates": coordinates,
        }
    return {
        "timing_semantics": (
            "ranks exclusive_duration_ns; raw inclusive durations remain in each observation "
            "and are not summed"
        ),
        "architectures": architectures,
    }


def validate_bundle(bundle: Mapping[str, Any], *, require_arm_x86: bool = True) -> None:
    try:
        _validate_bundle(bundle, require_arm_x86=require_arm_x86)
    except ProfileArtifactError:
        raise
    except (AttributeError, IndexError, KeyError, OSError, TypeError, ValueError) as error:
        raise ProfileArtifactError(f"profile bundle is malformed: {error}") from error


def _validate_bundle(bundle: Mapping[str, Any], *, require_arm_x86: bool) -> None:
    errors = sorted(
        Draft202012Validator(_schema()).iter_errors(bundle), key=lambda item: list(item.path)
    )
    if errors:
        first = errors[0]
        location = "/".join(str(item) for item in first.path) or "<root>"
        raise ProfileArtifactError(
            f"profile schema validation failed at {location}: {first.message}"
        )
    protocol = bundle["protocol"]
    if tuple(protocol["profiles"]) != DEFAULT_PROFILES or tuple(protocol["cases"]) != DEFAULT_CASES:
        raise ProfileArtifactError("profile bundle does not cover the canonical D1 aliases/cases")
    if protocol["repetitions"] != DEFAULT_REPETITIONS:
        raise ProfileArtifactError("profile bundle does not use the frozen observation repetitions")
    if protocol.get("event_capacity") != 4096 or protocol.get("samplers") != SAMPLER_PROTOCOLS:
        raise ProfileArtifactError("profile observation/sampler settings are not frozen D1 values")
    workload_identity = protocol["workload"]
    workload_relative = Path(workload_identity.get("path", ""))
    if workload_relative.is_absolute() or ".." in workload_relative.parts:
        raise ProfileArtifactError("profile workload path must be safe and repository-relative")
    workload_path = repository_root() / workload_relative
    if _sha256_path(workload_path) != workload_identity.get("sha256"):
        raise ProfileArtifactError("profile workload changed after capture")
    schema_relative = Path(workload_identity.get("schema_path", ""))
    if schema_relative.is_absolute() or ".." in schema_relative.parts:
        raise ProfileArtifactError(
            "profile workload schema path must be safe and repository-relative"
        )
    if _sha256_path(repository_root() / schema_relative) != workload_identity.get("schema_sha256"):
        raise ProfileArtifactError("profile workload schema changed after capture")
    workload = load_workload(workload_path)
    selected_cases = select_cases(workload, case_ids=protocol["cases"])
    expected_payloads: dict[str, dict[str, Any]] = {}
    for case in selected_cases:
        payload = materialize_case(case)
        expected_payloads[case["case_id"]] = {
            "boundary": case["boundary"],
            "release_name": case.get("release_name"),
            "source_kind": case["source"].get("kind"),
            "media_count": len(payload.buffers),
            "request_count": len(payload.messages),
            "input_fingerprint": payload.input_fingerprint,
            "logical_input_fingerprint": payload.logical_input_fingerprint,
            "media_scopes": _expected_media_scopes(payload),
        }
    if protocol["thread_budgets"] != [1, 4] or protocol["thread_regimes"] != {
        "one": 1,
        "production": 4,
    }:
        raise ProfileArtifactError("profile thread mapping must be one=1 and production=4")
    captures = bundle["captures"]
    architectures = [capture["host"]["architecture_family"] for capture in captures]
    if len(architectures) != len(set(architectures)):
        raise ProfileArtifactError("profile bundle has duplicate architecture captures")
    if require_arm_x86 and set(architectures) != {"arm64", "x86_64"}:
        raise ProfileArtifactError("combined evidence requires exactly ARM64 and x86_64 captures")
    expected_coordinates = _operation_coordinates(protocol)
    source_coordinates = {
        (capture["source"]["revision"], capture["source"]["tree_sha256"]) for capture in captures
    }
    if len(source_coordinates) != 1:
        raise ProfileArtifactError("architecture captures must use the exact same source")
    expected_revision, expected_digest = next(iter(source_coordinates))
    evidence_paths = {
        identity["path"]
        for capture in captures
        for identity in (
            capture["build"]["log"],
            capture["build"]["wheel"],
            capture["phase_c_report"],
            capture["paired_benchmark"],
            *(
                item
                for sampled in capture["sampled_profiles"]
                for item in (
                    sampled["artifacts"]["raw"],
                    sampled["artifacts"]["collapsed"],
                    sampled["worker"]["result"],
                )
            ),
        )
    }
    digest, file_count, total_bytes = _source_tree_digest_at_revision(
        expected_revision, evidence_paths
    )
    if digest != expected_digest:
        raise ProfileArtifactError("profile source digest differs from its recorded Git tree")
    for capture in captures:
        source = capture["source"]
        if source["file_count"] is not None and source["file_count"] != file_count:
            raise ProfileArtifactError(
                "profile source file count differs from its recorded Git tree"
            )
        if source["total_bytes"] is not None and source["total_bytes"] != total_bytes:
            raise ProfileArtifactError(
                "profile source byte count differs from its recorded Git tree"
            )
    for capture in captures:
        source = capture["source"]
        if source["dirty"] is not False:
            raise ProfileArtifactError("profile evidence must come from exact clean source")
        host = capture["host"]
        _validate_host_provenance(host)
        cpu_text = str(host.get("cpu_description", "")).lower()
        if any(marker in cpu_text for marker in ("qemu", "tcg", "software emulation")):
            raise ProfileArtifactError("profile host contains an emulation marker")
        observed_coordinates: set[tuple[str, str, int, int]] = set()
        observed_signatures: dict[tuple[str, str, int], Mapping[str, Any]] = {}
        observed_fingerprints: dict[tuple[str, str, int], str] = {}
        observed_logical_fingerprints: dict[tuple[str, str, int], str] = {}
        foreign_architecture = host["architecture_family"] != architecture_family()
        for operation in capture["observations"]:
            coordinate = (
                operation["profile_alias"],
                operation["case_id"],
                operation["thread_budget"],
                operation["repetition"],
            )
            if coordinate in observed_coordinates:
                raise ProfileArtifactError("profile operation coordinates are duplicated")
            observed_coordinates.add(coordinate)
            expected_payload = expected_payloads[operation["case_id"]]
            if any(
                operation.get(field) != expected_payload[field]
                for field in (
                    "boundary",
                    "release_name",
                    "media_count",
                    "logical_input_fingerprint",
                )
            ):
                raise ProfileArtifactError(
                    "observed operation is relabeled from its workload payload"
                )
            if (
                not (
                    foreign_architecture and expected_payload["source_kind"] == "generated_encoded"
                )
                and operation.get("input_fingerprint") != expected_payload["input_fingerprint"]
            ):
                raise ProfileArtifactError(
                    "observed operation exact input differs from its workload payload"
                )
            sample_coordinate = coordinate[:3]
            prior_signature = observed_signatures.setdefault(
                sample_coordinate, operation["output_signature"]
            )
            if prior_signature != operation["output_signature"]:
                raise ProfileArtifactError("observed output signature changed across repetitions")
            prior_fingerprint = observed_fingerprints.setdefault(
                sample_coordinate, operation["input_fingerprint"]
            )
            if prior_fingerprint != operation["input_fingerprint"]:
                raise ProfileArtifactError("observed input fingerprint changed across repetitions")
            prior_logical_fingerprint = observed_logical_fingerprints.setdefault(
                sample_coordinate, operation["logical_input_fingerprint"]
            )
            if prior_logical_fingerprint != operation["logical_input_fingerprint"]:
                raise ProfileArtifactError(
                    "observed logical input fingerprint changed across repetitions"
                )
            expected_regime = "one" if operation["thread_budget"] == 1 else "production"
            if operation["thread_regime"] != expected_regime:
                raise ProfileArtifactError("operation thread regime is relabeled")
            _validate_thread_settings(
                operation["thread_settings"],
                operation["thread_budget"],
                label="operation",
            )
            if operation["observation"].get("event_capacity") != protocol["event_capacity"]:
                raise ProfileArtifactError("observation event capacity differs from protocol")
            validate_observation(
                operation["observation"],
                media_count=operation["media_count"],
                request_count=expected_payload["request_count"],
                retained_output_bytes=sum(
                    item["nbytes"] for item in operation["output_signature"].values()
                ),
                expected_media_scopes=expected_payload["media_scopes"],
            )
        if observed_coordinates != expected_coordinates:
            raise ProfileArtifactError(
                "profile capture does not cover the protocol coordinate matrix"
            )
        sampled_build_runtime = (
            capture.get("build", {}).get("candidate_identity", {}).get("runtime_identity")
        )
        if not isinstance(sampled_build_runtime, Mapping):
            raise ProfileArtifactError("profile build lacks its candidate runtime identity")
        sampled_coordinates = {
            (
                sampled["profile_alias"],
                sampled["case_id"],
                sampled["thread_budget"],
            )
            for sampled in capture["sampled_profiles"]
        }
        if sampled_coordinates != _sample_coordinates(protocol) or len(sampled_coordinates) != len(
            capture["sampled_profiles"]
        ):
            raise ProfileArtifactError(
                "sampled profiles do not cover the protocol coordinate matrix"
            )
        sampled_artifact_paths = [
            identity["path"]
            for sampled in capture["sampled_profiles"]
            for identity in (
                sampled["artifacts"]["raw"],
                sampled["artifacts"]["collapsed"],
                sampled["worker"]["result"],
            )
        ]
        if len(sampled_artifact_paths) != len(set(sampled_artifact_paths)):
            raise ProfileArtifactError("sampled coordinates reuse raw/collapsed/worker artifacts")
        for sampled in capture["sampled_profiles"]:
            if sampled.get("architecture_family") != host["architecture_family"]:
                raise ProfileArtifactError("sampled profile architecture differs from its host")
            validate_sampled_profile(
                sampled,
                protocol=protocol,
                observed_signatures=observed_signatures,
            )
            if sampled["worker"]["runtime_identity"] != sampled_build_runtime:
                raise ProfileArtifactError(
                    "sample worker runtime differs from the capture's profiled build runtime"
                )
            sampled_coordinate = (
                sampled["profile_alias"],
                sampled["case_id"],
                sampled["thread_budget"],
            )
            if sampled["input_fingerprint"] != observed_fingerprints[sampled_coordinate]:
                raise ProfileArtifactError("sampled and observed input fingerprints differ")
            if (
                sampled["logical_input_fingerprint"]
                != observed_logical_fingerprints[sampled_coordinate]
            ):
                raise ProfileArtifactError("sampled and observed logical input fingerprints differ")
        benchmark = capture["paired_benchmark"]
        _, benchmark_bytes = _read_authenticated_artifact(benchmark)
        try:
            benchmark_result = json.loads(benchmark_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProfileArtifactError("paired benchmark artifact is not valid JSON") from error
        if not isinstance(benchmark_result, Mapping):
            raise ProfileArtifactError("paired benchmark artifact must contain an object")
        _validate_paired_benchmark(
            benchmark_result,
            profiles=protocol["profiles"],
            cases=protocol["cases"],
            budgets=protocol["thread_budgets"],
        )
        if benchmark_result.get("workload") != protocol["workload"]:
            raise ProfileArtifactError("paired benchmark and profile workload provenance differ")
        if benchmark_result.get("protocol", {}).get("candidate_identity") != benchmark.get(
            "candidate_identity"
        ):
            raise ProfileArtifactError("paired benchmark identity differs from raw artifact")
        if benchmark_result.get("architecture_family") != benchmark["architecture_family"]:
            raise ProfileArtifactError("paired benchmark architecture differs from raw artifact")
        if benchmark["architecture_family"] != capture["host"]["architecture_family"]:
            raise ProfileArtifactError("paired benchmark architecture differs from profile capture")
        build = capture["build"]
        if build.get("mode") != "release" or build.get("label") != PROFILE_BUILD_LABEL:
            raise ProfileArtifactError("profile build mode/label is not the profiled release")
        _validate_profile_build_command(build.get("command", ""))
        build_runtime = build.get("candidate_identity", {}).get("runtime_identity")
        benchmark_runtime = benchmark.get("candidate_identity", {}).get("runtime_identity")
        if not isinstance(build_runtime, Mapping) or build_runtime != benchmark_runtime:
            raise ProfileArtifactError("observed and paired benchmark runtime identities differ")
        native_hash = build_runtime.get("native_artifact_sha256")
        if not isinstance(native_hash, str) or len(native_hash) != 64:
            raise ProfileArtifactError("profile build lacks the native artifact hash")
        _, build_log = _read_authenticated_artifact(build.get("log", {}))
        _, wheel = _read_authenticated_artifact(build.get("wheel", {}))
        if not build_log or not wheel:
            raise ProfileArtifactError("profile build log and wheel evidence must be non-empty")
        _validate_wheel_runtime(wheel, build_runtime)
        phase_c_path, phase_c_bytes = _read_authenticated_artifact(capture["phase_c_report"])
        try:
            phase_c_report = json.loads(phase_c_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProfileArtifactError("authenticated Phase C report is not valid JSON") from error
        phase_c_gate = benchmark_result["release_eligibility"]["phase_c"]
        if phase_c_gate.get("report_sha256") != capture["phase_c_report"]["sha256"]:
            raise ProfileArtifactError("paired Phase C report hash differs from bundled report")
        phase_c_runtime = phase_c_gate.get("evidence", {}).get("candidate_runtime_identity")
        if (
            phase_c_runtime != build_runtime
            or phase_c_report.get("candidate", {}).get("runtime_identity") != build_runtime
        ):
            raise ProfileArtifactError("Phase C report runtime differs from profiled wheel runtime")
        assets_relative = Path(str(phase_c_gate.get("assets_root", "")))
        if assets_relative.is_absolute() or ".." in assets_relative.parts:
            raise ProfileArtifactError(
                "Phase C assets root must be portable and repository-relative"
            )
        try:
            validate_phase_c_report(
                phase_c_report,
                assets_root=repository_root() / assets_relative,
            )
        except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
            raise ProfileArtifactError(
                "authenticated Phase C report is semantically invalid"
            ) from error
        gate_material = {
            "report_sha256": capture["phase_c_report"]["sha256"],
            "evidence": phase_c_gate.get("evidence"),
        }
        if (
            phase_c_gate.get("gate_fingerprint")
            != hashlib.sha256(_canonical_json(gate_material)).hexdigest()
        ):
            raise ProfileArtifactError("Phase C gate fingerprint is stale or tampered")
        if not phase_c_path.is_file():
            raise ProfileArtifactError("Phase C report artifact is missing")
        paired_coordinates: set[tuple[str, str, int]] = set()
        for pair in benchmark_result["pairs"]:
            candidate = pair["implementations"]["candidate"]
            coordinate = (
                candidate["profile_alias"],
                candidate["case_id"],
                pair["thread_budget"],
            )
            paired_coordinates.add(coordinate)
            if candidate["input_fingerprint"] != observed_fingerprints.get(coordinate):
                raise ProfileArtifactError("paired candidate input differs from profile workload")
            if candidate["logical_input_fingerprint"] != observed_logical_fingerprints.get(
                coordinate
            ):
                raise ProfileArtifactError(
                    "paired candidate logical input differs from profile workload"
                )
            if candidate["output_signature"] != observed_signatures.get(coordinate):
                raise ProfileArtifactError("paired candidate output differs from profile output")
        if paired_coordinates != _sample_coordinates(protocol):
            raise ProfileArtifactError("paired benchmark omits profile coordinates")
    expected_summary = summarize_bundle({**bundle, "summary": {}})
    if bundle["summary"] != expected_summary:
        raise ProfileArtifactError("profile summary does not match raw observations")


def _validate_paired_benchmark(
    result: Mapping[str, Any],
    *,
    profiles: Sequence[str],
    cases: Sequence[str],
    budgets: Sequence[int],
    live_runtime: bool = False,
    phase_c_report_override: Path | None = None,
) -> None:
    try:
        if live_runtime:
            validate_result(result, phase_c_report_override=phase_c_report_override)
        else:
            validate_result_portable(result)
    except BenchmarkProtocolError as error:
        raise ProfileArtifactError(f"paired benchmark validation failed: {error}") from error
    protocol = result["protocol"]
    if protocol["candidate_adapter"] != BASELINE_ADAPTER:
        raise ProfileArtifactError("profile evidence must pair with the unobserved qwen-mm adapter")
    if protocol["self_test_only"]:
        raise ProfileArtifactError("synthetic paired benchmark cannot support profile evidence")
    if protocol["build_labels"] != [PROFILE_BUILD_LABEL]:
        raise ProfileArtifactError("paired benchmark must use the distinct profiled-release build")
    phase_c = result.get("release_eligibility", {}).get("phase_c", {})
    if phase_c.get("status") != "pass":
        raise ProfileArtifactError(
            "paired benchmark must have a passing exact-runtime Phase C gate"
        )
    if protocol["thread_regimes"] != ["one", "production"] or protocol.get(
        "thread_budget_mapping"
    ) != {"one": 1, "production": 4}:
        raise ProfileArtifactError("paired benchmark must map one=1 and production=4")
    if set(protocol["profiles"]) != set(profiles) or set(protocol["cases"]) != set(cases):
        raise ProfileArtifactError(
            "paired benchmark and profile capture select different cases/profiles"
        )
    paired_budgets = {pair["thread_budget"] for pair in result["pairs"]}
    if paired_budgets != set(budgets):
        raise ProfileArtifactError(
            "paired benchmark and profile capture use different thread budgets"
        )


def _capture_operation(
    *,
    adapter: Any,
    payload: Any,
    profile: str,
    case: Mapping[str, Any],
    budget: int,
    repetition: int,
    thread_settings: Mapping[str, Any],
) -> dict[str, Any]:
    arrays = normalize_outputs(adapter.run(payload), payload)
    signature = output_signature(arrays)
    retained_output_bytes = sum(item["nbytes"] for item in signature.values())
    report = copy.deepcopy(dict(adapter.observation_report()))
    validate_observation(
        report,
        media_count=len(payload.buffers),
        request_count=len(payload.messages),
        retained_output_bytes=retained_output_bytes,
        expected_media_scopes=_expected_media_scopes(payload),
    )
    return {
        "profile_alias": profile,
        "case_id": case["case_id"],
        "release_name": case.get("release_name"),
        "boundary": case["boundary"],
        "thread_budget": budget,
        "thread_regime": "one" if budget == 1 else "production",
        "thread_settings": copy.deepcopy(dict(thread_settings)),
        "repetition": repetition,
        "media_count": len(payload.buffers),
        "input_fingerprint": payload.input_fingerprint,
        "logical_input_fingerprint": payload.logical_input_fingerprint,
        "output_signature": signature,
        "observation": report,
    }


def capture_bundle(
    *,
    workload_path: Path,
    benchmark_result_path: Path,
    profiles: Sequence[str],
    cases: Sequence[str],
    thread_budgets: Sequence[int],
    repetitions: int,
    build_command: str,
    build_log_path: Path,
    wheel_path: Path,
    phase_c_report_path: Path,
    artifact_directory: Path,
    benchmark_phase_c_source_report_path: Path | None = None,
    artifact_publish_directory: Path | None = None,
    py_spy: str = "py-spy",
    sampler_rate_hz: int = 99,
    sampler_duration_seconds: int = 2,
    source_revision: str | None = None,
    source_digest: str | None = None,
    source_clean: bool = False,
    event_capacity: int = 4096,
) -> dict[str, Any]:
    if repetitions <= 0:
        raise ProfileArtifactError("profile repetitions must be positive")
    if event_capacity != 4096:
        raise ProfileArtifactError("D1 observation capacity is frozen to 4096 events")
    if sampler_rate_hz != 99 or sampler_duration_seconds != 2:
        raise ProfileArtifactError("D1 sampler settings are frozen to 99 Hz for 2 seconds")
    if list(thread_budgets) != list(DEFAULT_THREAD_BUDGETS):
        raise ProfileArtifactError("D1 profile thread budgets are locked to one=1, production=4")
    if tuple(profiles) != DEFAULT_PROFILES or tuple(cases) != DEFAULT_CASES:
        raise ProfileArtifactError(
            "D1 profile capture requires the complete frozen profile/case matrix"
        )
    _validate_profile_build_command(build_command)
    workload_path = (
        workload_path if workload_path.is_absolute() else repository_root() / workload_path
    )
    workload = load_workload(workload_path)
    selected = select_cases(workload, case_ids=cases)
    benchmark = _load_json(benchmark_result_path)
    _validate_paired_benchmark(
        benchmark,
        profiles=profiles,
        cases=cases,
        budgets=thread_budgets,
        live_runtime=True,
        phase_c_report_override=benchmark_phase_c_source_report_path,
    )

    def publish(path: Path) -> Path | None:
        return (
            None if artifact_publish_directory is None else artifact_publish_directory / path.name
        )

    phase_c_identity = _artifact_identity(
        phase_c_report_path, published_path=publish(phase_c_report_path)
    )
    build_log_identity = _artifact_identity(build_log_path, published_path=publish(build_log_path))
    wheel_identity = _artifact_identity(wheel_path, published_path=publish(wheel_path))
    if build_log_identity["bytes"] <= 0 or wheel_identity["bytes"] <= 0:
        raise ProfileArtifactError("profile build log and wheel artifacts must be non-empty")
    build_candidate_identity = candidate_artifact_identity(DEFAULT_ADAPTER)
    build_runtime = build_candidate_identity.get("runtime_identity")
    if not isinstance(build_runtime, Mapping):
        raise ProfileArtifactError("profiled wheel runtime identity cannot be resolved")
    _validate_wheel_runtime(wheel_path.read_bytes(), build_runtime)
    host = _host_provenance()
    if host["architecture_family"] not in {"arm64", "x86_64"}:
        raise ProfileArtifactError("profile evidence requires a native ARM64 or x86_64 host")
    source = _source_identity(
        revision=source_revision,
        source_digest=source_digest,
        declared_clean=source_clean,
    )
    if source["dirty"]:
        raise ProfileArtifactError("profile capture must start from an exact clean source")
    observations: list[dict[str, Any]] = []
    for profile in profiles:
        for case in selected:
            payload = materialize_case(case)
            for budget in thread_budgets:
                thread_settings = configure_threads(budget, configure_torch=False)
                context = AdapterContext(profile, PROFILE_BUILD_LABEL, budget)
                adapter = load_adapter(DEFAULT_ADAPTER, context)
                if hasattr(adapter, "_event_capacity"):
                    adapter._event_capacity = event_capacity
                for repetition in range(repetitions):
                    observations.append(
                        _capture_operation(
                            adapter=adapter,
                            payload=payload,
                            profile=profile,
                            case=case,
                            budget=budget,
                            repetition=repetition,
                            thread_settings=thread_settings,
                        )
                    )
    sampled_profiles = []
    for profile in profiles:
        for case in selected:
            for budget in thread_budgets:
                try:
                    sampled_profiles.append(
                        capture_sampled_profile(
                            profile=profile,
                            case_id=case["case_id"],
                            thread_budget=budget,
                            workload_path=workload_path,
                            artifact_directory=artifact_directory,
                            artifact_publish_directory=artifact_publish_directory,
                            py_spy=py_spy,
                            rate_hz=sampler_rate_hz,
                            duration_seconds=sampler_duration_seconds,
                        )
                    )
                except ProfileArtifactError as error:
                    raise ProfileArtifactError(
                        "sampled profile failed at "
                        f"profile={profile}, case={case['case_id']}, threads={budget}: {error}"
                    ) from error
    protocol = {
        "workload": workload_provenance(workload_path),
        "observed_adapter": DEFAULT_ADAPTER,
        "paired_adapter": BASELINE_ADAPTER,
        "profiles": list(profiles),
        "cases": list(cases),
        "thread_budgets": list(thread_budgets),
        "thread_regimes": {"one": 1, "production": 4},
        "repetitions": repetitions,
        "event_capacity": event_capacity,
        "samplers": copy.deepcopy(SAMPLER_PROTOCOLS),
        "build_labels": [PROFILE_BUILD_LABEL],
        "production_thread_budget": 4,
    }
    benchmark_identity = {
        **_artifact_identity(
            benchmark_result_path,
            published_path=publish(benchmark_result_path),
        ),
        "architecture_family": benchmark["architecture_family"],
        "schema_id": benchmark["schema_id"],
        "candidate_identity": benchmark["protocol"]["candidate_identity"],
    }
    bundle: dict[str, Any] = {
        "schema_id": PROFILE_SCHEMA_ID,
        "schema_version": PROFILE_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": protocol,
        "captures": [
            {
                "host": host,
                "source": source,
                "build": {
                    "label": PROFILE_BUILD_LABEL,
                    "mode": "release",
                    "command": build_command,
                    "candidate_identity": build_candidate_identity,
                    "log": build_log_identity,
                    "wheel": wheel_identity,
                },
                "phase_c_report": phase_c_identity,
                "capture_command": shlex.join(sys.argv),
                "profiler": {
                    "name": OBSERVATION_SCHEMA_VERSION,
                    "kind": "bounded_stage_instrumentation",
                    "timing": "inclusive and exclusive monotonic wall nanoseconds",
                },
                "paired_benchmark": benchmark_identity,
                "observations": observations,
                "sampled_profiles": sampled_profiles,
            }
        ],
        "summary": {},
    }
    bundle["summary"] = summarize_bundle(bundle)
    schema_errors = list(Draft202012Validator(_schema()).iter_errors(bundle))
    if schema_errors:
        raise ProfileArtifactError(
            f"captured profile bundle violates schema: {schema_errors[0].message}"
        )
    return bundle


def merge_bundles(bundles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not bundles:
        raise ProfileArtifactError("at least one profile bundle is required")
    for bundle in bundles:
        validate_bundle(bundle, require_arm_x86=False)
    protocol = bundles[0]["protocol"]
    if any(bundle["protocol"] != protocol for bundle in bundles[1:]):
        raise ProfileArtifactError("profile bundles use different protocols")
    merged = {
        "schema_id": PROFILE_SCHEMA_ID,
        "schema_version": PROFILE_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": copy.deepcopy(protocol),
        "captures": [
            copy.deepcopy(capture) for bundle in bundles for capture in bundle["captures"]
        ],
        "summary": {},
    }
    merged["summary"] = summarize_bundle(merged)
    validate_bundle(merged, require_arm_x86=len(merged["captures"]) > 1)
    return merged


def render_report(bundle: Mapping[str, Any]) -> str:
    validate_bundle(bundle, require_arm_x86=len(bundle["captures"]) > 1)
    protocol = bundle["protocol"]
    lines = [
        "# qwen-mm whole-operation profile v1",
        "",
        f"- Source revision: `{bundle['captures'][0]['source']['revision']}`",
        f"- Profiles: `{', '.join(protocol['profiles'])}`",
        f"- Cases: `{', '.join(protocol['cases'])}`",
        f"- Equal thread budgets: `{', '.join(map(str, protocol['thread_budgets']))}`",
        "- Timing ranking: exclusive stage duration; inclusive spans are retained but not summed.",
        "- Build: locked Maturin `profiled-release` evidence wheel.",
        "",
    ]
    for architecture, summary in bundle["summary"]["architectures"].items():
        capture = next(
            item
            for item in bundle["captures"]
            if item["host"]["architecture_family"] == architecture
        )
        _, benchmark_bytes = _read_authenticated_artifact(capture["paired_benchmark"])
        benchmark = json.loads(benchmark_bytes)
        phase_c = benchmark["release_eligibility"]["phase_c"]
        runtime = capture["build"]["candidate_identity"]["runtime_identity"]
        sampler = capture["sampled_profiles"][0]["sampler"]
        lines.extend(
            (
                f"## {architecture}",
                "",
                "### Provenance",
                "",
                f"- Host: `{capture['host']['system']} {capture['host']['release']}`; "
                f"CPU `{capture['host']['cpu_description']}`.",
                f"- Build command: `{capture['build']['command']}`",
                f"- Build evidence: wheel `{capture['build']['wheel']['sha256']}`; native "
                f"`{runtime['native_artifact_sha256']}`; log `{capture['build']['log']['sha256']}`.",
                f"- Capture command: `{capture['capture_command']}`",
                f"- Sampler: `{sampler['version']}`; mode "
                f"`{'native' if sampler['native'] else 'Python-only'}`; binary "
                f"`{sampler['binary_sha256']}`; "
                f"representative full command `{sampler['command']}`. Per-coordinate commands are "
                "retained in the bundle.",
                f"- Phase C: `pass`; report `{capture['phase_c_report']['sha256']}`; gate "
                f"`{phase_c['gate_fingerprint']}`; runtime native "
                f"`{phase_c['evidence']['candidate_runtime_identity']['native_artifact_sha256']}`.",
                "",
                "### Observed exclusive-time bottlenecks",
                "",
            )
        )
        lines.extend(("| Rank | Stage | Exclusive time | Share |", "| ---: | --- | ---: | ---: |"))
        for rank, row in enumerate(summary["timing"][:12], 1):
            lines.append(
                f"| {rank} | `{row['stage']}` | {row['exclusive_duration_ns'] / 1e6:.3f} ms | "
                f"{row['share_of_observed_stage_time']:.1%} |"
            )
        lines.extend(("", "### OS-sampled stack hotspots by coordinate", ""))
        lines.extend(
            (
                "| Profile/case/threads | Top inclusive frame | Top self frame | Samples |",
                "| --- | --- | --- | ---: |",
            )
        )
        for coordinate in summary["coordinates"]:
            label = (
                f"{coordinate['profile_alias']}/{coordinate['case_id']}/"
                f"{coordinate['thread_budget']}"
            )
            inclusive = coordinate["sampled_inclusive"][0]
            self_frame = coordinate["sampled_self"][0]
            lines.append(
                f"| `{label}` | `{inclusive['frame']}` ({inclusive['share']:.1%}) | "
                f"`{self_frame['frame']}` ({self_frame['share']:.1%}) | "
                f"{sum(row['samples'] for row in coordinate['sampled_self'])} |"
            )
        lines.extend(("", "### Allocation bottlenecks", ""))
        lines.extend(
            ("| Rank | Buffer | Class | Allocated | Count |", "| ---: | --- | --- | ---: | ---: |")
        )
        for rank, row in enumerate(summary["allocations"][:12], 1):
            lines.append(
                f"| {rank} | `{row['buffer']}` | `{row['class']}` | "
                f"{row['allocated_bytes']:,} B | {row['allocation_count']} |"
            )
        lines.extend(("", "### Copy bottlenecks", ""))
        lines.extend(("| Rank | Copy | Copied | Copies |", "| ---: | --- | ---: | ---: |"))
        for rank, row in enumerate(summary["copies"][:12], 1):
            lines.append(
                f"| {rank} | `{row['copy']}` | {row['copied_bytes']:,} B | {row['copy_count']} |"
            )
        lines.extend(("", "### Buffer-lifetime bottlenecks", ""))
        lines.extend(("| Rank | Buffer | Class | Total lifetime |", "| ---: | --- | --- | ---: |"))
        for rank, row in enumerate(summary["buffer_lifetimes"][:12], 1):
            lines.append(
                f"| {rank} | `{row['buffer']}` | `{row['class']}` | "
                f"{row['total_lifetime_ns'] / 1e6:.3f} ms |"
            )
        lines.append("")
    return "\n".join(lines)


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_ints(value: str) -> list[int]:
    try:
        return [int(item) for item in _csv_strings(value)]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "thread budgets must be comma-separated integers"
        ) from error


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture and validate qwen-mm profile v1 bundles")
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("_sample_worker", help=argparse.SUPPRESS)
    worker.add_argument("--config", type=Path, required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--workload", type=Path, default=Path("benchmarks/workloads-v2.json"))
    capture.add_argument("--benchmark-result", type=Path, required=True)
    capture.add_argument("--profiles", default=",".join(DEFAULT_PROFILES))
    capture.add_argument("--cases", default=",".join(DEFAULT_CASES))
    capture.add_argument("--thread-budgets", default="1,4")
    capture.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    capture.add_argument("--event-capacity", type=int, default=4096)
    capture.add_argument("--build-command", required=True)
    capture.add_argument("--build-log", type=Path, required=True)
    capture.add_argument("--wheel", type=Path, required=True)
    capture.add_argument("--phase-c-report", type=Path, required=True)
    capture.add_argument("--benchmark-phase-c-source-report", type=Path)
    capture.add_argument("--artifact-directory", type=Path, required=True)
    capture.add_argument("--artifact-publish-directory", type=Path)
    capture.add_argument("--py-spy", default="py-spy")
    capture.add_argument("--sampler-rate-hz", type=int, default=99)
    capture.add_argument("--sampler-duration-seconds", type=int, default=2)
    capture.add_argument("--source-revision")
    capture.add_argument("--source-digest")
    capture.add_argument("--source-clean", action="store_true")
    capture.add_argument("--output", type=Path, required=True)

    merge = commands.add_parser("merge")
    merge.add_argument("bundles", type=Path, nargs="+")
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--report", type=Path)

    validate = commands.add_parser("validate")
    validate.add_argument("bundle", type=Path)
    validate.add_argument("--allow-single-architecture", action="store_true")

    report = commands.add_parser("report")
    report.add_argument("bundle", type=Path)
    report.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "_sample_worker":
            _sample_worker(args)
        elif args.command == "capture":
            bundle = capture_bundle(
                workload_path=args.workload,
                benchmark_result_path=args.benchmark_result,
                profiles=_csv_strings(args.profiles),
                cases=_csv_strings(args.cases),
                thread_budgets=_csv_ints(args.thread_budgets),
                repetitions=args.repetitions,
                build_command=args.build_command,
                build_log_path=args.build_log,
                wheel_path=args.wheel,
                phase_c_report_path=args.phase_c_report,
                benchmark_phase_c_source_report_path=args.benchmark_phase_c_source_report,
                artifact_directory=args.artifact_directory,
                artifact_publish_directory=args.artifact_publish_directory,
                py_spy=args.py_spy,
                sampler_rate_hz=args.sampler_rate_hz,
                sampler_duration_seconds=args.sampler_duration_seconds,
                source_revision=args.source_revision,
                source_digest=args.source_digest,
                source_clean=args.source_clean,
                event_capacity=args.event_capacity,
            )
            _write_json(args.output, bundle)
        elif args.command == "merge":
            bundle = merge_bundles([_load_json(path) for path in args.bundles])
            validate_bundle(bundle, require_arm_x86=True)
            _write_json(args.output, bundle)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(render_report(bundle), encoding="utf-8")
        elif args.command == "validate":
            validate_bundle(
                _load_json(args.bundle),
                require_arm_x86=not args.allow_single_architecture,
            )
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(render_report(_load_json(args.bundle)), encoding="utf-8")
    except (BenchmarkProtocolError, ProfileArtifactError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
