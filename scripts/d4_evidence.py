"""Build and validate D4 certification evidence from two authenticated raw ZIPs.

The ZIPs remain the source of truth.  Validation reopens both archives, repeats
their portable benchmark and Phase C authentication, rebuilds every derived
capture and gate, and compares the result byte-for-byte as canonical JSON.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parent
REFERENCE_SOURCE = REPOSITORY_ROOT / "reference" / "src"
for import_path in (SCRIPT_DIRECTORY, REFERENCE_SOURCE):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from d4_capture_support import (  # noqa: E402
    BUILD_LABELS,
    CAPTURE_INPUT_PATHS,
    COMPACT_BUILD_LABELS,
    COMPACT_CASES_BY_THREAD_BUDGET,
    COMPACT_PROCESS_REPETITIONS,
    COMPACT_THREAD_BUDGETS,
    MODAL_VM_SANDBOX_ATTESTATION_MODE,
    PROFILES,
    THREAD_BUDGETS,
    _provenance_affinity_masks,
    assert_assets_identity,
    local_linux_toolchain_pins,
    read_capture_archive,
    toolchain_pins,
    wheel_benchmark_artifact_identity,
)
from modal_benchmark_support import committed_source_tree_digest  # noqa: E402
from modal_d3_support import sha256_bytes  # noqa: E402
from profile_capture_support import wheel_contents_identity  # noqa: E402
from qwen_mm_reference.benchmark_v2 import (  # noqa: E402
    _validate_phase_c_report,
    validate_result_authenticated_portable,
)
from qwen_mm_reference.performance_certification_v1 import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    CASES,
    RANDOM_SEED,
    WORK_UNITS,
    build_certification,
    render_report,
    validate_certification,
)

ASSETS_ROOT = REPOSITORY_ROOT / "reference" / ".cache" / "huggingface"
GENERATED_PREFIXES = (
    ".kd/",
    "benchmarks/performance-certification-v1/",
    "benchmarks/performance-evidence-v1/",
)
TOLERANCE_POLICY = (
    "exact integers; exact timed stability; pixel per-occurrence lossless=1e-6 "
    "otherwise workload float_atol"
)
TIMING_FLOOR_POLICY = (
    "minimum_samples one-operation latency samples; supplemental individually clocked "
    "exact-stability operations aggregate only to minimum_seconds and are excluded from "
    "latency distributions"
)
PROTOCOL = {
    "mode": "dedicated",
    "process_repetitions": COMPACT_PROCESS_REPETITIONS,
    "random_seed": RANDOM_SEED,
    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    "order": "randomized AB/BA per process repetition",
    "warmups": 3,
    "minimum_samples": 30,
    "minimum_seconds": 5.0,
    "timing_instrumentation": "none",
    "timing_floor_policy": TIMING_FLOOR_POLICY,
    "memory_pass": "separate from timing",
    "profiles": list(PROFILES),
    "cases": [
        "image24",
        "image1",
        "rgb24",
        "text_long",
        "ragged24",
        "images_16",
    ],
    "thread_budgets": list(COMPACT_THREAD_BUDGETS),
    "production_thread_budget": 8,
    "cache_policy": "cache workloads not selected in compact matrix",
}
LEGACY_PROTOCOL = {
    "mode": "dedicated",
    "process_repetitions": 5,
    "random_seed": RANDOM_SEED,
    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    "order": "randomized AB/BA per process repetition",
    "warmups": 3,
    "minimum_samples": 30,
    "minimum_seconds": 5.0,
    "timing_instrumentation": "none",
    "timing_floor_policy": TIMING_FLOOR_POLICY,
    "memory_pass": "separate from timing",
    "profiles": list(PROFILES),
    "cases": list(CASES),
    "thread_budgets": list(THREAD_BUDGETS),
    "production_thread_budget": 8,
    "cache_policy": "repeat24_cached unsupported/non-gating; uncached timing forbidden",
}


def _suite(provenance: Mapping[str, Any]) -> str:
    return "compact" if provenance.get("suite") == "compact" else "exhaustive"


def _budgets(provenance: Mapping[str, Any]) -> tuple[int, ...]:
    return COMPACT_THREAD_BUDGETS if _suite(provenance) == "compact" else THREAD_BUDGETS


def _build_labels(provenance: Mapping[str, Any]) -> tuple[str, ...]:
    return COMPACT_BUILD_LABELS if _suite(provenance) == "compact" else BUILD_LABELS


class D4EvidenceError(RuntimeError):
    """Raised when raw inputs cannot truthfully produce certification evidence."""


def _fail(message: str) -> None:
    raise D4EvidenceError(message)


def _object(value: Any, name: str, fields: set[str] | frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        actual = set(value) if isinstance(value, Mapping) else set()
        _fail(
            f"{name} has a closed shape "
            f"(missing={sorted(set(fields) - actual)}, extra={sorted(actual - set(fields))})"
        )
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{name} must be a list")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{name} must be an integer >= {minimum}")
    return value


def _sha(value: Any, name: str) -> str:
    text = _string(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        _fail(f"{name} must be a lowercase SHA-256 digest")
    return text


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    except (TypeError, ValueError) as error:
        raise D4EvidenceError("evidence contains non-canonical JSON data") from error


def _digest(value: Any) -> str:
    return sha256_bytes(_canonical(value))


def _json_member(files: Mapping[str, bytes], name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(files[name])
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise D4EvidenceError(f"archive member is missing or invalid JSON: {name}") from error
    if not isinstance(value, Mapping):
        _fail(f"archive member must contain an object: {name}")
    return value


def _materialize_phase_c_evidence(
    files: Mapping[str, bytes],
    *,
    build_label: str,
    position: str,
    destination: Path,
) -> Path:
    prefix = f"phase-c/{build_label}/{position}/"
    for member_name, member_data in files.items():
        if not member_name.startswith(prefix):
            continue
        relative = Path(member_name.removeprefix(prefix))
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(member_data)
    report = destination / "report.json"
    if not report.is_file():
        _fail(f"{build_label}/{position} Phase C report was not materialized")
    return report


def _command(value: Any, name: str) -> str:
    values = _list(value, name)
    if not values or not all(isinstance(item, str) and item for item in values):
        _fail(f"{name} must contain command arguments")
    return shlex.join(values)


def _git(*arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise D4EvidenceError(f"Git currentness check failed: {' '.join(arguments)}") from error


def _allowed_generated(path: str) -> bool:
    normalized = path.rstrip("/") + ("/" if path.endswith("/") else "")
    return path == ".kd" or any(
        path.startswith(prefix) or normalized.startswith(prefix) for prefix in GENERATED_PREFIXES
    )


def _assert_current_source(revision: str, source: Mapping[str, Any]) -> None:
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        _fail("captured source revision is not an exact Git commit")
    if _git("rev-parse", "--verify", f"{revision}^{{commit}}") != revision:
        _fail("captured source revision does not resolve exactly")
    try:
        _git("merge-base", "--is-ancestor", revision, "HEAD")
    except D4EvidenceError as error:
        raise D4EvidenceError("current HEAD does not descend from the captured revision") from error
    changed = [
        path
        for path in _git("diff", "--name-only", f"{revision}..HEAD").splitlines()
        if path and not _allowed_generated(path)
    ]
    if changed:
        _fail(f"code or protocol changed after capture: {changed}")
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    dirty: list[str] = []
    for line in status.splitlines():
        path = line[3:]
        if " -> " in path:
            old, new = path.split(" -> ", 1)
            paths = (old, new)
        else:
            paths = (path,)
        dirty.extend(item for item in paths if not _allowed_generated(item))
    if dirty:
        _fail(f"working tree has non-evidence changes after capture: {dirty}")
    expected = _object(source, "provenance.source", {"sha256", "file_count", "bytes"})
    digest, file_count, byte_count = committed_source_tree_digest(REPOSITORY_ROOT, revision)
    if expected != {"sha256": digest, "file_count": file_count, "bytes": byte_count}:
        _fail("captured source payload differs from its immutable Git revision")


def _git_input_identity(revision: str, path: str) -> dict[str, Any]:
    try:
        value = subprocess.run(
            ["git", "show", f"{revision}:{path}"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise D4EvidenceError(f"captured contract is unavailable from Git: {path}") from error
    return {"bytes": len(value), "sha256": sha256_bytes(value)}


def _identity(provenances: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    first = provenances[0]
    revision = _string(first["source_revision"], "source_revision")
    source = first["source"]
    assets = first["assets"]
    inputs = first["capture_inputs"]
    for provenance in provenances[1:]:
        for field in ("source_revision", "source", "assets", "capture_inputs"):
            if provenance[field] != first[field]:
                _fail(f"cross-host capture {field} drifted")
    _assert_current_source(revision, source)
    expected_inputs = _object(inputs, "capture_inputs", set(CAPTURE_INPUT_PATHS))
    for path in CAPTURE_INPUT_PATHS:
        record = _object(expected_inputs[path], f"capture_inputs.{path}", {"bytes", "sha256"})
        if record != _git_input_identity(revision, path):
            _fail(f"captured contract differs from the captured Git revision: {path}")
    asset_record = _object(
        assets,
        "assets",
        {"schema_id", "schema_version", "tree_sha256", "entry_count", "logical_bytes", "entries"},
    )
    assert_assets_identity(ASSETS_ROOT, asset_record)
    return {
        "source_revision": revision,
        "source_sha256": _sha(source["sha256"], "source.sha256"),
        "benchmark_schema_sha256": _sha(
            expected_inputs["benchmarks/result-schema-v3.json"]["sha256"], "benchmark schema"
        ),
        "workload_sha256": _sha(
            expected_inputs["benchmarks/workloads-v2.json"]["sha256"], "workload"
        ),
        "profile_schema_sha256": _sha(
            expected_inputs["benchmarks/profile-schema-v2.json"]["sha256"], "profile schema"
        ),
        "model_registry_sha256": _sha(
            expected_inputs["reference/models.json"]["sha256"], "model registry"
        ),
        "assets_sha256": _sha(asset_record["tree_sha256"], "assets tree"),
        "protocol_sha256": _sha(
            expected_inputs["benchmarks/performance-certification-schema-v1.json"]["sha256"],
            "certification protocol",
        ),
    }


def _validate_provenance(value: Mapping[str, Any], architecture: str) -> Mapping[str, Any]:
    common = {
        "schema_id",
        "schema_version",
        "claim",
        "architecture_family",
        "started_at",
        "completed_at",
        "source_revision",
        "source",
        "assets",
        "capture_inputs",
        "host",
        "build_wheel_sha256",
        "build_native_sha256",
        "toolchain_pins",
        "sample_pruning",
        "noise_cv_max",
        "environment",
    }
    compact = value.get("suite") == "compact"
    if compact:
        common.add("suite")
    if architecture == "arm64":
        fields = common | {"host_label"}
    else:
        provider = value.get("provider")
        if provider == "modal":
            fields = common | {"provider", "modal"}
        elif provider == "local_linux":
            fields = common | {"provider", "local_linux"}
        else:
            _fail("x86_64.provenance.provider must be modal or local_linux")
    provenance = _object(value, f"{architecture}.provenance", fields)
    if (
        provenance["schema_id"] != "qwen-mm-d4-raw-capture-provenance-v1"
        or provenance["schema_version"] != 1
        or provenance["claim"]
        != "raw controlled-host input for the separate D4 certification evaluator"
        or provenance["architecture_family"] != architecture
        or (compact and provenance["suite"] != "compact")
        or provenance["sample_pruning"] != "forbidden"
        or provenance["noise_cv_max"] != 0.05
    ):
        _fail(f"{architecture} raw provenance changed the D4 capture contract")
    if architecture == "x86_64":
        if provenance["provider"] == "modal":
            vm_sandbox = (
                provenance.get("host", {}).get("resource_attestation", {}).get("mode")
                == MODAL_VM_SANDBOX_ATTESTATION_MODE
            )
            modal = _object(
                provenance["modal"],
                "x86_64.modal",
                {"client_version", "environment"}
                | ({"sandbox_control", "pricing_snapshot", "lifecycle"} if vm_sandbox else set()),
            )
            _string(modal["client_version"], "x86_64.modal.client_version")
            if not isinstance(modal["environment"], Mapping):
                _fail("x86_64.modal.environment must be an object")
            if vm_sandbox and not all(
                isinstance(modal.get(field), Mapping)
                for field in ("sandbox_control", "pricing_snapshot", "lifecycle")
            ):
                _fail("x86_64.modal VM Sandbox provenance must contain object records")
        else:
            local = _object(
                provenance["local_linux"],
                "x86_64.local_linux",
                {"host_label", "allocation_id"},
            )
            _string(local["host_label"], "x86_64.local_linux.host_label")
            _string(local["allocation_id"], "x86_64.local_linux.allocation_id")
        expected_pins = (
            toolchain_pins(suite="compact" if compact else "exhaustive")
            if provenance["provider"] == "modal"
            else local_linux_toolchain_pins()
        )
        if provenance["toolchain_pins"] != expected_pins:
            _fail(f"x86_64 {provenance['provider']} toolchain/resource pins changed")
    elif provenance["toolchain_pins"] != toolchain_pins(
        suite="compact" if compact else "exhaustive"
    ):
        _fail("arm64 toolchain/resource pins changed")
    for field in ("build_wheel_sha256", "build_native_sha256"):
        mapping = _object(
            provenance[field], f"{architecture}.{field}", set(_build_labels(provenance))
        )
        for label, raw_digest in mapping.items():
            _sha(raw_digest, f"{architecture}.{field}.{label}")
    _object(
        provenance["environment"],
        f"{architecture}.environment",
        {
            "PYTHONHASHSEED",
            "LC_ALL",
            "TZ",
            "UV_DEFAULT_INDEX",
            "removed_package_index_variables",
        },
    )
    parsed_timestamps: dict[str, datetime] = {}
    for timestamp in ("started_at", "completed_at"):
        try:
            parsed = datetime.fromisoformat(str(provenance[timestamp]).replace("Z", "+00:00"))
        except ValueError as error:
            raise D4EvidenceError(f"{architecture}.{timestamp} is invalid") from error
        if parsed.tzinfo is None:
            _fail(f"{architecture}.{timestamp} lacks a timezone")
        parsed_timestamps[timestamp] = parsed
    if parsed_timestamps["started_at"] >= parsed_timestamps["completed_at"]:
        _fail(f"{architecture} capture timestamps are stale or inverted")
    return provenance


def _archive(path: Path, expected_architecture: str) -> dict[str, Any]:
    value = path.read_bytes()
    files = read_capture_archive(value, phase_c_assets_root=ASSETS_ROOT)
    provenance = _validate_provenance(_json_member(files, "provenance.json"), expected_architecture)
    try:
        with zipfile.ZipFile(path) as archive:
            manifest = archive.read("artifact-manifest.json")
    except (OSError, KeyError, zipfile.BadZipFile) as error:
        raise D4EvidenceError("validated raw ZIP could not be reopened") from error
    return {
        "path": path,
        "value": value,
        "files": files,
        "provenance": provenance,
        "sha256": sha256_bytes(value),
        "bytes": len(value),
        "manifest_sha256": sha256_bytes(manifest),
        "provenance_sha256": sha256_bytes(files["provenance.json"]),
    }


def _host(provenance: Mapping[str, Any], architecture: str) -> dict[str, Any]:
    suite = _suite(provenance)
    budgets = _budgets(provenance)
    if architecture == "arm64":
        raw = _object(
            provenance["host"],
            "arm64.host",
            {
                "system",
                "machine",
                "platform",
                "uname",
                "hostname",
                "cpu_model",
                "physical_cpu_count",
                "logical_cpu_count",
                "memory_bytes",
                "affinity",
            },
        )
        affinity = _object(raw["affinity"], "arm64.host.affinity", {"mode", "reason", "masks"})
        raw_masks = affinity["masks"]
        expected_compact_masks = {f"t{budget}": None for budget in budgets}
        expected_full_masks = {f"t{budget}": None for budget in THREAD_BUDGETS}
        if affinity["mode"] != "unavailable" or raw_masks not in (
            expected_compact_masks,
            expected_full_masks,
        ):
            _fail("ARM capture falsely claims process affinity")
        topology = {
            "cpu_model": raw["cpu_model"],
            "physical_cpu_count": raw["physical_cpu_count"],
            "logical_cpu_count": raw["logical_cpu_count"],
        }
        provider = "local"
        instance_type = _string(provenance["host_label"], "arm64.host_label")
        allocation_id = _string(raw["hostname"], "arm64.hostname")
        masks = {f"t{budget}": None for budget in budgets}
        cgroup = {"mode": "not-applicable-native-macos"}
        allocated_cpu_count = _integer(raw["physical_cpu_count"], "arm64 physical CPUs", minimum=8)
        allocated_memory = _integer(raw["memory_bytes"], "arm64 memory", minimum=1)
        affinity_source = "macos-unavailable"
        unavailable_reason = _string(affinity["reason"], "arm64 affinity reason")
        if raw["system"] != "Darwin" or raw["machine"] not in {"arm64", "aarch64"}:
            _fail("ARM host provenance is not native macOS ARM")
    else:
        raw = _object(
            provenance["host"],
            "x86_64.host",
            {
                "system",
                "machine",
                "platform",
                "uname",
                "hostname",
                "cpu_description",
                "lscpu",
                "lscpu_parse",
                "logical_cpu_count",
                "proc_meminfo_total_bytes",
                "resource_attestation",
            },
        )
        provider = provenance["provider"]
        if provider == "modal":
            modal_common = {
                "mode",
                "requested_resources_bound_by",
                "requested_physical_cores",
                "requested_memory_mib",
                "nonpreemptible",
                "single_use_container",
                "visible_affinity",
                "physical_core_masks",
                "cgroup_limits",
            }
            attestation_fields = (
                modal_common
                | {
                    "nonpreemptible_basis",
                    "sandbox_id",
                    "resolved_modal_image_id",
                    "vm_runtime",
                    "virtualization",
                    "api_resource_request",
                    "affinity_enforcement",
                }
                if raw["resource_attestation"].get("mode") == MODAL_VM_SANDBOX_ATTESTATION_MODE
                else modal_common
            )
        else:
            attestation_fields = {
                "mode",
                "allocated_physical_cores",
                "allocated_memory_bytes",
                "power_policy",
                "affinity_enforcement",
                "exclusive_physical_cores",
                "dedicated_capture",
                "stable",
                "visible_affinity",
                "physical_core_masks",
                "cgroup_limits",
            }
        attestation = _object(
            raw["resource_attestation"],
            "x86_64.host.resource_attestation",
            attestation_fields,
        )
        recomputed_masks = _provenance_affinity_masks(
            provenance,
            architecture,
            thread_budgets=budgets,
            suite=suite,
        )
        masks = {f"t{budget}": recomputed_masks[budget] for budget in budgets}
        topology = {"lscpu": raw["lscpu"], "lscpu_parse": raw["lscpu_parse"]}
        cgroup = {"mode": attestation["mode"], "limits": attestation["cgroup_limits"]}
        if provider == "modal":
            if (
                attestation["nonpreemptible"] is not True
                or attestation["single_use_container"] is not True
            ):
                _fail("x86 capture lacks nonpreemptible single-use placement")
            requested_cpu_count = int(attestation["requested_physical_cores"])
            instance_type = (
                f"cpu-{requested_cpu_count}-memory-{attestation['requested_memory_mib']}MiB"
            )
            modal = provenance["modal"]
            modal_environment = modal["environment"]
            allocation_id = (
                _string(attestation["sandbox_id"], "x86_64.modal.sandbox_id")
                if attestation["mode"] == MODAL_VM_SANDBOX_ATTESTATION_MODE
                else next(
                    (
                        str(modal_environment[name])
                        for name in ("MODAL_TASK_ID", "MODAL_CONTAINER_ID", "MODAL_APP_ID")
                        if isinstance(modal_environment.get(name), str) and modal_environment[name]
                    ),
                    _string(raw["hostname"], "x86_64.hostname"),
                )
            )
            allocated_cpu_count = requested_cpu_count
            allocated_memory = (
                _integer(attestation["requested_memory_mib"], "x86 allocated memory", minimum=1)
                * 1024
                * 1024
            )
            affinity_source = "taskset+sched_getaffinity"
        else:
            local = provenance["local_linux"]
            instance_type = _string(local["host_label"], "x86_64.local_linux.host_label")
            allocation_id = _string(local["allocation_id"], "x86_64.local_linux.allocation_id")
            allocated_cpu_count = _integer(
                attestation["allocated_physical_cores"],
                "x86 allocated physical CPUs",
                minimum=8,
            )
            allocated_memory = _integer(
                attestation["allocated_memory_bytes"],
                "x86 allocated memory",
                minimum=1,
            )
            affinity_source = "cgroup_v2_cpuset+taskset+sched_getaffinity"
        unavailable_reason = None
        if raw["system"] != "Linux" or raw["machine"] not in {"x86_64", "amd64"}:
            _fail("x86 host provenance is not native Linux x86_64")
    result = {
        "host_fingerprint": _digest(raw),
        "baseline": "current-m4" if architecture == "arm64" else "controlled-linux-x86",
        "system": raw["system"],
        "machine": architecture,
        "cpu_model": raw["cpu_model"] if architecture == "arm64" else raw["cpu_description"],
        "physical_cpu_count": (
            raw["physical_cpu_count"] if architecture == "arm64" else allocated_cpu_count
        ),
        "logical_cpu_count": raw["logical_cpu_count"],
        "memory_bytes": raw[
            "memory_bytes" if architecture == "arm64" else "proc_meminfo_total_bytes"
        ],
        "controlled": True,
        "provider": provider,
        "instance_type": instance_type,
        "allocation_id": allocation_id,
        "affinity_available": architecture == "x86_64",
        "affinity_source": affinity_source,
        "affinity_unavailable_reason": unavailable_reason,
        "allocated_cpu_count": allocated_cpu_count,
        "allocated_memory_bytes": allocated_memory,
        "cgroup_sha256": _digest(cgroup),
        "physical_core_topology_sha256": _digest(topology),
        "physical_core_masks": dict(masks),
    }
    return result


def _build_record(
    raw: Mapping[str, Any], label: str, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    fields = {
        "build_label",
        "identity",
        "plan",
        "commands",
        "build_environment",
        "build_host",
        "toolchain",
        "runtime",
        "runtime_reconciliation",
        "reference_sync",
        "packages",
    }
    compact = provenance.get("suite") == "compact"
    if compact:
        fields.add("suite")
    build = _object(raw, f"builds.{label}", fields)
    if build["build_label"] != label or (compact and build["suite"] != "compact"):
        _fail(f"build JSON metadata differs from its archive path: {label}")
    commands = _object(
        build["commands"],
        f"builds.{label}.commands",
        {"create_venv", "sync_reference", "build_wheel", "install_retained_wheel"},
    )
    identity = _object(build["identity"], f"builds.{label}.identity", {"python", "wheel"})
    wheel = _object(
        identity["wheel"], f"builds.{label}.identity.wheel", {"name", "bytes", "sha256", "contents"}
    )
    runtime = _object(
        build["runtime"],
        f"builds.{label}.runtime",
        {"package_version", "package_origin", "native_origin", "native_bytes", "native_sha256"},
    )
    wheel_sha = _sha(wheel["sha256"], f"{label}.wheel_sha256")
    native_sha = _sha(runtime["native_sha256"], f"{label}.native_sha256")
    if (
        provenance["build_wheel_sha256"].get(label) != wheel_sha
        or provenance["build_native_sha256"].get(label) != native_sha
    ):
        _fail(f"{label} build artifacts differ from provenance")
    build_command = _command(commands["build_wheel"], f"{label}.build_wheel")
    expected_native_flag = "RUSTFLAGS=-C target-cpu=native"
    if (expected_native_flag in commands["build_wheel"]) is (label == "shipping"):
        _fail(f"{label} build has the wrong native optimization override")
    return {
        "label": label,
        "optimization": "portable-release" if label == "shipping" else "target-cpu=native",
        "wheel_sha256": wheel_sha,
        "native_sha256": native_sha,
        "build_command": build_command,
        "rustflags": [] if label == "shipping" else ["-C", "target-cpu=native"],
    }


def _toolchain(raw: Mapping[str, Any]) -> dict[str, Any]:
    toolchain = _object(
        raw["toolchain"], "build.toolchain", {"python", "maturin", "rustc", "cargo"}
    )
    output: dict[str, str] = {}
    for name in ("python", "maturin", "rustc", "cargo"):
        record = _object(toolchain[name], f"build.toolchain.{name}", {"command", "output"})
        _command(record["command"], f"build.toolchain.{name}.command")
        output[name] = _string(record["output"], f"build.toolchain.{name}.output")
    build_host = _object(raw["build_host"], "build.build_host", {"os_release", "uname"})
    os_release = _object(
        build_host["os_release"], "build.build_host.os_release", {"command", "output"}
    )
    _command(os_release["command"], "build.build_host.os_release.command")
    return {
        **output,
        "os_release": _string(os_release["output"], "build.build_host.os_release.output"),
        "environment_sha256": _digest(raw["build_environment"]),
    }


def _retained_wheel(
    files: Mapping[str, bytes],
    raw_build: Mapping[str, Any],
    build: Mapping[str, Any],
    *,
    architecture: str,
    build_label: str,
    temporary: Path,
) -> dict[str, Any]:
    wheel_record = raw_build["identity"]["wheel"]
    wheel_name = _string(wheel_record["name"], f"{build_label}.wheel.name")
    wheel_member = f"builds/{build_label}/{wheel_name}"
    retained_wheels = sorted(
        name
        for name in files
        if name.startswith(f"builds/{build_label}/") and name.endswith(".whl")
    )
    if retained_wheels != [wheel_member]:
        _fail(
            f"{architecture}/{build_label} must retain exactly its one declared wheel: "
            f"{retained_wheels}"
        )
    wheel_bytes = files[wheel_member]
    if (
        len(wheel_bytes) != wheel_record["bytes"]
        or sha256_bytes(wheel_bytes) != wheel_record["sha256"]
    ):
        _fail(f"{architecture}/{build_label} retained wheel differs from build identity")
    wheel_path = temporary / wheel_name
    wheel_path.write_bytes(wheel_bytes)
    wheel_contents = wheel_contents_identity(wheel_path)
    benchmark_artifact = wheel_benchmark_artifact_identity(wheel_path)
    if wheel_contents != wheel_record["contents"]:
        _fail(f"{architecture}/{build_label} retained wheel contents differ from identity")
    reconciliation = _object(
        raw_build["runtime_reconciliation"],
        f"{architecture}.{build_label}.runtime_reconciliation",
        {"verified", "wheel_contents", "installed_runtime"},
    )
    if (
        reconciliation["verified"] is not True
        or reconciliation["wheel_contents"] != wheel_contents
        or reconciliation["installed_runtime"] != raw_build["runtime"]
        or wheel_contents["native_artifact_sha256"] != build["native_sha256"]
        or wheel_contents["native_bytes"] != raw_build["runtime"]["native_bytes"]
    ):
        _fail(f"{architecture}/{build_label} wheel/runtime reconciliation is stale")
    return {**wheel_contents, "benchmark_artifact": benchmark_artifact}


def _phase_c_record(
    report: Mapping[str, Any],
    report_bytes: bytes,
    identity: Mapping[str, str],
    build: Mapping[str, Any],
) -> dict[str, Any]:
    current = report.get("current_candidate")
    if (
        report.get("schema_id") != "qwen-mm-phase-c-conformance-overlay-v2"
        or report.get("schema_version") != 2
        or not isinstance(current, Mapping)
    ):
        _fail("Phase C report is not current resize-v2 overlay evidence")
    capture = current.get("capture")
    if not isinstance(capture, Mapping):
        _fail("Phase C resize-v2 overlay lacks installed-wheel capture evidence")
    runtime = capture.get("runtime_identity")
    wheel = capture.get("wheel")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("native_artifact_sha256") != build["native_sha256"]
        or not isinstance(wheel, Mapping)
        or wheel.get("sha256") != build["wheel_sha256"]
    ):
        _fail("Phase C report differs from its retained build artifacts")
    profiles = report.get("provenance", {}).get("profiles_inherited_from_v1")
    if profiles != list(PROFILES) or capture.get("case_count") != 45:
        _fail("Phase C resize-v2 overlay has incomplete profile/holdout scope")
    return {
        "status": report.get("status"),
        "report_sha256": sha256_bytes(report_bytes),
        "source_sha256": identity["source_sha256"],
        "wheel_sha256": build["wheel_sha256"],
        "native_sha256": build["native_sha256"],
        "assets_sha256": identity["assets_sha256"],
        "profiles": profiles,
        "failed_cases": 0,
        "skipped_cases": 0,
    }


def _assert_raw_protocol(
    result: Mapping[str, Any], *, architecture: str, build: str, budget: int, suite: str
) -> None:
    protocol = result.get("protocol")
    if suite == "compact":
        expected_cases = list(COMPACT_CASES_BY_THREAD_BUDGET[budget])
        expected_repetitions = COMPACT_PROCESS_REPETITIONS
    else:
        expected_cases = (
            ["image24"]
            if budget in {2, 4}
            else [case for case in CASES if case != "repeat24_cached"]
        )
        expected_repetitions = 5
    if not isinstance(protocol, Mapping) or (
        result.get("mode") != "dedicated"
        or result.get("architecture_family") != architecture
        or protocol.get("reference_adapter") != "official"
        or protocol.get("candidate_adapter") != "qwen_mm.benchmark:create_adapter"
        or protocol.get("profiles") != list(PROFILES)
        or protocol.get("cases") != expected_cases
        or protocol.get("process_repetitions") != expected_repetitions
        or protocol.get("random_seed") != RANDOM_SEED
        or protocol.get("order") != "randomized AB/BA per process repetition"
        or protocol.get("warmups") != 3
        or protocol.get("minimum_samples") != 30
        or protocol.get("minimum_seconds") != 5.0
        or protocol.get("thread_regimes") != [f"t{budget}"]
        or protocol.get("thread_budget_mapping") != {f"t{budget}": budget}
        or protocol.get("build_labels") != [build]
        or protocol.get("timing_protocol") != "instrumentation-free-v1"
        or protocol.get("resource_census_position") != "after_all_timed_samples"
        or protocol.get("threshold_enforcement") != "none; D4 owns release performance gates"
        or protocol.get("self_test_only") is not False
    ):
        _fail(f"{architecture}/{build}/t{budget} changed the frozen raw protocol")


def _memory(worker: Mapping[str, Any], *, candidate: bool) -> dict[str, Any]:
    census = worker["resource_census"]
    rss = census["rss"]
    output_bytes = census["output_bytes"]
    base: dict[str, Any] = {
        "rss_before_bytes": rss["baseline_rss_bytes"],
        "rss_after_bytes": rss["rss_after_bytes"],
        "scoped_peak_rss_bytes": rss["peak_rss_bytes"],
        "retained_output_bytes": output_bytes,
        "external_transient_rss_bytes": rss["external_transient_rss_bytes"],
        "native_counters_available": candidate,
        "observation_duration_ns": None,
        "allocation_count": None,
        "allocated_bytes": None,
        "copy_count": None,
        "copied_bytes": None,
        "transient_live_bytes": None,
        "peak_transient_live_bytes": None,
        "buffer_census_complete": False,
        "dropped_events": None,
        "buffers": [],
        "copies": [],
    }
    native = census["native_observed"]
    if not candidate:
        if native is not None:
            _fail("official worker fabricated candidate native counters")
        return base
    if not isinstance(native, Mapping):
        _fail("candidate worker lacks native observed counters")
    allocations = native["allocations"]
    buffers = native["buffers"]
    base.update(
        observation_duration_ns=native["duration_ns"],
        allocation_count=allocations["allocation_count"],
        allocated_bytes=allocations["allocated_bytes"],
        copy_count=allocations["copy_count"],
        copied_bytes=allocations["copied_bytes"],
        transient_live_bytes=allocations["transient_live_bytes"],
        peak_transient_live_bytes=allocations["peak_transient_live_bytes"],
        buffer_census_complete=True,
        dropped_events=native["dropped_events"],
        buffers=[dict(record) for record in buffers["records"]],
        copies=[dict(record) for record in native["copies"]],
    )
    return base


def _conformance(
    worker: Mapping[str, Any], *, witness_sha: str, input_sha: str, messages_sha: str
) -> dict[str, Any]:
    signature = worker["output_signature"]
    if not isinstance(signature, Mapping):
        _fail("worker output signature is missing")
    output_keys = list(signature)
    dtypes = {
        key: record.get("dtype") if isinstance(record, Mapping) else None
        for key, record in signature.items()
    }
    conformance = worker["conformance"]
    metrics = worker["resource_census"]["adapter_metrics"]
    if metrics["cache_supported"] is not False:
        _fail("timed worker does not explicitly reject adapter caching")
    return {
        "passed": (
            conformance["pre_measurement"] == "pass"
            and conformance["post_measurement"] == "pass"
            and conformance["all_measured_iterations_stable"] is True
        ),
        "witness_sha256": witness_sha,
        "input_sha256": input_sha,
        "messages_sha256": messages_sha,
        "output_keys": output_keys,
        "dtypes": dtypes,
        "values_sha256": _digest(signature),
        "float_atol": conformance["float_atol"],
        "tolerance_policy": TOLERANCE_POLICY,
        "all_timed_outputs_match": conformance["all_measured_iterations_stable"],
        "fallback_work": False,
        "cache_used": False,
    }


def _implementation(
    worker: Mapping[str, Any], *, candidate: bool, witness: str, input_sha: str, messages_sha: str
) -> dict[str, Any]:
    settings = worker["thread_settings"]
    model = settings["total_budget_model"]
    expected_owner = "qwen_mm_processor_pool" if candidate else "official_torch_intraop"
    if model["owner"] != expected_owner:
        _fail("worker total thread-budget owner changed")
    torch = settings["torch"]
    affinity = worker["affinity"]
    available = affinity["status"] == "attested"
    if available:
        requested = affinity["requested_cpus"]
        observed = affinity["observed_cpus"]
    else:
        if affinity["status"] != "unavailable":
            _fail("worker affinity is neither attested nor explicitly unavailable")
        requested = []
        observed = []
    raw_conformance = _conformance(
        worker, witness_sha=witness, input_sha=input_sha, messages_sha=messages_sha
    )
    return {
        "process_nonce": worker["worker_nonce"],
        "process_pid": worker["process_id"],
        "thread_budget": model["total_budget"],
        "thread_control": {
            "owner": "candidate-processor" if candidate else "official-torch",
            "torch_intraop_threads": torch["num_threads"],
            "torch_interop_threads": torch["num_interop_threads"],
            "processor_thread_budget": model["total_budget"] if candidate else None,
            "environment": dict(settings["environment"]),
        },
        "affinity": {
            "available": available,
            "source": affinity["mechanism"] or affinity["reason"],
            "requested_cpus": list(requested),
            "observed_cpus": list(observed),
        },
        "samples": [
            {
                "sequence": sample["sequence"],
                "wall_ms": sample["wall_ms"],
                "cpu_ms": sample["cpu_ms"],
            }
            for sample in worker["samples"]
        ],
        "timing_floor": dict(worker["timing_floor"]),
        "memory": _memory(worker, candidate=candidate),
        "pre_conformance": raw_conformance,
        "post_conformance": dict(raw_conformance),
    }


def _pair(
    raw: Mapping[str, Any],
    *,
    architecture: str,
    build: str,
    profile: str,
    case_id: str,
    budget: int,
) -> dict[str, Any]:
    if (
        raw.get("profile_alias") != profile
        or raw.get("case_id") != case_id
        or raw.get("build_label") != build
        or raw.get("thread_budget") != budget
        or raw.get("thread_regime") != f"t{budget}"
        or raw.get("work_units") != WORK_UNITS[case_id]
        or raw.get("cache_mode") != ("separated" if case_id == "repeat24_separated" else "disabled")
    ):
        _fail("raw pair metadata differs from its authenticated matrix coordinate")
    implementations = raw["implementations"]
    reference = implementations["reference"]
    candidate = implementations["candidate"]
    inputs = {reference["input_fingerprint"], candidate["input_fingerprint"]}
    logical_inputs = {
        reference["logical_input_fingerprint"],
        candidate["logical_input_fingerprint"],
    }
    messages = {reference["messages_fingerprint"], candidate["messages_fingerprint"]}
    if len(inputs) != 1 or len(logical_inputs) != 1 or len(messages) != 1:
        _fail("paired workers received different exact/logical inputs or messages")
    input_sha = next(iter(inputs))
    logical_input_sha = next(iter(logical_inputs))
    messages_sha = next(iter(messages))
    witness = _digest(
        {
            "input_sha256": input_sha,
            "logical_input_sha256": logical_input_sha,
            "messages_sha256": messages_sha,
            "reference": {
                "output_signature": reference["output_signature"],
                "conformance": reference["conformance"],
            },
            "candidate": {
                "output_signature": candidate["output_signature"],
                "conformance": candidate["conformance"],
            },
        }
    )
    repetition = raw["repetition"]
    return {
        "pair_id": f"{architecture}/{build}/{profile}/{case_id}/t{budget}/r{repetition}",
        "repetition": repetition,
        "order": list(raw["order"]),
        "input_sha256": input_sha,
        "logical_input_sha256": logical_input_sha,
        "messages_sha256": messages_sha,
        "conformance_witness_sha256": witness,
        "implementations": {
            "reference": _implementation(
                reference,
                candidate=False,
                witness=witness,
                input_sha=input_sha,
                messages_sha=messages_sha,
            ),
            "candidate": _implementation(
                candidate,
                candidate=True,
                witness=witness,
                input_sha=input_sha,
                messages_sha=messages_sha,
            ),
        },
    }


def _observations(
    files: Mapping[str, bytes],
    *,
    architecture: str,
    build: str,
    results: Mapping[int, Mapping[str, Any]],
    suite: str = "exhaustive",
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    by_coordinate: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for budget, result in results.items():
        for raw_pair in result["pairs"]:
            coordinate = (raw_pair["profile_alias"], raw_pair["case_id"], budget)
            by_coordinate.setdefault(coordinate, []).append(raw_pair)
    if suite == "compact":
        expected = {
            (profile, case_id, budget)
            for profile in PROFILES
            for budget in COMPACT_THREAD_BUDGETS
            for case_id in COMPACT_CASES_BY_THREAD_BUDGET[budget]
        }
        if set(by_coordinate) != expected:
            _fail(
                "compact raw benchmark matrix is incomplete "
                f"(missing={sorted(expected - set(by_coordinate))}, "
                f"extra={sorted(set(by_coordinate) - expected)})"
            )
        for profile in PROFILES:
            for budget in COMPACT_THREAD_BUDGETS:
                for case_id in COMPACT_CASES_BY_THREAD_BUDGET[budget]:
                    pairs = by_coordinate[(profile, case_id, budget)]
                    if len(pairs) != COMPACT_PROCESS_REPETITIONS:
                        _fail(
                            f"{profile}/{case_id}/t{budget} does not contain "
                            f"{COMPACT_PROCESS_REPETITIONS} pairs"
                        )
                    order_seeds = {pair["order_seed"] for pair in pairs}
                    if len(order_seeds) != 1:
                        _fail(f"{profile}/{case_id}/t{budget} changed its order seed")
                    observations.append(
                        {
                            "profile": profile,
                            "case_id": case_id,
                            "cache_mode": "disabled",
                            "thread_budget": budget,
                            "order_seed": next(iter(order_seeds)),
                            "work_units": WORK_UNITS[case_id],
                            "support_status": "supported",
                            "support_reason": None,
                            "timing_kind": "uncached",
                            "pairs": [
                                _pair(
                                    pair,
                                    architecture=architecture,
                                    build=build,
                                    profile=profile,
                                    case_id=case_id,
                                    budget=budget,
                                )
                                for pair in sorted(pairs, key=lambda item: item["repetition"])
                            ],
                        }
                    )
        return observations
    if suite != "exhaustive":
        _fail(f"unsupported D4 evidence suite: {suite}")
    unsupported = _json_member(files, f"captures/{build}/repeat24_cached-unsupported.json")
    unsupported_coordinates = unsupported.get("coordinates")
    if not isinstance(unsupported_coordinates, list):
        _fail("cached unsupported coordinates are missing")
    for profile in PROFILES:
        for budget in (1, 8):
            matching = [
                value
                for value in unsupported_coordinates
                if value.get("profile_alias") == profile and value.get("thread_budget") == budget
            ]
            if len(matching) != 1 or set(matching[0]) != {
                "profile_alias",
                "thread_budget",
                "reference_cache_supported",
                "candidate_cache_supported",
                "attested_from_case",
                "witness_pair_count",
                "witness_sha256",
            }:
                _fail("cached unsupported attestation is incomplete or duplicated")
            record = matching[0]
            if (
                record["reference_cache_supported"] is not False
                or record["candidate_cache_supported"] is not False
                or record["attested_from_case"] != "repeat24_uncached"
                or record["witness_pair_count"] != 5
            ):
                _fail("cached unsupported attestation was relabeled")
            _sha(record["witness_sha256"], "cached unsupported witness")
    expected = {
        (profile, case, thread)
        for profile in PROFILES
        for case in CASES
        for thread in (1, 8)
        if case != "repeat24_cached"
    } | {(profile, "image24", thread) for profile in PROFILES for thread in (2, 4)}
    if set(by_coordinate) != expected:
        _fail(
            "raw benchmark matrix is incomplete "
            f"(missing={sorted(expected - set(by_coordinate))}, extra={sorted(set(by_coordinate) - expected)})"
        )
    for profile in PROFILES:
        for case_id in CASES:
            for budget in (1, 8) if case_id != "image24" else (1, 2, 4, 8):
                if case_id == "repeat24_cached":
                    observations.append(
                        {
                            "profile": profile,
                            "case_id": case_id,
                            "cache_mode": "enabled",
                            "thread_budget": budget,
                            "order_seed": None,
                            "work_units": WORK_UNITS[case_id],
                            "support_status": "unsupported",
                            "support_reason": "adapter_cache_supported_false",
                            "timing_kind": "unsupported",
                            "pairs": [],
                        }
                    )
                    continue
                pairs = by_coordinate[(profile, case_id, budget)]
                if len(pairs) != 5:
                    _fail(f"{profile}/{case_id}/t{budget} does not contain five pairs")
                order_seeds = {pair["order_seed"] for pair in pairs}
                if len(order_seeds) != 1:
                    _fail(f"{profile}/{case_id}/t{budget} changed its order seed")
                observations.append(
                    {
                        "profile": profile,
                        "case_id": case_id,
                        "cache_mode": "separated"
                        if case_id == "repeat24_separated"
                        else "disabled",
                        "thread_budget": budget,
                        "order_seed": next(iter(order_seeds)),
                        "work_units": WORK_UNITS[case_id],
                        "support_status": "supported",
                        "support_reason": None,
                        "timing_kind": "uncached",
                        "pairs": [
                            _pair(
                                pair,
                                architecture=architecture,
                                build=build,
                                profile=profile,
                                case_id=case_id,
                                budget=budget,
                            )
                            for pair in sorted(pairs, key=lambda item: item["repetition"])
                        ],
                    }
                )
    return observations


def _capture(
    archive: Mapping[str, Any], *, build_label: str, identity: Mapping[str, str]
) -> dict[str, Any]:
    files = archive["files"]
    provenance = archive["provenance"]
    architecture = provenance["architecture_family"]
    suite = _suite(provenance)
    budgets = _budgets(provenance)
    raw_build = _json_member(files, f"builds/{build_label}/build.json")
    capture_fields = {
        "schema_id",
        "schema_version",
        "build_label",
        "executed",
        "created_at",
        "phase_c_pre_and_post",
        "sample_pruning",
        "plan",
    }
    if suite == "compact":
        capture_fields.add("suite")
    capture_metadata = _object(
        _json_member(files, f"builds/{build_label}/capture.json"),
        f"builds.{build_label}.capture",
        capture_fields,
    )
    expected_capture_schema = (
        "qwen-mm-d4-compact-build-capture-v1"
        if suite == "compact"
        else "qwen-mm-d4-build-capture-v1"
    )
    if (
        capture_metadata["schema_id"] != expected_capture_schema
        or capture_metadata["schema_version"] != 1
        or (suite == "compact" and capture_metadata["suite"] != "compact")
        or capture_metadata["build_label"] != build_label
        or capture_metadata["executed"] is not True
        or capture_metadata["phase_c_pre_and_post"] is not True
        or capture_metadata["sample_pruning"] != "forbidden"
        or capture_metadata["plan"] != raw_build.get("plan")
    ):
        _fail(f"{architecture}/{build_label} build capture metadata is inconsistent")
    build = _build_record(raw_build, build_label, provenance)
    results: dict[int, Mapping[str, Any]] = {}
    expected_runtime: Mapping[str, Any] | None = None
    pre_name = f"phase-c/{build_label}/pre/report.json"
    pre_report = _json_member(files, pre_name)
    post_name = f"phase-c/{build_label}/post/report.json"
    post_report = _json_member(files, post_name)
    with tempfile.TemporaryDirectory(prefix="qwen-mm-d4-evidence-") as temporary:
        temporary_root = Path(temporary)
        wheel_contents = _retained_wheel(
            files,
            raw_build,
            build,
            architecture=architecture,
            build_label=build_label,
            temporary=temporary_root,
        )
        pre_root = temporary_root / "phase-c-pre"
        pre_path = _materialize_phase_c_evidence(
            files,
            build_label=build_label,
            position="pre",
            destination=pre_root,
        )
        for budget in budgets:
            result = _json_member(files, f"captures/{build_label}/t{budget}/result.json")
            _assert_raw_protocol(
                result,
                architecture=architecture,
                build=build_label,
                budget=budget,
                suite=suite,
            )
            runtime = (
                result.get("protocol", {}).get("candidate_identity", {}).get("runtime_identity")
            )
            if not isinstance(runtime, Mapping):
                _fail(f"{architecture}/{build_label}/t{budget} lacks runtime identity")
            if runtime.get("native_artifact_sha256") != build["native_sha256"]:
                _fail(f"{architecture}/{build_label}/t{budget} runtime differs from retained build")
            if (
                runtime.get("native_artifact_sha256") != wheel_contents["native_artifact_sha256"]
                or runtime.get("package_artifact_sha256")
                != wheel_contents["package_artifact_sha256"]
                or runtime.get("version") != raw_build["runtime"]["package_version"]
            ):
                _fail(
                    f"{architecture}/{build_label}/t{budget} package/native bytes differ "
                    "from the retained wheel"
                )
            candidate_identity = result.get("protocol", {}).get("candidate_identity", {})
            if (
                not isinstance(candidate_identity, Mapping)
                or candidate_identity.get("artifact_sha256")
                != wheel_contents["benchmark_artifact"]["sha256"]
            ):
                _fail(
                    f"{architecture}/{build_label}/t{budget} adapter wrapper differs "
                    "from the retained wheel"
                )
            if expected_runtime is None:
                expected_runtime = runtime
            elif runtime != expected_runtime:
                _fail(f"{architecture}/{build_label} runtime changed within its matrix")
            try:
                validate_result_authenticated_portable(
                    result,
                    expected_runtime_identity=runtime,
                    phase_c_report_override=pre_path,
                    phase_c_assets_root_override=ASSETS_ROOT,
                )
            except (RuntimeError, TypeError, ValueError) as error:
                raise D4EvidenceError(
                    f"{architecture}/{build_label}/t{budget} failed authenticated validation"
                ) from error
            results[budget] = result
    if expected_runtime is None:
        _fail(f"{architecture}/{build_label} has no authenticated runtime")
    with tempfile.TemporaryDirectory(prefix="qwen-mm-d4-post-phase-c-") as temporary:
        post_root = Path(temporary)
        _materialize_phase_c_evidence(
            files,
            build_label=build_label,
            position="post",
            destination=post_root,
        )
        try:
            _validate_phase_c_report(
                post_report,
                root=REPOSITORY_ROOT,
                assets_root=ASSETS_ROOT,
                candidate_identity={"resolved": True, "runtime_identity": dict(expected_runtime)},
                report_evidence_root=post_root,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise D4EvidenceError(f"{architecture}/{build_label} post Phase C failed") from error
    toolchain = _toolchain(raw_build)
    commands = [
        _command(value, f"builds.{build_label}.commands.{name}")
        for name, value in raw_build["commands"].items()
    ]
    plan = raw_build["plan"]
    commands.extend(
        [
            _command(plan["pre_conformance"], "plan.pre_conformance"),
            *[
                _command(coordinate["command"], "plan.timed_matrix.command")
                for coordinate in plan["timed_matrix"]
            ],
            _command(plan["post_conformance"], "plan.post_conformance"),
        ]
    )
    return {
        "capture_id": f"{architecture}-{build_label}",
        "architecture": architecture,
        "identity": dict(identity),
        "build": build,
        "host": _host(provenance, architecture),
        "toolchain": toolchain,
        "environment": {
            name: provenance["environment"][name] for name in ("PYTHONHASHSEED", "LC_ALL", "TZ")
        },
        "commands": commands,
        "raw_archive_sha256": archive["sha256"],
        "raw_archive_bytes": archive["bytes"],
        "raw_manifest_sha256": archive["manifest_sha256"],
        "raw_provenance_sha256": archive["provenance_sha256"],
        "protocol": dict(PROTOCOL if suite == "compact" else LEGACY_PROTOCOL),
        "phase_c": {
            "before": _phase_c_record(pre_report, files[pre_name], identity, build),
            "after": _phase_c_record(post_report, files[post_name], identity, build),
        },
        "observations": _observations(
            files,
            architecture=architecture,
            build=build_label,
            results=results,
            suite=suite,
        ),
    }


def rebuild_from_archives(arm64_zip: Path, x86_64_zip: Path, *, created_at: str) -> dict[str, Any]:
    archives = [
        _archive(arm64_zip.resolve(), "arm64"),
        _archive(x86_64_zip.resolve(), "x86_64"),
    ]
    identity = _identity([archive["provenance"] for archive in archives])
    suites = {_suite(archive["provenance"]) for archive in archives}
    if len(suites) != 1:
        _fail("cross-host capture suite drifted")
    suite = next(iter(suites))
    build_labels = COMPACT_BUILD_LABELS if suite == "compact" else BUILD_LABELS
    captures = [
        _capture(archive, build_label=build, identity=identity)
        for archive in archives
        for build in build_labels
    ]
    return build_certification(captures, current_identity=identity, created_at=created_at)


def validate_from_archives(
    artifact: Mapping[str, Any], arm64_zip: Path, x86_64_zip: Path
) -> dict[str, Any]:
    created_at = artifact.get("created_at")
    if not isinstance(created_at, str):
        _fail("certification artifact lacks created_at")
    rebuilt = rebuild_from_archives(arm64_zip, x86_64_zip, created_at=created_at)
    validate_certification(rebuilt, current_identity=rebuilt["current_identity"])
    if _canonical(artifact) != _canonical(rebuilt):
        _fail("certification artifact differs from the two authenticated raw archives")
    return rebuilt


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as output:
        temporary = Path(output.name)
        output.write(value)
        output.flush()
    try:
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm64", type=Path, required=True)
    parser.add_argument("--x86-64", dest="x86_64", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        try:
            artifact = json.loads(args.artifact.read_bytes())
        except (OSError, json.JSONDecodeError) as error:
            raise D4EvidenceError(
                "certification artifact is unavailable or invalid JSON"
            ) from error
        rebuilt = validate_from_archives(artifact, args.arm64, args.x86_64)
        if args.report is not None:
            expected_report = (
                render_report(rebuilt, current_identity=rebuilt["current_identity"]) + "\n"
            )
            try:
                observed_report = args.report.read_text(encoding="utf-8")
            except OSError as error:
                raise D4EvidenceError("certification report is unavailable") from error
            if observed_report != expected_report:
                _fail("certification report differs from the rebuilt authenticated result")
        print(f"validated D4 certification: {args.artifact}")
        return
    created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    artifact = rebuild_from_archives(args.arm64, args.x86_64, created_at=created_at)
    _atomic_write(args.artifact, json.dumps(artifact, indent=2, sort_keys=True).encode() + b"\n")
    report_path = args.report or args.artifact.with_suffix(".md")
    report = render_report(artifact, current_identity=artifact["current_identity"])
    _atomic_write(report_path, report.encode() + b"\n")
    print(f"wrote D4 certification ({artifact['certification_status']}): {args.artifact}")


if __name__ == "__main__":
    main()
