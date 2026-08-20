from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import re
import secrets
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark_protocol import (
    RESULT_SCHEMA_ID,
    RESULT_SCHEMA_VERSION,
    THREAD_ENVIRONMENT_NAMES,
    TIMING_FLOOR_POLICY,
    TIMING_SAMPLE_FIELDS,
    BenchmarkProtocolError,
    architecture_family,
    environment_metadata,
    load_workload,
    materialize_case,
    media_source_dimensions,
    percentile,
    physical_cpu_count,
    run_worker,
    select_cases,
    summarize_samples,
    total_thread_budget_model,
    workload_provenance,
)
from .fixtures import repository_root
from .phase_c_conformance import (
    expected_candidate_case_ids,
)
from .phase_c_conformance import (
    source_inputs as phase_c_source_inputs,
)
from .phase_c_conformance import (
    validate_report as validate_phase_c_report,
)
from .phase_c_overlay_v2 import SCHEMA_ID as PHASE_C_OVERLAY_SCHEMA_ID
from .phase_c_overlay_v2 import SCHEMA_VERSION as PHASE_C_OVERLAY_SCHEMA_VERSION
from .phase_c_overlay_v2 import validate_overlay as validate_phase_c_overlay
from .resize_conformance_v2 import COMPARISON_ID as RESIZE_V2_COMPARISON_ID
from .resize_conformance_v2 import validate_witness as validate_resize_v2_witness

DEFAULT_WORKLOAD = Path("benchmarks/workloads-v2.json")
DEFAULT_PHASE_C_REPORT = Path("reference/phase-c/v2/report.json")
DEFAULT_PHASE_C_ASSETS_ROOT = Path("reference/.cache/huggingface")
PHASE_C_REPORT_SCHEMA_ID = "qwen-mm-phase-c-conformance-report-v1"
PHASE_C_REPORT_SCHEMA_VERSION = 1
PHASE_C_PROFILES = {"qwen3-vl-8b", "qwen3.5-9b"}
IMAGE_BOUNDARIES = {"encoded_to_numpy", "rgb_to_vllm_ready"}
PHASE_C_GATE_STATUSES = {
    "artifact_mismatch",
    "invalid",
    "missing",
    "not_applicable",
    "pass",
    "stale",
}
OUTPUT_COMPARISON_POLICY = {
    "official_candidate_still_image": RESIZE_V2_COMPARISON_ID,
    "non_image": "qwen-mm-exact-output-comparison-v1",
    "candidate_self_comparison": "qwen-mm-exact-output-comparison-v1",
    "pre_post_witnesses": "required",
}
MODE_DEFAULTS: dict[str, dict[str, Any]] = {
    "smoke": {
        "process_repetitions": 1,
        "warmups": 1,
        "minimum_samples": 3,
        "minimum_seconds": 0.0,
        "tag": "smoke",
        "thread_regimes": ["one"],
        "build_labels": ["shipping"],
    },
    "dedicated": {
        "process_repetitions": 5,
        "warmups": 3,
        "minimum_samples": 30,
        "minimum_seconds": 5.0,
        "tag": "release",
        "thread_regimes": ["one", "production"],
        "build_labels": ["shipping", "native"],
    },
}


class PhaseCGateError(BenchmarkProtocolError):
    def __init__(self, status: str, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.reason_code = reason_code


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else repository_root() / path


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _display_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def _qwen_runtime_identity() -> dict[str, Any] | None:
    try:
        package_spec = importlib.util.find_spec("qwen_mm")
        native_spec = importlib.util.find_spec("qwen_mm._native")
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    package_origin = None if package_spec is None else package_spec.origin
    native_origin = None if native_spec is None else native_spec.origin
    if (
        package_origin is None
        or native_origin is None
        or not Path(package_origin).is_file()
        or not Path(native_origin).is_file()
    ):
        return None
    try:
        version = importlib.metadata.version("qwen-mm")
    except importlib.metadata.PackageNotFoundError:
        try:
            version = str(importlib.import_module("qwen_mm").__version__)
        except (AttributeError, ImportError):
            return None
    try:
        return {
            "package": "qwen_mm",
            "version": version,
            "package_artifact_sha256": _sha256_path(Path(package_origin)),
            "native_module": "qwen_mm._native",
            "native_artifact_sha256": _sha256_path(Path(native_origin)),
        }
    except OSError:
        return None


def candidate_artifact_identity(adapter_spec: str) -> dict[str, Any]:
    """Compute the import artifact identity for a candidate adapter.

    The adapter wrapper and the qwen-mm package/native runtime are hashed
    separately. Phase C attests the runtime; result validation independently
    re-derives the wrapper identity instead of trusting serialized hashes.
    """

    if adapter_spec in {"official", "synthetic"}:
        return {
            "adapter_spec": adapter_spec,
            "kind": adapter_spec,
            "resolved": False,
        }
    if ":" not in adapter_spec:
        return {
            "adapter_spec": adapter_spec,
            "kind": "module",
            "resolved": False,
        }
    module_name, _ = adapter_spec.split(":", 1)
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None
    origin = None if spec is None else spec.origin
    if origin is None or not Path(origin).is_file():
        return {
            "adapter_spec": adapter_spec,
            "kind": "module",
            "resolved": False,
        }

    top_level = module_name.partition(".")[0]
    distributions = sorted(importlib.metadata.packages_distributions().get(top_level, ()))
    versions: dict[str, str] = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    origin_path = Path(origin).resolve()
    try:
        artifact_sha256 = _sha256_path(origin_path)
    except OSError:
        return {
            "adapter_spec": adapter_spec,
            "kind": "module",
            "resolved": False,
        }
    identity: dict[str, Any] = {
        "adapter_spec": adapter_spec,
        "kind": "module",
        "resolved": True,
        "module": module_name,
        "origin": str(origin_path),
        "artifact_sha256": artifact_sha256,
        "distributions": versions,
        "runtime_identity": _qwen_runtime_identity(),
    }
    return identity


def _stable_candidate_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Drop host-specific paths while retaining artifact and package identity."""

    return {key: value for key, value in identity.items() if key not in {"native_origin", "origin"}}


def _portable_candidate_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Retain architecture-independent adapter identity for foreign-host validation."""

    return {
        key: identity.get(key)
        for key in ("adapter_spec", "kind", "resolved", "module", "artifact_sha256")
    }


def _current_portable_candidate_identity(adapter_spec: str) -> dict[str, Any]:
    """Hash the repository adapter wrapper without importing qwen-mm or its extension."""

    if adapter_spec != "qwen_mm.benchmark:create_adapter":
        raise BenchmarkProtocolError(
            "authenticated portable validation requires the frozen qwen-mm benchmark adapter"
        )
    adapter_path = (
        repository_root() / "crates/qwen-mm-python/python/qwen_mm/benchmark.py"
    ).resolve()
    try:
        artifact_sha256 = _sha256_path(adapter_path)
    except OSError as error:
        raise BenchmarkProtocolError("qwen-mm benchmark adapter wrapper is missing") from error
    return {
        "adapter_spec": adapter_spec,
        "kind": "module",
        "resolved": True,
        "module": "qwen_mm.benchmark",
        "artifact_sha256": artifact_sha256,
    }


def _validate_runtime_identity(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "package",
        "version",
        "package_artifact_sha256",
        "native_module",
        "native_artifact_sha256",
    }:
        raise BenchmarkProtocolError("benchmark candidate runtime identity is incomplete")
    if value.get("package") != "qwen_mm" or value.get("native_module") != "qwen_mm._native":
        raise BenchmarkProtocolError("benchmark candidate runtime identity names are invalid")
    if not isinstance(value.get("version"), str) or not value["version"]:
        raise BenchmarkProtocolError("benchmark candidate runtime version is invalid")
    for field in ("package_artifact_sha256", "native_artifact_sha256"):
        digest = value.get(field)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise BenchmarkProtocolError(f"benchmark candidate runtime {field} is invalid")
    return value


def _phase_c_has_performance_claim(value: Any) -> bool:
    forbidden = {
        "benchmark",
        "duration",
        "latency",
        "performance",
        "speed",
        "speedup",
        "throughput",
        "timing",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(token in forbidden for token in normalized.split("_")):
                return True
            if _phase_c_has_performance_claim(child):
                return True
    elif isinstance(value, list):
        return any(_phase_c_has_performance_claim(child) for child in value)
    elif isinstance(value, str):
        return (
            re.search(
                r"\b(?:faster|latency|performance|speed|speedup|throughput|timing)\b|wall[_ -]?ms",
                value,
                flags=re.IGNORECASE,
            )
            is not None
        )
    return False


def _validate_phase_c_report(
    report: Mapping[str, Any],
    *,
    root: Path,
    assets_root: Path,
    candidate_identity: Mapping[str, Any],
    report_evidence_root: Path | None = None,
) -> dict[str, Any]:
    if _phase_c_has_performance_claim(report):
        raise PhaseCGateError(
            "invalid",
            "performance_claim",
            "Phase C correctness evidence must not contain performance claims",
        )
    is_resize_v2_overlay = report.get("schema_id") == PHASE_C_OVERLAY_SCHEMA_ID
    try:
        if is_resize_v2_overlay:
            validate_phase_c_overlay(report, evidence_root=report_evidence_root)
        else:
            validate_phase_c_report(report, assets_root=assets_root)
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
        detail = str(error)
        stale = any(token in detail.lower() for token in ("drifted", "mismatch", "stale"))
        raise PhaseCGateError(
            "stale" if stale else "invalid",
            "report_currentness" if stale else "report_validation",
            detail,
        ) from error

    effective_report = report
    if is_resize_v2_overlay:
        base_record = report["base_phase_c_v1"]["artifact"]
        effective_report = _load_json(root / base_record["path"])
    scope = effective_report["scope"]
    declared = scope["declared_case_ids"]
    canonical_ids = expected_candidate_case_ids(root)
    if declared != canonical_ids:
        raise PhaseCGateError(
            "stale",
            "case_inventory_changed",
            "Phase C report does not match the canonical candidate case inventory",
        )
    expected_input_paths = [path.as_posix() for path in phase_c_source_inputs()]
    reported_inputs = effective_report.get("inputs")
    if not is_resize_v2_overlay and (
        not isinstance(reported_inputs, list)
        or [item.get("path") for item in reported_inputs] != expected_input_paths
    ):
        raise PhaseCGateError(
            "stale",
            "input_inventory_changed",
            "Phase C report does not match the canonical authenticated input inventory",
        )

    candidate = effective_report.get("candidate")
    expected_runtime = (
        report.get("current_candidate", {}).get("capture", {}).get("runtime_identity")
        if is_resize_v2_overlay
        else (None if not isinstance(candidate, Mapping) else candidate.get("runtime_identity"))
    )
    observed_runtime = candidate_identity.get("runtime_identity")
    if (
        not candidate_identity.get("resolved")
        or not isinstance(observed_runtime, Mapping)
        or expected_runtime != observed_runtime
    ):
        raise PhaseCGateError(
            "artifact_mismatch",
            "candidate_runtime_mismatch",
            "Phase C candidate runtime does not match the benchmark candidate runtime",
        )
    return {
        "schema_id": (
            PHASE_C_OVERLAY_SCHEMA_ID if is_resize_v2_overlay else PHASE_C_REPORT_SCHEMA_ID
        ),
        "schema_version": (
            PHASE_C_OVERLAY_SCHEMA_VERSION
            if is_resize_v2_overlay
            else PHASE_C_REPORT_SCHEMA_VERSION
        ),
        "contract_id": report.get("contract_id"),
        "profiles": sorted(PHASE_C_PROFILES),
        "declared_case_ids": declared,
        "candidate_runtime_identity": dict(observed_runtime),
        "inputs": reported_inputs,
        **(
            {
                "base_report_sha256": report["base_phase_c_v1"]["artifact"]["sha256"],
                "resize_holdout_sha256": report["resize_v2"]["holdout_manifest"]["sha256"],
                "production_rgb8_sha256": report["resize_v2"]["production_rgb8"]["sha256"],
            }
            if is_resize_v2_overlay
            else {}
        ),
    }


def evaluate_image_release_gate(
    *,
    mode: str,
    selected_cases: Sequence[Mapping[str, Any]],
    candidate_identity: Mapping[str, Any],
    phase_c_report_path: Path | None,
    phase_c_assets_root: Path | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate the C3 prerequisite while keeping all pre-D4 results diagnostic."""

    effective_root = (root or repository_root()).resolve()
    has_images = any(case.get("boundary") in IMAGE_BOUNDARIES for case in selected_cases)
    base: dict[str, Any] = {
        "releasable": False,
        "performance_status": "diagnostic_only",
        "phase_c": {
            "status": "not_applicable" if not has_images else "missing",
            "reason_codes": [] if not has_images else ["report_missing"],
        },
    }
    if not has_images:
        return base

    report_path = phase_c_report_path or effective_root / DEFAULT_PHASE_C_REPORT
    if not report_path.is_absolute():
        report_path = effective_root / report_path
    assets_root = phase_c_assets_root or effective_root / DEFAULT_PHASE_C_ASSETS_ROOT
    if not assets_root.is_absolute():
        assets_root = effective_root / assets_root
    phase_c = base["phase_c"]
    phase_c["report_path"] = _display_path(report_path, effective_root)
    phase_c["assets_root"] = _display_path(assets_root, effective_root)
    phase_c["mode_note"] = (
        "dedicated measurement remains diagnostic until D4"
        if mode == "dedicated"
        else "smoke measurement is never release evidence"
    )
    if not report_path.is_file():
        return base

    try:
        report = _load_json(report_path)
        if not isinstance(report, Mapping):
            raise PhaseCGateError("invalid", "report_type", "Phase C report must be a JSON object")
        report_sha256 = _sha256_path(report_path)
        evidence = _validate_phase_c_report(
            report,
            root=effective_root,
            assets_root=assets_root,
            candidate_identity=candidate_identity,
            report_evidence_root=report_path.parent,
        )
    except (json.JSONDecodeError, OSError) as error:
        phase_c.update(status="invalid", reason_codes=["report_unreadable"], detail=str(error))
        return base
    except PhaseCGateError as error:
        phase_c.update(
            status=error.status,
            reason_codes=[error.reason_code],
            detail=str(error),
        )
        return base

    fingerprint_material = {
        "report_sha256": report_sha256,
        "evidence": evidence,
    }
    phase_c.update(
        status="pass",
        reason_codes=[],
        report_sha256=report_sha256,
        gate_fingerprint=hashlib.sha256(_canonical_json(fingerprint_material)).hexdigest(),
        evidence=evidence,
    )
    return base


def bootstrap_confidence_interval(
    values: Sequence[float], *, seed: int, resamples: int = 2_000
) -> dict[str, float | int]:
    if not values:
        raise BenchmarkProtocolError("cannot bootstrap an empty speedup set")
    materialized = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(materialized), size=(resamples, len(materialized)))
    medians = np.median(materialized[indices], axis=1)
    return {
        "method": "paired process-repetition bootstrap of median speedup",
        "resamples": resamples,
        "lower": float(np.quantile(medians, 0.025)),
        "upper": float(np.quantile(medians, 0.975)),
    }


def _thread_budget(regime: str, *, production_thread_budget: int | None = None) -> int:
    if regime == "one":
        return 1
    if regime == "production":
        return production_thread_budget or physical_cpu_count()
    numeric = regime[1:] if re.fullmatch(r"t[1-9][0-9]*", regime) else regime
    try:
        value = int(numeric)
    except ValueError as error:
        raise BenchmarkProtocolError(f"invalid thread regime: {regime}") from error
    if value <= 0:
        raise BenchmarkProtocolError("thread budget must be positive")
    return value


def randomized_orders(repetitions: int, *, seed: int) -> list[list[str]]:
    if repetitions <= 0:
        raise BenchmarkProtocolError("process repetitions must be positive")
    rng = random.Random(seed)
    orders: list[list[str]] = []
    for repetition in range(repetitions):
        order = ["reference", "candidate"]
        if rng.randrange(2):
            order.reverse()
        if repetition > 0 and order == orders[-1] and repetitions > 2:
            order.reverse()
        orders.append(order)
    return orders


def _run_subprocess_worker(
    *,
    config: Mapping[str, Any],
    workload_path: Path,
    case_id: str,
    temporary_directory: Path,
) -> dict[str, Any]:
    token = (
        f"{config['implementation']}-{case_id}-{config['profile_alias']}-"
        f"{config['build_label']}-{config['thread_budget']}-{config['repetition']}"
    )
    config_path = temporary_directory / f"{token}-config.json"
    output_path = temporary_directory / f"{token}-output.json"
    _write_json(config_path, config)
    command = [
        sys.executable,
        "-m",
        "qwen_mm_reference.benchmark_v2",
        "_worker",
        "--config",
        str(config_path),
        "--workload",
        str(workload_path),
        "--case",
        case_id,
        "--output",
        str(output_path),
    ]
    requested_affinity = config.get("affinity_cpus")
    if platform.system() == "Linux" and requested_affinity is not None:
        taskset = shutil.which("taskset")
        if taskset is None:
            raise BenchmarkProtocolError("Linux affinity was requested but taskset is unavailable")
        cpu_list = ",".join(str(cpu) for cpu in requested_affinity)
        command = [taskset, "--cpu-list", cpu_list, *command]
    worker_environment = os.environ.copy()
    budget_model = total_thread_budget_model(
        str(config["adapter_spec"]), int(config["thread_budget"])
    )
    worker_environment.update(budget_model["environment"])
    completed = subprocess.run(
        command,
        cwd=repository_root(),
        check=False,
        capture_output=True,
        text=True,
        env=worker_environment,
    )
    if completed.returncode != 0:
        raise BenchmarkProtocolError(
            f"benchmark worker failed ({' '.join(command)}):\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    result = _load_json(output_path)
    result["worker_stdout"] = completed.stdout.strip()
    result["worker_stderr"] = completed.stderr.strip()
    return result


def _pair_key(pair: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(pair["profile_alias"]),
        str(pair["case_id"]),
        str(pair["build_label"]),
        int(pair["thread_budget"]),
    )


def summarize_pairs(pairs: Sequence[Mapping[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[_pair_key(pair)].append(pair)
    summaries: list[dict[str, Any]] = []
    for index, (key, group) in enumerate(sorted(grouped.items())):
        profile_alias, case_id, build_label, thread_budget = key
        reference_samples = [
            float(sample["wall_ms"])
            for pair in group
            for sample in pair["implementations"]["reference"]["samples"]
        ]
        candidate_samples = [
            float(sample["wall_ms"])
            for pair in group
            for sample in pair["implementations"]["candidate"]["samples"]
        ]
        speedups = [float(pair["paired_speedup_p50"]) for pair in group]
        summaries.append(
            {
                "profile_alias": profile_alias,
                "case_id": case_id,
                "release_name": group[0].get("release_name"),
                "build_label": build_label,
                "thread_budget": thread_budget,
                "process_repetitions": len(group),
                "reference": {
                    "sample_count": len(reference_samples),
                    "p50_ms": percentile(reference_samples, 0.50),
                    "p90_ms": percentile(reference_samples, 0.90),
                    "p99_qualified": len(reference_samples) >= 100,
                    "p99_ms": (
                        percentile(reference_samples, 0.99)
                        if len(reference_samples) >= 100
                        else None
                    ),
                },
                "candidate": {
                    "sample_count": len(candidate_samples),
                    "p50_ms": percentile(candidate_samples, 0.50),
                    "p90_ms": percentile(candidate_samples, 0.90),
                    "p99_qualified": len(candidate_samples) >= 100,
                    "p99_ms": (
                        percentile(candidate_samples, 0.99)
                        if len(candidate_samples) >= 100
                        else None
                    ),
                },
                "paired_process_speedups": speedups,
                "speedup_p50": statistics.median(speedups),
                "speedup_bootstrap_95_ci": bootstrap_confidence_interval(
                    speedups, seed=seed + index
                ),
            }
        )
    return summaries


def _validate_release_eligibility(
    result: Mapping[str, Any],
    *,
    verify_live_runtime: bool = True,
    phase_c_report_override: Path | None = None,
) -> None:
    eligibility = result.get("release_eligibility")
    if not isinstance(eligibility, Mapping):
        raise BenchmarkProtocolError("benchmark result lacks release eligibility metadata")
    if eligibility.get("releasable") is not False:
        raise BenchmarkProtocolError(
            "pre-D4 benchmark results must remain diagnostic-only and unreleasable"
        )
    if eligibility.get("performance_status") != "diagnostic_only":
        raise BenchmarkProtocolError("benchmark performance status must be diagnostic_only")
    phase_c = eligibility.get("phase_c")
    if not isinstance(phase_c, Mapping) or phase_c.get("status") not in PHASE_C_GATE_STATUSES:
        raise BenchmarkProtocolError("benchmark result has invalid Phase C gate metadata")
    reason_codes = phase_c.get("reason_codes")
    if not isinstance(reason_codes, list) or not all(
        isinstance(reason, str) for reason in reason_codes
    ):
        raise BenchmarkProtocolError("Phase C gate reason codes must be strings")
    if phase_c.get("status") in {"not_applicable", "pass"} and reason_codes:
        raise BenchmarkProtocolError("passing or inapplicable Phase C gate cannot have reasons")
    if phase_c.get("status") not in {"not_applicable", "pass"} and not reason_codes:
        raise BenchmarkProtocolError("blocked Phase C gate requires a reason code")

    protocol = result.get("protocol")
    if not isinstance(protocol, Mapping):
        raise BenchmarkProtocolError("benchmark result lacks protocol metadata")
    candidate_adapter = protocol.get("candidate_adapter")
    recorded_candidate_identity = protocol.get("candidate_identity")
    if not isinstance(candidate_adapter, str) or not isinstance(
        recorded_candidate_identity, Mapping
    ):
        raise BenchmarkProtocolError("benchmark result lacks candidate artifact identity")
    current_candidate_identity = recorded_candidate_identity
    if verify_live_runtime:
        current_candidate_identity = candidate_artifact_identity(candidate_adapter)
        if _stable_candidate_identity(current_candidate_identity) != _stable_candidate_identity(
            recorded_candidate_identity
        ):
            raise BenchmarkProtocolError(
                "benchmark candidate adapter or runtime artifact changed after measurement"
            )
    boundaries = protocol.get("case_boundaries")
    if not isinstance(boundaries, Mapping):
        raise BenchmarkProtocolError("benchmark protocol lacks case boundary metadata")
    has_images = any(boundary in IMAGE_BOUNDARIES for boundary in boundaries.values())
    if not has_images:
        if phase_c.get("status") != "not_applicable":
            raise BenchmarkProtocolError("text-only benchmark must mark Phase C not_applicable")
        return
    if phase_c.get("status") == "not_applicable":
        raise BenchmarkProtocolError("image benchmark cannot mark Phase C not_applicable")

    if phase_c.get("status") == "pass" and verify_live_runtime:
        report_path = phase_c.get("report_path")
        assets_root = phase_c.get("assets_root")
        if not isinstance(report_path, str) or not isinstance(assets_root, str):
            raise BenchmarkProtocolError("passing Phase C gate lacks reproducibility metadata")
        selected_cases = [{"boundary": boundary} for boundary in boundaries.values()]
        recomputed = evaluate_image_release_gate(
            mode=str(result.get("mode")),
            selected_cases=selected_cases,
            candidate_identity=current_candidate_identity,
            phase_c_report_path=(
                phase_c_report_override
                if phase_c_report_override is not None
                else Path(report_path)
            ),
            phase_c_assets_root=Path(assets_root),
        )
        if phase_c_report_override is not None:
            recomputed["phase_c"]["report_path"] = report_path
        if recomputed != eligibility:
            raise BenchmarkProtocolError("Phase C gate is stale or has been tampered with")


def _authenticated_workload(result: Mapping[str, Any]) -> dict[str, Any]:
    provenance = result.get("workload")
    if not isinstance(provenance, Mapping):
        raise BenchmarkProtocolError("benchmark result lacks workload provenance")
    root = repository_root().resolve()

    def resolve_recorded_path(field: str) -> Path:
        value = provenance.get(field)
        if not isinstance(value, str) or not value or Path(value).is_absolute():
            raise BenchmarkProtocolError(f"benchmark workload {field} must be repository-relative")
        path = (root / value).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise BenchmarkProtocolError(
                f"benchmark workload {field} escapes the repository"
            ) from error
        if not path.is_file():
            raise BenchmarkProtocolError(f"benchmark workload {field} is missing")
        return path

    workload_path = resolve_recorded_path("path")
    schema_path = resolve_recorded_path("schema_path")
    expected_schema = (root / "benchmarks" / "workload-schema-v2.json").resolve()
    if schema_path != expected_schema:
        raise BenchmarkProtocolError("benchmark workload schema path was relabeled")
    try:
        workload_sha256 = _sha256_path(workload_path)
        schema_sha256 = _sha256_path(schema_path)
    except OSError as error:
        raise BenchmarkProtocolError("benchmark workload provenance is unreadable") from error
    if provenance.get("sha256") != workload_sha256:
        raise BenchmarkProtocolError("benchmark workload changed after measurement")
    if provenance.get("schema_sha256") != schema_sha256:
        raise BenchmarkProtocolError("benchmark workload schema changed after measurement")
    try:
        return load_workload(workload_path)
    except (json.JSONDecodeError, OSError) as error:
        raise BenchmarkProtocolError("authenticated benchmark workload is invalid") from error


def _finite_number(value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkProtocolError("benchmark sample values must be numeric")
    number = float(value)
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        raise BenchmarkProtocolError("benchmark sample values must be finite and non-negative")
    return number


def _nonnegative_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkProtocolError(f"benchmark {field} must be a non-negative integer")
    return value


def _optional_nonnegative_integer(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer(value, field=field)


def _validate_resource_scope(value: Any) -> None:
    fields = {
        "request_index",
        "message_index",
        "content_item_index",
        "media_index",
        "input_index",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise BenchmarkProtocolError("native resource scope is invalid")
    if any(
        index is not None and (isinstance(index, bool) or not isinstance(index, int) or index < 0)
        for index in value.values()
    ):
        raise BenchmarkProtocolError("native resource scope index is invalid")


def _validate_resource_census(
    value: Any, *, primary_adapter: str, implementation_name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "timing_separation",
        "adapter_spec",
        "output_conformance",
        "output_bytes",
        "rss",
        "adapter_metrics",
        "native_observed",
    }:
        raise BenchmarkProtocolError("benchmark resource census is incomplete")
    expected_adapter = (
        "qwen_mm.benchmark:create_observed_adapter"
        if primary_adapter == "qwen_mm.benchmark:create_adapter"
        else primary_adapter
    )
    if (
        value.get("timing_separation") != "after_all_timed_samples"
        or value.get("adapter_spec") != expected_adapter
        or value.get("output_conformance") != "exact_pass"
        or _nonnegative_integer(value.get("output_bytes"), field="resource output bytes") <= 0
    ):
        raise BenchmarkProtocolError("benchmark resource census was relabeled")
    rss = value.get("rss")
    if not isinstance(rss, Mapping) or set(rss) != {
        "source",
        "sampler",
        "baseline_rss_bytes",
        "peak_rss_bytes",
        "retained_rss_bytes",
        "rss_after_bytes",
        "retained_rss_delta_bytes",
        "transient_rss_bytes",
        "external_transient_rss_bytes",
        "sample_count",
    }:
        raise BenchmarkProtocolError("benchmark RSS census is incomplete")
    if rss.get("source") not in {"darwin-mach-task-info", "linux-procfs-statm"}:
        raise BenchmarkProtocolError("benchmark RSS census source is invalid")
    if rss.get("sampler") != "os_current_rss_sampler_thread":
        raise BenchmarkProtocolError("benchmark RSS census did not use OS current RSS")
    baseline = _nonnegative_integer(rss.get("baseline_rss_bytes"), field="RSS baseline")
    peak = _nonnegative_integer(rss.get("peak_rss_bytes"), field="RSS peak")
    retained = _nonnegative_integer(rss.get("retained_rss_bytes"), field="retained RSS")
    rss_after = _nonnegative_integer(rss.get("rss_after_bytes"), field="RSS after")
    retained_delta = _nonnegative_integer(
        rss.get("retained_rss_delta_bytes"), field="retained RSS delta"
    )
    transient = _nonnegative_integer(rss.get("transient_rss_bytes"), field="transient RSS")
    external_transient = _nonnegative_integer(
        rss.get("external_transient_rss_bytes"), field="external transient RSS"
    )
    samples = _nonnegative_integer(rss.get("sample_count"), field="RSS sample count")
    if (
        samples < 2
        or peak < max(baseline, retained)
        or rss_after != retained
        or retained_delta != max(0, retained - baseline)
        or transient != max(0, peak - baseline - value["output_bytes"])
        or external_transient != transient
    ):
        raise BenchmarkProtocolError("benchmark RSS census envelope is inconsistent")
    metrics = value.get("adapter_metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != {
        "allocation_count",
        "copy_count",
        "copied_bytes",
        "retained_final_output_bytes",
        "peak_transient_live_bytes",
        "copy_count_scope",
        "cache_supported",
    }:
        raise BenchmarkProtocolError("benchmark adapter resource counters are incomplete")
    for field in (
        "allocation_count",
        "copy_count",
        "copied_bytes",
        "retained_final_output_bytes",
        "peak_transient_live_bytes",
    ):
        _optional_nonnegative_integer(metrics.get(field), field=f"adapter {field}")
    if metrics.get("copy_count_scope") is not None and not isinstance(
        metrics.get("copy_count_scope"), str
    ):
        raise BenchmarkProtocolError("benchmark copy-count scope is invalid")
    if metrics.get("cache_supported") not in {None, True, False}:
        raise BenchmarkProtocolError("benchmark cache-support counter is invalid")

    native = value.get("native_observed")
    expects_native = implementation_name == "candidate" and expected_adapter.endswith(
        ":create_observed_adapter"
    )
    if expects_native != (native is not None):
        raise BenchmarkProtocolError("candidate native resource census is missing or misplaced")
    if native is None:
        return value
    if not isinstance(native, Mapping) or set(native) != {
        "schema_version",
        "outcome",
        "dropped_events",
        "duration_ns",
        "counter_scope",
        "allocations",
        "buffers",
        "copies",
        "calls",
    }:
        raise BenchmarkProtocolError("native observed resource census is incomplete")
    if (
        native.get("schema_version") != "qwen-mm-observation-v1"
        or native.get("outcome") != "success"
        or native.get("dropped_events") != 0
        or not isinstance(native.get("counter_scope"), str)
        or not native["counter_scope"]
    ):
        raise BenchmarkProtocolError("native observed resource census is not lossless")
    duration_ns = _nonnegative_integer(
        native.get("duration_ns"), field="native observation duration"
    )
    allocations = native.get("allocations")
    allocation_fields = {
        "allocation_count",
        "allocated_bytes",
        "copy_count",
        "copied_bytes",
        "transient_live_bytes",
        "peak_transient_live_bytes",
        "retained_final_output_bytes",
    }
    if not isinstance(allocations, Mapping) or set(allocations) != allocation_fields:
        raise BenchmarkProtocolError("native allocation census is incomplete")
    for field in allocation_fields:
        _nonnegative_integer(allocations.get(field), field=f"native {field}")
    buffers = native.get("buffers")
    if not isinstance(buffers, Mapping) or set(buffers) != {
        "count",
        "total_bytes",
        "bytes_by_class",
        "records",
    }:
        raise BenchmarkProtocolError("native buffer census is incomplete")
    count = _nonnegative_integer(buffers.get("count"), field="native buffer count")
    total = _nonnegative_integer(buffers.get("total_bytes"), field="native buffer bytes")
    by_class = buffers.get("bytes_by_class")
    if not isinstance(by_class, Mapping) or any(
        not isinstance(name, str)
        or not name
        or _nonnegative_integer(size, field="native buffer class bytes") < 0
        for name, size in by_class.items()
    ):
        raise BenchmarkProtocolError("native buffer class census is invalid")
    records = buffers.get("records")
    if not isinstance(records, list) or len(records) != count:
        raise BenchmarkProtocolError("native buffer record census is incomplete")
    record_bytes_by_class: dict[str, int] = {}
    for sequence, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != {
            "sequence",
            "name",
            "class",
            "scope",
            "bytes",
            "allocated_at_ns",
            "released_at_ns",
        }:
            raise BenchmarkProtocolError("native buffer record is incomplete")
        if (
            record.get("sequence") != sequence
            or not isinstance(record.get("name"), str)
            or not record["name"]
            or record.get("class") not in {"retained_output", "discarded_output", "transient"}
            or not isinstance(record.get("scope"), Mapping)
        ):
            raise BenchmarkProtocolError("native buffer record is invalid")
        _validate_resource_scope(record["scope"])
        size = _nonnegative_integer(record.get("bytes"), field="native buffer record bytes")
        allocated_at = _nonnegative_integer(
            record.get("allocated_at_ns"), field="native buffer allocation timestamp"
        )
        released_at = _optional_nonnegative_integer(
            record.get("released_at_ns"), field="native buffer release timestamp"
        )
        if allocated_at > duration_ns or (
            released_at is not None and not allocated_at <= released_at <= duration_ns
        ):
            raise BenchmarkProtocolError("native buffer record lifetime is invalid")
        buffer_class = str(record["class"])
        record_bytes_by_class[buffer_class] = record_bytes_by_class.get(buffer_class, 0) + size
    copies = native.get("copies")
    if not isinstance(copies, list):
        raise BenchmarkProtocolError("native copy record census is incomplete")
    copied_bytes = 0
    for sequence, copy_event in enumerate(copies):
        if not isinstance(copy_event, Mapping) or set(copy_event) != {
            "sequence",
            "name",
            "scope",
            "bytes",
        }:
            raise BenchmarkProtocolError("native copy record is incomplete")
        if (
            copy_event.get("sequence") != sequence
            or not isinstance(copy_event.get("name"), str)
            or not copy_event["name"]
            or not isinstance(copy_event.get("scope"), Mapping)
        ):
            raise BenchmarkProtocolError("native copy record is invalid")
        _validate_resource_scope(copy_event["scope"])
        copied_bytes += _nonnegative_integer(
            copy_event.get("bytes"), field="native copy record bytes"
        )
    if (
        allocations["allocation_count"] != count
        or allocations["allocated_bytes"] != total
        or sum(by_class.values()) != total
        or record_bytes_by_class != dict(by_class)
        or allocations["copy_count"] != len(copies)
        or allocations["copied_bytes"] != copied_bytes
        or allocations["retained_final_output_bytes"] != value["output_bytes"]
    ):
        raise BenchmarkProtocolError("native buffer and allocation censuses differ")
    for adapter_field, native_field in (
        ("allocation_count", "allocation_count"),
        ("copy_count", "copy_count"),
        ("copied_bytes", "copied_bytes"),
        ("retained_final_output_bytes", "retained_final_output_bytes"),
        ("peak_transient_live_bytes", "peak_transient_live_bytes"),
    ):
        if metrics[adapter_field] != allocations[native_field]:
            raise BenchmarkProtocolError("native and adapter resource counters differ")
    calls = native.get("calls")
    call_fields = {
        "public_python_calls",
        "native_batch_calls",
        "native_visual_calls",
        "python_callbacks",
        "hugging_face_calls",
        "qwen_vl_utils_calls",
        "pillow_calls",
        "torchvision_calls",
    }
    if not isinstance(calls, Mapping) or set(calls) != call_fields:
        raise BenchmarkProtocolError("native call census is incomplete")
    for field in call_fields:
        _nonnegative_integer(calls.get(field), field=f"native {field}")
    if (
        calls["public_python_calls"] != 1
        or calls["native_batch_calls"] != 1
        or any(
            calls[field] != 0
            for field in (
                "python_callbacks",
                "hugging_face_calls",
                "qwen_vl_utils_calls",
                "pillow_calls",
                "torchvision_calls",
            )
        )
    ):
        raise BenchmarkProtocolError("native resource census observed fallback or extra work")
    return value


def _validate_instrumentation_free_worker(
    implementation: Mapping[str, Any],
    *,
    implementation_name: str,
    primary_adapter: str,
    work_units: int,
    protocol: Mapping[str, Any],
) -> None:
    expected_scope = {
        "clocks": ["perf_counter_ns", "process_time_ns"],
        "boundary": "adapter.run+required_output_normalization",
        "instrumentation": "none",
        "resource_census": "separate_after_all_timed_samples",
    }
    if implementation.get("timing_scope") != expected_scope:
        raise BenchmarkProtocolError("benchmark worker timing scope is not instrumentation-free")
    for field in ("warmups", "minimum_samples"):
        expected = protocol.get(field)
        observed = implementation.get(field)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, int)
            or observed < (0 if field == "warmups" else 1)
            or observed != expected
        ):
            raise BenchmarkProtocolError(f"benchmark worker {field} differs from protocol")
    minimum_seconds = _finite_number(implementation.get("minimum_seconds"))
    if minimum_seconds != _finite_number(protocol.get("minimum_seconds")):
        raise BenchmarkProtocolError("benchmark worker minimum seconds differs from protocol")
    samples = implementation.get("samples")
    if not isinstance(samples, list) or len(samples) != implementation["minimum_samples"]:
        raise BenchmarkProtocolError(
            "benchmark worker did not retain exactly the declared raw samples"
        )
    raw_sample_elapsed_ms = 0.0
    for sequence, sample in enumerate(samples):
        if not isinstance(sample, Mapping) or set(sample) != set(TIMING_SAMPLE_FIELDS):
            raise BenchmarkProtocolError("benchmark timing sample fields are invalid")
        if sample.get("sequence") != sequence:
            raise BenchmarkProtocolError("benchmark timing sample sequence is invalid")
        wall = _finite_number(sample.get("wall_ms"), positive=True)
        cpu = _finite_number(sample.get("cpu_ms"))
        throughput = _finite_number(sample.get("throughput_per_s"), positive=True)
        utilization = _finite_number(sample.get("core_utilization"))
        if not math.isclose(throughput, work_units / (wall / 1_000), rel_tol=1e-12):
            raise BenchmarkProtocolError("benchmark sample throughput differs from raw wall time")
        if not math.isclose(utilization, cpu / wall, rel_tol=1e-12, abs_tol=1e-15):
            raise BenchmarkProtocolError("benchmark sample utilization differs from raw clocks")
        allowed_cores = float(
            implementation["thread_settings"]["total_budget_model"]["total_budget"]
        ) + float(protocol["cpu_utilization_tolerance_cores"])
        if utilization > allowed_cores:
            raise BenchmarkProtocolError("benchmark sample exceeded its total CPU-thread budget")
        raw_sample_elapsed_ms += wall
    timing_floor = implementation.get("timing_floor")
    timing_floor_fields = {
        "required_seconds",
        "raw_sample_iteration_count",
        "raw_sample_elapsed_wall_ms",
        "supplemental_iteration_count",
        "supplemental_elapsed_wall_ms",
        "supplemental_elapsed_cpu_ms",
        "total_iteration_count",
        "total_elapsed_wall_ms",
    }
    if not isinstance(timing_floor, Mapping) or set(timing_floor) != timing_floor_fields:
        raise BenchmarkProtocolError("benchmark timing-floor evidence is invalid")
    supplemental_iterations = timing_floor.get("supplemental_iteration_count")
    total_iterations = timing_floor.get("total_iteration_count")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (supplemental_iterations, total_iterations)
    ):
        raise BenchmarkProtocolError("benchmark timing-floor iteration counters are invalid")
    supplemental_wall = _finite_number(timing_floor.get("supplemental_elapsed_wall_ms"))
    supplemental_cpu = _finite_number(timing_floor.get("supplemental_elapsed_cpu_ms"))
    total_wall = _finite_number(timing_floor.get("total_elapsed_wall_ms"), positive=True)
    if (
        _finite_number(timing_floor.get("required_seconds")) != minimum_seconds
        or timing_floor.get("raw_sample_iteration_count") != len(samples)
        or not math.isclose(
            _finite_number(timing_floor.get("raw_sample_elapsed_wall_ms")),
            raw_sample_elapsed_ms,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or total_iterations != len(samples) + supplemental_iterations
        or not math.isclose(
            total_wall, raw_sample_elapsed_ms + supplemental_wall, rel_tol=1e-12, abs_tol=1e-12
        )
    ):
        raise BenchmarkProtocolError("benchmark timing-floor counters do not reconcile")
    if supplemental_wall == 0.0:
        if supplemental_iterations != 0 or supplemental_cpu != 0.0:
            raise BenchmarkProtocolError("empty supplemental timing floor has nonzero counters")
    elif supplemental_iterations == 0 or supplemental_cpu / supplemental_wall > allowed_cores:
        raise BenchmarkProtocolError("supplemental timing floor exceeded its CPU-thread budget")
    if total_wall / 1_000 + 1e-12 < minimum_seconds:
        raise BenchmarkProtocolError("benchmark worker did not meet the elapsed-time floor")
    census = _validate_resource_census(
        implementation.get("resource_census"),
        primary_adapter=primary_adapter,
        implementation_name=implementation_name,
    )
    if implementation.get("summary") != summarize_samples(samples, census):
        raise BenchmarkProtocolError("benchmark worker summary differs from raw samples/census")


def _validate_output_conformance_witness(
    conformance: Any,
    *,
    require_resize_v2: bool,
    reference_output_signature: Mapping[str, Any] | None = None,
    candidate_output_signature: Mapping[str, Any] | None = None,
    source_dimensions: Sequence[tuple[int, int]] = (),
) -> None:
    if not isinstance(conformance, Mapping):
        raise BenchmarkProtocolError("benchmark output conformance metadata is missing")
    pre_witness = conformance.get("pre_measurement_witness")
    post_witness = conformance.get("post_measurement_witness")
    exact = {
        "comparison_id": "qwen-mm-exact-output-comparison-v1",
        "contract_id": "qwen-mm-compat-v1",
        "passed": True,
    }
    if require_resize_v2:
        if not isinstance(pre_witness, Mapping) or not isinstance(post_witness, Mapping):
            raise BenchmarkProtocolError("resize-v2 conformance witness is missing")
        if reference_output_signature is None or candidate_output_signature is None:
            raise BenchmarkProtocolError("resize-v2 paired output signatures are missing")
        try:
            validate_resize_v2_witness(
                pre_witness,
                reference_output_signature=reference_output_signature,
                candidate_output_signature=candidate_output_signature,
                source_dimensions=source_dimensions,
            )
            validate_resize_v2_witness(
                post_witness,
                reference_output_signature=reference_output_signature,
                candidate_output_signature=candidate_output_signature,
                source_dimensions=source_dimensions,
            )
        except ValueError as error:
            raise BenchmarkProtocolError(str(error)) from error
        if pre_witness != post_witness:
            raise BenchmarkProtocolError("resize-v2 pre/post conformance witnesses differ")
        return
    if pre_witness != exact or post_witness != exact:
        raise BenchmarkProtocolError("exact pre/post conformance witness is invalid")


def _validate_pair_output_signatures(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], payload: Any
) -> None:
    expected_keys = ["input_ids", "attention_mask", "mm_token_type_ids"]
    if payload.buffers:
        expected_keys.extend(("pixel_values", "image_grid_thw"))
    signatures: list[Mapping[str, Any]] = []
    for implementation_name, implementation in (
        ("reference", reference),
        ("candidate", candidate),
    ):
        signature = implementation.get("output_signature")
        if not isinstance(signature, Mapping) or list(signature) != expected_keys:
            raise BenchmarkProtocolError(
                f"benchmark {implementation_name} output signature key/order is invalid"
            )
        for name, descriptor in signature.items():
            if not isinstance(descriptor, Mapping) or set(descriptor) != {
                "dtype",
                "shape",
                "strides",
                "nbytes",
                "sha256",
            }:
                raise BenchmarkProtocolError("benchmark output signature descriptor is incomplete")
            dtype = descriptor.get("dtype")
            shape = descriptor.get("shape")
            strides = descriptor.get("strides")
            nbytes = descriptor.get("nbytes")
            digest = descriptor.get("sha256")
            expected_dtype = "float32" if name == "pixel_values" else "int64"
            if (
                dtype != expected_dtype
                or not isinstance(shape, list)
                or len(shape) != 2
                or any(
                    isinstance(item, bool) or not isinstance(item, int) or item < 0
                    for item in shape
                )
                or not isinstance(strides, list)
                or len(strides) != 2
                or any(
                    isinstance(item, bool) or not isinstance(item, int) or item <= 0
                    for item in strides
                )
                or strides != [shape[1] * np.dtype(dtype).itemsize, np.dtype(dtype).itemsize]
                or isinstance(nbytes, bool)
                or nbytes != math.prod(shape) * np.dtype(dtype).itemsize
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise BenchmarkProtocolError("benchmark output signature descriptor is invalid")
        signatures.append(signature)
    reference_signature, candidate_signature = signatures
    for name in expected_keys:
        if name == "pixel_values":
            reference_layout = {
                key: value for key, value in reference_signature[name].items() if key != "sha256"
            }
            candidate_layout = {
                key: value for key, value in candidate_signature[name].items() if key != "sha256"
            }
            if reference_layout != candidate_layout:
                raise BenchmarkProtocolError("paired pixel output layouts differ")
        elif reference_signature[name] != candidate_signature[name]:
            raise BenchmarkProtocolError(f"paired non-pixel output signature differs: {name}")


def _validate_case_bindings(
    result: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    *,
    portable: bool,
    require_output_comparison_policy: bool = True,
) -> dict[str, str]:
    workload = _authenticated_workload(result)
    by_id = {case["case_id"]: case for case in workload["cases"]}
    protocol = result.get("protocol")
    if not isinstance(protocol, Mapping):
        raise BenchmarkProtocolError("benchmark result lacks protocol metadata")

    list_fields = ("profiles", "cases", "build_labels", "thread_regimes")
    values: dict[str, list[str]] = {}
    for field in list_fields:
        raw = protocol.get(field)
        if (
            not isinstance(raw, list)
            or not raw
            or not all(isinstance(item, str) and item for item in raw)
            or len(raw) != len(set(raw))
        ):
            raise BenchmarkProtocolError(f"benchmark protocol {field} inventory is invalid")
        values[field] = raw
    thread_mapping = protocol.get("thread_budget_mapping")
    if (
        not isinstance(thread_mapping, Mapping)
        or set(thread_mapping) != set(values["thread_regimes"])
        or any(
            not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0
            for budget in thread_mapping.values()
        )
    ):
        raise BenchmarkProtocolError("benchmark protocol thread budget mapping is invalid")
    for regime, budget in thread_mapping.items():
        if regime == "one" and budget != 1:
            raise BenchmarkProtocolError("one-thread regime must map to one thread")
        if re.fullmatch(r"t[1-9][0-9]*", regime) and budget != int(regime[1:]):
            raise BenchmarkProtocolError("explicit tN regime differs from its thread budget")
    instrumentation_free = protocol.get("timing_protocol") is not None
    output_comparison_policy = protocol.get("output_comparison_policy")
    if require_output_comparison_policy and output_comparison_policy != OUTPUT_COMPARISON_POLICY:
        raise BenchmarkProtocolError("benchmark output comparison policy is invalid")
    if not require_output_comparison_policy and output_comparison_policy is not None:
        raise BenchmarkProtocolError("legacy benchmark result contains a post-v2 policy")
    if instrumentation_free:
        if (
            protocol.get("timing_protocol") != "instrumentation-free-v1"
            or protocol.get("timing_sample_fields") != list(TIMING_SAMPLE_FIELDS)
            or protocol.get("timing_floor_policy") != TIMING_FLOOR_POLICY
            or protocol.get("resource_census_position") != "after_all_timed_samples"
            or protocol.get("fixed_thread_environment") != list(THREAD_ENVIRONMENT_NAMES)
            or protocol.get("cpu_utilization_tolerance_cores") != 0.25
        ):
            raise BenchmarkProtocolError("benchmark timing protocol metadata is invalid")
        affinity_mapping = protocol.get("affinity_cpu_mapping")
        if not isinstance(affinity_mapping, Mapping) or set(affinity_mapping) != set(
            values["thread_regimes"]
        ):
            raise BenchmarkProtocolError("benchmark affinity mapping is incomplete")
        for regime, cpus in affinity_mapping.items():
            if cpus is not None and (
                not isinstance(cpus, list)
                or len(cpus) != thread_mapping[regime]
                or any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in cpus)
                or len(cpus) != len(set(cpus))
            ):
                raise BenchmarkProtocolError("benchmark affinity CPU mapping is invalid")
    else:
        affinity_mapping = {regime: None for regime in values["thread_regimes"]}
    unknown_cases = sorted(set(values["cases"]) - by_id.keys())
    if unknown_cases:
        raise BenchmarkProtocolError(
            f"benchmark protocol references unknown workload cases: {', '.join(unknown_cases)}"
        )
    expected_boundaries = {case_id: by_id[case_id]["boundary"] for case_id in values["cases"]}
    if protocol.get("case_boundaries") != expected_boundaries:
        raise BenchmarkProtocolError("benchmark protocol case boundaries were relabeled")

    repetitions = protocol.get("process_repetitions")
    seed = protocol.get("random_seed")
    if (
        not isinstance(repetitions, int)
        or isinstance(repetitions, bool)
        or repetitions <= 0
        or not isinstance(seed, int)
        or isinstance(seed, bool)
    ):
        raise BenchmarkProtocolError("benchmark protocol repetition or seed metadata is invalid")
    expected_pair_ids = [
        f"{profile}/{case_id}/{build}/{regime}/{repetition}"
        for profile in values["profiles"]
        for case_id in values["cases"]
        for build in values["build_labels"]
        for regime in values["thread_regimes"]
        for repetition in range(repetitions)
    ]
    if not all(isinstance(pair, Mapping) for pair in pairs):
        raise BenchmarkProtocolError("benchmark pairs must be objects")
    if [pair.get("pair_id") for pair in pairs] != expected_pair_ids:
        raise BenchmarkProtocolError(
            "benchmark pair inventory does not match the recorded protocol"
        )

    payloads: dict[str, Any] = {}
    for case_id in values["cases"]:
        try:
            payloads[case_id] = materialize_case(by_id[case_id])
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise BenchmarkProtocolError(
                f"authenticated benchmark case cannot be materialized: {case_id}"
            ) from error
    if instrumentation_free and protocol.get("case_work_units") != {
        case_id: payloads[case_id].work_units for case_id in values["cases"]
    }:
        raise BenchmarkProtocolError("benchmark case work units were relabeled")
    foreign_architecture = portable and result.get("architecture_family") != architecture_family()
    process_ids: list[int] = []
    worker_nonces: list[str] = []
    for pair in pairs:
        case_id = pair.get("case_id")
        if not isinstance(case_id, str) or case_id not in payloads:
            raise BenchmarkProtocolError("benchmark pair references an undeclared workload case")
        case = by_id[case_id]
        payload = payloads[case_id]
        if instrumentation_free and pair.get("work_units") != payload.work_units:
            raise BenchmarkProtocolError("benchmark pair work units were relabeled")
        for field, expected in (
            ("boundary", case["boundary"]),
            ("cache_mode", case["cache_mode"]),
            ("release_name", case.get("release_name")),
        ):
            if pair.get(field) != expected:
                raise BenchmarkProtocolError(f"benchmark pair {case_id} {field} was relabeled")
        profile = pair.get("profile_alias")
        build = pair.get("build_label")
        regime = pair.get("thread_regime")
        repetition = pair.get("repetition")
        expected_pair_id = f"{profile}/{case_id}/{build}/{regime}/{repetition}"
        if pair.get("pair_id") != expected_pair_id:
            raise BenchmarkProtocolError("benchmark pair coordinate labels do not match its ID")
        schedule_index = (
            (
                values["profiles"].index(profile) * len(values["cases"])
                + values["cases"].index(case_id)
            )
            * len(values["build_labels"])
            + values["build_labels"].index(build)
        ) * len(values["thread_regimes"]) + values["thread_regimes"].index(regime)
        order_seed = seed + schedule_index
        if pair.get("order_seed") != order_seed:
            raise BenchmarkProtocolError("benchmark pair order seed differs from its raw schedule")
        if pair.get("order") != randomized_orders(repetitions, seed=order_seed)[repetition]:
            raise BenchmarkProtocolError("benchmark pair AB/BA order differs from its raw seed")
        thread_budget = pair.get("thread_budget")
        if (
            not isinstance(thread_budget, int)
            or isinstance(thread_budget, bool)
            or thread_budget <= 0
            or regime not in thread_mapping
            or thread_budget != thread_mapping[regime]
        ):
            raise BenchmarkProtocolError(
                "benchmark pair thread budget differs from its protocol regime mapping"
            )
        implementations = pair.get("implementations")
        if not isinstance(implementations, Mapping) or set(implementations) != {
            "reference",
            "candidate",
        }:
            raise BenchmarkProtocolError(
                "benchmark pair must contain reference and candidate implementations"
            )
        for implementation_name, implementation in implementations.items():
            if implementation_name not in {"reference", "candidate"} or not isinstance(
                implementation, Mapping
            ):
                raise BenchmarkProtocolError("benchmark pair implementation metadata is invalid")
            expected_adapter = protocol.get(
                "reference_adapter" if implementation_name == "reference" else "candidate_adapter"
            )
            expected_fields = [
                ("implementation", implementation_name),
                ("adapter_spec", expected_adapter),
                ("oracle_adapter_spec", protocol.get("reference_adapter")),
                ("profile_alias", profile),
                ("build_label", build),
                ("case_id", case_id),
                ("boundary", case["boundary"]),
                ("cache_mode", case["cache_mode"]),
                ("release_name", case.get("release_name")),
                ("logical_input_fingerprint", payload.logical_input_fingerprint),
                ("messages_fingerprint", payload.messages_fingerprint),
            ]
            if instrumentation_free:
                expected_fields.append(("work_units", payload.work_units))
            if not (
                foreign_architecture and case.get("source", {}).get("kind") == "generated_encoded"
            ):
                expected_fields.append(("input_fingerprint", payload.input_fingerprint))
            if any(implementation.get(field) != expected for field, expected in expected_fields):
                raise BenchmarkProtocolError(
                    f"benchmark {implementation_name} worker metadata was relabeled"
                )
            thread_settings = implementation.get("thread_settings")
            if (
                not isinstance(thread_settings, Mapping)
                or thread_settings.get("budget") != thread_budget
            ):
                raise BenchmarkProtocolError("benchmark worker thread budget differs from its pair")
            budget_model = total_thread_budget_model(str(expected_adapter), thread_budget)
            expected_environment = (
                budget_model["environment"]
                if instrumentation_free
                else {name: str(thread_budget) for name in THREAD_ENVIRONMENT_NAMES}
            )
            if thread_settings.get("environment") != expected_environment:
                raise BenchmarkProtocolError(
                    "benchmark worker thread environment is not exactly budget-pinned"
                )
            if instrumentation_free and thread_settings.get("total_budget_model") != budget_model:
                raise BenchmarkProtocolError("benchmark worker total-thread model was relabeled")
            if instrumentation_free:
                expected_torch = (
                    {
                        "num_threads": budget_model["torch_thread_budget"],
                        "num_interop_threads": 1,
                    }
                    if protocol.get("reference_adapter") == "official"
                    else {}
                )
                if thread_settings.get("torch") != expected_torch:
                    raise BenchmarkProtocolError(
                        "benchmark worker Torch pools differ from the total-thread model"
                    )
            process_id = implementation.get("process_id")
            if isinstance(process_id, bool) or not isinstance(process_id, int) or process_id <= 0:
                raise BenchmarkProtocolError("benchmark worker process ID is invalid")
            process_ids.append(process_id)
            if instrumentation_free:
                nonce = implementation.get("worker_nonce")
                if (
                    not isinstance(nonce, str)
                    or len(nonce) != 64
                    or any(character not in "0123456789abcdef" for character in nonce)
                ):
                    raise BenchmarkProtocolError("benchmark worker nonce is invalid")
                worker_nonces.append(nonce)
                affinity = implementation.get("affinity")
                if not isinstance(affinity, Mapping):
                    raise BenchmarkProtocolError("benchmark worker affinity attestation is missing")
                requested = affinity_mapping[regime]
                if affinity.get("requested_cpus") != requested:
                    raise BenchmarkProtocolError("benchmark worker affinity request was relabeled")
                worker_system = implementation.get("environment", {}).get("system")
                if worker_system == "Linux":
                    expected_status = "attested" if requested is not None else "observed"
                    expected_mechanism = (
                        "taskset+sched_getaffinity"
                        if requested is not None
                        else "sched_getaffinity"
                    )
                    observed = affinity.get("observed_cpus")
                    if (
                        affinity.get("status") != expected_status
                        or affinity.get("mechanism") != expected_mechanism
                        or affinity.get("reason") is not None
                        or not isinstance(observed, list)
                        or not observed
                        or (requested is not None and observed != requested)
                    ):
                        raise BenchmarkProtocolError("Linux worker affinity was not child-attested")
                elif worker_system == "Darwin":
                    if (
                        affinity.get("status") != "unavailable"
                        or affinity.get("mechanism") is not None
                        or affinity.get("observed_cpus") is not None
                        or not isinstance(affinity.get("reason"), str)
                    ):
                        raise BenchmarkProtocolError(
                            "macOS worker affinity must be explicitly unavailable"
                        )
                _validate_instrumentation_free_worker(
                    implementation,
                    implementation_name=implementation_name,
                    primary_adapter=str(expected_adapter),
                    work_units=payload.work_units,
                    protocol=protocol,
                )
        if require_output_comparison_policy:
            reference_implementation = implementations["reference"]
            candidate_implementation = implementations["candidate"]
            _validate_pair_output_signatures(
                reference_implementation, candidate_implementation, payload
            )
            _validate_output_conformance_witness(
                reference_implementation.get("conformance"), require_resize_v2=False
            )
            candidate_is_distinct = bool(payload.buffers) and protocol.get(
                "candidate_adapter"
            ) != protocol.get("reference_adapter")
            _validate_output_conformance_witness(
                candidate_implementation.get("conformance"),
                require_resize_v2=candidate_is_distinct,
                reference_output_signature=reference_implementation.get("output_signature"),
                candidate_output_signature=candidate_implementation.get("output_signature"),
                source_dimensions=(
                    media_source_dimensions(payload) if candidate_is_distinct else ()
                ),
            )

    if len(process_ids) != len(set(process_ids)):
        raise BenchmarkProtocolError("benchmark workers did not use unique fresh process IDs")
    if instrumentation_free and len(worker_nonces) != len(set(worker_nonces)):
        raise BenchmarkProtocolError("benchmark workers did not use unique nonces")

    if list(summaries) != summarize_pairs(pairs, seed=seed):
        raise BenchmarkProtocolError("benchmark summaries do not match the authenticated pairs")
    return {case_id: payload.input_fingerprint for case_id, payload in payloads.items()}


def _validate_result_portable_body(
    result: Mapping[str, Any], *, require_output_comparison_policy: bool
) -> None:
    if result.get("mode") not in MODE_DEFAULTS:
        raise BenchmarkProtocolError("benchmark result has an invalid mode")
    family = result.get("architecture_family")
    if family not in {"arm64", "x86_64", "other"}:
        raise BenchmarkProtocolError("benchmark result has an invalid architecture family")
    pairs = result.get("pairs")
    summaries = result.get("summaries")
    if not isinstance(pairs, list) or not pairs or not isinstance(summaries, list):
        raise BenchmarkProtocolError("benchmark result pairs and summaries must be arrays")
    _validate_case_bindings(
        result,
        pairs,
        summaries,
        portable=True,
        require_output_comparison_policy=require_output_comparison_policy,
    )
    for pair in pairs:
        implementations = pair.get("implementations")
        if not isinstance(implementations, Mapping):
            raise BenchmarkProtocolError("benchmark pair lacks implementations")
        if set(implementations) != {"reference", "candidate"}:
            raise BenchmarkProtocolError("benchmark pair must contain reference and candidate")
        for field in (
            "input_fingerprint",
            "logical_input_fingerprint",
            "messages_fingerprint",
        ):
            fingerprints = {
                implementation.get(field) for implementation in implementations.values()
            }
            if any(
                not isinstance(fingerprint, str)
                or len(fingerprint) != 64
                or any(character not in "0123456789abcdef" for character in fingerprint)
                for fingerprint in fingerprints
            ):
                raise BenchmarkProtocolError(f"benchmark worker {field} is not SHA-256")
            if len(fingerprints) != 1:
                raise BenchmarkProtocolError("paired implementations received different inputs")
        for implementation in implementations.values():
            if implementation.get("environment", {}).get("architecture_family") != family:
                raise BenchmarkProtocolError("mixed architectures are not allowed in one result")
            conformance = implementation.get("conformance", {})
            if conformance.get("pre_measurement") != "pass":
                raise BenchmarkProtocolError("pre-measurement conformance did not pass")
            if conformance.get("post_measurement") != "pass":
                raise BenchmarkProtocolError("post-measurement conformance did not pass")
            if conformance.get("all_measured_iterations_stable") is not True:
                raise BenchmarkProtocolError("measured outputs were not stable")
    _validate_release_eligibility(result, verify_live_runtime=False)


def validate_result_portable(result: Mapping[str, Any]) -> None:
    """Validate current immutable v3 evidence without importing the measured wheel."""

    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise BenchmarkProtocolError("unsupported benchmark result schema version")
    if result.get("schema_id") != RESULT_SCHEMA_ID:
        raise BenchmarkProtocolError("benchmark result schema ID mismatch")
    _validate_result_portable_body(result, require_output_comparison_policy=True)


def validate_legacy_result_v2_portable(
    result: Mapping[str, Any], *, artifact_bytes: bytes, expected_artifact_sha256: str
) -> None:
    """Validate explicitly authenticated historical benchmark-result-v2 evidence.

    The caller must supply a trusted canonical artifact digest.  This separate
    entry point cannot be reached by relabeling a v3 result through the normal
    validator, and it rejects the post-v2 output-comparison policy.
    """

    if (
        not isinstance(expected_artifact_sha256, str)
        or len(expected_artifact_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_artifact_sha256)
    ):
        raise BenchmarkProtocolError("legacy benchmark authentication digest is invalid")
    if not isinstance(artifact_bytes, bytes):
        raise BenchmarkProtocolError("legacy benchmark raw artifact bytes are required")
    if result.get("schema_version") != 2 or result.get("schema_id") != (
        "qwen-mm-benchmark-result-v2"
    ):
        raise BenchmarkProtocolError("not a historical benchmark result v2")
    observed = hashlib.sha256(artifact_bytes).hexdigest()
    if observed != expected_artifact_sha256:
        raise BenchmarkProtocolError("legacy benchmark result authentication failed")
    try:
        parsed = json.loads(artifact_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkProtocolError("legacy benchmark artifact is not valid JSON") from error
    if parsed != result:
        raise BenchmarkProtocolError("legacy benchmark mapping differs from authenticated bytes")
    _validate_result_portable_body(result, require_output_comparison_policy=False)


def validate_result_authenticated_portable(
    result: Mapping[str, Any],
    *,
    expected_runtime_identity: Mapping[str, Any],
    phase_c_report_override: Path | None = None,
    phase_c_assets_root_override: Path | None = None,
) -> None:
    """Validate foreign-architecture evidence against an authenticated runtime.

    The adapter wrapper is re-hashed from the current source checkout, while the
    native runtime is compared with the wheel-authenticated identity supplied by
    the containing host evidence. This deliberately never imports or compares
    the local machine's native extension with the foreign measured runtime.
    """

    validate_result_portable(result)
    protocol = result.get("protocol")
    if not isinstance(protocol, Mapping):
        raise BenchmarkProtocolError("benchmark result lacks protocol metadata")
    adapter = protocol.get("candidate_adapter")
    recorded = protocol.get("candidate_identity")
    if not isinstance(adapter, str) or not isinstance(recorded, Mapping):
        raise BenchmarkProtocolError("benchmark result lacks candidate artifact identity")
    current = _current_portable_candidate_identity(adapter)
    if _portable_candidate_identity(current) != _portable_candidate_identity(recorded):
        raise BenchmarkProtocolError(
            "benchmark candidate adapter artifact changed after foreign-host measurement"
        )
    expected_runtime = _validate_runtime_identity(expected_runtime_identity)
    recorded_runtime = _validate_runtime_identity(recorded.get("runtime_identity"))
    if dict(recorded_runtime) != dict(expected_runtime):
        raise BenchmarkProtocolError(
            "benchmark candidate runtime differs from authenticated host evidence"
        )
    phase_c = result.get("release_eligibility", {}).get("phase_c", {})
    if isinstance(phase_c, Mapping) and phase_c.get("status") == "pass":
        evidence = phase_c.get("evidence")
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("candidate_runtime_identity") != expected_runtime
        ):
            raise BenchmarkProtocolError(
                "benchmark Phase C runtime differs from authenticated host evidence"
            )
        report_path = phase_c.get("report_path")
        if not isinstance(report_path, str) or not report_path:
            raise BenchmarkProtocolError("portable benchmark Phase C report path is missing")
        root = repository_root().resolve()
        if phase_c_report_override is None:
            if Path(report_path).is_absolute() or ".." in Path(report_path).parts:
                raise BenchmarkProtocolError(
                    "portable benchmark Phase C report path must be safe and repository-relative"
                )
            durable_report = (root / report_path).resolve()
        else:
            durable_report = phase_c_report_override.resolve()
        try:
            report_sha256 = _sha256_path(durable_report)
            if phase_c_report_override is None:
                durable_report.relative_to(root)
        except (OSError, ValueError) as error:
            raise BenchmarkProtocolError(
                "portable benchmark Phase C report is unavailable"
            ) from error
        if phase_c.get("report_sha256") != report_sha256:
            raise BenchmarkProtocolError(
                "portable benchmark Phase C report hash differs from durable evidence"
            )
        if phase_c_report_override is not None:
            if phase_c_assets_root_override is None:
                raise BenchmarkProtocolError(
                    "archived Phase C validation requires its authenticated assets root"
                )
            try:
                report = _load_json(durable_report)
                authenticated = _validate_phase_c_report(
                    report,
                    root=root,
                    assets_root=phase_c_assets_root_override.resolve(),
                    candidate_identity={
                        "resolved": True,
                        "runtime_identity": dict(expected_runtime),
                    },
                )
            except (json.JSONDecodeError, OSError, PhaseCGateError) as error:
                raise BenchmarkProtocolError(
                    "archived Phase C report failed authenticated validation"
                ) from error
            if authenticated != evidence:
                raise BenchmarkProtocolError(
                    "archived Phase C evidence differs from the benchmark gate witness"
                )
        fingerprint = hashlib.sha256(
            _canonical_json({"report_sha256": report_sha256, "evidence": evidence})
        ).hexdigest()
        if phase_c.get("gate_fingerprint") != fingerprint:
            raise BenchmarkProtocolError("portable benchmark Phase C gate fingerprint is stale")


def validate_result(
    result: Mapping[str, Any], *, phase_c_report_override: Path | None = None
) -> None:
    """Validate result evidence and re-authenticate the current local candidate runtime."""

    validate_result_portable(result)
    _validate_case_bindings(
        result,
        result["pairs"],
        result["summaries"],
        portable=False,
    )
    _validate_release_eligibility(
        result,
        verify_live_runtime=True,
        phase_c_report_override=phase_c_report_override,
    )


def run_benchmark(
    *,
    workload_path: Path,
    mode: str,
    reference_adapter: str,
    candidate_adapter: str,
    profiles: Sequence[str],
    case_ids: Sequence[str] = (),
    seed: int = 20260731,
    process_repetitions: int | None = None,
    warmups: int | None = None,
    minimum_samples: int | None = None,
    minimum_seconds: float | None = None,
    thread_regimes: Sequence[str] = (),
    build_labels: Sequence[str] = (),
    production_thread_budget: int | None = None,
    affinity_cpus: Sequence[int] | None = None,
    phase_c_report_path: Path | None = None,
    phase_c_assets_root: Path | None = None,
    phase_c_publish_report_path: Path | None = None,
) -> dict[str, Any]:
    if mode not in MODE_DEFAULTS:
        raise BenchmarkProtocolError(f"unknown benchmark mode: {mode}")
    defaults = MODE_DEFAULTS[mode]
    workload_path = _resolve_repo_path(workload_path)
    workload = load_workload(workload_path)
    model_registry = _load_json(repository_root() / "reference" / "models.json")
    unknown_profiles = sorted(set(profiles) - model_registry.keys())
    if unknown_profiles:
        raise BenchmarkProtocolError(f"unknown benchmark profiles: {', '.join(unknown_profiles)}")
    selected = select_cases(
        workload,
        case_ids=case_ids,
        tag=None if case_ids else str(defaults["tag"]),
    )
    candidate_identity = candidate_artifact_identity(candidate_adapter)
    repetitions = int(
        defaults["process_repetitions"] if process_repetitions is None else process_repetitions
    )
    effective_warmups = int(defaults["warmups"] if warmups is None else warmups)
    effective_samples = int(
        defaults["minimum_samples"] if minimum_samples is None else minimum_samples
    )
    effective_seconds = float(
        defaults["minimum_seconds"] if minimum_seconds is None else minimum_seconds
    )
    effective_regimes = list(thread_regimes or defaults["thread_regimes"])
    effective_builds = list(build_labels or defaults["build_labels"])
    if (
        not profiles
        or isinstance(effective_warmups, bool)
        or effective_warmups < 0
        or isinstance(effective_samples, bool)
        or effective_samples <= 0
        or not math.isfinite(effective_seconds)
        or effective_seconds < 0
        or repetitions <= 0
        or (
            production_thread_budget is not None
            and (
                isinstance(production_thread_budget, bool)
                or not isinstance(production_thread_budget, int)
                or production_thread_budget <= 0
            )
        )
        or len(effective_regimes) != len(set(effective_regimes))
        or len(effective_builds) != len(set(effective_builds))
    ):
        raise BenchmarkProtocolError("invalid benchmark protocol values")
    affinity_pool = None if affinity_cpus is None else list(affinity_cpus)
    if affinity_pool is not None and (
        not affinity_pool
        or any(
            isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in affinity_pool
        )
        or len(affinity_pool) != len(set(affinity_pool))
    ):
        raise BenchmarkProtocolError("affinity CPUs must be unique non-negative integers")
    thread_budget_mapping = {
        regime: _thread_budget(regime, production_thread_budget=production_thread_budget)
        for regime in effective_regimes
    }
    if affinity_pool is not None and max(thread_budget_mapping.values()) > len(affinity_pool):
        raise BenchmarkProtocolError("affinity CPU pool is smaller than a requested thread regime")
    affinity_mapping = {
        regime: (
            None
            if affinity_pool is None
            else sorted(affinity_pool)[: thread_budget_mapping[regime]]
        )
        for regime in effective_regimes
    }

    pairs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="qwen-mm-benchmark-v2-") as temporary:
        temporary_directory = Path(temporary)
        schedule_index = 0
        for profile_alias in profiles:
            for case in selected:
                for build_label in effective_builds:
                    for regime in effective_regimes:
                        thread_budget = thread_budget_mapping[regime]
                        orders = randomized_orders(repetitions, seed=seed + schedule_index)
                        schedule_index += 1
                        for repetition, order in enumerate(orders):
                            implementations: dict[str, Any] = {}
                            for implementation in order:
                                adapter_spec = (
                                    reference_adapter
                                    if implementation == "reference"
                                    else candidate_adapter
                                )
                                config = {
                                    "implementation": implementation,
                                    "adapter_spec": adapter_spec,
                                    "oracle_adapter_spec": reference_adapter,
                                    "profile_alias": profile_alias,
                                    "build_label": build_label,
                                    "thread_budget": thread_budget,
                                    "thread_regime": regime,
                                    "repetition": repetition,
                                    "warmups": effective_warmups,
                                    "minimum_samples": effective_samples,
                                    "minimum_seconds": effective_seconds,
                                    "affinity_cpus": affinity_mapping[regime],
                                    "worker_nonce": secrets.token_hex(32),
                                }
                                implementations[implementation] = _run_subprocess_worker(
                                    config=config,
                                    workload_path=workload_path,
                                    case_id=case["case_id"],
                                    temporary_directory=temporary_directory,
                                )
                            for field in (
                                "input_fingerprint",
                                "logical_input_fingerprint",
                                "messages_fingerprint",
                            ):
                                fingerprints = {
                                    result[field] for result in implementations.values()
                                }
                                if len(fingerprints) != 1:
                                    raise BenchmarkProtocolError(
                                        f"{case['case_id']}: paired subprocess input mismatch"
                                    )
                            reference_p50 = implementations["reference"]["summary"]["wall_ms"][
                                "p50"
                            ]
                            candidate_p50 = implementations["candidate"]["summary"]["wall_ms"][
                                "p50"
                            ]
                            pairs.append(
                                {
                                    "pair_id": (
                                        f"{profile_alias}/{case['case_id']}/{build_label}/"
                                        f"{regime}/{repetition}"
                                    ),
                                    "profile_alias": profile_alias,
                                    "case_id": case["case_id"],
                                    "release_name": case.get("release_name"),
                                    "boundary": case["boundary"],
                                    "cache_mode": case["cache_mode"],
                                    "work_units": implementations["reference"]["work_units"],
                                    "build_label": build_label,
                                    "thread_regime": regime,
                                    "thread_budget": thread_budget,
                                    "repetition": repetition,
                                    "order": order,
                                    "order_seed": seed + schedule_index - 1,
                                    "implementations": implementations,
                                    "paired_speedup_p50": reference_p50 / candidate_p50,
                                }
                            )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "schema_id": RESULT_SCHEMA_ID,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "architecture_family": architecture_family(),
        "environment": environment_metadata(),
        "workload": workload_provenance(workload_path),
        "protocol": {
            "reference_adapter": reference_adapter,
            "candidate_adapter": candidate_adapter,
            "profiles": list(profiles),
            "profile_models": {profile: model_registry[profile] for profile in profiles},
            "cases": [case["case_id"] for case in selected],
            "case_boundaries": {case["case_id"]: case["boundary"] for case in selected},
            "case_work_units": {
                case["case_id"]: materialize_case(case).work_units for case in selected
            },
            "candidate_identity": candidate_identity,
            "process_repetitions": repetitions,
            "random_seed": seed,
            "order": "randomized AB/BA per process repetition",
            "warmups": effective_warmups,
            "minimum_samples": effective_samples,
            "minimum_seconds": effective_seconds,
            "thread_regimes": effective_regimes,
            "thread_budget_mapping": {
                regime: thread_budget_mapping[regime] for regime in effective_regimes
            },
            "production_thread_budget": production_thread_budget,
            "affinity_cpu_mapping": affinity_mapping,
            "build_labels": effective_builds,
            "timing_protocol": "instrumentation-free-v1",
            "output_comparison_policy": OUTPUT_COMPARISON_POLICY,
            "timing_sample_fields": list(TIMING_SAMPLE_FIELDS),
            "timing_floor_policy": TIMING_FLOOR_POLICY,
            "resource_census_position": "after_all_timed_samples",
            "fixed_thread_environment": list(THREAD_ENVIRONMENT_NAMES),
            "cpu_utilization_tolerance_cores": 0.25,
            "p99_minimum_samples": 100,
            "threshold_enforcement": "none; D4 owns release performance gates",
            "self_test_only": "synthetic" in {reference_adapter, candidate_adapter},
        },
        "pairs": pairs,
        "summaries": summarize_pairs(pairs, seed=seed),
    }
    result["release_eligibility"] = evaluate_image_release_gate(
        mode=mode,
        selected_cases=selected,
        candidate_identity=candidate_identity,
        phase_c_report_path=phase_c_report_path,
        phase_c_assets_root=phase_c_assets_root,
    )
    if phase_c_publish_report_path is not None:
        publish_path = Path(phase_c_publish_report_path)
        if publish_path.is_absolute() or ".." in publish_path.parts or not publish_path.parts:
            raise BenchmarkProtocolError(
                "future Phase C publication path must be safe and repository-relative"
            )
        if result["release_eligibility"]["phase_c"]["status"] == "pass":
            result["release_eligibility"]["phase_c"]["report_path"] = publish_path.as_posix()
    validate_result(result, phase_c_report_override=phase_c_report_path)
    return result


def render_report(result: Mapping[str, Any], *, phase_c_report_override: Path | None = None) -> str:
    validate_result(result, phase_c_report_override=phase_c_report_override)
    eligibility = result["release_eligibility"]
    phase_c = eligibility["phase_c"]
    reason_text = ", ".join(phase_c["reason_codes"]) or "current passing correctness gate"
    lines = [
        "# qwen-mm paired benchmark v2",
        "",
        f"- Created: `{result['created_at']}`",
        f"- Mode: `{result['mode']}`",
        f"- Architecture family: `{result['architecture_family']}`",
        f"- Process repetitions: `{result['protocol']['process_repetitions']}`",
        f"- Threshold enforcement: {result['protocol']['threshold_enforcement']}",
        "- Performance status: `DIAGNOSTIC ONLY`",
        "- Releasable: `false`",
        f"- Phase C correctness prerequisite: `{phase_c['status']}` ({reason_text})",
        "",
        "**DIAGNOSTIC ONLY:** these measurements are not a release performance claim. "
        "D4 owns performance certification.",
        "",
        "Each measured workload passed paired pre/post output checks and measured-output "
        "stability. These checks do not replace the complete Phase C conformance gate.",
        "",
        "| Profile | Case | Build | Threads | Reference p50 | Candidate p50 | Speedup | 95% CI |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in result["summaries"]:
        interval = summary["speedup_bootstrap_95_ci"]
        lines.append(
            "| {profile} | {case} | {build} | {threads} | {reference:.3f} ms | "
            "{candidate:.3f} ms | {speedup:.3f}x | {lower:.3f}–{upper:.3f}x |".format(
                profile=summary["profile_alias"],
                case=summary["case_id"],
                build=summary["build_label"],
                threads=summary["thread_budget"],
                reference=summary["reference"]["p50_ms"],
                candidate=summary["candidate"]["p50_ms"],
                speedup=summary["speedup_p50"],
                lower=interval["lower"],
                upper=interval["upper"],
            )
        )
    lines.extend(
        (
            "",
            "p99 is reported only when an implementation has at least 100 raw samples. "
            "ARM and x86 measurements are intentionally stored in separate result files.",
            "",
        )
    )
    return "\n".join(lines)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_cpu_csv(value: str) -> list[int] | None:
    if not value.strip():
        return None
    try:
        return [int(item) for item in _parse_csv(value)]
    except ValueError as error:
        raise BenchmarkProtocolError("affinity CPUs must be comma-separated integers") from error


def _worker_command(args: argparse.Namespace) -> None:
    config = _load_json(args.config)
    workload = load_workload(args.workload)
    cases = select_cases(workload, case_ids=[args.case])
    result = run_worker(config, cases[0])
    _write_json(args.output, result)
    print(f"wrote worker result {args.output}")


def _run_command(args: argparse.Namespace) -> None:
    result = run_benchmark(
        workload_path=args.workload,
        mode=args.mode,
        reference_adapter=args.reference_adapter,
        candidate_adapter=args.candidate_adapter,
        profiles=_parse_csv(args.profiles),
        case_ids=_parse_csv(args.cases),
        seed=args.seed,
        process_repetitions=args.process_repetitions,
        warmups=args.warmups,
        minimum_samples=args.minimum_samples,
        minimum_seconds=args.minimum_seconds,
        thread_regimes=_parse_csv(args.thread_regimes),
        build_labels=_parse_csv(args.build_labels),
        production_thread_budget=args.production_thread_budget,
        affinity_cpus=_parse_cpu_csv(args.affinity_cpus),
        phase_c_report_path=args.phase_c_report,
        phase_c_assets_root=args.phase_c_assets_root,
        phase_c_publish_report_path=args.phase_c_publish_report,
    )
    _write_json(args.output, result)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            render_report(result, phase_c_report_override=args.phase_c_report),
            encoding="utf-8",
        )
    print(f"wrote paired benchmark result {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired qwen-mm benchmark result protocol v3.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run paired fresh-process measurements")
    run_parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    run_parser.add_argument("--mode", choices=sorted(MODE_DEFAULTS), default="smoke")
    run_parser.add_argument("--reference-adapter", default="official")
    run_parser.add_argument("--candidate-adapter", required=True)
    run_parser.add_argument("--profiles", default="qwen3-vl-8b,qwen3.5-9b")
    run_parser.add_argument("--cases", default="")
    run_parser.add_argument("--seed", type=int, default=20260731)
    run_parser.add_argument("--process-repetitions", type=int)
    run_parser.add_argument("--warmups", type=int)
    run_parser.add_argument("--minimum-samples", type=int)
    run_parser.add_argument("--minimum-seconds", type=float)
    run_parser.add_argument("--thread-regimes", default="")
    run_parser.add_argument("--build-labels", default="")
    run_parser.add_argument(
        "--affinity-cpus",
        default="",
        help="ordered CPU pool; each tN worker is taskset-pinned to the first N CPUs on Linux",
    )
    run_parser.add_argument(
        "--production-thread-budget",
        type=int,
        help="explicit equal production budget for cross-host evidence",
    )
    run_parser.add_argument(
        "--phase-c-report",
        type=Path,
        default=DEFAULT_PHASE_C_REPORT,
        help="current passing Phase C correctness report; never a performance certification",
    )
    run_parser.add_argument(
        "--phase-c-assets-root",
        type=Path,
        default=DEFAULT_PHASE_C_ASSETS_ROOT,
        help="authenticated profile assets used to validate the selected Phase C report",
    )
    run_parser.add_argument(
        "--phase-c-publish-report",
        type=Path,
        help="future safe repository-relative report path recorded in returned evidence",
    )
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--report", type=Path)
    run_parser.set_defaults(func=_run_command)

    worker_parser = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--config", type=Path, required=True)
    worker_parser.add_argument("--workload", type=Path, required=True)
    worker_parser.add_argument("--case", required=True)
    worker_parser.add_argument("--output", type=Path, required=True)
    worker_parser.set_defaults(func=_worker_command)

    validate_parser = subparsers.add_parser("validate", help="validate a result file")
    validate_parser.add_argument("result", type=Path)
    validate_parser.add_argument(
        "--phase-c-source-report",
        type=Path,
        help="capture-time source report for validating a future publication path",
    )
    validate_parser.set_defaults(
        func=lambda args: validate_result(
            _load_json(args.result), phase_c_report_override=args.phase_c_source_report
        )
    )

    portable_parser = subparsers.add_parser(
        "validate-portable",
        help="validate authenticated foreign-architecture result evidence",
    )
    portable_parser.add_argument("result", type=Path)
    portable_parser.add_argument("--runtime-authentication", type=Path, required=True)

    def portable_command(args: argparse.Namespace) -> None:
        authentication = _load_json(args.runtime_authentication)
        expected = authentication.get("benchmark_runtime_identity")
        if not isinstance(expected, Mapping):
            raise BenchmarkProtocolError("runtime authentication lacks benchmark_runtime_identity")
        validate_result_authenticated_portable(
            _load_json(args.result), expected_runtime_identity=expected
        )

    portable_parser.set_defaults(func=portable_command)

    report_parser = subparsers.add_parser("report", help="render a Markdown result report")
    report_parser.add_argument("result", type=Path)
    report_parser.add_argument("--output", type=Path, required=True)

    def report_command(args: argparse.Namespace) -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_report(_load_json(args.result)), encoding="utf-8")

    report_parser.set_defaults(func=report_command)
    args = parser.parse_args()
    try:
        args.func(args)
    except BenchmarkProtocolError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
