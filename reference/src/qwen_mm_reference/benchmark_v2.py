from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import random
import re
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
    BenchmarkProtocolError,
    architecture_family,
    environment_metadata,
    load_workload,
    materialize_case,
    percentile,
    physical_cpu_count,
    run_worker,
    select_cases,
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

DEFAULT_WORKLOAD = Path("benchmarks/workloads-v2.json")
DEFAULT_PHASE_C_REPORT = Path("reference/phase-c/v1/report.json")
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
) -> dict[str, Any]:
    if _phase_c_has_performance_claim(report):
        raise PhaseCGateError(
            "invalid",
            "performance_claim",
            "Phase C correctness evidence must not contain performance claims",
        )
    try:
        validate_phase_c_report(report, assets_root=assets_root)
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
        detail = str(error)
        stale = any(token in detail.lower() for token in ("drifted", "mismatch", "stale"))
        raise PhaseCGateError(
            "stale" if stale else "invalid",
            "report_currentness" if stale else "report_validation",
            detail,
        ) from error

    scope = report["scope"]
    declared = scope["declared_case_ids"]
    canonical_ids = expected_candidate_case_ids(root)
    if declared != canonical_ids:
        raise PhaseCGateError(
            "stale",
            "case_inventory_changed",
            "Phase C report does not match the canonical candidate case inventory",
        )
    expected_input_paths = [path.as_posix() for path in phase_c_source_inputs()]
    reported_inputs = report.get("inputs")
    if (
        not isinstance(reported_inputs, list)
        or [item.get("path") for item in reported_inputs] != expected_input_paths
    ):
        raise PhaseCGateError(
            "stale",
            "input_inventory_changed",
            "Phase C report does not match the canonical authenticated input inventory",
        )

    candidate = report.get("candidate")
    expected_runtime = (
        None if not isinstance(candidate, Mapping) else candidate.get("runtime_identity")
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
        "schema_id": PHASE_C_REPORT_SCHEMA_ID,
        "schema_version": PHASE_C_REPORT_SCHEMA_VERSION,
        "contract_id": "qwen-mm-compat-v1",
        "profiles": sorted(PHASE_C_PROFILES),
        "declared_case_ids": declared,
        "candidate_runtime_identity": dict(observed_runtime),
        "inputs": reported_inputs,
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


def _thread_budget(regime: str) -> int:
    if regime == "one":
        return 1
    if regime == "production":
        return physical_cpu_count()
    try:
        value = int(regime)
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
    completed = subprocess.run(
        command,
        cwd=repository_root(),
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
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


def _validate_release_eligibility(result: Mapping[str, Any]) -> None:
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

    if phase_c.get("status") == "pass":
        report_path = phase_c.get("report_path")
        assets_root = phase_c.get("assets_root")
        if not isinstance(report_path, str) or not isinstance(assets_root, str):
            raise BenchmarkProtocolError("passing Phase C gate lacks reproducibility metadata")
        selected_cases = [{"boundary": boundary} for boundary in boundaries.values()]
        recomputed = evaluate_image_release_gate(
            mode=str(result.get("mode")),
            selected_cases=selected_cases,
            candidate_identity=current_candidate_identity,
            phase_c_report_path=Path(report_path),
            phase_c_assets_root=Path(assets_root),
        )
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


def _validate_case_bindings(
    result: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
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

    fingerprints: dict[str, str] = {}
    for case_id in values["cases"]:
        try:
            fingerprints[case_id] = materialize_case(by_id[case_id]).input_fingerprint
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise BenchmarkProtocolError(
                f"authenticated benchmark case cannot be materialized: {case_id}"
            ) from error
    for pair in pairs:
        case_id = pair.get("case_id")
        if not isinstance(case_id, str) or case_id not in fingerprints:
            raise BenchmarkProtocolError("benchmark pair references an undeclared workload case")
        case = by_id[case_id]
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
        thread_budget = pair.get("thread_budget")
        if (
            not isinstance(thread_budget, int)
            or isinstance(thread_budget, bool)
            or thread_budget <= 0
        ):
            raise BenchmarkProtocolError("benchmark pair thread budget is invalid")
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
            expected_fields = (
                ("implementation", implementation_name),
                ("adapter_spec", expected_adapter),
                ("oracle_adapter_spec", protocol.get("reference_adapter")),
                ("profile_alias", profile),
                ("build_label", build),
                ("case_id", case_id),
                ("boundary", case["boundary"]),
                ("cache_mode", case["cache_mode"]),
                ("release_name", case.get("release_name")),
                ("input_fingerprint", fingerprints[case_id]),
            )
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

    if list(summaries) != summarize_pairs(pairs, seed=seed):
        raise BenchmarkProtocolError("benchmark summaries do not match the authenticated pairs")
    return fingerprints


def validate_result(result: Mapping[str, Any]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise BenchmarkProtocolError("unsupported benchmark result schema version")
    if result.get("schema_id") != RESULT_SCHEMA_ID:
        raise BenchmarkProtocolError("benchmark result schema ID mismatch")
    if result.get("mode") not in MODE_DEFAULTS:
        raise BenchmarkProtocolError("benchmark result has an invalid mode")
    family = result.get("architecture_family")
    if family not in {"arm64", "x86_64", "other"}:
        raise BenchmarkProtocolError("benchmark result has an invalid architecture family")
    pairs = result.get("pairs")
    summaries = result.get("summaries")
    if not isinstance(pairs, list) or not pairs or not isinstance(summaries, list):
        raise BenchmarkProtocolError("benchmark result pairs and summaries must be arrays")
    _validate_case_bindings(result, pairs, summaries)
    for pair in pairs:
        implementations = pair.get("implementations")
        if not isinstance(implementations, Mapping):
            raise BenchmarkProtocolError("benchmark pair lacks implementations")
        if set(implementations) != {"reference", "candidate"}:
            raise BenchmarkProtocolError("benchmark pair must contain reference and candidate")
        fingerprints = {
            implementation["input_fingerprint"] for implementation in implementations.values()
        }
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
    _validate_release_eligibility(result)


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
    phase_c_report_path: Path | None = None,
    phase_c_assets_root: Path | None = None,
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
    repetitions = process_repetitions or int(defaults["process_repetitions"])
    effective_warmups = int(defaults["warmups"] if warmups is None else warmups)
    effective_samples = int(
        defaults["minimum_samples"] if minimum_samples is None else minimum_samples
    )
    effective_seconds = float(
        defaults["minimum_seconds"] if minimum_seconds is None else minimum_seconds
    )
    effective_regimes = list(thread_regimes or defaults["thread_regimes"])
    effective_builds = list(build_labels or defaults["build_labels"])
    if not profiles or effective_warmups < 0 or effective_samples <= 0 or effective_seconds < 0:
        raise BenchmarkProtocolError("invalid benchmark protocol values")

    pairs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="qwen-mm-benchmark-v2-") as temporary:
        temporary_directory = Path(temporary)
        schedule_index = 0
        for profile_alias in profiles:
            for case in selected:
                for build_label in effective_builds:
                    for regime in effective_regimes:
                        thread_budget = _thread_budget(regime)
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
                                }
                                implementations[implementation] = _run_subprocess_worker(
                                    config=config,
                                    workload_path=workload_path,
                                    case_id=case["case_id"],
                                    temporary_directory=temporary_directory,
                                )
                            fingerprints = {
                                result["input_fingerprint"] for result in implementations.values()
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
                                    "build_label": build_label,
                                    "thread_regime": regime,
                                    "thread_budget": thread_budget,
                                    "repetition": repetition,
                                    "order": order,
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
            "candidate_identity": candidate_identity,
            "process_repetitions": repetitions,
            "random_seed": seed,
            "order": "randomized AB/BA per process repetition",
            "warmups": effective_warmups,
            "minimum_samples": effective_samples,
            "minimum_seconds": effective_seconds,
            "thread_regimes": effective_regimes,
            "build_labels": effective_builds,
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
    validate_result(result)
    return result


def render_report(result: Mapping[str, Any]) -> str:
    validate_result(result)
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
        phase_c_report_path=args.phase_c_report,
        phase_c_assets_root=args.phase_c_assets_root,
    )
    _write_json(args.output, result)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(result), encoding="utf-8")
    print(f"wrote paired benchmark result {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired qwen-mm benchmark protocol v2.")
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
    validate_parser.set_defaults(func=lambda args: validate_result(_load_json(args.result)))

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
