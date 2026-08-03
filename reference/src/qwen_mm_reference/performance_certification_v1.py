from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

SCHEMA_ID = "qwen-mm-performance-certification-v1"
SCHEMA_VERSION = 1
BOOTSTRAP_RESAMPLES = 20_000
RANDOM_SEED = 20260731
CPU_UTILIZATION_TOLERANCE_CORES = 0.25
ARCHITECTURES = ("arm64", "x86_64")
BUILDS = ("shipping", "native")
PROFILES = ("qwen3-vl-8b", "qwen3.5-9b")
CASES = (
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
    "repeat24_cached",
    "repeat24_separated",
    "images_1",
    "images_4",
    "images_16",
    "images_32",
    "images_64",
)
IMAGE_CASES = frozenset(CASES) - {"text_short", "text_long"}
CACHED_CASE = "repeat24_cached"
FLOAT_ATOL = {
    case_id: (
        1e-6
        if case_id in {"text_short", "text_long", "aligned24", "minmax_boundaries", "rgb24"}
        else 0.007844
    )
    for case_id in CASES
}
WORK_UNITS = {case_id: 1 for case_id in CASES}
WORK_UNITS.update({"jpeg24_requests": 24, "minmax_boundaries": 12})
PRODUCTION_THREAD_BUDGET = 8
THREAD_BUDGETS = (1, 2, 4, 8)
IDENTITY_FIELDS = frozenset(
    {
        "source_revision",
        "source_sha256",
        "benchmark_schema_sha256",
        "workload_sha256",
        "profile_schema_sha256",
        "model_registry_sha256",
        "assets_sha256",
        "protocol_sha256",
    }
)
THREAD_ENVIRONMENT_NAMES = frozenset(
    {
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
    }
)
SCOPE_FIELDS = frozenset(
    {"request_index", "message_index", "content_item_index", "media_index", "input_index"}
)
BUFFER_SEMANTICS = {
    "input_ids": ("int64", "vector", False, "retained_output"),
    "attention_mask": ("int64", "vector", False, "retained_output"),
    "mm_token_type_ids": ("int64", "vector", False, "retained_output"),
    "image_grid_thw": ("int64", "vector", False, "retained_output"),
    "pixel_values": ("float32", "packed_patch", False, "retained_output"),
    "binding.owned_media": ("uint8", "HWC", True, "transient"),
    "decoded_rgb": ("uint8", "HWC", True, "transient"),
    "prepared_rgb": ("uint8", "HWC", True, "transient"),
    "resize.packed_source": ("uint8", "HWC", True, "transient"),
    "resize.horizontal.source_copy": ("uint8", "HWC", True, "transient"),
    "resize.noop.source_copy": ("uint8", "HWC", True, "transient"),
    "resize.horizontal.destination": ("uint8", "HWC", True, "transient"),
    "resize.vertical.destination": ("uint8", "HWC", True, "transient"),
    "resize.horizontal.weights_f64": ("float64", "vector", False, "transient"),
    "resize.horizontal.bounds": ("usize_pair", "vector", False, "transient"),
    "resize.horizontal.coefficients_i32": ("int32", "vector", False, "transient"),
    "resize.vertical.weights_f64": ("float64", "vector", False, "transient"),
    "resize.vertical.bounds": ("usize_pair", "vector", False, "transient"),
    "resize.vertical.coefficients_i32": ("int32", "vector", False, "transient"),
}
COPY_NAMES = frozenset({"binding.owned_media", "resize.packed_source", "resize.noop.source_copy"})


class PerformanceCertificationError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise PerformanceCertificationError(message)


def _object(
    value: Any,
    name: str,
    fields: set[str] | frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be an object")
    actual = set(value)
    if actual != set(fields):
        missing = sorted(set(fields) - actual)
        extra = sorted(actual - set(fields))
        _fail(f"{name} has a closed shape (missing={missing}, extra={extra})")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{name} must be a list")
    return value


def _string(value: Any, name: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        _fail(f"{name} must be a{' non-empty' if nonempty else ''} string")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        _fail(f"{name} must be a finite {'positive ' if positive else ''}number")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{name} must be a boolean")
    return value


def _sha256(value: Any, name: str) -> str:
    result = _string(value, name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        _fail(f"{name} must be a lowercase SHA-256 digest")
    return result


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PerformanceCertificationError("artifact is not canonical JSON data") from error


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        _fail("cannot summarize empty samples")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap(speedups: Sequence[float], *, seed: int) -> dict[str, float | int | str]:
    if not speedups:
        _fail("cannot bootstrap empty paired speedups")
    if all(value == speedups[0] for value in speedups):
        return {
            "method": "deterministic paired process-median bootstrap",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": seed,
            "lower": speedups[0],
            "upper": speedups[0],
        }
    generator = random.Random(seed)
    medians = [
        statistics.median(speedups[generator.randrange(len(speedups))] for _ in speedups)
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    return {
        "method": "deterministic paired process-median bootstrap",
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": seed,
        "lower": _percentile(medians, 0.025),
        "upper": _percentile(medians, 0.975),
    }


def _validate_identity(value: Any, name: str) -> dict[str, str]:
    identity = _object(value, name, IDENTITY_FIELDS)
    for field in IDENTITY_FIELDS - {"source_revision"}:
        _sha256(identity[field], f"{name}.{field}")
    revision = _string(identity["source_revision"], f"{name}.source_revision")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        _fail(f"{name}.source_revision must be an exact 40-character Git revision")
    return dict(identity)


def _validate_build(value: Any, architecture: str) -> Mapping[str, Any]:
    build = _object(
        value,
        f"{architecture}.build",
        {
            "label",
            "optimization",
            "wheel_sha256",
            "native_sha256",
            "build_command",
            "rustflags",
        },
    )
    label = _string(build["label"], f"{architecture}.build.label")
    if label not in BUILDS:
        _fail(f"{architecture}.build.label is unsupported")
    expected_optimization = "portable-release" if label == "shipping" else "target-cpu=native"
    if build["optimization"] != expected_optimization:
        _fail(f"{architecture}/{label} has the wrong optimization identity")
    _sha256(build["wheel_sha256"], f"{architecture}/{label}.wheel_sha256")
    _sha256(build["native_sha256"], f"{architecture}/{label}.native_sha256")
    _string(build["build_command"], f"{architecture}/{label}.build_command")
    rustflags = _list(build["rustflags"], f"{architecture}/{label}.rustflags")
    if not all(isinstance(item, str) and item for item in rustflags):
        _fail(f"{architecture}/{label}.rustflags must contain non-empty strings")
    if label == "shipping" and rustflags:
        _fail("shipping must use the repository's portable release flags")
    if label == "native" and rustflags != ["-C", "target-cpu=native"]:
        _fail("native must be an authenticated -C target-cpu=native build")
    return build


def _validate_host(value: Any, architecture: str) -> Mapping[str, Any]:
    host = _object(
        value,
        f"{architecture}.host",
        {
            "host_fingerprint",
            "baseline",
            "system",
            "machine",
            "cpu_model",
            "physical_cpu_count",
            "logical_cpu_count",
            "memory_bytes",
            "controlled",
            "provider",
            "instance_type",
            "allocation_id",
            "affinity_available",
            "affinity_source",
            "affinity_unavailable_reason",
            "allocated_cpu_count",
            "allocated_memory_bytes",
            "cgroup_sha256",
            "physical_core_topology_sha256",
            "physical_core_masks",
        },
    )
    _sha256(host["host_fingerprint"], f"{architecture}.host.host_fingerprint")
    expected_baseline = "current-m4" if architecture == "arm64" else "controlled-linux-x86"
    if host["baseline"] != expected_baseline:
        _fail(f"{architecture} has the wrong controlled-host baseline")
    if architecture == "arm64":
        if host["system"] != "Darwin" or host["machine"] != "arm64":
            _fail("ARM certification requires native macOS arm64")
        if "M4" not in _string(host["cpu_model"], "arm64.host.cpu_model"):
            _fail("ARM certification requires the current Apple M4 baseline")
    else:
        if host["system"] != "Linux" or host["machine"] != "x86_64":
            _fail("x86 certification requires native Linux x86_64")
        _string(host["cpu_model"], "x86_64.host.cpu_model")
    for field in (
        "physical_cpu_count",
        "logical_cpu_count",
        "memory_bytes",
        "allocated_cpu_count",
        "allocated_memory_bytes",
    ):
        _integer(host[field], f"{architecture}.host.{field}", minimum=1)
    if host["physical_cpu_count"] < PRODUCTION_THREAD_BUDGET:
        _fail(f"{architecture} host has fewer than eight physical cores")
    if host["allocated_cpu_count"] < PRODUCTION_THREAD_BUDGET:
        _fail(f"{architecture} allocation has fewer than eight CPUs")
    if host["controlled"] is not True:
        _fail(f"{architecture} host is not attested controlled")
    for field in ("provider", "instance_type", "allocation_id", "affinity_source"):
        _string(host[field], f"{architecture}.host.{field}")
    available = _boolean(host["affinity_available"], f"{architecture}.host.affinity_available")
    reason = host["affinity_unavailable_reason"]
    if reason is not None and not isinstance(reason, str):
        _fail(f"{architecture}.host.affinity_unavailable_reason must be string or null")
    if available and reason is not None:
        _fail(f"{architecture} available affinity cannot have an unavailable reason")
    if not available and (not isinstance(reason, str) or not reason):
        _fail(f"{architecture} unavailable affinity requires an explicit reason")
    if architecture == "x86_64" and not available:
        _fail("controlled Linux x86 certification requires fixed CPU affinity")
    _sha256(host["cgroup_sha256"], f"{architecture}.host.cgroup_sha256")
    _sha256(
        host["physical_core_topology_sha256"],
        f"{architecture}.host.physical_core_topology_sha256",
    )
    masks = _object(
        host["physical_core_masks"],
        f"{architecture}.host.physical_core_masks",
        {"t1", "t2", "t4", "t8"},
    )
    prior: list[int] = []
    for budget in THREAD_BUDGETS:
        raw_mask = masks[f"t{budget}"]
        if not available:
            if raw_mask is not None:
                _fail(f"{architecture} cannot claim a physical-core mask without affinity")
            continue
        mask = _list(raw_mask, f"{architecture}.host.physical_core_masks.t{budget}")
        if (
            len(mask) != budget
            or len(mask) != len(set(mask))
            or any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in mask)
        ):
            _fail(f"{architecture}.host.physical_core_masks.t{budget} is invalid")
        if prior and mask[: len(prior)] != prior:
            _fail(f"{architecture} physical-core masks must be nested and stable")
        prior = mask
    return host


def _validate_toolchain(value: Any, name: str) -> None:
    toolchain = _object(
        value,
        name,
        {"rustc", "cargo", "python", "maturin", "os_release", "environment_sha256"},
    )
    for field in ("rustc", "cargo", "python", "maturin", "os_release"):
        _string(toolchain[field], f"{name}.{field}")
    _sha256(toolchain["environment_sha256"], f"{name}.environment_sha256")


def _validate_protocol(value: Any, name: str) -> Mapping[str, Any]:
    protocol = _object(
        value,
        name,
        {
            "mode",
            "process_repetitions",
            "random_seed",
            "bootstrap_resamples",
            "order",
            "warmups",
            "minimum_samples",
            "minimum_seconds",
            "timing_instrumentation",
            "timing_floor_policy",
            "memory_pass",
            "profiles",
            "cases",
            "thread_budgets",
            "production_thread_budget",
            "cache_policy",
        },
    )
    expected = {
        "mode": "dedicated",
        "process_repetitions": 5,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "order": "randomized AB/BA per process repetition",
        "warmups": 3,
        "minimum_samples": 30,
        "minimum_seconds": 5.0,
        "timing_instrumentation": "none",
        "timing_floor_policy": (
            "minimum_samples one-operation latency samples; supplemental individually clocked "
            "exact-stability operations aggregate only to minimum_seconds and are excluded from "
            "latency distributions"
        ),
        "memory_pass": "separate from timing",
        "profiles": list(PROFILES),
        "cases": list(CASES),
        "thread_budgets": list(THREAD_BUDGETS),
        "production_thread_budget": PRODUCTION_THREAD_BUDGET,
        "cache_policy": "repeat24_cached unsupported/non-gating; uncached timing forbidden",
    }
    for field, expected_value in expected.items():
        if protocol[field] != expected_value:
            _fail(f"{name}.{field} does not match the frozen D4 protocol")
    if protocol["random_seed"] != RANDOM_SEED:
        _fail(f"{name}.random_seed does not match the frozen D4 seed")
    return protocol


def _validate_phase_c(
    value: Any,
    name: str,
    *,
    identity: Mapping[str, Any],
    build: Mapping[str, Any],
) -> None:
    phase_c = _object(
        value,
        name,
        {"before", "after"},
    )
    fields = {
        "status",
        "report_sha256",
        "source_sha256",
        "wheel_sha256",
        "native_sha256",
        "assets_sha256",
        "profiles",
        "failed_cases",
        "skipped_cases",
    }
    for position in ("before", "after"):
        report = _object(phase_c[position], f"{name}.{position}", fields)
        if report["status"] != "pass":
            _fail(f"{name}.{position} must pass complete Phase C conformance")
        _sha256(report["report_sha256"], f"{name}.{position}.report_sha256")
        for field, expected in (
            ("source_sha256", identity["source_sha256"]),
            ("wheel_sha256", build["wheel_sha256"]),
            ("native_sha256", build["native_sha256"]),
            ("assets_sha256", identity["assets_sha256"]),
        ):
            _sha256(report[field], f"{name}.{position}.{field}")
            if report[field] != expected:
                _fail(f"{name}.{position}.{field} is stale")
        if report["profiles"] != list(PROFILES):
            _fail(f"{name}.{position} does not cover both profiles")
        if report["failed_cases"] != 0 or report["skipped_cases"] != 0:
            _fail(f"{name}.{position} has failed or skipped conformance cases")
    if phase_c["before"]["report_sha256"] == phase_c["after"]["report_sha256"]:
        _fail(f"{name} reused one Phase C report instead of fresh pre/post runs")


def _validate_conformance(value: Any, name: str, *, case_id: str) -> Mapping[str, Any]:
    record = _object(
        value,
        name,
        {
            "passed",
            "witness_sha256",
            "input_sha256",
            "messages_sha256",
            "output_keys",
            "dtypes",
            "values_sha256",
            "float_atol",
            "tolerance_policy",
            "all_timed_outputs_match",
            "fallback_work",
            "cache_used",
        },
    )
    if record["passed"] is not True:
        _fail(f"{name} did not pass")
    for field in (
        "witness_sha256",
        "input_sha256",
        "messages_sha256",
        "values_sha256",
    ):
        _sha256(record[field], f"{name}.{field}")
    expected_keys = ["input_ids", "attention_mask", "mm_token_type_ids"]
    expected_dtypes = {key: "int64" for key in expected_keys}
    if case_id in IMAGE_CASES:
        expected_keys.extend(("pixel_values", "image_grid_thw"))
        expected_dtypes.update(pixel_values="float32", image_grid_thw="int64")
    if record["output_keys"] != expected_keys or record["dtypes"] != expected_dtypes:
        _fail(f"{name} has missing/reordered keys or non-contract dtypes")
    if _number(record["float_atol"], f"{name}.float_atol") != FLOAT_ATOL[case_id]:
        _fail(f"{name}.float_atol widens or changes the frozen workload tolerance")
    if record["tolerance_policy"] != (
        "exact integers; exact timed stability; pixel per-occurrence lossless=1e-6 "
        "otherwise workload float_atol"
    ):
        _fail(f"{name}.tolerance_policy is not the frozen comparison policy")
    if record["all_timed_outputs_match"] is not True:
        _fail(f"{name} has a timed output that changed from the conformance baseline")
    if record["fallback_work"] is not False:
        _fail(f"{name} observed fallback work")
    if record["cache_used"] is not False:
        _fail(f"{name} observed cache use")
    return record


def _validate_memory(
    value: Any,
    name: str,
    *,
    candidate: bool,
) -> Mapping[str, Any]:
    memory = _object(
        value,
        name,
        {
            "rss_before_bytes",
            "rss_after_bytes",
            "scoped_peak_rss_bytes",
            "retained_output_bytes",
            "external_transient_rss_bytes",
            "native_counters_available",
            "observation_duration_ns",
            "allocation_count",
            "allocated_bytes",
            "copy_count",
            "copied_bytes",
            "transient_live_bytes",
            "peak_transient_live_bytes",
            "buffer_census_complete",
            "dropped_events",
            "buffers",
            "copies",
        },
    )
    for field in (
        "rss_before_bytes",
        "rss_after_bytes",
        "scoped_peak_rss_bytes",
        "retained_output_bytes",
        "external_transient_rss_bytes",
    ):
        _integer(memory[field], f"{name}.{field}")
    if memory["scoped_peak_rss_bytes"] < max(memory["rss_before_bytes"], memory["rss_after_bytes"]):
        _fail(f"{name}.scoped_peak_rss_bytes is below a boundary RSS observation")
    computed_external = max(
        0,
        memory["scoped_peak_rss_bytes"]
        - memory["rss_before_bytes"]
        - memory["retained_output_bytes"],
    )
    if memory["external_transient_rss_bytes"] != computed_external:
        _fail(
            f"{name}.external_transient_rss_bytes must equal "
            "max(0, scoped peak - baseline RSS - retained output bytes)"
        )
    buffers = _list(memory["buffers"], f"{name}.buffers")
    copies = _list(memory["copies"], f"{name}.copies")
    counter_fields = (
        "observation_duration_ns",
        "allocation_count",
        "allocated_bytes",
        "copy_count",
        "copied_bytes",
        "transient_live_bytes",
        "peak_transient_live_bytes",
        "dropped_events",
    )
    if not candidate:
        if memory["native_counters_available"] is not False:
            _fail(f"{name} must not fabricate unavailable official native counters")
        if any(memory[field] is not None for field in counter_fields):
            _fail(f"{name} official native counters must be explicit nulls")
        if memory["buffer_census_complete"] is not False or buffers or copies:
            _fail(f"{name} official native allocation census must be explicitly unavailable")
        return memory
    if memory["native_counters_available"] is not True:
        _fail(f"{name} candidate native counters are required")
    for field in counter_fields:
        _integer(memory[field], f"{name}.{field}")
    if memory["observation_duration_ns"] <= 0:
        _fail(f"{name}.observation_duration_ns must be positive")
    if memory["buffer_census_complete"] is not True or memory["dropped_events"] != 0:
        _fail(f"{name} has an incomplete candidate allocation census")
    if memory["transient_live_bytes"] != 0:
        _fail(f"{name} retained candidate transient buffers after success")
    transient_events: list[tuple[int, int]] = []
    for index, raw_buffer in enumerate(buffers):
        buffer = _object(
            raw_buffer,
            f"{name}.buffers[{index}]",
            {
                "sequence",
                "name",
                "class",
                "scope",
                "bytes",
                "allocated_at_ns",
                "released_at_ns",
            },
        )
        if buffer["sequence"] != index:
            _fail(f"{name}.buffers must use contiguous sequence numbers")
        buffer_name = _string(buffer["name"], f"{name}.buffers[{index}].name")
        semantics = BUFFER_SEMANTICS.get(buffer_name)
        if semantics is None:
            _fail(f"{name}.buffers[{index}] has an unknown semantic name")
        if buffer["class"] not in {"retained_output", "discarded_output", "transient"}:
            _fail(f"{name}.buffers[{index}].class is unsupported")
        scope = _object(buffer["scope"], f"{name}.buffers[{index}].scope", SCOPE_FIELDS)
        for field, scope_value in scope.items():
            if scope_value is not None:
                _integer(scope_value, f"{name}.buffers[{index}].scope.{field}")
        _integer(buffer["bytes"], f"{name}.buffers[{index}].bytes")
        allocated_at = _integer(
            buffer["allocated_at_ns"], f"{name}.buffers[{index}].allocated_at_ns"
        )
        if allocated_at > memory["observation_duration_ns"]:
            _fail(f"{name}.buffers[{index}] allocation is outside the observation")
        released_at = buffer["released_at_ns"]
        if released_at is not None:
            _integer(released_at, f"{name}.buffers[{index}].released_at_ns")
            if not allocated_at <= released_at <= memory["observation_duration_ns"]:
                _fail(f"{name}.buffers[{index}] release is outside its lifetime")
        if buffer["class"] == "transient":
            if released_at is None:
                _fail(f"{name}.buffers[{index}] left a transient unreleased")
            transient_events.extend(
                ((allocated_at, buffer["bytes"]), (released_at, -buffer["bytes"]))
            )
        elif buffer["class"] == "retained_output":
            if released_at is not None:
                _fail(f"{name}.buffers[{index}] released retained output")
        else:
            _fail(f"{name}.buffers[{index}] contains discarded output on success")
        dtype, layout, full_image, expected_class = semantics
        if buffer["class"] != expected_class:
            _fail(f"{name}.buffers[{index}] class contradicts its static semantic inventory")
        if dtype == "float32" and full_image and layout in {"HWC", "CHW"}:
            _fail(f"{name} contains a forbidden full float32 {layout} image")
    if memory["allocation_count"] != len(buffers):
        _fail(f"{name}.allocation_count does not match the complete buffer census")
    if memory["allocated_bytes"] != sum(buffer["bytes"] for buffer in buffers):
        _fail(f"{name}.allocated_bytes does not match the complete buffer census")
    retained = sum(buffer["bytes"] for buffer in buffers if buffer["class"] == "retained_output")
    if memory["retained_output_bytes"] != retained:
        _fail(f"{name}.retained_output_bytes does not match the complete buffer census")
    for index, raw_copy in enumerate(copies):
        copy = _object(
            raw_copy,
            f"{name}.copies[{index}]",
            {"sequence", "name", "scope", "bytes"},
        )
        if copy["sequence"] != index:
            _fail(f"{name}.copies must use contiguous sequence numbers")
        if copy["name"] not in COPY_NAMES:
            _fail(f"{name}.copies[{index}] has an unknown semantic name")
        scope = _object(copy["scope"], f"{name}.copies[{index}].scope", SCOPE_FIELDS)
        for field, scope_value in scope.items():
            if scope_value is not None:
                _integer(scope_value, f"{name}.copies[{index}].scope.{field}")
        _integer(copy["bytes"], f"{name}.copies[{index}].bytes")
    if memory["copy_count"] != len(copies):
        _fail(f"{name}.copy_count does not match the complete copy census")
    if memory["copied_bytes"] != sum(copy["bytes"] for copy in copies):
        _fail(f"{name}.copied_bytes does not match the complete copy census")
    grouped: dict[int, dict[str, int]] = {}
    for timestamp, delta in transient_events:
        values = grouped.setdefault(timestamp, {"allocate": 0, "release": 0})
        values["allocate" if delta >= 0 else "release"] += abs(delta)
    live = 0
    minimum_peak = 0
    maximum_peak = 0
    for timestamp in sorted(grouped):
        allocated = grouped[timestamp]["allocate"]
        released = grouped[timestamp]["release"]
        if released > live + allocated:
            _fail(f"{name} transient buffer lifetime underflowed")
        maximum_peak = max(maximum_peak, live + allocated)
        live = live + allocated - released
        minimum_peak = max(minimum_peak, live)
    if live != 0 or not minimum_peak <= memory["peak_transient_live_bytes"] <= maximum_peak:
        _fail(f"{name}.peak_transient_live_bytes does not reconcile with lifetimes")
    return memory


def _validate_implementation(
    value: Any,
    name: str,
    *,
    thread_budget: int,
    affinity_available: bool,
    expected_affinity: Sequence[int] | None,
    candidate: bool,
    case_id: str,
) -> Mapping[str, Any]:
    implementation = _object(
        value,
        name,
        {
            "process_nonce",
            "process_pid",
            "thread_budget",
            "thread_control",
            "affinity",
            "samples",
            "timing_floor",
            "memory",
            "pre_conformance",
            "post_conformance",
        },
    )
    _sha256(implementation["process_nonce"], f"{name}.process_nonce")
    _integer(implementation["process_pid"], f"{name}.process_pid", minimum=1)
    if implementation["thread_budget"] != thread_budget:
        _fail(f"{name} has unequal thread budget")
    thread_control = _object(
        implementation["thread_control"],
        f"{name}.thread_control",
        {
            "owner",
            "torch_intraop_threads",
            "torch_interop_threads",
            "processor_thread_budget",
            "environment",
        },
    )
    environment = _object(
        thread_control["environment"],
        f"{name}.thread_control.environment",
        THREAD_ENVIRONMENT_NAMES,
    )
    for field, field_value in environment.items():
        _string(field_value, f"{name}.thread_control.environment.{field}")
    _integer(
        thread_control["torch_intraop_threads"],
        f"{name}.thread_control.torch_intraop_threads",
        minimum=1,
    )
    _integer(
        thread_control["torch_interop_threads"],
        f"{name}.thread_control.torch_interop_threads",
        minimum=1,
    )
    if thread_control["torch_interop_threads"] != 1:
        _fail(f"{name} must keep Torch inter-op at one")
    if candidate:
        _integer(
            thread_control["processor_thread_budget"],
            f"{name}.thread_control.processor_thread_budget",
            minimum=1,
        )
        if (
            thread_control["owner"] != "candidate-processor"
            or thread_control["processor_thread_budget"] != thread_budget
            or thread_control["torch_intraop_threads"] != 1
            or any(value != "1" for value in environment.values())
        ):
            _fail(f"{name} does not prove one Processor-owned total thread budget")
    else:
        expected_environment = {field: "1" for field in THREAD_ENVIRONMENT_NAMES}
        expected_environment["OMP_NUM_THREADS"] = str(thread_budget)
        if (
            thread_control["owner"] != "official-torch"
            or thread_control["processor_thread_budget"] is not None
            or thread_control["torch_intraop_threads"] != thread_budget
            or dict(environment) != expected_environment
        ):
            _fail(f"{name} does not prove one official Torch-owned total thread budget")
    affinity = _object(
        implementation["affinity"],
        f"{name}.affinity",
        {"available", "source", "requested_cpus", "observed_cpus"},
    )
    if affinity["available"] is not affinity_available:
        _fail(f"{name}.affinity availability contradicts the host attestation")
    _string(affinity["source"], f"{name}.affinity.source")
    requested_cpus = _list(affinity["requested_cpus"], f"{name}.affinity.requested_cpus")
    observed_cpus = _list(affinity["observed_cpus"], f"{name}.affinity.observed_cpus")
    for field, cpus in (("requested_cpus", requested_cpus), ("observed_cpus", observed_cpus)):
        if any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in cpus):
            _fail(f"{name}.affinity.{field} must contain non-negative CPU indices")
        if len(cpus) != len(set(cpus)):
            _fail(f"{name}.affinity.{field} must be unique")
    if affinity_available and (
        requested_cpus != observed_cpus
        or len(observed_cpus) != thread_budget
        or requested_cpus != list(expected_affinity or ())
    ):
        _fail(f"{name} did not receive its exact requested fixed CPU set")
    if not affinity_available and (requested_cpus or observed_cpus):
        _fail(f"{name} cannot claim CPU affinity when placement is unavailable")
    samples = _list(implementation["samples"], f"{name}.samples")
    if len(samples) != 30:
        _fail(f"{name} must contain exactly 30 raw timing samples")
    total_wall_seconds = 0.0
    for index, raw_sample in enumerate(samples):
        sample = _object(raw_sample, f"{name}.samples[{index}]", {"sequence", "wall_ms", "cpu_ms"})
        if sample.get("sequence") != index:
            _fail(f"{name}.samples must use contiguous sequence numbers")
        wall = _number(sample["wall_ms"], f"{name}.samples[{index}].wall_ms", positive=True)
        cpu = _number(sample["cpu_ms"], f"{name}.samples[{index}].cpu_ms", positive=True)
        if cpu / wall > thread_budget + CPU_UTILIZATION_TOLERANCE_CORES:
            _fail(f"{name}.samples[{index}] exceeds a plausible declared CPU budget")
        total_wall_seconds += wall / 1_000.0
    timing_floor = _object(
        implementation["timing_floor"],
        f"{name}.timing_floor",
        {
            "required_seconds",
            "raw_sample_iteration_count",
            "raw_sample_elapsed_wall_ms",
            "supplemental_iteration_count",
            "supplemental_elapsed_wall_ms",
            "supplemental_elapsed_cpu_ms",
            "total_iteration_count",
            "total_elapsed_wall_ms",
        },
    )
    if _number(timing_floor["required_seconds"], f"{name}.timing_floor.required_seconds") != 5.0:
        _fail(f"{name}.timing_floor changed the frozen five-second floor")
    if timing_floor["raw_sample_iteration_count"] != len(samples):
        _fail(f"{name}.timing_floor raw sample count does not reconcile")
    raw_elapsed = _number(
        timing_floor["raw_sample_elapsed_wall_ms"],
        f"{name}.timing_floor.raw_sample_elapsed_wall_ms",
        positive=True,
    )
    if not math.isclose(raw_elapsed, total_wall_seconds * 1_000, rel_tol=1e-12, abs_tol=1e-12):
        _fail(f"{name}.timing_floor raw elapsed time does not reconcile")
    supplemental_iterations = _integer(
        timing_floor["supplemental_iteration_count"],
        f"{name}.timing_floor.supplemental_iteration_count",
    )
    supplemental_wall = _number(
        timing_floor["supplemental_elapsed_wall_ms"],
        f"{name}.timing_floor.supplemental_elapsed_wall_ms",
    )
    supplemental_cpu = _number(
        timing_floor["supplemental_elapsed_cpu_ms"],
        f"{name}.timing_floor.supplemental_elapsed_cpu_ms",
    )
    if supplemental_wall == 0:
        if supplemental_iterations != 0 or supplemental_cpu != 0:
            _fail(f"{name}.timing_floor empty supplemental pass has nonzero counters")
    elif (
        supplemental_iterations == 0
        or supplemental_cpu / supplemental_wall > thread_budget + CPU_UTILIZATION_TOLERANCE_CORES
    ):
        _fail(f"{name}.timing_floor supplemental pass exceeds the CPU budget")
    if timing_floor["total_iteration_count"] != len(samples) + supplemental_iterations:
        _fail(f"{name}.timing_floor total iterations do not reconcile")
    total_elapsed = _number(
        timing_floor["total_elapsed_wall_ms"],
        f"{name}.timing_floor.total_elapsed_wall_ms",
        positive=True,
    )
    if not math.isclose(
        total_elapsed, raw_elapsed + supplemental_wall, rel_tol=1e-12, abs_tol=1e-12
    ):
        _fail(f"{name}.timing_floor total elapsed time does not reconcile")
    if total_elapsed < 5_000:
        _fail(f"{name} has fewer than five seconds of timing samples")
    _validate_memory(implementation["memory"], f"{name}.memory", candidate=candidate)
    before = _validate_conformance(
        implementation["pre_conformance"], f"{name}.pre_conformance", case_id=case_id
    )
    after = _validate_conformance(
        implementation["post_conformance"], f"{name}.post_conformance", case_id=case_id
    )
    if before != after:
        _fail(f"{name} output changed between pre/post conformance")
    return implementation


def _expected_coordinates() -> set[tuple[str, str, int]]:
    coordinates = {
        (profile, case, thread)
        for profile in PROFILES
        for case in CASES
        for thread in (1, PRODUCTION_THREAD_BUDGET)
    }
    coordinates.update((profile, "image24", thread) for profile in PROFILES for thread in (2, 4))
    return coordinates


def _timed_coordinates() -> list[tuple[str, str, int]]:
    return sorted(
        coordinate for coordinate in _expected_coordinates() if coordinate[1] != CACHED_CASE
    )


def _expected_order_seed(profile: str, case_id: str, thread_budget: int) -> int:
    """Mirror the four independently authenticated t1/t2/t4/t8 runner schedules."""

    if thread_budget in (2, 4):
        if case_id != "image24":
            _fail("intermediate thread budgets are only defined for image24")
        schedule_index = PROFILES.index(profile)
    else:
        timed_cases = tuple(case for case in CASES if case != CACHED_CASE)
        schedule_index = PROFILES.index(profile) * len(timed_cases) + timed_cases.index(case_id)
    return RANDOM_SEED + schedule_index


def _randomized_orders(seed: int) -> list[list[str]]:
    generator = random.Random(seed)
    orders: list[list[str]] = []
    for repetition in range(5):
        order = ["reference", "candidate"]
        if generator.randrange(2):
            order.reverse()
        if repetition > 0 and order == orders[-1]:
            order.reverse()
        orders.append(order)
    return orders


def _validate_observation(
    value: Any,
    name: str,
    *,
    architecture: str,
    build_label: str,
    affinity_available: bool,
    physical_core_masks: Mapping[str, Any],
    random_seed: int,
) -> tuple[str, str, int]:
    observation = _object(
        value,
        name,
        {
            "profile",
            "case_id",
            "cache_mode",
            "thread_budget",
            "order_seed",
            "work_units",
            "support_status",
            "support_reason",
            "timing_kind",
            "pairs",
        },
    )
    profile = _string(observation["profile"], f"{name}.profile")
    case_id = _string(observation["case_id"], f"{name}.case_id")
    thread_budget = _integer(observation["thread_budget"], f"{name}.thread_budget", minimum=1)
    if profile not in PROFILES or case_id not in CASES:
        _fail(f"{name} has an unsupported profile/case")
    if thread_budget not in THREAD_BUDGETS:
        _fail(f"{name} has an unsupported thread budget")
    _integer(observation["work_units"], f"{name}.work_units", minimum=1)
    if observation["work_units"] != WORK_UNITS[case_id]:
        _fail(f"{name}.work_units does not match the frozen workload")
    pairs = _list(observation["pairs"], f"{name}.pairs")
    if case_id == CACHED_CASE:
        if (
            observation["cache_mode"] != "enabled"
            or observation["support_status"] != "unsupported"
            or observation["support_reason"] != "adapter_cache_supported_false"
            or observation["timing_kind"] != "unsupported"
            or observation["order_seed"] is not None
            or pairs
        ):
            _fail(f"{name} must retain repeat24_cached as unsupported/non-gating without timing")
        return profile, case_id, thread_budget
    if (
        observation["cache_mode"]
        != ("separated" if case_id == "repeat24_separated" else "disabled")
        or observation["support_status"] != "supported"
        or observation["support_reason"] is not None
        or observation["timing_kind"] != "uncached"
    ):
        _fail(f"{name} has mislabeled cache timing")
    if random_seed != RANDOM_SEED:
        _fail(f"{name} does not use the frozen D4 random seed")
    expected_order_seed = _expected_order_seed(profile, case_id, thread_budget)
    if observation["order_seed"] != expected_order_seed:
        _fail(f"{name}.order_seed is not canonically derived from the capture schedule")
    if len(pairs) != 5:
        _fail(f"{name} requires five fresh process pairs")
    repetitions: set[int] = set()
    orders: set[tuple[str, str]] = set()
    coordinate_inputs: set[tuple[str, str, str]] = set()
    coordinate_affinities: set[tuple[int, ...]] = set()
    process_nonces: set[str] = set()
    process_pids: set[int] = set()
    expected_orders = _randomized_orders(expected_order_seed)
    for pair_index, raw_pair in enumerate(pairs):
        pair_name = f"{name}.pairs[{pair_index}]"
        pair = _object(
            raw_pair,
            pair_name,
            {
                "pair_id",
                "repetition",
                "order",
                "input_sha256",
                "logical_input_sha256",
                "messages_sha256",
                "conformance_witness_sha256",
                "implementations",
            },
        )
        expected_pair_id = (
            f"{architecture}/{build_label}/{profile}/{case_id}/t{thread_budget}/r{pair_index}"
        )
        if pair["pair_id"] != expected_pair_id:
            _fail(f"{pair_name}.pair_id is not canonical")
        repetition = _integer(pair["repetition"], f"{pair_name}.repetition")
        if repetition != pair_index:
            _fail(f"{pair_name}.repetition is out of canonical sequence")
        repetitions.add(repetition)
        order = _list(pair["order"], f"{pair_name}.order")
        if order not in (["reference", "candidate"], ["candidate", "reference"]):
            _fail(f"{pair_name}.order must be AB or BA")
        if order != expected_orders[pair_index]:
            _fail(f"{pair_name}.order does not match the authenticated random seed")
        orders.add(tuple(order))
        input_sha = _sha256(pair["input_sha256"], f"{pair_name}.input_sha256")
        logical_input_sha = _sha256(
            pair["logical_input_sha256"], f"{pair_name}.logical_input_sha256"
        )
        messages_sha = _sha256(pair["messages_sha256"], f"{pair_name}.messages_sha256")
        witness_sha = _sha256(
            pair["conformance_witness_sha256"],
            f"{pair_name}.conformance_witness_sha256",
        )
        coordinate_inputs.add((input_sha, logical_input_sha, messages_sha))
        implementations = _object(
            pair["implementations"],
            f"{pair_name}.implementations",
            {"reference", "candidate"},
        )
        for implementation_name in ("reference", "candidate"):
            implementation = _validate_implementation(
                implementations[implementation_name],
                f"{pair_name}.implementations.{implementation_name}",
                thread_budget=thread_budget,
                affinity_available=affinity_available,
                expected_affinity=physical_core_masks[f"t{thread_budget}"],
                candidate=implementation_name == "candidate",
                case_id=case_id,
            )
            nonce = implementation["process_nonce"]
            if nonce in process_nonces:
                _fail(f"{name} reused a process nonce instead of a fresh process")
            process_nonces.add(nonce)
            process_pid = implementation["process_pid"]
            if process_pid in process_pids:
                _fail(f"{name} reused a PID within one benchmark result")
            process_pids.add(process_pid)
            for position in ("pre_conformance", "post_conformance"):
                conformance = implementation[position]
                if (
                    conformance["input_sha256"] != input_sha
                    or conformance["messages_sha256"] != messages_sha
                    or conformance["witness_sha256"] != witness_sha
                ):
                    _fail(f"{pair_name} conformance is not bound to its exact input/witness")
        reference_affinity = implementations["reference"]["affinity"]
        candidate_affinity = implementations["candidate"]["affinity"]
        if reference_affinity != candidate_affinity:
            _fail(f"{pair_name} reference/candidate CPU placement differs")
        coordinate_affinities.add(tuple(reference_affinity["observed_cpus"]))
        reference_pre = implementations["reference"]["pre_conformance"]
        candidate_pre = implementations["candidate"]["pre_conformance"]
        for field in ("output_keys", "dtypes", "float_atol", "tolerance_policy"):
            if reference_pre[field] != candidate_pre[field]:
                _fail(f"{pair_name} reference/candidate {field} differs")
    if repetitions != set(range(5)):
        _fail(f"{name} repetitions must be exactly 0..4")
    if len(orders) != 2:
        _fail(f"{name} must contain both AB and BA process order")
    if len(coordinate_inputs) != 1:
        _fail(f"{name} input/messages changed across process pairs")
    if len(coordinate_affinities) != 1:
        _fail(f"{name} CPU placement changed across process pairs")
    return profile, case_id, thread_budget


def _validate_capture(value: Any, index: int, expected_identity: Mapping[str, Any]) -> None:
    name = f"captures[{index}]"
    capture = _object(
        value,
        name,
        {
            "capture_id",
            "architecture",
            "identity",
            "build",
            "host",
            "toolchain",
            "environment",
            "commands",
            "raw_archive_sha256",
            "raw_archive_bytes",
            "raw_manifest_sha256",
            "raw_provenance_sha256",
            "protocol",
            "phase_c",
            "observations",
        },
    )
    architecture = _string(capture["architecture"], f"{name}.architecture")
    if architecture not in ARCHITECTURES:
        _fail(f"{name}.architecture is unsupported")
    identity = _validate_identity(capture["identity"], f"{name}.identity")
    if identity != expected_identity:
        _fail(f"{name}.identity is stale or cross-source")
    build = _validate_build(capture["build"], architecture)
    label = build["label"]
    if capture["capture_id"] != f"{architecture}-{label}":
        _fail(f"{name}.capture_id is not canonical")
    host = _validate_host(capture["host"], architecture)
    _validate_toolchain(capture["toolchain"], f"{name}.toolchain")
    environment = _object(
        capture["environment"],
        f"{name}.environment",
        {"PYTHONHASHSEED", "LC_ALL", "TZ"},
    )
    for field in environment:
        _string(environment[field], f"{name}.environment.{field}")
    commands = _list(capture["commands"], f"{name}.commands")
    if not commands or not all(isinstance(command, str) and command for command in commands):
        _fail(f"{name}.commands must preserve non-empty reproduction commands")
    for field in ("raw_archive_sha256", "raw_manifest_sha256", "raw_provenance_sha256"):
        _sha256(capture[field], f"{name}.{field}")
    _integer(capture["raw_archive_bytes"], f"{name}.raw_archive_bytes", minimum=1)
    _validate_protocol(capture["protocol"], f"{name}.protocol")
    _validate_phase_c(
        capture["phase_c"],
        f"{name}.phase_c",
        identity=identity,
        build=build,
    )
    observations = _list(capture["observations"], f"{name}.observations")
    coordinates = {
        _validate_observation(
            observation,
            f"{name}.observations[{observation_index}]",
            architecture=architecture,
            build_label=label,
            affinity_available=host["affinity_available"],
            physical_core_masks=host["physical_core_masks"],
            random_seed=capture["protocol"]["random_seed"],
        )
        for observation_index, observation in enumerate(observations)
    }
    if len(coordinates) != len(observations):
        _fail(f"{name} has duplicate observation coordinates")
    if coordinates != _expected_coordinates():
        missing = sorted(_expected_coordinates() - coordinates)
        extra = sorted(coordinates - _expected_coordinates())
        _fail(f"{name} has an incomplete D4 matrix (missing={missing}, extra={extra})")
    capture_nonces = [
        implementation["process_nonce"]
        for observation in observations
        for pair in observation["pairs"]
        for implementation in pair["implementations"].values()
    ]
    if len(capture_nonces) != len(set(capture_nonces)):
        _fail(f"{name} reused a process nonce across benchmark results")


def _sample_metrics(samples: Sequence[Mapping[str, Any]], work_units: int) -> dict[str, Any]:
    walls = [float(sample["wall_ms"]) for sample in samples]
    cpus = [float(sample["cpu_ms"]) for sample in samples]
    throughput = [work_units * 1_000.0 / wall for wall in walls]
    utilization = [cpu / wall for cpu, wall in zip(cpus, walls, strict=True)]
    return {
        "sample_count": len(samples),
        "wall_ms": {
            "p50": _percentile(walls, 0.50),
            "p90": _percentile(walls, 0.90),
            "p99": _percentile(walls, 0.99) if len(walls) >= 100 else None,
            "p99_qualified": len(walls) >= 100,
        },
        "cpu_ms": {"p50": _percentile(cpus, 0.50), "p90": _percentile(cpus, 0.90)},
        "throughput_per_s": {
            "p50": _percentile(throughput, 0.50),
            "p90": _percentile(throughput, 0.90),
        },
        "core_utilization": {
            "p50": _percentile(utilization, 0.50),
            "p90": _percentile(utilization, 0.90),
        },
    }


def _implementation_metrics(
    implementations: Sequence[Mapping[str, Any]], work_units: int
) -> dict[str, Any]:
    """Give each fresh process equal weight in all reported latency metrics."""

    process_metrics = [
        _sample_metrics(implementation["samples"], work_units) for implementation in implementations
    ]

    def median(path: tuple[str, str]) -> float:
        return statistics.median(float(metric[path[0]][path[1]]) for metric in process_metrics)

    p99_qualified = all(metric["wall_ms"]["p99_qualified"] for metric in process_metrics)
    return {
        "sample_count": sum(metric["sample_count"] for metric in process_metrics),
        "iteration_count": sum(
            implementation["timing_floor"]["total_iteration_count"]
            for implementation in implementations
        ),
        "measured_seconds": sum(
            implementation["timing_floor"]["total_elapsed_wall_ms"]
            for implementation in implementations
        )
        / 1_000.0,
        "wall_ms": {
            "p50": median(("wall_ms", "p50")),
            "p90": median(("wall_ms", "p90")),
            "p99": median(("wall_ms", "p99")) if p99_qualified else None,
            "p99_qualified": p99_qualified,
        },
        "cpu_ms": {
            "p50": median(("cpu_ms", "p50")),
            "p90": median(("cpu_ms", "p90")),
        },
        "throughput_per_s": {
            "p50": median(("throughput_per_s", "p50")),
            "p90": median(("throughput_per_s", "p90")),
        },
        "core_utilization": {
            "p50": median(("core_utilization", "p50")),
            "p90": median(("core_utilization", "p90")),
        },
    }


def _memory_metrics(implementations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mandatory_fields = (
        "rss_before_bytes",
        "rss_after_bytes",
        "scoped_peak_rss_bytes",
        "external_transient_rss_bytes",
        "retained_output_bytes",
    )
    optional_fields = (
        "peak_transient_live_bytes",
        "allocation_count",
        "allocated_bytes",
        "copy_count",
        "copied_bytes",
    )
    result: dict[str, Any] = {
        field: max(implementation["memory"][field] for implementation in implementations)
        for field in mandatory_fields
    }
    available = all(
        implementation["memory"]["native_counters_available"] for implementation in implementations
    )
    result["native_counters_available"] = available
    result["buffer_census_complete"] = all(
        implementation["memory"]["buffer_census_complete"] for implementation in implementations
    )
    result.update(
        {
            field: (
                max(implementation["memory"][field] for implementation in implementations)
                if available
                else None
            )
            for field in optional_fields
        }
    )
    return result


def _summaries(captures: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for capture in sorted(
        captures, key=lambda item: (item["architecture"], item["build"]["label"])
    ):
        architecture = capture["architecture"]
        build = capture["build"]["label"]
        random_seed = capture["protocol"]["random_seed"]
        observations = sorted(
            capture["observations"],
            key=lambda item: (item["profile"], item["case_id"], item["thread_budget"]),
        )
        for observation in observations:
            base: dict[str, Any] = {
                "architecture": architecture,
                "build": build,
                "profile": observation["profile"],
                "case_id": observation["case_id"],
                "thread_budget": observation["thread_budget"],
                "status": observation["support_status"],
            }
            if observation["support_status"] == "unsupported":
                base["reason"] = "adapter_cache_supported_false; non-gating and not timed"
                summaries.append(base)
                continue
            reference_implementations = [
                pair["implementations"]["reference"] for pair in observation["pairs"]
            ]
            candidate_implementations = [
                pair["implementations"]["candidate"] for pair in observation["pairs"]
            ]
            speedups = [
                _percentile([float(sample["wall_ms"]) for sample in reference["samples"]], 0.50)
                / _percentile([float(sample["wall_ms"]) for sample in candidate["samples"]], 0.50)
                for reference, candidate in zip(
                    reference_implementations, candidate_implementations, strict=True
                )
            ]
            paired_memory_ratios: list[float | None] = []
            for reference, candidate in zip(
                reference_implementations, candidate_implementations, strict=True
            ):
                official = reference["memory"]["external_transient_rss_bytes"]
                candidate_envelope = max(
                    candidate["memory"]["external_transient_rss_bytes"],
                    candidate["memory"]["peak_transient_live_bytes"],
                )
                paired_memory_ratios.append(
                    None if official == 0 else candidate_envelope / official
                )
            coordinate = (
                f"{architecture}/{build}/{observation['profile']}/"
                f"{observation['case_id']}/t{observation['thread_budget']}"
            )
            coordinate_seed = random_seed + int.from_bytes(
                hashlib.sha256(coordinate.encode("utf-8")).digest()[:4], "big"
            )
            base.update(
                {
                    "reason": None,
                    "reference": _implementation_metrics(
                        reference_implementations, observation["work_units"]
                    ),
                    "candidate": _implementation_metrics(
                        candidate_implementations, observation["work_units"]
                    ),
                    "memory": {
                        "reference": _memory_metrics(reference_implementations),
                        "candidate": _memory_metrics(candidate_implementations),
                    },
                    "paired_process_speedups": speedups,
                    "paired_memory_ratios": paired_memory_ratios,
                    "speedup_p50": statistics.median(speedups),
                    "speedup_bootstrap_95_ci": _bootstrap(speedups, seed=coordinate_seed),
                }
            )
            summaries.append(base)
    return summaries


def _gate(gate_id: str, passed: bool, reason: str) -> dict[str, str]:
    return {"gate_id": gate_id, "status": "pass" if passed else "miss", "reason": reason}


def _gates(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    by_key = {
        (
            summary["architecture"],
            summary["build"],
            summary["profile"],
            summary["case_id"],
            summary["thread_budget"],
        ): summary
        for summary in summaries
    }
    gates: list[dict[str, str]] = []
    for architecture in ARCHITECTURES:
        for profile in PROFILES:
            for case_id in ("image24", "ragged24"):
                summary = by_key[(architecture, "shipping", profile, case_id, 8)]
                lower = summary["speedup_bootstrap_95_ci"]["lower"]
                gates.append(
                    _gate(
                        f"headline/{architecture}/{profile}/{case_id}",
                        lower >= 2.0,
                        f"shipping t8 paired-bootstrap lower={lower:.3f}x; required >=2.000x",
                    )
                )
    targets = {"qwen3-vl-8b": 111.0, "qwen3.5-9b": 111.6}
    for profile, target in targets.items():
        summary = by_key[("arm64", "shipping", profile, "image24", 8)]
        observed = summary["candidate"]["wall_ms"]["p50"]
        gates.append(
            _gate(
                f"m4-point/{profile}/image24",
                observed < target,
                f"shipping t8 candidate p50={observed:.3f}ms; required <{target:.1f}ms",
            )
        )
    for architecture in ARCHITECTURES:
        for profile in PROFILES:
            for case_id in ("image1", "text_short", "text_long"):
                for thread_budget in (1, 8):
                    summary = by_key[(architecture, "shipping", profile, case_id, thread_budget)]
                    ratio = (
                        summary["candidate"]["wall_ms"]["p50"]
                        / summary["reference"]["wall_ms"]["p50"]
                    )
                    gates.append(
                        _gate(
                            f"regression/{architecture}/{profile}/{case_id}/t{thread_budget}",
                            ratio <= 1.05,
                            f"shipping candidate/reference p50={ratio:.4f}; required <=1.0500",
                        )
                    )
    for architecture in ARCHITECTURES:
        for profile in PROFILES:
            t1 = by_key[(architecture, "shipping", profile, "image24", 1)]["candidate"]["wall_ms"][
                "p50"
            ]
            for thread_budget in (2, 4, 8):
                tn = by_key[(architecture, "shipping", profile, "image24", thread_budget)][
                    "candidate"
                ]["wall_ms"]["p50"]
                efficiency = t1 / (thread_budget * tn)
                gates.append(
                    _gate(
                        f"efficiency/{architecture}/{profile}/image24/t{thread_budget}",
                        efficiency >= 0.60,
                        f"shipping E_{thread_budget}={efficiency:.4f}; required >=0.6000",
                    )
                )
    for summary in summaries:
        if (
            summary["build"] != "shipping"
            or summary["status"] != "supported"
            or summary["case_id"] not in IMAGE_CASES
        ):
            continue
        ratios = summary["paired_memory_ratios"]
        passed = all(ratio is not None and ratio <= 0.5 for ratio in ratios)
        ratio_text = (
            "unmeasurable"
            if any(ratio is None for ratio in ratios)
            else f"max-paired={max(ratios):.4f}"
        )
        gates.append(
            _gate(
                (
                    f"memory/{summary['architecture']}/{summary['profile']}/"
                    f"{summary['case_id']}/t{summary['thread_budget']}"
                ),
                passed,
                "shipping per-process candidate max(external RSS, exact native peak)/official "
                f"external RSS ratio={ratio_text}; every official must be >0 and every paired "
                "ratio <=0.5000",
            )
        )
    return gates


def build_certification(
    captures: Sequence[Mapping[str, Any]],
    *,
    current_identity: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    identity = _validate_identity(current_identity, "current_identity")
    if not isinstance(captures, list):
        _fail("captures must be a list")
    if len(captures) != 4:
        _fail("certification requires exactly four ARM/x86 shipping/native captures")
    for index, capture in enumerate(captures):
        _validate_capture(capture, index, identity)
    capture_keys = [(capture["architecture"], capture["build"]["label"]) for capture in captures]
    expected_capture_keys = {
        (architecture, build) for architecture in ARCHITECTURES for build in BUILDS
    }
    if set(capture_keys) != expected_capture_keys:
        _fail("captures must contain each architecture/build exactly once")
    if len(set(capture_keys)) != 4:
        _fail("capture architecture/build pairs must be unique")
    for architecture in ARCHITECTURES:
        matching = [capture for capture in captures if capture["architecture"] == architecture]
        shipping = next(capture for capture in matching if capture["build"]["label"] == "shipping")
        native = next(capture for capture in matching if capture["build"]["label"] == "native")
        if shipping["host"]["host_fingerprint"] != native["host"]["host_fingerprint"]:
            _fail(f"{architecture} shipping/native captures used different hosts")
        if shipping["host"] != native["host"]:
            _fail(f"{architecture} shipping/native host attestations differ")
        for field in (
            "raw_archive_sha256",
            "raw_archive_bytes",
            "raw_manifest_sha256",
            "raw_provenance_sha256",
        ):
            if shipping[field] != native[field]:
                _fail(f"{architecture} shipping/native raw archive bindings differ")
        if shipping["protocol"] != native["protocol"]:
            _fail(f"{architecture} shipping/native captures used different protocols")
        if shipping["toolchain"] != native["toolchain"]:
            _fail(f"{architecture} shipping/native captures used different toolchains")
        if shipping["environment"] != native["environment"]:
            _fail(f"{architecture} shipping/native captures used different environments")
        if shipping["build"]["wheel_sha256"] == native["build"]["wheel_sha256"]:
            _fail(f"{architecture} shipping/native wheel artifacts are not distinct")
        if shipping["build"]["native_sha256"] == native["build"]["native_sha256"]:
            _fail(f"{architecture} shipping/native extension artifacts are not distinct")
    protocols = [capture["protocol"] for capture in captures]
    if any(protocol != protocols[0] for protocol in protocols[1:]):
        _fail("ARM/x86 captures used different frozen protocols")
    cross_host_inputs: dict[tuple[str, str, int], set[tuple[str, str]]] = {}
    architecture_inputs: dict[tuple[str, str, str, int], set[tuple[str, str]]] = {}
    for capture in captures:
        for observation in capture["observations"]:
            if observation["support_status"] == "unsupported":
                continue
            coordinate = (
                observation["profile"],
                observation["case_id"],
                observation["thread_budget"],
            )
            first_pair = observation["pairs"][0]
            cross_host_inputs.setdefault(coordinate, set()).add(
                (first_pair["logical_input_sha256"], first_pair["messages_sha256"])
            )
            architecture_inputs.setdefault((capture["architecture"], *coordinate), set()).add(
                (first_pair["input_sha256"], first_pair["messages_sha256"])
            )
    drifted_cross_host = [
        coordinate for coordinate, values in cross_host_inputs.items() if len(values) != 1
    ]
    if drifted_cross_host:
        _fail(f"cross-architecture logical input drift detected: {sorted(drifted_cross_host)}")
    drifted_architecture = [
        coordinate for coordinate, values in architecture_inputs.items() if len(values) != 1
    ]
    if drifted_architecture:
        _fail(f"same-architecture exact input drift detected: {sorted(drifted_architecture)}")
    timestamp = _string(created_at, "created_at")
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise PerformanceCertificationError("created_at must be an ISO-8601 timestamp") from error
    if parsed_timestamp.tzinfo is None:
        _fail("created_at must include an explicit timezone")
    summaries = _summaries(captures)
    gates = _gates(summaries)
    passed = all(gate["status"] == "pass" for gate in gates)
    artifact = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "current_identity": identity,
        "captures": list(captures),
        "summaries": summaries,
        "gates": gates,
        "certification_status": "pass" if passed else "miss",
        "releasable": passed,
        "claims": {
            "architectures_aggregated": False,
            "native_build_gating": False,
            "video_performance": False,
            "vllm_production": False,
        },
    }
    _canonical_json(artifact)
    return artifact


def validate_certification(
    artifact: Mapping[str, Any], *, current_identity: Mapping[str, Any]
) -> None:
    value = _object(
        artifact,
        "certification",
        {
            "schema_id",
            "schema_version",
            "created_at",
            "current_identity",
            "captures",
            "summaries",
            "gates",
            "certification_status",
            "releasable",
            "claims",
        },
    )
    if value["schema_id"] != SCHEMA_ID or value["schema_version"] != SCHEMA_VERSION:
        _fail("unsupported performance certification schema")
    expected_identity = _validate_identity(current_identity, "current_identity")
    if value["current_identity"] != expected_identity:
        _fail("certification is stale for the current source/schema/workload identity")
    expected = build_certification(
        value["captures"],
        current_identity=expected_identity,
        created_at=value["created_at"],
    )
    if _canonical_json(value) != _canonical_json(expected):
        _fail("certification derived summaries, gates, claims, or conclusion were forged")


def render_report(artifact: Mapping[str, Any], *, current_identity: Mapping[str, Any]) -> str:
    validate_certification(artifact, current_identity=current_identity)
    status = artifact["certification_status"].upper()
    lines = [
        "# qwen-mm image performance certification v1",
        "",
        f"- Status: `{status}`",
        f"- Releasable: `{str(artifact['releasable']).lower()}`",
        "- Gating build: `shipping` (portable release); native results are supplemental.",
        "- Architectures: macOS ARM and Linux x86 are evaluated separately and never aggregated.",
        "- Scope: CPU preprocessing only; this report makes no video or vLLM production claim.",
        "- `repeat24_cached`: unsupported/non-gating and deliberately not timed.",
        "",
        "## Gate conclusion",
        "",
    ]
    misses = [gate for gate in artifact["gates"] if gate["status"] == "miss"]
    if misses:
        lines.append(f"Certification missed {len(misses)} unchanged gate(s):")
        lines.append("")
        lines.extend(f"- `{gate['gate_id']}`: {gate['reason']}" for gate in misses)
    else:
        lines.append("All unchanged D4 shipping-build gates passed.")
    lines.extend(
        [
            "",
            "## Headline shipping results",
            "",
            "| Architecture | Profile | Case | Candidate p50 | Speedup | 95% CI |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for summary in artifact["summaries"]:
        if (
            summary["build"] == "shipping"
            and summary["case_id"] in {"image24", "ragged24"}
            and summary["thread_budget"] == 8
        ):
            interval = summary["speedup_bootstrap_95_ci"]
            lines.append(
                "| {architecture} | {profile} | {case} | {candidate:.3f} ms | "
                "{speedup:.3f}x | {lower:.3f}–{upper:.3f}x |".format(
                    architecture=summary["architecture"],
                    profile=summary["profile"],
                    case=summary["case_id"],
                    candidate=summary["candidate"]["wall_ms"]["p50"],
                    speedup=summary["speedup_p50"],
                    lower=interval["lower"],
                    upper=interval["upper"],
                )
            )
    lines.extend(
        [
            "",
            "Raw timing samples, separate scoped-memory observations, allocation census data, "
            "pre/post conformance witnesses, build artifacts, host placement, and reproduction "
            "commands are retained in the versioned JSON artifact.",
            "",
        ]
    )
    return "\n".join(lines)
