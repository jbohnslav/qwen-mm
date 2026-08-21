from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path
from typing import Any

import jsonschema
import qwen_mm_reference.performance_certification_v1 as certification
from qwen_mm_reference.performance_certification_v1 import (
    PerformanceCertificationError,
    build_certification,
    render_report,
    validate_certification,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "benchmarks" / "performance-certification-schema-v1.json"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def current_identity() -> dict[str, str]:
    return {
        "source_revision": "a" * 40,
        "source_sha256": digest("source"),
        "benchmark_schema_sha256": digest("benchmark-schema"),
        "workload_sha256": digest("workload"),
        "profile_schema_sha256": digest("profile-schema"),
        "model_registry_sha256": digest("models"),
        "assets_sha256": digest("assets"),
        "protocol_sha256": digest("protocol"),
    }


def protocol(seed: int) -> dict[str, Any]:
    return {
        "mode": "dedicated",
        "process_repetitions": 5,
        "random_seed": seed,
        "bootstrap_resamples": 20_000,
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
        "profiles": list(certification.PROFILES),
        "cases": list(certification.CASES),
        "thread_budgets": list(certification.THREAD_BUDGETS),
        "production_thread_budget": 8,
        "cache_policy": "repeat24_cached unsupported/non-gating; uncached timing forbidden",
    }


def conformance_record(coordinate: str, implementation: str) -> dict[str, Any]:
    case_id = coordinate.split("/")[1]
    output_keys = ["input_ids", "attention_mask", "mm_token_type_ids"]
    dtypes = {key: "int64" for key in output_keys}
    if case_id in certification.IMAGE_CASES:
        output_keys.extend(("pixel_values", "image_grid_thw"))
        dtypes.update(pixel_values="float32", image_grid_thw="int64")
    return {
        "passed": True,
        "witness_sha256": digest(f"witness/{coordinate}"),
        "input_sha256": digest(f"input/{coordinate}"),
        "messages_sha256": digest(f"messages/{coordinate}"),
        "output_keys": output_keys,
        "dtypes": dtypes,
        "values_sha256": digest(f"values/{coordinate}/{implementation}"),
        "float_atol": certification.FLOAT_ATOL[case_id],
        "tolerance_policy": (
            "exact integers; exact timed stability; pixel per-occurrence lossless=1e-6 "
            "otherwise workload float_atol"
        ),
        "all_timed_outputs_match": True,
        "fallback_work": False,
        "cache_used": False,
    }


def memory_record(implementation: str) -> dict[str, Any]:
    external = 200 if implementation == "reference" else 100
    retained = 100
    before = 1_000
    # Deliberately differs from baseline + output bytes so the fixture catches
    # accidental subtraction of allocator-retained rss_after context.
    after = 1_050
    transient = 200 if implementation == "reference" else 100
    base: dict[str, Any] = {
        "rss_before_bytes": before,
        "rss_after_bytes": after,
        "scoped_peak_rss_bytes": before + retained + external,
        "retained_output_bytes": retained,
        "external_transient_rss_bytes": external,
        "native_counters_available": implementation == "candidate",
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
    if implementation == "candidate":
        scope = {
            "request_index": 0,
            "message_index": 0,
            "content_item_index": 0,
            "media_index": 0,
            "input_index": 0,
        }
        base.update(
            observation_duration_ns=100,
            allocation_count=2,
            allocated_bytes=retained + transient,
            copy_count=0,
            copied_bytes=0,
            transient_live_bytes=0,
            peak_transient_live_bytes=transient,
            buffer_census_complete=True,
            dropped_events=0,
            buffers=[
                {
                    "sequence": 0,
                    "name": "prepared_rgb",
                    "class": "transient",
                    "scope": scope,
                    "bytes": transient,
                    "allocated_at_ns": 10,
                    "released_at_ns": 50,
                },
                {
                    "sequence": 1,
                    "name": "pixel_values",
                    "class": "retained_output",
                    "scope": {
                        "request_index": None,
                        "message_index": None,
                        "content_item_index": None,
                        "media_index": None,
                        "input_index": None,
                    },
                    "bytes": retained,
                    "allocated_at_ns": 60,
                    "released_at_ns": None,
                },
            ],
        )
    return base


def wall_time(case_id: str, thread_budget: int, implementation: str) -> float:
    if case_id == "image24":
        if implementation == "candidate":
            return {1: 800.0, 2: 400.0, 4: 200.0, 8: 100.0}[thread_budget]
        return 250.0 if thread_budget == 8 else 800.0
    if case_id == "ragged24":
        return 100.0 if implementation == "candidate" else 250.0
    return 200.0


def implementation_record(
    architecture: str,
    coordinate: str,
    case_id: str,
    thread_budget: int,
    implementation: str,
    repetition: int,
) -> dict[str, Any]:
    wall = wall_time(case_id, thread_budget, implementation)
    sample_count = 30
    cpu = wall * min(thread_budget * 0.8, 4.0)
    raw_elapsed = sample_count * wall
    supplemental_iterations = max(0, int((5_000 - raw_elapsed + wall - 1) // wall))
    supplemental_wall = supplemental_iterations * wall
    conformance = conformance_record(coordinate, implementation)
    return {
        "process_nonce": digest(
            f"process/{architecture}/{coordinate}/{repetition}/{implementation}"
        ),
        "process_pid": 10_000 + repetition + (100 if implementation == "candidate" else 0),
        "thread_budget": thread_budget,
        "thread_control": {
            "owner": "candidate-processor" if implementation == "candidate" else "official-torch",
            "torch_intraop_threads": thread_budget if implementation == "reference" else 1,
            "torch_interop_threads": 1,
            "processor_thread_budget": thread_budget if implementation == "candidate" else None,
            "environment": {
                name: (
                    str(thread_budget)
                    if implementation == "reference" and name == "OMP_NUM_THREADS"
                    else "1"
                )
                for name in certification.THREAD_ENVIRONMENT_NAMES
            },
        },
        "affinity": {
            "available": architecture == "x86_64",
            "source": "taskset" if architecture == "x86_64" else "macos-unavailable",
            "requested_cpus": list(range(thread_budget)) if architecture == "x86_64" else [],
            "observed_cpus": list(range(thread_budget)) if architecture == "x86_64" else [],
        },
        "samples": [
            {"sequence": sequence, "wall_ms": wall, "cpu_ms": cpu}
            for sequence in range(sample_count)
        ],
        "timing_floor": {
            "required_seconds": 5.0,
            "raw_sample_iteration_count": sample_count,
            "raw_sample_elapsed_wall_ms": raw_elapsed,
            "supplemental_iteration_count": supplemental_iterations,
            "supplemental_elapsed_wall_ms": supplemental_wall,
            "supplemental_elapsed_cpu_ms": supplemental_iterations * cpu,
            "total_iteration_count": sample_count + supplemental_iterations,
            "total_elapsed_wall_ms": raw_elapsed + supplemental_wall,
        },
        "memory": memory_record(implementation),
        "pre_conformance": conformance,
        "post_conformance": copy.deepcopy(conformance),
    }


def observation(
    architecture: str,
    build: str,
    profile: str,
    case_id: str,
    thread_budget: int,
    *,
    process_repetitions: int = 5,
) -> dict[str, Any]:
    coordinate = f"{profile}/{case_id}/t{thread_budget}"
    cache_mode = (
        "enabled"
        if case_id == "repeat24_cached"
        else ("separated" if case_id == "repeat24_separated" else "disabled")
    )
    base: dict[str, Any] = {
        "profile": profile,
        "case_id": case_id,
        "cache_mode": cache_mode,
        "thread_budget": thread_budget,
        "order_seed": None,
        "work_units": certification.WORK_UNITS[case_id],
        "support_status": "supported",
        "support_reason": None,
        "timing_kind": "uncached",
        "pairs": [],
    }
    if case_id == "repeat24_cached":
        base.update(
            support_status="unsupported",
            support_reason="adapter_cache_supported_false",
            timing_kind="unsupported",
        )
        return base
    base["order_seed"] = certification._expected_order_seed(
        profile,
        case_id,
        thread_budget,
        process_repetitions=process_repetitions,
    )
    for repetition in range(process_repetitions):
        implementations = {
            name: implementation_record(
                architecture,
                coordinate,
                case_id,
                thread_budget,
                name,
                repetition,
            )
            for name in ("reference", "candidate")
        }
        base["pairs"].append(
            {
                "pair_id": (
                    f"{architecture}/{build}/{profile}/{case_id}/t{thread_budget}/r{repetition}"
                ),
                "repetition": repetition,
                "order": certification._randomized_orders(
                    base["order_seed"], repetitions=process_repetitions
                )[repetition],
                "input_sha256": digest(f"input/{coordinate}"),
                "logical_input_sha256": digest(f"logical-input/{coordinate}"),
                "messages_sha256": digest(f"messages/{coordinate}"),
                "conformance_witness_sha256": digest(f"witness/{coordinate}"),
                "implementations": implementations,
            }
        )
    return base


def capture(architecture: str, build: str, identity: dict[str, str]) -> dict[str, Any]:
    wheel_sha = digest(f"{architecture}/{build}/wheel")
    native_sha = digest(f"{architecture}/{build}/native")
    phase_report = {
        "status": "pass",
        "report_sha256": digest(f"{architecture}/{build}/phase-c"),
        "source_sha256": identity["source_sha256"],
        "wheel_sha256": wheel_sha,
        "native_sha256": native_sha,
        "assets_sha256": identity["assets_sha256"],
        "profiles": list(certification.PROFILES),
        "failed_cases": 0,
        "skipped_cases": 0,
    }
    observations = [
        observation(architecture, build, profile, case_id, thread_budget)
        for profile, case_id, thread_budget in sorted(certification._expected_coordinates())
    ]
    is_arm = architecture == "arm64"
    return {
        "capture_id": f"{architecture}-{build}",
        "architecture": architecture,
        "identity": copy.deepcopy(identity),
        "build": {
            "label": build,
            "optimization": "portable-release" if build == "shipping" else "target-cpu=native",
            "wheel_sha256": wheel_sha,
            "native_sha256": native_sha,
            "build_command": f"build {architecture} {build}",
            "rustflags": [] if build == "shipping" else ["-C", "target-cpu=native"],
        },
        "host": {
            "host_fingerprint": digest(f"host/{architecture}"),
            "baseline": "current-m4" if is_arm else "controlled-linux-x86",
            "system": "Darwin" if is_arm else "Linux",
            "machine": architecture,
            "cpu_model": "Apple M4" if is_arm else "AMD EPYC controlled fixture",
            "physical_cpu_count": 16,
            "logical_cpu_count": 16,
            "memory_bytes": 64_000_000_000,
            "controlled": True,
            "provider": "local" if is_arm else "modal",
            "instance_type": "m4" if is_arm else "cpu-16",
            "allocation_id": f"allocation-{architecture}",
            "affinity_available": not is_arm,
            "affinity_source": "taskset" if not is_arm else "macos-unavailable",
            "affinity_unavailable_reason": (
                "macOS has no supported process CPU affinity API" if is_arm else None
            ),
            "allocated_cpu_count": 16,
            "allocated_memory_bytes": 64_000_000_000,
            "cgroup_sha256": digest(f"cgroup/{architecture}"),
            "physical_core_topology_sha256": digest(f"topology/{architecture}"),
            "physical_core_masks": {
                f"t{budget}": None if is_arm else list(range(budget))
                for budget in certification.THREAD_BUDGETS
            },
        },
        "toolchain": {
            "rustc": "rustc fixture",
            "cargo": "cargo fixture",
            "python": "3.11.fixture",
            "maturin": "1.14.1",
            "os_release": "fixture",
            "environment_sha256": digest(f"environment/{architecture}"),
        },
        "environment": {"PYTHONHASHSEED": "0", "LC_ALL": "C", "TZ": "UTC"},
        "commands": [f"run {architecture} {build}"],
        "raw_archive_sha256": digest(f"archive/{architecture}"),
        "raw_archive_bytes": 1_000_000,
        "raw_manifest_sha256": digest(f"manifest/{architecture}"),
        "raw_provenance_sha256": digest(f"provenance/{architecture}"),
        "protocol": protocol(certification.RANDOM_SEED),
        "phase_c": {
            "before": phase_report,
            "after": {
                **copy.deepcopy(phase_report),
                "report_sha256": digest(f"{architecture}/{build}/phase-c-after"),
            },
        },
        "observations": observations,
    }


def captures(identity: dict[str, str]) -> list[dict[str, Any]]:
    return [
        capture(architecture, build, identity)
        for architecture in certification.ARCHITECTURES
        for build in certification.BUILDS
    ]


def compact_captures(identity: dict[str, str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for architecture in certification.ARCHITECTURES:
        value = capture(architecture, "shipping", identity)
        value["protocol"] = {
            **protocol(certification.RANDOM_SEED),
            "process_repetitions": 3,
            "cases": list(certification.COMPACT_CASES),
            "thread_budgets": list(certification.COMPACT_THREAD_BUDGETS),
            "cache_policy": "cache workloads not selected in compact matrix",
        }
        value["observations"] = [
            observation(
                architecture,
                "shipping",
                profile,
                case_id,
                thread_budget,
                process_repetitions=3,
            )
            for profile, case_id, thread_budget in sorted(
                certification._expected_coordinates(process_repetitions=3)
            )
        ]
        output.append(value)
    return output


class PerformanceCertificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.identity = current_identity()
        cls.artifact = build_certification(
            captures(cls.identity),
            current_identity=cls.identity,
            created_at="2026-08-03T00:00:00Z",
        )

    def assert_rejected(self, artifact: dict[str, Any]) -> None:
        with self.assertRaises(PerformanceCertificationError):
            validate_certification(artifact, current_identity=self.identity)

    def test_valid_closed_artifact_passes_schema_and_renders_scope(self) -> None:
        validate_certification(self.artifact, current_identity=self.identity)
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(self.artifact, schema)
        report = render_report(self.artifact, current_identity=self.identity)
        self.assertIn("Status: `PASS`", report)
        self.assertIn("no video or vLLM production claim", report)
        self.assertIn("unsupported/non-gating", report)
        self.assertEqual(report.count("## Headline shipping results"), 1)
        self.assertFalse(self.artifact["claims"]["architectures_aggregated"])
        self.assertFalse(self.artifact["claims"]["native_build_gating"])
        measured = next(summary for summary in self.artifact["summaries"] if "memory" in summary)
        self.assertEqual(measured["memory"]["candidate"]["scoped_peak_rss_bytes"], 1_200)
        self.assertFalse(
            any(
                dtype == "float32" and full_image and layout in {"HWC", "CHW"}
                for dtype, layout, full_image, _buffer_class in certification.BUFFER_SEMANTICS.values()
            )
        )

    def test_bootstrap_is_deterministic_and_uses_twenty_thousand_resamples(self) -> None:
        first = certification._bootstrap([1.9, 2.0, 2.1, 2.2, 2.3], seed=71)
        second = certification._bootstrap([1.9, 2.0, 2.1, 2.2, 2.3], seed=71)
        self.assertEqual(first, second)
        self.assertEqual(first["resamples"], 20_000)

    def test_compact_shipping_archives_validate_and_report_every_selected_gate(self) -> None:
        artifact = build_certification(
            compact_captures(self.identity),
            current_identity=self.identity,
            created_at="2026-08-21T00:00:00Z",
        )
        validate_certification(artifact, current_identity=self.identity)
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.validate(artifact, schema)
        self.assertEqual(
            [capture["capture_id"] for capture in artifact["captures"]],
            ["arm64-shipping", "x86_64-shipping"],
        )
        gate_ids = {gate["gate_id"] for gate in artifact["gates"]}
        self.assertIn("speed/arm64/qwen3-vl-8b/image1/t1", gate_ids)
        self.assertIn("regression/arm64/qwen3-vl-8b/text_long/t1", gate_ids)
        self.assertIn("efficiency/x86_64/qwen3.5-9b/image24/t8", gate_ids)
        self.assertIn("memory/x86_64/qwen3.5-9b/images_16/t8", gate_ids)
        self.assertFalse(any("/native/" in gate_id for gate_id in gate_ids))
        report = render_report(artifact, current_identity=self.identity)
        self.assertIn("compact shipping-only matrix", report)
        self.assertIn("Certification missed", report)

    def test_noise_miss_is_reported_after_the_complete_matrix(self) -> None:
        noisy_captures = captures(self.identity)
        native = next(
            capture for capture in noisy_captures if capture["capture_id"] == "arm64-native"
        )
        measured = next(
            observation
            for observation in native["observations"]
            if observation["profile"] == "qwen3-vl-8b"
            and observation["case_id"] == "text_short"
            and observation["thread_budget"] == 1
        )
        for pair, wall in zip(measured["pairs"], [160.0, 180.0, 200.0, 220.0, 240.0], strict=True):
            implementation = pair["implementations"]["reference"]
            cpu = wall * 0.8
            for sample in implementation["samples"]:
                sample.update(wall_ms=wall, cpu_ms=cpu)
            raw_elapsed = len(implementation["samples"]) * wall
            supplemental_iterations = max(0, int((5_000 - raw_elapsed + wall - 1) // wall))
            implementation["timing_floor"].update(
                raw_sample_elapsed_wall_ms=raw_elapsed,
                supplemental_iteration_count=supplemental_iterations,
                supplemental_elapsed_wall_ms=supplemental_iterations * wall,
                supplemental_elapsed_cpu_ms=supplemental_iterations * cpu,
                total_iteration_count=len(implementation["samples"]) + supplemental_iterations,
                total_elapsed_wall_ms=raw_elapsed + supplemental_iterations * wall,
            )

        artifact = build_certification(
            noisy_captures,
            current_identity=self.identity,
            created_at="2026-08-03T00:00:00Z",
        )
        validate_certification(artifact, current_identity=self.identity)
        gate = next(
            gate
            for gate in artifact["gates"]
            if gate["gate_id"] == "noise/arm64/native/qwen3-vl-8b/text_short/t1/reference"
        )
        self.assertEqual(gate["status"], "miss")
        self.assertEqual(artifact["certification_status"], "miss")
        self.assertFalse(artifact["releasable"])
        self.assertIn(gate["gate_id"], render_report(artifact, current_identity=self.identity))

    def test_rejects_nonfinite_bool_missing_and_forged_values(self) -> None:
        mutations = {
            "nan": lambda value: value["captures"][0]["observations"][0]["pairs"][0][
                "implementations"
            ]["reference"]["samples"][0].update(wall_ms=float("nan")),
            "bool": lambda value: value["captures"][0]["observations"][0]["pairs"][0][
                "implementations"
            ]["reference"]["samples"][0].update(cpu_ms=True),
            "missing": lambda value: value["captures"][0]["host"].pop("cpu_model"),
            "forged-derived": lambda value: value["summaries"][0].update(speedup_p50=999.0),
            "extra": lambda value: value.update(unversioned_escape_hatch=True),
            "raw-archive": lambda value: value["captures"][0].update(raw_archive_sha256="0" * 64),
            "physical-mask": lambda value: value["captures"][2]["host"][
                "physical_core_masks"
            ].update(t8=list(range(1, 9))),
            "timing-floor": lambda value: value["captures"][0]["observations"][0]["pairs"][0][
                "implementations"
            ]["reference"]["timing_floor"].update(total_elapsed_wall_ms=9_999.0),
            "extra-sample": lambda value: value["captures"][0]["observations"][0]["pairs"][0][
                "implementations"
            ]["reference"]["samples"].append(
                {
                    **value["captures"][0]["observations"][0]["pairs"][0]["implementations"][
                        "reference"
                    ]["samples"][-1],
                    "sequence": 30,
                }
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                tampered = copy.deepcopy(self.artifact)
                mutate(tampered)
                self.assert_rejected(tampered)

    def test_rejects_cross_architecture_drift_and_stale_hashes(self) -> None:
        drifted = copy.deepcopy(self.artifact)
        x86 = next(
            capture for capture in drifted["captures"] if capture["capture_id"] == "x86_64-shipping"
        )
        measured = next(
            observation
            for observation in x86["observations"]
            if observation["case_id"] == "image24" and observation["thread_budget"] == 8
        )
        for pair in measured["pairs"]:
            pair["input_sha256"] = digest("drifted-input")
            pair["logical_input_sha256"] = digest("drifted-logical-input")
            for implementation in pair["implementations"].values():
                for position in ("pre_conformance", "post_conformance"):
                    implementation[position]["input_sha256"] = digest("drifted-input")
        self.assert_rejected(drifted)

        stale_identity = copy.deepcopy(self.identity)
        stale_identity["workload_sha256"] = digest("new-workload")
        with self.assertRaises(PerformanceCertificationError):
            validate_certification(self.artifact, current_identity=stale_identity)

    def test_rejects_cached_relabeling_build_alias_and_forbidden_candidate_buffer(self) -> None:
        cached = copy.deepcopy(self.artifact)
        cached_observation = next(
            observation
            for observation in cached["captures"][0]["observations"]
            if observation["case_id"] == "repeat24_cached"
        )
        source_observation = next(
            observation
            for observation in cached["captures"][0]["observations"]
            if observation["case_id"] == "repeat24_uncached"
            and observation["thread_budget"] == cached_observation["thread_budget"]
            and observation["profile"] == cached_observation["profile"]
        )
        cached_observation.update(
            support_status="supported",
            support_reason=None,
            timing_kind="uncached",
            pairs=copy.deepcopy(source_observation["pairs"]),
        )
        self.assert_rejected(cached)

        aliased = copy.deepcopy(self.artifact)
        arm_shipping = next(
            capture for capture in aliased["captures"] if capture["capture_id"] == "arm64-shipping"
        )
        arm_native = next(
            capture for capture in aliased["captures"] if capture["capture_id"] == "arm64-native"
        )
        arm_native["build"]["wheel_sha256"] = arm_shipping["build"]["wheel_sha256"]
        arm_native["phase_c"]["before"]["wheel_sha256"] = arm_shipping["build"]["wheel_sha256"]
        arm_native["phase_c"]["after"]["wheel_sha256"] = arm_shipping["build"]["wheel_sha256"]
        self.assert_rejected(aliased)

        forbidden = copy.deepcopy(self.artifact)
        measured_observation = next(
            observation
            for observation in forbidden["captures"][0]["observations"]
            if observation["support_status"] == "supported"
        )
        buffer = measured_observation["pairs"][0]["implementations"]["candidate"]["memory"][
            "buffers"
        ][0]
        buffer["name"] = "forged_full_float32_hwc"
        self.assert_rejected(forbidden)


if __name__ == "__main__":
    unittest.main()
