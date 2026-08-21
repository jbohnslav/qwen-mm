from __future__ import annotations

import copy
import hashlib
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
REFERENCE_SOURCE = SCRIPTS.parent / "reference" / "src"
for import_path in (SCRIPTS, REFERENCE_SOURCE):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import d4_evidence as evidence  # noqa: E402
import qwen_mm_reference.performance_certification_v1 as certification  # noqa: E402


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def asset_identity() -> dict[str, object]:
    return {
        "schema_id": "qwen-mm-logical-directory-identity-v2",
        "schema_version": 2,
        "tree_sha256": digest("assets"),
        "entry_count": 1,
        "logical_bytes": 1,
        "entries": [{"path": "asset", "bytes": 1, "sha256": digest("asset")}],
    }


def affinity_enforcement() -> dict[str, object]:
    def phase(native_id_offset: int) -> dict[str, object]:
        return {
            "probes": [
                {
                    "native_thread_count": thread_count,
                    "wall_seconds": 1.0,
                    "process_cpu_seconds": 1.0,
                    "process_cpu_wall_ratio": 1.0,
                    "worker_reports": [
                        {
                            "worker_index": worker_index,
                            "native_thread_id": native_id_offset + worker_index,
                            "affinity": [0],
                            "observed_cpus": [0],
                            "iterations": 100,
                        }
                        for worker_index in range(thread_count)
                    ],
                }
                for thread_count in (1, 2, 4)
            ]
        }

    return {
        "method": "taskset-t1-python-native-threads-v1",
        "expected_affinity": [0],
        "probe_seconds": 1.0,
        "cpu_wall_ratio_max": 1.25,
        "pre": phase(100),
        "post": phase(200),
    }


def provenance(architecture: str, *, provider: str = "modal") -> dict[str, object]:
    capture_inputs = {
        path: {"bytes": index + 1, "sha256": digest(path)}
        for index, path in enumerate(evidence.CAPTURE_INPUT_PATHS)
    }
    common: dict[str, object] = {
        "schema_id": "qwen-mm-d4-raw-capture-provenance-v1",
        "schema_version": 1,
        "claim": "raw controlled-host input for the separate D4 certification evaluator",
        "architecture_family": architecture,
        "started_at": "2026-08-03T00:00:00Z",
        "completed_at": "2026-08-03T01:00:00Z",
        "source_revision": "a" * 40,
        "source": {"sha256": digest("source"), "file_count": 10, "bytes": 100},
        "assets": asset_identity(),
        "capture_inputs": capture_inputs,
        "build_wheel_sha256": {"shipping": digest("sw"), "native": digest("nw")},
        "build_native_sha256": {"shipping": digest("sn"), "native": digest("nn")},
        "toolchain_pins": evidence.toolchain_pins(),
        "sample_pruning": "forbidden",
        "noise_cv_max": 0.05,
        "environment": {
            "PYTHONHASHSEED": "0",
            "LC_ALL": "C",
            "TZ": "UTC",
            "UV_DEFAULT_INDEX": "https://pypi.org/simple",
            "removed_package_index_variables": [],
        },
    }
    if architecture == "arm64":
        common["host_label"] = "local-m4"
        common["host"] = {
            "system": "Darwin",
            "machine": "arm64",
            "platform": "macOS",
            "uname": "Darwin fixture",
            "hostname": "m4-host",
            "cpu_model": "Apple M4",
            "physical_cpu_count": 10,
            "logical_cpu_count": 10,
            "memory_bytes": 32 * 1024**3,
            "affinity": {
                "mode": "unavailable",
                "reason": "macOS has no supported taskset/sched_setaffinity equivalent",
                "masks": {f"t{budget}": None for budget in evidence.THREAD_BUDGETS},
            },
        }
    else:
        common["provider"] = provider
        if provider == "modal":
            common["modal"] = {
                "client_version": "1.4.0",
                "environment": {"MODAL_TASK_ID": "task-1"},
            }
        else:
            common["toolchain_pins"] = evidence.local_linux_toolchain_pins()
            common["local_linux"] = {
                "host_label": "dedicated-linux-fixture",
                "allocation_id": "capture-1",
            }
        common["host"] = {
            "system": "Linux",
            "machine": "x86_64",
            "platform": "Linux fixture",
            "uname": "Linux fixture",
            "hostname": "modal-host",
            "cpu_description": "AMD EPYC",
            "lscpu": {"lscpu": [{"field": "Model name:", "data": "AMD EPYC"}]},
            "lscpu_parse": "\n".join(
                ["# CPU,Core,Socket,Online"] + [f"{cpu},{cpu},0,Y" for cpu in range(16)]
            ),
            "logical_cpu_count": 16,
            "proc_meminfo_total_bytes": 32 * 1024**3,
            "resource_attestation": {
                "mode": "cgroup_v2_limits",
                "requested_resources_bound_by": "source-authenticated Modal function decorator",
                "requested_physical_cores": 16,
                "requested_memory_mib": 32768,
                "nonpreemptible": True,
                "single_use_container": True,
                "visible_affinity": list(range(16)),
                "physical_core_masks": {
                    f"t{budget}": list(range(budget)) for budget in evidence.THREAD_BUDGETS
                },
                "cgroup_limits": {
                    "/sys/fs/cgroup/cpu.max": "1600000 100000",
                    "/sys/fs/cgroup/cpuset.cpus.effective": "0-15",
                    "/sys/fs/cgroup/memory.max": str(32 * 1024**3),
                },
            },
        }
        if provider == "local_linux":
            common["host"]["hostname"] = "local-linux-host"
            common["host"]["resource_attestation"] = {
                "mode": "cgroup_v2_cpuset",
                "allocated_physical_cores": 16,
                "allocated_memory_bytes": 32 * 1024**3,
                "power_policy": {
                    "paths": {
                        f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/{name}": "performance"
                        for cpu in range(16)
                        for name in (
                            "scaling_governor",
                            "energy_performance_preference",
                            "scaling_min_freq",
                            "scaling_max_freq",
                        )
                    }
                    | {
                        "/sys/devices/system/cpu/intel_pstate/no_turbo": "1",
                        "/sys/devices/system/cpu/cpufreq/boost": "0",
                    },
                    "unavailable": {},
                },
                "affinity_enforcement": affinity_enforcement(),
                "exclusive_physical_cores": False,
                "dedicated_capture": True,
                "stable": True,
                "visible_affinity": list(range(16)),
                "physical_core_masks": {
                    f"t{budget}": list(range(budget)) for budget in evidence.THREAD_BUDGETS
                },
                "cgroup_limits": {
                    "/sys/fs/cgroup/cpu.max": "1600000 100000",
                    "/sys/fs/cgroup/cpuset.cpus.effective": "0-15",
                    "/sys/fs/cgroup/memory.max": str(32 * 1024**3),
                },
            }
    return common


def output_signature(image: bool = True) -> dict[str, dict[str, object]]:
    keys = ["input_ids", "attention_mask", "mm_token_type_ids"]
    if image:
        keys += ["pixel_values", "image_grid_thw"]
    return {
        key: {
            "dtype": "float32" if key == "pixel_values" else "int64",
            "shape": [1],
            "strides": [4 if key == "pixel_values" else 8],
            "nbytes": 4 if key == "pixel_values" else 8,
            "sha256": digest(key),
        }
        for key in keys
    }


def worker(candidate: bool = True) -> dict[str, object]:
    scope = {
        "request_index": 0,
        "message_index": 0,
        "content_item_index": 0,
        "media_index": 0,
        "input_index": 0,
    }
    retained_scope = {name: None for name in scope}
    buffers = [
        {
            "sequence": 0,
            "name": "prepared_rgb",
            "class": "transient",
            "scope": scope,
            "bytes": 100,
            "allocated_at_ns": 10,
            "released_at_ns": 50,
        },
        {
            "sequence": 1,
            "name": "pixel_values",
            "class": "retained_output",
            "scope": retained_scope,
            "bytes": 100,
            "allocated_at_ns": 60,
            "released_at_ns": None,
        },
    ]
    native = (
        {
            "schema_version": "qwen-mm-observation-v1",
            "outcome": "success",
            "dropped_events": 0,
            "duration_ns": 100,
            "counter_scope": "one operation",
            "allocations": {
                "allocation_count": 2,
                "allocated_bytes": 200,
                "copy_count": 0,
                "copied_bytes": 0,
                "transient_live_bytes": 0,
                "peak_transient_live_bytes": 100,
                "retained_final_output_bytes": 100,
            },
            "buffers": {
                "count": 2,
                "total_bytes": 200,
                "bytes_by_class": {"transient": 100, "retained_output": 100},
                "records": buffers,
            },
            "copies": [],
            "calls": {
                "public_python_calls": 1,
                "native_batch_calls": 1,
                "native_visual_calls": 1,
                "python_callbacks": 0,
                "hugging_face_calls": 0,
                "qwen_vl_utils_calls": 0,
                "pillow_calls": 0,
                "torchvision_calls": 0,
            },
        }
        if candidate
        else None
    )
    wall = 200.0
    samples = [
        {
            "sequence": index,
            "wall_ms": wall,
            "cpu_ms": wall * 0.8,
            "throughput_per_s": 5.0,
            "core_utilization": 0.8,
        }
        for index in range(30)
    ]
    return {
        "input_fingerprint": digest("exact-input"),
        "logical_input_fingerprint": digest("logical-input"),
        "messages_fingerprint": digest("messages"),
        "worker_nonce": digest(f"nonce/{candidate}"),
        "process_id": 101 if candidate else 100,
        "thread_settings": {
            "budget": 1,
            "environment": {name: "1" for name in certification.THREAD_ENVIRONMENT_NAMES},
            "torch": {"num_threads": 1, "num_interop_threads": 1},
            "total_budget_model": {
                "total_budget": 1,
                "owner": "qwen_mm_processor_pool" if candidate else "official_torch_intraop",
                "inactive_runtime_budget": 1,
                "environment": {name: "1" for name in certification.THREAD_ENVIRONMENT_NAMES},
                "torch_thread_budget": 1,
            },
        },
        "affinity": {
            "requested_cpus": None,
            "status": "unavailable",
            "mechanism": None,
            "observed_cpus": None,
            "reason": "macOS unsupported",
        },
        "samples": samples,
        "timing_floor": {
            "required_seconds": 5.0,
            "raw_sample_iteration_count": 30,
            "raw_sample_elapsed_wall_ms": 6000.0,
            "supplemental_iteration_count": 0,
            "supplemental_elapsed_wall_ms": 0.0,
            "supplemental_elapsed_cpu_ms": 0.0,
            "total_iteration_count": 30,
            "total_elapsed_wall_ms": 6000.0,
        },
        "resource_census": {
            "rss": {
                "baseline_rss_bytes": 1000,
                "rss_after_bytes": 1050,
                "peak_rss_bytes": 1200 if candidate else 1300,
                "external_transient_rss_bytes": 100 if candidate else 200,
            },
            "output_bytes": 100,
            "native_observed": native,
            "adapter_metrics": {"cache_supported": False},
        },
        "output_signature": output_signature(),
        "conformance": {
            "pre_measurement": "pass",
            "post_measurement": "pass",
            "all_measured_iterations_stable": True,
            "float_atol": certification.FLOAT_ATOL["image24"],
        },
    }


class D4EvidenceTests(unittest.TestCase):
    def test_compact_protocol_is_accepted_by_certification_evaluator(self) -> None:
        validated = certification._validate_protocol(evidence.PROTOCOL, "compact.protocol")
        self.assertEqual(validated["process_repetitions"], 3)
        self.assertEqual(validated["thread_budgets"], [1, 8])

    def test_identity_authenticates_cross_host_contracts_and_rejects_drift(self) -> None:
        arm = provenance("arm64")
        x86 = provenance("x86_64")
        captured_inputs = arm["capture_inputs"]
        with (
            mock.patch.object(evidence, "_assert_current_source"),
            mock.patch.object(
                evidence,
                "_git_input_identity",
                side_effect=lambda _revision, path: captured_inputs[path],
            ),
            mock.patch.object(evidence, "assert_assets_identity"),
        ):
            identity = evidence._identity([arm, x86])
            self.assertEqual(identity["source_revision"], "a" * 40)
            self.assertEqual(identity["assets_sha256"], digest("assets"))

            tampered = copy.deepcopy(x86)
            tampered["capture_inputs"]["benchmarks/workloads-v2.json"]["sha256"] = digest(
                "tampered"
            )
            with self.assertRaisesRegex(evidence.D4EvidenceError, "capture_inputs drifted"):
                evidence._identity([arm, tampered])

            missing = copy.deepcopy(arm)
            missing["capture_inputs"].pop("reference/models.json")
            with self.assertRaisesRegex(evidence.D4EvidenceError, "closed shape"):
                evidence._identity([missing, missing])

    def test_host_preserves_physical_masks_and_rejects_cross_architecture_relabel(self) -> None:
        x86 = provenance("x86_64")
        host = evidence._host(x86, "x86_64")
        self.assertEqual(host["physical_core_masks"]["t8"], list(range(8)))
        self.assertTrue(host["affinity_available"])
        self.assertEqual(host["allocation_id"], "task-1")

        relabeled = copy.deepcopy(x86)
        relabeled["host"]["machine"] = "arm64"
        with self.assertRaisesRegex(evidence.D4EvidenceError, "not native Linux"):
            evidence._host(relabeled, "x86_64")

    def test_local_linux_provider_has_authenticated_controls_without_modal_fields(self) -> None:
        x86 = provenance("x86_64", provider="local_linux")
        validated = evidence._validate_provenance(x86, "x86_64")
        host = evidence._host(validated, "x86_64")
        self.assertEqual(host["provider"], "local_linux")
        self.assertEqual(host["instance_type"], "dedicated-linux-fixture")
        self.assertEqual(host["allocation_id"], "capture-1")
        self.assertEqual(host["allocated_cpu_count"], 16)
        self.assertTrue(host["affinity_available"])
        self.assertNotIn("modal", validated)

        forged_modal = copy.deepcopy(x86)
        forged_modal["modal"] = {"client_version": "test", "environment": {}}
        with self.assertRaisesRegex(evidence.D4EvidenceError, "closed shape"):
            evidence._validate_provenance(forged_modal, "x86_64")

        unstable = copy.deepcopy(x86)
        unstable["host"]["resource_attestation"]["stable"] = False
        with self.assertRaisesRegex(ValueError, "stable host controls"):
            evidence._host(unstable, "x86_64")

        false_exclusivity = copy.deepcopy(x86)
        false_exclusivity["host"]["resource_attestation"]["exclusive_physical_cores"] = True
        with self.assertRaisesRegex(ValueError, "cannot claim exclusive"):
            evidence._host(false_exclusivity, "x86_64")

        over_budget = copy.deepcopy(x86)
        over_budget_probe = over_budget["host"]["resource_attestation"]["affinity_enforcement"][
            "post"
        ]["probes"][2]
        over_budget_probe["process_cpu_seconds"] = 1.3
        over_budget_probe["process_cpu_wall_ratio"] = 1.3
        with self.assertRaisesRegex(ValueError, "exceeded CPU/wall limit"):
            evidence._host(over_budget, "x86_64")

    def test_worker_transform_preserves_samples_floor_and_native_census(self) -> None:
        raw = worker(candidate=True)
        transformed = evidence._implementation(
            raw,
            candidate=True,
            witness=digest("witness"),
            input_sha=digest("input"),
            messages_sha=digest("messages"),
        )
        certification._validate_implementation(
            transformed,
            "candidate",
            thread_budget=1,
            affinity_available=False,
            expected_affinity=None,
            candidate=True,
            case_id="image24",
        )
        self.assertEqual(len(transformed["samples"]), 30)
        self.assertEqual(transformed["timing_floor"]["total_iteration_count"], 30)
        self.assertEqual(transformed["memory"]["allocated_bytes"], 200)
        self.assertEqual(
            transformed["memory"]["buffers"],
            raw["resource_census"]["native_observed"]["buffers"]["records"],
        )

        official_with_invented_counters = worker(candidate=False)
        official_with_invented_counters["resource_census"]["native_observed"] = raw[
            "resource_census"
        ]["native_observed"]
        with self.assertRaisesRegex(evidence.D4EvidenceError, "fabricated"):
            evidence._memory(official_with_invented_counters, candidate=False)

    def test_pair_keeps_exact_logical_and_messages_identities_distinct(self) -> None:
        raw_pair = {
            "profile_alias": "qwen3-vl-8b",
            "case_id": "image24",
            "build_label": "shipping",
            "thread_budget": 1,
            "thread_regime": "t1",
            "work_units": 1,
            "cache_mode": "disabled",
            "repetition": 0,
            "order": ["reference", "candidate"],
            "implementations": {
                "reference": worker(candidate=False),
                "candidate": worker(candidate=True),
            },
        }
        transformed = evidence._pair(
            raw_pair,
            architecture="arm64",
            build="shipping",
            profile="qwen3-vl-8b",
            case_id="image24",
            budget=1,
        )
        self.assertEqual(transformed["input_sha256"], digest("exact-input"))
        self.assertEqual(transformed["logical_input_sha256"], digest("logical-input"))
        self.assertEqual(transformed["messages_sha256"], digest("messages"))
        self.assertEqual(
            len(
                {
                    transformed[name]
                    for name in (
                        "input_sha256",
                        "logical_input_sha256",
                        "messages_sha256",
                    )
                }
            ),
            3,
        )

    def test_retained_wheel_reconciles_exact_archive_bytes_and_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheel_path = root / "qwen_mm-test.whl"
            package = b"package-bytes"
            native = b"native-bytes"
            benchmark = b"def create_adapter(): pass\n"
            with zipfile.ZipFile(wheel_path, "w") as archive:
                archive.writestr("qwen_mm/__init__.py", package)
                archive.writestr("qwen_mm/_native.test.so", native)
                archive.writestr("qwen_mm/benchmark.py", benchmark)
            wheel_bytes = wheel_path.read_bytes()
            contents = evidence.wheel_contents_identity(wheel_path)
            runtime = {
                "package_version": "0.1.0",
                "package_origin": "/venv/qwen_mm/__init__.py",
                "native_origin": "/venv/qwen_mm/_native.test.so",
                "native_bytes": len(native),
                "native_sha256": hashlib.sha256(native).hexdigest(),
            }
            raw_build = {
                "identity": {
                    "wheel": {
                        "name": wheel_path.name,
                        "bytes": len(wheel_bytes),
                        "sha256": hashlib.sha256(wheel_bytes).hexdigest(),
                        "contents": contents,
                    }
                },
                "runtime": runtime,
                "runtime_reconciliation": {
                    "verified": True,
                    "wheel_contents": contents,
                    "installed_runtime": runtime,
                },
            }
            files = {f"builds/shipping/{wheel_path.name}": wheel_bytes}
            build = {"native_sha256": hashlib.sha256(native).hexdigest()}
            retained = evidence._retained_wheel(
                files,
                raw_build,
                build,
                architecture="x86_64",
                build_label="shipping",
                temporary=root,
            )
            self.assertEqual(
                retained,
                {
                    **contents,
                    "benchmark_artifact": {
                        "member": "qwen_mm/benchmark.py",
                        "bytes": len(benchmark),
                        "sha256": hashlib.sha256(benchmark).hexdigest(),
                    },
                },
            )
            files["builds/shipping/foreign.whl"] = wheel_bytes
            with self.assertRaisesRegex(evidence.D4EvidenceError, "exactly its one"):
                evidence._retained_wheel(
                    files,
                    raw_build,
                    build,
                    architecture="x86_64",
                    build_label="shipping",
                    temporary=root,
                )

    def test_observation_inventory_rejects_missing_coordinates(self) -> None:
        results = {budget: {"pairs": []} for budget in evidence.THREAD_BUDGETS}
        files = {
            "captures/shipping/repeat24_cached-unsupported.json": evidence._canonical(
                {"coordinates": []}
            )
        }
        with self.assertRaisesRegex(evidence.D4EvidenceError, "cached unsupported"):
            evidence._observations(
                files,
                architecture="arm64",
                build="shipping",
                results=results,
            )

    def test_compact_observations_require_exact_selected_three_pair_matrix(self) -> None:
        results: dict[int, dict[str, list[dict[str, object]]]] = {}
        for budget in evidence.COMPACT_THREAD_BUDGETS:
            pairs: list[dict[str, object]] = []
            for profile in evidence.PROFILES:
                for case_id in evidence.COMPACT_CASES_BY_THREAD_BUDGET[budget]:
                    for repetition in range(evidence.COMPACT_PROCESS_REPETITIONS):
                        pairs.append(
                            {
                                "profile_alias": profile,
                                "case_id": case_id,
                                "order_seed": 20260731,
                                "repetition": repetition,
                            }
                        )
            results[budget] = {"pairs": pairs}

        with mock.patch.object(evidence, "_pair", side_effect=lambda pair, **_kwargs: pair):
            observations = evidence._observations(
                {},
                architecture="arm64",
                build="shipping",
                results=results,
                suite="compact",
            )

        self.assertEqual(len(observations), 14)
        self.assertEqual(sum(len(item["pairs"]) for item in observations), 42)
        self.assertTrue(all(item["support_status"] == "supported" for item in observations))

        del results[1]["pairs"][-evidence.COMPACT_PROCESS_REPETITIONS :]
        with self.assertRaisesRegex(evidence.D4EvidenceError, "compact raw benchmark matrix"):
            evidence._observations(
                {},
                architecture="arm64",
                build="shipping",
                results=results,
                suite="compact",
            )

    def test_validate_rebuilds_archives_and_rejects_forged_artifact(self) -> None:
        rebuilt = {
            "created_at": "2026-08-03T00:00:00Z",
            "current_identity": {"source_revision": "a" * 40},
            "value": "authentic",
        }
        with (
            mock.patch.object(evidence, "rebuild_from_archives", return_value=rebuilt) as rebuild,
            mock.patch.object(evidence, "validate_certification"),
        ):
            evidence.validate_from_archives(rebuilt, Path("arm.zip"), Path("x86.zip"))
            rebuild.assert_called_once_with(
                Path("arm.zip"), Path("x86.zip"), created_at=rebuilt["created_at"]
            )
            forged = {**rebuilt, "value": "forged"}
            with self.assertRaisesRegex(evidence.D4EvidenceError, "differs"):
                evidence.validate_from_archives(forged, Path("arm.zip"), Path("x86.zip"))

    def test_currentness_rejects_code_drift(self) -> None:
        source = {"sha256": digest("source"), "file_count": 1, "bytes": 1}

        def git(*arguments: str) -> str:
            if arguments[:2] == ("diff", "--name-only"):
                return "crates/qwen-mm-core/src/lib.rs"
            if arguments[:2] == ("status", "--porcelain=v1"):
                return ""
            if arguments[:2] == ("rev-parse", "--verify"):
                return "a" * 40
            return ""

        with (
            mock.patch.object(evidence, "_git", side_effect=git),
            mock.patch.object(
                evidence,
                "committed_source_tree_digest",
                return_value=(digest("source"), 1, 1),
            ),
        ):
            with self.assertRaisesRegex(evidence.D4EvidenceError, "changed after capture"):
                evidence._assert_current_source("a" * 40, source)


if __name__ == "__main__":
    unittest.main()
