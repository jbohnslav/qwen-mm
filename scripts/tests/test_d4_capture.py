from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import d4_capture_support as support  # noqa: E402
import d4_local  # noqa: E402
import d4_worker  # noqa: E402


def _base_files() -> dict[str, bytes]:
    files = {
        name: b"log\n" if name.endswith(".log") else b"{}\n"
        for name in support.REQUIRED_BASE_MEMBERS
    }
    index = {
        "build_labels": list(support.BUILD_LABELS),
        "thread_budgets": list(support.THREAD_BUDGETS),
        "files": sorted(set(files) - {"capture-index.json"}),
    }
    files["capture-index.json"] = (json.dumps(index) + "\n").encode()
    for label in support.BUILD_LABELS:
        files[f"captures/{label}/repeat24_cached-unsupported.json"] = (
            json.dumps(
                {
                    "schema_id": "qwen-mm-d4-cached-unsupported-v1",
                    "schema_version": 1,
                    "build_label": label,
                    "case_id": "repeat24_cached",
                    "cache_mode": "enabled",
                    "support_status": "unsupported",
                    "support_reason": "adapter_cache_supported_false",
                    "timing_kind": "unsupported",
                    "timed_pair_count": 0,
                    "timed_sample_count": 0,
                    "coordinates": [
                        {
                            "profile_alias": profile,
                            "thread_budget": budget,
                            "reference_cache_supported": False,
                            "candidate_cache_supported": False,
                        }
                        for profile in support.PROFILES
                        for budget in (1, support.PRODUCTION_THREAD_BUDGET)
                    ],
                }
            )
            + "\n"
        ).encode()
        for budget in support.THREAD_BUDGETS:
            files[f"captures/{label}/t{budget}/noise.json"] = (
                json.dumps(
                    {
                        "sample_pruning": "forbidden",
                        "pass": True,
                        "assessments": [
                            {
                                "pass": True,
                                "observation_count": 5,
                                "retained_observation_count": 5,
                                "pruned_observation_count": 0,
                            }
                        ],
                    }
                )
                + "\n"
            ).encode()
            files[f"captures/{label}/t{budget}/result.json"] = b'{"pairs": []}\n'
    index["files"] = sorted(set(files) - {"capture-index.json"})
    files["capture-index.json"] = (json.dumps(index) + "\n").encode()
    return files


class D4CaptureSupportTests(unittest.TestCase):
    def test_clean_venv_uses_exact_locked_reference_sync(self) -> None:
        venv = Path("/working/venvs/shipping")
        command = support.reference_sync_command(venv=venv)
        self.assertEqual(
            command,
            [
                "env",
                "VIRTUAL_ENV=/working/venvs/shipping",
                "uv",
                "sync",
                "--locked",
                "--package",
                "qwen-mm-reference",
                "--active",
                "--inexact",
                "--default-index",
                support.PYPI_INDEX,
            ],
        )
        plan = d4_local.build_plan(Path("/working"))
        for build in plan["builds"].values():
            self.assertNotIn("--system-site-packages", build["create_venv"])
            self.assertEqual(
                build["sync"], support.reference_sync_command(venv=Path(build["venv"]))
            )

    def test_capture_environment_normalizes_package_indexes(self) -> None:
        base = {
            "HOME": "/home/capture",
            "PATH": "/usr/bin",
            "PIP_INDEX_URL": "https://ambient.invalid/simple",
            "SECRET_TOKEN": "must-not-be-captured",
            "UV_EXTRA_INDEX_URL": "https://ambient.invalid/extra",
            "RUSTFLAGS": "-C target-cpu=ambient",
        }
        environment = support.normalized_capture_environment(base)
        self.assertEqual(environment["UV_DEFAULT_INDEX"], support.PYPI_INDEX)
        self.assertNotIn("RUSTFLAGS", environment)
        self.assertTrue(
            all(name not in environment for name in support.PYPI_OVERRIDE_ENVIRONMENT_NAMES)
        )
        evidence = support.build_environment_evidence(environment)
        self.assertEqual(evidence["set"]["UV_DEFAULT_INDEX"], support.PYPI_INDEX)
        self.assertNotIn("SECRET_TOKEN", evidence["set"])
        self.assertIn("RUSTFLAGS", evidence["unset"])

    def test_build_invariants_fail_if_foreign_host_toolchain_changes(self) -> None:
        invariant = {
            "build_environment": {"set": {"UV_DEFAULT_INDEX": support.PYPI_INDEX}},
            "build_host": {"uname": "Linux worker"},
            "packages": ["qwen-mm==0.1.0"],
            "toolchain": {
                "python": "Python 3.11.15",
                "maturin": "maturin 1.14.1",
                "rustc": "rustc 1.94.0",
                "cargo": "cargo 1.94.0",
            },
        }
        support.assert_build_invariants({label: invariant for label in support.BUILD_LABELS})
        with self.assertRaisesRegex(support.D4CaptureError, "toolchain"):
            support.assert_build_invariants(
                {
                    "shipping": invariant,
                    "native": {
                        **invariant,
                        "toolchain": {**invariant["toolchain"], "rustc": "rustc changed"},
                    },
                }
            )
        normalized = support.normalize_build_artifact_paths(
            "qwen-mm @ file:///tmp/capture/native/qwen_mm.whl",
            {Path("/tmp/capture/native"): "<retained-build>"},
        )
        self.assertEqual(normalized, "qwen-mm @ file://<retained-build>/qwen_mm.whl")

    def test_capture_inputs_include_all_certification_schemas(self) -> None:
        self.assertTrue(
            {
                "benchmarks/performance-certification-schema-v1.json",
                "benchmarks/profile-schema-v1.json",
                "benchmarks/result-schema-v2.json",
                "benchmarks/workload-schema-v2.json",
            }.issubset(support.CAPTURE_INPUT_PATHS)
        )

    def test_installed_native_must_match_retained_wheel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            wheel = Path(temporary) / "qwen_mm-0.1.0-test.whl"
            native = b"native-extension-bytes"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("qwen_mm/__init__.py", b"from . import _native\n")
                archive.writestr("qwen_mm/_native.test.so", native)
            runtime = {
                "native_bytes": len(native),
                "native_sha256": hashlib.sha256(native).hexdigest(),
            }
            result = support.reconcile_installed_runtime(wheel=wheel, runtime=runtime)
            self.assertTrue(result["verified"])
            self.assertEqual(
                result["wheel_contents"]["native_artifact_sha256"],
                runtime["native_sha256"],
            )
            with self.assertRaisesRegex(support.D4CaptureError, "differs"):
                support.reconcile_installed_runtime(
                    wheel=wheel,
                    runtime={**runtime, "native_sha256": "0" * 64},
                )

    def test_shipping_and_native_builds_are_distinct_and_locked(self) -> None:
        shipping = support.wheel_build_command(
            python=Path("/venv/shipping/bin/python"),
            output=Path("/wheels/shipping"),
            build_label="shipping",
            cargo_target_dir=Path("/cargo/shipping"),
        )
        native = support.wheel_build_command(
            python=Path("/venv/native/bin/python"),
            output=Path("/wheels/native"),
            build_label="native",
            cargo_target_dir=Path("/cargo/native"),
        )
        self.assertEqual(shipping[:2], ["env", "CARGO_TARGET_DIR=/cargo/shipping"])
        self.assertEqual(native[0], "env")
        self.assertIn("CARGO_TARGET_DIR=/cargo/native", native)
        self.assertIn("RUSTFLAGS=-C target-cpu=native", native)
        self.assertIn("--release", shipping)
        self.assertIn("--locked", shipping)
        self.assertNotIn("target-cpu=native", " ".join(shipping))

    def test_physical_core_masks_are_nested_and_skip_smt_siblings(self) -> None:
        topology = """# CPU,Core,Socket,Online
0,0,0,Y
8,0,0,Y
1,1,0,Y
9,1,0,Y
2,2,0,Y
10,2,0,Y
3,3,0,Y
11,3,0,Y
4,4,0,Y
12,4,0,Y
5,5,0,Y
13,5,0,Y
6,6,0,Y
14,6,0,Y
7,7,0,Y
15,7,0,Y
"""
        masks = support.physical_core_masks(topology, allowed_cpus=range(16))
        self.assertEqual(masks[1], (0,))
        self.assertEqual(masks[2], (0, 1))
        self.assertEqual(masks[4], (0, 1, 2, 3))
        self.assertEqual(masks[8], tuple(range(8)))

    def test_physical_core_masks_fail_when_allocation_is_too_small(self) -> None:
        topology = "\n".join(f"{cpu},{cpu},0,Y" for cpu in range(7))
        with self.assertRaisesRegex(support.D4CaptureError, "8 required"):
            support.physical_core_masks(topology, allowed_cpus=range(7))

    def test_local_affinity_is_explicitly_unavailable(self) -> None:
        provenance = support.local_affinity_provenance()
        self.assertEqual(provenance["mode"], "unavailable")
        self.assertTrue(all(value is None for value in provenance["masks"].values()))

    def test_modal_resource_attestation_requires_complete_reservation(self) -> None:
        masks = {budget: tuple(range(budget)) for budget in support.THREAD_BUDGETS}
        cgroups = {
            "/sys/fs/cgroup/cpu.max": "1600000 100000",
            "/sys/fs/cgroup/cpuset.cpus.effective": "0-15",
            "/sys/fs/cgroup/memory.max": str(32 * 1024 * 1024 * 1024),
        }
        result = support.modal_resource_attestation(
            cgroup_limits=cgroups,
            platform_text="Linux",
            uname="Linux",
            visible_affinity=list(range(16)),
            masks=masks,
        )
        self.assertEqual(result["mode"], support.CGROUP_ATTESTATION_MODE)
        undersized = dict(cgroups)
        undersized["/sys/fs/cgroup/cpu.max"] = "800000 100000"
        with self.assertRaisesRegex(support.D4CaptureError, "CPU quota"):
            support.modal_resource_attestation(
                cgroup_limits=undersized,
                platform_text="Linux",
                uname="Linux",
                visible_affinity=list(range(16)),
                masks=masks,
            )

    def test_noise_rule_keeps_every_observation(self) -> None:
        result = support.noise_assessment([100.0, 101.0, 99.0, 100.5, 99.5])
        self.assertTrue(result["pass"])
        self.assertEqual(result["observation_count"], 5)
        self.assertEqual(result["retained_observation_count"], 5)
        self.assertEqual(result["pruned_observation_count"], 0)
        noisy = support.noise_assessment([80.0, 90.0, 100.0, 110.0, 120.0])
        self.assertFalse(noisy["pass"])

    def test_capture_archive_round_trip_and_tamper_rejection(self) -> None:
        files = _base_files()
        archive = support.create_capture_archive(files)
        self.assertEqual(support.read_capture_archive(archive), files)
        malformed = dict(files)
        index = json.loads(malformed["capture-index.json"])
        index["thread_budgets"] = [1, 8]
        malformed["capture-index.json"] = json.dumps(index).encode()
        with self.assertRaisesRegex(support.D4CaptureError, "build/thread matrix"):
            support.read_capture_archive(support.create_capture_archive(malformed))

        cached_tamper = _base_files()
        cached_name = "captures/shipping/repeat24_cached-unsupported.json"
        cached = json.loads(cached_tamper[cached_name])
        cached["coordinates"][0]["candidate_cache_supported"] = True
        cached_tamper[cached_name] = json.dumps(cached).encode()
        with self.assertRaisesRegex(support.D4CaptureError, "zero timing"):
            support.read_capture_archive(support.create_capture_archive(cached_tamper))

    def test_cpu_list_parser_fails_closed(self) -> None:
        self.assertEqual(support.parse_cpu_list("0-2,8,10-11"), {0, 1, 2, 8, 10, 11})
        with self.assertRaises(support.D4CaptureError):
            support.parse_cpu_list("3-1")

    def test_worker_plan_wraps_the_complete_matrix_in_phase_c(self) -> None:
        masks = {budget: tuple(range(budget)) for budget in support.THREAD_BUDGETS}
        plan = d4_worker.capture_plan(
            python=Path("/venv/shipping/bin/python"),
            wheel=Path("/wheels/qwen_mm.whl"),
            build_label="shipping",
            assets_root=Path("/assets"),
            output_root=Path("/capture"),
            affinity_masks=masks,
        )
        self.assertEqual(len(plan["timed_matrix"]), 4)
        self.assertIn("qwen_mm_reference.phase_c_conformance", plan["pre_conformance"])
        self.assertIn("qwen_mm_reference.phase_c_conformance", plan["post_conformance"])
        for budget, coordinate in zip(support.THREAD_BUDGETS, plan["timed_matrix"], strict=True):
            self.assertEqual(coordinate["thread_budget"], budget)
            self.assertEqual(coordinate["command"][:2], ["taskset", "--cpu-list"])
            self.assertIn(f"t{budget}", coordinate["command"])
            self.assertIn("--mode", coordinate["command"])
            self.assertIn("dedicated", coordinate["command"])
            self.assertIn("5", coordinate["command"])
            self.assertIn("30", coordinate["command"])


if __name__ == "__main__":
    unittest.main()
