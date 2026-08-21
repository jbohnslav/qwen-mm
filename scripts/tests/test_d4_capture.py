from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import d4_capture_support as support  # noqa: E402
import d4_local  # noqa: E402
import d4_smoke  # noqa: E402
import d4_worker  # noqa: E402


def _base_files() -> dict[str, bytes]:
    files = {
        name: b"log\n" if name.endswith(".log") else b"{}\n"
        for name in support.REQUIRED_BASE_MEMBERS
    }
    index = {
        "architecture_family": "arm64",
        "build_labels": list(support.BUILD_LABELS),
        "thread_budgets": list(support.THREAD_BUDGETS),
        "files": sorted(set(files) - {"capture-index.json"}),
    }
    files["capture-index.json"] = (json.dumps(index) + "\n").encode()
    files["provenance.json"] = (
        json.dumps(
            {
                "architecture_family": "arm64",
                "build_wheel_sha256": {label: "a" * 64 for label in support.BUILD_LABELS},
                "build_native_sha256": {label: "b" * 64 for label in support.BUILD_LABELS},
                "host": {
                    "affinity": {"masks": {f"t{budget}": None for budget in support.THREAD_BUDGETS}}
                },
            }
        )
        + "\n"
    ).encode()
    for label in support.BUILD_LABELS:
        noise_by_budget = {}
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
        for budget_index, budget in enumerate(support.THREAD_BUDGETS, start=1):
            result = {
                "architecture_family": "arm64",
                "created_at": f"2026-08-03T00:0{budget_index}:00+00:00",
                "release_eligibility": {
                    "phase_c": {
                        "report_sha256": hashlib.sha256(
                            files[f"phase-c/{label}/pre/report.json"]
                        ).hexdigest()
                    }
                },
                "protocol": {
                    "build_labels": [label],
                    "candidate_identity": {"artifact_sha256": "c" * 64},
                    "thread_regimes": [f"t{budget}"],
                    "random_seed": support.D4_RANDOM_SEED,
                },
                "pairs": [
                    {
                        "profile_alias": "qwen3-vl-8b",
                        "case_id": "text_short",
                        "implementations": {
                            implementation: {"summary": {"wall_ms": {"p50": 100.0}}}
                            for implementation in ("reference", "candidate")
                        },
                    }
                    for _ in range(5)
                ],
            }
            files[f"captures/{label}/t{budget}/noise.json"] = (
                json.dumps(support.result_noise_assessment(result)) + "\n"
            ).encode()
            noise_by_budget[budget] = support.result_noise_assessment(result)
            files[f"captures/{label}/t{budget}/result.json"] = (json.dumps(result) + "\n").encode()
        files[f"builds/{label}/failures.json"] = (
            json.dumps(
                support.capture_failure_report(build_label=label, noise_by_budget=noise_by_budget)
            )
            + "\n"
        ).encode()
    index["files"] = sorted(set(files) - {"capture-index.json"})
    files["capture-index.json"] = (json.dumps(index) + "\n").encode()
    return files


class D4CaptureSupportTests(unittest.TestCase):
    def test_smoke_preserves_virtualenv_interpreter_path(self) -> None:
        interpreter = Path(".venv/bin/python")
        selected = d4_smoke.absolute_without_resolving(interpreter)
        self.assertEqual(selected, Path.cwd() / interpreter)
        self.assertNotEqual(selected, interpreter.resolve())

    def test_smoke_rebuilds_current_native_extension_before_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python = root / "venv/bin/python"
            with mock.patch.object(d4_smoke.subprocess, "run") as run:
                d4_smoke.run_smoke(
                    python=python,
                    assets_root=root / "assets",
                    phase_c_report=root / "phase-c.json",
                    output=root / "result.json",
                    report=root / "report.md",
                )
        self.assertEqual(len(run.call_args_list), 3)
        build = run.call_args_list[0]
        self.assertTrue(any(Path(item).name == "maturin" for item in build.args[0]))
        self.assertIn("develop", build.args[0])
        self.assertIn("--release", build.args[0])
        self.assertIn("--skip-install", build.args[0])
        self.assertNotIn("uv", build.args[0])
        self.assertEqual(build.kwargs["env"]["VIRTUAL_ENV"], str(python.parent.parent))

    def test_source_payload_must_match_immutable_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
            (root / ".gitignore").write_text(".DS_Store\n", encoding="utf-8")
            (root / "source.txt").write_text("recorded\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=D4 Test",
                    "-c",
                    "user.email=d4@example.invalid",
                    "commit",
                    "--quiet",
                    "-m",
                    "recorded",
                ],
                cwd=root,
                check=True,
            )
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            source = support.source_payload_identity(root)
            support.assert_source_payload_matches_revision(root, revision, source)

            (root / ".DS_Store").write_bytes(b"generated metadata")
            contaminated = support.source_payload_identity(root)
            with self.assertRaisesRegex(support.D4CaptureError, "immutable Git revision"):
                support.assert_source_payload_matches_revision(root, revision, contaminated)

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

    def test_raw_inventory_keeps_every_referenced_phase_c_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "phase-c/shipping/pre/outputs/case.json"
            output.parent.mkdir(parents=True)
            output.write_text("{}\n", encoding="utf-8")
            files = d4_local._collect_files(root)
        self.assertEqual(files["phase-c/shipping/pre/outputs/case.json"], b"{}\n")
        for build_label in support.BUILD_LABELS:
            for position in ("pre", "post"):
                for relative in support.PHASE_C_OUTPUT_MEMBERS:
                    self.assertIn(
                        f"phase-c/{build_label}/{position}/outputs/{relative}",
                        support.REQUIRED_BASE_MEMBERS,
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
                "benchmarks/profile-schema-v2.json",
                "benchmarks/result-schema-v3.json",
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

    def test_build_artifact_authentication_accepts_byte_identical_outputs(self) -> None:
        plan = d4_local.build_plan(Path("/capture"))
        same = "a" * 64
        support.assert_build_variant_artifacts(
            plan,
            native_hashes={"shipping": same, "native": same},
            wheel_hashes={"shipping": same, "native": same},
        )

        broken = json.loads(json.dumps(plan))
        broken["builds"]["native"]["build"] = [
            item
            for item in broken["builds"]["native"]["build"]
            if item != "RUSTFLAGS=-C target-cpu=native"
        ]
        with self.assertRaisesRegex(support.D4CaptureError, "optimization override"):
            support.assert_build_variant_artifacts(
                broken,
                native_hashes={"shipping": same, "native": same},
                wheel_hashes={"shipping": same, "native": same},
            )

        shared = json.loads(json.dumps(plan))
        shared["builds"]["native"]["venv"] = shared["builds"]["shipping"]["venv"]
        shared["builds"]["native"]["create_venv"][-1] = shared["builds"]["shipping"]["venv"]
        native_build = shared["builds"]["native"]["build"]
        interpreter = native_build.index("--interpreter") + 1
        native_build[interpreter] = str(Path(shared["builds"]["shipping"]["venv"]) / "bin/python")
        with self.assertRaisesRegex(support.D4CaptureError, "distinct paths"):
            support.assert_build_variant_artifacts(
                shared,
                native_hashes={"shipping": same, "native": same},
                wheel_hashes={"shipping": same, "native": same},
            )

    def test_failed_local_workspace_is_retained_and_success_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            failure = Path(temporary) / "failed"
            failure.mkdir()
            with (
                mock.patch.object(d4_local.tempfile, "mkdtemp", return_value=str(failure)),
                self.assertRaisesRegex(RuntimeError, "boom"),
                d4_local._capture_workspace(),
            ):
                raise RuntimeError("boom")
            self.assertTrue(failure.is_dir())

            success = Path(temporary) / "success"
            success.mkdir()
            with (
                mock.patch.object(d4_local.tempfile, "mkdtemp", return_value=str(success)),
                d4_local._capture_workspace(),
            ):
                (success / "evidence.txt").write_text("ok", encoding="utf-8")
            self.assertFalse(success.exists())

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

    def test_local_capture_requires_the_current_m4_baseline(self) -> None:
        d4_local.assert_local_baseline(system="Darwin", machine="arm64", cpu_model="Apple M4")
        with self.assertRaisesRegex(support.D4CaptureError, "M4 baseline"):
            d4_local.assert_local_baseline(
                system="Darwin", machine="arm64", cpu_model="Apple M3 Max"
            )

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

    def test_modal_vm_sandbox_archive_attestation_rebuilds_exact_controls(self) -> None:
        topology = "# CPU,Core,Socket,Online\n" + "\n".join(f"{cpu},{cpu},0,Y" for cpu in range(16))
        masks = {budget: tuple(range(budget)) for budget in support.THREAD_BUDGETS}

        def phase(offset: int) -> dict[str, object]:
            return {
                "probes": [
                    {
                        "native_thread_count": count,
                        "wall_seconds": 1.0,
                        "process_cpu_seconds": 1.0,
                        "process_cpu_wall_ratio": 1.0,
                        "worker_reports": [
                            {
                                "worker_index": index,
                                "native_thread_id": offset + index,
                                "affinity": [0],
                                "observed_cpus": [0],
                                "iterations": 100,
                            }
                            for index in range(count)
                        ],
                    }
                    for count in (1, 2, 4)
                ]
            }

        attestation = {
            "mode": support.MODAL_VM_SANDBOX_ATTESTATION_MODE,
            "requested_resources_bound_by": "Modal Sandbox.create request/limit tuples",
            "requested_physical_cores": support.MODAL_CPU,
            "requested_memory_mib": support.MODAL_MEMORY_MIB,
            "nonpreemptible": True,
            "nonpreemptible_basis": "Modal CPU-only Sandbox runtime semantics",
            "single_use_container": True,
            "sandbox_id": "sb-test",
            "resolved_modal_image_id": "im-test",
            "vm_runtime": True,
            "virtualization": "kvm",
            "visible_affinity": list(range(16)),
            "physical_core_masks": {
                f"t{budget}": list(masks[budget]) for budget in support.THREAD_BUDGETS
            },
            "cgroup_limits": {"/sys/fs/cgroup/cpuset.cpus.effective": "0-15"},
            "api_resource_request": {
                "cpu_request_and_hard_limit": [support.MODAL_CPU, support.MODAL_CPU],
                "memory_request_and_hard_limit_mib": [
                    support.MODAL_MEMORY_MIB,
                    support.MODAL_MEMORY_MIB,
                ],
                "vm_runtime": True,
                "nonpreemptible": True,
                "single_use": True,
            },
            "affinity_enforcement": {
                "method": "taskset-t1-python-native-threads-v1",
                "expected_affinity": [0],
                "probe_seconds": 1.0,
                "cpu_wall_ratio_max": 1.25,
                "pre": phase(100),
                "post": phase(200),
            },
        }
        self.assertEqual(
            support.validate_modal_vm_sandbox_attestation(
                attestation=attestation,
                lscpu_parse=topology,
                recomputed_masks=masks,
            ),
            attestation,
        )
        mutated = copy.deepcopy(attestation)
        mutated["api_resource_request"]["cpu_request_and_hard_limit"] = [16.0, 32.0]
        with self.assertRaisesRegex(support.D4CaptureError, "resource/topology"):
            support.validate_modal_vm_sandbox_attestation(
                attestation=mutated,
                lscpu_parse=topology,
                recomputed_masks=masks,
            )

    def test_local_linux_resource_attestation_requires_fixed_dedicated_cgroup(self) -> None:
        masks = {budget: tuple(range(budget)) for budget in support.THREAD_BUDGETS}
        cgroups = {
            "/sys/fs/cgroup/cpu.max": "1600000 100000",
            "/sys/fs/cgroup/cpuset.cpus.effective": "0-15",
            "/sys/fs/cgroup/memory.max": str(32 * 1024**3),
        }
        result = support.local_linux_resource_attestation(
            cgroup_limits=cgroups,
            allocated_physical_cores=16,
            allocated_memory_bytes=32 * 1024**3,
            power_policy={
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
            exclusive_physical_cores=False,
            dedicated_capture=True,
            stable=True,
            visible_affinity=list(range(16)),
            masks=masks,
        )
        self.assertEqual(result["mode"], "cgroup_v2_cpuset")
        self.assertEqual(result["allocated_physical_cores"], 16)
        self.assertNotIn("requested_resources_bound_by", result)
        self.assertNotIn("nonpreemptible", result)
        self.assertFalse(result["exclusive_physical_cores"])

        with self.assertRaisesRegex(support.D4CaptureError, "stable host controls"):
            support.local_linux_resource_attestation(
                cgroup_limits=cgroups,
                allocated_physical_cores=16,
                allocated_memory_bytes=32 * 1024**3,
                power_policy=result["power_policy"],
                exclusive_physical_cores=False,
                dedicated_capture=True,
                stable=False,
                visible_affinity=list(range(16)),
                masks=masks,
            )
        undersized = dict(cgroups)
        undersized["/sys/fs/cgroup/cpu.max"] = "800000 100000"
        with self.assertRaisesRegex(support.D4CaptureError, "CPU quota"):
            support.local_linux_resource_attestation(
                cgroup_limits=undersized,
                allocated_physical_cores=16,
                allocated_memory_bytes=32 * 1024**3,
                power_policy=result["power_policy"],
                exclusive_physical_cores=False,
                dedicated_capture=True,
                stable=True,
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

    def test_archived_phase_c_must_bound_a_fresh_ordered_capture_window(self) -> None:
        native_sha = "a" * 64
        package_sha = "b" * 64
        revision = "c" * 40
        build = {
            "runtime": {"native_sha256": native_sha, "package_version": "0.1.0"},
            "runtime_reconciliation": {"wheel_contents": {"package_artifact_sha256": package_sha}},
            "identity": {"wheel": {"name": "qwen_mm-test.whl", "sha256": "d" * 64}},
        }

        def report(created_at: str) -> bytes:
            return json.dumps(
                {
                    "schema_id": "qwen-mm-phase-c-conformance-overlay-v2",
                    "schema_version": 2,
                    "status": "pass",
                    "passed": True,
                    "created_at": created_at,
                    "current_candidate": {
                        "revision": revision,
                        "capture": {
                            "runtime_identity": {
                                "package": "qwen_mm",
                                "version": "0.1.0",
                                "package_artifact_sha256": package_sha,
                                "native_module": "qwen_mm._native",
                                "native_artifact_sha256": native_sha,
                            },
                            "wheel": {"sha256": "d" * 64},
                        },
                    },
                }
            ).encode()

        files = {
            "builds/shipping/capture.json": json.dumps(
                {"created_at": "2026-08-03T00:04:00+00:00"}
            ).encode(),
            "phase-c/shipping/pre/report.json": report("2026-08-03T00:01:00+00:00"),
            "phase-c/shipping/post/report.json": report("2026-08-03T00:03:00+00:00"),
            "builds/shipping/qwen_mm-test.whl": b"wheel",
        }
        provenance = {
            "started_at": "2026-08-03T00:00:00+00:00",
            "completed_at": "2026-08-03T00:05:00+00:00",
            "source_revision": revision,
        }
        with (
            mock.patch("qwen_mm_reference.phase_c_overlay_v2.validate_overlay") as validate,
            mock.patch("qwen_mm_reference.benchmark_v2._validate_phase_c_report") as validate_gate,
        ):
            support.validate_archived_phase_c(
                files, provenance, "shipping", build, assets_root=None
            )
        self.assertIsNotNone(validate.call_args.kwargs["evidence_root"])
        self.assertIsNotNone(validate_gate.call_args.kwargs["report_evidence_root"])
        v1_files = dict(files)
        v1_report = {
            "schema_id": "qwen-mm-phase-c-conformance-report-v1",
            "schema_version": 1,
            "status": "pass",
            "passed": True,
            "created_at": "2026-08-03T00:01:00+00:00",
        }
        v1_files["phase-c/shipping/pre/report.json"] = json.dumps(v1_report).encode()
        with self.assertRaisesRegex(support.D4CaptureError, "archived Phase C pre is invalid"):
            support.validate_archived_phase_c(
                v1_files, provenance, "shipping", build, assets_root=None
            )
        files["phase-c/shipping/post/report.json"] = report("2026-08-03T00:00:30+00:00")
        with (
            mock.patch("qwen_mm_reference.phase_c_overlay_v2.validate_overlay"),
            mock.patch("qwen_mm_reference.benchmark_v2._validate_phase_c_report"),
            self.assertRaisesRegex(support.D4CaptureError, "stale or inverted"),
        ):
            support.validate_archived_phase_c(
                files, provenance, "shipping", build, assets_root=None
            )

    def test_private_environment_mutation_fails_closed_with_path_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            venv = root / "venv"
            package = venv / "lib/python3.11/site-packages/PIL/Image.py"
            package.parent.mkdir(parents=True)
            package.write_text("original\n", encoding="utf-8")
            evidence = root / "environment-integrity.json"
            support.initialize_private_environment_integrity(
                venv, build_label="shipping", evidence_path=evidence
            )
            package.write_text("mutated\n", encoding="utf-8")
            with self.assertRaisesRegex(support.D4CaptureError, "mutated.*PIL/Image.py"):
                support.verify_private_environment_integrity(
                    venv, evidence_path=evidence, checkpoint="before-t1"
                )
            recorded = json.loads(evidence.read_text(encoding="utf-8"))
            self.assertFalse(recorded["checkpoints"][-1]["matches_baseline"])
            self.assertEqual(
                recorded["checkpoints"][-1]["differences"][0]["path"],
                "lib/python3.11/site-packages/PIL/Image.py",
            )

    def test_complete_private_environment_checkpoint_evidence_validates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            venv = root / "venv"
            package = venv / "lib/python3.11/site-packages/qwen_mm/__init__.py"
            package.parent.mkdir(parents=True)
            package.write_text("__version__ = 'test'\n", encoding="utf-8")
            evidence = root / "environment-integrity.json"
            support.initialize_private_environment_integrity(
                venv, build_label="shipping", evidence_path=evidence
            )
            checkpoints = [
                "before-pre-phase-c-v2",
                "after-pre-phase-c-v2",
                *(
                    f"{position}-t{budget}"
                    for budget in support.THREAD_BUDGETS
                    for position in ("before", "after")
                ),
                "before-post-phase-c-v2",
                "after-post-phase-c-v2",
                "before-archive",
            ]
            for checkpoint in checkpoints:
                support.verify_private_environment_integrity(
                    venv, evidence_path=evidence, checkpoint=checkpoint
                )
            support.validate_private_environment_integrity(
                json.loads(evidence.read_text(encoding="utf-8")), build_label="shipping"
            )

    def test_capture_archive_round_trip_and_tamper_rejection(self) -> None:
        build_plan = d4_local.build_plan(Path("/capture"))["builds"]

        def archived_build(
            _files: dict[str, bytes],
            _provenance: dict[str, object],
            build_label: str,
        ) -> dict[str, object]:
            lane = build_plan[build_label]
            return {
                "commands": {
                    "create_venv": lane["create_venv"],
                    "build_wheel": lane["build"],
                },
                "_retained_benchmark_artifact": {"sha256": "c" * 64},
            }

        def phase_c_hashes(
            files: dict[str, bytes],
            _provenance: dict[str, object],
            build_label: str,
            _build: dict[str, object],
            *,
            assets_root: Path | None,
        ) -> dict[str, str]:
            del assets_root
            return {
                position: hashlib.sha256(
                    files[f"phase-c/{build_label}/{position}/report.json"]
                ).hexdigest()
                for position in ("pre", "post")
            } | {
                "pre_created_at": "2026-08-03T00:00:00+00:00",
                "post_created_at": "2026-08-03T00:05:00+00:00",
                "capture_created_at": "2026-08-03T00:06:00+00:00",
            }

        with (
            mock.patch("qwen_mm_reference.benchmark_v2.validate_result_portable"),
            mock.patch.object(
                support,
                "validate_archived_build",
                side_effect=archived_build,
            ),
            mock.patch.object(support, "validate_archived_phase_c", side_effect=phase_c_hashes),
            mock.patch.object(support, "validate_d4_result_contract"),
        ):
            files = _base_files()
            archive = support.create_capture_archive(files)
            self.assertEqual(support.read_capture_archive(archive), files)

            noisy = _base_files()
            result_name = "captures/shipping/t1/result.json"
            noisy_result = json.loads(noisy[result_name])
            for pair, median in zip(
                noisy_result["pairs"], [80.0, 90.0, 100.0, 110.0, 120.0], strict=True
            ):
                for implementation in ("reference", "candidate"):
                    pair["implementations"][implementation]["summary"]["wall_ms"]["p50"] = median
            noisy[result_name] = (json.dumps(noisy_result) + "\n").encode()
            noisy["captures/shipping/t1/noise.json"] = (
                json.dumps(support.result_noise_assessment(noisy_result)) + "\n"
            ).encode()
            shipping_noise = {
                budget: json.loads(noisy[f"captures/shipping/t{budget}/noise.json"])
                for budget in support.THREAD_BUDGETS
            }
            noisy["builds/shipping/failures.json"] = (
                json.dumps(
                    support.capture_failure_report(
                        build_label="shipping", noise_by_budget=shipping_noise
                    )
                )
                + "\n"
            ).encode()
            self.assertEqual(
                support.read_capture_archive(support.create_capture_archive(noisy)), noisy
            )
            forged_failures = dict(noisy)
            report = json.loads(forged_failures["builds/shipping/failures.json"])
            report["failures"] = []
            report["failure_count"] = 0
            report["status"] = "pass"
            forged_failures["builds/shipping/failures.json"] = (json.dumps(report) + "\n").encode()
            with self.assertRaisesRegex(
                support.D4CaptureError, "failure report differs from raw noise"
            ):
                support.read_capture_archive(support.create_capture_archive(forged_failures))

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

            stale_timing = _base_files()
            result_name = "captures/shipping/t1/result.json"
            stale_result = json.loads(stale_timing[result_name])
            stale_result["created_at"] = "2026-08-03T00:00:00+00:00"
            stale_timing[result_name] = json.dumps(stale_result).encode()
            with self.assertRaisesRegex(support.D4CaptureError, "timestamps are stale"):
                support.read_capture_archive(support.create_capture_archive(stale_timing))

    def test_cpu_list_parser_fails_closed(self) -> None:
        self.assertEqual(support.parse_cpu_list("0-2,8,10-11"), {0, 1, 2, 8, 10, 11})
        with self.assertRaises(support.D4CaptureError):
            support.parse_cpu_list("3-1")

    def test_smoke_command_is_small_real_adapter_gut_check(self) -> None:
        command = d4_worker.smoke_command(
            python=Path("/venv/bin/python"),
            assets_root=Path("/assets"),
            phase_c_report=Path("/phase-c/report.json"),
            output=Path("/output/result.json"),
            report=Path("/output/report.md"),
        )
        self.assertIn("official", command)
        self.assertIn("qwen_mm.benchmark:create_adapter", command)
        self.assertEqual(
            command[command.index("--cases") + 1],
            ",".join(d4_worker.SMOKE_CASES),
        )
        self.assertEqual(command[command.index("--process-repetitions") + 1], "1")
        self.assertEqual(command[command.index("--warmups") + 1], "1")
        self.assertEqual(command[command.index("--minimum-samples") + 1], "3")
        self.assertEqual(command[command.index("--minimum-seconds") + 1], "0")
        self.assertEqual(command[command.index("--thread-regimes") + 1], "t1")
        self.assertEqual(command[command.index("--build-labels") + 1], "shipping")

    def test_worker_plan_wraps_the_compact_matrix_in_phase_c(self) -> None:
        masks = {budget: tuple(range(budget)) for budget in d4_worker.COMPACT_THREAD_BUDGETS}
        plan = d4_worker.compact_capture_plan(
            python=Path("/venv/shipping/bin/python"),
            wheel=Path("/wheels/qwen_mm.whl"),
            build_label="shipping",
            assets_root=Path("/assets"),
            output_root=Path("/capture"),
            affinity_masks=masks,
        )
        self.assertEqual(len(plan["timed_matrix"]), 2)
        self.assertIn("qwen_mm_reference.phase_c_overlay_v2", plan["pre_conformance"])
        self.assertIn("qwen_mm_reference.phase_c_overlay_v2", plan["post_conformance"])
        self.assertIn("--reuse-committed-production-evidence", plan["pre_conformance"])
        self.assertNotIn("qwen_mm_reference.phase_c_conformance", plan["pre_conformance"])
        for budget, coordinate in zip(
            d4_worker.COMPACT_THREAD_BUDGETS, plan["timed_matrix"], strict=True
        ):
            self.assertEqual(coordinate["thread_budget"], budget)
            self.assertEqual(coordinate["command"][:2], ["taskset", "--cpu-list"])
            self.assertIn(f"t{budget}", coordinate["command"])
            self.assertIn("--mode", coordinate["command"])
            self.assertIn("dedicated", coordinate["command"])
            self.assertEqual(
                coordinate["command"][coordinate["command"].index("--cases") + 1],
                ",".join(d4_worker.COMPACT_CASES_BY_THREAD_BUDGET[budget]),
            )
            self.assertEqual(
                coordinate["command"][coordinate["command"].index("--process-repetitions") + 1],
                "3",
            )
            self.assertEqual(
                coordinate["command"][coordinate["command"].index("--minimum-samples") + 1],
                "30",
            )
            self.assertEqual(
                coordinate["command"][coordinate["command"].index("--minimum-seconds") + 1],
                "5",
            )
            seed_index = coordinate["command"].index("--seed")
            self.assertEqual(coordinate["command"][seed_index + 1], str(support.D4_RANDOM_SEED))

    def test_worker_collects_noise_failures_and_finishes_capture_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coordinates = []
            for budget in support.THREAD_BUDGETS:
                result_path = root / "captures" / "shipping" / f"t{budget}" / "result.json"
                result_path.parent.mkdir(parents=True, exist_ok=True)
                medians = (
                    [80.0, 90.0, 100.0, 110.0, 120.0]
                    if budget in {1, 4}
                    else [99.0, 100.0, 101.0, 100.5, 99.5]
                )
                result_path.write_text(
                    json.dumps(
                        {
                            "pairs": [
                                {
                                    "profile_alias": "qwen3-vl-8b",
                                    "case_id": "image24",
                                    "implementations": {
                                        implementation: {"summary": {"wall_ms": {"p50": median}}}
                                        for implementation in ("reference", "candidate")
                                    },
                                }
                                for median in medians
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                coordinates.append(
                    {
                        "thread_budget": budget,
                        "command": [f"benchmark-t{budget}"],
                        "output": str(result_path),
                    }
                )
            plan = {
                "pre_conformance": ["pre-conformance"],
                "timed_matrix": coordinates,
                "post_conformance": ["post-conformance"],
            }
            commands: list[list[str]] = []

            def run_logged(command: list[str], **_kwargs: object) -> None:
                commands.append(list(command))

            with (
                mock.patch.object(d4_worker, "capture_plan", return_value=plan),
                mock.patch.object(d4_worker, "installed_build_identity", return_value={}),
                mock.patch.object(d4_worker, "normalized_capture_environment", return_value={}),
                mock.patch.object(d4_worker, "initialize_private_environment_integrity"),
                mock.patch.object(d4_worker, "verify_private_environment_integrity"),
                mock.patch.object(
                    d4_worker,
                    "phase_c_overlay_validation_command",
                    return_value=["validate-phase-c"],
                ),
                mock.patch.object(
                    d4_worker, "benchmark_validation_command", return_value=["validate-benchmark"]
                ),
                mock.patch.object(d4_worker, "_cached_unsupported_attestations", return_value=[]),
                mock.patch.object(d4_worker, "_run_logged", side_effect=run_logged),
            ):
                d4_worker.run_capture(
                    python=root / "venv/bin/python",
                    wheel=root / "qwen_mm.whl",
                    build_label="shipping",
                    assets_root=root / "assets",
                    output_root=root,
                    affinity_masks={budget: None for budget in support.THREAD_BUDGETS},
                    execute=True,
                )

            for budget in support.THREAD_BUDGETS:
                self.assertIn([f"benchmark-t{budget}"], commands)
            self.assertIn(["post-conformance"], commands)
            failures = json.loads(
                (root / "builds/shipping/failures.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failures["status"], "miss")
            self.assertEqual(failures["failure_count"], 4)
            self.assertEqual(
                [failure["gate_id"] for failure in failures["failures"]],
                [
                    "noise/shipping/qwen3-vl-8b/image24/t1/candidate",
                    "noise/shipping/qwen3-vl-8b/image24/t1/reference",
                    "noise/shipping/qwen3-vl-8b/image24/t4/candidate",
                    "noise/shipping/qwen3-vl-8b/image24/t4/reference",
                ],
            )

    def test_compact_worker_collects_failures_and_finishes_capture_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coordinates = []
            for budget in d4_worker.COMPACT_THREAD_BUDGETS:
                result_path = root / "captures" / "shipping" / f"t{budget}" / "result.json"
                result_path.parent.mkdir(parents=True, exist_ok=True)
                medians = [80.0, 100.0, 120.0] if budget == 1 else [99.0, 100.0, 101.0]
                result_path.write_text(
                    json.dumps(
                        {
                            "pairs": [
                                {
                                    "profile_alias": "qwen3-vl-8b",
                                    "case_id": "image24",
                                    "implementations": {
                                        implementation: {"summary": {"wall_ms": {"p50": median}}}
                                        for implementation in ("reference", "candidate")
                                    },
                                }
                                for median in medians
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                coordinates.append(
                    {
                        "thread_budget": budget,
                        "command": [f"compact-t{budget}"],
                        "output": str(result_path),
                    }
                )
            plan = {
                "suite": "compact",
                "pre_conformance": ["pre-conformance"],
                "timed_matrix": coordinates,
                "post_conformance": ["post-conformance"],
            }
            commands: list[list[str]] = []

            def run_logged(command: list[str], **_kwargs: object) -> None:
                commands.append(list(command))

            with (
                mock.patch.object(d4_worker, "compact_capture_plan", return_value=plan),
                mock.patch.object(d4_worker, "installed_build_identity", return_value={}),
                mock.patch.object(d4_worker, "normalized_capture_environment", return_value={}),
                mock.patch.object(d4_worker, "initialize_private_environment_integrity"),
                mock.patch.object(d4_worker, "verify_private_environment_integrity"),
                mock.patch.object(
                    d4_worker,
                    "phase_c_overlay_validation_command",
                    return_value=["validate-phase-c"],
                ),
                mock.patch.object(
                    d4_worker, "benchmark_validation_command", return_value=["validate-benchmark"]
                ),
                mock.patch.object(d4_worker, "_run_logged", side_effect=run_logged),
            ):
                d4_worker.run_compact_capture(
                    python=root / "venv/bin/python",
                    wheel=root / "qwen_mm.whl",
                    assets_root=root / "assets",
                    output_root=root,
                    affinity_masks={budget: None for budget in d4_worker.COMPACT_THREAD_BUDGETS},
                    execute=True,
                )

            self.assertIn(["post-conformance"], commands)
            failures = json.loads(
                (root / "builds/shipping/failures.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failures["status"], "miss")
            self.assertEqual(failures["failure_count"], 2)


if __name__ == "__main__":
    unittest.main()
