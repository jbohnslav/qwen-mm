from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SUPPORT_PATH = Path(__file__).resolve().parents[1] / "modal_d3_support.py"
SPEC = importlib.util.spec_from_file_location("modal_d3_support", SUPPORT_PATH)
assert SPEC is not None and SPEC.loader is not None
support = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = support
SPEC.loader.exec_module(support)


def _digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fixture() -> tuple[dict[str, bytes], tuple[str, int, int], dict[str, str], dict[str, object]]:
    patch_bytes = {
        "copy-only": b"copy patch\n",
        "no-lut": b"no lut patch\n",
        "candidate": b"candidate patch\n",
    }
    expected_diffs = {
        "baseline": support.EMPTY_SHA256,
        **{name: support.sha256_bytes(value) for name, value in patch_bytes.items()},
    }
    binaries = {variant: f"native {variant}".encode() for variant in support.VARIANTS}
    native_hashes = {variant: support.sha256_bytes(value) for variant, value in binaries.items()}
    latin_orders = (
        ("baseline", "copy-only", "no-lut", "candidate"),
        ("copy-only", "no-lut", "candidate", "baseline"),
        ("no-lut", "candidate", "baseline", "copy-only"),
        ("candidate", "baseline", "copy-only", "no-lut"),
    )
    records = []
    matrix_run_nonce = "d" * 32
    sequence = 0
    for coordinate_index, (profile, case) in enumerate(
        (profile, case) for profile in support.PROFILES for case in support.CASES
    ):
        order = latin_orders[coordinate_index % 4]
        for order_index, variant in enumerate(order):
            output_signature = {
                "sha256": hashlib.sha256(f"{profile}/{case}".encode()).hexdigest(),
                "arrays": {},
            }
            sidecar_value = {"profile": profile, "case": case}
            samples = [{"wall_ms": 10.0 + index, "cpu_ms": 9.0 + index} for index in range(7)]
            record = {
                "variant": variant,
                "profile_alias": profile,
                "case_id": case,
                "thread_budget": 1,
                "thread_environment": {name: "1" for name in support.THREAD_NAMES},
                "sample_count": 7,
                "samples": samples,
                "wall_ms_p50": 13.0,
                "cpu_ms_p50": 12.0,
                "peak_rss_bytes": 1_000_000,
                "source": {
                    "implementation_digest": hashlib.sha256(variant.encode()).hexdigest(),
                    "implementation_diff_sha256": expected_diffs[variant],
                },
                "native_module": {"sha256": native_hashes[variant]},
                "harness": {"aggregate_sha256": "a" * 64},
                "protocol": {
                    "input_fingerprint": hashlib.sha256(case.encode()).hexdigest(),
                    "logical_input_fingerprint": hashlib.sha256(case.encode()).hexdigest(),
                },
                "assets": {"profile_fingerprint": hashlib.sha256(profile.encode()).hexdigest()},
                "toolchain": {"rustc": "test", "packages": {"qwen-mm": "0.1.0"}},
                "official_output_signature": output_signature,
                "observed_output_signature": copy.deepcopy(output_signature),
                "authenticated_output_signature": copy.deepcopy(output_signature),
                "official_metadata_sidecar": {
                    "sha256": _digest(sidecar_value),
                    "value": sidecar_value,
                },
                "observation": {
                    "duration_ms": 15.0,
                    "allocations": {
                        "allocation_count": 3,
                        "allocated_bytes": 300,
                        "copy_count": 1,
                        "copied_bytes": 100,
                        "transient_live_bytes": 0,
                        "peak_transient_live_bytes": 200,
                        "retained_final_output_bytes": 100,
                    },
                    "stage_exclusive_ms": {"native.media.resize": 10.0},
                    "buffer_bytes": {"prepared_rgb": 200, "pixel_values": 100},
                    "copy_bytes": {"binding.owned_media": 100},
                    "dropped_events": 0,
                },
                "modal_protocol": {
                    "sequence": sequence,
                    "coordinate_index": coordinate_index,
                    "order_index": order_index,
                    "variant_order": list(order),
                    "fresh_subprocess": True,
                    "pinned_cpu": 7,
                    "wrapper_pid": 1_000 + sequence,
                    "wrapper_sched_getaffinity": [7],
                    "measurement_pid": 2_000 + sequence,
                    "measurement_sched_getaffinity": [7],
                    "matrix_run_nonce": matrix_run_nonce,
                    "coordinate_run_nonce": hashlib.sha256(
                        f"{matrix_run_nonce}:{sequence}:{variant}:{profile}:{case}".encode()
                    ).hexdigest(),
                },
            }
            records.append(record)
            sequence += 1
    summary = support.validate_matrix(records, expected_diff_sha256=expected_diffs, pinned_cpu=7)
    source = ("f" * 64, 100, 10_000)
    assets = {
        "schema_id": "qwen-mm-logical-directory-identity-v2",
        "schema_version": 2,
        "tree_sha256": "e" * 64,
        "entry_count": 2,
        "logical_bytes": 100,
        "entries": [],
    }
    provenance = {
        "schema_id": support.SCHEMA_ID,
        "schema_version": support.SCHEMA_VERSION,
        "execution": "controlled_same_worker_relative_variant_matrix",
        "performance_claim_scope": "D3 same-worker relative variant selection",
        "d4_certification": False,
        "elapsed_seconds": 600.0,
        "cost_estimate": {
            "elapsed_worker_usd": 600.0
            * support.NONPREEMPTIBLE_MULTIPLIER
            * (
                support.MODAL_CPU * support.CPU_USD_PER_PHYSICAL_CORE_SECOND
                + (support.MODAL_MEMORY_MIB / 1024) * support.MEMORY_USD_PER_GIB_SECOND
            ),
            "basis_seconds": 600.0,
            "scope": "elapsed function worker only; excludes image construction and provider billing adjustments",
        },
        "uploaded_source": {"sha256": source[0], "file_count": source[1], "bytes": source[2]},
        "assets": assets,
        "matrix_run_nonce": matrix_run_nonce,
        "expected_diff_sha256": expected_diffs,
        "variant_native_modules": {
            variant: {
                "sha256": native_hashes[variant],
                "file": "ELF 64-bit LSB shared object, x86-64",
            }
            for variant in support.VARIANTS
        },
        "host": {
            "system": "Linux",
            "machine": "x86_64",
            "platform": "Linux-6.0.0-x86_64-with-glibc2.36",
            "uname": "Linux test 6.0.0 x86_64 GNU/Linux",
            "requested_physical_cores": 16.0,
            "requested_memory_mib": 32_768,
            "nonpreemptible": True,
            "single_use_container": True,
            "logical_cpu_count": 32,
            "visible_cpu_affinity": list(range(16)),
            "pinned_cpu": 7,
            "proc_meminfo_total_bytes": 64 * 1024 * 1024 * 1024,
            "resource_attestation": {
                "mode": support.CGROUP_ATTESTATION_MODE,
                "requested_resources_bound_by": support.RESOURCE_BINDING,
            },
            "lscpu": {"lscpu": [{"field": "CPU(s):", "data": "32"}]},
            "cgroup_limits": {
                "/sys/fs/cgroup/cpu.max": "1600000 100000",
                "/sys/fs/cgroup/cpuset.cpus.effective": "0-31",
                "/sys/fs/cgroup/memory.max": str(32_768 * 1024 * 1024),
            },
        },
        "image": {
            "base_image": support.BASE_IMAGE,
            "rust_version": f"rustc {support.RUST_VERSION} (test 2026-01-01)",
            "uv_version": f"uv {support.UV_VERSION} (test 2026-01-01 x86_64-linux)",
        },
        "protocol": {
            "profiles": list(support.PROFILES),
            "cases": list(support.CASES),
            "variants": list(support.VARIANTS),
            "coordinates": 24,
            "warmups": 2,
            "samples": 7,
            "thread_budget": 1,
            "cooldown_seconds": support.COOLDOWN_SECONDS,
            "latin_orders": [list(order) for order in support.LATIN_ORDERS],
            "fresh_subprocess_per_coordinate": True,
            "taskset_cpu": 7,
        },
        "pricing_snapshot": {
            "cpu_usd_per_physical_core_second": support.CPU_USD_PER_PHYSICAL_CORE_SECOND,
            "memory_usd_per_gib_second": support.MEMORY_USD_PER_GIB_SECOND,
            "nonpreemptible_multiplier": support.NONPREEMPTIBLE_MULTIPLIER,
        },
    }
    files = {
        "provenance.json": (json.dumps(provenance) + "\n").encode(),
        "summary.json": (json.dumps(summary) + "\n").encode(),
    }
    files.update({f"patches/{name}.patch": value for name, value in patch_bytes.items()})
    files.update({f"binaries/{name}.so": value for name, value in binaries.items()})
    for variant in support.VARIANTS:
        files[f"logs/build-{variant}.log"] = (
            b"maturin develop --release\nFinished `release` profile\n"
        )
    files["logs/measure.log"] = b"$ taskset -c 7 coordinate\n" * 24
    for record in records:
        files[
            f"coordinates/{record['variant']}/{record['profile_alias']}-{record['case_id']}.json"
        ] = (json.dumps(record) + "\n").encode()
    return files, source, expected_diffs, assets


class ModalD3SupportTests(unittest.TestCase):
    def test_integrity_archive_round_trip_binds_local_source_and_patches(self) -> None:
        files, source, expected_diffs, assets = _fixture()
        artifact = support.create_archive(files)
        validated = support.validate_archive(artifact)
        self.assertEqual(validated["summary"]["coordinate_count"], 24)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "d3.zip"
            support.write_archive_atomically(
                artifact,
                destination,
                expected_source=source,
                expected_diff_sha256=expected_diffs,
                expected_assets=assets,
            )
            self.assertEqual(destination.read_bytes(), artifact)
            with self.assertRaisesRegex(support.ModalD3EvidenceError, "local upload"):
                support.write_archive_atomically(
                    artifact,
                    destination,
                    expected_source=("0" * 64, source[1], source[2]),
                    expected_diff_sha256=expected_diffs,
                    expected_assets=assets,
                )

    def test_matrix_rejects_affinity_medians_drops_and_order(self) -> None:
        files, _, expected_diffs, _ = _fixture()
        records = [
            json.loads(value) for name, value in files.items() if name.startswith("coordinates/")
        ]
        mutations = (
            (
                "affinity",
                lambda record: record["modal_protocol"].update(
                    {"measurement_sched_getaffinity": [8]}
                ),
            ),
            ("medians", lambda record: record.update({"wall_ms_p50": 999.0})),
            ("samples", lambda record: record["samples"][0].update({"wall_ms": True})),
            ("dropped", lambda record: record["observation"].update({"dropped_events": 1})),
            (
                "reconcile with buffers",
                lambda record: record["observation"]["allocations"].update(
                    {"allocated_bytes": 301}
                ),
            ),
            (
                "run nonce",
                lambda record: record["modal_protocol"].update({"coordinate_run_nonce": "0" * 64}),
            ),
            ("Latin-square", lambda record: record["modal_protocol"].update({"order_index": 3})),
        )
        for message, mutate in mutations:
            changed = copy.deepcopy(records)
            mutate(changed[0])
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(support.ModalD3EvidenceError, message),
            ):
                support.validate_matrix(changed, expected_diff_sha256=expected_diffs, pinned_cpu=7)

    def test_archive_rejects_underprovisioned_cpuset_and_wrong_matrix_nonce(self) -> None:
        files, _, _, _ = _fixture()
        provenance = json.loads(files["provenance.json"])
        provenance["host"]["cgroup_limits"]["/sys/fs/cgroup/cpuset.cpus.effective"] = "0-14"
        files["provenance.json"] = (json.dumps(provenance) + "\n").encode()
        with self.assertRaisesRegex(support.ModalD3EvidenceError, "cpuset"):
            support.validate_archive(support.create_archive(files))

        files, _, _, _ = _fixture()
        provenance = json.loads(files["provenance.json"])
        provenance["matrix_run_nonce"] = "e" * 32
        files["provenance.json"] = (json.dumps(provenance) + "\n").encode()
        with self.assertRaisesRegex(support.ModalD3EvidenceError, "matrix run nonce"):
            support.validate_archive(support.create_archive(files))

    def test_archive_accepts_observed_capacity_gvisor_fallback(self) -> None:
        files, _, _, _ = _fixture()
        provenance = json.loads(files["provenance.json"])
        host = provenance["host"]
        host.update(
            {
                "platform": "Linux-4.19.0-gvisor-x86_64-with-glibc2.36",
                "uname": "Linux modal 4.19.0-gvisor x86_64 GNU/Linux",
                "logical_cpu_count": 24,
                "visible_cpu_affinity": list(range(24)),
                "proc_meminfo_total_bytes": 810_988_670_976,
                "lscpu": {"lscpu": [{"field": "CPU(s):", "data": "24"}]},
                "cgroup_limits": {},
                "resource_attestation": {
                    "mode": support.GVISOR_ATTESTATION_MODE,
                    "requested_resources_bound_by": support.RESOURCE_BINDING,
                },
            }
        )
        files["provenance.json"] = (json.dumps(provenance) + "\n").encode()
        support.validate_archive(support.create_archive(files))

        host["proc_meminfo_total_bytes"] = 31 * 1024 * 1024 * 1024
        files["provenance.json"] = (json.dumps(provenance) + "\n").encode()
        with self.assertRaisesRegex(support.ModalD3EvidenceError, "resource or affinity"):
            support.validate_archive(support.create_archive(files))

    def test_archive_rejects_incomplete_log_and_wrong_protocol(self) -> None:
        files, _, _, _ = _fixture()
        files["logs/measure.log"] = b"$ taskset -c 7 coordinate\n" * 23
        with self.assertRaisesRegex(support.ModalD3EvidenceError, "24 taskset"):
            support.validate_archive(support.create_archive(files))

        files, _, _, _ = _fixture()
        provenance = json.loads(files["provenance.json"])
        provenance["protocol"]["samples"] = 5
        files["provenance.json"] = (json.dumps(provenance) + "\n").encode()
        with self.assertRaisesRegex(support.ModalD3EvidenceError, "protocol"):
            support.validate_archive(support.create_archive(files))

    def test_archive_rejects_too_many_members_before_reading(self) -> None:
        files, _, _, _ = _fixture()
        for index in range(support.MAX_ARCHIVE_MEMBERS):
            files[f"extra/{index}.txt"] = b"x"
        with self.assertRaisesRegex(support.ModalD3EvidenceError, "too many members"):
            support.validate_archive(support.create_archive(files))


if __name__ == "__main__":
    unittest.main()
