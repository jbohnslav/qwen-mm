from __future__ import annotations

import copy
import hashlib
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import qwen_mm_reference.benchmark_protocol as benchmark_protocol
import qwen_mm_reference.benchmark_v2 as benchmark_v2
from qwen_mm_reference.benchmark_protocol import (
    BenchmarkProtocolError,
    compare_outputs,
    load_workload,
    materialize_case,
    normalize_outputs,
    run_worker,
    select_cases,
)
from qwen_mm_reference.benchmark_v2 import (
    PHASE_C_REPORT_SCHEMA_ID,
    _validate_release_eligibility,
    bootstrap_confidence_interval,
    candidate_artifact_identity,
    evaluate_image_release_gate,
    randomized_orders,
    render_report,
    run_benchmark,
    validate_result,
    validate_result_authenticated_portable,
    validate_result_portable,
)
from qwen_mm_reference.fixtures import repository_root

WORKLOAD_PATH = repository_root() / "benchmarks" / "workloads-v2.json"
WORKLOAD_SCHEMA_PATH = repository_root() / "benchmarks" / "workload-schema-v2.json"
RESULT_SCHEMA_PATH = repository_root() / "benchmarks" / "result-schema-v2.json"
IMAGE_CASE = {"case_id": "image", "boundary": "encoded_to_numpy"}
PHASE_C_CASE_IDS = ["image", "text"]
RUNTIME_IDENTITY = {
    "package": "qwen_mm",
    "version": "0.1.0",
    "package_artifact_sha256": "c" * 64,
    "native_module": "qwen_mm._native",
    "native_artifact_sha256": "d" * 64,
}


def worker_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "implementation": "candidate",
        "adapter_spec": "synthetic",
        "oracle_adapter_spec": "synthetic",
        "profile_alias": "qwen3-vl-8b",
        "build_label": "shipping",
        "thread_budget": 1,
        "warmups": 1,
        "minimum_samples": 3,
        "minimum_seconds": 0.0,
    }
    config.update(overrides)
    return config


def write_phase_c_report(
    root: Path,
    candidate_identity: dict[str, Any],
    *,
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    tracked = root / "candidate-source.txt"
    tracked.write_text("candidate source\n", encoding="utf-8")
    report: dict[str, object] = {
        "schema_id": PHASE_C_REPORT_SCHEMA_ID,
        "schema_version": 1,
        "contract_id": "qwen-mm-compat-v1",
        "status": "pass",
        "passed": True,
        "scope": {
            "kind": "text_image",
            "profiles": ["qwen3-vl-8b", "qwen3.5-9b"],
            "declared_case_ids": PHASE_C_CASE_IDS,
            "executed_case_ids": PHASE_C_CASE_IDS,
            "skipped_case_ids": [],
        },
        "candidate": {
            "runtime_identity": candidate_identity["runtime_identity"],
        },
        "inputs": [
            {
                "path": "candidate-source.txt",
                "sha256": hashlib.sha256(tracked.read_bytes()).hexdigest(),
            }
        ],
        "results": [
            {
                "case_id": case_id,
                "candidate_executed": True,
                "passed": True,
                "issues": [],
            }
            for case_id in PHASE_C_CASE_IDS
        ],
    }
    if mutate is not None:
        mutate(report)
    path = root / "reference" / "phase-c" / "v1" / "report.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


class WorkloadTests(unittest.TestCase):
    @staticmethod
    def _foreign_candidate_result(result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        adapter = "qwen_mm.benchmark:create_adapter"
        identity = candidate_artifact_identity(adapter)
        if identity.get("resolved") is not True:
            raise AssertionError("portable regression requires the built qwen-mm adapter wrapper")
        foreign_runtime = {
            "package": "qwen_mm",
            "version": "0.1.0",
            "package_artifact_sha256": "a" * 64,
            "native_module": "qwen_mm._native",
            "native_artifact_sha256": "b" * 64,
        }
        identity["runtime_identity"] = foreign_runtime
        result["architecture_family"] = "x86_64"
        result["protocol"]["candidate_adapter"] = adapter
        result["protocol"]["candidate_identity"] = identity
        for pair in result["pairs"]:
            for implementation in pair["implementations"].values():
                implementation["environment"]["architecture_family"] = "x86_64"
            pair["implementations"]["candidate"]["adapter_spec"] = adapter
        return result, foreign_runtime

    def test_authenticated_portable_validation_accepts_foreign_runtime_and_rejects_tampering(
        self,
    ) -> None:
        result = run_benchmark(
            workload_path=WORKLOAD_PATH,
            mode="smoke",
            reference_adapter="synthetic",
            candidate_adapter="synthetic",
            profiles=["qwen3-vl-8b"],
            case_ids=["text_short"],
            process_repetitions=1,
            warmups=0,
            minimum_samples=1,
            minimum_seconds=0.0,
            thread_regimes=["one"],
            build_labels=["profiled-release"],
            seed=41,
        )
        result, foreign_runtime = self._foreign_candidate_result(result)

        validate_result_authenticated_portable(result, expected_runtime_identity=foreign_runtime)
        same_architecture = copy.deepcopy(result)
        same_architecture["architecture_family"] = benchmark_v2.architecture_family()
        for pair in same_architecture["pairs"]:
            for implementation in pair["implementations"].values():
                implementation["environment"]["architecture_family"] = same_architecture[
                    "architecture_family"
                ]
        validate_result_authenticated_portable(
            same_architecture, expected_runtime_identity=foreign_runtime
        )

        mutations = {
            "adapter": lambda value: value["protocol"]["candidate_identity"].update(
                artifact_sha256="0" * 64
            ),
            "runtime": lambda value: value["protocol"]["candidate_identity"][
                "runtime_identity"
            ].update(native_artifact_sha256="0" * 64),
            "workload": lambda value: value["workload"].update(sha256="0" * 64),
            "thread": lambda value: value["pairs"][0]["implementations"]["candidate"][
                "thread_settings"
            ]["environment"].update(OMP_NUM_THREADS="9"),
            "pair": lambda value: value["pairs"][0]["implementations"]["candidate"].update(
                input_fingerprint="tampered"
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                tampered = copy.deepcopy(result)
                mutate(tampered)
                with self.assertRaises(BenchmarkProtocolError):
                    validate_result_authenticated_portable(
                        tampered, expected_runtime_identity=foreign_runtime
                    )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "result.json"
            authentication_path = root / "runtime.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            authentication_path.write_text(
                json.dumps({"benchmark_runtime_identity": foreign_runtime}), encoding="utf-8"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "qwen_mm_reference.benchmark_v2",
                    "validate-portable",
                    str(result_path),
                    "--runtime-authentication",
                    str(authentication_path),
                ],
                cwd=repository_root(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_authenticated_portable_validation_binds_durable_phase_c_path_and_hash(self) -> None:
        result = run_benchmark(
            workload_path=WORKLOAD_PATH,
            mode="smoke",
            reference_adapter="synthetic",
            candidate_adapter="synthetic",
            profiles=["qwen3-vl-8b"],
            case_ids=["image1"],
            process_repetitions=1,
            warmups=0,
            minimum_samples=1,
            minimum_seconds=0.0,
            thread_regimes=["one"],
            build_labels=["profiled-release"],
            seed=43,
        )
        result, foreign_runtime = self._foreign_candidate_result(result)
        root = repository_root()
        with tempfile.TemporaryDirectory(prefix=".portable-phase-c-", dir=root) as directory:
            report = Path(directory) / "report.json"
            report.write_text('{"status":"pass"}\n', encoding="utf-8")
            relative = report.relative_to(root).as_posix()
            report_sha256 = hashlib.sha256(report.read_bytes()).hexdigest()
            evidence = {"candidate_runtime_identity": foreign_runtime}
            fingerprint = hashlib.sha256(
                json.dumps(
                    {"report_sha256": report_sha256, "evidence": evidence},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            result["release_eligibility"]["phase_c"] = {
                "status": "pass",
                "reason_codes": [],
                "report_path": relative,
                "assets_root": "reference/.cache/huggingface",
                "mode_note": "smoke measurement is never release evidence",
                "report_sha256": report_sha256,
                "gate_fingerprint": fingerprint,
                "evidence": evidence,
            }
            validate_result_authenticated_portable(
                result, expected_runtime_identity=foreign_runtime
            )
            mutations = {
                "temporary-path": lambda value: value["release_eligibility"]["phase_c"].update(
                    report_path="/tmp/vanished-phase-c.json"
                ),
                "relabel": lambda value: value["release_eligibility"]["phase_c"].update(
                    report_path="benchmarks/workloads-v2.json"
                ),
                "hash": lambda value: value["release_eligibility"]["phase_c"].update(
                    report_sha256="0" * 64
                ),
                "fingerprint": lambda value: value["release_eligibility"]["phase_c"].update(
                    gate_fingerprint="0" * 64
                ),
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    tampered = copy.deepcopy(result)
                    mutate(tampered)
                    with self.assertRaises(BenchmarkProtocolError):
                        validate_result_authenticated_portable(
                            tampered, expected_runtime_identity=foreign_runtime
                        )

    def test_generated_encoded_portability_keeps_exact_and_logical_bindings(self) -> None:
        workload = load_workload(WORKLOAD_PATH)
        case = select_cases(workload, case_ids=["ragged24"])[0]
        case["source"].update(count=2, formats=["JPEG", "PNG"], shapes=[[7, 11]], seed=47)
        workload["cases"] = [case]
        with tempfile.TemporaryDirectory(prefix=".logical-input-", dir=repository_root()) as value:
            workload_path = Path(value) / "workload.json"
            workload_path.write_text(json.dumps(workload), encoding="utf-8")
            result = run_benchmark(
                workload_path=workload_path,
                mode="smoke",
                reference_adapter="synthetic",
                candidate_adapter="synthetic",
                profiles=["qwen3-vl-8b"],
                case_ids=["ragged24"],
                process_repetitions=1,
                warmups=0,
                minimum_samples=1,
                minimum_seconds=0.0,
                thread_regimes=["one"],
                build_labels=["profiled-release"],
                seed=47,
            )
            exact = {
                implementation["input_fingerprint"]
                for implementation in result["pairs"][0]["implementations"].values()
            }
            logical = {
                implementation["logical_input_fingerprint"]
                for implementation in result["pairs"][0]["implementations"].values()
            }
            self.assertEqual(len(exact), 1)
            self.assertEqual(len(logical), 1)

            exact_mutation = copy.deepcopy(result)
            for implementation in exact_mutation["pairs"][0]["implementations"].values():
                implementation["input_fingerprint"] = "e" * 64
            with self.assertRaisesRegex(BenchmarkProtocolError, "relabeled"):
                validate_result(exact_mutation)

            foreign = copy.deepcopy(exact_mutation)
            foreign["architecture_family"] = (
                "x86_64" if benchmark_v2.architecture_family() == "arm64" else "arm64"
            )
            for implementation in foreign["pairs"][0]["implementations"].values():
                implementation["environment"]["architecture_family"] = foreign[
                    "architecture_family"
                ]
            validate_result_portable(foreign)

            mismatched_exact = copy.deepcopy(foreign)
            mismatched_exact["pairs"][0]["implementations"]["candidate"]["input_fingerprint"] = (
                "f" * 64
            )
            with self.assertRaisesRegex(BenchmarkProtocolError, "different inputs"):
                validate_result_portable(mismatched_exact)

            invalid_exact = copy.deepcopy(foreign)
            for implementation in invalid_exact["pairs"][0]["implementations"].values():
                implementation["input_fingerprint"] = "not-a-sha256"
            with self.assertRaisesRegex(BenchmarkProtocolError, "not SHA-256"):
                validate_result_portable(invalid_exact)

            changed_logical = copy.deepcopy(foreign)
            for implementation in changed_logical["pairs"][0]["implementations"].values():
                implementation["logical_input_fingerprint"] = "a" * 64
            with self.assertRaisesRegex(BenchmarkProtocolError, "relabeled"):
                validate_result_portable(changed_logical)

    def test_result_schema_closes_phase_c_release_states(self) -> None:
        schema = json.loads(RESULT_SCHEMA_PATH.read_text(encoding="utf-8"))
        eligibility = schema["properties"]["release_eligibility"]
        self.assertEqual(eligibility["properties"]["releasable"], {"const": False})
        phase_c = eligibility["properties"]["phase_c"]
        self.assertFalse(phase_c["additionalProperties"])
        pass_rule = phase_c["allOf"][0]["then"]
        self.assertEqual(
            set(pass_rule["required"]),
            {"report_path", "assets_root", "report_sha256", "gate_fingerprint", "evidence"},
        )
        self.assertEqual(pass_rule["properties"]["reason_codes"], {"maxItems": 0})
        blocked_rule = phase_c["allOf"][1]["then"]
        self.assertEqual(blocked_rule["properties"]["reason_codes"], {"minItems": 1})

    def test_release_matrix_covers_a5_contract(self) -> None:
        workload = load_workload(WORKLOAD_PATH)
        case_ids = {case["case_id"] for case in workload["cases"]}
        required = {
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
        }
        self.assertTrue(required <= case_ids)
        self.assertEqual(len(case_ids), len(workload["cases"]))

    def test_same_case_materializes_stable_messages_buffers_and_fingerprint(self) -> None:
        workload = load_workload(WORKLOAD_PATH)
        case = select_cases(workload, case_ids=["image1"])[0]

        first = materialize_case(case)
        second = materialize_case(case)

        self.assertEqual(first.input_fingerprint, second.input_fingerprint)
        self.assertEqual(first.input_fingerprint, first.logical_input_fingerprint)
        self.assertEqual(first.messages, second.messages)
        self.assertEqual(first.buffers[0].value, second.buffers[0].value)
        self.assertEqual(first.boundary, "encoded_to_numpy")

    def test_ragged_generator_covers_formats_and_shapes(self) -> None:
        workload = load_workload(WORKLOAD_PATH)
        case = select_cases(workload, case_ids=["ragged24"])[0]
        case["source"] = copy.deepcopy(case["source"])
        case["source"].update(count=3, shapes=[[31, 47], [32, 64], [65, 33]])

        payload = materialize_case(case)

        self.assertEqual([media.format for media in payload.buffers], ["JPEG", "PNG", "WEBP"])
        self.assertEqual(len({media.value for media in payload.buffers}), 3)

    def test_generated_encoded_fingerprint_binds_logical_source_not_codec_bytes(self) -> None:
        workload = load_workload(WORKLOAD_PATH)
        case = select_cases(workload, case_ids=["ragged24"])[0]
        case["source"] = copy.deepcopy(case["source"])
        case["source"].update(count=2, formats=["JPEG", "PNG"], shapes=[[7, 11]], seed=17)

        with mock.patch.object(benchmark_protocol, "_encode_rgb", return_value=b"codec-a"):
            first = materialize_case(case)
        with mock.patch.object(benchmark_protocol, "_encode_rgb", return_value=b"codec-b"):
            second = materialize_case(case)

        self.assertNotEqual(first.buffers[0].value, second.buffers[0].value)
        self.assertNotEqual(first.input_fingerprint, second.input_fingerprint)
        self.assertEqual(first.logical_input_fingerprint, second.logical_input_fingerprint)

        original_generated_rgb = benchmark_protocol._generated_rgb

        def changed_rgb(height: int, width: int, seed: int) -> np.ndarray:
            value = original_generated_rgb(height, width, seed).copy()
            value[0, 0, 0] ^= np.uint8(1)
            value.setflags(write=False)
            return value

        mutations = {
            "logical-source": None,
            "format": {"formats": ["WEBP", "PNG"]},
            "shape": {"shapes": [[11, 7]]},
            "seed": {"seed": 18},
        }
        for name, source_update in mutations.items():
            with self.subTest(name=name):
                changed_case = copy.deepcopy(case)
                if source_update is not None:
                    changed_case["source"].update(source_update)
                patches = [mock.patch.object(benchmark_protocol, "_encode_rgb", return_value=b"x")]
                if name == "logical-source":
                    patches.append(
                        mock.patch.object(benchmark_protocol, "_generated_rgb", changed_rgb)
                    )
                with ExitStack() as stack:
                    for patch in patches:
                        stack.enter_context(patch)
                    changed = materialize_case(changed_case)
                self.assertNotEqual(
                    first.logical_input_fingerprint, changed.logical_input_fingerprint
                )

    def test_fixture_encoded_fingerprint_still_binds_raw_bytes(self) -> None:
        workload = load_workload(WORKLOAD_PATH)
        case = select_cases(workload, case_ids=["image1"])[0]
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "fixture.jpg"
            with mock.patch.object(
                benchmark_protocol, "fixture_directory", return_value=Path(directory)
            ):
                fixture.write_bytes(b"fixture-a")
                first = materialize_case(case)
                fixture.write_bytes(b"fixture-b")
                second = materialize_case(case)

        self.assertNotEqual(first.input_fingerprint, second.input_fingerprint)
        self.assertEqual(first.input_fingerprint, first.logical_input_fingerprint)
        self.assertEqual(second.input_fingerprint, second.logical_input_fingerprint)

    def test_raw_rgb_payload_is_caller_owned_read_only_uint8(self) -> None:
        workload = load_workload(WORKLOAD_PATH)
        case = select_cases(workload, case_ids=["rgb24"])[0]
        case["source"] = copy.deepcopy(case["source"])
        case["source"].update(count=2, shapes=[[17, 19]])

        payload = materialize_case(case)

        self.assertEqual(payload.boundary, "rgb_to_vllm_ready")
        self.assertEqual(payload.input_fingerprint, payload.logical_input_fingerprint)
        for media in payload.buffers:
            self.assertIsInstance(media.value, np.ndarray)
            self.assertEqual(media.value.dtype, np.uint8)
            self.assertFalse(media.value.flags.writeable)


class ConformanceTests(unittest.TestCase):
    def setUp(self) -> None:
        workload = load_workload(WORKLOAD_PATH)
        self.payload = materialize_case(select_cases(workload, case_ids=["image1"])[0])
        self.outputs = {
            "input_ids": np.zeros((1, 4), dtype=np.int64),
            "attention_mask": np.ones((1, 4), dtype=np.int64),
            "mm_token_type_ids": np.zeros((1, 4), dtype=np.int64),
            "pixel_values": np.zeros((4, 1536), dtype=np.float32),
            "image_grid_thw": np.asarray([[1, 2, 2]], dtype=np.int64),
        }

    def test_contract_rejects_wrong_dtype_and_key_order(self) -> None:
        broken_dtype = {**self.outputs, "pixel_values": self.outputs["pixel_values"].astype("f8")}
        with self.assertRaisesRegex(BenchmarkProtocolError, "dtype"):
            normalize_outputs(broken_dtype, self.payload)

        broken_order = dict(reversed(list(self.outputs.items())))
        with self.assertRaisesRegex(BenchmarkProtocolError, "key/order"):
            normalize_outputs(broken_order, self.payload)

    def test_comparison_allows_contract_tolerance_but_stability_is_exact(self) -> None:
        expected = normalize_outputs(self.outputs, self.payload)
        actual = {name: value.copy() for name, value in expected.items()}
        actual["pixel_values"][0, 0] = self.payload.float_atol / 2

        compare_outputs(expected, actual, self.payload)
        with self.assertRaisesRegex(BenchmarkProtocolError, "values differ"):
            compare_outputs(expected, actual, self.payload, exact_float=True)

    def test_mixed_codec_pixels_keep_lossless_occurrences_strict(self) -> None:
        workload = load_workload(WORKLOAD_PATH)
        case = select_cases(workload, case_ids=["ragged24"])[0]
        case["source"] = copy.deepcopy(case["source"])
        case["source"].update(count=3, shapes=[[31, 47]])
        payload = materialize_case(case)
        expected = {
            "input_ids": np.zeros((1, 4), dtype=np.int64),
            "attention_mask": np.ones((1, 4), dtype=np.int64),
            "mm_token_type_ids": np.zeros((1, 4), dtype=np.int64),
            "pixel_values": np.zeros((3, 1536), dtype=np.float32),
            "image_grid_thw": np.tile(np.asarray([[1, 1, 1]], dtype=np.int64), (3, 1)),
        }
        actual = {name: value.copy() for name, value in expected.items()}
        actual["pixel_values"][0, 0] = 1e-4
        compare_outputs(expected, actual, payload)

        actual["pixel_values"][1, 0] = 1e-4
        with self.assertRaisesRegex(BenchmarkProtocolError, "values differ"):
            compare_outputs(expected, actual, payload)


class ReleaseGuardTests(unittest.TestCase):
    candidate_identity = {
        "adapter_spec": "candidate.adapter:create",
        "kind": "module",
        "resolved": True,
        "module": "candidate.adapter",
        "origin": "/temporary/site-packages/candidate/adapter.py",
        "artifact_sha256": "a" * 64,
        "distributions": {"qwen-mm": "0.1.0"},
        "runtime_identity": RUNTIME_IDENTITY,
    }

    def authority(self, root: Path) -> ExitStack:
        def validate(report: dict[str, Any], *, assets_root: Path) -> None:
            del assets_root
            if (
                report.get("schema_id") != PHASE_C_REPORT_SCHEMA_ID
                or report.get("schema_version") != 1
            ):
                raise ValueError("unsupported Phase C report schema")
            if report.get("status") != "pass" or report.get("passed") is not True:
                raise ValueError("Phase C report is not passing")
            scope = report.get("scope", {})
            if scope.get("profiles") != ["qwen3-vl-8b", "qwen3.5-9b"]:
                raise ValueError("Phase C report does not cover both profiles")
            if (
                scope.get("declared_case_ids") != PHASE_C_CASE_IDS
                or scope.get("executed_case_ids") != PHASE_C_CASE_IDS
                or scope.get("skipped_case_ids") != []
            ):
                raise ValueError("Phase C candidate inventory drifted")
            results = report.get("results")
            if not isinstance(results, list) or {item.get("case_id") for item in results} != set(
                PHASE_C_CASE_IDS
            ):
                raise ValueError("Phase C result rows do not match the declared inventory")
            if any(
                item.get("candidate_executed") is not True
                or item.get("passed") is not True
                or item.get("issues") != []
                for item in results
            ):
                raise ValueError("candidate case did not pass")
            tracked = root / "candidate-source.txt"
            expected_inputs = [
                {
                    "path": "candidate-source.txt",
                    "sha256": hashlib.sha256(tracked.read_bytes()).hexdigest(),
                }
            ]
            if report.get("inputs") != expected_inputs:
                raise ValueError("Phase C authenticated source inputs are stale")

        stack = ExitStack()
        stack.enter_context(
            mock.patch.object(benchmark_v2, "validate_phase_c_report", side_effect=validate)
        )
        stack.enter_context(
            mock.patch.object(
                benchmark_v2,
                "expected_candidate_case_ids",
                return_value=PHASE_C_CASE_IDS,
            )
        )
        stack.enter_context(
            mock.patch.object(
                benchmark_v2,
                "phase_c_source_inputs",
                return_value=[Path("candidate-source.txt")],
            )
        )
        return stack

    def evaluate(self, root: Path, path: Path | None, **identity: object) -> dict[str, Any]:
        candidate_identity = {**self.candidate_identity, **identity}
        return evaluate_image_release_gate(
            mode="dedicated",
            selected_cases=[IMAGE_CASE],
            candidate_identity=candidate_identity,
            phase_c_report_path=path,
            root=root,
        )

    def test_complete_current_report_qualifies_correctness_but_not_performance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = write_phase_c_report(root, self.candidate_identity)
            with self.authority(root):
                eligibility = self.evaluate(root, path)

        self.assertEqual(eligibility["phase_c"]["status"], "pass")
        self.assertEqual(eligibility["phase_c"]["reason_codes"], [])
        self.assertEqual(len(eligibility["phase_c"]["report_sha256"]), 64)
        self.assertEqual(len(eligibility["phase_c"]["gate_fingerprint"]), 64)
        self.assertFalse(eligibility["releasable"])
        self.assertEqual(eligibility["performance_status"], "diagnostic_only")

    def test_missing_report_and_artifact_mismatch_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = self.evaluate(root, root / "missing.json")
            path = write_phase_c_report(root, self.candidate_identity)
            changed_runtime = {**RUNTIME_IDENTITY, "native_artifact_sha256": "b" * 64}
            with self.authority(root):
                mismatch = self.evaluate(root, path, runtime_identity=changed_runtime)

        self.assertEqual(missing["phase_c"]["status"], "missing")
        self.assertEqual(missing["phase_c"]["reason_codes"], ["report_missing"])
        self.assertEqual(mismatch["phase_c"]["status"], "artifact_mismatch")
        self.assertEqual(mismatch["phase_c"]["reason_codes"], ["candidate_runtime_mismatch"])
        self.assertFalse(missing["releasable"])
        self.assertFalse(mismatch["releasable"])

    def test_stale_input_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = write_phase_c_report(root, self.candidate_identity)
            (root / "candidate-source.txt").write_text("changed\n", encoding="utf-8")
            with self.authority(root):
                eligibility = self.evaluate(root, path)

        self.assertEqual(eligibility["phase_c"]["status"], "stale")
        self.assertEqual(eligibility["phase_c"]["reason_codes"], ["report_currentness"])

    def test_incomplete_tampered_or_performance_bearing_reports_fail_closed(self) -> None:
        def missing_profile(report: dict[str, Any]) -> None:
            report["scope"]["profiles"] = ["qwen3-vl-8b"]

        def missing_case(report: dict[str, Any]) -> None:
            report["scope"]["executed_case_ids"] = ["text"]

        def skipped_case(report: dict[str, Any]) -> None:
            report["scope"]["skipped_case_ids"] = ["image"]

        def minimized_inventory(report: dict[str, Any]) -> None:
            report["scope"]["declared_case_ids"] = ["image"]
            report["scope"]["executed_case_ids"] = ["image"]
            report["results"] = [report["results"][0]]

        def failed_case_under_passing_top_level(report: dict[str, Any]) -> None:
            report["results"][0]["passed"] = False
            report["results"][0]["issues"] = [{"kind": "mismatch"}]

        mutations: dict[str, tuple[str, str, Callable[[dict[str, Any]], None]]] = {
            "schema": (
                "invalid",
                "report_validation",
                lambda report: report.__setitem__("schema_id", "broken"),
            ),
            "not-passing": (
                "invalid",
                "report_validation",
                lambda report: report.__setitem__("passed", False),
            ),
            "profile": ("invalid", "report_validation", missing_profile),
            "case": ("stale", "report_currentness", missing_case),
            "skip": ("stale", "report_currentness", skipped_case),
            "minimized": ("stale", "report_currentness", minimized_inventory),
            "per-case-failure": (
                "invalid",
                "report_validation",
                failed_case_under_passing_top_level,
            ),
            "performance": (
                "invalid",
                "performance_claim",
                lambda report: report.__setitem__("performance", {"speedup": 2.0}),
            ),
            "performance-string": (
                "invalid",
                "performance_claim",
                lambda report: report.__setitem__("notes", "candidate has a 2x speedup"),
            ),
        }
        for name, (status, reason, mutate) in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = write_phase_c_report(root, self.candidate_identity, mutate=mutate)
                with self.authority(root):
                    eligibility = self.evaluate(root, path)
                self.assertEqual(eligibility["phase_c"]["status"], status)
                self.assertEqual(eligibility["phase_c"]["reason_codes"], [reason])

    def test_canonical_input_substitution_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = write_phase_c_report(root, self.candidate_identity)
            unrelated = root / "unrelated.txt"
            unrelated.write_text("candidate source\n", encoding="utf-8")
            report = json.loads(path.read_text(encoding="utf-8"))
            report["inputs"][0]["path"] = "unrelated.txt"
            path.write_text(json.dumps(report), encoding="utf-8")

            with self.authority(root):
                eligibility = self.evaluate(root, path)

        self.assertIn(eligibility["phase_c"]["status"], {"invalid", "stale"})
        self.assertNotEqual(eligibility["phase_c"]["status"], "pass")

    def test_result_validation_rejects_changed_or_removed_adapter(self) -> None:
        for mutation in ("changed", "removed"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                adapter = root / "candidate_adapter.py"
                adapter.write_text("def create(context):\n    return context\n", encoding="utf-8")
                with mock.patch.object(sys, "path", [str(root), *sys.path]):
                    importlib.invalidate_caches()
                    identity = candidate_artifact_identity("candidate_adapter:create")
                    self.assertTrue(identity["resolved"])
                    result = {
                        "mode": "smoke",
                        "protocol": {
                            "candidate_adapter": "candidate_adapter:create",
                            "candidate_identity": identity,
                            "case_boundaries": {"image": "encoded_to_numpy"},
                        },
                        "release_eligibility": {
                            "releasable": False,
                            "performance_status": "diagnostic_only",
                            "phase_c": {
                                "status": "missing",
                                "reason_codes": ["report_missing"],
                            },
                        },
                    }
                    _validate_release_eligibility(result)
                    if mutation == "changed":
                        adapter.write_text(
                            "def create(context):\n    return {'changed': context}\n",
                            encoding="utf-8",
                        )
                    else:
                        adapter.unlink()
                    importlib.invalidate_caches()
                    with self.assertRaisesRegex(BenchmarkProtocolError, "artifact changed"):
                        _validate_release_eligibility(result)

    def test_text_only_scope_does_not_require_phase_c(self) -> None:
        eligibility = evaluate_image_release_gate(
            mode="dedicated",
            selected_cases=[{"case_id": "text", "boundary": "structured_messages_to_numpy"}],
            candidate_identity=self.candidate_identity,
            phase_c_report_path=None,
        )

        self.assertEqual(eligibility["phase_c"]["status"], "not_applicable")
        self.assertFalse(eligibility["releasable"])


class ProtocolTests(unittest.TestCase):
    def test_worker_records_complete_metrics_and_conformance(self) -> None:
        workload = load_workload(WORKLOAD_PATH)
        case = select_cases(workload, case_ids=["text_short"])[0]

        result = run_worker(worker_config(), case)

        self.assertEqual(result["conformance"]["pre_measurement"], "pass")
        self.assertTrue(result["conformance"]["all_measured_iterations_stable"])
        self.assertEqual(result["conformance"]["post_measurement"], "pass")
        self.assertEqual(len(result["samples"]), 3)
        sample = result["samples"][0]
        for field in (
            "wall_ms",
            "cpu_ms",
            "throughput_per_s",
            "core_utilization",
            "peak_rss_bytes",
            "transient_live_bytes",
            "allocation_count",
            "copy_count",
        ):
            self.assertIn(field, sample)
        self.assertFalse(result["summary"]["wall_ms"]["p99_qualified"])
        self.assertIsNone(result["summary"]["wall_ms"]["p99"])

    def test_schedule_and_bootstrap_are_seeded(self) -> None:
        first = randomized_orders(5, seed=7)
        second = randomized_orders(5, seed=7)
        self.assertEqual(first, second)
        self.assertIn(["reference", "candidate"], first)
        self.assertIn(["candidate", "reference"], first)

        interval = bootstrap_confidence_interval([1.0, 2.0, 3.0], seed=11, resamples=200)
        self.assertLessEqual(interval["lower"], 2.0)
        self.assertGreaterEqual(interval["upper"], 2.0)

    def test_synthetic_pair_runs_in_fresh_subprocesses_and_renders_report(self) -> None:
        result = run_benchmark(
            workload_path=WORKLOAD_PATH,
            mode="smoke",
            reference_adapter="synthetic",
            candidate_adapter="synthetic",
            profiles=["qwen3-vl-8b"],
            case_ids=["text_short"],
            process_repetitions=2,
            warmups=0,
            minimum_samples=2,
            minimum_seconds=0.0,
            thread_regimes=["one"],
            build_labels=["shipping"],
            seed=13,
        )

        validate_result(result)
        self.assertEqual(len(result["pairs"]), 2)
        process_ids = {
            implementation["process_id"]
            for pair in result["pairs"]
            for implementation in pair["implementations"].values()
        }
        self.assertEqual(len(process_ids), 4)
        self.assertTrue(result["protocol"]["self_test_only"])
        self.assertEqual(result["release_eligibility"]["phase_c"]["status"], "not_applicable")
        self.assertFalse(result["release_eligibility"]["releasable"])
        report = render_report(result)
        self.assertIn("paired benchmark v2", report)
        self.assertIn("qwen3-vl-8b", report)
        self.assertIn("D4 owns release performance gates", report)
        self.assertIn("DIAGNOSTIC ONLY", report)
        self.assertIn("Releasable: `false`", report)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            self.assertGreater(path.stat().st_size, 0)

    def test_synthetic_image_smoke_is_usable_but_never_release_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing_report = Path(temporary) / "missing-phase-c-report.json"
            result = run_benchmark(
                workload_path=WORKLOAD_PATH,
                mode="smoke",
                reference_adapter="synthetic",
                candidate_adapter="synthetic",
                profiles=["qwen3-vl-8b"],
                case_ids=["image1"],
                process_repetitions=1,
                warmups=0,
                minimum_samples=1,
                minimum_seconds=0.0,
                seed=19,
                phase_c_report_path=missing_report,
            )

            validate_result(result)
            eligibility = result["release_eligibility"]
            self.assertEqual(eligibility["phase_c"]["status"], "missing")
            self.assertFalse(eligibility["releasable"])
            self.assertIn("DIAGNOSTIC ONLY", render_report(result))

            forged = copy.deepcopy(result)
            forged["release_eligibility"]["phase_c"].update(
                status="pass",
                reason_codes=[],
                report_sha256="a" * 64,
                gate_fingerprint="b" * 64,
                evidence={},
            )
            with self.assertRaisesRegex(BenchmarkProtocolError, "stale or has been tampered"):
                validate_result(forged)

    def test_result_validation_binds_workload_protocol_pairs_and_workers(self) -> None:
        result = run_benchmark(
            workload_path=WORKLOAD_PATH,
            mode="smoke",
            reference_adapter="synthetic",
            candidate_adapter="synthetic",
            profiles=["qwen3-vl-8b"],
            case_ids=["image1"],
            process_repetitions=1,
            warmups=0,
            minimum_samples=1,
            minimum_seconds=0.0,
            seed=29,
        )

        def gate_bypass(value: dict[str, Any]) -> None:
            value["protocol"]["case_boundaries"]["image1"] = "structured_messages_to_numpy"
            value["release_eligibility"]["phase_c"].update(status="not_applicable", reason_codes=[])

        def pair_relabel(value: dict[str, Any]) -> None:
            value["pairs"][0]["boundary"] = "structured_messages_to_numpy"

        def protocol_case_relabel(value: dict[str, Any]) -> None:
            value["protocol"]["cases"] = ["text_short"]
            value["protocol"]["case_boundaries"] = {"text_short": "structured_messages_to_numpy"}

        def worker_relabel(value: dict[str, Any]) -> None:
            value["pairs"][0]["implementations"]["candidate"]["case_id"] = "text_short"

        def workload_hash_relabel(value: dict[str, Any]) -> None:
            value["workload"]["sha256"] = "0" * 64

        def workload_path_relabel(value: dict[str, Any]) -> None:
            value["workload"]["path"] = "benchmarks/workload-schema-v2.json"
            value["workload"]["sha256"] = hashlib.sha256(
                WORKLOAD_SCHEMA_PATH.read_bytes()
            ).hexdigest()

        def summary_relabel(value: dict[str, Any]) -> None:
            value["summaries"][0]["case_id"] = "text_short"

        def pair_budget_relabel(value: dict[str, Any]) -> None:
            pair = value["pairs"][0]
            pair["thread_budget"] = 2
            for implementation in pair["implementations"].values():
                implementation["thread_settings"]["budget"] = 2
                implementation["thread_settings"]["environment"] = {
                    name: "2" for name in implementation["thread_settings"]["environment"]
                }

        def worker_thread_environment_relabel(value: dict[str, Any]) -> None:
            value["pairs"][0]["implementations"]["candidate"]["thread_settings"]["environment"][
                "OMP_NUM_THREADS"
            ] = "99"

        mutations = {
            "gate-bypass": gate_bypass,
            "pair": pair_relabel,
            "protocol-case": protocol_case_relabel,
            "worker": worker_relabel,
            "workload-hash": workload_hash_relabel,
            "workload-path": workload_path_relabel,
            "summary": summary_relabel,
            "pair-budget": pair_budget_relabel,
            "worker-thread-environment": worker_thread_environment_relabel,
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                forged = copy.deepcopy(result)
                mutate(forged)
                with self.assertRaises(BenchmarkProtocolError):
                    validate_result(forged)

    def test_result_validation_rejects_release_status_tampering(self) -> None:
        result = run_benchmark(
            workload_path=WORKLOAD_PATH,
            mode="smoke",
            reference_adapter="synthetic",
            candidate_adapter="synthetic",
            profiles=["qwen3-vl-8b"],
            case_ids=["text_short"],
            process_repetitions=1,
            warmups=0,
            minimum_samples=1,
            minimum_seconds=0.0,
            seed=23,
        )
        result["release_eligibility"]["releasable"] = True

        with self.assertRaisesRegex(BenchmarkProtocolError, "unreleasable"):
            validate_result(result)

    def test_result_validation_rejects_cross_input_pair(self) -> None:
        result = run_benchmark(
            workload_path=WORKLOAD_PATH,
            mode="smoke",
            reference_adapter="synthetic",
            candidate_adapter="synthetic",
            profiles=["qwen3-vl-8b"],
            case_ids=["text_short"],
            process_repetitions=1,
            warmups=0,
            minimum_samples=1,
            minimum_seconds=0.0,
            seed=17,
        )
        result["pairs"][0]["implementations"]["candidate"]["input_fingerprint"] = "broken"

        with self.assertRaisesRegex(BenchmarkProtocolError, "relabeled|different inputs"):
            validate_result(result)


if __name__ == "__main__":
    unittest.main()
