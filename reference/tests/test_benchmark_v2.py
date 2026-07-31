from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
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
    bootstrap_confidence_interval,
    randomized_orders,
    render_report,
    run_benchmark,
    validate_result,
)
from qwen_mm_reference.fixtures import repository_root

WORKLOAD_PATH = repository_root() / "benchmarks" / "workloads-v2.json"


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


class WorkloadTests(unittest.TestCase):
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

    def test_raw_rgb_payload_is_caller_owned_read_only_uint8(self) -> None:
        workload = load_workload(WORKLOAD_PATH)
        case = select_cases(workload, case_ids=["rgb24"])[0]
        case["source"] = copy.deepcopy(case["source"])
        case["source"].update(count=2, shapes=[[17, 19]])

        payload = materialize_case(case)

        self.assertEqual(payload.boundary, "rgb_to_vllm_ready")
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
        report = render_report(result)
        self.assertIn("paired benchmark v2", report)
        self.assertIn("qwen3-vl-8b", report)
        self.assertIn("D4 owns release performance gates", report)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            self.assertGreater(path.stat().st_size, 0)

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

        with self.assertRaisesRegex(BenchmarkProtocolError, "different inputs"):
            validate_result(result)


if __name__ == "__main__":
    unittest.main()
