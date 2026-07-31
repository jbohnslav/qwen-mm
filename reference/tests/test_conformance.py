from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from qwen_mm_reference.conformance import (
    _output_tolerance,
    check_invariants,
    compare_manifests,
)
from qwen_mm_reference.corpus import (
    generate_seeded_cases,
    load_and_validate,
    minimize_case,
    promote_case,
)
from qwen_mm_reference.fixtures import repository_root


def load_manifest(profile: str, case: str) -> dict:
    path = repository_root() / "reference" / "goldens" / "v1" / profile / case / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


class ComparatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.expected = load_manifest("qwen3-vl-8b", "multimodal-smoke")

    def test_identical_manifest_passes(self) -> None:
        report = compare_manifests(self.expected, copy.deepcopy(self.expected))

        self.assertTrue(report.passed, report.to_dict())

    def test_output_key_order_is_exact(self) -> None:
        actual = copy.deepcopy(self.expected)
        actual["output"]["keys"][3:5] = reversed(actual["output"]["keys"][3:5])

        report = compare_manifests(self.expected, actual)

        self.assertFalse(report.passed)
        self.assertIn("key_or_order_mismatch", {issue["kind"] for issue in report.issues})

    def test_dtype_and_stride_mismatches_are_localized(self) -> None:
        actual = copy.deepcopy(self.expected)
        descriptor = actual["output"]["arrays"]["input_ids"]
        descriptor["dtype"] = "int32"
        descriptor["strides"] = [descriptor["strides"][0] // 2, 4]

        report = compare_manifests(self.expected, actual)

        paths = {issue["path"] for issue in report.issues}
        self.assertIn("output.arrays.input_ids.dtype", paths)
        self.assertIn("output.arrays.input_ids.strides", paths)

    def test_pixel_mismatch_has_required_diagnostics_and_coordinate(self) -> None:
        actual = copy.deepcopy(self.expected)
        descriptor = actual["output"]["arrays"]["pixel_values"]
        descriptor["data"][3][300] += 0.25

        report = compare_manifests(self.expected, actual)

        issue = next(issue for issue in report.issues if issue["kind"] == "numeric_array_mismatch")
        self.assertEqual(issue["stage"], "outputs")
        self.assertGreater(issue["maximum_absolute_error"], 0.2)
        self.assertGreater(issue["maximum_ulp_error"], 0)
        self.assertGreater(issue["rmse"], 0)
        self.assertEqual(set(issue["absolute_error_percentiles"]), {"p50", "p90", "p99"})
        self.assertEqual(issue["count_over_threshold"], 1)
        coordinate = issue["first_difference"]
        self.assertEqual(coordinate["patch"], 3)
        self.assertEqual(coordinate["column"], 300)
        self.assertEqual(coordinate["channel"], 1)
        self.assertEqual(coordinate["media_occurrence"], 0)
        self.assertEqual(coordinate["request_index"], 0)

    def test_pixel_difference_inside_frozen_bound_passes(self) -> None:
        actual = copy.deepcopy(self.expected)
        actual["output"]["arrays"]["pixel_values"]["data"][0][0] += 0.001

        report = compare_manifests(self.expected, actual)

        self.assertTrue(report.passed, report.to_dict())

    def test_mixed_lossless_and_lossy_images_get_per_occurrence_bounds(self) -> None:
        manifest = copy.deepcopy(self.expected)
        first = manifest["input"]["media"][0]
        first["source_properties"] = {"format": "PNG"}
        second = copy.deepcopy(first)
        second["source_properties"] = {"format": "JPEG"}
        manifest["input"]["media"].append(second)
        grid = manifest["output"]["arrays"]["image_grid_thw"]
        grid["data"].append(grid["data"][0])

        atol, rtol = _output_tolerance(manifest, "pixel_values")

        self.assertEqual(rtol, 0.0)
        self.assertEqual(atol.shape, (32, 1))
        self.assertEqual(float(atol[0, 0]), 1e-6)
        self.assertGreater(float(atol[-1, 0]), 0.007)

    def test_expected_errors_compare_by_stable_category_only(self) -> None:
        expected = load_manifest("qwen3-vl-8b", "corrupt-image")
        actual = copy.deepcopy(expected)
        actual["error"]["message"] = "candidate-specific diagnostic"
        actual["error"]["exception_type"] = "RustMediaError"

        self.assertTrue(compare_manifests(expected, actual).passed)
        actual["error"]["category"] = "unsupported_media"
        report = compare_manifests(expected, actual)
        self.assertIn("exact_mismatch", {issue["kind"] for issue in report.issues})

    def test_signature_only_mismatch_requests_materialized_values(self) -> None:
        expected = load_manifest("qwen3-vl-8b", "image24")
        actual = copy.deepcopy(expected)
        actual["output"]["arrays"]["pixel_values"]["sha256"] = "0" * 64

        report = compare_manifests(expected, actual)

        issue = next(
            issue for issue in report.issues if issue["kind"] == "array_values_unavailable"
        )
        self.assertIn("materialize", issue["hint"])

    def test_placeholder_invariant_detects_corruption(self) -> None:
        actual = copy.deepcopy(self.expected)
        prompt = actual["stages"]["expanded_prompts"][0]
        prompt["text"] = prompt["text"].replace("<|image_pad|>", "", 1)

        report = check_invariants(actual)

        self.assertIn("placeholder_patch_invariant", {issue["kind"] for issue in report.issues})


class CorpusTests(unittest.TestCase):
    def test_catalog_has_complete_executable_coverage(self) -> None:
        catalog, rules = load_and_validate()

        self.assertGreaterEqual(len(catalog["entries"]), 15)
        self.assertGreaterEqual(len(rules["rules"]), 24)

    def test_seeded_live_cases_are_reproducible_and_order_sensitive(self) -> None:
        first = generate_seeded_cases(12345, 4)
        second = generate_seeded_cases(12345, 4)
        different = generate_seeded_cases(54321, 4)

        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertEqual(first[0]["generation"]["seed"], 12345)

    def test_minimizer_removes_irrelevant_requests_messages_and_content(self) -> None:
        case = {
            "schema_version": 1,
            "case_id": "failure",
            "requests": [
                {"messages": [{"role": "user", "content": [{"type": "text", "text": "ok"}]}]},
                {
                    "messages": [
                        {"role": "system", "content": "irrelevant"},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "irrelevant"},
                                {"type": "text", "text": "TRIGGER"},
                            ],
                        },
                    ]
                },
            ],
        }

        def fails(candidate: dict) -> bool:
            return "TRIGGER" in json.dumps(candidate)

        minimized = minimize_case(case, fails)

        self.assertTrue(fails(minimized))
        self.assertEqual(len(minimized["requests"]), 1)
        self.assertEqual(len(minimized["requests"][0]["messages"]), 1)
        self.assertEqual(len(minimized["requests"][0]["messages"][0]["content"]), 1)
        self.assertEqual(minimized["minimized_from"], "failure")

    def test_promote_writes_canonical_regression_without_overwrite(self) -> None:
        case = generate_seeded_cases(7, 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            destination = promote_case(case, "regression-7", Path(directory))
            promoted = json.loads(destination.read_text(encoding="utf-8"))

            self.assertEqual(promoted["case_id"], "regression-7")
            self.assertTrue(promoted["promoted_regression"])
            with self.assertRaises(FileExistsError):
                promote_case(case, "regression-7", Path(directory))


if __name__ == "__main__":
    unittest.main()
