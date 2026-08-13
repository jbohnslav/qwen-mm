from __future__ import annotations

import copy
import unittest

import numpy as np
from qwen_mm_reference.resize_conformance_v2 import (
    COMPARISON_ID,
    compare_final_tensors,
    patchify_rgb8,
    unpatchify_rgb8,
    validate_witness,
)


def _bound_signatures(witness: dict) -> tuple[dict, dict]:
    bindings = witness["bindings"]
    grid = {key: value for key, value in bindings["image_grid_thw"].items() if key != "rows"}
    return (
        {"pixel_values": bindings["reference_pixel_values"], "image_grid_thw": grid},
        {"pixel_values": bindings["candidate_pixel_values"], "image_grid_thw": grid},
    )


class ResizeConformanceV2Tests(unittest.TestCase):
    def test_final_tensor_comparison_reports_every_occurrence_and_channel(self) -> None:
        reference_images = [
            np.full((32, 32, 3), (32, 96, 160), dtype=np.uint8),
            np.full((32, 64, 3), (64, 128, 192), dtype=np.uint8),
        ]
        candidate_images = [image.copy() for image in reference_images]
        candidate_images[0][8:16, 8:16, 0] += 3
        candidate_images[1][4:12, 20:28, 2] -= 2
        reference = np.concatenate([patchify_rgb8(image) for image in reference_images])
        candidate = np.concatenate([patchify_rgb8(image) for image in candidate_images])
        grids = np.asarray([[1, 2, 2], [1, 2, 4]], dtype=np.int64)

        witness = compare_final_tensors(
            reference, candidate, grids, source_dimensions=[(31, 31), (31, 63)]
        )

        self.assertEqual(witness["comparison_id"], COMPARISON_ID)
        self.assertTrue(witness["passed"])
        self.assertEqual([item["occurrence"] for item in witness["occurrences"]], [0, 1])
        self.assertTrue(
            all(
                [channel["channel"] for channel in occurrence["channels"]] == ["R", "G", "B"]
                for occurrence in witness["occurrences"]
            )
        )
        self.assertTrue(
            all(occurrence["downstream_transform_exact"] for occurrence in witness["occurrences"])
        )
        reference_signature, candidate_signature = _bound_signatures(witness)
        validate_witness(
            witness,
            reference_output_signature=reference_signature,
            candidate_output_signature=candidate_signature,
            source_dimensions=[(31, 31), (31, 63)],
        )

    def test_final_tensor_comparison_rejects_noncanonical_downstream_difference(self) -> None:
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        reference = patchify_rgb8(image)
        candidate = reference.copy()
        candidate[0, 0] = np.nextafter(candidate[0, 0], np.float32(1.0))
        candidate[0, 256] = candidate[0, 0]

        witness = compare_final_tensors(
            reference,
            candidate,
            np.asarray([[1, 2, 2]], dtype=np.int64),
            source_dimensions=[(31, 31)],
        )

        self.assertTrue(witness["occurrences"][0]["channels"][0]["passed"])
        self.assertFalse(witness["occurrences"][0]["downstream_transform_exact"])
        self.assertFalse(witness["passed"])

    def test_final_tensor_comparison_rejects_quality_gate_failure(self) -> None:
        reference = patchify_rgb8(np.zeros((32, 32, 3), dtype=np.uint8))
        candidate = patchify_rgb8(np.full((32, 32, 3), 255, dtype=np.uint8))

        witness = compare_final_tensors(
            reference,
            candidate,
            np.asarray([[1, 2, 2]], dtype=np.int64),
            source_dimensions=[(31, 31)],
        )

        self.assertFalse(witness["passed"])
        self.assertTrue(
            all(not channel["passed"] for channel in witness["occurrences"][0]["channels"])
        )

    def test_witness_validator_rejects_missing_diagnostic(self) -> None:
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        tensor = patchify_rgb8(image)
        witness = compare_final_tensors(
            tensor,
            tensor.copy(),
            np.asarray([[1, 2, 2]], dtype=np.int64),
            source_dimensions=[(31, 31)],
        )
        broken = copy.deepcopy(witness)
        del broken["occurrences"][0]["channels"][0]["absolute_error_p90"]
        reference_signature, candidate_signature = _bound_signatures(witness)

        with self.assertRaisesRegex(ValueError, "diagnostics"):
            validate_witness(
                broken,
                reference_output_signature=reference_signature,
                candidate_output_signature=candidate_signature,
                source_dimensions=[(31, 31)],
            )

    def test_patch_layout_round_trip_is_exact(self) -> None:
        image = np.arange(32 * 64 * 3, dtype=np.uint32).reshape(32, 64, 3).astype(np.uint8)
        tensor = patchify_rgb8(image)
        np.testing.assert_array_equal(unpatchify_rgb8(tensor, 32, 64), image)

    def test_no_op_occurrence_rejects_any_rgb_drift(self) -> None:
        reference_rgb = np.full((32, 32, 3), 128, dtype=np.uint8)
        candidate_rgb = reference_rgb.copy()
        candidate_rgb[0, 0, 0] += 1
        witness = compare_final_tensors(
            patchify_rgb8(reference_rgb),
            patchify_rgb8(candidate_rgb),
            np.asarray([[1, 2, 2]], dtype=np.int64),
            source_dimensions=[(32, 32)],
        )

        self.assertFalse(witness["passed"])
        self.assertEqual(witness["occurrences"][0]["comparison_mode"], "no_op_exact")
        self.assertFalse(witness["occurrences"][0]["no_op_exact"])

    def test_transplanted_witness_rejects_unrelated_output_signatures(self) -> None:
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        tensor = patchify_rgb8(image)
        witness = compare_final_tensors(
            tensor,
            tensor.copy(),
            np.asarray([[1, 2, 2]], dtype=np.int64),
            source_dimensions=[(31, 31)],
        )
        reference_signature, candidate_signature = _bound_signatures(witness)
        transplanted = copy.deepcopy(candidate_signature)
        transplanted["pixel_values"]["sha256"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "candidate pixels"):
            validate_witness(
                witness,
                reference_output_signature=reference_signature,
                candidate_output_signature=transplanted,
                source_dimensions=[(31, 31)],
            )

    def test_witness_validator_rejects_hostile_channel_value(self) -> None:
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        tensor = patchify_rgb8(image)
        witness = compare_final_tensors(
            tensor,
            tensor.copy(),
            np.asarray([[1, 2, 2]], dtype=np.int64),
            source_dimensions=[(31, 31)],
        )
        reference_signature, candidate_signature = _bound_signatures(witness)
        witness["occurrences"][0]["channels"][0] = "hostile"
        with self.assertRaisesRegex(ValueError, "per-channel"):
            validate_witness(
                witness,
                reference_output_signature=reference_signature,
                candidate_output_signature=candidate_signature,
                source_dimensions=[(31, 31)],
            )
