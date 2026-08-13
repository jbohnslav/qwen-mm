"""Versioned still-image output comparison for the resize-v2 contract.

The final Qwen tensor is an exactly reversible representation of normalized
RGB8 values.  This module uses that property to make the resize relaxation
explicit and narrow: structure is checked by the caller, each reconstructed
RGB occurrence/channel must pass the frozen resize-v2 quality gates, and each
final tensor slice must be the exact canonical normalize/patchify result for
its reconstructed RGB8 witness.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .resize_quality_v2 import CONTRACT_ID, GATES, channel_metrics

COMPARISON_ID = "qwen-mm-still-image-final-output-comparison-v2"
CHANNELS = ("R", "G", "B")
SIGNATURE_FIELDS = {"dtype", "shape", "strides", "nbytes", "sha256"}


def _array_signature(array: np.ndarray[Any, Any]) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "strides": list(array.strides),
        "nbytes": int(array.nbytes),
        "sha256": hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest(),
    }


def unpatchify_rgb8(pixel_values: np.ndarray[Any, Any], height: int, width: int) -> np.ndarray:
    """Invert the frozen Qwen patch layout into its represented HWC RGB8 image."""

    patch = 16
    merge = 2
    if height <= 0 or width <= 0 or height % (patch * merge) or width % (patch * merge):
        raise ValueError("prepared RGB dimensions must be positive multiples of 32")
    expected_rows = (height // patch) * (width // patch)
    if pixel_values.shape != (expected_rows, 1536):
        raise ValueError(
            f"pixel_values shape {pixel_values.shape} does not match {(expected_rows, 1536)}"
        )
    output = np.empty((height, width, 3), dtype=np.uint8)
    row = 0
    for outer_y in range(height // (patch * merge)):
        for outer_x in range(width // (patch * merge)):
            for merge_y in range(merge):
                for merge_x in range(merge):
                    values = pixel_values[row].reshape(3, 2, patch, patch)
                    if not np.array_equal(values[:, 0], values[:, 1]):
                        raise ValueError(f"image temporal planes differ at patch row {row}")
                    prepared = np.rint(values[:, 0] * np.float32(127.5) + np.float32(127.5))
                    prepared = np.clip(prepared, 0, 255).astype(np.uint8).transpose(1, 2, 0)
                    y = (outer_y * merge + merge_y) * patch
                    x = (outer_x * merge + merge_x) * patch
                    output[y : y + patch, x : x + patch] = prepared
                    row += 1
    return output


def patchify_rgb8(prepared: np.ndarray[Any, Any]) -> np.ndarray:
    """Apply the frozen RGB8 normalize/duplicate/patchify transform exactly."""

    if (
        prepared.dtype != np.uint8
        or prepared.ndim != 3
        or prepared.shape[2] != 3
        or prepared.shape[0] % 32
        or prepared.shape[1] % 32
    ):
        raise ValueError("prepared RGB must be uint8 HWC3 with dimensions divisible by 32")
    patch = 16
    rows: list[np.ndarray[Any, Any]] = []
    for outer_y in range(prepared.shape[0] // 32):
        for outer_x in range(prepared.shape[1] // 32):
            for merge_y in range(2):
                for merge_x in range(2):
                    y = (outer_y * 2 + merge_y) * patch
                    x = (outer_x * 2 + merge_x) * patch
                    values = prepared[y : y + patch, x : x + patch].transpose(2, 0, 1)
                    normalized = (values.astype(np.float32) - np.float32(127.5)) / np.float32(127.5)
                    rows.append(np.repeat(normalized[:, None], 2, axis=1).reshape(-1))
    return np.stack(rows).astype(np.float32, copy=False)


def compare_prepared_occurrence(
    reference: np.ndarray[Any, Any],
    candidate: np.ndarray[Any, Any],
    *,
    occurrence: int,
) -> dict[str, Any]:
    """Return the mandatory per-channel resize-v2 quality witness."""

    if (
        reference.dtype != np.uint8
        or candidate.dtype != np.uint8
        or reference.ndim != 3
        or reference.shape[-1:] != (3,)
        or reference.shape != candidate.shape
    ):
        raise ValueError("resize-v2 occurrence comparison requires equal-shape HWC RGB8")
    channels = [
        {
            "channel": name,
            **channel_metrics(reference[..., index], candidate[..., index]),
        }
        for index, name in enumerate(CHANNELS)
    ]
    return {
        "occurrence": occurrence,
        "shape": list(reference.shape),
        "channels": channels,
        "passed": all(channel["passed"] for channel in channels),
    }


def compare_final_tensors(
    reference: np.ndarray[Any, Any],
    candidate: np.ndarray[Any, Any],
    grids: np.ndarray[Any, Any],
    *,
    source_dimensions: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Compare final tensors under resize-v2 and return an auditable witness.

    Dtypes, shapes, strides, keys, and grids are deliberately caller-owned
    exact invariants.  This function rejects any non-canonical downstream
    transform instead of treating final-tensor error as another tolerance.
    """

    if reference.dtype != np.float32 or candidate.dtype != np.float32:
        raise ValueError("resize-v2 final tensors must both be float32")
    if reference.shape != candidate.shape or reference.strides != candidate.strides:
        raise ValueError("resize-v2 final tensor shape or strides differ")
    if grids.dtype != np.int64 or grids.ndim != 2 or grids.shape[1:] != (3,):
        raise ValueError("resize-v2 grids must be an Nx3 int64 array")
    if len(source_dimensions) != len(grids) or any(
        not isinstance(dimensions, tuple)
        or len(dimensions) != 2
        or any(
            isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
            for dimension in dimensions
        )
        for dimensions in source_dimensions
    ):
        raise ValueError("resize-v2 source dimensions must cover every occurrence")

    occurrences: list[dict[str, Any]] = []
    cursor = 0
    for occurrence, row in enumerate(grids):
        temporal, grid_height, grid_width = (int(value) for value in row)
        if temporal != 1 or grid_height <= 0 or grid_width <= 0:
            raise ValueError("resize-v2 still-image grid must have positive shape and temporal=1")
        count = temporal * grid_height * grid_width
        end = cursor + count
        if end > len(reference):
            raise ValueError("resize-v2 occurrence rows exceed final tensor")
        height = grid_height * 16
        width = grid_width * 16
        reference_slice = reference[cursor:end]
        candidate_slice = candidate[cursor:end]
        reference_rgb = unpatchify_rgb8(reference_slice, height, width)
        candidate_rgb = unpatchify_rgb8(candidate_slice, height, width)
        occurrence_witness = compare_prepared_occurrence(
            reference_rgb, candidate_rgb, occurrence=occurrence
        )
        source_height, source_width = source_dimensions[occurrence]
        no_op = (source_height, source_width) == (height, width)
        no_op_exact = not no_op or bool(np.array_equal(reference_rgb, candidate_rgb))
        downstream_exact = bool(
            np.array_equal(patchify_rgb8(reference_rgb), reference_slice)
            and np.array_equal(patchify_rgb8(candidate_rgb), candidate_slice)
        )
        occurrence_witness["pixel_rows"] = [cursor, end]
        occurrence_witness["grid"] = [temporal, grid_height, grid_width]
        occurrence_witness["source_shape"] = [source_height, source_width, 3]
        occurrence_witness["comparison_mode"] = "no_op_exact" if no_op else "resize_quality"
        occurrence_witness["no_op_exact"] = no_op_exact
        occurrence_witness["downstream_transform_exact"] = downstream_exact
        occurrence_witness["passed"] = (
            occurrence_witness["passed"] and downstream_exact and no_op_exact
        )
        occurrences.append(occurrence_witness)
        cursor = end
    if cursor != len(reference):
        raise ValueError("resize-v2 grids do not account for every final tensor row")
    return {
        "comparison_id": COMPARISON_ID,
        "contract_id": CONTRACT_ID,
        "unit": "each still-image occurrence and each R/G/B channel independently",
        "gates": {
            "maximum_absolute_error_max": GATES["maximum_absolute_error"],
            "rmse_max": GATES["rmse"],
            "absolute_error_p99_max_nearest_rank": GATES["absolute_error_p99"],
            "absolute_signed_mean_bias_max": GATES["absolute_signed_mean_bias"],
            "canonical_windowed_ssim_min": GATES["ssim"],
        },
        "bindings": {
            "reference_pixel_values": _array_signature(reference),
            "candidate_pixel_values": _array_signature(candidate),
            "image_grid_thw": {
                **_array_signature(grids),
                "rows": grids.tolist(),
            },
            "occurrence_count": len(grids),
            "pixel_row_count": len(reference),
        },
        "occurrences": occurrences,
        "passed": all(item["passed"] for item in occurrences),
    }


def _validate_signature(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != SIGNATURE_FIELDS:
        raise ValueError(f"resize-v2 {name} signature is incomplete")
    dtype = value.get("dtype")
    shape = value.get("shape")
    strides = value.get("strides")
    nbytes = value.get("nbytes")
    digest = value.get("sha256")
    if (
        not isinstance(dtype, str)
        or not isinstance(shape, list)
        or not shape
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape)
        or not isinstance(strides, list)
        or len(strides) != len(shape)
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in strides)
        or isinstance(nbytes, bool)
        or not isinstance(nbytes, int)
        or nbytes < 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"resize-v2 {name} signature is invalid")
    try:
        itemsize = np.dtype(dtype).itemsize
    except TypeError as error:
        raise ValueError(f"resize-v2 {name} dtype is invalid") from error
    expected_nbytes = math.prod(shape) * itemsize
    expected_strides: list[int] = []
    stride = itemsize
    for extent in reversed(shape):
        expected_strides.append(stride)
        stride *= extent
    if nbytes != expected_nbytes or strides != list(reversed(expected_strides)):
        raise ValueError(f"resize-v2 {name} size or strides are inconsistent")
    return value


def validate_witness(
    value: Mapping[str, Any],
    *,
    reference_output_signature: Mapping[str, Any],
    candidate_output_signature: Mapping[str, Any],
    source_dimensions: Sequence[tuple[int, int]],
) -> None:
    """Validate a serialized witness without access to discarded output arrays."""

    if any(
        not isinstance(dimensions, tuple)
        or len(dimensions) != 2
        or any(
            isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
            for dimension in dimensions
        )
        for dimensions in source_dimensions
    ):
        raise ValueError("resize-v2 authenticated source dimensions are invalid")

    if (
        value.get("comparison_id") != COMPARISON_ID
        or value.get("contract_id") != CONTRACT_ID
        or value.get("unit") != "each still-image occurrence and each R/G/B channel independently"
        or value.get("passed") is not True
        or value.get("gates")
        != {
            "maximum_absolute_error_max": GATES["maximum_absolute_error"],
            "rmse_max": GATES["rmse"],
            "absolute_error_p99_max_nearest_rank": GATES["absolute_error_p99"],
            "absolute_signed_mean_bias_max": GATES["absolute_signed_mean_bias"],
            "canonical_windowed_ssim_min": GATES["ssim"],
        }
    ):
        raise ValueError("resize-v2 witness identity, gates, or status is invalid")
    bindings = value.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "reference_pixel_values",
        "candidate_pixel_values",
        "image_grid_thw",
        "occurrence_count",
        "pixel_row_count",
    }:
        raise ValueError("resize-v2 witness bindings are incomplete")
    reference_binding = _validate_signature(
        bindings["reference_pixel_values"], name="reference pixel"
    )
    candidate_binding = _validate_signature(
        bindings["candidate_pixel_values"], name="candidate pixel"
    )
    grid_binding = bindings["image_grid_thw"]
    if not isinstance(grid_binding, Mapping) or set(grid_binding) != SIGNATURE_FIELDS | {"rows"}:
        raise ValueError("resize-v2 grid binding is incomplete")
    grid_signature = _validate_signature(
        {key: grid_binding[key] for key in SIGNATURE_FIELDS}, name="grid"
    )
    if reference_binding != reference_output_signature.get("pixel_values"):
        raise ValueError("resize-v2 reference pixels are not bound to paired output")
    if candidate_binding != candidate_output_signature.get("pixel_values"):
        raise ValueError("resize-v2 candidate pixels are not bound to paired output")
    if grid_signature != reference_output_signature.get(
        "image_grid_thw"
    ) or grid_signature != candidate_output_signature.get("image_grid_thw"):
        raise ValueError("resize-v2 grid is not bound exactly to both paired outputs")
    if (
        reference_binding["dtype"] != "float32"
        or candidate_binding["dtype"] != "float32"
        or reference_binding["shape"] != candidate_binding["shape"]
        or reference_binding["shape"][1:] != [1536]
        or reference_binding["strides"] != candidate_binding["strides"]
        or grid_signature["dtype"] != "int64"
        or grid_signature["shape"][1:] != [3]
    ):
        raise ValueError("resize-v2 bound pixel/grid layouts are invalid")
    rows = grid_binding.get("rows")
    if not isinstance(rows, list) or any(
        not isinstance(row, list)
        or len(row) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in row)
        for row in rows
    ):
        raise ValueError("resize-v2 bound grid rows are invalid")
    grid_array = np.asarray(rows, dtype=np.int64)
    if _array_signature(grid_array) != grid_signature:
        raise ValueError("resize-v2 bound grid content differs from its signature")
    occurrence_count = len(rows)
    if (
        isinstance(bindings.get("occurrence_count"), bool)
        or not isinstance(bindings.get("occurrence_count"), int)
        or bindings.get("occurrence_count") != occurrence_count
        or len(source_dimensions) != occurrence_count
        or isinstance(bindings.get("pixel_row_count"), bool)
        or not isinstance(bindings.get("pixel_row_count"), int)
    ):
        raise ValueError("resize-v2 bound occurrence count is invalid")
    occurrences = value.get("occurrences")
    if not isinstance(occurrences, Sequence) or len(occurrences) != occurrence_count:
        raise ValueError("resize-v2 witness occurrence inventory is invalid")
    cursor = 0
    for index, occurrence in enumerate(occurrences):
        if (
            not isinstance(occurrence, Mapping)
            or occurrence.get("occurrence") != index
            or occurrence.get("passed") is not True
            or occurrence.get("downstream_transform_exact") is not True
        ):
            raise ValueError("resize-v2 occurrence witness did not pass")
        shape = occurrence.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or shape[2] != 3
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape
            )
            or shape[0] % 32
            or shape[1] % 32
        ):
            raise ValueError("resize-v2 occurrence shape is invalid")
        row_count = (shape[0] // 16) * (shape[1] // 16)
        grid = rows[index]
        source_height, source_width = source_dimensions[index]
        expected_shape = [grid[1] * 16, grid[2] * 16, 3]
        no_op = [source_height, source_width, 3] == expected_shape
        if (
            occurrence.get("grid") != grid
            or shape != expected_shape
            or occurrence.get("source_shape") != [source_height, source_width, 3]
            or occurrence.get("comparison_mode") != ("no_op_exact" if no_op else "resize_quality")
            or occurrence.get("no_op_exact") is not True
        ):
            raise ValueError("resize-v2 occurrence geometry or no-op classification is invalid")
        if occurrence.get("pixel_rows") != [cursor, cursor + row_count]:
            raise ValueError("resize-v2 occurrence pixel rows are invalid")
        cursor += row_count
        channels = occurrence.get("channels")
        if (
            not isinstance(channels, list)
            or not all(isinstance(item, Mapping) for item in channels)
            or [item.get("channel") for item in channels] != list(CHANNELS)
        ):
            raise ValueError("resize-v2 per-channel witness is incomplete")
        for channel in channels:
            metrics = channel if isinstance(channel, Mapping) else {}
            gates = metrics.get("gates")
            if metrics.get("passed") is not True or not isinstance(gates, Mapping):
                raise ValueError("resize-v2 channel witness did not pass")
            required = {
                "maximum_absolute_error",
                "rmse",
                "absolute_error_p99",
                "signed_mean_bias",
                "absolute_signed_mean_bias",
                "ssim",
                "mae",
                "psnr",
                "absolute_error_p50",
                "absolute_error_p90",
            }
            if set(metrics) != required | {"channel", "gates", "passed"}:
                raise ValueError("resize-v2 required channel diagnostics are incomplete")
            numeric_fields = required - {"psnr"}
            if any(
                isinstance(metrics[field], bool)
                or not isinstance(metrics[field], (int, float))
                or not math.isfinite(float(metrics[field]))
                for field in numeric_fields
            ):
                raise ValueError("resize-v2 channel diagnostics are not finite numbers")
            psnr = metrics["psnr"]
            if psnr != "Infinity" and (
                isinstance(psnr, bool)
                or not isinstance(psnr, (int, float))
                or not math.isfinite(float(psnr))
            ):
                raise ValueError("resize-v2 PSNR diagnostic is invalid")
            if not (
                0.0
                <= metrics["absolute_error_p50"]
                <= metrics["absolute_error_p90"]
                <= metrics["absolute_error_p99"]
                <= metrics["maximum_absolute_error"]
                <= 255.0
                and 0.0 <= metrics["mae"] <= metrics["rmse"] <= 255.0
                and -255.0 <= metrics["signed_mean_bias"] <= 255.0
                and math.isclose(
                    metrics["absolute_signed_mean_bias"],
                    abs(metrics["signed_mean_bias"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and metrics["absolute_signed_mean_bias"] <= metrics["mae"] + 1e-12
                and -1.0 <= metrics["ssim"] <= 1.0
            ):
                raise ValueError("resize-v2 channel diagnostics are inconsistent")
            expected_psnr: float | str = (
                "Infinity" if metrics["rmse"] == 0.0 else 20.0 * math.log10(255.0 / metrics["rmse"])
            )
            if expected_psnr == "Infinity":
                if psnr != expected_psnr:
                    raise ValueError("resize-v2 PSNR diagnostic is inconsistent")
            elif not isinstance(psnr, (int, float)) or not math.isclose(
                float(psnr), expected_psnr, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError("resize-v2 PSNR diagnostic is inconsistent")
            expected_gates = {
                "maximum_absolute_error": metrics["maximum_absolute_error"]
                <= GATES["maximum_absolute_error"],
                "rmse": metrics["rmse"] <= GATES["rmse"],
                "absolute_error_p99": metrics["absolute_error_p99"] <= GATES["absolute_error_p99"],
                "absolute_signed_mean_bias": metrics["absolute_signed_mean_bias"]
                <= GATES["absolute_signed_mean_bias"],
                "ssim": metrics["ssim"] >= GATES["ssim"],
            }
            if gates != expected_gates or not all(expected_gates.values()):
                raise ValueError("resize-v2 channel gate inventory is invalid")
    if cursor != bindings.get("pixel_row_count") or cursor != reference_binding["shape"][0]:
        raise ValueError("resize-v2 bound pixel row count is invalid")
