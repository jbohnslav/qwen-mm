from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from qwen_mm_reference.golden import validate_manifest

EXACT_OUTPUTS = {
    "input_ids",
    "attention_mask",
    "mm_token_type_ids",
    "image_grid_thw",
    "video_grid_thw",
}
PIXEL_OUTPUTS = {"pixel_values", "pixel_values_videos"}
STRUCTURAL_ARRAY_FIELDS = (
    "shape",
    "dtype",
    "strides",
    "byte_order",
    "c_contiguous",
    "nbytes",
)


@dataclass
class ComparisonReport:
    expected_case: str
    actual_case: str
    issues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.issues

    def add(self, stage: str, path: str, kind: str, **details: Any) -> None:
        self.issues.append(
            {
                "stage": stage,
                "path": path,
                "kind": kind,
                **details,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "expected_case": self.expected_case,
            "actual_case": self.actual_case,
            "passed": self.passed,
            "issue_count": len(self.issues),
            "issues": self.issues,
        }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _array_from_descriptor(
    descriptor: Mapping[str, Any], root: Path | None
) -> np.ndarray[Any, Any] | None:
    storage = descriptor.get("storage")
    dtype = np.dtype(descriptor["dtype"])
    if storage == "inline_json":
        return np.asarray(descriptor["data"], dtype=dtype)
    if storage == "npy":
        if root is None:
            return None
        return np.load(root / descriptor["path"], allow_pickle=False)
    return None


def _json_equal(
    report: ComparisonReport,
    stage: str,
    path: str,
    expected: Any,
    actual: Any,
) -> None:
    if expected != actual:
        report.add(stage, path, "exact_mismatch", expected=expected, actual=actual)


def _float_equal(expected: float, actual: float, atol: float) -> bool:
    return math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=atol)


def _compare_video_metadata(
    report: ComparisonReport, expected: Any, actual: Any, atol: float
) -> None:
    path = "stages.video_metadata"
    if not isinstance(expected, list) or not isinstance(actual, list):
        _json_equal(report, "video_metadata", path, expected, actual)
        return
    if len(expected) != len(actual):
        report.add(
            "video_metadata",
            path,
            "length_mismatch",
            expected=len(expected),
            actual=len(actual),
        )
        return
    float_fields = {"fps", "sample_fps", "total_num_frames"}
    for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
        if set(expected_item) != set(actual_item):
            report.add(
                "video_metadata",
                f"{path}[{index}]",
                "key_mismatch",
                expected=list(expected_item),
                actual=list(actual_item),
            )
            continue
        for key in expected_item:
            item_path = f"{path}[{index}].{key}"
            if key in float_fields:
                if not _float_equal(expected_item[key], actual_item[key], atol):
                    report.add(
                        "video_metadata",
                        item_path,
                        "numeric_mismatch",
                        expected=expected_item[key],
                        actual=actual_item[key],
                        atol=atol,
                    )
            else:
                _json_equal(
                    report,
                    "video_metadata",
                    item_path,
                    expected_item[key],
                    actual_item[key],
                )


def _ordered_float_bits(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    if values.dtype == np.float32:
        signed = values.view(np.int32).astype(np.int64)
        return np.where(signed < 0, 0x80000000 - signed, signed + 0x80000000)
    if values.dtype == np.float64:
        signed = values.view(np.int64)
        # Object integers avoid overflow at the signed endpoints. Diagnostics are
        # not on the performance path.
        objects = signed.astype(object)
        return np.where(objects < 0, (1 << 63) - objects, objects + (1 << 63))
    raise TypeError(f"ULP diagnostics require float32 or float64, got {values.dtype}")


def _max_ulp(expected: np.ndarray[Any, Any], actual: np.ndarray[Any, Any]) -> int:
    finite = np.isfinite(expected) & np.isfinite(actual)
    if not np.any(finite):
        return 0
    left = _ordered_float_bits(expected[finite])
    right = _ordered_float_bits(actual[finite])
    return int(np.max(np.abs(left - right)))


def _pixel_coordinate(
    index: tuple[int, ...],
    *,
    name: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    coordinate: dict[str, Any] = {"index": list(index)}
    if len(index) != 2 or name not in PIXEL_OUTPUTS:
        return coordinate

    patch, column = index
    coordinate.update(
        {
            "patch": patch,
            "column": column,
            "temporal_offset": column // (3 * 16 * 16),
            "channel": (column // (16 * 16)) % 3,
            "patch_y": (column // 16) % 16,
            "patch_x": column % 16,
        }
    )
    grid_name = "image_grid_thw" if name == "pixel_values" else "video_grid_thw"
    grid_descriptor = manifest.get("output", {}).get("arrays", {}).get(grid_name)
    if not grid_descriptor:
        return coordinate
    grid = _array_from_descriptor(grid_descriptor, None)
    if grid is None:
        return coordinate
    offset = 0
    for occurrence, row in enumerate(grid):
        patch_count = int(np.prod(row, dtype=np.int64))
        if patch < offset + patch_count:
            coordinate["media_occurrence"] = occurrence
            coordinate["patch_in_occurrence"] = patch - offset
            kind = "image" if name == "pixel_values" else "video"
            media = [
                item
                for item in manifest.get("input", {}).get("media", [])
                if item.get("kind") == kind
            ]
            if occurrence < len(media):
                coordinate["request_index"] = media[occurrence].get("request_index")
                coordinate["message_index"] = media[occurrence].get("message_index")
                coordinate["content_index"] = media[occurrence].get("content_index")
            break
        offset += patch_count
    return coordinate


def _numeric_diagnostics(
    expected: np.ndarray[Any, Any],
    actual: np.ndarray[Any, Any],
    *,
    atol: float | np.ndarray[Any, Any],
    rtol: float,
    name: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    expected64 = expected.astype(np.float64)
    actual64 = actual.astype(np.float64)
    absolute = np.abs(actual64 - expected64)
    bound = atol + rtol * np.abs(expected64)
    differing = ~np.isclose(actual64, expected64, rtol=rtol, atol=atol, equal_nan=True)
    first = tuple(int(value) for value in np.argwhere(differing)[0])
    finite_absolute = absolute[np.isfinite(absolute)]
    if finite_absolute.size:
        percentiles = {
            "p50": float(np.percentile(finite_absolute, 50)),
            "p90": float(np.percentile(finite_absolute, 90)),
            "p99": float(np.percentile(finite_absolute, 99)),
        }
        maximum = float(np.max(finite_absolute))
        rmse = float(np.sqrt(np.mean(np.square(finite_absolute))))
    else:
        percentiles = {"p50": math.inf, "p90": math.inf, "p99": math.inf}
        maximum = math.inf
        rmse = math.inf
    tolerance_details: dict[str, Any]
    if np.isscalar(atol):
        tolerance_details = {"atol": float(atol)}
    else:
        broadcast_atol = np.broadcast_to(atol, expected.shape)
        tolerance_details = {
            "atol_min": float(np.min(broadcast_atol)),
            "atol_max": float(np.max(broadcast_atol)),
            "atol_at_first_difference": float(broadcast_atol[first]),
        }
    return {
        **tolerance_details,
        "rtol": rtol,
        "maximum_absolute_error": maximum,
        "maximum_ulp_error": _max_ulp(expected, actual),
        "rmse": rmse,
        "absolute_error_percentiles": percentiles,
        "count_over_threshold": int(np.count_nonzero(absolute > bound)),
        "first_difference": _pixel_coordinate(first, name=name, manifest=manifest),
        "expected_at_first_difference": float(expected[first]),
        "actual_at_first_difference": float(actual[first]),
    }


def _compare_array(
    report: ComparisonReport,
    *,
    stage: str,
    path: str,
    name: str,
    expected_descriptor: Mapping[str, Any],
    actual_descriptor: Mapping[str, Any],
    expected_root: Path | None,
    actual_root: Path | None,
    exact: bool,
    atol: float | np.ndarray[Any, Any] = 0.0,
    rtol: float = 0.0,
    manifest: Mapping[str, Any],
) -> None:
    structural_mismatch = False
    for field_name in STRUCTURAL_ARRAY_FIELDS:
        expected_value = expected_descriptor.get(field_name)
        actual_value = actual_descriptor.get(field_name)
        if expected_value != actual_value:
            structural_mismatch = True
            report.add(
                stage,
                f"{path}.{field_name}",
                "array_structure_mismatch",
                expected=expected_value,
                actual=actual_value,
            )
    if structural_mismatch:
        return

    expected_array = _array_from_descriptor(expected_descriptor, expected_root)
    actual_array = _array_from_descriptor(actual_descriptor, actual_root)
    if expected_array is None or actual_array is None:
        if expected_descriptor.get("sha256") != actual_descriptor.get("sha256"):
            report.add(
                stage,
                path,
                "array_values_unavailable",
                expected_sha256=expected_descriptor.get("sha256"),
                actual_sha256=actual_descriptor.get("sha256"),
                hint="materialize both arrays as inline_json or npy for tolerant diagnostics",
            )
        return

    if exact:
        if not np.array_equal(expected_array, actual_array, equal_nan=True):
            differing = np.argwhere(expected_array != actual_array)
            first = tuple(int(value) for value in differing[0])
            report.add(
                stage,
                path,
                "exact_array_mismatch",
                first_difference={"index": list(first)},
                expected_at_first_difference=expected_array[first].item(),
                actual_at_first_difference=actual_array[first].item(),
                differing_values=int(np.count_nonzero(expected_array != actual_array)),
            )
        return

    if not np.allclose(expected_array, actual_array, rtol=rtol, atol=atol, equal_nan=True):
        report.add(
            stage,
            path,
            "numeric_array_mismatch",
            **_numeric_diagnostics(
                expected_array,
                actual_array,
                atol=atol,
                rtol=rtol,
                name=name,
                manifest=manifest,
            ),
        )


def _source_is_lossy(properties: Mapping[str, Any]) -> bool:
    format_name = str(properties.get("format", "")).upper()
    return format_name == "JPEG" or (
        format_name == "WEBP" and properties.get("lossless") is not True
    )


def _output_tolerance(
    manifest: Mapping[str, Any], name: str
) -> tuple[float | np.ndarray[Any, Any], float]:
    numeric = manifest["comparison_policy"]["numeric"]
    if name == "pixel_values_videos":
        policy = numeric["raw_frame_video_final_tensor"]
        return float(policy["atol"]), float(policy["rtol"])
    strict = numeric["normalize_and_patchify"]
    lossy = numeric["jpeg_or_lossy_webp_final_tensor"]
    strict_atol = float(strict["atol"])
    lossy_atol = float(lossy["atol_value"])
    media = [
        item for item in manifest.get("input", {}).get("media", []) if item.get("kind") == "image"
    ]
    grid_descriptor = manifest.get("output", {}).get("arrays", {}).get("image_grid_thw")
    grid = _array_from_descriptor(grid_descriptor, None) if grid_descriptor else None
    if grid is None or len(media) != len(grid):
        return (
            lossy_atol
            if any(_source_is_lossy(item.get("source_properties", {})) for item in media)
            else strict_atol
        ), 0.0
    per_patch: list[float] = []
    for item, row in zip(media, grid, strict=True):
        tolerance = (
            lossy_atol if _source_is_lossy(item.get("source_properties", {})) else strict_atol
        )
        per_patch.extend([tolerance] * int(np.prod(row, dtype=np.int64)))
    return np.asarray(per_patch, dtype=np.float64)[:, np.newaxis], 0.0


def _compare_prepared_media(
    report: ComparisonReport,
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    expected_root: Path | None,
    actual_root: Path | None,
) -> None:
    expected_prepared = expected.get("stages", {}).get("prepared_media", {})
    actual_prepared = actual.get("stages", {}).get("prepared_media", {})
    for kind in ("images", "videos"):
        expected_items = expected_prepared.get(kind, [])
        actual_items = actual_prepared.get(kind, [])
        if len(expected_items) != len(actual_items):
            report.add(
                "prepared_media",
                f"stages.prepared_media.{kind}",
                "length_mismatch",
                expected=len(expected_items),
                actual=len(actual_items),
            )
            continue
        for index, (expected_item, actual_item) in enumerate(
            zip(expected_items, actual_items, strict=True)
        ):
            base = f"stages.prepared_media.{kind}[{index}]"
            _json_equal(
                report,
                "prepared_media",
                f"{base}.occurrence",
                expected_item.get("occurrence"),
                actual_item.get("occurrence"),
            )
            _json_equal(
                report,
                "prepared_media",
                f"{base}.layout",
                expected_item.get("layout"),
                actual_item.get("layout"),
            )
            if "array" not in expected_item or "array" not in actual_item:
                report.add(
                    "prepared_media",
                    f"{base}.array",
                    "missing_array",
                    expected="array" in expected_item,
                    actual="array" in actual_item,
                )
                continue
            if kind == "videos":
                policy = expected["comparison_policy"]["numeric"][
                    "torchvision_video_resize_0_255_float32"
                ]
                atol = float(policy["atol"])
                exact = False
            else:
                media = [
                    item
                    for item in expected.get("input", {}).get("media", [])
                    if item.get("kind") == "image"
                ]
                source = media[index].get("source_properties", {}) if index < len(media) else {}
                format_name = str(source.get("format", "")).upper()
                exact = format_name in {"PNG", "RAW", ""} or (
                    format_name == "WEBP" and source.get("lossless") is True
                )
                atol = 0.0 if exact else 1.0
            _compare_array(
                report,
                stage="prepared_media",
                path=f"{base}.array",
                name=f"prepared_{kind}",
                expected_descriptor=expected_item["array"],
                actual_descriptor=actual_item["array"],
                expected_root=expected_root,
                actual_root=actual_root,
                exact=exact,
                atol=atol,
                manifest=expected,
            )


def _check_array_descriptor(
    report: ComparisonReport,
    descriptor: Mapping[str, Any],
    path: str,
) -> None:
    try:
        dtype = np.dtype(descriptor["dtype"])
        shape = tuple(int(value) for value in descriptor["shape"])
    except (KeyError, TypeError, ValueError) as error:
        report.add("invariants", path, "invalid_array_descriptor", error=str(error))
        return
    expected_nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if descriptor.get("nbytes") != expected_nbytes:
        report.add(
            "invariants",
            f"{path}.nbytes",
            "nbytes_invariant",
            expected=expected_nbytes,
            actual=descriptor.get("nbytes"),
        )


def _check_modality_invariants(
    report: ComparisonReport,
    manifest: Mapping[str, Any],
    root: Path | None,
    *,
    kind: str,
) -> None:
    grid_name = "image_grid_thw" if kind == "image" else "video_grid_thw"
    pixel_name = "pixel_values" if kind == "image" else "pixel_values_videos"
    arrays = manifest.get("output", {}).get("arrays", {})
    occurrences = [
        item for item in manifest.get("input", {}).get("media", []) if item.get("kind") == kind
    ]
    if not occurrences:
        if grid_name in arrays or pixel_name in arrays:
            report.add(
                "invariants",
                "output.arrays",
                "unexpected_modality_output",
                modality=kind,
            )
        return
    if grid_name not in arrays or pixel_name not in arrays:
        report.add("invariants", "output.arrays", "missing_modality_output", modality=kind)
        return
    grid = _array_from_descriptor(arrays[grid_name], root)
    if grid is None:
        return
    if grid.ndim != 2 or grid.shape[1] != 3:
        report.add(
            "invariants",
            f"output.arrays.{grid_name}",
            "grid_shape_invariant",
            actual=list(grid.shape),
        )
        return
    if grid.shape[0] != len(occurrences):
        report.add(
            "invariants",
            f"output.arrays.{grid_name}",
            "grid_occurrence_invariant",
            expected=len(occurrences),
            actual=int(grid.shape[0]),
        )
    patch_rows = int(sum(int(np.prod(row, dtype=np.int64)) for row in grid))
    pixel_shape = arrays[pixel_name].get("shape", [])
    if len(pixel_shape) != 2 or pixel_shape[0] != patch_rows or pixel_shape[1] != 1536:
        report.add(
            "invariants",
            f"output.arrays.{pixel_name}.shape",
            "grid_patch_invariant",
            expected=[patch_rows, 1536],
            actual=pixel_shape,
        )
    offsets = [
        item
        for request in manifest.get("stages", {}).get("replacement_offsets", [])
        for item in request
        if item.get("type") == kind
    ]
    if len(offsets) != len(occurrences):
        report.add(
            "invariants",
            "stages.replacement_offsets",
            "placeholder_occurrence_invariant",
            modality=kind,
            expected=len(occurrences),
            actual=len(offsets),
        )
    token = "<|image_pad|>" if kind == "image" else "<|video_pad|>"
    observed_placeholders = sum(
        prompt.get("text", "").count(token)
        for prompt in manifest.get("stages", {}).get("expanded_prompts", [])
    )
    expected_placeholders = int(sum(int(np.prod(row, dtype=np.int64)) // 4 for row in grid))
    if observed_placeholders != expected_placeholders:
        report.add(
            "invariants",
            "stages.expanded_prompts",
            "placeholder_patch_invariant",
            modality=kind,
            expected=expected_placeholders,
            actual=observed_placeholders,
        )


def check_invariants(manifest: Mapping[str, Any], *, root: Path | None = None) -> ComparisonReport:
    report = ComparisonReport(
        str(manifest.get("case_id", "<unknown>")), str(manifest.get("case_id", "<unknown>"))
    )
    if manifest.get("status") != "success":
        return report
    arrays = manifest.get("output", {}).get("arrays", {})
    keys = manifest.get("output", {}).get("keys", [])
    if set(arrays) != set(keys):
        report.add(
            "invariants",
            "output.keys",
            "array_key_invariant",
            expected=sorted(keys),
            actual=sorted(arrays),
        )
    for name, descriptor in arrays.items():
        _check_array_descriptor(report, descriptor, f"output.arrays.{name}")
    text_shapes = [
        arrays[name].get("shape")
        for name in EXACT_OUTPUTS
        if name in arrays and name not in {"image_grid_thw", "video_grid_thw"}
    ]
    valid_text_shapes = [
        shape for shape in text_shapes if isinstance(shape, list) and len(shape) == 2
    ]
    if len(valid_text_shapes) != len(text_shapes):
        report.add(
            "invariants",
            "output.arrays",
            "text_shape_invariant",
            shapes=text_shapes,
        )
    elif valid_text_shapes and any(
        shape != valid_text_shapes[0] for shape in valid_text_shapes[1:]
    ):
        report.add("invariants", "output.arrays", "text_shape_invariant", shapes=text_shapes)
    request_count = len(manifest.get("input", {}).get("requests", []))
    if valid_text_shapes and valid_text_shapes[0][0] != request_count:
        report.add(
            "invariants",
            "output.arrays.input_ids.shape",
            "batch_size_invariant",
            expected=request_count,
            actual=valid_text_shapes[0][0],
        )
    _check_modality_invariants(report, manifest, root, kind="image")
    _check_modality_invariants(report, manifest, root, kind="video")
    return report


def compare_manifests(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    expected_root: Path | None = None,
    actual_root: Path | None = None,
) -> ComparisonReport:
    report = ComparisonReport(
        str(expected.get("case_id", "<unknown>")), str(actual.get("case_id", "<unknown>"))
    )
    for field_name in ("case_id", "profile_alias", "status"):
        _json_equal(
            report,
            "envelope",
            field_name,
            expected.get(field_name),
            actual.get(field_name),
        )
    if expected.get("status") != actual.get("status"):
        return report
    if expected.get("status") == "expected_error":
        _json_equal(
            report,
            "error",
            "error.category",
            expected.get("error", {}).get("category"),
            actual.get("error", {}).get("category"),
        )
        return report
    if expected.get("status") != "success":
        return report

    _json_equal(
        report,
        "input",
        "input.media",
        expected.get("input", {}).get("media"),
        actual.get("input", {}).get("media"),
    )
    for field_name in (
        "rendered_prompts",
        "expanded_prompts",
        "replacement_offsets",
        "video_kwargs",
    ):
        _json_equal(
            report,
            field_name,
            f"stages.{field_name}",
            expected.get("stages", {}).get(field_name),
            actual.get("stages", {}).get(field_name),
        )
    fps_policy = expected["comparison_policy"]["numeric"]["raw_frame_fps"]
    _compare_video_metadata(
        report,
        expected.get("stages", {}).get("video_metadata"),
        actual.get("stages", {}).get("video_metadata"),
        float(fps_policy["atol_hz"]),
    )
    _compare_prepared_media(
        report,
        expected,
        actual,
        expected_root=expected_root,
        actual_root=actual_root,
    )

    expected_output = expected.get("output", {})
    actual_output = actual.get("output", {})
    expected_keys = expected_output.get("keys", [])
    actual_keys = actual_output.get("keys", [])
    if expected_keys != actual_keys:
        report.add(
            "outputs",
            "output.keys",
            "key_or_order_mismatch",
            expected=expected_keys,
            actual=actual_keys,
        )
    expected_arrays = expected_output.get("arrays", {})
    actual_arrays = actual_output.get("arrays", {})
    for name in expected_keys:
        if name not in actual_arrays:
            report.add("outputs", f"output.arrays.{name}", "missing_array")
            continue
        if name not in expected_arrays:
            continue
        exact = name in EXACT_OUTPUTS
        atol, rtol = (0.0, 0.0) if exact else _output_tolerance(expected, name)
        _compare_array(
            report,
            stage="outputs",
            path=f"output.arrays.{name}",
            name=name,
            expected_descriptor=expected_arrays[name],
            actual_descriptor=actual_arrays[name],
            expected_root=expected_root,
            actual_root=actual_root,
            exact=exact,
            atol=atol,
            rtol=rtol,
            manifest=expected,
        )
    for name in actual_arrays.keys() - expected_arrays.keys():
        report.add("outputs", f"output.arrays.{name}", "unexpected_array")

    for invariant_issue in check_invariants(actual, root=actual_root).issues:
        report.issues.append(invariant_issue)
    return report


def _manifest_paths(paths: Sequence[Path]) -> list[Path]:
    manifests: list[Path] = []
    for path in paths:
        manifests.extend(sorted(path.rglob("manifest.json")) if path.is_dir() else [path])
    return sorted(set(manifests))


def _print_report(report: ComparisonReport) -> None:
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare qwen-mm candidate manifests with the pinned staged oracle."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    compare_parser = subparsers.add_parser("compare", help="compare two manifests")
    compare_parser.add_argument("--expected", type=Path, required=True)
    compare_parser.add_argument("--actual", type=Path, required=True)
    check_parser = subparsers.add_parser(
        "check", help="validate committed manifests and their cross-stage invariants"
    )
    check_parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    if args.command == "compare":
        expected = _load_json(args.expected)
        actual = _load_json(args.actual)
        report = compare_manifests(
            expected,
            actual,
            expected_root=args.expected.parent,
            actual_root=args.actual.parent,
        )
        _print_report(report)
        raise SystemExit(0 if report.passed else 1)

    failed = False
    for path in _manifest_paths(args.paths):
        manifest = _load_json(path)
        validate_manifest(manifest)
        report = check_invariants(manifest, root=path.parent)
        if report.passed:
            print(f"conformant golden: {path}")
        else:
            failed = True
            _print_report(report)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
