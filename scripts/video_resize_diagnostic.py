"""Report installed-wheel resize differences and official half-rounding witnesses.

This diagnostic never changes the video oracle's 2e-6 tolerance. It emits both
observed mismatches and known Linux regression coordinates, even on a host where
the native and official uint8 results agree. Output is a diagnostic report, not
an acceptance gate or a replacement for video_oracle.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from video_oracle import (
    PROFILES,
    ROOT,
    VIDEO_ATOL,
    cases,
    load_processors,
    official_inputs,
    reference_video,
    runtime_identity,
    unpatchify_video,
)

RESIZE_CASES = ("decoded_spatial_resize", "decoded_total_pixel_budget")
LINUX_REGRESSION_POINTS = (
    (2, 2, 10, 59),
    (3, 2, 10, 59),
    (2, 2, 42, 59),
    (2, 2, 74, 59),
    (0, 2, 42, 59),
)


def interpolate(data: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    return torch.nn.functional.interpolate(
        data, size=shape, mode="bicubic", align_corners=False, antialias=True
    )


def float_witness(value: float) -> dict[str, Any]:
    value32 = np.float32(value)
    half = float(np.floor(value32) + 0.5)
    return {
        "float32": float(value32),
        "float32_bits": f"0x{value32.view(np.uint32).item():08x}",
        "nearest_half": half,
        "signed_distance_from_half": float(value32) - half,
        "clamped_round_ties_even_u8": int(np.rint(np.clip(value32, 0, 255))),
    }


def diagnose(
    profile: str,
    case: Any,
    cache: Path,
    thread_budget: int,
    max_mismatches: int,
    near_half_count: int,
) -> dict[str, Any]:
    item = next(
        content
        for content in case.requests[0]["messages"][0]["content"]
        if content.get("type") == "video"
    )
    source = np.ascontiguousarray(item["video"])
    data = torch.from_numpy(source).permute(0, 3, 1, 2)
    reference, _, route = reference_video(source, item)
    target = tuple(reference.shape[-2:])
    height, width = source.shape[1:3]

    # These are the actual official float32 operations before TorchVision's
    # uint8 clamp/round. Separating the two passes exposes the horizontal
    # intermediates; equality with the whole operation is measured below.
    floating = data.float()
    whole = interpolate(floating, target)
    horizontal = interpolate(floating, (height, target[1]))
    separable = interpolate(horizontal, target)
    quantized = whole.clamp(0, 255).round().to(torch.uint8).numpy()
    np.testing.assert_array_equal(quantized, reference.numpy().astype(np.uint8))

    native, official = load_processors(profile, cache, thread_budget)
    expected, _ = official_inputs(official, case)
    prepared = native.prepare_batch(case.requests, padding_side=case.padding_side)
    np.testing.assert_array_equal(expected["video_grid_thw"], prepared["video_grid_thw"])
    grid = expected["video_grid_thw"][0]
    wanted = unpatchify_video(expected["pixel_values_videos"], grid)
    got = unpatchify_video(prepared["pixel_values_videos"], grid)
    mismatches = np.argwhere(got != wanted)

    # Basis-vector interpolation exposes the actual official vertical kernel's
    # impulse response. No native coefficient implementation is used here.
    # Width-one antialias interpolation has a separate upstream edge path. Use
    # two identical columns so the vertical probe exercises the ordinary path;
    # use two identical rows for the corresponding horizontal probe.
    basis = torch.eye(height, dtype=torch.float32).reshape(height, 1, height, 1)
    basis = basis.expand(-1, -1, -1, 2).contiguous()
    vertical_response = interpolate(basis, (target[0], 2))[:, 0, :, 0].numpy()
    horizontal_basis = torch.eye(width, dtype=torch.float32).reshape(width, 1, 1, width)
    horizontal_basis = horizontal_basis.expand(-1, -1, 2, -1).contiguous()
    horizontal_response = interpolate(horizontal_basis, (2, target[1]))[:, 0, 0, :].numpy()
    whole_values = whole.numpy()
    horizontal_values = horizontal.numpy()
    separable_values = separable.numpy()

    def pixel(point: tuple[int, int, int, int]) -> dict[str, Any]:
        frame, channel, y, x = point
        rows = np.flatnonzero(vertical_response[:, y])
        columns = np.flatnonzero(horizontal_response[:, x])
        return {
            "frame_channel_y_x": list(point),
            "reference_u8": int(wanted[point]),
            "native_u8": int(got[point]),
            "official_float_before_u8_round": float_witness(whole_values[point]),
            "official_separable_float_before_u8_round": float_witness(separable_values[point]),
            "horizontal_impulse_response": [
                {
                    "source_x": int(column),
                    "coefficient_f32": float(horizontal_response[column, x]),
                    "coefficient_f32_bits": f"0x{horizontal_response[column, x].view(np.uint32).item():08x}",
                }
                for column in columns
            ],
            "horizontal_intermediates": [
                {
                    "source_y": int(row),
                    "horizontal_float": float_witness(horizontal_values[frame, channel, row, x]),
                    "vertical_impulse_response_f32": float(vertical_response[row, y]),
                    "vertical_impulse_response_f32_bits": f"0x{vertical_response[row, y].view(np.uint32).item():08x}",
                    "source_channel_values_at_horizontal_kernel_columns": source[
                        frame, row, columns, channel
                    ].tolist(),
                }
                for row in rows
            ],
        }

    known_points = LINUX_REGRESSION_POINTS if case.name == "decoded_spatial_resize" else ()
    half_distances = np.abs(whole_values - (np.floor(whole_values) + np.float32(0.5)))
    nearest = np.argsort(half_distances.ravel(), kind="stable")[:near_half_count]
    near_points = [
        tuple(map(int, np.unravel_index(index, whole_values.shape))) for index in nearest
    ]
    normalized_difference = np.abs(
        prepared["pixel_values_videos"].astype(np.float64) - expected["pixel_values_videos"]
    )
    return {
        "profile": profile,
        "case": case.name,
        "reference_route": route,
        "source_thwc": list(source.shape),
        "target_tchw": list(reference.shape),
        "torch_strides": {
            "source_tchw": list(data.stride()),
            "source_float_tchw": list(floating.stride()),
            "official_output_tchw": list(whole.stride()),
            "official_horizontal_tchw": list(horizontal.stride()),
        },
        "grid_thw": grid.tolist(),
        "options": {key: value for key, value in item.items() if key not in ("type", "video")},
        "uint8_mismatch_count": len(mismatches),
        "normalized_mismatch_count_at_oracle_tolerance": int(
            np.count_nonzero(normalized_difference > VIDEO_ATOL)
        ),
        "max_normalized_absolute_difference": float(normalized_difference.max()),
        "official_whole_vs_separable_max_absolute_difference": float(
            np.max(np.abs(whole_values - separable_values))
        ),
        "mismatches": [pixel(tuple(map(int, point))) for point in mismatches[:max_mismatches]],
        "known_linux_regression_points": [pixel(point) for point in known_points],
        "nearest_half_rounding_points": [pixel(point) for point in near_points],
        "non_video_arrays_exact": {
            key: bool(np.array_equal(prepared[key], value))
            for key, value in expected.items()
            if key != "pixel_values_videos"
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "reference/.cache/huggingface")
    parser.add_argument("--profile", choices=PROFILES, default=PROFILES[0])
    parser.add_argument("--case", choices=RESIZE_CASES, default=RESIZE_CASES[0])
    parser.add_argument("--thread-budget", type=int, default=2)
    parser.add_argument("--max-mismatches", type=int, default=32)
    parser.add_argument("--near-half-count", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("/tmp/video-resize-diagnostic.json"))
    args = parser.parse_args()
    if min(args.thread_budget, args.max_mismatches, args.near_half_count) <= 0:
        parser.error("thread budget and report counts must be positive")
    torch.set_num_threads(args.thread_budget)
    identity = runtime_identity()
    identity.update(
        torch_cpu_capability=torch.backends.cpu.get_cpu_capability(),
        aten_cpu_capability_override=os.environ.get("ATEN_CPU_CAPABILITY"),
        diagnostic_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    with tempfile.TemporaryDirectory(prefix="qwen-mm-resize-diagnostic-") as temporary:
        case = next(
            case for case in cases(Path(temporary), include_files=False) if case.name == args.case
        )
        result = diagnose(
            args.profile,
            case,
            args.cache_dir,
            args.thread_budget,
            args.max_mismatches,
            args.near_half_count,
        )
    report = {
        "schema_id": "qwen-mm-video-resize-diagnostic-v1",
        "runtime": identity,
        "video_oracle_atol_unchanged": VIDEO_ATOL,
        "scope": "diagnostic only; official float32 interpolation and uint8 rounding witnesses",
        "result": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if isinstance(value, (str, int, float))}
        )
    )
    print(f"Diagnostic report: {args.output}")


if __name__ == "__main__":
    main()
