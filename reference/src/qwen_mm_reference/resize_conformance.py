"""Generate the deterministic B5 Pillow/TorchVision resize-stage oracle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import smart_resize
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tv_functional

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "reference" / "resize" / "v1"
COMPATIBILITY_PATH = REPOSITORY_ROOT / "reference" / "compatibility" / "v1.json"
GENERATOR_COMMAND = (
    "./scripts/with-cargo.sh uv run --locked --no-sync "
    "--package qwen-mm-reference python -m qwen_mm_reference.resize_conformance"
)


CASE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "aligned-no-op-ramp",
        "source": [64, 96],
        "pattern": "ramp",
        "tags": ["no_op", "aligned", "ramp", "landscape"],
    },
    {
        "id": "factor-below-impulse",
        "source": [31, 31],
        "pattern": "impulse",
        "tags": ["upsample", "factor_boundary", "min_boundary", "impulse"],
    },
    {
        "id": "factor-at-checkerboard",
        "source": [32, 32],
        "pattern": "checkerboard",
        "tags": ["upsample", "factor_boundary", "checkerboard"],
    },
    {
        "id": "factor-above-independent-rgb",
        "source": [33, 33],
        "pattern": "independent_rgb",
        "tags": ["upsample", "factor_boundary", "independent_rgb"],
    },
    {
        "id": "off-grid-landscape-checkerboard",
        "source": [63, 95],
        "pattern": "checkerboard",
        "tags": ["off_grid", "landscape", "checkerboard"],
    },
    {
        "id": "off-grid-portrait-edges",
        "source": [95, 63],
        "pattern": "edges",
        "tags": ["off_grid", "portrait", "edges"],
    },
    {
        "id": "horizontal-only-padded-independent-rgb",
        "source": [64, 95],
        "pattern": "independent_rgb",
        "row_padding": 7,
        "tags": [
            "upsample",
            "off_grid",
            "landscape",
            "horizontal_only",
            "padded_stride",
            "independent_rgb",
        ],
    },
    {
        "id": "vertical-only-checkerboard",
        "source": [95, 64],
        "pattern": "checkerboard",
        "tags": [
            "upsample",
            "off_grid",
            "portrait",
            "vertical_only",
            "checkerboard",
        ],
    },
    {
        "id": "min-below-ramp",
        "source": [32, 128],
        "pattern": "ramp",
        "min_pixels": 4095,
        "max_pixels": 16_777_216,
        "tags": ["min_boundary", "ramp", "landscape"],
    },
    {
        "id": "min-at-independent-rgb",
        "source": [32, 128],
        "pattern": "independent_rgb",
        "min_pixels": 4096,
        "max_pixels": 16_777_216,
        "tags": ["min_boundary", "independent_rgb", "landscape"],
    },
    {
        "id": "min-above-checkerboard",
        "source": [32, 128],
        "pattern": "checkerboard",
        "min_pixels": 4097,
        "max_pixels": 16_777_216,
        "tags": ["upsample", "min_boundary", "checkerboard", "landscape"],
    },
    {
        "id": "max-below-noise",
        "source": [128, 128],
        "pattern": "noise",
        "min_pixels": 4096,
        "max_pixels": 16_383,
        "tags": ["downsample", "max_boundary", "high_frequency"],
    },
    {
        "id": "max-at-impulse",
        "source": [128, 128],
        "pattern": "impulse",
        "min_pixels": 4096,
        "max_pixels": 16_384,
        "tags": ["no_op", "max_boundary", "impulse"],
    },
    {
        "id": "max-above-edges",
        "source": [128, 128],
        "pattern": "edges",
        "min_pixels": 4096,
        "max_pixels": 16_385,
        "tags": ["no_op", "max_boundary", "edges"],
    },
    {
        "id": "representative-downsample-noise",
        "source": [257, 385],
        "pattern": "noise",
        "min_pixels": 4096,
        "max_pixels": 6144,
        "tags": ["downsample", "off_grid", "high_frequency", "landscape"],
    },
    {
        "id": "representative-upsample-portrait",
        "source": [17, 9],
        "pattern": "independent_rgb",
        "tags": ["upsample", "off_grid", "portrait", "independent_rgb"],
    },
    {
        "id": "representative-upsample-landscape",
        "source": [9, 17],
        "pattern": "edges",
        "tags": ["upsample", "off_grid", "landscape", "edges"],
    },
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256(path.read_bytes())


def _pattern(height: int, width: int, name: str) -> np.ndarray:
    y, x = np.indices((height, width), dtype=np.uint32)
    if name == "ramp":
        red = x * 255 // max(width - 1, 1)
        green = y * 255 // max(height - 1, 1)
        blue = (x * 17 + y * 29) % 256
        return np.stack((red, green, blue), axis=-1).astype(np.uint8)
    if name == "checkerboard":
        red = ((x + y) % 2) * 255
        green = ((x // 2 + y // 3) % 2) * 191
        blue = ((x // 5 + y // 7) % 2) * 127
        return np.stack((red, green, blue), axis=-1).astype(np.uint8)
    if name == "independent_rgb":
        red = (x * 37 + 11) % 256
        green = (y * 53 + 29) % 256
        blue = (x * 7 + y * 13 + 101) % 256
        return np.stack((red, green, blue), axis=-1).astype(np.uint8)
    if name == "edges":
        red = np.where(x < width // 2, 0, 255)
        green = np.where(y < height // 2, 255, 0)
        blue = np.where(x + y < (width + height) // 2, 31, 223)
        return np.stack((red, green, blue), axis=-1).astype(np.uint8)
    if name == "impulse":
        output = np.zeros((height, width, 3), dtype=np.uint8)
        output[height // 2, width // 2] = (255, 127, 63)
        output[max(height // 3, 0), max(width // 4, 0)] = (17, 251, 89)
        return output
    if name == "noise":
        output = np.empty(height * width * 3, dtype=np.uint8)
        state = 0x51A7E5ED
        for index in range(output.size):
            state = (1_664_525 * state + 1_013_904_223) & 0xFFFF_FFFF
            output[index] = state >> 24
        return output.reshape(height, width, 3)
    raise ValueError(f"unknown pattern: {name}")


def _append(buffer: bytearray, data: bytes) -> dict[str, Any]:
    record = {
        "offset": len(buffer),
        "byte_length": len(data),
        "sha256": _sha256(data),
    }
    buffer.extend(data)
    return record


def _versions() -> dict[str, str]:
    names = ("Pillow", "numpy", "qwen-vl-utils", "torch", "torchvision")
    return {name: importlib.metadata.version(name) for name in names}


def generate(output_directory: Path) -> None:
    compatibility = json.loads(COMPATIBILITY_PATH.read_text(encoding="utf-8"))
    versions = _versions()
    expected_versions = {name: compatibility["packages"][name] for name in versions}
    if versions != expected_versions:
        raise RuntimeError(
            f"locked package mismatch: expected {expected_versions}, observed {versions}"
        )

    torch.set_num_threads(1)
    source_blob = bytearray()
    image_blob = bytearray()
    video_blob = bytearray()
    cases: list[dict[str, Any]] = []

    for specification in CASE_SPECS:
        source_height, source_width = specification["source"]
        source = np.ascontiguousarray(
            _pattern(source_height, source_width, specification["pattern"])
        )
        min_pixels = specification.get("min_pixels")
        max_pixels = specification.get("max_pixels")
        destination_height, destination_width = smart_resize(
            source_height,
            source_width,
            factor=32,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        pillow_output = np.asarray(
            Image.fromarray(source, mode="RGB").resize((destination_width, destination_height)),
            dtype=np.uint8,
        )
        tensor = torch.from_numpy(source.copy()).permute(2, 0, 1)
        torchvision_output = tv_functional.resize(
            tensor,
            [destination_height, destination_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        torchvision_hwc = (
            torchvision_output.permute(1, 2, 0).contiguous().numpy().astype("<f4", copy=False)
        )

        packed_stride = source_width * 3
        source_stride = packed_stride + specification.get("row_padding", 0)
        if source_stride == packed_stride:
            source_bytes = source.tobytes(order="C")
        else:
            stored_source = np.full((source_height, source_stride), 0xA5, dtype=np.uint8)
            stored_source[:, :packed_stride] = source.reshape(source_height, packed_stride)
            source_bytes = stored_source.tobytes(order="C")
        image_bytes = np.ascontiguousarray(pillow_output).tobytes(order="C")
        video_bytes = torchvision_hwc.tobytes(order="C")
        case = {
            "id": specification["id"],
            "tags": specification["tags"],
            "pattern": specification["pattern"],
            "source": {
                "height": source_height,
                "width": source_width,
                "stride_bytes": source_stride,
                **_append(source_blob, source_bytes),
            },
            "geometry_options": {
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            },
            "destination": {
                "height": destination_height,
                "width": destination_width,
            },
            "pillow_image_rgb8": {
                "layout": "HWC RGB",
                "dtype": "uint8",
                "shape": [destination_height, destination_width, 3],
                **_append(image_blob, image_bytes),
            },
            "torchvision_video_rgb_f32": {
                "layout": "HWC RGB (transposed from the oracle TCHW result)",
                "dtype": "float32-le",
                "domain": "0..255",
                "shape": [destination_height, destination_width, 3],
                **_append(video_blob, video_bytes),
            },
        }
        cases.append(case)

    output_directory.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "sources.rgb8.bin": bytes(source_blob),
        "pillow-image-rgb8.bin": bytes(image_blob),
        "torchvision-video-f32le.bin": bytes(video_blob),
    }
    for name, data in artifacts.items():
        (output_directory / name).write_bytes(data)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "contract_id": compatibility["contract_id"],
        "stage_id": "qwen-mm-resize-stage-v1",
        "generator": {
            "command": GENERATOR_COMMAND,
            "source": "reference/src/qwen_mm_reference/resize_conformance.py",
            "source_sha256": _sha256_file(Path(__file__)),
        },
        "oracle": {
            "image": "Pillow Image.resize default (Resampling.BICUBIC)",
            "video": (
                "torchvision.transforms.functional.resize on a CPU uint8 CHW tensor; "
                "the wrapper casts to float32, uses InterpolationMode.BICUBIC with "
                "antialias=True, clamps and rounds back to uint8, then qwen calls float()"
            ),
            "packages": versions,
            "compatibility_manifest": {
                "path": "reference/compatibility/v1.json",
                "sha256": _sha256_file(COMPATIBILITY_PATH),
            },
            "profile_fingerprints": {
                alias: profile["fingerprint"]
                for alias, profile in compatibility["profiles"].items()
            },
            "qwen_vl_utils_source_sha256": compatibility["source_files"][
                "qwen_vl_utils/vision_process.py"
            ],
            "capture_platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
        },
        "comparison_policy": {
            "pillow_image_rgb8": {"absolute_byte_error_max": 1},
            "torchvision_video_rgb_f32": {"rtol": 0.0, "atol": 0.0001},
            "diagnostics": [
                "maximum_absolute_error",
                "rmse",
                "absolute_error_p50_p90_p99",
                "count_nonzero_differences",
                "count_over_tolerance",
                "maximum_ulp_distance",
                "first_differing_case_pixel_channel",
            ],
        },
        "candidate_requirements": {
            "coordinate_transform": "half-pixel centers (align_corners=false)",
            "image_kernel": "cubic convolution a=-0.5 with downscale antialias",
            "video_kernel": (
                "TorchVision float32 Keys bicubic-antialias convolution followed by "
                "clamp, ties-to-even uint8 quantization, and float32 exposure"
            ),
            "color_domain": "non-linear RGB; no linear-light conversion",
            "clamping": "image/video uint8 outputs clamp to 0..255",
        },
        "coverage_tags": sorted(
            {tag for specification in CASE_SPECS for tag in specification["tags"]}
        ),
        "artifacts": {
            name: {"byte_length": len(data), "sha256": _sha256(data)}
            for name, data in artifacts.items()
        },
        "cases": cases,
    }
    encoded = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    (output_directory / "manifest.json").write_bytes(encoded)
    print(f"wrote {len(cases)} resize cases to {output_directory}")
    print(f"manifest sha256: {_sha256(encoded)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    generate(arguments.output)


if __name__ == "__main__":
    main()
