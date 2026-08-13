"""Generate and compare the frozen still-image resize quality holdout v2.

The holdout is candidate-blind: this module generates only deterministic RGB8
sources and official Pillow 12.3.0 outputs.  Candidate implementations write a
single RGB8 blob with the same per-case offsets as ``pillow-image-rgb8.bin``;
``compare`` authenticates the corpus and evaluates the v2 per-channel gates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from qwen_vl_utils import smart_resize

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "reference" / "resize" / "v2"
COMPATIBILITY_PATH = ROOT / "reference" / "compatibility" / "v1.json"
CONTRACT_PATH = ROOT / "docs" / "image-resize-contract-v2.md"
FIXTURE_MANIFEST_PATH = ROOT / "fixtures" / "baseline" / "manifest.json"
FIXTURE_GENERATOR_PATH = ROOT / "reference" / "src" / "qwen_mm_reference" / "fixtures.py"
NASA_ASSET_PATH = ROOT / "reference" / "resize" / "v2" / "assets" / "nasa-blue-marble-2004-12.jpg"
NASA_ASSET_SHA256 = "3d38a8ecaceb1cdb51ced7f831e9ac18d261950049616feff6892efcae0dcc26"
NASA_SOURCE_URL = (
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74218/world.200412.3x5400x2700.jpg"
)
NASA_USAGE_POLICY_URL = "https://www.nasa.gov/nasa-brand-center/"
GENERATOR_PATH = Path(__file__)
SEED = 20_260_813
CONTRACT_ID = "qwen-mm-still-image-resize-v2"
STAGE_ID = "qwen-mm-still-image-resize-holdout-v2"
GENERATOR_COMMAND = (
    "./scripts/with-cargo.sh uv run --locked --no-sync --package "
    "qwen-mm-reference python -m qwen_mm_reference.resize_quality_v2 generate"
)

GATES: dict[str, float] = {
    "maximum_absolute_error": 32.0,
    "rmse": 5.0,
    "absolute_error_p99": 16.0,
    "absolute_signed_mean_bias": 2.0,
    "ssim": 0.98,
}


def _case(
    case_id: str,
    source: tuple[int, int],
    pattern: str,
    tags: Iterable[str],
    *,
    destination: tuple[int, int] | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    row_padding: int = 0,
    value: tuple[int, int, int] | None = None,
    seed_offset: int = 0,
    channel_order: tuple[int, int, int] | None = None,
    relation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "source": source,
        "pattern": pattern,
        "tags": list(tags),
        "destination": destination,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "row_padding": row_padding,
        "value": value,
        "seed_offset": seed_offset,
        "channel_order": channel_order,
        "relation": relation,
    }


CASE_SPECS: tuple[dict[str, Any], ...] = (
    _case("noop-padded-ramp", (64, 96), "ramp", ("no_op", "padded_stride", "ramp"), row_padding=7),
    _case(
        "solid-black-up",
        (33, 47),
        "solid",
        ("constant", "black", "upsample"),
        destination=(96, 128),
        value=(0, 0, 0),
    ),
    _case(
        "solid-white-down",
        (129, 193),
        "solid",
        ("constant", "white", "downsample"),
        destination=(64, 96),
        value=(255, 255, 255),
    ),
    _case(
        "solid-midgray-anisotropic",
        (31, 257),
        "solid",
        ("constant", "midgray", "extreme_anisotropic"),
        destination=(256, 32),
        value=(128, 128, 128),
    ),
    _case(
        "solid-red-up",
        (35, 51),
        "solid",
        ("constant", "primary", "red", "upsample"),
        destination=(96, 128),
        value=(255, 0, 0),
    ),
    _case(
        "solid-green-down",
        (171, 259),
        "solid",
        ("constant", "primary", "green", "downsample"),
        destination=(64, 96),
        value=(0, 255, 0),
    ),
    _case(
        "solid-blue-one-axis",
        (64, 257),
        "solid",
        ("constant", "primary", "blue", "horizontal_only"),
        destination=(64, 96),
        value=(0, 0, 255),
    ),
    _case(
        "ramp-horizontal-down",
        (73, 301),
        "horizontal_ramp",
        ("ramp", "downsample", "horizontal_only"),
        destination=(73, 96),
    ),
    _case(
        "ramp-vertical-up",
        (41, 83),
        "vertical_ramp",
        ("ramp", "upsample", "vertical_only"),
        destination=(160, 83),
    ),
    _case(
        "ramp-independent-anisotropic",
        (37, 401),
        "ramp",
        ("ramp", "extreme_anisotropic"),
        destination=(224, 64),
    ),
    _case("impulse-up", (37, 53), "impulse", ("impulse", "upsample"), destination=(128, 160)),
    _case("impulse-down", (193, 257), "impulse", ("impulse", "downsample"), destination=(64, 96)),
    _case(
        "edge-vertical-down",
        (97, 291),
        "vertical_edge",
        ("edge", "downsample"),
        destination=(64, 96),
    ),
    _case(
        "edge-horizontal-up",
        (39, 71),
        "horizontal_edge",
        ("edge", "upsample"),
        destination=(128, 160),
    ),
    _case(
        "checkerboard-up",
        (35, 51),
        "checkerboard",
        ("checkerboard", "upsample", "high_frequency"),
        destination=(128, 160),
    ),
    _case(
        "checkerboard-down",
        (193, 257),
        "checkerboard",
        ("checkerboard", "downsample", "high_frequency"),
        destination=(64, 96),
    ),
    _case(
        "nyquist-vertical-stripes",
        (127, 383),
        "vertical_stripes",
        ("one_pixel_stripes", "nyquist", "downsample"),
        destination=(64, 96),
    ),
    _case(
        "nyquist-horizontal-stripes",
        (383, 127),
        "horizontal_stripes",
        ("one_pixel_stripes", "nyquist", "downsample"),
        destination=(96, 64),
    ),
    _case(
        "independent-rgb-down",
        (173, 263),
        "independent_rgb",
        ("independent_channels", "downsample"),
        destination=(64, 96),
    ),
    _case(
        "alternating-channels-up",
        (43, 59),
        "alternating_channels",
        ("alternating_channels", "upsample"),
        destination=(128, 160),
    ),
    _case(
        "red-only-edge",
        (93, 151),
        "red_only",
        ("single_channel", "red", "edge", "downsample"),
        destination=(64, 96),
    ),
    _case(
        "green-only-ramp",
        (47, 71),
        "green_only",
        ("single_channel", "green", "ramp", "upsample"),
        destination=(128, 160),
    ),
    _case(
        "blue-only-noise",
        (137, 211),
        "blue_only",
        ("single_channel", "blue", "noise", "downsample"),
        destination=(64, 96),
        seed_offset=6,
    ),
    _case(
        "identical-channels-down",
        (151, 227),
        "identical_channels",
        ("identical_channels", "downsample"),
        destination=(64, 96),
    ),
    _case(
        "channel-permutation-base",
        (79, 113),
        "independent_rgb",
        ("channel_permutation", "base"),
        destination=(96, 128),
        relation={"group": "channel-permutation", "order": [0, 1, 2]},
    ),
    _case(
        "channel-permutation-bgr",
        (79, 113),
        "independent_rgb",
        ("channel_permutation", "permuted"),
        destination=(96, 128),
        channel_order=(2, 1, 0),
        relation={"group": "channel-permutation", "order": [2, 1, 0]},
    ),
    _case(
        "noise-up", (47, 61), "noise", ("noise", "upsample"), destination=(128, 160), seed_offset=1
    ),
    _case(
        "noise-down",
        (211, 317),
        "noise",
        ("noise", "downsample"),
        destination=(64, 96),
        seed_offset=2,
    ),
    _case(
        "factor-below",
        (31, 31),
        "noise",
        ("factor_boundary", "factor_below", "upsample"),
        seed_offset=3,
    ),
    _case("factor-at", (32, 32), "ramp", ("factor_boundary", "factor_at", "no_op")),
    _case(
        "factor-above", (33, 33), "independent_rgb", ("factor_boundary", "factor_above", "upsample")
    ),
    _case(
        "min-below",
        (32, 128),
        "horizontal_ramp",
        ("min_boundary", "min_below"),
        min_pixels=4095,
        max_pixels=16_777_216,
    ),
    _case(
        "min-at",
        (32, 128),
        "independent_rgb",
        ("min_boundary", "min_at", "no_op"),
        min_pixels=4096,
        max_pixels=16_777_216,
    ),
    _case(
        "min-above",
        (32, 128),
        "noise",
        ("min_boundary", "min_above", "upsample"),
        min_pixels=4097,
        max_pixels=16_777_216,
        seed_offset=4,
    ),
    _case(
        "max-below",
        (128, 128),
        "noise",
        ("max_boundary", "max_below", "downsample"),
        min_pixels=4096,
        max_pixels=16_383,
        seed_offset=5,
    ),
    _case(
        "max-at",
        (128, 128),
        "impulse",
        ("max_boundary", "max_at", "no_op"),
        min_pixels=4096,
        max_pixels=16_384,
    ),
    _case(
        "max-above",
        (128, 128),
        "vertical_edge",
        ("max_boundary", "max_above", "no_op"),
        min_pixels=4096,
        max_pixels=16_385,
    ),
)

FIXTURE_CROPS: tuple[dict[str, Any], ...] = (
    {
        "id": "procedural-fixture-crop-00-down",
        "filename": "image-00.jpg",
        "crop": (37, 41, 677, 521),
        "destination": (128, 192),
        "tags": ["procedural_fixture_crop", "downsample", "landscape"],
    },
    {
        "id": "procedural-fixture-crop-07-up",
        "filename": "image-07.jpg",
        "crop": (211, 153, 467, 345),
        "destination": (384, 512),
        "tags": ["procedural_fixture_crop", "upsample", "landscape"],
    },
    {
        "id": "procedural-fixture-crop-13-portrait",
        "filename": "image-13.jpg",
        "crop": (451, 73, 707, 649),
        "destination": (256, 128),
        "tags": ["procedural_fixture_crop", "downsample", "portrait"],
    },
    {
        "id": "procedural-fixture-crop-21-one-axis",
        "filename": "image-21.jpg",
        "crop": (101, 233, 741, 489),
        "destination": (256, 192),
        "tags": ["procedural_fixture_crop", "horizontal_only"],
    },
)

NASA_CROPS: tuple[dict[str, Any], ...] = (
    {
        "id": "natural-blue-marble-africa-down",
        "crop": (2800, 500, 3600, 1100),
        "destination": (128, 192),
        "tags": ["natural_image", "nasa_blue_marble", "downsample", "landscape"],
    },
    {
        "id": "natural-blue-marble-americas-portrait",
        "crop": (800, 400, 1300, 1150),
        "destination": (256, 160),
        "tags": ["natural_image", "nasa_blue_marble", "downsample", "portrait"],
    },
    {
        "id": "natural-blue-marble-asia-up",
        "crop": (4100, 600, 4420, 840),
        "destination": (384, 512),
        "tags": ["natural_image", "nasa_blue_marble", "upsample", "landscape"],
    },
    {
        "id": "natural-blue-marble-ocean-anisotropic",
        "crop": (1800, 1700, 2700, 2000),
        "destination": (256, 128),
        "tags": ["natural_image", "nasa_blue_marble", "extreme_anisotropic"],
    },
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256(path.read_bytes())


def _append(buffer: bytearray, data: bytes) -> dict[str, Any]:
    record = {"offset": len(buffer), "byte_length": len(data), "sha256": _sha256(data)}
    buffer.extend(data)
    return record


def _versions() -> dict[str, str]:
    names = ("Pillow", "numpy", "qwen-vl-utils")
    return {name: importlib.metadata.version(name) for name in names}


def _pattern(height: int, width: int, specification: dict[str, Any]) -> np.ndarray:
    name = specification["pattern"]
    y, x = np.indices((height, width), dtype=np.uint32)
    if name == "solid":
        output = np.empty((height, width, 3), dtype=np.uint8)
        output[...] = specification["value"]
    elif name == "ramp":
        output = np.stack(
            (x * 255 // max(width - 1, 1), y * 255 // max(height - 1, 1), (x * 11 + y * 17) % 256),
            axis=-1,
        ).astype(np.uint8)
    elif name == "horizontal_ramp":
        plane = (x * 255 // max(width - 1, 1)).astype(np.uint8)
        output = np.stack((plane, plane, plane), axis=-1)
    elif name == "vertical_ramp":
        plane = (y * 255 // max(height - 1, 1)).astype(np.uint8)
        output = np.stack((plane, plane, plane), axis=-1)
    elif name == "impulse":
        output = np.zeros((height, width, 3), dtype=np.uint8)
        output[height // 2, width // 2] = (255, 127, 63)
        output[height // 3, width // 4] = (17, 251, 89)
    elif name == "vertical_edge":
        output = np.stack(
            (
                np.where(x < width // 2, 0, 255),
                np.where(x < width // 2, 255, 0),
                np.where(x < width // 2, 31, 223),
            ),
            axis=-1,
        ).astype(np.uint8)
    elif name == "horizontal_edge":
        output = np.stack(
            (
                np.where(y < height // 2, 0, 255),
                np.where(y < height // 2, 255, 0),
                np.where(y < height // 2, 31, 223),
            ),
            axis=-1,
        ).astype(np.uint8)
    elif name == "checkerboard":
        output = np.stack(
            (((x + y) % 2) * 255, ((x + y) % 2) * 191, ((x + y) % 2) * 127), axis=-1
        ).astype(np.uint8)
    elif name == "vertical_stripes":
        plane = (x % 2 * 255).astype(np.uint8)
        output = np.stack((plane, 255 - plane, plane), axis=-1)
    elif name == "horizontal_stripes":
        plane = (y % 2 * 255).astype(np.uint8)
        output = np.stack((plane, plane, 255 - plane), axis=-1)
    elif name == "independent_rgb":
        output = np.stack(
            ((x * 37 + 11) % 256, (y * 53 + 29) % 256, (x * 7 + y * 13 + 101) % 256), axis=-1
        ).astype(np.uint8)
    elif name == "alternating_channels":
        output = np.stack(((x % 2) * 255, (y % 2) * 255, ((x + y) % 2) * 255), axis=-1).astype(
            np.uint8
        )
    elif name in {"red_only", "green_only", "blue_only"}:
        if name == "red_only":
            plane = np.where(x + y < (width + height) // 2, 0, 255).astype(np.uint8)
            channel = 0
        elif name == "green_only":
            plane = (x * 255 // max(width - 1, 1)).astype(np.uint8)
            channel = 1
        else:
            rng = np.random.default_rng(SEED + specification["seed_offset"])
            plane = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
            channel = 2
        output = np.zeros((height, width, 3), dtype=np.uint8)
        output[..., channel] = plane
    elif name == "identical_channels":
        plane = ((x * 19 + y * 31 + ((x ^ y) & 15) * 7) % 256).astype(np.uint8)
        output = np.stack((plane, plane, plane), axis=-1)
    elif name == "noise":
        rng = np.random.default_rng(SEED + specification["seed_offset"])
        output = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    else:
        raise ValueError(f"unknown pattern: {name}")
    order = specification.get("channel_order")
    return np.ascontiguousarray(output if order is None else output[..., list(order)])


def _fixture_sources() -> tuple[list[tuple[dict[str, Any], np.ndarray]], dict[str, Any]]:
    manifest = json.loads(FIXTURE_MANIFEST_PATH.read_text(encoding="utf-8"))
    records = {entry["filename"]: entry for entry in manifest["images"]}
    cases: list[tuple[dict[str, Any], np.ndarray]] = []
    used: list[dict[str, Any]] = []
    for crop_spec in FIXTURE_CROPS:
        filename = crop_spec["filename"]
        path = ROOT / "fixtures" / "baseline" / "image24" / filename
        record = records[filename]
        if path.stat().st_size != record["bytes"] or _sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"authenticated baseline fixture mismatch: {filename}")
        with Image.open(path) as encoded:
            decoded = np.asarray(encoded.convert("RGB").crop(crop_spec["crop"]), dtype=np.uint8)
        specification = {
            **crop_spec,
            "pattern": "decoded_procedural_fixture_crop",
            "source": tuple(decoded.shape[:2]),
            "row_padding": 0,
            "min_pixels": None,
            "max_pixels": None,
            "relation": None,
        }
        cases.append((specification, np.ascontiguousarray(decoded)))
        used.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": record["sha256"],
                "byte_length": record["bytes"],
                "crop_box_ltrb": list(crop_spec["crop"]),
            }
        )
    return cases, {
        "description": "project-local procedurally generated baseline JPEG fixtures; decoded RGB crops only; not natural photographs",
        "manifest": {
            "path": FIXTURE_MANIFEST_PATH.relative_to(ROOT).as_posix(),
            "sha256": _sha256_file(FIXTURE_MANIFEST_PATH),
        },
        "generator": {
            "path": FIXTURE_GENERATOR_PATH.relative_to(ROOT).as_posix(),
            "sha256": _sha256_file(FIXTURE_GENERATOR_PATH),
        },
        "used_files": used,
    }


def _natural_sources() -> tuple[list[tuple[dict[str, Any], np.ndarray]], dict[str, Any]]:
    if _sha256_file(NASA_ASSET_PATH) != NASA_ASSET_SHA256:
        raise RuntimeError("authenticated NASA Blue Marble source mismatch")
    with Image.open(NASA_ASSET_PATH) as encoded:
        if encoded.size != (5400, 2700) or encoded.mode != "RGB":
            raise RuntimeError(
                f"unexpected NASA source properties: size={encoded.size}, mode={encoded.mode}"
            )
        decoded = encoded.copy()
    cases: list[tuple[dict[str, Any], np.ndarray]] = []
    for crop_spec in NASA_CROPS:
        crop = np.asarray(decoded.crop(crop_spec["crop"]), dtype=np.uint8)
        specification = {
            **crop_spec,
            "pattern": "nasa_blue_marble_natural_crop",
            "source": tuple(crop.shape[:2]),
            "row_padding": 0,
            "min_pixels": None,
            "max_pixels": None,
            "relation": None,
        }
        cases.append((specification, np.ascontiguousarray(crop)))
    return cases, {
        "credit": "NASA Earth Observatory, Blue Marble",
        "asset": {
            "path": NASA_ASSET_PATH.relative_to(ROOT).as_posix(),
            "byte_length": NASA_ASSET_PATH.stat().st_size,
            "sha256": NASA_ASSET_SHA256,
            "source_url": NASA_SOURCE_URL,
        },
        "usage_policy": {
            "url": NASA_USAGE_POLICY_URL,
            "note": (
                "NASA generally has no objection to public reproduction subject to "
                "its media-usage and branding guidelines; NASA endorsement is not implied."
            ),
        },
        "crops": [
            {
                "id": crop["id"],
                "crop_box_ltrb": list(crop["crop"]),
                "destination_hw": list(crop["destination"]),
            }
            for crop in NASA_CROPS
        ],
    }


def _destination(specification: dict[str, Any], height: int, width: int) -> tuple[int, int, str]:
    explicit = specification.get("destination")
    if explicit is not None:
        return int(explicit[0]), int(explicit[1]), "explicit fixed holdout plan"
    destination_height, destination_width = smart_resize(
        height,
        width,
        factor=32,
        min_pixels=specification.get("min_pixels"),
        max_pixels=specification.get("max_pixels"),
    )
    return destination_height, destination_width, "qwen_vl_utils.smart_resize factor=32"


def generate(output_directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    compatibility = json.loads(COMPATIBILITY_PATH.read_text(encoding="utf-8"))
    versions = _versions()
    expected = {name: compatibility["packages"][name] for name in versions}
    if versions != expected:
        raise RuntimeError(f"locked package mismatch: expected {expected}, observed {versions}")

    fixture_cases, fixture_provenance = _fixture_sources()
    natural_cases, natural_provenance = _natural_sources()
    inputs: list[tuple[dict[str, Any], np.ndarray, dict[str, Any] | None]] = [
        (specification, _pattern(*specification["source"], specification), None)
        for specification in CASE_SPECS
    ]
    inputs.extend(
        (
            specification,
            source,
            {"filename": specification["filename"], "crop_box_ltrb": list(specification["crop"])},
        )
        for specification, source in fixture_cases
    )
    inputs.extend(
        (
            specification,
            source,
            {
                "asset": NASA_ASSET_PATH.relative_to(ROOT).as_posix(),
                "crop_box_ltrb": list(specification["crop"]),
            },
        )
        for specification, source in natural_cases
    )

    source_blob = bytearray()
    pillow_blob = bytearray()
    cases: list[dict[str, Any]] = []
    for specification, source, source_provenance in inputs:
        source_height, source_width, _ = source.shape
        destination_height, destination_width, planner = _destination(
            specification, source_height, source_width
        )
        if min(destination_height, destination_width) <= 5:
            raise RuntimeError(f"{specification['id']}: result too small for reflect-101 padding")
        pillow = np.asarray(
            Image.fromarray(source, mode="RGB").resize((destination_width, destination_height)),
            dtype=np.uint8,
        )
        packed_stride = source_width * 3
        source_stride = packed_stride + specification.get("row_padding", 0)
        if source_stride == packed_stride:
            stored_source = source.tobytes(order="C")
        else:
            padded = np.full((source_height, source_stride), 0xA5, dtype=np.uint8)
            padded[:, :packed_stride] = source.reshape(source_height, packed_stride)
            stored_source = padded.tobytes(order="C")

        exact_invariants: list[str] = []
        if (source_height, source_width) == (destination_height, destination_width):
            exact_invariants.append("no_op_source_bytes")
        if specification["pattern"] == "solid":
            exact_invariants.append("constant_rgb")
        if specification["pattern"] == "identical_channels":
            exact_invariants.append("identical_channels")
        if specification["pattern"] in {"red_only", "green_only", "blue_only"}:
            exact_invariants.append("zero_source_channels_remain_zero")

        case = {
            "id": specification["id"],
            "tags": specification["tags"],
            "pattern": specification["pattern"],
            "seed": (
                SEED + specification["seed_offset"]
                if specification["pattern"] in {"noise", "blue_only"}
                else None
            ),
            "source": {
                "height": source_height,
                "width": source_width,
                "stride_bytes": source_stride,
                **_append(source_blob, stored_source),
            },
            "source_provenance": source_provenance,
            "resize_plan": {
                "planner": planner,
                "factor": 32 if specification.get("destination") is None else None,
                "min_pixels": specification.get("min_pixels"),
                "max_pixels": specification.get("max_pixels"),
            },
            "destination": {"height": destination_height, "width": destination_width},
            "pillow_image_rgb8": {
                "layout": "HWC RGB",
                "dtype": "uint8",
                "shape": [destination_height, destination_width, 3],
                **_append(pillow_blob, np.ascontiguousarray(pillow).tobytes(order="C")),
            },
            "exact_invariants": exact_invariants,
            "relation": specification.get("relation"),
        }
        cases.append(case)

    artifacts = {
        "sources.rgb8.bin": bytes(source_blob),
        "pillow-image-rgb8.bin": bytes(pillow_blob),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, data in artifacts.items():
        (output_directory / name).write_bytes(data)
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "contract_id": CONTRACT_ID,
        "stage_id": STAGE_ID,
        "holdout_seed": SEED,
        "candidate_blind": True,
        "immutability": "Do not regenerate after any candidate is inspected against this holdout.",
        "generator": {
            "command": GENERATOR_COMMAND,
            "source": GENERATOR_PATH.relative_to(ROOT).as_posix(),
            "source_sha256": _sha256_file(GENERATOR_PATH),
        },
        "oracle": {
            "image": "Pillow Image.resize default (Resampling.BICUBIC)",
            "packages": versions,
            "compatibility_manifest": {
                "path": COMPATIBILITY_PATH.relative_to(ROOT).as_posix(),
                "sha256": _sha256_file(COMPATIBILITY_PATH),
            },
            "contract_document": {
                "path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
                "sha256": _sha256_file(CONTRACT_PATH),
            },
            "capture_platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
        },
        "fixture_provenance": fixture_provenance,
        "natural_image_provenance": natural_provenance,
        "comparison_policy": {
            "unit": "each resized media occurrence and each R/G/B channel independently",
            "gates": {
                "maximum_absolute_error_max": GATES["maximum_absolute_error"],
                "rmse_max": GATES["rmse"],
                "absolute_error_p99_max_nearest_rank": GATES["absolute_error_p99"],
                "absolute_signed_mean_bias_max": GATES["absolute_signed_mean_bias"],
                "canonical_windowed_ssim_min": GATES["ssim"],
            },
            "diagnostics": [
                "mae",
                "psnr_255",
                "absolute_error_p50_nearest_rank",
                "absolute_error_p90_nearest_rank",
                "maximum_absolute_error",
                "rmse",
                "absolute_error_p99_nearest_rank",
                "signed_mean_bias",
                "canonical_windowed_ssim",
            ],
            "ssim": {
                "dtype": "float64",
                "padding": "reflect-101, five pixels",
                "window": [11, 11],
                "sigma": 1.5,
                "population_statistics": True,
                "c1": (0.01 * 255) ** 2,
                "c2": (0.03 * 255) ** 2,
            },
        },
        "coverage_tags": sorted({tag for case in cases for tag in case["tags"]}),
        "artifacts": {
            name: {"byte_length": len(data), "sha256": _sha256(data)}
            for name, data in artifacts.items()
        },
        "cases": cases,
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(cases)} candidate-blind resize holdout cases to {output_directory}")
    return manifest


def _read_verified(path: Path, record: dict[str, Any]) -> bytes:
    data = path.read_bytes()
    if len(data) != record["byte_length"] or _sha256(data) != record["sha256"]:
        raise RuntimeError(f"artifact authentication failed: {path}")
    return data


def verify_corpus(directory: Path = DEFAULT_OUTPUT) -> tuple[dict[str, Any], bytes, bytes]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("contract_id") != CONTRACT_ID or manifest.get("holdout_seed") != SEED:
        raise RuntimeError("not the frozen resize-v2 holdout")
    generator = ROOT / manifest["generator"]["source"]
    if _sha256_file(generator) != manifest["generator"]["source_sha256"]:
        raise RuntimeError("holdout generator source hash mismatch")
    for provenance_key in ("compatibility_manifest", "contract_document"):
        record = manifest["oracle"][provenance_key]
        if _sha256_file(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"holdout provenance mismatch: {provenance_key}")
    fixture = manifest["fixture_provenance"]
    for record in (fixture["manifest"], fixture["generator"], *fixture["used_files"]):
        path = ROOT / record["path"]
        if _sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"fixture provenance mismatch: {record['path']}")
    natural_asset = manifest["natural_image_provenance"]["asset"]
    natural_path = ROOT / natural_asset["path"]
    if (
        natural_path.stat().st_size != natural_asset["byte_length"]
        or _sha256_file(natural_path) != natural_asset["sha256"]
    ):
        raise RuntimeError("natural-image provenance mismatch")
    sources = _read_verified(
        directory / "sources.rgb8.bin", manifest["artifacts"]["sources.rgb8.bin"]
    )
    pillow = _read_verified(
        directory / "pillow-image-rgb8.bin", manifest["artifacts"]["pillow-image-rgb8.bin"]
    )
    for case in manifest["cases"]:
        for blob, key in ((sources, "source"), (pillow, "pillow_image_rgb8")):
            record = case[key]
            data = blob[record["offset"] : record["offset"] + record["byte_length"]]
            if len(data) != record["byte_length"] or _sha256(data) != record["sha256"]:
                raise RuntimeError(f"case authentication failed: {case['id']}/{key}")
    return manifest, sources, pillow


def _nearest_rank(values: np.ndarray, percentile: float) -> float:
    flat = np.ravel(values)
    rank = max(1, math.ceil(percentile * flat.size))
    return float(np.partition(flat, rank - 1)[rank - 1])


def _gaussian_filter_reflect101(image: np.ndarray) -> np.ndarray:
    coordinates = np.arange(-5, 6, dtype=np.float64)
    kernel = np.exp(-(coordinates * coordinates) / (2.0 * 1.5**2))
    kernel /= kernel.sum(dtype=np.float64)
    padded_x = np.pad(image, ((0, 0), (5, 5)), mode="reflect")
    horizontal = np.zeros_like(image, dtype=np.float64)
    for index, weight in enumerate(kernel):
        horizontal += weight * padded_x[:, index : index + image.shape[1]]
    padded_y = np.pad(horizontal, ((5, 5), (0, 0)), mode="reflect")
    output = np.zeros_like(image, dtype=np.float64)
    for index, weight in enumerate(kernel):
        output += weight * padded_y[index : index + image.shape[0], :]
    return output


def canonical_ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    if reference.shape != candidate.shape or reference.ndim != 2 or min(reference.shape) <= 5:
        raise ValueError("SSIM requires equal 2-D images large enough for five-pixel reflection")
    reference_f64 = reference.astype(np.float64, copy=False)
    candidate_f64 = candidate.astype(np.float64, copy=False)
    mu_reference = _gaussian_filter_reflect101(reference_f64)
    mu_candidate = _gaussian_filter_reflect101(candidate_f64)
    variance_reference = (
        _gaussian_filter_reflect101(reference_f64 * reference_f64) - mu_reference * mu_reference
    )
    variance_candidate = (
        _gaussian_filter_reflect101(candidate_f64 * candidate_f64) - mu_candidate * mu_candidate
    )
    covariance = (
        _gaussian_filter_reflect101(reference_f64 * candidate_f64) - mu_reference * mu_candidate
    )
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    local = ((2.0 * mu_reference * mu_candidate + c1) * (2.0 * covariance + c2)) / (
        (mu_reference * mu_reference + mu_candidate * mu_candidate + c1)
        * (variance_reference + variance_candidate + c2)
    )
    return float(np.mean(local, dtype=np.float64))


def channel_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if (
        reference.shape != candidate.shape
        or reference.dtype != np.uint8
        or candidate.dtype != np.uint8
    ):
        raise ValueError("metrics require equal-shape uint8 channels")
    error = candidate.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(error)
    mse = float(np.mean(error * error, dtype=np.float64))
    signed_bias = float(np.mean(error, dtype=np.float64))
    metrics: dict[str, Any] = {
        "maximum_absolute_error": float(np.max(absolute)),
        "rmse": math.sqrt(mse),
        "absolute_error_p99": _nearest_rank(absolute, 0.99),
        "signed_mean_bias": signed_bias,
        "absolute_signed_mean_bias": abs(signed_bias),
        "ssim": canonical_ssim(reference, candidate),
        "mae": float(np.mean(absolute, dtype=np.float64)),
        "psnr": "Infinity" if mse == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse)),
        "absolute_error_p50": _nearest_rank(absolute, 0.50),
        "absolute_error_p90": _nearest_rank(absolute, 0.90),
    }
    metrics["gates"] = {
        "maximum_absolute_error": metrics["maximum_absolute_error"]
        <= GATES["maximum_absolute_error"],
        "rmse": metrics["rmse"] <= GATES["rmse"],
        "absolute_error_p99": metrics["absolute_error_p99"] <= GATES["absolute_error_p99"],
        "absolute_signed_mean_bias": metrics["absolute_signed_mean_bias"]
        <= GATES["absolute_signed_mean_bias"],
        "ssim": metrics["ssim"] >= GATES["ssim"],
    }
    metrics["passed"] = all(metrics["gates"].values())
    return metrics


def _array(blob: bytes, record: dict[str, Any]) -> np.ndarray:
    data = blob[record["offset"] : record["offset"] + record["byte_length"]]
    return np.frombuffer(data, dtype=np.uint8).reshape(record["shape"])


def _packed_source(blob: bytes, case: dict[str, Any]) -> np.ndarray:
    record = case["source"]
    data = blob[record["offset"] : record["offset"] + record["byte_length"]]
    rows = np.frombuffer(data, dtype=np.uint8).reshape(record["height"], record["stride_bytes"])
    return rows[:, : record["width"] * 3].reshape(record["height"], record["width"], 3)


def compare_candidate(candidate_path: Path, directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    manifest, sources, pillow = verify_corpus(directory)
    candidate = candidate_path.read_bytes()
    expected_length = manifest["artifacts"]["pillow-image-rgb8.bin"]["byte_length"]
    if len(candidate) != expected_length:
        raise RuntimeError(
            f"candidate blob length mismatch: expected {expected_length}, observed {len(candidate)}"
        )
    cases: list[dict[str, Any]] = []
    candidate_arrays: dict[str, np.ndarray] = {}
    for case in manifest["cases"]:
        reference_array = _array(pillow, case["pillow_image_rgb8"])
        candidate_array = _array(candidate, case["pillow_image_rgb8"])
        candidate_arrays[case["id"]] = candidate_array
        channels = []
        for channel_index, channel_name in enumerate(("R", "G", "B")):
            metrics = channel_metrics(
                reference_array[..., channel_index], candidate_array[..., channel_index]
            )
            channels.append({"channel": channel_name, **metrics})
        exact: dict[str, bool] = {}
        for invariant in case["exact_invariants"]:
            if invariant == "no_op_source_bytes":
                exact[invariant] = bool(
                    np.array_equal(candidate_array, _packed_source(sources, case))
                )
            elif invariant == "constant_rgb":
                source_pixel = _packed_source(sources, case)[0, 0]
                exact[invariant] = bool(np.all(candidate_array == source_pixel))
            elif invariant == "identical_channels":
                exact[invariant] = bool(
                    np.array_equal(candidate_array[..., 0], candidate_array[..., 1])
                    and np.array_equal(candidate_array[..., 1], candidate_array[..., 2])
                )
            elif invariant == "zero_source_channels_remain_zero":
                source_array = _packed_source(sources, case)
                zero_channels = [
                    channel for channel in range(3) if not np.any(source_array[..., channel])
                ]
                exact[invariant] = bool(
                    zero_channels
                    and all(not np.any(candidate_array[..., channel]) for channel in zero_channels)
                )
            else:
                raise RuntimeError(f"unknown exact invariant: {invariant}")
        cases.append(
            {
                "id": case["id"],
                "channels": channels,
                "exact_invariants": exact,
                "passed": all(channel["passed"] for channel in channels) and all(exact.values()),
            }
        )

    relation_passes: list[dict[str, Any]] = []
    base = candidate_arrays["channel-permutation-base"]
    permuted = candidate_arrays["channel-permutation-bgr"]
    relation_passes.append(
        {
            "id": "channel-permutation",
            "passed": bool(np.array_equal(permuted, base[..., [2, 1, 0]])),
        }
    )
    passed = all(case["passed"] for case in cases) and all(
        relation["passed"] for relation in relation_passes
    )
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "holdout_manifest": {
            "path": (directory / "manifest.json").relative_to(ROOT).as_posix()
            if (directory / "manifest.json").is_relative_to(ROOT)
            else str(directory / "manifest.json"),
            "sha256": _sha256_file(directory / "manifest.json"),
        },
        "candidate": {
            "path": str(candidate_path),
            "byte_length": len(candidate),
            "sha256": _sha256(candidate),
        },
        "passed": passed,
        "cases": cases,
        "cross_case_exact_invariants": relation_passes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--fixture", type=Path, default=DEFAULT_OUTPUT)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--fixture", type=Path, default=DEFAULT_OUTPUT)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "generate":
        generate(arguments.output)
    elif arguments.command == "verify":
        manifest, _, _ = verify_corpus(arguments.fixture)
        print(f"verified {len(manifest['cases'])} resize-v2 holdout cases")
    else:
        report = compare_candidate(arguments.candidate, arguments.fixture)
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if arguments.output is None:
            print(encoded, end="")
        else:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(encoded, encoding="utf-8")
        if not report["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
