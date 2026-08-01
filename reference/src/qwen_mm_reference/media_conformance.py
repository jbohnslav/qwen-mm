"""Generate the deterministic B6 encoded-media/prepared-RGB oracle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, features
from qwen_vl_utils import smart_resize
from qwen_vl_utils.vision_process import to_rgb

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "reference" / "media" / "v1"
COMPATIBILITY_PATH = REPOSITORY_ROOT / "reference" / "compatibility" / "v1.json"
GENERATOR_COMMAND = (
    "./scripts/with-cargo.sh uv run --locked --no-sync "
    "--package qwen-mm-reference python -m qwen_mm_reference.media_conformance"
)
EXPECTED_LIBJPEG_TURBO = "3.1.4.1"
EXPECTED_LIBWEBP = "1.6.0"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _append(buffer: bytearray, data: bytes) -> dict[str, Any]:
    record = {"offset": len(buffer), "byte_length": len(data), "sha256": _sha256(data)}
    buffer.extend(data)
    return record


def _rgb_pattern(height: int, width: int, seed: int) -> np.ndarray:
    y, x = np.indices((height, width), dtype=np.uint32)
    return np.stack(
        (
            (x * 37 + y * 11 + seed) % 256,
            (x * 7 + y * 53 + seed * 3) % 256,
            (x * 19 + y * 23 + seed * 5) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def _l_pattern(height: int, width: int) -> np.ndarray:
    y, x = np.indices((height, width), dtype=np.uint32)
    return ((x * 29 + y * 47 + 13) % 256).astype(np.uint8)


def _rgba_pattern(height: int, width: int) -> np.ndarray:
    rgb = _rgb_pattern(height, width, 17)
    y, x = np.indices((height, width), dtype=np.uint32)
    alpha = ((x * 31 + y * 43) % 256).astype(np.uint8)
    alpha.reshape(-1)[:9] = np.array([0, 1, 2, 63, 127, 128, 129, 254, 255], dtype=np.uint8)
    return np.concatenate((rgb, alpha[..., None]), axis=-1)


def _cmyk_pattern(height: int, width: int) -> np.ndarray:
    y, x = np.indices((height, width), dtype=np.uint32)
    return np.stack(
        (
            (x * 17 + 3) % 256,
            (y * 31 + 7) % 256,
            (x * 5 + y * 13 + 11) % 256,
            (x * 23 + y * 19 + 29) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def _encode(image: Image.Image, image_format: str, **kwargs: Any) -> bytes:
    output = io.BytesIO()
    image.save(output, format=image_format, **kwargs)
    return output.getvalue()


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFF_FFFF)
    )


def _oversized_png() -> bytes:
    ihdr = struct.pack(">IIBBBBB", 32_769, 1, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IEND", b"")


def _pixel_bomb_png() -> bytes:
    ihdr = struct.pack(">IIBBBBB", 8_193, 8_193, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IEND", b"")


def _malformed_png_tuple(
    bit_depth: int,
    color_type: int,
    *,
    compression_method: int = 0,
    corrupt_ihdr_crc: bool = False,
) -> bytes:
    ihdr = struct.pack(">IIBBBBB", 64, 64, bit_depth, color_type, compression_method, 0, 0)
    encoded = bytearray(b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IEND", b""))
    if corrupt_ihdr_crc:
        encoded[32] ^= 1
    return bytes(encoded)


def _malformed_apng_control(payload: bytes) -> bytes:
    ihdr = struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"acTL", payload)
        + _png_chunk(b"IEND", b"")
    )


def _oversized_jpeg() -> bytes:
    return bytes(
        [
            0xFF,
            0xD8,
            0xFF,
            0xC0,
            0x00,
            0x11,
            0x08,
            0x80,
            0x01,
            0x00,
            0x01,
            0x03,
            0x01,
            0x11,
            0x00,
            0x02,
            0x11,
            0x00,
            0x03,
            0x11,
            0x00,
        ]
    )


def _oversized_webp() -> bytes:
    payload = bytes([0, 0, 0, 0, 0x00, 0x80, 0x00, 0, 0, 0])
    chunk = b"VP8X" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk


def _oriented_jpeg(orientation: int) -> bytes:
    exif = Image.Exif()
    exif[274] = orientation
    return _encode(
        Image.fromarray(_rgb_pattern(64, 96, 40 + orientation), mode="RGB"),
        "JPEG",
        quality=91,
        subsampling=0,
        optimize=False,
        progressive=False,
        exif=exif,
    )


def _with_adobe_transform(encoded: bytes, transform: int) -> bytes:
    adobe = encoded.find(b"Adobe")
    if adobe < 4 or adobe + 12 > len(encoded):
        raise RuntimeError("CMYK JPEG does not contain the expected Adobe APP14 marker")
    if encoded[adobe - 4 : adobe - 2] != b"\xff\xee":
        raise RuntimeError("Adobe signature is not inside APP14")
    modified = bytearray(encoded)
    modified[adobe + 11] = transform
    return bytes(modified)


def _image_specs() -> list[dict[str, Any]]:
    rgba = Image.fromarray(_rgba_pattern(64, 96), mode="RGBA")
    palette = Image.fromarray((_l_pattern(64, 64) % 4).astype(np.uint8), mode="P")
    palette.putpalette(
        [value for index in range(256) for value in (index, 255 - index, index // 2)]
    )
    la = np.stack((_l_pattern(64, 64), np.flipud(_l_pattern(64, 64))), axis=-1)
    cmyk_jpeg = _encode(
        Image.fromarray(_cmyk_pattern(64, 64), mode="CMYK"),
        "JPEG",
        quality=90,
        optimize=False,
        progressive=False,
    )
    return [
        {
            "id": "png-rgb-aligned-no-op",
            "format": "png",
            "encoded": _encode(
                Image.fromarray(_rgb_pattern(64, 96, 1), mode="RGB"), "PNG", compress_level=9
            ),
            "options": {},
            "tolerance": 0,
            "tags": ["png", "rgb", "no_op", "aligned", "lossless"],
        },
        {
            "id": "png-luma-off-grid-resize",
            "format": "png",
            "encoded": _encode(
                Image.fromarray(_l_pattern(65, 97), mode="L"), "PNG", compress_level=9
            ),
            "options": {},
            "tolerance": 0,
            "tags": ["png", "grayscale", "resize", "lossless"],
        },
        {
            "id": "png-rgba-white-composite",
            "format": "png",
            "encoded": _encode(rgba, "PNG", compress_level=9),
            "options": {},
            "tolerance": 0,
            "tags": ["png", "rgba", "alpha", "white_composite", "lossless"],
        },
        {
            "id": "png-rgb-trns-ignored",
            "format": "png",
            "encoded": _encode(
                Image.fromarray(_rgb_pattern(64, 96, 20), mode="RGB"),
                "PNG",
                compress_level=9,
                transparency=(20, 60, 100),
            ),
            "options": {},
            "tolerance": 0,
            "tags": ["png", "rgb", "trns", "alpha_ignored", "lossless"],
        },
        {
            "id": "png-luma-trns-ignored",
            "format": "png",
            "encoded": _encode(
                Image.fromarray(_l_pattern(64, 96), mode="L"),
                "PNG",
                compress_level=9,
                transparency=128,
            ),
            "options": {},
            "tolerance": 0,
            "tags": ["png", "grayscale", "trns", "alpha_ignored", "lossless"],
        },
        {
            "id": "jpeg-rgb-aligned-no-op",
            "format": "jpeg",
            "encoded": _encode(
                Image.fromarray(_rgb_pattern(64, 96, 2), mode="RGB"),
                "JPEG",
                quality=90,
                subsampling=0,
                optimize=False,
                progressive=False,
            ),
            "options": {},
            "tolerance": 1,
            "tags": ["jpeg", "rgb", "no_op", "aligned", "lossy"],
        },
        {
            "id": "jpeg-rgb-off-grid-resize",
            "format": "jpeg",
            "encoded": _encode(
                Image.fromarray(_rgb_pattern(71, 109, 3), mode="RGB"),
                "JPEG",
                quality=87,
                subsampling=2,
                optimize=False,
                progressive=True,
            ),
            "options": {},
            "tolerance": 1,
            "tags": ["jpeg", "rgb", "resize", "progressive", "lossy"],
        },
        {
            "id": "jpeg-cmyk-aligned-no-op",
            "format": "jpeg",
            "encoded": cmyk_jpeg,
            "options": {},
            "tolerance": 1,
            "tags": ["jpeg", "cmyk", "no_op", "lossy", "adobe_app14"],
        },
        {
            "id": "jpeg-ycck-adobe-transform-2",
            "format": "jpeg",
            "encoded": _with_adobe_transform(cmyk_jpeg, 2),
            "options": {},
            "tolerance": 1,
            "tags": ["jpeg", "ycck", "cmyk", "lossy", "adobe_transform_2"],
        },
        {
            "id": "jpeg-luma-aligned-no-op",
            "format": "jpeg",
            "encoded": _encode(
                Image.fromarray(_l_pattern(64, 64), mode="L"),
                "JPEG",
                quality=90,
                optimize=False,
                progressive=False,
            ),
            "options": {},
            "tolerance": 1,
            "tags": ["jpeg", "grayscale", "no_op", "lossy"],
        },
        *[
            {
                "id": f"jpeg-exif-orientation-{orientation}-ignored",
                "format": "jpeg",
                "encoded": _oriented_jpeg(orientation),
                "options": {},
                "tolerance": 1,
                "tags": [
                    "jpeg",
                    f"exif_orientation_{orientation}",
                    "stored_raster",
                    "lossy",
                ],
            }
            for orientation in range(1, 9)
        ],
        {
            "id": "webp-lossless-rgb-aligned-no-op",
            "format": "webp",
            "encoded": _encode(
                Image.fromarray(_rgb_pattern(64, 96, 5), mode="RGB"),
                "WEBP",
                lossless=True,
                method=6,
            ),
            "options": {},
            "tolerance": 0,
            "tags": ["webp", "rgb", "no_op", "lossless"],
        },
        {
            "id": "webp-lossless-rgb-off-grid-resize",
            "format": "webp",
            "encoded": _encode(
                Image.fromarray(_rgb_pattern(71, 109, 18), mode="RGB"),
                "WEBP",
                lossless=True,
                method=6,
            ),
            "options": {},
            "tolerance": 0,
            "tags": ["webp", "rgb", "resize", "off_grid", "lossless"],
        },
        {
            "id": "webp-lossless-rgba-white-composite",
            "format": "webp",
            "encoded": _encode(rgba, "WEBP", lossless=True, method=6, exact=True),
            "options": {},
            "tolerance": 0,
            "tags": ["webp", "rgba", "alpha", "white_composite", "lossless"],
        },
        {
            "id": "webp-lossy-aligned-no-op",
            "format": "webp",
            "encoded": _encode(
                Image.fromarray(_rgb_pattern(64, 96, 8), mode="RGB"),
                "WEBP",
                lossless=False,
                quality=80,
                method=6,
            ),
            "options": {},
            "tolerance": 1,
            "tags": ["webp", "rgb", "no_op", "lossy"],
        },
        {
            "id": "webp-lossy-off-grid-resize",
            "format": "webp",
            "encoded": _encode(
                Image.fromarray(_rgb_pattern(71, 109, 6), mode="RGB"),
                "WEBP",
                lossless=False,
                quality=80,
                method=6,
            ),
            "options": {},
            "tolerance": 1,
            "tags": ["webp", "rgb", "resize", "lossy"],
        },
        {
            "id": "webp-lossy-rgba-white-composite",
            "format": "webp",
            "encoded": _encode(
                rgba,
                "WEBP",
                lossless=False,
                quality=80,
                method=6,
                exact=True,
            ),
            "options": {},
            "tolerance": 1,
            "tags": [
                "webp",
                "rgba",
                "vp8x",
                "alpha",
                "white_composite",
                "lossy",
            ],
        },
        {
            "id": "png-unusual-aspect-200",
            "format": "png",
            "encoded": _encode(
                Image.fromarray(_rgb_pattern(32, 6400, 7), mode="RGB"), "PNG", compress_level=9
            ),
            "options": {},
            "tolerance": 0,
            "tags": ["png", "unusual_aspect", "aspect_200", "lossless"],
        },
        {
            "id": "png-la-rejected-by-v1-envelope",
            "format": "png",
            "encoded": _encode(Image.fromarray(la, mode="LA"), "PNG", compress_level=9),
            "expected_error": "unsupported_media",
            "tags": ["png", "la", "alpha", "unsupported_mode"],
        },
        {
            "id": "png-palette-rejected",
            "format": "png",
            "encoded": _encode(palette, "PNG", compress_level=9),
            "expected_error": "unsupported_media",
            "tags": ["png", "palette", "unsupported_mode"],
        },
        {
            "id": "png-16bit-rejected",
            "format": "png",
            "encoded": _encode(Image.fromarray(_l_pattern(64, 64).astype(np.uint16) * 257), "PNG"),
            "expected_error": "unsupported_media",
            "tags": ["png", "16bit", "unsupported_mode"],
        },
    ]


def _oracle(encoded: bytes, options: dict[str, int]) -> tuple[bytes, int, int]:
    with Image.open(io.BytesIO(encoded)) as source:
        source.load()
        rgb = to_rgb(source)
        destination_height, destination_width = smart_resize(
            rgb.height,
            rgb.width,
            factor=32,
            min_pixels=options.get("min_pixels"),
            max_pixels=options.get("max_pixels"),
        )
        if rgb.size != (destination_width, destination_height):
            rgb = rgb.resize((destination_width, destination_height))
        array = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        return array.tobytes(order="C"), destination_height, destination_width


def generate(output_directory: Path) -> None:
    compatibility_bytes = COMPATIBILITY_PATH.read_bytes()
    compatibility = json.loads(compatibility_bytes)
    versions = {
        name: importlib.metadata.version(name) for name in ("Pillow", "numpy", "qwen-vl-utils")
    }
    expected_versions = {name: compatibility["packages"][name] for name in versions}
    if versions != expected_versions:
        raise RuntimeError(
            f"locked package mismatch: expected {expected_versions}, observed {versions}"
        )
    codec_versions = {
        "libjpeg_turbo": features.version("libjpeg_turbo"),
        "libwebp": features.version("webp"),
    }
    expected_codecs = {
        "libjpeg_turbo": EXPECTED_LIBJPEG_TURBO,
        "libwebp": EXPECTED_LIBWEBP,
    }
    if codec_versions != expected_codecs:
        raise RuntimeError(
            f"locked codec mismatch: expected {expected_codecs}, observed {codec_versions}"
        )

    encoded_blob = bytearray()
    prepared_blob = bytearray()
    cases: list[dict[str, Any]] = []
    specs = _image_specs()

    # Damage and resource cases are committed beside successful color/codec cases.
    valid_by_format = {spec["format"]: spec["encoded"] for spec in specs if "tolerance" in spec}
    animated = io.BytesIO()
    apng = io.BytesIO()
    frame0 = Image.fromarray(_rgb_pattern(64, 64, 9), mode="RGB")
    frame1 = Image.fromarray(_rgb_pattern(64, 64, 10), mode="RGB")
    frame0.save(
        animated,
        format="WEBP",
        save_all=True,
        append_images=[frame1],
        duration=100,
        loop=0,
        lossless=True,
    )
    frame0.save(apng, format="PNG", save_all=True, append_images=[frame1], duration=100, loop=0)
    gif = _encode(frame0, "GIF")
    specs.extend(
        [
            {
                "id": "jpeg-missing-terminal-eoi",
                "format": "jpeg",
                "encoded": valid_by_format["jpeg"][:-2],
                "expected_error": "media_decode",
                "tags": ["jpeg", "truncated", "missing_eoi"],
            },
            {
                "id": "jpeg-truncated",
                "format": "jpeg",
                "encoded": valid_by_format["jpeg"][:-64],
                "expected_error": "media_decode",
                "tags": ["jpeg", "truncated"],
            },
            {
                "id": "png-truncated",
                "format": "png",
                "encoded": valid_by_format["png"][:40],
                "expected_error": "media_decode",
                "tags": ["png", "truncated"],
            },
            {
                "id": "webp-truncated",
                "format": "webp",
                "encoded": valid_by_format["webp"][:-16],
                "expected_error": "media_decode",
                "tags": ["webp", "truncated"],
            },
            {
                "id": "jpeg-oversized-header",
                "format": "jpeg",
                "encoded": _oversized_jpeg(),
                "expected_error": "resource_limit",
                "tags": ["jpeg", "malicious_header", "edge_limit"],
            },
            {
                "id": "png-oversized-header",
                "format": "png",
                "encoded": _oversized_png(),
                "expected_error": "resource_limit",
                "tags": ["png", "malicious_header", "edge_limit"],
            },
            {
                "id": "webp-oversized-header",
                "format": "webp",
                "encoded": _oversized_webp(),
                "expected_error": "resource_limit",
                "tags": ["webp", "malicious_header", "edge_limit"],
            },
            {
                "id": "png-pixel-bomb-header",
                "format": "png",
                "encoded": _pixel_bomb_png(),
                "expected_error": "resource_limit",
                "tags": ["png", "malicious_header", "decoded_pixel_limit"],
            },
            {
                "id": "png-malformed-rgb-bit-depth",
                "format": "png",
                "encoded": _malformed_png_tuple(3, 2),
                "expected_error": "media_decode",
                "tags": ["png", "malformed_header", "invalid_bit_depth"],
            },
            {
                "id": "png-malformed-reserved-color-type",
                "format": "png",
                "encoded": _malformed_png_tuple(8, 1),
                "expected_error": "media_decode",
                "tags": ["png", "malformed_header", "reserved_color_type"],
            },
            {
                "id": "png-malformed-method-before-excluded-la",
                "format": "png",
                "encoded": _malformed_png_tuple(8, 4, compression_method=1),
                "expected_error": "media_decode",
                "tags": ["png", "malformed_header", "invalid_method", "la"],
            },
            {
                "id": "png-malformed-crc-before-excluded-la",
                "format": "png",
                "encoded": _malformed_png_tuple(8, 4, corrupt_ihdr_crc=True),
                "expected_error": "media_decode",
                "tags": ["png", "malformed_header", "invalid_crc", "la"],
            },
            {
                "id": "apng-malformed-control-length",
                "format": "png",
                "encoded": _malformed_apng_control(struct.pack(">I", 1)),
                "expected_error": "media_decode",
                "tags": ["png", "apng", "malformed_header", "invalid_chunk_length"],
            },
            {
                "id": "apng-malformed-zero-frames",
                "format": "png",
                "encoded": _malformed_apng_control(struct.pack(">II", 0, 0)),
                "expected_error": "media_decode",
                "tags": ["png", "apng", "malformed_header", "zero_frames"],
            },
            {
                "id": "jpeg-malformed-baseline-precision",
                "format": "jpeg",
                "encoded": bytes(
                    [
                        0xFF,
                        0xD8,
                        0xFF,
                        0xC0,
                        0x00,
                        0x11,
                        0x0C,
                        0x00,
                        0x40,
                        0x00,
                        0x40,
                        0x03,
                        0x01,
                        0x11,
                        0x00,
                        0x02,
                        0x11,
                        0x00,
                        0x03,
                        0x11,
                        0x00,
                    ]
                ),
                "expected_error": "media_decode",
                "tags": ["jpeg", "malformed_header", "invalid_precision"],
            },
            {
                "id": "jpeg-malformed-zero-components",
                "format": "jpeg",
                "encoded": b"\xff\xd8\xff\xc0\x00\x08\x08\x00\x40\x00\x40\x00",
                "expected_error": "media_decode",
                "tags": ["jpeg", "malformed_header", "invalid_component_count"],
            },
            {
                "id": "jpeg-malformed-excluded-component-sampling",
                "format": "jpeg",
                "encoded": b"\xff\xd8\xff\xc0\x00\x0e\x08\x00\x40\x00\x40\x02\x01\x01\x00\x02\x11\x00",
                "expected_error": "media_decode",
                "tags": ["jpeg", "malformed_header", "invalid_sampling", "components_2"],
            },
            {
                "id": "webp-malformed-reserved-animation-flags",
                "format": "webp",
                "encoded": b"RIFF\x16\x00\x00\x00WEBPVP8X\x0a\x00\x00\x00\x83\x00\x00\x00\x3f\x00\x00\x3f\x00\x00",
                "expected_error": "media_decode",
                "tags": ["webp", "malformed_header", "reserved_flags"],
            },
            {
                "id": "webp-malformed-lossless-version",
                "format": "webp",
                "encoded": b"RIFF\x11\x00\x00\x00WEBPVP8L\x05\x00\x00\x00\x2f\x00\x00\x00\xe0",
                "expected_error": "media_decode",
                "tags": ["webp", "malformed_header", "invalid_version"],
            },
            {
                "id": "webp-malformed-extended-length",
                "format": "webp",
                "encoded": b"RIFF\x17\x00\x00\x00WEBPVP8X\x0b\x00\x00\x00\x02\x00\x00\x00\x3f\x00\x00\x3f\x00\x00\x00",
                "expected_error": "media_decode",
                "tags": ["webp", "malformed_header", "invalid_chunk_length"],
            },
            {
                "id": "webp-chunk-outside-declared-riff",
                "format": "webp",
                "encoded": b"RIFF\x04\x00\x00\x00WEBPVP8X\x0a\x00\x00\x00\x02\x00\x00\x00\x3f\x00\x00\x3f\x00\x00",
                "expected_error": "media_decode",
                "tags": ["webp", "malformed_header", "riff_extent"],
            },
            {
                "id": "png-valid-aspect-201-rejected",
                "format": "png",
                "encoded": _encode(
                    Image.fromarray(_rgb_pattern(32, 6_432, 19), mode="RGB"),
                    "PNG",
                    compress_level=9,
                ),
                "expected_error": "media_geometry",
                "tags": ["png", "valid_decode", "aspect_201", "media_geometry"],
            },
            {
                "id": "webp-animated-rejected",
                "format": "webp",
                "encoded": animated.getvalue(),
                "expected_error": "unsupported_media",
                "tags": ["webp", "animated", "unsupported_media"],
            },
            {
                "id": "apng-animated-rejected",
                "format": "png",
                "encoded": apng.getvalue(),
                "expected_error": "unsupported_media",
                "tags": ["png", "apng", "animated", "unsupported_media"],
            },
            {
                "id": "gif-codec-rejected",
                "format": "jpeg",
                "encoded": gif,
                "expected_error": "unsupported_media",
                "tags": ["gif", "unsupported_codec"],
            },
            {
                "id": "jpeg-declared-png-mismatch",
                "format": "jpeg",
                "encoded": valid_by_format["png"],
                "expected_error": "media_decode",
                "tags": ["format_mismatch", "media_decode"],
            },
        ]
    )

    for specification in specs:
        encoded = specification["encoded"]
        case: dict[str, Any] = {
            "id": specification["id"],
            "format": specification["format"],
            "tags": specification["tags"],
            "options": specification.get("options", {}),
            "encoded": _append(encoded_blob, encoded),
        }
        if "expected_error" in specification:
            case["expected_error"] = {"category": specification["expected_error"]}
        else:
            prepared, height, width = _oracle(encoded, case["options"])
            case["prepared_rgb8"] = {
                "height": height,
                "width": width,
                "row_stride_bytes": width * 3,
                "absolute_byte_error_max": specification["tolerance"],
                **_append(prepared_blob, prepared),
            }
        cases.append(case)

    output_directory.mkdir(parents=True, exist_ok=True)
    encoded_path = output_directory / "encoded.bin"
    prepared_path = output_directory / "prepared-rgb8.bin"
    encoded_path.write_bytes(encoded_blob)
    prepared_path.write_bytes(prepared_blob)
    manifest = {
        "schema_version": 1,
        "contract_id": compatibility["contract_id"],
        "stage": "prepared_hwc_rgb8",
        "generator_command": GENERATOR_COMMAND,
        "provenance": {
            "generator": {
                "path": str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)),
                "sha256": _sha256(Path(__file__).read_bytes()),
            },
            "compatibility_manifest": {
                "path": str(COMPATIBILITY_PATH.relative_to(REPOSITORY_ROOT)),
                "sha256": _sha256(compatibility_bytes),
            },
            "profile_fingerprints": {
                alias: profile["fingerprint"]
                for alias, profile in compatibility["profiles"].items()
            },
        },
        "oracle": {
            **versions,
            **codec_versions,
            "capture_platform": platform.platform(),
            "python": sys.version,
            "exif_policy": "stored raster; no transpose",
            "rgba_policy": "white background via Pillow paste with alpha mask",
            "la_policy": "unsupported_media by accepted ADR envelope",
        },
        "artifacts": {
            "encoded.bin": {"byte_length": len(encoded_blob), "sha256": _sha256(encoded_blob)},
            "prepared-rgb8.bin": {
                "byte_length": len(prepared_blob),
                "sha256": _sha256(prepared_blob),
            },
        },
        "cases": cases,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["integrity"] = {
        "algorithm": "sha256",
        "canonical_json_without_integrity_sha256": _sha256(canonical),
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    generate(arguments.output)


if __name__ == "__main__":
    main()
