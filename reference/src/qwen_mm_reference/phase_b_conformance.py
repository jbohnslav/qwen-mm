from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .fixtures import repository_root, sha256_file
from .golden import collect_provenance, export_case

ROOT = repository_root()
OUTPUT_DIRECTORY = ROOT / "reference/phase-b/v1"
GENERATOR_PATH = Path(__file__)
COMPATIBILITY_PATH = ROOT / "reference/compatibility/v1.json"
PROFILES = ("qwen3-vl-8b", "qwen3.5-9b")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _pattern(height: int, width: int, seed: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
    y, x = np.indices((height, width), dtype=np.uint32)
    return np.stack(
        (
            (x * 17 + y * 3 + seed * 29) % 256,
            (x * 5 + y * 19 + seed * 47) % 256,
            (x * 11 + y * 7 + seed * 61) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def _save(
    asset_directory: Path,
    image: np.ndarray[Any, Any],
    name: str,
    format_name: str,
    **options: Any,
) -> Path:
    path = asset_directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, "RGB").save(path, format=format_name, **options)
    return path


def _repo_relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _padded_rows(image: np.ndarray[Any, Any], padding: int) -> tuple[bytes, int]:
    height, width, _ = image.shape
    packed = width * 3
    stride = packed + padding
    output = bytearray((height - 1) * stride + packed)
    for row in range(height):
        start = row * stride
        output[start : start + packed] = image[row].tobytes(order="C")
        if row + 1 < height:
            output[start + packed : start + stride] = bytes([0xA0 + row % 31]) * padding
    return bytes(output), stride


def _build_sources(
    asset_directory: Path,
) -> tuple[dict[str, dict[str, Any]], bytes, dict[str, Path]]:
    raw_a = _pattern(65, 97, 1)
    raw_b = _pattern(79, 55, 2)
    codec = _pattern(73, 91, 3)
    paths = {
        "raw-a": _save(asset_directory, raw_a, "raw-a.png", "PNG", compress_level=9),
        "raw-b": _save(asset_directory, raw_b, "raw-b.png", "PNG", compress_level=9),
        "jpeg": _save(asset_directory, codec, "codec.jpg", "JPEG", quality=88, subsampling=2),
        "png": _save(asset_directory, codec, "codec.png", "PNG", compress_level=9),
        "webp-lossless": _save(
            asset_directory, codec, "lossless.webp", "WEBP", lossless=True, method=6
        ),
        "webp-lossy": _save(asset_directory, codec, "lossy.webp", "WEBP", quality=83, method=6),
    }
    padded_a, padded_a_stride = _padded_rows(raw_a, 7)
    values: dict[str, tuple[bytes, dict[str, Any]]] = {
        "raw-a": (
            padded_a,
            {
                "kind": "raw_rgb8",
                "height": 65,
                "width": 97,
                "row_stride": padded_a_stride,
                "padded": True,
            },
        ),
        "raw-b": (
            raw_b.tobytes(order="C"),
            {"kind": "raw_rgb8", "height": 79, "width": 55, "row_stride": 165},
        ),
        "jpeg": (
            paths["jpeg"].read_bytes(),
            {"kind": "encoded", "format": "jpeg", "height": 73, "width": 91},
        ),
        "png": (
            paths["png"].read_bytes(),
            {"kind": "encoded", "format": "png", "height": 73, "width": 91},
        ),
        "webp-lossless": (
            paths["webp-lossless"].read_bytes(),
            {
                "kind": "encoded",
                "format": "webp",
                "height": 73,
                "width": 91,
                "lossless": True,
            },
        ),
        "webp-lossy": (
            paths["webp-lossy"].read_bytes(),
            {
                "kind": "encoded",
                "format": "webp",
                "height": 73,
                "width": 91,
                "lossless": False,
            },
        ),
    }
    blob = bytearray()
    records: dict[str, dict[str, Any]] = {}
    for source_id in sorted(values):
        data, metadata = values[source_id]
        offset = len(blob)
        blob.extend(data)
        records[source_id] = {
            **metadata,
            "offset": offset,
            "length": len(data),
            "sha256": _sha256(data),
        }
    return records, bytes(blob), paths


def _image(source: str, input_index: int, **options: int) -> dict[str, Any]:
    return {"type": "image", "source": source, "input_index": input_index, "options": options}


def _text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}


def _cases() -> list[dict[str, Any]]:
    return [
        {
            "id": "text-only",
            "inputs": [],
            "messages": [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Reply: ready"},
            ],
            "options": {"add_generation_prompt": True},
            "tags": ["text-only", "conditional-keys"],
        },
        {
            "id": "one-raw-rgb",
            "inputs": ["raw-a"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        _image(
                            "raw-a",
                            0,
                            resized_height=64,
                            resized_width=96,
                        ),
                        _text("Describe the raw image."),
                    ],
                }
            ],
            "options": {"add_generation_prompt": True},
            "tags": ["one-image", "raw-rgb", "lossless"],
        },
        {
            "id": "one-jpeg",
            "inputs": ["jpeg"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        _text("Before."),
                        _image("jpeg", 0, resized_height=96, resized_width=64),
                        _text("After."),
                    ],
                }
            ],
            "options": {"add_generation_prompt": True, "add_vision_id": True},
            "tags": ["one-image", "jpeg", "lossy", "vision-id"],
        },
        {
            "id": "interleaved-codecs",
            "inputs": ["png", "webp-lossless", "webp-lossy"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        _image("png", 0, resized_height=64, resized_width=64),
                        _text(" png / "),
                        _image("webp-lossless", 1, resized_height=64, resized_width=96),
                        _text(" lossless / "),
                        _image("webp-lossy", 2, resized_height=96, resized_width=64),
                    ],
                }
            ],
            "options": {"add_generation_prompt": True},
            "tags": ["multi-image", "interleaved", "png", "webp", "lossless", "lossy"],
        },
        {
            "id": "repeat-index-order",
            "inputs": ["raw-a", "raw-b"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        _image("raw-b", 1, resized_height=64, resized_width=64),
                        _text(" then "),
                        _image("raw-a", 0, resized_height=64, resized_width=96),
                        _text(" and again "),
                        _image("raw-b", 1, resized_height=96, resized_width=64),
                    ],
                }
            ],
            "options": {"add_generation_prompt": False},
            "tags": ["multi-image", "interleaved", "repeated-reference", "index-order"],
        },
    ]


def _oracle_case(case: dict[str, Any], asset_paths: dict[str, Path]) -> dict[str, Any]:
    messages = []
    for message in case["messages"]:
        content = message["content"]
        if isinstance(content, str):
            oracle_content: str | list[dict[str, Any]] = content
        else:
            oracle_content = []
            for item in content:
                if item["type"] == "text":
                    oracle_content.append({"type": "text", "text": item["text"]})
                    continue
                oracle_content.append(
                    {
                        "type": "image",
                        "image": {"path": _repo_relative(asset_paths[item["source"]])},
                        **item["options"],
                    }
                )
        messages.append({"role": message["role"], "content": oracle_content})
    return {
        "schema_version": 1,
        "case_id": case["id"],
        "requests": [{"messages": messages, "options": case["options"]}],
    }


def _array_from_descriptor(descriptor: dict[str, Any], root: Path) -> np.ndarray[Any, Any]:
    if descriptor["storage"] == "npy":
        return np.load(root / descriptor["path"], allow_pickle=False)
    return np.asarray(descriptor["data"], dtype=descriptor["dtype"])


def _append_array(descriptor: dict[str, Any], root: Path, blob: bytearray) -> dict[str, Any]:
    array = np.ascontiguousarray(_array_from_descriptor(descriptor, root))
    data = array.tobytes(order="C")
    offset = len(blob)
    blob.extend(data)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "strides": list(array.strides),
        "offset": offset,
        "length": len(data),
        "sha256": _sha256(data),
    }


def generate(output_directory: Path = OUTPUT_DIRECTORY) -> dict[str, Any]:
    cases = _cases()
    expected_blob = bytearray()
    expectations: dict[str, dict[str, Any]] = {case["id"]: {} for case in cases}
    profiles: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="phase-b-assets-", dir=ROOT / "reference") as assets:
        source_records, source_blob, asset_paths = _build_sources(Path(assets))
        with tempfile.TemporaryDirectory(prefix="qwen-mm-phase-b-") as temporary:
            temporary_root = Path(temporary)
            for profile_alias in PROFILES:
                compatibility, provenance = collect_provenance(profile_alias)
                profiles[profile_alias] = {
                    "fingerprint": compatibility["profiles"][profile_alias]["fingerprint"],
                    "model_id": compatibility["profiles"][profile_alias]["model_id"],
                    "revision": compatibility["profiles"][profile_alias]["revision"],
                    "packages": {
                        name: record["observed"] for name, record in provenance["packages"].items()
                    },
                }
                for case in cases:
                    oracle_root = temporary_root / profile_alias / case["id"]
                    golden = export_case(
                        _oracle_case(case, asset_paths),
                        profile_alias=profile_alias,
                        output_directory=oracle_root,
                        inline_max_bytes=0,
                        write_arrays=True,
                    )
                    if golden["status"] != "success":
                        raise RuntimeError(f"oracle case failed: {profile_alias}/{case['id']}")
                    stages = golden["stages"]
                    expectations[case["id"]][profile_alias] = {
                        "rendered_prompt": stages["rendered_prompts"][0],
                        "expanded_prompt": stages["expanded_prompts"][0],
                        "replacements": stages["replacement_offsets"][0],
                        "official_keys": golden["output"]["keys"],
                        "prepared_images": [
                            _append_array(item["array"], oracle_root, expected_blob)
                            for item in stages["prepared_media"]["images"]
                        ],
                        "arrays": {
                            name: _append_array(descriptor, oracle_root, expected_blob)
                            for name, descriptor in golden["output"]["arrays"].items()
                        },
                    }

    for case in cases:
        case["expectations"] = expectations[case["id"]]
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "sources.bin").write_bytes(source_blob)
    (output_directory / "expected.bin").write_bytes(expected_blob)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "contract_id": "qwen-mm-compat-v1",
        "profiles": profiles,
        "generator": {
            "module": "qwen_mm_reference.phase_b_conformance",
            "command": "make phase-b-conformance-regenerate",
            "source_sha256": sha256_file(GENERATOR_PATH),
            "compatibility_sha256": sha256_file(COMPATIBILITY_PATH),
        },
        "blobs": {
            "sources": {
                "path": "sources.bin",
                "length": len(source_blob),
                "sha256": _sha256(source_blob),
            },
            "expected": {
                "path": "expected.bin",
                "length": len(expected_blob),
                "sha256": _sha256(bytes(expected_blob)),
            },
        },
        "sources": source_records,
        "cases": cases,
    }
    manifest["integrity"] = {
        "algorithm": "sha256",
        "canonical_json_sha256": _sha256(_canonical(manifest)),
    }
    _write_json(output_directory / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase B end-to-end oracle corpus")
    parser.add_argument("--output-directory", type=Path, default=OUTPUT_DIRECTORY)
    args = parser.parse_args()
    manifest = generate(args.output_directory)
    print(
        json.dumps(
            {
                "cases": len(manifest["cases"]),
                "profiles": list(manifest["profiles"]),
                "sources_sha256": manifest["blobs"]["sources"]["sha256"],
                "expected_sha256": manifest["blobs"]["expected"]["sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
