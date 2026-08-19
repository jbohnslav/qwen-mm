from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_ID = "qwen-mm-phase-c-conformance-report-v1"
SCHEMA_VERSION = 1
REPORT_PATH = Path("reference/phase-c/v1/report.json")
SUMMARY_PATH = Path("reference/phase-c/v1/summary.md")
SCHEMA_PATH = Path("reference/phase-c/v1/schema-v1.json")
PROFILES = ("qwen3-vl-8b", "qwen3.5-9b")
FORBIDDEN_CANDIDATE_MODULES = ("PIL", "torch", "torchvision", "transformers", "qwen_vl_utils")
PHASE_E_EXCLUSIONS = (
    "golden:qwen3-vl-8b/multimodal-smoke:raw-video-component",
    "chat:qwen3-vl-8b-visual-padding:mixed-video-case",
    "chat:qwen3.5-9b-visual-padding:mixed-video-case",
    "live:visual-occurrence-order:image-video-interleave-leg",
    "live:resource-and-overflow-matrix:raw_frames",
    "rule:nframes-half-up-to-even",
    "rule:nframes-half-down-to-even",
    "rule:fps-default-min",
    "rule:fps-clamped-max",
    "rule:fps-and-nframes-exclusive",
    "rule:nframes-below-factor",
    "rule:sample-indices-five",
    "rule:video-placeholder-count",
)
PERFORMANCE_WORDS = ("latency", "throughput", "speedup", "timing", "wall_ms")
CANONICAL_COMPARATOR = {
    "policy_id": "qwen-mm-compat-v1-comparison",
    "prepared_lossless_absolute_byte_error_max": 0,
    "prepared_lossy_absolute_byte_error_max": 1,
    "final_strict_atol": 1.0e-6,
    "final_lossy_atol": 0.007844090461730957,
    "diagnostic_fields": [
        "maximum_absolute_error",
        "maximum_ulp_error",
        "rmse",
        "absolute_error_percentiles",
        "count_over_threshold",
        "first_difference",
    ],
}
_OBSERVED_CANDIDATE_PROVENANCE: dict[str, Any] | None = None
RAW_LAYOUT_IDS = (
    "contiguous-hwc",
    "row-padded-hwc",
    "negative-row-stride",
    "channel-sliced-invalid",
)
RESOURCE_AXIS_IDS = (
    "requests",
    "messages",
    "content_items",
    "text_bytes",
    "media_occurrences_request",
    "media_occurrences_batch",
    "encoded_bytes_item",
    "encoded_bytes_batch",
    "decoded_pixels",
    "edge_length",
    "prepared_pixels",
    "tokens_request",
    "tokens_batch",
    "output_bytes",
)
BOUNDARY_POSITIONS = ("one-below", "at", "one-above")
HETEROGENEOUS_ORDERS = ("identity", "permutation-0", "permutation-1")
SCHEMA_ERROR_IDS = (
    "request-unknown-key",
    "request-missing-messages",
    "messages-not-list",
    "message-unknown-key",
    "message-missing-role",
    "invalid-role",
    "wrong-content-type",
    "missing-image-reference",
    "out-of-range-image-reference",
    "mixed-valid-invalid-batch",
)
GEOMETRY_CASE_IDS = (
    "default-min-below",
    "default-min-edge",
    "explicit-min",
    "explicit-max",
    "exact-budget",
    "explicit-paired",
    "explicit-unpaired",
    "aspect-ratio-200",
    "aspect-ratio-over-200",
    "tie-48-up-even",
    "tie-80-down-even",
    "tie-112-up-even",
)
BASE_SOURCE_INPUTS = (
    Path("reference/compatibility/v1.json"),
    Path("reference/conformance/v1/corpus.json"),
    Path("reference/conformance/v1/rules.json"),
    Path("reference/conformance/v1/chat.json"),
    Path("reference/media/v1/manifest.json"),
    Path("reference/media/v1/encoded.bin"),
    Path("reference/media/v1/prepared-rgb8.bin"),
    Path("reference/phase-b/v1/manifest.json"),
    Path("reference/phase-b/v1/sources.bin"),
    Path("reference/phase-b/v1/expected.bin"),
    Path("reference/goldens/schema-v1.json"),
    Path("Cargo.lock"),
    Path("uv.lock"),
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def source_inputs() -> list[Path]:
    """Return every authenticated input or gate-affecting source, deterministically."""
    root = repository_root()
    paths = set(BASE_SOURCE_INPUTS)
    directories = (
        (root / "reference" / "goldens" / "v1", {".json", ".npy"}),
        (root / "reference" / "cases" / "v1", {".json", ".bin"}),
        (root / "fixtures" / "baseline", {".json", ".jpg", ".jpeg", ".png", ".webp"}),
        (root / "crates" / "qwen-mm-core" / "src", {".rs"}),
        (root / "crates" / "qwen-mm-python" / "src", {".rs"}),
        (root / "crates" / "qwen-mm-python" / "python", {".py"}),
        (root / "reference" / "src" / "qwen_mm_reference", {".py"}),
    )
    paths.update(
        path.relative_to(root)
        for directory, suffixes in directories
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes and "__pycache__" not in path.parts
    )
    paths.update(
        {
            Path("Cargo.toml"),
            Path("crates/qwen-mm-core/Cargo.toml"),
            Path("crates/qwen-mm-python/Cargo.toml"),
            Path("pyproject.toml"),
            Path("reference/pyproject.toml"),
            Path("reference/models.json"),
            Path("reference/src/qwen_mm_reference/conformance.py"),
            Path("reference/src/qwen_mm_reference/golden.py"),
            Path("reference/src/qwen_mm_reference/phase_c_conformance.py"),
            Path("reference/phase-c/v1/schema-v1.json"),
            Path("scripts/test-phase-c-conformance.sh"),
            Path("scripts/with-cargo.sh"),
            Path("scripts/cargo.sh"),
            Path("Makefile"),
            Path("rust-toolchain.toml"),
        }
    )
    return sorted(path for path in paths if (root / path).is_file())


def expected_candidate_case_ids(root: Path | None = None) -> list[str]:
    """Canonical installed-wheel inventory, independently derived from source fixtures."""
    root = repository_root() if root is None else root
    phase_b = _json(root / "reference/phase-b/v1/manifest.json")
    chat = _json(root / "reference/conformance/v1/chat.json")
    media = _json(root / "reference/media/v1/manifest.json")
    ids = {
        *(f"phase-b:{profile}:{case['id']}" for profile in PROFILES for case in phase_b["cases"]),
        *(
            f"chat:{case['id']}"
            for case in chat["cases"]
            if case["id"] not in {"qwen3-vl-8b-visual-padding", "qwen3.5-9b-visual-padding"}
        ),
        *(f"media:{profile}:{case['id']}" for profile in PROFILES for case in media["cases"]),
        *(
            f"golden:{profile}:{case_id}"
            for profile in PROFILES
            for case_id in ("text-smoke", "corrupt-image", "image24")
        ),
        *(f"raw-layout:{profile}:{layout}" for profile in PROFILES for layout in RAW_LAYOUT_IDS),
        *(
            f"resource:{profile}:{axis}:{position}"
            for profile in PROFILES
            for axis in RESOURCE_AXIS_IDS
            for position in BOUNDARY_POSITIONS
        ),
        *(
            f"heterogeneous:{profile}:{order}"
            for profile in PROFILES
            for order in HETEROGENEOUS_ORDERS
        ),
        *(
            f"schema-error:{profile}:{case_id}"
            for profile in PROFILES
            for case_id in SCHEMA_ERROR_IDS
        ),
        *(f"arithmetic:{profile}:explicit-resized-pixel-overflow" for profile in PROFILES),
        *(f"geometry:{profile}:{case_id}" for profile in PROFILES for case_id in GEOMETRY_CASE_IDS),
    }
    return sorted(ids)


def expected_installed_boundary_rule_map() -> dict[str, Any]:
    geometry = {
        "resize-default-min-below": "default-min-below",
        "resize-default-min-edge": "default-min-edge",
        "resize-explicit-min": "explicit-min",
        "resize-explicit-max": "explicit-max",
        "resize-exact-budget": "exact-budget",
        "aspect-ratio-200": "aspect-ratio-200",
        "aspect-ratio-over-200": "aspect-ratio-over-200",
        "explicit-dimensions-paired": "explicit-paired",
        "explicit-dimensions-unpaired": "explicit-unpaired",
        "image-placeholder-count": "exact-budget",
    }
    mapping: dict[str, Any] = {
        rule_id: {
            "classification": "installed-boundary",
            "candidate_case_ids": [f"geometry:{profile}:{candidate_id}" for profile in PROFILES],
            "fixture_authentication_id": f"rule:{rule_id}",
        }
        for rule_id, candidate_id in geometry.items()
    }
    mapping["checked-product-overflow"] = {
        "classification": "installed-boundary",
        "candidate_case_ids": [
            f"arithmetic:{profile}:explicit-resized-pixel-overflow" for profile in PROFILES
        ],
        "fixture_authentication_id": "rule:checked-product-overflow",
    }
    public_rounding = {
        "round-half-2-even": "tie-48-up-even",
        "round-half-2-even-down": "tie-80-down-even",
        "round-half-4-even": "tie-112-up-even",
    }
    for rule_id, candidate_id in public_rounding.items():
        mapping[rule_id] = {
            "classification": "installed-boundary",
            "candidate_case_ids": [f"geometry:{profile}:{candidate_id}" for profile in PROFILES],
            "fixture_authentication_id": f"rule:{rule_id}",
        }
    for rule_id in ("round-half-0-even", "checked-product-largest"):
        mapping[rule_id] = {
            "classification": "intrinsic-helper-only",
            "candidate_case_ids": [],
            "fixture_authentication_id": f"rule:{rule_id}",
        }
    return dict(sorted(mapping.items()))


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wheel_runtime_identity(wheel_path: Path) -> dict[str, str]:
    """Authenticate the package artifacts embedded in the supplied wheel."""
    with zipfile.ZipFile(wheel_path) as archive:
        names = archive.namelist()
        package_members = [name for name in names if name == "qwen_mm/__init__.py"]
        native_members = [
            name
            for name in names
            if name.startswith("qwen_mm/_native.") and name.endswith((".so", ".pyd", ".dylib"))
        ]
        metadata_members = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(package_members) != 1 or len(native_members) != 1 or len(metadata_members) != 1:
            raise ValueError(
                "wheel does not contain one qwen_mm package, native module, and METADATA"
            )
        metadata = archive.read(metadata_members[0]).decode("utf-8")
        versions = [
            line.partition(":")[2].strip()
            for line in metadata.splitlines()
            if line.startswith("Version:")
        ]
        if len(versions) != 1 or not versions[0]:
            raise ValueError("wheel METADATA does not contain one distribution version")
        return {
            "package": "qwen_mm",
            "version": versions[0],
            "package_artifact_sha256": _sha256_bytes(archive.read(package_members[0])),
            "native_module": "qwen_mm._native",
            "native_artifact_sha256": _sha256_bytes(archive.read(native_members[0])),
        }


def _assert_wheel_runtime_binding(
    wheel_path: Path, runtime_identity: Mapping[str, Any]
) -> dict[str, str]:
    wheel_identity = _wheel_runtime_identity(wheel_path)
    if dict(runtime_identity) != wheel_identity:
        raise ValueError(
            "candidate runtime identity does not match supplied wheel artifacts: "
            f"candidate={dict(runtime_identity)}, wheel={wheel_identity}"
        )
    return wheel_identity


def _descriptor(array: np.ndarray[Any, Any], *, path: str | None = None) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    value: dict[str, Any] = {
        "dtype": str(array.dtype),
        "byte_order": _byte_order(array.dtype),
        "shape": list(array.shape),
        "strides": list(array.strides),
        "c_contiguous": bool(array.flags.c_contiguous),
        "nbytes": int(array.nbytes),
        "sha256": _sha256_bytes(memoryview(contiguous).cast("B")),
    }
    if path is not None:
        value["path"] = path
    return value


def _byte_order(dtype: np.dtype[Any]) -> str:
    if dtype.itemsize == 1 or dtype.byteorder == "|":
        return "not-applicable"
    if dtype.byteorder == "<" or (dtype.byteorder == "=" and sys.byteorder == "little"):
        return "little"
    return "big"


def _ordered_float_bits(array: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    bits = np.asarray(array, dtype=np.float32).view(np.int32).astype(np.int64)
    return np.where(bits < 0, np.int64(0x80000000) - bits, bits)


def numeric_diagnostics(
    expected: np.ndarray[Any, Any],
    actual: np.ndarray[Any, Any],
    atol: float,
    *,
    pixel_metadata: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    expected64 = np.asarray(expected, dtype=np.float64)
    actual64 = np.asarray(actual, dtype=np.float64)
    absolute = np.abs(expected64 - actual64)
    beyond = absolute > atol
    differing = np.argwhere(beyond)
    first = tuple(int(item) for item in differing[0]) if len(differing) else None
    ulp = np.abs(_ordered_float_bits(expected) - _ordered_float_bits(actual))
    first_difference: dict[str, Any] | None = None
    if first is not None:
        first_difference = {
            "index": list(first),
            "expected": float(expected[first]),
            "actual": float(actual[first]),
        }
        if pixel_metadata is not None and len(first) == 2:
            row, column = first
            for occurrence, metadata in enumerate(pixel_metadata):
                start, end = metadata["pixel_rows"]
                if start <= row < end:
                    local_row = row - start
                    grid = metadata["geometry"]["image_grid_thw"]
                    outer_width = int(grid[2]) // 2
                    outer_index, merge_index = divmod(local_row, 4)
                    outer_y, outer_x = divmod(outer_index, outer_width)
                    merge_y, merge_x = divmod(merge_index, 2)
                    channel, remainder = divmod(column, 2 * 16 * 16)
                    temporal, remainder = divmod(remainder, 16 * 16)
                    patch_y, patch_x = divmod(remainder, 16)
                    first_difference["semantic_coordinate"] = {
                        "request": metadata["request_index"],
                        "media_occurrence": occurrence,
                        "patch": local_row,
                        "outer_y": outer_y,
                        "outer_x": outer_x,
                        "merge_y": merge_y,
                        "merge_x": merge_x,
                        "temporal": temporal,
                        "channel": channel,
                        "patch_y": patch_y,
                        "patch_x": patch_x,
                    }
                    break
    return {
        "maximum_absolute_error": float(absolute.max(initial=0.0)),
        "maximum_ulp_error": int(ulp.max(initial=0)),
        "rmse": float(np.sqrt(np.mean(np.square(absolute)))) if absolute.size else 0.0,
        "absolute_error_percentiles": {
            name: float(np.quantile(absolute, quantile)) if absolute.size else 0.0
            for name, quantile in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99))
        },
        "count_over_threshold": int(np.count_nonzero(beyond)),
        "first_difference": first_difference,
    }


def unpatchify_image(pixel_values: np.ndarray[Any, Any], height: int, width: int) -> np.ndarray:
    """Invert the frozen Qwen image patch layout into prepared HWC RGB8."""
    patch = 16
    merge = 2
    grid_height = height // patch
    grid_width = width // patch
    expected_rows = grid_height * grid_width
    if pixel_values.shape != (expected_rows, 1536):
        raise ValueError(
            f"pixel_values shape {pixel_values.shape} does not match {(expected_rows, 1536)}"
        )
    output = np.empty((height, width, 3), dtype=np.uint8)
    row = 0
    for outer_y in range(grid_height // merge):
        for outer_x in range(grid_width // merge):
            for merge_y in range(merge):
                for merge_x in range(merge):
                    patch_values = pixel_values[row].reshape(3, 2, patch, patch)
                    if not np.array_equal(patch_values[:, 0], patch_values[:, 1]):
                        raise ValueError(f"image temporal planes differ at patch row {row}")
                    prepared = np.rint(patch_values[:, 0] * np.float32(127.5) + np.float32(127.5))
                    prepared = np.clip(prepared, 0, 255).astype(np.uint8).transpose(1, 2, 0)
                    y = (outer_y * merge + merge_y) * patch
                    x = (outer_x * merge + merge_x) * patch
                    output[y : y + patch, x : x + patch] = prepared
                    row += 1
    return output


def patchify_rgb(prepared: np.ndarray[Any, Any]) -> np.ndarray:
    """Frozen NumPy oracle for normalize/patchify from authenticated prepared RGB."""
    height, width, channels = prepared.shape
    if channels != 3 or height % 32 or width % 32:
        raise ValueError("prepared RGB must be HWC3 with dimensions divisible by 32")
    rows: list[np.ndarray[Any, Any]] = []
    for outer_y in range(height // 32):
        for outer_x in range(width // 32):
            for merge_y in range(2):
                for merge_x in range(2):
                    y = (outer_y * 2 + merge_y) * 16
                    x = (outer_x * 2 + merge_x) * 16
                    patch = prepared[y : y + 16, x : x + 16].transpose(2, 0, 1)
                    normalized = (patch.astype(np.float32) - np.float32(127.5)) / np.float32(127.5)
                    rows.append(np.repeat(normalized[:, None], 2, axis=1).reshape(-1))
    return np.stack(rows).astype(np.float32, copy=False)


def _candidate_message(message: Mapping[str, Any]) -> dict[str, Any]:
    content = message["content"]
    if isinstance(content, list):
        normalized = []
        for item in content:
            if item["type"] == "text":
                normalized.append({"type": "text", "text": item["text"]})
                continue
            normalized_item = {
                name: copy.deepcopy(item[name])
                for name in (
                    "type",
                    "input_index",
                    "buffer_index",
                    "image",
                    "video",
                    "options",
                    "min_pixels",
                    "max_pixels",
                    "resized_height",
                    "resized_width",
                )
                if name in item
            }
            normalized.append(normalized_item)
        content = normalized
    value = {"role": message["role"], "content": copy.deepcopy(content)}
    if "reasoning_content" in message:
        value["reasoning_content"] = message["reasoning_content"]
    if calls := message.get("tool_calls"):
        converted = []
        for call in calls:
            function = {"name": call["name"], "arguments": call["arguments_json"]}
            converted.append({"function": function} if call["shape"] == "function" else function)
        value["tool_calls"] = converted
    return value


def _candidate_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    source = dict(options or {})
    value = {
        name: source[name]
        for name in ("add_generation_prompt", "add_vision_id", "enable_thinking")
        if name in source
    }
    if "tools_json" in source:
        value["tools"] = source["tools_json"]
    for name in source.get("excluded", []):
        value[name] = "arbitrary" if name == "chat_template" else True
    return value


def _load_candidate_image(spec: Mapping[str, Any]) -> Any:
    path = Path(spec["path"])
    data = path.read_bytes()
    offset = int(spec.get("offset", 0))
    length = int(spec.get("length", len(data) - offset))
    payload = data[offset : offset + length]
    if _sha256_bytes(payload) != spec["sha256"]:
        raise ValueError(f"candidate input hash mismatch: {path}")
    kind = spec["kind"]
    if kind == "encoded":
        return {"data": payload, "format": spec["format"]}
    height, width, row_stride = (int(spec[name]) for name in ("height", "width", "row_stride"))
    owner = bytearray(payload)
    base = np.ndarray((height, width, 3), dtype=np.uint8, buffer=owner, strides=(row_stride, 3, 1))
    layout = spec.get("layout", "positive")
    if layout == "negative-row":
        return base[::-1]
    if layout == "channel-strided":
        expanded = np.zeros((height, width, 6), dtype=np.uint8)
        expanded[:, :, ::2] = base
        return expanded[:, :, ::2]
    return base


def _create_candidate_processor(module: Any, case: Mapping[str, Any]) -> Any:
    return module.Processor(case["profile"], case["assets_directory"], limits=case.get("limits"))


def candidate_worker(case_path: Path, output_path: Path) -> int:
    accessed: list[str] = []

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event != "open" or not args:
            return
        path = args[0]
        if isinstance(path, (str, bytes, os.PathLike)):
            text = os.fsdecode(path)
            accessed.append(text)
            lowered = text.lower()
            if "/expected/" in lowered or lowered.endswith("/manifest.json"):
                raise PermissionError(f"candidate attempted oracle access: {text}")

    sys.addaudithook(audit)
    case = _json(case_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import importlib.metadata

    import qwen_mm
    import qwen_mm._native as native

    requests = []
    owners = []
    for request in case.get("requests", []):
        if "_raw_request" in request:
            requests.append(copy.deepcopy(request["_raw_request"]))
            continue
        images = [_load_candidate_image(spec) for spec in request.get("images", [])]
        owners.extend(images)
        requests.append(
            {
                "messages": [_candidate_message(message) for message in request["messages"]],
                "images": images,
                "options": _candidate_options(request.get("options")),
            }
        )
    result: dict[str, Any]
    try:
        processor = _create_candidate_processor(qwen_mm, case)
        prepared = processor.prepare_batch(requests)
        arrays: dict[str, Any] = {}
        arrays_directory = output_path.parent / "arrays"
        arrays_directory.mkdir(exist_ok=True)
        for name, array in prepared.arrays.items():
            array_path = arrays_directory / f"{name}.npy"
            np.save(array_path, array, allow_pickle=False)
            arrays[name] = _descriptor(array, path=str(array_path.relative_to(output_path.parent)))
        result = {
            "status": "success",
            "official_keys": prepared.official_keys(),
            "arrays": arrays,
            "metadata": prepared.metadata,
        }
    except qwen_mm.QwenMMError as error:
        result = {
            "status": "expected_error",
            "error": {
                "category": error.category,
                "type": type(error).__name__,
                "context": error.context,
            },
        }
    forbidden = sorted(
        name
        for name in FORBIDDEN_CANDIDATE_MODULES
        if any(module == name or module.startswith(name + ".") for module in sys.modules)
    )
    prefix = Path(sys.prefix).resolve()
    package_path = Path(qwen_mm.__file__).resolve()
    native_path = Path(native.__file__).resolve()
    result["isolation"] = {
        "forbidden_imports": forbidden,
        "package_path": str(package_path),
        "native_path": str(native_path),
        "sys_prefix": str(prefix),
        "installed_origin": package_path.is_relative_to(prefix)
        and native_path.is_relative_to(prefix),
        "oracle_accesses": sorted(
            path
            for path in accessed
            if "/expected/" in path.lower() or path.endswith("manifest.json")
        ),
    }
    try:
        distribution_version = importlib.metadata.version("qwen-mm")
    except importlib.metadata.PackageNotFoundError:
        distribution_version = qwen_mm.__version__
    result["runtime_identity"] = {
        "package": "qwen_mm",
        "version": distribution_version,
        "package_artifact_sha256": _sha256_file(package_path),
        "native_module": "qwen_mm._native",
        "native_artifact_sha256": _sha256_file(native_path),
    }
    result["gate_invocation_identity"] = {
        "entrypoint": "qwen_mm:Processor",
        "python_isolated_mode": True,
    }
    result["candidate_environment"] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "abi_tag": sys.implementation.cache_tag,
        "platform": platform.platform(),
        "package_origin": str(package_path.relative_to(prefix)),
        "native_origin": str(native_path.relative_to(prefix)),
        "installed_origin": True,
    }
    _write_json(output_path, result)
    return 0


def _run_candidate(
    candidate_python: Path, case: Mapping[str, Any], directory: Path
) -> tuple[dict[str, Any], Path]:
    case_path = directory / "case.json"
    output_path = directory / "actual" / "result.json"
    _write_json(case_path, case)
    completed = subprocess.run(
        [
            str(candidate_python),
            "-B",
            "-I",
            str(Path(__file__).resolve()),
            "_candidate",
            "--case",
            str(case_path),
            "--output",
            str(output_path),
        ],
        cwd=repository_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"candidate process failed: {completed.returncode}\n{completed.stdout}\n{completed.stderr}"
        )
    result = _json(output_path)
    isolation = result.get("isolation", {})
    if isolation.get("forbidden_imports") or isolation.get("oracle_accesses"):
        raise RuntimeError(f"candidate isolation violation: {isolation}")
    if isolation.get("installed_origin") is not True:
        raise RuntimeError(f"candidate did not import the installed wheel: {isolation}")
    global _OBSERVED_CANDIDATE_PROVENANCE
    provenance = {
        "runtime_identity": result["runtime_identity"],
        "environment": result["candidate_environment"],
    }
    if _OBSERVED_CANDIDATE_PROVENANCE is None:
        _OBSERVED_CANDIDATE_PROVENANCE = provenance
    elif _OBSERVED_CANDIDATE_PROVENANCE != provenance:
        raise RuntimeError("candidate runtime identity changed between gate subprocesses")
    return result, output_path.parent


def _load_actual_array(result: Mapping[str, Any], root: Path, name: str) -> np.ndarray:
    descriptor = result["arrays"][name]
    return np.load(root / descriptor["path"], allow_pickle=False)


def _result(case_id: str, tier: str, profile: str, issues: Sequence[Mapping[str, Any]]) -> dict:
    return {
        "case_id": case_id,
        "tier": tier,
        "profile": profile,
        "passed": not issues,
        "issues": list(issues),
    }


def _compare_exact_array(
    issues: list[dict[str, Any]], name: str, expected: np.ndarray, actual: np.ndarray
) -> None:
    for field, expected_value, actual_value in (
        ("dtype", str(expected.dtype), str(actual.dtype)),
        ("byte_order", _byte_order(expected.dtype), _byte_order(actual.dtype)),
        ("shape", list(expected.shape), list(actual.shape)),
        ("strides", list(expected.strides), list(actual.strides)),
        ("c_contiguous", True, bool(actual.flags.c_contiguous)),
    ):
        if expected_value != actual_value:
            issues.append(
                {
                    "stage": "outputs",
                    "path": f"{name}.{field}",
                    "expected": expected_value,
                    "actual": actual_value,
                }
            )
    if expected.shape == actual.shape and not np.array_equal(expected, actual):
        difference = np.argwhere(expected != actual)[0]
        index = tuple(int(item) for item in difference)
        issues.append(
            {
                "stage": "outputs",
                "path": name,
                "kind": "exact_array_mismatch",
                "first_difference": {
                    "index": list(index),
                    "expected": expected[index].item(),
                    "actual": actual[index].item(),
                },
            }
        )


def _compare_float_array(
    issues: list[dict[str, Any]],
    name: str,
    expected: np.ndarray,
    actual: np.ndarray,
    atol: float,
    *,
    pixel_metadata: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    if (
        str(expected.dtype) != str(actual.dtype)
        or expected.shape != actual.shape
        or expected.strides != actual.strides
    ):
        _compare_exact_array(issues, name, expected, actual)
        return
    diagnostics = numeric_diagnostics(expected, actual, atol, pixel_metadata=pixel_metadata)
    if diagnostics["count_over_threshold"]:
        issues.append(
            {"stage": "outputs", "path": name, "kind": "numeric_array_mismatch", **diagnostics}
        )


def _blob_array(descriptor: Mapping[str, Any], blob: bytes) -> np.ndarray:
    dtype = np.dtype(descriptor["dtype"])
    start = int(descriptor["offset"])
    length = int(descriptor["length"])
    value = np.frombuffer(blob[start : start + length], dtype=dtype).reshape(descriptor["shape"])
    if _sha256_bytes(memoryview(np.ascontiguousarray(value)).cast("B")) != descriptor["sha256"]:
        raise ValueError("authenticated binary descriptor mismatch")
    return value


def _golden_array(descriptor: Mapping[str, Any], root: Path) -> np.ndarray:
    if descriptor.get("storage") == "inline_json":
        return np.asarray(descriptor["data"], dtype=np.dtype(descriptor["dtype"])).reshape(
            descriptor["shape"]
        )
    if descriptor.get("storage") == "npy":
        return np.load(root / descriptor["path"], allow_pickle=False)
    raise ValueError("golden array is signature-only")


def _assets_directory(assets_root: Path, profile: str) -> Path:
    models = _json(repository_root() / "reference/models.json")
    record = models[profile]
    cache_name = "models--" + record["model_id"].replace("/", "--")
    return assets_root / cache_name / "snapshots" / record["revision"]


def _image_spec_from_phase_b(source: Mapping[str, Any], blob_path: Path) -> dict[str, Any]:
    kind = source["kind"]
    common = {
        "path": str(blob_path),
        "offset": source["offset"],
        "length": source["length"],
        "sha256": source["sha256"],
    }
    if kind == "encoded":
        return {**common, "kind": "encoded", "format": source["format"]}
    return {
        **common,
        "kind": "raw",
        "height": source["height"],
        "width": source["width"],
        "row_stride": source["row_stride"],
        "layout": "positive",
    }


def run_phase_b(
    candidate_python: Path, assets_root: Path, output: Path
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    root = repository_root()
    manifest = _json(root / "reference/phase-b/v1/manifest.json")
    sources_blob_path = root / "reference/phase-b/v1/sources.bin"
    expected_blob = (root / "reference/phase-b/v1/expected.bin").read_bytes()
    results: list[dict[str, Any]] = []
    declared: list[str] = []
    artifact_identity: dict[str, Any] | None = None
    for case in manifest["cases"]:
        for profile in PROFILES:
            case_id = f"phase-b:{profile}:{case['id']}"
            declared.append(case_id)
            request = {
                "messages": copy.deepcopy(case["messages"]),
                "options": copy.deepcopy(case["options"]),
                "images": [
                    _image_spec_from_phase_b(manifest["sources"][name], sources_blob_path)
                    for name in case["inputs"]
                ],
            }
            actual, actual_root = _run_candidate(
                candidate_python,
                {
                    "profile": profile,
                    "assets_directory": str(_assets_directory(assets_root, profile)),
                    "requests": [request],
                },
                output / "phase-b" / profile / case["id"],
            )
            artifact_identity = artifact_identity or {
                "runtime_identity": actual["runtime_identity"],
                "environment": actual["candidate_environment"],
                "isolation": actual["isolation"],
            }
            issues: list[dict[str, Any]] = []
            expected = case["expectations"][profile]
            if actual.get("status") != "success":
                issues.append({"stage": "envelope", "expected": "success", "actual": actual})
            else:
                if actual["official_keys"] != expected["official_keys"]:
                    issues.append(
                        {
                            "stage": "outputs",
                            "path": "official_keys",
                            "expected": expected["official_keys"],
                            "actual": actual["official_keys"],
                        }
                    )
                text = actual["metadata"]["text"][0]
                for name in ("rendered_prompt", "expanded_prompt"):
                    if text[name] != expected[name]["text"]:
                        issues.append({"stage": "text", "path": name, "kind": "exact_mismatch"})
                actual_replacements = [
                    {
                        "type": item["modality"],
                        "original_codepoint_span": item["rendered_code_points"],
                        "expanded_codepoint_span": item["expanded_code_points"],
                        "expanded_token_span": item["expanded_tokens"],
                    }
                    for item in text["replacements"]
                ]
                expected_replacements = [
                    {
                        "type": item["type"],
                        "original_codepoint_span": item["original_codepoint_span"],
                        "expanded_codepoint_span": item["expanded_codepoint_span"],
                        "expanded_token_span": item["expanded_token_span"],
                    }
                    for item in expected["replacements"]
                ]
                if actual_replacements != expected_replacements:
                    issues.append(
                        {
                            "stage": "replacement_ranges",
                            "expected": expected_replacements,
                            "actual": actual_replacements,
                        }
                    )
                expected_occurrences = []
                for message_index, message in enumerate(case["messages"]):
                    if not isinstance(message["content"], list):
                        continue
                    for content_item_index, item in enumerate(message["content"]):
                        if item["type"] == "image":
                            expected_occurrences.append(
                                {
                                    "request_index": 0,
                                    "message_index": message_index,
                                    "content_item_index": content_item_index,
                                    "input_index": item["input_index"],
                                }
                            )
                actual_occurrences = (
                    [
                        {name: item[name] for name in expected_occurrences[0]}
                        for item in actual["metadata"]["images"]
                    ]
                    if expected_occurrences
                    else []
                )
                if actual_occurrences != expected_occurrences:
                    issues.append(
                        {
                            "stage": "occurrence_order",
                            "expected": expected_occurrences,
                            "actual": actual_occurrences,
                        }
                    )
                for name, descriptor in expected["arrays"].items():
                    expected_array = _blob_array(descriptor, expected_blob)
                    actual_array = _load_actual_array(actual, actual_root, name)
                    if name == "pixel_values":
                        _compare_float_array(
                            issues,
                            name,
                            expected_array,
                            actual_array,
                            1.0e-6,
                            pixel_metadata=actual["metadata"]["images"],
                        )
                    else:
                        _compare_exact_array(issues, name, expected_array, actual_array)
                if expected["prepared_images"]:
                    pixels = _load_actual_array(actual, actual_root, "pixel_values")
                    for index, descriptor in enumerate(expected["prepared_images"]):
                        metadata = actual["metadata"]["images"][index]
                        start, end = metadata["pixel_rows"]
                        prepared = unpatchify_image(
                            pixels[start:end],
                            metadata["geometry"]["height"],
                            metadata["geometry"]["width"],
                        )
                        expected_prepared = _blob_array(descriptor, expected_blob)
                        if not np.array_equal(prepared, expected_prepared):
                            issues.append(
                                {
                                    "stage": "prepared_media",
                                    "path": f"images[{index}]",
                                    **numeric_diagnostics(expected_prepared, prepared, 0.0),
                                }
                            )
            results.append(_result(case_id, "committed-phase-b", profile, issues))
    if artifact_identity is None:
        raise RuntimeError("Phase B produced no candidate identity")
    return results, declared, artifact_identity


def run_chat(
    candidate_python: Path, assets_root: Path, output: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    root = repository_root()
    document = _json(root / "reference/conformance/v1/chat.json")
    excluded = {"qwen3-vl-8b-visual-padding", "qwen3.5-9b-visual-padding"}
    results: list[dict[str, Any]] = []
    declared: list[str] = []
    for case in document["cases"]:
        if case["id"] in excluded:
            continue
        case_id = f"chat:{case['id']}"
        declared.append(case_id)
        profile = case["profile"]
        assets_profile = profile if profile in PROFILES else PROFILES[0]
        actual, actual_root = _run_candidate(
            candidate_python,
            {
                "profile": profile,
                "assets_directory": str(_assets_directory(assets_root, assets_profile)),
                "requests": [
                    {
                        "messages": request["messages"],
                        "options": request.get("options", {}),
                        "images": [],
                    }
                    for request in case["requests"]
                ],
            },
            output / "chat" / case["id"],
        )
        issues: list[dict[str, Any]] = []
        expected_error = case.get("expected_error")
        expected_status = "expected_error" if expected_error else "success"
        if actual.get("status") != expected_status:
            issues.append(
                {"stage": "envelope", "expected": expected_status, "actual": actual.get("status")}
            )
        elif expected_error:
            if actual["error"]["category"] != expected_error["category"]:
                issues.append(
                    {
                        "stage": "error",
                        "path": "category",
                        "expected": expected_error["category"],
                        "actual": actual["error"]["category"],
                    }
                )
        else:
            expected = case["expected"]
            if actual["official_keys"] != ["input_ids", "attention_mask", "mm_token_type_ids"]:
                issues.append({"stage": "outputs", "path": "official_keys", "kind": "mismatch"})
            rendered = [item["rendered_prompt"] for item in actual["metadata"]["text"]]
            expanded = [item["expanded_prompt"] for item in actual["metadata"]["text"]]
            if rendered != expected["rendered_prompts"]:
                issues.append(
                    {"stage": "text", "path": "rendered_prompts", "kind": "exact_mismatch"}
                )
            if expanded != expected["expanded_prompts"]:
                issues.append(
                    {"stage": "text", "path": "expanded_prompts", "kind": "exact_mismatch"}
                )
            for name in ("input_ids", "attention_mask", "mm_token_type_ids"):
                expected_array = np.asarray(expected[name], dtype=np.int64)
                actual_array = _load_actual_array(actual, actual_root, name)
                _compare_exact_array(issues, name, expected_array, actual_array)
        results.append(_result(case_id, "live-chat", profile, issues))
    return results, declared


def _media_input_spec(case: Mapping[str, Any], encoded_path: Path) -> dict[str, Any]:
    descriptor = case["encoded"]
    return {
        "kind": "encoded",
        "path": str(encoded_path),
        "offset": descriptor["offset"],
        "length": descriptor["byte_length"],
        "sha256": descriptor["sha256"],
        "format": case["format"],
    }


def run_media(
    candidate_python: Path, assets_root: Path, output: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    from qwen_mm_reference.golden import export_case

    root = repository_root()
    manifest = _json(root / "reference/media/v1/manifest.json")
    encoded_path = root / "reference/media/v1/encoded.bin"
    encoded_blob = encoded_path.read_bytes()
    prepared_blob = (root / "reference/media/v1/prepared-rgb8.bin").read_bytes()
    successes = [case for case in manifest["cases"] if "expected_error" not in case]
    failures = [case for case in manifest["cases"] if "expected_error" in case]
    results: list[dict[str, Any]] = []
    declared: list[str] = []
    cache = root / "reference/.cache"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phase-c-media-", dir=cache) as temporary:
        inputs = Path(temporary) / "inputs"
        inputs.mkdir()
        oracle_requests = []
        candidate_requests = []
        for case in successes:
            encoded = case["encoded"]
            payload = encoded_blob[encoded["offset"] : encoded["offset"] + encoded["byte_length"]]
            suffix = {"jpeg": ".jpg", "png": ".png", "webp": ".webp"}[case["format"]]
            image_path = inputs / f"{case['id']}{suffix}"
            image_path.write_bytes(payload)
            relative = image_path.relative_to(root).as_posix()
            oracle_requests.append(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": {"path": relative}},
                                {"type": "text", "text": case["id"]},
                            ],
                        }
                    ],
                    "options": {"add_generation_prompt": True},
                }
            )
            candidate_requests.append(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "input_index": 0},
                                {"type": "text", "text": case["id"]},
                            ],
                        }
                    ],
                    "options": {"add_generation_prompt": True},
                    "images": [_media_input_spec(case, encoded_path)],
                }
            )
        for profile in PROFILES:
            expected_directory = Path(temporary) / "expected" / profile
            expected = export_case(
                {
                    "schema_version": 1,
                    "case_id": f"phase-c-media-success-{profile}",
                    "requests": oracle_requests,
                },
                profile_alias=profile,
                output_directory=expected_directory,
                inline_max_bytes=1_048_576,
                write_arrays=True,
            )
            actual, actual_root = _run_candidate(
                candidate_python,
                {
                    "profile": profile,
                    "assets_directory": str(_assets_directory(assets_root, profile)),
                    "requests": candidate_requests,
                },
                output / "media" / profile / "success-matrix",
            )
            aggregate_issues: list[dict[str, Any]] = []
            if actual.get("status") != "success":
                aggregate_issues.append(
                    {"stage": "envelope", "expected": "success", "actual": actual}
                )
            else:
                if actual["official_keys"] != expected["output"]["keys"]:
                    aggregate_issues.append(
                        {
                            "stage": "outputs",
                            "path": "official_keys",
                            "expected": expected["output"]["keys"],
                            "actual": actual["official_keys"],
                        }
                    )
                rendered = [item["rendered_prompt"] for item in actual["metadata"]["text"]]
                expanded = [item["expanded_prompt"] for item in actual["metadata"]["text"]]
                expected_rendered = [
                    item["text"] for item in expected["stages"]["rendered_prompts"]
                ]
                expected_expanded = [
                    item["text"] for item in expected["stages"]["expanded_prompts"]
                ]
                if rendered != expected_rendered:
                    aggregate_issues.append(
                        {"stage": "text", "path": "rendered_prompts", "kind": "exact_mismatch"}
                    )
                if expanded != expected_expanded:
                    aggregate_issues.append(
                        {"stage": "text", "path": "expanded_prompts", "kind": "exact_mismatch"}
                    )
                for name in ("input_ids", "attention_mask", "mm_token_type_ids", "image_grid_thw"):
                    expected_array = _golden_array(
                        expected["output"]["arrays"][name], expected_directory
                    )
                    actual_array = _load_actual_array(actual, actual_root, name)
                    _compare_exact_array(aggregate_issues, name, expected_array, actual_array)
                expected_pixels = _golden_array(
                    expected["output"]["arrays"]["pixel_values"], expected_directory
                )
                actual_pixels = _load_actual_array(actual, actual_root, "pixel_values")
                for index, media_case in enumerate(successes):
                    case_id = f"media:{profile}:{media_case['id']}"
                    declared.append(case_id)
                    issues = list(aggregate_issues)
                    metadata = actual["metadata"]["images"][index]
                    start, end = metadata["pixel_rows"]
                    prepared = unpatchify_image(
                        actual_pixels[start:end],
                        metadata["geometry"]["height"],
                        metadata["geometry"]["width"],
                    )
                    descriptor = media_case["prepared_rgb8"]
                    expected_prepared = np.frombuffer(
                        prepared_blob[
                            descriptor["offset"] : descriptor["offset"] + descriptor["byte_length"]
                        ],
                        dtype=np.uint8,
                    ).reshape(descriptor["height"], descriptor["width"], 3)
                    prepared_diagnostics = numeric_diagnostics(
                        expected_prepared, prepared, float(descriptor["absolute_byte_error_max"])
                    )
                    if prepared_diagnostics["count_over_threshold"]:
                        issues.append(
                            {
                                "stage": "prepared_media",
                                "path": media_case["id"],
                                **prepared_diagnostics,
                            }
                        )
                    tolerance = 0.007844090461730957 if "lossy" in media_case["tags"] else 1.0e-6
                    diagnostic_metadata = copy.deepcopy(metadata)
                    diagnostic_metadata["pixel_rows"] = [0, end - start]
                    final_diagnostics = numeric_diagnostics(
                        expected_pixels[start:end],
                        actual_pixels[start:end],
                        tolerance,
                        pixel_metadata=[diagnostic_metadata],
                    )
                    if final_diagnostics["count_over_threshold"]:
                        issues.append(
                            {
                                "stage": "outputs",
                                "path": f"pixel_values[{start}:{end}]",
                                **final_diagnostics,
                            }
                        )
                    results.append(_result(case_id, "live-media", profile, issues))
            if aggregate_issues and actual.get("status") != "success":
                for media_case in successes:
                    case_id = f"media:{profile}:{media_case['id']}"
                    declared.append(case_id)
                    results.append(_result(case_id, "live-media", profile, aggregate_issues))

            for media_case in failures:
                case_id = f"media:{profile}:{media_case['id']}"
                declared.append(case_id)
                request = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "input_index": 0},
                                {"type": "text", "text": media_case["id"]},
                            ],
                        }
                    ],
                    "options": {},
                    "images": [_media_input_spec(media_case, encoded_path)],
                }
                observed, _ = _run_candidate(
                    candidate_python,
                    {
                        "profile": profile,
                        "assets_directory": str(_assets_directory(assets_root, profile)),
                        "requests": [request],
                    },
                    output / "media" / profile / media_case["id"],
                )
                issues = []
                expected_category = media_case["expected_error"]["category"]
                if (
                    observed.get("status") != "expected_error"
                    or observed.get("error", {}).get("category") != expected_category
                ):
                    issues.append(
                        {
                            "stage": "error",
                            "expected": expected_category,
                            "actual": observed,
                        }
                    )
                results.append(_result(case_id, "live-media", profile, issues))
    return results, declared


def _candidate_requests_from_golden_case(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = repository_root()
    requests = []
    for request in case["requests"]:
        images = []
        messages = []
        for message in request["messages"]:
            content = message["content"]
            if isinstance(content, list):
                converted = []
                for item in content:
                    if item["type"] == "text":
                        converted.append({"type": "text", "text": item["text"]})
                        continue
                    if item["type"] != "image":
                        raise ValueError("Phase C golden conversion received video")
                    source = item["image"]
                    image_path = root / source["path"]
                    data = image_path.read_bytes()
                    format_name = image_path.suffix.lower().lstrip(".")
                    if format_name not in {"jpeg", "jpg", "png", "webp"}:
                        format_name = "jpeg"
                    if format_name == "jpg":
                        format_name = "jpeg"
                    input_index = len(images)
                    images.append(
                        {
                            "kind": "encoded",
                            "path": str(image_path),
                            "offset": 0,
                            "length": len(data),
                            "sha256": _sha256_bytes(data),
                            "format": format_name,
                        }
                    )
                    converted_item = {"type": "image", "input_index": input_index}
                    for name in ("min_pixels", "max_pixels", "resized_height", "resized_width"):
                        if name in item:
                            converted_item[name] = item[name]
                    converted.append(converted_item)
                content = converted
            messages.append({**message, "content": content})
        requests.append(
            {
                "messages": messages,
                "options": request.get("options", {}),
                "images": images,
            }
        )
    return requests


def _golden_expected_sidecar(
    replacements_by_request: Sequence[Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Translate golden replacement metadata into the public sidecar contract."""
    sidecar: list[dict[str, Any]] = []
    grid_row = 0
    for request_index, replacements in enumerate(replacements_by_request):
        for replacement in replacements:
            if replacement["type"] != "image":
                continue
            sidecar.append(
                {
                    "request_index": request_index,
                    "grid_row": grid_row,
                    "replacement": {
                        "code_points": replacement["original_codepoint_span"],
                        "tokens": replacement["expanded_token_span"],
                    },
                }
            )
            grid_row += 1
    return sidecar


def _compare_golden_result(
    expected: Mapping[str, Any], expected_root: Path, actual: Mapping[str, Any], actual_root: Path
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if actual.get("status") != expected["status"]:
        return [{"stage": "envelope", "expected": expected["status"], "actual": actual}]
    if expected["status"] == "expected_error":
        if actual.get("error", {}).get("category") != expected["error"]["category"]:
            issues.append(
                {
                    "stage": "error",
                    "expected": expected["error"]["category"],
                    "actual": actual.get("error", {}).get("category"),
                }
            )
        return issues
    if actual["official_keys"] != expected["output"]["keys"]:
        issues.append(
            {
                "stage": "outputs",
                "path": "official_keys",
                "expected": expected["output"]["keys"],
                "actual": actual["official_keys"],
            }
        )
    rendered = [item["rendered_prompt"] for item in actual["metadata"]["text"]]
    expanded = [item["expanded_prompt"] for item in actual["metadata"]["text"]]
    if rendered != [item["text"] for item in expected["stages"]["rendered_prompts"]]:
        issues.append({"stage": "text", "path": "rendered_prompts", "kind": "exact_mismatch"})
    if expanded != [item["text"] for item in expected["stages"]["expanded_prompts"]]:
        issues.append({"stage": "text", "path": "expanded_prompts", "kind": "exact_mismatch"})
    expected_replacements = expected["stages"]["replacement_offsets"]
    actual_replacements = [
        [
            {
                "type": replacement["modality"],
                "original_codepoint_span": replacement["rendered_code_points"],
                "expanded_codepoint_span": replacement["expanded_code_points"],
                "expanded_token_span": replacement["expanded_tokens"],
            }
            for replacement in text["replacements"]
        ]
        for text in actual["metadata"]["text"]
    ]
    simplified_expected_replacements = [
        [
            {
                "type": replacement["type"],
                "original_codepoint_span": replacement["original_codepoint_span"],
                "expanded_codepoint_span": replacement["expanded_codepoint_span"],
                "expanded_token_span": replacement["expanded_token_span"],
            }
            for replacement in request
        ]
        for request in expected_replacements
    ]
    if actual_replacements != simplified_expected_replacements:
        issues.append(
            {
                "stage": "replacement_ranges",
                "expected": simplified_expected_replacements,
                "actual": actual_replacements,
            }
        )
    for name in expected["output"]["keys"]:
        expected_descriptor = expected["output"]["arrays"][name]
        actual_descriptor = actual["arrays"][name]
        for field in ("dtype", "byte_order", "shape", "strides", "c_contiguous"):
            if expected_descriptor[field] != actual_descriptor[field]:
                issues.append(
                    {
                        "stage": "outputs",
                        "path": f"{name}.{field}",
                        "expected": expected_descriptor[field],
                        "actual": actual_descriptor[field],
                    }
                )
        try:
            expected_array = _golden_array(expected_descriptor, expected_root)
        except ValueError:
            if expected_descriptor["sha256"] != actual_descriptor["sha256"]:
                issues.append(
                    {
                        "stage": "outputs",
                        "path": name,
                        "kind": "signature_mismatch",
                        "expected_sha256": expected_descriptor["sha256"],
                        "actual_sha256": actual_descriptor["sha256"],
                    }
                )
        else:
            actual_array = _load_actual_array(actual, actual_root, name)
            if name == "pixel_values":
                _compare_float_array(
                    issues,
                    name,
                    expected_array,
                    actual_array,
                    0.007844090461730957,
                    pixel_metadata=actual["metadata"]["images"],
                )
            else:
                _compare_exact_array(issues, name, expected_array, actual_array)
    try:
        attention = _golden_array(expected["output"]["arrays"]["attention_mask"], expected_root)
        grids = (
            _golden_array(expected["output"]["arrays"]["image_grid_thw"], expected_root)
            if "image_grid_thw" in expected["output"]["arrays"]
            else np.empty((0, 3), dtype=np.int64)
        )
    except ValueError:
        pass
    else:
        columns = int(attention.shape[1])
        grid_cursor = 0
        pixel_cursor = 0
        expected_request_layouts = []
        expected_sidecar = _golden_expected_sidecar(simplified_expected_replacements)
        for request_index, replacements in enumerate(simplified_expected_replacements):
            token_count = int(attention[request_index].sum())
            request_grid_start = grid_cursor
            request_pixel_start = pixel_cursor
            for replacement in replacements:
                if replacement["type"] != "image":
                    continue
                patch_rows = int(np.prod(grids[grid_cursor], dtype=np.int64))
                grid_cursor += 1
                pixel_cursor += patch_rows
            expected_request_layouts.append(
                {
                    "request_index": request_index,
                    "text_elements": [request_index * columns, (request_index + 1) * columns],
                    "token_count": token_count,
                    "right_padding": columns - token_count,
                    "image_grid_rows": [request_grid_start, grid_cursor],
                    "pixel_rows": [request_pixel_start, pixel_cursor],
                }
            )
        if actual["metadata"]["request_layouts"] != expected_request_layouts:
            issues.append(
                {
                    "stage": "request_layouts",
                    "expected": expected_request_layouts,
                    "actual": actual["metadata"]["request_layouts"],
                }
            )
        if actual["metadata"]["sidecar"]["images"] != expected_sidecar:
            issues.append(
                {
                    "stage": "sidecar",
                    "expected": expected_sidecar,
                    "actual": actual["metadata"]["sidecar"]["images"],
                }
            )
        if actual["metadata"]["sidecar"]["videos"] != []:
            issues.append({"stage": "sidecar", "path": "videos", "kind": "unexpected"})
        media = [item for item in expected["input"]["media"] if item["kind"] == "image"]
        if len(actual["metadata"]["images"]) != len(media):
            issues.append(
                {
                    "stage": "occurrence_order",
                    "expected": len(media),
                    "actual": len(actual["metadata"]["images"]),
                }
            )
        else:
            for index, (observed, source) in enumerate(
                zip(actual["metadata"]["images"], media, strict=True)
            ):
                for key, expected_value in (
                    ("request_index", source["request_index"]),
                    ("message_index", source["message_index"]),
                    ("content_item_index", source["content_index"]),
                    ("grid_row", index),
                    ("source_height", source["source_properties"]["height"]),
                    ("source_width", source["source_properties"]["width"]),
                ):
                    if observed[key] != expected_value:
                        issues.append(
                            {
                                "stage": "occurrence_order",
                                "path": f"images[{index}].{key}",
                                "expected": expected_value,
                                "actual": observed[key],
                            }
                        )
                layout = actual["metadata"]["image_layouts"][index]
                for key in (
                    "request_index",
                    "message_index",
                    "content_item_index",
                    "input_index",
                    "grid_row",
                    "pixel_rows",
                    "geometry",
                    "cache_key",
                ):
                    if layout[key] != observed[key]:
                        issues.append(
                            {
                                "stage": "image_layouts",
                                "path": f"images[{index}].{key}",
                                "kind": "range_or_order_mismatch",
                            }
                        )
    if "pixel_values" in actual.get("arrays", {}):
        pixels = _load_actual_array(actual, actual_root, "pixel_values")
        expected_prepared = expected["stages"]["prepared_media"]["images"]
        for index, item in enumerate(actual["metadata"]["images"]):
            start, end = item["pixel_rows"]
            prepared = unpatchify_image(
                pixels[start:end], item["geometry"]["height"], item["geometry"]["width"]
            )
            descriptor = expected_prepared[index]["array"]
            try:
                expected_array = _golden_array(descriptor, expected_root)
            except ValueError:
                if _descriptor(prepared)["sha256"] != descriptor["sha256"]:
                    issues.append(
                        {
                            "stage": "prepared_media",
                            "path": f"images[{index}]",
                            "kind": "signature_mismatch",
                            "expected_sha256": descriptor["sha256"],
                            "actual_sha256": _descriptor(prepared)["sha256"],
                        }
                    )
            else:
                diagnostics = numeric_diagnostics(expected_array, prepared, 1.0)
                if diagnostics["count_over_threshold"]:
                    issues.append(
                        {
                            "stage": "prepared_media",
                            "path": f"images[{index}]",
                            **diagnostics,
                        }
                    )
    return issues


def run_committed_goldens(
    candidate_python: Path, assets_root: Path, output: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    from qwen_mm_reference.golden import export_case

    root = repository_root()
    results: list[dict[str, Any]] = []
    declared: list[str] = []
    cases = {
        name: _json(root / f"reference/cases/v1/{name}.json")
        for name in ("text-smoke", "corrupt-image", "image24")
    }
    for profile in PROFILES:
        for name in ("text-smoke", "corrupt-image", "image24"):
            case_id = f"golden:{profile}:{name}"
            declared.append(case_id)
            case = cases[name]
            committed = root / f"reference/goldens/v1/{profile}/{name}/manifest.json"
            expected_root = committed.parent
            if committed.is_file():
                expected = _json(committed)
            else:
                expected_root = output / "oracle" / profile / name
                expected = export_case(
                    case,
                    profile_alias=profile,
                    output_directory=expected_root,
                    inline_max_bytes=1_048_576,
                    write_arrays=False,
                )
            actual, actual_root = _run_candidate(
                candidate_python,
                {
                    "profile": profile,
                    "assets_directory": str(_assets_directory(assets_root, profile)),
                    "requests": _candidate_requests_from_golden_case(case),
                },
                output / "goldens" / profile / name,
            )
            issues = _compare_golden_result(expected, expected_root, actual, actual_root)
            results.append(_result(case_id, "committed-golden", profile, issues))
    return results, declared


def _raw_request(spec: Mapping[str, Any], text: str = "Describe the raw image.") -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "input_index": 0,
                        "options": {"resized_height": 64, "resized_width": 96},
                    },
                    {"type": "text", "text": text},
                ],
            }
        ],
        "options": {"add_generation_prompt": True},
        "images": [dict(spec)],
    }


def run_raw_layouts(
    candidate_python: Path, assets_root: Path, output: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    root = repository_root()
    phase_b = _json(root / "reference/phase-b/v1/manifest.json")
    source = phase_b["sources"]["raw-a"]
    blob_path = root / "reference/phase-b/v1/sources.bin"
    blob = blob_path.read_bytes()
    payload = blob[source["offset"] : source["offset"] + source["length"]]
    padded = _image_spec_from_phase_b(source, blob_path)
    input_directory = output / "raw-layout-inputs"
    input_directory.mkdir(parents=True, exist_ok=True)
    tight = b"".join(
        payload[row * source["row_stride"] : row * source["row_stride"] + source["width"] * 3]
        for row in range(source["height"])
    )
    tight_path = input_directory / "raw-a-tight.rgb8"
    tight_path.write_bytes(tight)
    contiguous = {
        "kind": "raw",
        "path": str(tight_path),
        "offset": 0,
        "length": len(tight),
        "sha256": _sha256_bytes(tight),
        "height": source["height"],
        "width": source["width"],
        "row_stride": source["width"] * 3,
        "layout": "positive",
    }
    layouts = {
        "contiguous-hwc": (contiguous, "success"),
        "row-padded-hwc": (padded, "success"),
        "negative-row-stride": ({**padded, "layout": "negative-row"}, "media_geometry"),
        "channel-sliced-invalid": ({**padded, "layout": "channel-strided"}, "media_geometry"),
    }
    results: list[dict[str, Any]] = []
    declared: list[str] = []
    for profile in PROFILES:
        success_arrays: dict[str, dict[str, np.ndarray]] = {}
        success_metadata: dict[str, Any] = {}
        for layout, (spec, expectation) in layouts.items():
            case_id = f"raw-layout:{profile}:{layout}"
            declared.append(case_id)
            actual, actual_root = _run_candidate(
                candidate_python,
                {
                    "profile": profile,
                    "assets_directory": str(_assets_directory(assets_root, profile)),
                    "requests": [_raw_request(spec)],
                },
                output / "raw-layout" / profile / layout,
            )
            issues: list[dict[str, Any]] = []
            if expectation == "success":
                if actual.get("status") != "success":
                    issues.append({"stage": "envelope", "expected": "success", "actual": actual})
                else:
                    success_arrays[layout] = {
                        name: _load_actual_array(actual, actual_root, name)
                        for name in actual["official_keys"]
                    }
                    success_metadata[layout] = actual["metadata"]
            elif (
                actual.get("status") != "expected_error"
                or actual.get("error", {}).get("category") != expectation
            ):
                issues.append({"stage": "error", "expected": expectation, "actual": actual})
            results.append(_result(case_id, "live-raw-layout", profile, issues))
        if set(success_arrays) == {"contiguous-hwc", "row-padded-hwc"}:
            for name in success_arrays["contiguous-hwc"]:
                if not np.array_equal(
                    success_arrays["contiguous-hwc"][name], success_arrays["row-padded-hwc"][name]
                ):
                    issue = {
                        "stage": "raw_layout",
                        "path": name,
                        "kind": "positive_stride_changed_output",
                    }
                    for result in results[-4:]:
                        if result["case_id"].endswith(("contiguous-hwc", "row-padded-hwc")):
                            result["issues"].append(issue)
                            result["passed"] = False
            for key in ("text", "images", "request_layouts", "image_layouts", "sidecar"):
                if (
                    success_metadata["contiguous-hwc"][key]
                    != success_metadata["row-padded-hwc"][key]
                ):
                    issue = {"stage": "raw_layout", "path": f"metadata.{key}", "kind": "mismatch"}
                    for result in results[-4:]:
                        if result["case_id"].endswith(("contiguous-hwc", "row-padded-hwc")):
                            result["issues"].append(issue)
                            result["passed"] = False
    return results, declared


def run_rules() -> tuple[list[dict[str, Any]], list[str]]:
    from qwen_mm_reference.corpus import evaluate_rule

    root = repository_root()
    document = _json(root / "reference/conformance/v1/rules.json")
    excluded = {
        value.removeprefix("rule:") for value in PHASE_E_EXCLUSIONS if value.startswith("rule:")
    }
    results: list[dict[str, Any]] = []
    declared: list[str] = []
    for rule in document["rules"]:
        if rule["id"] in excluded:
            continue
        case_id = f"rule:{rule['id']}"
        declared.append(case_id)
        issues: list[dict[str, Any]] = []
        try:
            observed = evaluate_rule(rule)
        except Exception as error:  # noqa: BLE001 - exception type is fixture evidence
            expected = rule.get("expected_error")
            if expected is None or type(error).__name__ != expected["exception"]:
                issues.append(
                    {
                        "stage": "rule",
                        "expected": expected,
                        "actual": {"type": type(error).__name__, "message": str(error)},
                    }
                )
        else:
            if "expected_error" in rule or observed != rule.get("expected"):
                issues.append(
                    {
                        "stage": "rule",
                        "expected": rule.get("expected_error", rule.get("expected")),
                        "actual": observed,
                    }
                )
        result = _result(case_id, "reference-fixture-authentication", "both", issues)
        result["candidate_executed"] = False
        result["classification"] = (
            "helper-only oracle fixture; installed-boundary mapping is recorded separately"
        )
        results.append(result)
    return results, declared


def _resource_boundary_axes(
    raw_spec: Mapping[str, Any], encoded_spec: Mapping[str, Any]
) -> dict[str, tuple[str, list[dict[str, Any]], int | None]]:
    text_request = {
        "messages": [{"role": "user", "content": "x"}],
        "options": {"add_generation_prompt": True},
        "images": [],
    }
    content_request = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            }
        ],
        "options": {},
        "images": [],
    }
    raw_request = _raw_request(raw_spec)
    encoded_request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "input_index": 0},
                    {"type": "text", "text": "encoded"},
                ],
            }
        ],
        "options": {},
        "images": [encoded_spec],
    }
    return {
        "requests": ("requests_per_batch", [text_request], None),
        "messages": ("messages_per_request", [text_request], None),
        "content_items": ("content_items_per_request", [content_request], None),
        "text_bytes": ("text_bytes_per_request", [text_request], None),
        "media_occurrences_request": ("media_per_request", [raw_request], None),
        "media_occurrences_batch": ("media_per_batch", [raw_request], None),
        "encoded_bytes_item": ("encoded_bytes_per_item", [encoded_request], None),
        "encoded_bytes_batch": ("encoded_bytes_per_batch", [encoded_request], None),
        "decoded_pixels": ("decoded_pixels_per_image_or_frame", [raw_request], None),
        # The zero-limit preflight reports the first checked edge (height), while
        # the public limit applies independently to both height and width.
        "edge_length": (
            "decoded_edge_length",
            [raw_request],
            max(int(raw_spec["height"]), int(raw_spec["width"])),
        ),
        "prepared_pixels": ("prepared_image_pixels_per_occurrence", [raw_request], None),
        "tokens_request": ("rendered_tokens_per_request", [text_request], None),
        "tokens_batch": ("rendered_tokens_per_batch", [text_request], None),
        "output_bytes": ("materialized_output_bytes_per_batch", [text_request], None),
    }


def _resource_probe_actual_value(
    probe: Mapping[str, Any], authoritative_actual: int | None
) -> tuple[int, dict[str, Any] | None]:
    if (
        probe.get("status") == "expected_error"
        and probe.get("error", {}).get("category") == "resource_limit"
    ):
        try:
            probed_actual = int(probe["error"]["context"]["actual"])
        except (KeyError, TypeError, ValueError):
            pass
        else:
            return authoritative_actual or probed_actual, None
    return authoritative_actual or 1, {
        "stage": "resource",
        "kind": "zero_limit_probe_failed",
        "actual": probe,
    }


def run_resource_boundaries(
    candidate_python: Path, assets_root: Path, output: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    root = repository_root()
    phase_b = _json(root / "reference/phase-b/v1/manifest.json")
    sources_path = root / "reference/phase-b/v1/sources.bin"
    raw_spec = _image_spec_from_phase_b(phase_b["sources"]["raw-a"], sources_path)
    encoded_spec = _image_spec_from_phase_b(phase_b["sources"]["jpeg"], sources_path)
    axes = _resource_boundary_axes(raw_spec, encoded_spec)
    results: list[dict[str, Any]] = []
    declared: list[str] = []
    for profile in PROFILES:
        for axis, (limit_name, requests, authoritative_actual) in axes.items():
            base = {
                "profile": profile,
                "assets_directory": str(_assets_directory(assets_root, profile)),
                "requests": requests,
            }
            probe, _ = _run_candidate(
                candidate_python,
                {**base, "limits": {limit_name: 0}},
                output / "resources" / profile / axis / "probe",
            )
            actual_value, probe_issue = _resource_probe_actual_value(probe, authoritative_actual)
            observed_success: dict[str, dict[str, np.ndarray]] = {}
            for position, limit, expectation in (
                ("one-below", max(0, actual_value - 1), "resource_limit"),
                ("at", actual_value, "success"),
                ("one-above", actual_value + 1, "success"),
            ):
                case_id = f"resource:{profile}:{axis}:{position}"
                declared.append(case_id)
                actual, actual_root = _run_candidate(
                    candidate_python,
                    {**base, "limits": {limit_name: limit}},
                    output / "resources" / profile / axis / position,
                )
                issues: list[dict[str, Any]] = []
                if probe_issue is not None:
                    issues.append(probe_issue)
                if expectation == "success":
                    if actual.get("status") != "success":
                        issues.append(
                            {"stage": "resource", "expected": "success", "actual": actual}
                        )
                    else:
                        observed_success[position] = {
                            name: _load_actual_array(actual, actual_root, name)
                            for name in actual["official_keys"]
                        }
                elif (
                    actual.get("status") != "expected_error"
                    or actual.get("error", {}).get("category") != expectation
                ):
                    issues.append({"stage": "resource", "expected": expectation, "actual": actual})
                results.append(_result(case_id, "live-resource-boundary", profile, issues))
            if set(observed_success) == {"at", "one-above"}:
                for name in observed_success["at"]:
                    if not np.array_equal(
                        observed_success["at"][name], observed_success["one-above"][name]
                    ):
                        issue = {
                            "stage": "resource",
                            "path": name,
                            "kind": "boundary_changed_output",
                        }
                        for result in results[-2:]:
                            result["issues"].append(issue)
                            result["passed"] = False
    return results, declared


def run_heterogeneous_permutations(
    candidate_python: Path, assets_root: Path, output: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    root = repository_root()
    phase_b = _json(root / "reference/phase-b/v1/manifest.json")
    source_path = root / "reference/phase-b/v1/sources.bin"
    by_id = {case["id"]: case for case in phase_b["cases"]}
    selected = [by_id[name] for name in ("text-only", "one-raw-rgb", "one-jpeg")]
    requests = [
        {
            "messages": case["messages"],
            "options": case["options"],
            "images": [
                _image_spec_from_phase_b(phase_b["sources"][name], source_path)
                for name in case["inputs"]
            ],
        }
        for case in selected
    ]
    orders = {"identity": [0, 1, 2], "permutation-0": [2, 0, 1], "permutation-1": [1, 2, 0]}
    results: list[dict[str, Any]] = []
    declared: list[str] = []
    for profile in PROFILES:
        runs: dict[str, tuple[dict[str, Any], Path]] = {}
        for name, order in orders.items():
            case_id = f"heterogeneous:{profile}:{name}"
            declared.append(case_id)
            actual, actual_root = _run_candidate(
                candidate_python,
                {
                    "profile": profile,
                    "assets_directory": str(_assets_directory(assets_root, profile)),
                    "requests": [requests[index] for index in order],
                },
                output / "heterogeneous" / profile / name,
            )
            runs[name] = (actual, actual_root)
            issues = (
                []
                if actual.get("status") == "success"
                else [{"stage": "envelope", "expected": "success", "actual": actual}]
            )
            results.append(_result(case_id, "live-heterogeneous-permutation", profile, issues))
        if any(value[0].get("status") != "success" for value in runs.values()):
            continue
        identity, identity_root = runs["identity"]
        identity_arrays = {
            name: _load_actual_array(identity, identity_root, name)
            for name in identity["official_keys"]
        }
        for run_index, (name, order) in enumerate(list(orders.items())[1:], start=1):
            actual, actual_root = runs[name]
            actual_arrays = {
                key: _load_actual_array(actual, actual_root, key) for key in actual["official_keys"]
            }
            issues = results[-3 + run_index]["issues"]
            for row, original in enumerate(order):
                identity_layout = identity["metadata"]["request_layouts"][original]
                actual_layout = actual["metadata"]["request_layouts"][row]
                if identity_layout["token_count"] != actual_layout["token_count"]:
                    issues.append(
                        {
                            "stage": "permutation",
                            "path": f"request[{original}].token_count",
                            "kind": "mismatch",
                        }
                    )
                    continue
                token_count = identity_layout["token_count"]
                for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
                    if not np.array_equal(identity_arrays[key][original], actual_arrays[key][row]):
                        issues.append(
                            {
                                "stage": "permutation",
                                "path": f"request[{original}].{key}",
                                "kind": "full_padded_row_order_dependent",
                                "token_count": token_count,
                            }
                        )
                for key in ("token_count", "right_padding"):
                    if identity_layout[key] != actual_layout[key]:
                        issues.append(
                            {
                                "stage": "permutation",
                                "path": f"request[{original}].{key}",
                                "kind": "order_dependent",
                            }
                        )
                expected_text_elements = [
                    row * actual_arrays["input_ids"].shape[1],
                    (row + 1) * actual_arrays["input_ids"].shape[1],
                ]
                if actual_layout["text_elements"] != expected_text_elements:
                    issues.append(
                        {
                            "stage": "permutation",
                            "path": f"request[{original}].text_elements",
                            "kind": "range_mismatch",
                        }
                    )
                if identity["metadata"]["text"][original] != actual["metadata"]["text"][row]:
                    issues.append(
                        {
                            "stage": "permutation",
                            "path": f"request[{original}].text_metadata",
                            "kind": "order_dependent",
                        }
                    )
            identity_images = {
                item["request_index"]: item for item in identity["metadata"]["images"]
            }
            actual_images = {item["request_index"]: item for item in actual["metadata"]["images"]}
            for row, original in enumerate(order):
                if original not in identity_images:
                    continue
                expected_item = identity_images[original]
                observed_item = actual_images[row]
                for key in ("input_index", "geometry", "cache_key"):
                    if expected_item[key] != observed_item[key]:
                        issues.append(
                            {
                                "stage": "permutation",
                                "path": f"request[{original}].{key}",
                                "kind": "order_dependent",
                            }
                        )
                expected_start, expected_end = expected_item["pixel_rows"]
                actual_start, actual_end = observed_item["pixel_rows"]
                if not np.array_equal(
                    identity_arrays["pixel_values"][expected_start:expected_end],
                    actual_arrays["pixel_values"][actual_start:actual_end],
                ):
                    issues.append(
                        {
                            "stage": "permutation",
                            "path": f"request[{original}].pixel_values",
                            "kind": "order_dependent",
                        }
                    )
                expected_grid = identity_arrays["image_grid_thw"][expected_item["grid_row"]]
                actual_grid = actual_arrays["image_grid_thw"][observed_item["grid_row"]]
                if not np.array_equal(expected_grid, actual_grid):
                    issues.append(
                        {
                            "stage": "permutation",
                            "path": f"request[{original}].image_grid_thw",
                            "kind": "order_dependent",
                        }
                    )
                layout = next(
                    item
                    for item in actual["metadata"]["image_layouts"]
                    if item["request_index"] == row
                )
                for key in (
                    "message_index",
                    "content_item_index",
                    "input_index",
                    "grid_row",
                    "pixel_rows",
                    "geometry",
                    "cache_key",
                ):
                    if layout[key] != observed_item[key]:
                        issues.append(
                            {
                                "stage": "permutation",
                                "path": f"request[{original}].image_layouts.{key}",
                                "kind": "range_or_order_mismatch",
                            }
                        )
                sidecar = next(
                    item
                    for item in actual["metadata"]["sidecar"]["images"]
                    if item["request_index"] == row
                )
                if sidecar["grid_row"] != observed_item["grid_row"]:
                    issues.append(
                        {
                            "stage": "permutation",
                            "path": f"request[{original}].sidecar.grid_row",
                            "kind": "order_mismatch",
                        }
                    )
            results[-3 + run_index]["passed"] = not issues
    return results, declared


def run_schema_and_arithmetic_errors(
    candidate_python: Path, assets_root: Path, output: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    root = repository_root()
    phase_b = _json(root / "reference/phase-b/v1/manifest.json")
    raw_spec = _image_spec_from_phase_b(
        phase_b["sources"]["raw-a"], root / "reference/phase-b/v1/sources.bin"
    )
    valid = {"messages": [{"role": "user", "content": "valid"}]}
    error_requests: dict[str, list[dict[str, Any]]] = {
        "request-unknown-key": [{"_raw_request": {**valid, "unknown": True}}],
        "request-missing-messages": [{"_raw_request": {}}],
        "messages-not-list": [{"_raw_request": {"messages": "bad"}}],
        "message-unknown-key": [
            {"_raw_request": {"messages": [{"role": "user", "content": "x", "unknown": 1}]}}
        ],
        "message-missing-role": [{"_raw_request": {"messages": [{"content": "x"}]}}],
        "invalid-role": [{"_raw_request": {"messages": [{"role": "alien", "content": "x"}]}}],
        "wrong-content-type": [{"_raw_request": {"messages": [{"role": "user", "content": 7}]}}],
        "missing-image-reference": [
            {"_raw_request": {"messages": [{"role": "user", "content": [{"type": "image"}]}]}}
        ],
        "out-of-range-image-reference": [
            {
                "_raw_request": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "image", "input_index": 1}],
                        }
                    ],
                    "images": [],
                }
            }
        ],
        "mixed-valid-invalid-batch": [
            {"_raw_request": valid},
            {"_raw_request": {"messages": [{"role": "alien", "content": "bad"}]}},
        ],
    }
    results: list[dict[str, Any]] = []
    declared: list[str] = []
    for profile in PROFILES:
        base = {
            "profile": profile,
            "assets_directory": str(_assets_directory(assets_root, profile)),
        }
        for error_id, requests in error_requests.items():
            case_id = f"schema-error:{profile}:{error_id}"
            declared.append(case_id)
            actual, _ = _run_candidate(
                candidate_python,
                {**base, "requests": requests},
                output / "schema-errors" / profile / error_id,
            )
            issues = []
            if (
                actual.get("status") != "expected_error"
                or actual.get("error", {}).get("category") != "invalid_request"
            ):
                issues.append({"stage": "schema", "expected": "invalid_request", "actual": actual})
            results.append(_result(case_id, "live-schema-error", profile, issues))
        arithmetic_id = f"arithmetic:{profile}:explicit-resized-pixel-overflow"
        declared.append(arithmetic_id)
        request = _raw_request(raw_spec, "overflow")
        image_item = request["messages"][0]["content"][0]
        image_item["options"] = {
            "resized_height": 4_294_967_296,
            "resized_width": 4_294_967_296,
        }
        actual, _ = _run_candidate(
            candidate_python,
            {**base, "requests": [request]},
            output / "arithmetic" / profile,
        )
        issues = []
        if (
            actual.get("status") != "expected_error"
            or actual.get("error", {}).get("category") != "arithmetic_overflow"
        ):
            issues.append(
                {"stage": "arithmetic", "expected": "arithmetic_overflow", "actual": actual}
            )
        results.append(_result(arithmetic_id, "live-arithmetic", profile, issues))
    return results, declared


def run_installed_geometry_rules(
    candidate_python: Path, assets_root: Path, output: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    definitions: dict[str, tuple[int, int, dict[str, int], tuple[int, int] | str]] = {
        "default-min-below": (31, 31, {}, (64, 64)),
        "default-min-edge": (63, 63, {}, (64, 64)),
        "explicit-min": (20, 20, {"min_pixels": 4096}, (64, 64)),
        "explicit-max": (2000, 2000, {"max_pixels": 4096}, (64, 64)),
        "exact-budget": (64, 96, {"min_pixels": 6144, "max_pixels": 6144}, (64, 96)),
        "explicit-paired": (
            20,
            20,
            {"resized_height": 64, "resized_width": 96},
            (64, 96),
        ),
        "explicit-unpaired": (20, 20, {"resized_height": 64}, "media_geometry"),
        "aspect-ratio-200": (100, 20000, {}, (96, 20000)),
        "aspect-ratio-over-200": (100, 20001, {}, "media_geometry"),
        "tie-48-up-even": (48, 64, {"min_pixels": 1}, (64, 64)),
        "tie-80-down-even": (80, 64, {"min_pixels": 1}, (64, 64)),
        "tie-112-up-even": (112, 64, {"min_pixels": 1}, (128, 64)),
    }
    input_directory = output / "geometry-inputs"
    input_directory.mkdir(parents=True, exist_ok=True)
    specs: dict[tuple[int, int], dict[str, Any]] = {}
    for height, width, _, _ in definitions.values():
        key = (height, width)
        if key in specs:
            continue
        payload = bytes(height * width * 3)
        path = input_directory / f"zero-{height}x{width}.rgb8"
        path.write_bytes(payload)
        specs[key] = {
            "kind": "raw",
            "path": str(path),
            "offset": 0,
            "length": len(payload),
            "sha256": _sha256_bytes(payload),
            "height": height,
            "width": width,
            "row_stride": width * 3,
            "layout": "positive",
        }
    results: list[dict[str, Any]] = []
    declared: list[str] = []
    for profile in PROFILES:
        for rule_id, (height, width, options, expectation) in definitions.items():
            case_id = f"geometry:{profile}:{rule_id}"
            declared.append(case_id)
            request = _raw_request(specs[(height, width)], rule_id)
            request["messages"][0]["content"][0]["options"] = options
            actual, actual_root = _run_candidate(
                candidate_python,
                {
                    "profile": profile,
                    "assets_directory": str(_assets_directory(assets_root, profile)),
                    "requests": [request],
                },
                output / "geometry" / profile / rule_id,
            )
            issues: list[dict[str, Any]] = []
            if isinstance(expectation, str):
                if (
                    actual.get("status") != "expected_error"
                    or actual.get("error", {}).get("category") != expectation
                ):
                    issues.append({"stage": "geometry", "expected": expectation, "actual": actual})
            elif actual.get("status") != "success":
                issues.append({"stage": "geometry", "expected": "success", "actual": actual})
            else:
                expected_height, expected_width = expectation
                metadata = actual["metadata"]["images"][0]
                geometry = metadata["geometry"]
                expected_grid = [1, expected_height // 16, expected_width // 16]
                expected_patches = int(np.prod(expected_grid, dtype=np.int64))
                expected_placeholders = expected_patches // 4
                for path, expected_value, actual_value in (
                    ("height", expected_height, geometry["height"]),
                    ("width", expected_width, geometry["width"]),
                    ("image_grid_thw", expected_grid, geometry["image_grid_thw"]),
                    ("patch_rows", expected_patches, geometry["patch_rows"]),
                    ("placeholder_count", expected_placeholders, geometry["placeholder_count"]),
                ):
                    if expected_value != actual_value:
                        issues.append(
                            {
                                "stage": "geometry",
                                "path": path,
                                "expected": expected_value,
                                "actual": actual_value,
                            }
                        )
                grid = _load_actual_array(actual, actual_root, "image_grid_thw")
                _compare_exact_array(
                    issues, "image_grid_thw", np.asarray([expected_grid], dtype=np.int64), grid
                )
                pixels = _load_actual_array(actual, actual_root, "pixel_values")
                expected_pixels = patchify_rgb(
                    np.zeros((expected_height, expected_width, 3), dtype=np.uint8)
                )
                _compare_float_array(
                    issues,
                    "pixel_values",
                    expected_pixels,
                    pixels,
                    1.0e-6,
                    pixel_metadata=actual["metadata"]["images"],
                )
            results.append(_result(case_id, "installed-geometry-rule", profile, issues))
    return results, declared


def _input_records() -> list[dict[str, str]]:
    root = repository_root()
    return [
        {"path": path.as_posix(), "sha256": _sha256_file(root / path)} for path in source_inputs()
    ]


def _asset_records(assets_root: Path) -> dict[str, Any]:
    root = repository_root()
    compatibility = _json(root / "reference/compatibility/v1.json")
    records: dict[str, Any] = {}
    for profile in PROFILES:
        directory = _assets_directory(assets_root, profile)
        files = []
        for name, expected in sorted(compatibility["profiles"][profile]["artifacts"].items()):
            path = directory / name
            observed = _sha256_file(path)
            if observed != expected:
                raise ValueError(f"profile asset mismatch for {profile}/{name}")
            files.append({"name": name, "sha256": observed})
        records[profile] = {
            "model_id": compatibility["profiles"][profile]["model_id"],
            "revision": compatibility["profiles"][profile]["revision"],
            "fingerprint": compatibility["profiles"][profile]["fingerprint"],
            "directory": directory.relative_to(assets_root).as_posix(),
            "files": files,
        }
    return records


def _source_fingerprint(inputs: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(
        _canonical([{"path": item["path"], "sha256": item["sha256"]} for item in inputs])
    )


def _has_forbidden_claim_key(value: Any) -> bool:
    forbidden = {
        "benchmark",
        "duration",
        "latency",
        "performance",
        "speed",
        "speedup",
        "throughput",
        "timing",
        "wall_ms",
    }
    if isinstance(value, Mapping):
        return any(
            any(token in forbidden for token in str(key).lower().replace("-", "_").split("_"))
            or _has_forbidden_claim_key(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_has_forbidden_claim_key(child) for child in value)
    return False


def _has_forbidden_claim_string(value: Any) -> bool:
    forbidden = {
        "benchmark",
        "faster",
        "latency",
        "performance",
        "slower",
        "speed",
        "speedup",
        "throughput",
        "timing",
    }
    if isinstance(value, str):
        return bool(forbidden.intersection(re.findall(r"[a-z0-9]+", value.lower())))
    if isinstance(value, Mapping):
        return any(_has_forbidden_claim_string(child) for child in value.values())
    if isinstance(value, list):
        return any(_has_forbidden_claim_string(child) for child in value)
    return False


def _validate_schema_contract() -> None:
    from jsonschema import Draft202012Validator

    schema = _json(repository_root() / SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    if schema.get("$id") != SCHEMA_ID or schema.get("additionalProperties") is not False:
        raise ValueError("Phase C JSON schema identity or closure is invalid")
    required = set(schema.get("required", []))
    expected = {
        "schema_id",
        "schema_version",
        "contract_id",
        "status",
        "passed",
        "scope",
        "candidate",
        "inputs",
        "source_fingerprint",
        "profile_assets",
        "comparator",
        "results",
        "provenance",
    }
    if not expected <= required:
        raise ValueError("Phase C JSON schema does not require the evidence envelope")
    candidate = schema.get("properties", {}).get("candidate", {})
    if candidate.get("additionalProperties") is not False:
        raise ValueError("Phase C JSON schema candidate envelope is permissive")
    runtime = candidate.get("properties", {}).get("runtime_identity", {})
    if runtime.get("additionalProperties") is not False:
        raise ValueError("Phase C JSON schema runtime identity is permissive")


def _validate_report_schema(report: Mapping[str, Any]) -> None:
    from jsonschema import Draft202012Validator

    schema = _json(repository_root() / SCHEMA_PATH)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(report), key=lambda error: list(error.absolute_path))
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise ValueError(f"Phase C report JSON Schema violation at {location}: {error.message}")


def validate_report(report: Mapping[str, Any], *, assets_root: Path) -> None:
    _validate_schema_contract()
    _validate_report_schema(report)
    if report.get("schema_id") != SCHEMA_ID or report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Phase C report schema")
    if report.get("contract_id") != "qwen-mm-compat-v1":
        raise ValueError("Phase C report contract mismatch")
    if report.get("status") != "pass" or report.get("passed") is not True:
        raise ValueError("Phase C report is not passing")
    claim_content = {
        key: value for key, value in report.items() if key not in {"inputs", "source_fingerprint"}
    }
    if _has_forbidden_claim_key(report) or _has_forbidden_claim_string(claim_content):
        raise ValueError("Phase C correctness report contains a forbidden claim field")
    created_at = report.get("created_at")
    try:
        created = datetime.fromisoformat(created_at)
    except (TypeError, ValueError) as error:
        raise ValueError("Phase C creation timestamp is invalid") from error
    if created.tzinfo is None or created.utcoffset() is None:
        raise ValueError("Phase C creation timestamp is invalid")
    scope = report.get("scope")
    if not isinstance(scope, Mapping) or scope.get("kind") != "text_image":
        raise ValueError("Phase C report scope is invalid")
    if scope.get("profiles") != list(PROFILES):
        raise ValueError("Phase C report does not cover both profiles")
    declared = scope.get("declared_case_ids")
    executed = scope.get("executed_case_ids")
    if not isinstance(declared, list) or not declared or len(declared) != len(set(declared)):
        raise ValueError("Phase C declared case inventory is invalid")
    if executed != declared or scope.get("skipped_case_ids") != []:
        raise ValueError("Phase C candidate case inventory is incomplete")
    canonical_ids = expected_candidate_case_ids()
    if declared != canonical_ids:
        missing = sorted(set(canonical_ids) - set(declared))
        extra = sorted(set(declared) - set(canonical_ids))
        raise ValueError(f"Phase C candidate inventory drifted: missing={missing}, extra={extra}")
    results = report.get("results")
    if not isinstance(results, list) or [item.get("case_id") for item in results] != declared:
        raise ValueError("Phase C result rows do not match the declared inventory")
    for item in results:
        if item.get("profile") not in PROFILES:
            if not (
                item.get("case_id") == "chat:unknown-profile"
                and item.get("tier") == "live-chat"
                and item.get("profile") == "qwen-unknown"
            ):
                raise ValueError(f"candidate case has invalid profile: {item.get('case_id')}")
        if item.get("candidate_executed") is not True:
            raise ValueError(f"candidate execution flag missing for {item.get('case_id')}")
        if item.get("passed") is not True or item.get("issues") != []:
            raise ValueError(f"candidate case did not pass: {item.get('case_id')}")
    if report.get("phase_e_exclusions") != list(PHASE_E_EXCLUSIONS):
        raise ValueError("Phase E exclusion inventory changed")
    if report.get("installed_boundary_rule_map") != expected_installed_boundary_rule_map():
        raise ValueError("installed boundary rule mapping changed or is incomplete")
    if report.get("comparator") != CANONICAL_COMPARATOR:
        raise ValueError("Phase C comparator policy changed")
    fixture_authentication = report.get("fixture_authentication")
    expected_fixture_ids = sorted(
        f"rule:{rule['id']}"
        for rule in _json(repository_root() / "reference/conformance/v1/rules.json")["rules"]
        if f"rule:{rule['id']}" not in PHASE_E_EXCLUSIONS
    )
    if (
        not isinstance(fixture_authentication, list)
        or sorted(item.get("case_id") for item in fixture_authentication) != expected_fixture_ids
    ):
        raise ValueError("Phase C rule fixture authentication is incomplete")
    for item in fixture_authentication:
        if (
            item.get("candidate_executed") is not False
            or not item.get("passed")
            or item.get("issues")
        ):
            raise ValueError("rule fixture authentication did not pass cleanly")
    expected_inputs = _input_records()
    if report.get("inputs") != expected_inputs:
        raise ValueError("Phase C authenticated source inputs are stale")
    if report.get("source_fingerprint") != _source_fingerprint(expected_inputs):
        raise ValueError("Phase C source fingerprint is stale")
    expected_assets = _asset_records(assets_root)
    reported_assets = report.get("profile_assets")
    if not isinstance(reported_assets, Mapping):
        raise ValueError("Phase C profile asset provenance is missing")
    for profile in PROFILES:
        for field in ("model_id", "revision", "fingerprint", "directory", "files"):
            if reported_assets.get(profile, {}).get(field) != expected_assets[profile][field]:
                raise ValueError(f"Phase C profile assets are stale for {profile}")
    runtime = report.get("candidate", {}).get("runtime_identity")
    required_runtime = {
        "package",
        "version",
        "package_artifact_sha256",
        "native_module",
        "native_artifact_sha256",
    }
    if not isinstance(runtime, Mapping) or set(runtime) != required_runtime:
        raise ValueError("Phase C candidate runtime identity is incomplete")
    for field in ("package_artifact_sha256", "native_artifact_sha256"):
        value = runtime[field]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("Phase C candidate runtime identity hash is invalid")
    environment = report.get("candidate", {}).get("environment")
    if not isinstance(environment, Mapping) or environment.get("numpy") != "2.4.6":
        raise ValueError("Phase C candidate environment is missing pinned NumPy")
    for field in (
        "python",
        "implementation",
        "abi_tag",
        "platform",
        "package_origin",
        "native_origin",
    ):
        if not isinstance(environment.get(field), str) or not environment[field]:
            raise ValueError(f"Phase C candidate environment lacks {field}")
    if environment.get("installed_origin") is not True:
        raise ValueError("Phase C candidate environment is not installed-wheel origin")
    wheel = report.get("candidate", {}).get("wheel")
    if not isinstance(wheel, Mapping) or set(wheel) != {
        "filename",
        "sha256",
        "tag",
        "build_command",
        "bound_runtime_identity",
    }:
        raise ValueError("Phase C wheel provenance is incomplete")
    if (
        not wheel["filename"].endswith(".whl")
        or len(wheel["sha256"]) != 64
        or not wheel["tag"]
        or wheel["build_command"] != "maturin build --locked"
        or wheel["bound_runtime_identity"] != runtime
    ):
        raise ValueError("Phase C wheel provenance is invalid")
    git = report.get("provenance", {}).get("git")
    current_git = _git_provenance(repository_root())
    if (
        not isinstance(git, Mapping)
        or git.get("gate_inputs_clean") is not True
        or git.get("gate_input_status") != []
        or current_git["gate_inputs_clean"] is not True
        or current_git["gate_input_status"] != []
        or not _git_revision_is_ancestor(repository_root(), git.get("revision"))
        or not _git_revision_matches_inputs(repository_root(), git.get("revision"), expected_inputs)
    ):
        raise ValueError("Phase C git provenance is stale or gate inputs are dirty")


def _git_provenance(root: Path) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    gate_paths = sorted(path.as_posix() for path in source_inputs())
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all", "--", *gate_paths],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    relevant = sorted(status)
    return {"revision": revision, "gate_inputs_clean": not relevant, "gate_input_status": relevant}


def _git_revision_is_ancestor(root: Path, revision: Any) -> bool:
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        return False
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", revision, "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def _git_revision_matches_inputs(
    root: Path, revision: Any, inputs: Sequence[Mapping[str, Any]]
) -> bool:
    """Require the recorded commit to contain each authenticated gate input verbatim."""
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        return False
    for record in inputs:
        path = record.get("path")
        expected_sha256 = record.get("sha256")
        if not isinstance(path, str) or not isinstance(expected_sha256, str):
            return False
        completed = subprocess.run(
            ["git", "show", f"{revision}:{path}"],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if completed.returncode or _sha256_bytes(completed.stdout) != expected_sha256:
            return False
    return True


def _write_summary(path: Path, report: Mapping[str, Any]) -> None:
    by_tier: dict[str, int] = {}
    for result in report["results"]:
        by_tier[result["tier"]] = by_tier.get(result["tier"], 0) + 1
    lines = [
        "# Phase C text/image conformance v1",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Candidate cases: {len(report['results'])}; skipped: {len(report['scope']['skipped_case_ids'])}.",
        "",
        (
            "Both pinned profiles passed the installed production-wheel text/image gate."
            if report["passed"]
            else "The installed production-wheel text/image gate failed; inspect report.json."
        ),
        "",
        "## Evidence counts",
        "",
    ]
    lines.extend(f"- `{tier}`: {count}" for tier, count in sorted(by_tier.items()))
    lines.extend(
        [
            "",
            "## Phase E exclusions",
            "",
            *[f"- `{item}`" for item in report["phase_e_exclusions"]],
            "",
            "The candidate process used isolated Python mode, imported the installed wheel, and loaded no forbidden oracle module.",
            "All source inputs, local profile assets, the wheel package, and the native extension are authenticated in `report.json`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_gate(
    *,
    candidate_python: Path,
    wheel_path: Path,
    assets_root: Path,
    output: Path,
    report_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    global _OBSERVED_CANDIDATE_PROVENANCE
    _OBSERVED_CANDIDATE_PROVENANCE = None
    output.mkdir(parents=True, exist_ok=True)
    candidate_results: list[dict[str, Any]] = []
    declared: list[str] = []
    canonical_ids = expected_candidate_case_ids()
    try:
        phase_b_results, phase_b_ids, candidate_provenance = run_phase_b(
            candidate_python, assets_root, output
        )
    except Exception as error:  # noqa: BLE001 - preserve a machine-readable failed gate
        phase_b_ids = [case_id for case_id in canonical_ids if case_id.startswith("phase-b:")]
        phase_b_results = [
            _result(
                case_id,
                "committed-phase-b",
                case_id.split(":")[1],
                [{"stage": "runner", "kind": "comparison_exception", "detail": str(error)}],
            )
            for case_id in phase_b_ids
        ]
        for result in phase_b_results:
            result["candidate_executed"] = False
        candidate_provenance = {
            "runtime_identity": {
                "package": "qwen_mm",
                "version": "unknown",
                "package_artifact_sha256": "0" * 64,
                "native_module": "qwen_mm._native",
                "native_artifact_sha256": "0" * 64,
            },
            "environment": {
                "python": "unknown",
                "implementation": "unknown",
                "numpy": "unknown",
                "abi_tag": "unknown",
                "platform": "unknown",
                "package_origin": "unknown",
                "native_origin": "unknown",
                "installed_origin": False,
            },
        }
    candidate_results.extend(phase_b_results)
    declared.extend(phase_b_ids)
    runners = (
        (run_chat, ("chat:",)),
        (run_media, ("media:",)),
        (run_committed_goldens, ("golden:",)),
        (run_raw_layouts, ("raw-layout:",)),
        (run_resource_boundaries, ("resource:",)),
        (run_heterogeneous_permutations, ("heterogeneous:",)),
        (run_schema_and_arithmetic_errors, ("schema-error:", "arithmetic:")),
        (run_installed_geometry_rules, ("geometry:",)),
    )
    for runner, prefixes in runners:
        try:
            rows, case_ids = runner(candidate_python, assets_root, output)
        except Exception as error:  # noqa: BLE001 - emit failed evidence instead of crashing
            case_ids = [case_id for case_id in canonical_ids if case_id.startswith(prefixes)]
            rows = [
                _result(
                    case_id,
                    "runner-failure",
                    case_id.split(":")[1] if case_id.split(":")[1] in PROFILES else "both",
                    [
                        {
                            "stage": "runner",
                            "kind": "comparison_exception",
                            "runner": runner.__name__,
                            "detail": str(error),
                        }
                    ],
                )
                for case_id in case_ids
            ]
            for result in rows:
                result["candidate_executed"] = False
        candidate_results.extend(rows)
        declared.extend(case_ids)
    fixture_authentication, _ = run_rules()
    for result in candidate_results:
        result.setdefault("candidate_executed", True)
    wheel_runtime_identity = _wheel_runtime_identity(wheel_path)
    try:
        _assert_wheel_runtime_binding(wheel_path, candidate_provenance["runtime_identity"])
    except ValueError as error:
        binding_issue = {
            "stage": "provenance",
            "kind": "wheel_runtime_mismatch",
            "detail": str(error),
        }
        for result in candidate_results:
            result["issues"].append(binding_issue)
            result["passed"] = False
    declared = sorted(declared)
    candidate_results.sort(key=lambda item: item["case_id"])
    executed = sorted(
        result["case_id"] for result in candidate_results if result["candidate_executed"] is True
    )
    skipped = sorted(set(declared) - set(executed))
    inputs = _input_records()
    passed = all(result["passed"] for result in candidate_results) and all(
        result["passed"] for result in fixture_authentication
    )
    root = repository_root()
    report = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "contract_id": "qwen-mm-compat-v1",
        "status": "pass" if passed else "fail",
        "passed": passed,
        "created_at": datetime.now(UTC).isoformat(),
        "scope": {
            "kind": "text_image",
            "profiles": list(PROFILES),
            "declared_case_ids": declared,
            "executed_case_ids": executed,
            "skipped_case_ids": skipped,
        },
        "candidate": {
            "runtime_identity": candidate_provenance["runtime_identity"],
            "environment": candidate_provenance["environment"],
            "wheel": {
                "filename": wheel_path.name,
                "sha256": _sha256_file(wheel_path),
                "tag": "-".join(wheel_path.name.removesuffix(".whl").rsplit("-", 3)[-3:]),
                "build_command": "maturin build --locked",
                "bound_runtime_identity": wheel_runtime_identity,
            },
            "gate_invocation_identity": {
                "entrypoint": "qwen_mm:Processor",
                "python_isolated_mode": True,
                "forbidden_modules": list(FORBIDDEN_CANDIDATE_MODULES),
            },
        },
        "inputs": inputs,
        "source_fingerprint": _source_fingerprint(inputs),
        "profile_assets": _asset_records(assets_root),
        "comparator": copy.deepcopy(CANONICAL_COMPARATOR),
        "installed_boundary_rule_map": expected_installed_boundary_rule_map(),
        "phase_e_exclusions": list(PHASE_E_EXCLUSIONS),
        "fixture_authentication": fixture_authentication,
        "results": candidate_results,
        "provenance": {
            "command": "make phase-c-conformance",
            "git": _git_provenance(root),
            "platform": {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "byte_order": sys.byteorder,
            },
        },
    }
    _write_json(report_path, report)
    _write_summary(summary_path, report)
    _validate_report_schema(report)
    if passed:
        validate_report(report, assets_root=assets_root)
    return report


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else repository_root() / path


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and validate Phase C installed-wheel evidence"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    candidate = subparsers.add_parser("_candidate")
    candidate.add_argument("--case", type=Path, required=True)
    candidate.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--candidate-python", type=Path, required=True)
    run.add_argument("--wheel", type=Path, required=True)
    run.add_argument("--assets-root", type=Path, default=Path("reference/.cache/huggingface"))
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--report", type=Path, default=REPORT_PATH)
    run.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--report", type=Path, default=REPORT_PATH)
    validate.add_argument("--assets-root", type=Path, default=Path("reference/.cache/huggingface"))
    args = parser.parse_args()
    if args.command == "_candidate":
        raise SystemExit(candidate_worker(args.case.resolve(), args.output.resolve()))
    assets_root = _resolve(args.assets_root)
    if args.command == "validate":
        validate_report(_json(_resolve(args.report)), assets_root=assets_root)
        print(f"valid Phase C report: {_resolve(args.report)}")
        return
    report = run_gate(
        candidate_python=_lexical_absolute(args.candidate_python),
        wheel_path=args.wheel.resolve(),
        assets_root=assets_root,
        output=_resolve(args.output),
        report_path=_resolve(args.report),
        summary_path=_resolve(args.summary),
    )
    print(json.dumps({"status": report["status"], "cases": len(report["results"])}, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
