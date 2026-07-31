from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import av
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError, features
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

from .bench import PACKAGE_NAMES, hugging_face_cache, split_video_metadata
from .fixtures import repository_root, sha256_file

GOLDEN_SCHEMA_PATH = Path("reference/goldens/schema-v1.json")
COMPATIBILITY_PATH = Path("reference/compatibility/v1.json")
WORKSPACE_LOCK_PATH = Path("uv.lock")
SEMANTIC_ENVIRONMENT_NAMES = (
    "MODEL_SEQ_LEN",
    "FORCE_QWENVL_VIDEO_READER",
    "TORCHCODEC_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TOKENIZERS_PARALLELISM",
)
OFFICIAL_ALWAYS_KEYS = {"input_ids", "attention_mask", "mm_token_type_ids"}
OFFICIAL_IMAGE_KEYS = {"pixel_values", "image_grid_thw"}
OFFICIAL_VIDEO_KEYS = {"pixel_values_videos", "video_grid_thw"}
STABLE_ERROR_CATEGORIES = {
    "invalid_request",
    "unsupported_option",
    "unsupported_media",
    "profile_mismatch",
    "media_decode",
    "media_geometry",
    "resource_limit",
    "arithmetic_overflow",
    "destination_too_small",
    "internal_invariant",
}


class GoldenExportError(RuntimeError):
    """An exporter failure with a stable compatibility category."""

    def __init__(self, category: str, message: str):
        if category not in STABLE_ERROR_CATEGORIES:
            raise ValueError(f"unknown stable error category: {category}")
        self.category = category
        super().__init__(message)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GoldenExportError("invalid_request", f"expected a JSON object in {path}")
    return value


def _repo_path(path: Path | str) -> Path:
    return repository_root() / Path(path)


def _relative_repo_path(path: Path) -> str:
    root = repository_root().resolve()
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise GoldenExportError(
            "invalid_request", f"media path is outside the repository: {path}"
        ) from error


def _sanitize_diagnostic(value: str) -> str:
    sanitized = value.replace(str(repository_root().resolve()), "$REPO")
    return re.sub(r"<_io\.BytesIO object at 0x[0-9A-Fa-f]+>", "<BytesIO>", sanitized)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _byte_order(dtype: np.dtype[Any]) -> str:
    if dtype.byteorder == "|":
        return "not-applicable"
    if dtype.byteorder == "<":
        return "little"
    if dtype.byteorder == ">":
        return "big"
    return sys.byteorder


def _safe_array_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", name).strip("._")


def describe_array(
    value: Any,
    *,
    name: str,
    inline_max_bytes: int,
    output_directory: Path | None = None,
    write_arrays: bool = False,
) -> dict[str, Any]:
    array = np.asarray(value)
    contiguous = np.ascontiguousarray(array)
    raw = memoryview(contiguous).cast("B")
    descriptor: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "strides": list(array.strides),
        "byte_order": _byte_order(array.dtype),
        "c_contiguous": bool(array.flags.c_contiguous),
        "nbytes": int(array.nbytes),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "hash_encoding": "C-order bytes in the declared dtype and byte order",
    }
    if array.nbytes <= inline_max_bytes:
        descriptor["storage"] = "inline_json"
        descriptor["data"] = array.tolist()
    elif write_arrays:
        if output_directory is None:
            raise ValueError("output_directory is required when write_arrays is true")
        arrays_directory = output_directory / "arrays"
        arrays_directory.mkdir(parents=True, exist_ok=True)
        array_path = arrays_directory / f"{_safe_array_name(name)}.npy"
        np.save(array_path, contiguous, allow_pickle=False)
        descriptor["storage"] = "npy"
        descriptor["path"] = array_path.relative_to(output_directory).as_posix()
        descriptor["file_sha256"] = sha256_file(array_path)
    else:
        descriptor["storage"] = "signature_only"
    return descriptor


def _text_descriptor(value: str) -> dict[str, Any]:
    encoded = value.encode("utf-8")
    return {
        "text": value,
        "utf8_base64": base64.b64encode(encoded).decode("ascii"),
        "bytes": len(encoded),
        "sha256": _sha256_bytes(encoded),
    }


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _installed_source_path(relative_path: str) -> Path:
    package_name = relative_path.split("/", 1)[0]
    spec = importlib.util.find_spec(package_name)
    if spec is None or not spec.submodule_search_locations:
        raise GoldenExportError("profile_mismatch", f"cannot locate installed {package_name}")
    package_root = Path(next(iter(spec.submodule_search_locations)))
    return package_root.parent / relative_path


def _profile_fingerprint(compatibility: Mapping[str, Any], alias: str) -> str:
    profile = copy.deepcopy(compatibility["profiles"][alias])
    profile.pop("fingerprint", None)
    payload = {
        "contract_id": compatibility["contract_id"],
        "python": compatibility["python"],
        "lock": compatibility["lock"],
        "packages": compatibility["packages"],
        "source_files": compatibility["source_files"],
        "environment": compatibility["environment"],
        "oracle_kwargs": compatibility["oracle_kwargs"],
        "profile": profile,
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


def _artifact_provenance(
    profile: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for filename, expected_sha256 in profile["artifacts"].items():
        try:
            path = Path(
                hf_hub_download(
                    profile["model_id"],
                    filename,
                    revision=profile["revision"],
                    cache_dir=hugging_face_cache(),
                    local_files_only=True,
                )
            )
            actual_sha256 = sha256_file(path)
            observed[filename] = {
                "bytes": path.stat().st_size,
                "expected_sha256": expected_sha256,
                "sha256": actual_sha256,
                "matches": actual_sha256 == expected_sha256,
            }
        except Exception as error:
            observed[filename] = {
                "expected_sha256": expected_sha256,
                "matches": False,
                "error_type": type(error).__name__,
                "error": _sanitize_diagnostic(str(error)),
            }
    return observed


def _codec_metadata() -> dict[str, Any]:
    pillow_codecs: dict[str, str | None] = {}
    for codec in sorted(features.get_supported_codecs()):
        pillow_codecs[codec] = features.version_codec(codec)
    return {
        "pillow": pillow_codecs,
        "pyav": {name: version for name, version in sorted(av.library_versions.items())},
    }


def collect_provenance(alias: str) -> tuple[dict[str, Any], dict[str, Any]]:
    compatibility_path = _repo_path(COMPATIBILITY_PATH)
    compatibility = _load_json(compatibility_path)
    if alias not in compatibility.get("profiles", {}):
        raise GoldenExportError("profile_mismatch", f"unknown profile alias: {alias}")
    profile = compatibility["profiles"][alias]

    expected_packages = compatibility["packages"]
    actual_packages = _package_versions()
    package_records = {
        name: {
            "expected": expected,
            "observed": actual_packages.get(name, "not-installed"),
            "matches": actual_packages.get(name) == expected,
        }
        for name, expected in expected_packages.items()
    }

    source_records: dict[str, dict[str, Any]] = {}
    for relative_path, expected_sha256 in compatibility["source_files"].items():
        path = _installed_source_path(relative_path)
        actual_sha256 = sha256_file(path) if path.exists() else None
        source_records[relative_path] = {
            "expected_sha256": expected_sha256,
            "sha256": actual_sha256,
            "matches": actual_sha256 == expected_sha256,
        }

    model_artifacts = _artifact_provenance(profile)
    computed_fingerprint = _profile_fingerprint(compatibility, alias)
    declared_lock = compatibility["lock"]
    declared_lock_path = _repo_path(declared_lock["path"])
    declared_lock_observed = (
        sha256_file(declared_lock_path) if declared_lock_path.exists() else None
    )
    workspace_lock_path = _repo_path(WORKSPACE_LOCK_PATH)
    workspace_lock_sha256 = sha256_file(workspace_lock_path)

    provenance = {
        "compatibility": {
            "contract_id": compatibility["contract_id"],
            "manifest_path": COMPATIBILITY_PATH.as_posix(),
            "manifest_sha256": sha256_file(compatibility_path),
            "schema_version": compatibility["schema_version"],
            "declared_lock": {
                **declared_lock,
                "observed_sha256": declared_lock_observed,
                "matches": declared_lock_observed == declared_lock["sha256"],
            },
        },
        "workspace_lock": {
            "path": WORKSPACE_LOCK_PATH.as_posix(),
            "sha256": workspace_lock_sha256,
        },
        "profile": {
            "alias": alias,
            "model_id": profile["model_id"],
            "revision": profile["revision"],
            "fingerprint": profile["fingerprint"],
            "computed_fingerprint": computed_fingerprint,
            "matches": computed_fingerprint == profile["fingerprint"],
            "classes": profile["classes"],
            "visual": profile["visual"],
            "tokenizer": profile["tokenizer"],
        },
        "packages": package_records,
        "source_files": source_records,
        "model_artifacts": model_artifacts,
        "environment": {name: os.environ.get(name) for name in SEMANTIC_ENVIRONMENT_NAMES},
        "platform": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "byte_order": sys.byteorder,
        },
        "codecs": _codec_metadata(),
    }

    mismatches = [
        *(f"package:{name}" for name, record in package_records.items() if not record["matches"]),
        *(f"source:{name}" for name, record in source_records.items() if not record["matches"]),
        *(f"artifact:{name}" for name, record in model_artifacts.items() if not record["matches"]),
    ]
    if computed_fingerprint != profile["fingerprint"]:
        mismatches.append("profile:fingerprint")
    if platform.python_version() != compatibility["python"]:
        mismatches.append("python:version")
    if mismatches:
        raise GoldenExportError(
            "profile_mismatch", "pinned environment mismatch: " + ", ".join(mismatches)
        )
    return compatibility, provenance


def _encoded_media_record(path_value: str) -> tuple[bytes, dict[str, Any]]:
    path = _repo_path(path_value)
    relative_path = _relative_repo_path(path)
    data = path.read_bytes()
    return data, {
        "path": relative_path,
        "bytes": len(data),
        "sha256": _sha256_bytes(data),
    }


def _decode_image(source: Mapping[str, Any], record: dict[str, Any]) -> Image.Image:
    if set(source) != {"path"} or not isinstance(source.get("path"), str):
        raise GoldenExportError(
            "invalid_request", "golden image sources require exactly one string path"
        )
    data, encoded = _encoded_media_record(source["path"])
    record["encoded"] = encoded
    image = Image.open(io.BytesIO(data))
    image.load()
    record["source_properties"] = {
        "format": image.format,
        "mode": image.mode,
        "width": image.width,
        "height": image.height,
        "exif_orientation": image.getexif().get(274),
    }
    if image.format == "WEBP":
        # VP8L is the lossless payload chunk, including inside an extended VP8X
        # container. The comparator needs this distinction because A1 freezes
        # exact prepared RGB for lossless WebP and a one-level bound for lossy.
        record["source_properties"]["lossless"] = b"VP8L" in data
    return image


def bind_requests(
    requests: Sequence[Mapping[str, Any]], media_records: list[dict[str, Any]]
) -> list[list[dict[str, Any]]]:
    bound_requests: list[list[dict[str, Any]]] = []
    occurrence = 0
    for request_index, request in enumerate(requests):
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise GoldenExportError("invalid_request", "each request needs non-empty messages")
        bound_messages = copy.deepcopy(messages)
        for message_index, message in enumerate(bound_messages):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for content_index, item in enumerate(content):
                if not isinstance(item, dict):
                    raise GoldenExportError("invalid_request", "content items must be objects")
                item_type = item.get("type")
                if item_type == "image":
                    record: dict[str, Any] = {
                        "kind": "image",
                        "occurrence": occurrence,
                        "request_index": request_index,
                        "message_index": message_index,
                        "content_index": content_index,
                    }
                    media_records.append(record)
                    item["image"] = _decode_image(item["image"], record)
                    occurrence += 1
                elif item_type == "video":
                    source = item.get("video")
                    if not isinstance(source, dict) or set(source) != {"frames"}:
                        raise GoldenExportError(
                            "unsupported_media", "golden videos require a frames descriptor"
                        )
                    frames = source["frames"]
                    if not isinstance(frames, list) or not frames:
                        raise GoldenExportError("media_geometry", "video frames cannot be empty")
                    record = {
                        "kind": "video",
                        "occurrence": occurrence,
                        "request_index": request_index,
                        "message_index": message_index,
                        "content_index": content_index,
                        "frames": [],
                        "options": {
                            key: _jsonable(value)
                            for key, value in item.items()
                            if key not in {"type", "video"}
                        },
                    }
                    media_records.append(record)
                    decoded_frames = []
                    for frame_index, frame_source in enumerate(frames):
                        frame_record: dict[str, Any] = {"frame_index": frame_index}
                        record["frames"].append(frame_record)
                        decoded_frames.append(_decode_image(frame_source, frame_record))
                    item["video"] = decoded_frames
                    occurrence += 1
        bound_requests.append(bound_messages)
    return bound_requests


@contextmanager
def _capture_replacements(processor: Any) -> Iterator[dict[str, list[str]]]:
    captured: dict[str, list[str]] = {"image": [], "video": []}
    original_image = processor.replace_image_token
    original_video = processor.replace_video_token

    def capture_image(*args: Any, **kwargs: Any) -> str:
        replacement = original_image(*args, **kwargs)
        captured["image"].append(replacement)
        return replacement

    def capture_video(*args: Any, **kwargs: Any) -> str:
        replacement = original_video(*args, **kwargs)
        captured["video"].append(replacement)
        return replacement

    processor.replace_image_token = capture_image
    processor.replace_video_token = capture_video
    try:
        yield captured
    finally:
        del processor.replace_image_token
        del processor.replace_video_token


def _split_videos(videos: Any) -> tuple[Any, Any]:
    return split_video_metadata(videos)


def _find_subsequence(haystack: list[int], needle: list[int], start: int) -> tuple[int, int]:
    if not needle:
        raise GoldenExportError("internal_invariant", "empty replacement token sequence")
    final_start = len(haystack) - len(needle)
    for index in range(start, final_start + 1):
        if haystack[index : index + len(needle)] == needle:
            return index, index + len(needle)
    raise GoldenExportError(
        "internal_invariant", "expanded replacement was not found in official input_ids"
    )


def _replacement_offsets(
    processor: Any,
    expanded_offsets: Sequence[Sequence[Mapping[str, Any]]],
    input_ids: np.ndarray,
) -> list[list[dict[str, Any]]]:
    exported: list[list[dict[str, Any]]] = []
    for batch_index, request_offsets in enumerate(expanded_offsets):
        row = np.asarray(input_ids[batch_index]).tolist()
        token_cursor = 0
        exported_request: list[dict[str, Any]] = []
        for offset in request_offsets:
            replacement = offset["replacement"]
            replacement_ids = processor.tokenizer.encode(replacement, add_special_tokens=False)
            token_start, token_end = _find_subsequence(row, replacement_ids, token_cursor)
            token_cursor = token_end
            replacement_bytes = replacement.encode("utf-8")
            exported_request.append(
                {
                    "type": offset["type"],
                    "original_codepoint_span": list(offset["span"]),
                    "expanded_codepoint_span": list(offset["new_span"]),
                    "expanded_token_span": [token_start, token_end],
                    "placeholder": offset["text"],
                    "replacement_utf8_bytes": len(replacement_bytes),
                    "replacement_sha256": _sha256_bytes(replacement_bytes),
                }
            )
        exported.append(exported_request)
    return exported


def _invocation_graph(
    compatibility: Mapping[str, Any], requests: Sequence[Mapping[str, Any]], video_kwargs: Any
) -> dict[str, Any]:
    render_calls = []
    for request_index, request in enumerate(requests):
        kwargs = {"tokenize": False, **request.get("options", {})}
        render_calls.append({"request_index": request_index, "kwargs": kwargs})
    processor_kwargs = {
        **compatibility["oracle_kwargs"]["processor"],
        **(_jsonable(video_kwargs) if video_kwargs else {}),
    }
    return {
        "nodes": [
            {
                "id": "bind_media",
                "call": "qwen_mm_reference.golden.bind_requests",
                "inputs": ["input.requests"],
                "outputs": ["bound structured messages", "input.media"],
            },
            {
                "id": "render_chat",
                "call": "processor.apply_chat_template",
                "depends_on": ["bind_media"],
                "calls": render_calls,
                "outputs": ["stages.rendered_prompts"],
            },
            {
                "id": "prepare_vision",
                "call": "qwen_vl_utils.process_vision_info",
                "depends_on": ["bind_media"],
                "kwargs": compatibility["oracle_kwargs"]["process_vision_info"],
                "outputs": ["stages.prepared_media", "stages.video_kwargs"],
            },
            {
                "id": "split_video_metadata",
                "call": "qwen_mm_reference.bench.split_video_metadata",
                "depends_on": ["prepare_vision"],
                "outputs": ["video values", "stages.video_metadata"],
            },
            {
                "id": "process",
                "call": "transformers.Qwen3VLProcessor.__call__",
                "depends_on": ["render_chat", "prepare_vision", "split_video_metadata"],
                "kwargs": processor_kwargs,
                "outputs": ["output.arrays", "captured multimodal replacements"],
            },
            {
                "id": "record_expanded_prompt",
                "call": "processor.get_text_with_replacements",
                "depends_on": ["process"],
                "note": "Replays replacements captured from the real processor call; media is not reprocessed.",
                "outputs": ["stages.expanded_prompts", "stages.replacement_offsets"],
            },
        ]
    }


def _stable_error(error: Exception, stage: str) -> tuple[str, Exception]:
    if isinstance(error, GoldenExportError):
        return error.category, error
    if isinstance(error, (UnidentifiedImageError, OSError)) and stage in {
        "bind_media",
        "prepare_vision",
    }:
        return "media_decode", error
    if isinstance(error, (AssertionError, ValueError)) and stage in {
        "prepare_vision",
        "processor",
    }:
        return "media_geometry", error
    if isinstance(error, (KeyError, TypeError, ValueError)):
        return "invalid_request", error
    return "internal_invariant", error


def _add_integrity(manifest: dict[str, Any]) -> None:
    manifest["integrity"] = {
        "algorithm": "sha256",
        "canonical_json_sha256": _sha256_bytes(_canonical_json_bytes(manifest)),
    }


def export_case(
    case: Mapping[str, Any],
    *,
    profile_alias: str,
    output_directory: Path,
    inline_max_bytes: int,
    write_arrays: bool,
) -> dict[str, Any]:
    schema_path = _repo_path(GOLDEN_SCHEMA_PATH)
    schema = _load_json(schema_path)
    compatibility, provenance = collect_provenance(profile_alias)
    requests = case.get("requests")
    if not isinstance(requests, list) or not requests:
        raise GoldenExportError("invalid_request", "case requires a non-empty requests list")
    if case.get("schema_version") != 1 or not isinstance(case.get("case_id"), str):
        raise GoldenExportError("invalid_request", "case requires schema_version 1 and case_id")

    media_records: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "schema": {
            "id": schema["schema_id"],
            "path": GOLDEN_SCHEMA_PATH.as_posix(),
            "sha256": sha256_file(schema_path),
        },
        "case_id": case["case_id"],
        "profile_alias": profile_alias,
        "input": {
            "requests": copy.deepcopy(requests),
            "media": media_records,
        },
        "provenance": provenance,
        "comparison_policy": schema["comparison_policy"],
        "storage_policy": {
            "inline_max_bytes": inline_max_bytes,
            "large_array_default": "npy" if write_arrays else "signature_only",
        },
    }
    expected_error = case.get("expected_error")
    stage = "bind_media"
    try:
        profile = compatibility["profiles"][profile_alias]
        processor = AutoProcessor.from_pretrained(
            profile["model_id"],
            revision=profile["revision"],
            cache_dir=hugging_face_cache(),
            local_files_only=True,
        )
        bound_requests = bind_requests(requests, media_records)

        stage = "render_chat"
        rendered_prompts = []
        for request, messages in zip(requests, bound_requests, strict=True):
            render_kwargs = {"tokenize": False, **request.get("options", {})}
            rendered_prompts.append(processor.apply_chat_template(messages, **render_kwargs))

        stage = "prepare_vision"
        images, videos, video_kwargs = process_vision_info(
            bound_requests,
            image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        video_values, video_metadata = _split_videos(videos)

        stage = "processor"
        processor_kwargs: dict[str, Any] = {
            "text": rendered_prompts,
            "images": images,
            "videos": video_values,
            "padding": True,
            "truncation": False,
            "do_resize": False,
            "do_sample_frames": False,
            "return_tensors": "np",
        }
        if video_metadata is not None:
            processor_kwargs["video_metadata"] = video_metadata
        if video_kwargs:
            processor_kwargs.update(video_kwargs)
        with _capture_replacements(processor) as replacements:
            outputs = processor(**processor_kwargs)

        stage = "expanded_prompt"
        expanded_prompts, replacement_offsets = processor.get_text_with_replacements(
            rendered_prompts.copy(),
            replacements["image"],
            replacements["video"],
        )
        exported_offsets = _replacement_offsets(
            processor, replacement_offsets, np.asarray(outputs["input_ids"])
        )
        if expected_error is not None:
            raise GoldenExportError("internal_invariant", "case succeeded but expected an error")

        stage = "serialize"
        prepared_images = []
        for index, image in enumerate(images or []):
            prepared_images.append(
                {
                    "occurrence": index,
                    "layout": "HWC RGB",
                    "array": describe_array(
                        np.asarray(image),
                        name=f"prepared__image__{index}",
                        inline_max_bytes=inline_max_bytes,
                        output_directory=output_directory,
                        write_arrays=write_arrays,
                    ),
                }
            )
        prepared_videos = []
        for index, video in enumerate(video_values or []):
            prepared_videos.append(
                {
                    "occurrence": index,
                    "layout": "TCHW RGB in the 0..255 float32 domain",
                    "array": describe_array(
                        video,
                        name=f"prepared__video__{index}",
                        inline_max_bytes=inline_max_bytes,
                        output_directory=output_directory,
                        write_arrays=write_arrays,
                    ),
                }
            )

        output_arrays = {
            name: describe_array(
                value,
                name=f"output__{name}",
                inline_max_bytes=inline_max_bytes,
                output_directory=output_directory,
                write_arrays=write_arrays,
            )
            for name, value in outputs.items()
        }
        manifest.update(
            {
                "status": "success",
                "invocation": _invocation_graph(compatibility, requests, video_kwargs),
                "stages": {
                    "rendered_prompts": [_text_descriptor(value) for value in rendered_prompts],
                    "prepared_media": {
                        "images": prepared_images,
                        "videos": prepared_videos,
                    },
                    "video_kwargs": _jsonable(video_kwargs),
                    "video_metadata": _jsonable(video_metadata),
                    "expanded_prompts": [_text_descriptor(value) for value in expanded_prompts],
                    "replacement_offsets": exported_offsets,
                },
                "output": {
                    "keys": list(outputs.keys()),
                    "arrays": output_arrays,
                },
            }
        )
    except Exception as error:
        category, diagnostic_error = _stable_error(error, stage)
        if expected_error is None:
            raise
        if not isinstance(expected_error, Mapping):
            raise GoldenExportError(
                "invalid_request", "expected_error must be an object"
            ) from error
        expected_category = expected_error.get("category")
        expected_stage = expected_error.get("stage")
        if category != expected_category or (
            expected_stage is not None and stage != expected_stage
        ):
            raise GoldenExportError(
                "internal_invariant",
                f"expected {expected_category} at {expected_stage}, got {category} at {stage}",
            ) from error
        manifest.update(
            {
                "status": "expected_error",
                "invocation": _invocation_graph(compatibility, requests, None),
                "expected_error": copy.deepcopy(expected_error),
                "error": {
                    "category": category,
                    "stage": stage,
                    "exception_module": type(diagnostic_error).__module__,
                    "exception_type": type(diagnostic_error).__name__,
                    "message": _sanitize_diagnostic(str(diagnostic_error)),
                    "args": [_sanitize_diagnostic(str(value)) for value in diagnostic_error.args],
                },
            }
        )

    _add_integrity(manifest)
    validate_manifest(manifest, schema=schema, verify_integrity=True)
    return manifest


def _require_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for segment in path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            raise ValueError(f"missing required manifest field: {path}")
        current = current[segment]
    return current


def _validate_text_descriptor(value: Mapping[str, Any]) -> None:
    text = _require_path(value, "text")
    encoded = text.encode("utf-8")
    if base64.b64decode(_require_path(value, "utf8_base64"), validate=True) != encoded:
        raise ValueError("prompt utf8_base64 does not match text")
    if value.get("bytes") != len(encoded) or value.get("sha256") != _sha256_bytes(encoded):
        raise ValueError("prompt byte count or SHA-256 does not match text")


def _validate_array_descriptor(value: Mapping[str, Any]) -> None:
    for field in (
        "shape",
        "dtype",
        "strides",
        "byte_order",
        "c_contiguous",
        "nbytes",
        "sha256",
        "hash_encoding",
        "storage",
    ):
        if field not in value:
            raise ValueError(f"array descriptor is missing {field}")
    if len(value["shape"]) != len(value["strides"]):
        raise ValueError("array shape and strides have different ranks")
    if value["storage"] == "inline_json":
        if "data" not in value:
            raise ValueError("inline array descriptor is missing data")
        array = np.asarray(value["data"], dtype=np.dtype(value["dtype"]))
        if list(array.shape) != value["shape"]:
            raise ValueError("inline array data has the wrong shape")
        if _sha256_bytes(np.ascontiguousarray(array).tobytes(order="C")) != value["sha256"]:
            raise ValueError("inline array data SHA-256 mismatch")
    elif value["storage"] == "npy":
        for field in ("path", "file_sha256"):
            if field not in value:
                raise ValueError(f"npy array descriptor is missing {field}")
    elif value["storage"] != "signature_only":
        raise ValueError(f"unknown array storage: {value['storage']}")


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    schema: Mapping[str, Any] | None = None,
    verify_integrity: bool = True,
) -> None:
    if schema is None:
        schema = _load_json(_repo_path(GOLDEN_SCHEMA_PATH))
    for path in schema["required_provenance"]:
        _require_path(manifest, path)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported golden manifest schema_version")
    if _require_path(manifest, "schema.id") != schema["schema_id"]:
        raise ValueError("golden manifest schema ID mismatch")
    if manifest.get("comparison_policy") != schema["comparison_policy"]:
        raise ValueError("golden manifest comparison policy mismatch")

    status = manifest.get("status")
    if status == "success":
        media = _require_path(manifest, "input.media")
        has_images = any(item.get("kind") == "image" for item in media)
        has_videos = any(item.get("kind") == "video" for item in media)
        expected_keys = set(OFFICIAL_ALWAYS_KEYS)
        if has_images:
            expected_keys.update(OFFICIAL_IMAGE_KEYS)
        if has_videos:
            expected_keys.update(OFFICIAL_VIDEO_KEYS)
        actual_keys = set(_require_path(manifest, "output.keys"))
        if actual_keys != expected_keys:
            raise ValueError(
                f"official output keys mismatch: expected {sorted(expected_keys)}, "
                f"got {sorted(actual_keys)}"
            )
        arrays = _require_path(manifest, "output.arrays")
        if set(arrays) != actual_keys:
            raise ValueError("output array descriptors do not match output.keys")
        for descriptor in arrays.values():
            _validate_array_descriptor(descriptor)
        for prompt in _require_path(manifest, "stages.rendered_prompts"):
            _validate_text_descriptor(prompt)
        for prompt in _require_path(manifest, "stages.expanded_prompts"):
            _validate_text_descriptor(prompt)
        prepared = _require_path(manifest, "stages.prepared_media")
        for item in [*prepared["images"], *prepared["videos"]]:
            _validate_array_descriptor(item["array"])
    elif status == "expected_error":
        category = _require_path(manifest, "error.category")
        if category not in STABLE_ERROR_CATEGORIES:
            raise ValueError(f"unknown stable error category: {category}")
        if category != _require_path(manifest, "expected_error.category"):
            raise ValueError("expected and observed error categories differ")
        _require_path(manifest, "error.exception_type")
        _require_path(manifest, "error.message")
    else:
        raise ValueError(f"unknown golden manifest status: {status}")

    if verify_integrity:
        integrity = _require_path(manifest, "integrity")
        unsigned = copy.deepcopy(dict(manifest))
        unsigned.pop("integrity", None)
        actual = _sha256_bytes(_canonical_json_bytes(unsigned))
        if (
            integrity.get("algorithm") != "sha256"
            or integrity.get("canonical_json_sha256") != actual
        ):
            raise ValueError("golden manifest integrity mismatch")


def _manifest_paths(paths: Sequence[Path]) -> list[Path]:
    manifests: list[Path] = []
    for path in paths:
        if path.is_dir():
            manifests.extend(sorted(path.rglob("manifest.json")))
        else:
            manifests.append(path)
    return sorted(set(manifests))


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export and validate deterministic qwen-mm golden manifests."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="export one case")
    export_parser.add_argument("--case", type=Path, required=True)
    export_parser.add_argument("--profile", required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--inline-max-bytes", type=int, default=1_048_576)
    export_parser.add_argument(
        "--write-arrays",
        action="store_true",
        help="write arrays larger than the inline limit as deterministic .npy files",
    )

    validate_parser = subparsers.add_parser("validate", help="validate manifests")
    validate_parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    if args.command == "export":
        if args.inline_max_bytes < 0:
            parser.error("--inline-max-bytes must be non-negative")
        case = _load_json(args.case)
        output_path = args.output
        output_directory = output_path.parent
        manifest = export_case(
            case,
            profile_alias=args.profile,
            output_directory=output_directory,
            inline_max_bytes=args.inline_max_bytes,
            write_arrays=args.write_arrays,
        )
        _write_manifest(output_path, manifest)
        print(
            f"exported {manifest['profile_alias']}/{manifest['case_id']} "
            f"({manifest['status']}) to {output_path}"
        )
        return

    manifests = _manifest_paths(args.paths)
    if not manifests:
        parser.error("no manifest.json files found")
    for path in manifests:
        validate_manifest(_load_json(path))
        print(f"valid: {path}")


if __name__ == "__main__":
    main()
