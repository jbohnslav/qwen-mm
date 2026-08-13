from __future__ import annotations

import copy
import ctypes
import gc
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
import platform
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .fixtures import fixture_directory, repository_root
from .resize_conformance_v2 import compare_final_tensors

WORKLOAD_SCHEMA_VERSION = 2
WORKLOAD_SCHEMA_ID = "qwen-mm-benchmark-workload-v2"
RESULT_SCHEMA_VERSION = 3
RESULT_SCHEMA_ID = "qwen-mm-benchmark-result-v3"
INTEGER_KEYS = ("input_ids", "attention_mask", "mm_token_type_ids", "image_grid_thw")
FLOAT_KEYS = ("pixel_values",)
THREAD_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
)
TIMING_SAMPLE_FIELDS = (
    "sequence",
    "wall_ms",
    "cpu_ms",
    "throughput_per_s",
    "core_utilization",
)
TIMING_FLOOR_POLICY = (
    "minimum_samples one-operation latency samples; supplemental individually clocked "
    "exact-stability operations aggregate only to minimum_seconds and are excluded from "
    "latency distributions"
)


class BenchmarkProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class MediaBuffer:
    kind: str
    value: bytes | np.ndarray
    format: str | None
    lossless: bool


@dataclass(frozen=True)
class CasePayload:
    case_id: str
    release_name: str | None
    boundary: str
    cache_mode: str
    float_atol: float
    messages: tuple[tuple[dict[str, Any], ...], ...]
    buffers: tuple[MediaBuffer, ...]
    work_units: int
    input_fingerprint: str
    logical_input_fingerprint: str
    messages_fingerprint: str


@dataclass(frozen=True)
class AdapterContext:
    profile_alias: str
    build_label: str
    thread_budget: int


class BenchmarkAdapter(Protocol):
    name: str

    def run(self, payload: CasePayload) -> Mapping[str, Any]: ...

    def metrics(self) -> Mapping[str, Any]: ...


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_workload(path: Path) -> dict[str, Any]:
    workload = json.loads(path.read_text(encoding="utf-8"))
    validate_workload(workload)
    return workload


def validate_workload(workload: Mapping[str, Any]) -> None:
    if workload.get("schema_version") != WORKLOAD_SCHEMA_VERSION:
        raise BenchmarkProtocolError("unsupported benchmark workload schema version")
    if workload.get("schema_id") != WORKLOAD_SCHEMA_ID:
        raise BenchmarkProtocolError("benchmark workload schema ID mismatch")
    cases = workload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise BenchmarkProtocolError("benchmark workload requires a non-empty cases list")
    seen: set[str] = set()
    boundaries = {
        "structured_messages_to_numpy",
        "encoded_to_numpy",
        "rgb_to_vllm_ready",
    }
    layouts = {"text", "one_request", "independent_requests"}
    cache_modes = {"disabled", "enabled", "separated"}
    for case in cases:
        if not isinstance(case, Mapping):
            raise BenchmarkProtocolError("benchmark cases must be objects")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise BenchmarkProtocolError("benchmark case IDs must be unique non-empty strings")
        seen.add(case_id)
        if case.get("boundary") not in boundaries:
            raise BenchmarkProtocolError(f"{case_id}: unsupported boundary")
        if case.get("layout") not in layouts:
            raise BenchmarkProtocolError(f"{case_id}: unsupported layout")
        if case.get("cache_mode") not in cache_modes:
            raise BenchmarkProtocolError(f"{case_id}: unsupported cache mode")
        if not isinstance(case.get("source"), Mapping):
            raise BenchmarkProtocolError(f"{case_id}: source must be an object")
        float_atol = case.get("float_atol")
        if not isinstance(float_atol, (int, float)) or float_atol < 0:
            raise BenchmarkProtocolError(f"{case_id}: float_atol must be non-negative")
        tags = case.get("tags")
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise BenchmarkProtocolError(f"{case_id}: tags must be strings")


def select_cases(
    workload: Mapping[str, Any], *, case_ids: Sequence[str] = (), tag: str | None = None
) -> list[dict[str, Any]]:
    by_id = {case["case_id"]: case for case in workload["cases"]}
    if case_ids:
        unknown = sorted(set(case_ids) - by_id.keys())
        if unknown:
            raise BenchmarkProtocolError(f"unknown benchmark cases: {', '.join(unknown)}")
        return [copy.deepcopy(by_id[case_id]) for case_id in case_ids]
    if tag is None:
        return copy.deepcopy(workload["cases"])
    return [copy.deepcopy(case) for case in workload["cases"] if tag in case["tags"]]


def _generated_rgb(height: int, width: int, seed: int) -> np.ndarray:
    y, x = np.indices((height, width), dtype=np.uint32)
    phase = np.uint32(seed & 0xFFFF)
    channels = np.stack(
        (
            (3 * x + y + phase) % 256,
            (x + 5 * y + 3 * phase) % 256,
            (7 * x + 2 * y + 11 * phase) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    channels.setflags(write=False)
    return channels


def _encode_rgb(array: np.ndarray, image_format: str) -> bytes:
    from PIL import Image

    image = Image.fromarray(array, mode="RGB")
    output = io.BytesIO()
    options: dict[str, Any] = {"format": image_format}
    if image_format == "JPEG":
        options.update(quality=90, subsampling=2, optimize=False, progressive=False)
    elif image_format == "PNG":
        options.update(compress_level=6, optimize=False)
    elif image_format == "WEBP":
        options.update(lossless=True, quality=100, method=6)
    image.save(output, **options)
    return output.getvalue()


def _materialize_buffers(source: Mapping[str, Any]) -> list[MediaBuffer]:
    kind = source.get("kind")
    if kind == "none":
        return []
    count = source.get("count")
    if not isinstance(count, int) or count <= 0:
        raise BenchmarkProtocolError("media source count must be positive")
    fixture_paths = sorted(fixture_directory().glob("*.jpg"))
    if kind in {"fixture_cycle", "fixture_repeat"} and not fixture_paths:
        raise BenchmarkProtocolError("baseline fixture directory is empty")
    if kind == "fixture_cycle":
        return [
            MediaBuffer(
                "encoded",
                fixture_paths[index % len(fixture_paths)].read_bytes(),
                "JPEG",
                False,
            )
            for index in range(count)
        ]
    if kind == "fixture_repeat":
        index = int(source.get("index", 0))
        value = fixture_paths[index].read_bytes()
        return [MediaBuffer("encoded", value, "JPEG", False) for _ in range(count)]
    shapes = source.get("shapes")
    if not isinstance(shapes, list) or not shapes:
        raise BenchmarkProtocolError("generated sources require shapes")
    seed = int(source.get("seed", 0))
    buffers: list[MediaBuffer] = []
    if kind == "generated_rgb":
        for index in range(count):
            height, width = shapes[index % len(shapes)]
            buffers.append(
                MediaBuffer("rgb", _generated_rgb(height, width, seed + index), None, True)
            )
        return buffers
    if kind == "generated_encoded":
        formats = source.get("formats")
        if not isinstance(formats, list) or not formats:
            raise BenchmarkProtocolError("generated encoded sources require formats")
        for index in range(count):
            height, width = shapes[index % len(shapes)]
            image_format = str(formats[index % len(formats)]).upper()
            rgb = _generated_rgb(height, width, seed + index)
            buffers.append(
                MediaBuffer(
                    "encoded",
                    _encode_rgb(rgb, image_format),
                    image_format,
                    image_format in {"PNG", "WEBP"},
                )
            )
        return buffers
    raise BenchmarkProtocolError(f"unsupported media source kind: {kind}")


def _message_content(buffer_indices: Sequence[int], instruction: str) -> list[dict[str, Any]]:
    content = [{"type": "image", "buffer_index": index} for index in buffer_indices]
    content.append({"type": "text", "text": instruction})
    return content


def _build_messages(
    case: Mapping[str, Any], buffer_count: int
) -> tuple[tuple[dict[str, Any], ...], ...]:
    layout = case["layout"]
    instruction = str(case.get("instruction", "Describe each image briefly."))
    if layout == "text":
        repetitions = int(case.get("text_repetitions", 1))
        text = "Explain the preprocessing contract briefly. " * repetitions
        return (({"role": "user", "content": [{"type": "text", "text": text}]},),)
    if layout == "one_request":
        message = {"role": "user", "content": _message_content(range(buffer_count), instruction)}
        return ((message,),)
    requests = []
    for index in range(buffer_count):
        message = {"role": "user", "content": _message_content([index], instruction)}
        requests.append((message,))
    return tuple(requests)


def _payload_fingerprint(
    case: Mapping[str, Any],
    messages: tuple[tuple[dict[str, Any], ...], ...],
    buffers: Sequence[MediaBuffer],
) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_json({"case": case, "messages": messages}))
    for media in buffers:
        digest.update(media.kind.encode("ascii"))
        digest.update((media.format or "").encode("ascii"))
        digest.update(b"1" if media.lossless else b"0")
        if isinstance(media.value, bytes):
            digest.update(media.value)
        else:
            digest.update(str(media.value.dtype).encode("ascii"))
            digest.update(_canonical_json(list(media.value.shape)))
            digest.update(memoryview(np.ascontiguousarray(media.value)).cast("B"))
    return digest.hexdigest()


def _logical_payload_fingerprint(
    case: Mapping[str, Any],
    messages: tuple[tuple[dict[str, Any], ...], ...],
    buffers: Sequence[MediaBuffer],
) -> str:
    source = case["source"]
    if source.get("kind") != "generated_encoded":
        return _payload_fingerprint(case, messages, buffers)
    digest = hashlib.sha256()
    digest.update(_canonical_json({"case": case, "messages": messages}))
    shapes = source["shapes"]
    formats = source["formats"]
    seed = int(source.get("seed", 0))
    for index in range(len(buffers)):
        height, width = shapes[index % len(shapes)]
        image_format = str(formats[index % len(formats)]).upper()
        rgb = _generated_rgb(height, width, seed + index)
        digest.update(
            _canonical_json(
                {
                    "kind": "generated_encoded",
                    "index": index,
                    "format": image_format,
                    "lossless": image_format in {"PNG", "WEBP"},
                    "dtype": str(rgb.dtype),
                    "shape": list(rgb.shape),
                }
            )
        )
        digest.update(memoryview(np.ascontiguousarray(rgb)).cast("B"))
    return digest.hexdigest()


def _messages_fingerprint(messages: tuple[tuple[dict[str, Any], ...], ...]) -> str:
    """Hash only the structured messages, separately from media and case policy."""

    return _sha256(_canonical_json({"messages": messages}))


def materialize_case(case: Mapping[str, Any]) -> CasePayload:
    buffers = _materialize_buffers(case["source"])
    messages = _build_messages(case, len(buffers))
    return CasePayload(
        case_id=case["case_id"],
        release_name=case.get("release_name"),
        boundary=case["boundary"],
        cache_mode=case["cache_mode"],
        float_atol=float(case["float_atol"]),
        messages=messages,
        buffers=tuple(buffers),
        work_units=max(1, len(messages)),
        input_fingerprint=_payload_fingerprint(case, messages, buffers),
        logical_input_fingerprint=_logical_payload_fingerprint(case, messages, buffers),
        messages_fingerprint=_messages_fingerprint(messages),
    )


def _expected_keys(payload: CasePayload) -> list[str]:
    keys = ["input_ids", "attention_mask", "mm_token_type_ids"]
    if payload.buffers:
        keys.extend(("pixel_values", "image_grid_thw"))
    return keys


def normalize_outputs(outputs: Mapping[str, Any], payload: CasePayload) -> dict[str, np.ndarray]:
    expected_keys = _expected_keys(payload)
    if list(outputs) != expected_keys:
        raise BenchmarkProtocolError(
            f"{payload.case_id}: output key/order mismatch: {list(outputs)!r} != {expected_keys!r}"
        )
    normalized = {name: np.asarray(value) for name, value in outputs.items()}
    for name in expected_keys:
        array = normalized[name]
        expected_dtype = np.dtype("float32" if name in FLOAT_KEYS else "int64")
        if array.dtype != expected_dtype:
            raise BenchmarkProtocolError(
                f"{payload.case_id}/{name}: dtype {array.dtype} != {expected_dtype}"
            )
        if array.ndim != 2:
            raise BenchmarkProtocolError(f"{payload.case_id}/{name}: expected rank 2")
        if not array.flags.c_contiguous:
            raise BenchmarkProtocolError(f"{payload.case_id}/{name}: output must be C-contiguous")
    if normalized["input_ids"].shape != normalized["attention_mask"].shape:
        raise BenchmarkProtocolError(f"{payload.case_id}: text array shapes differ")
    if normalized["input_ids"].shape != normalized["mm_token_type_ids"].shape:
        raise BenchmarkProtocolError(f"{payload.case_id}: token type array shape differs")
    if payload.buffers:
        if normalized["pixel_values"].shape[1] != 1536:
            raise BenchmarkProtocolError(f"{payload.case_id}: pixel width must be 1536")
        if normalized["image_grid_thw"].shape != (len(payload.buffers), 3):
            raise BenchmarkProtocolError(f"{payload.case_id}: image grid shape mismatch")
    return normalized


def compare_outputs(
    expected: Mapping[str, np.ndarray],
    actual: Mapping[str, np.ndarray],
    payload: CasePayload,
    *,
    exact_float: bool = False,
) -> None:
    if list(expected) != list(actual):
        raise BenchmarkProtocolError(f"{payload.case_id}: compared output keys differ")
    for name in expected:
        left = expected[name]
        right = actual[name]
        if left.dtype != right.dtype or left.shape != right.shape or left.strides != right.strides:
            raise BenchmarkProtocolError(
                f"{payload.case_id}/{name}: dtype, shape, or strides differ"
            )
        if name in INTEGER_KEYS or exact_float:
            equal = np.array_equal(left, right)
        elif name == "pixel_values":
            grid = expected.get("image_grid_thw")
            if grid is None or len(grid) != len(payload.buffers):
                raise BenchmarkProtocolError(
                    f"{payload.case_id}: cannot derive per-occurrence pixel tolerances"
                )
            per_patch: list[float] = []
            for media, row in zip(payload.buffers, grid, strict=True):
                tolerance = 1e-6 if media.lossless else payload.float_atol
                per_patch.extend([tolerance] * int(np.prod(row, dtype=np.int64)))
            if len(per_patch) != left.shape[0]:
                raise BenchmarkProtocolError(
                    f"{payload.case_id}: pixel rows do not match image grids"
                )
            tolerance_array = np.asarray(per_patch, dtype=np.float64)[:, np.newaxis]
            equal = bool(
                np.all(
                    np.less_equal(
                        np.abs(left.astype(np.float64) - right.astype(np.float64)),
                        tolerance_array,
                    )
                )
            )
        else:
            equal = np.allclose(left, right, rtol=0.0, atol=payload.float_atol, equal_nan=True)
        if not equal:
            maximum = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
            raise BenchmarkProtocolError(
                f"{payload.case_id}/{name}: values differ (maximum absolute error {maximum})"
            )


def compare_outputs_resize_v2(
    expected: Mapping[str, np.ndarray],
    actual: Mapping[str, np.ndarray],
    payload: CasePayload,
) -> dict[str, Any]:
    """Compare official and candidate outputs under the still-image resize-v2 contract.

    Only ``pixel_values`` may differ numerically.  All structure and every
    integer output remain exact; the pixel difference must produce a passing
    per-occurrence/per-channel RGB8 witness and both tensor slices must be the
    exact canonical downstream transform of their represented RGB8 values.
    """

    if not payload.buffers:
        compare_outputs(expected, actual, payload, exact_float=True)
        return {
            "comparison_id": "qwen-mm-exact-output-comparison-v1",
            "contract_id": "qwen-mm-compat-v1",
            "passed": True,
        }
    if list(expected) != list(actual):
        raise BenchmarkProtocolError(f"{payload.case_id}: compared output keys differ")
    for name in expected:
        left = expected[name]
        right = actual[name]
        if left.dtype != right.dtype or left.shape != right.shape or left.strides != right.strides:
            raise BenchmarkProtocolError(
                f"{payload.case_id}/{name}: dtype, shape, or strides differ"
            )
        if name != "pixel_values" and not np.array_equal(left, right):
            raise BenchmarkProtocolError(f"{payload.case_id}/{name}: values differ")
    try:
        witness = compare_final_tensors(
            expected["pixel_values"],
            actual["pixel_values"],
            expected["image_grid_thw"],
            source_dimensions=media_source_dimensions(payload),
        )
    except (OSError, ValueError) as error:
        raise BenchmarkProtocolError(f"{payload.case_id}/pixel_values: {error}") from error
    if not witness["passed"]:
        failed = [
            occurrence["occurrence"]
            for occurrence in witness["occurrences"]
            if not occurrence["passed"]
        ]
        raise BenchmarkProtocolError(
            f"{payload.case_id}/pixel_values: resize-v2 quality failed for occurrences {failed}"
        )
    return witness


def media_source_dimensions(payload: CasePayload) -> list[tuple[int, int]]:
    """Return authenticated source H/W for every still-image occurrence."""

    dimensions: list[tuple[int, int]] = []
    for media in payload.buffers:
        if isinstance(media.value, np.ndarray):
            dimensions.append((int(media.value.shape[0]), int(media.value.shape[1])))
        else:
            from PIL import Image

            with Image.open(io.BytesIO(media.value)) as image:
                dimensions.append((int(image.height), int(image.width)))
    return dimensions


def _compare_oracle_outputs(
    expected: Mapping[str, np.ndarray],
    actual: Mapping[str, np.ndarray],
    payload: CasePayload,
    *,
    require_resize_v2: bool,
) -> dict[str, Any]:
    if require_resize_v2:
        return compare_outputs_resize_v2(expected, actual, payload)
    compare_outputs(expected, actual, payload, exact_float=True)
    return {
        "comparison_id": "qwen-mm-exact-output-comparison-v1",
        "contract_id": "qwen-mm-compat-v1",
        "passed": True,
    }


def output_signature(outputs: Mapping[str, np.ndarray]) -> dict[str, Any]:
    signature: dict[str, Any] = {}
    for name, array in outputs.items():
        contiguous = np.ascontiguousarray(array)
        signature[name] = {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "strides": list(array.strides),
            "nbytes": int(array.nbytes),
            "sha256": _sha256(memoryview(contiguous).cast("B")),
        }
    return signature


class OfficialAdapter:
    name = "official-transformers"

    def __init__(self, context: AdapterContext) -> None:
        from transformers import AutoProcessor

        from .bench import hugging_face_cache, load_models

        model = load_models()[context.profile_alias]
        self._processor = AutoProcessor.from_pretrained(
            model["model_id"],
            revision=model["revision"],
            cache_dir=hugging_face_cache(),
            local_files_only=True,
        )
        self._metrics: dict[str, Any] = {}

    def run(self, payload: CasePayload) -> Mapping[str, Any]:
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        from .bench import split_video_metadata

        bound_requests = [
            [copy.deepcopy(message) for message in messages] for messages in payload.messages
        ]
        controlled_copies = 0
        for messages in bound_requests:
            for message in messages:
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for item in content:
                    if item.get("type") != "image":
                        continue
                    index = item.pop("buffer_index")
                    media = payload.buffers[index]
                    if isinstance(media.value, bytes):
                        image = Image.open(io.BytesIO(media.value))
                        image.load()
                    else:
                        image = Image.fromarray(media.value, mode="RGB")
                        controlled_copies += 1
                    item["image"] = image
        rendered = [
            self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in bound_requests
        ]
        images, videos, video_kwargs = process_vision_info(
            bound_requests,
            image_patch_size=self._processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        video_values, video_metadata = split_video_metadata(videos)
        kwargs: dict[str, Any] = {
            "text": rendered,
            "images": images,
            "videos": video_values,
            "padding": True,
            "truncation": False,
            "do_resize": False,
            "do_sample_frames": False,
            "return_tensors": "np",
        }
        if video_metadata is not None:
            kwargs["video_metadata"] = video_metadata
        if video_kwargs:
            kwargs.update(video_kwargs)
        outputs = self._processor(**kwargs)
        self._metrics = {
            "allocation_count": None,
            "copy_count": controlled_copies,
            "copy_count_scope": "controlled RGB boundary copies only",
            "transient_live_bytes": None,
            "cache_supported": False,
        }
        return outputs

    def metrics(self) -> Mapping[str, Any]:
        return self._metrics


class SyntheticAdapter:
    """Deterministic protocol self-test adapter; never valid for performance claims."""

    def __init__(self, context: AdapterContext) -> None:
        self.name = f"synthetic-{context.build_label}"
        self._metrics: dict[str, Any] = {}

    def run(self, payload: CasePayload) -> Mapping[str, Any]:
        batch = len(payload.messages)
        text_bytes = sum(
            len(str(item.get("text", "")).encode("utf-8"))
            for messages in payload.messages
            for message in messages
            for item in message.get("content", [])
        )
        sequence = max(4, min(64, 4 + text_bytes % 61))
        base = int(payload.input_fingerprint[:8], 16) % 10_000
        input_ids = np.arange(batch * sequence, dtype=np.int64).reshape(batch, sequence) + base
        outputs: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": np.ones((batch, sequence), dtype=np.int64),
            "mm_token_type_ids": np.zeros((batch, sequence), dtype=np.int64),
        }
        if payload.buffers:
            rows = len(payload.buffers) * 4
            outputs["pixel_values"] = np.full((rows, 1536), base / 10_000, dtype=np.float32)
            outputs["image_grid_thw"] = np.tile(
                np.asarray([[1, 2, 2]], dtype=np.int64), (len(payload.buffers), 1)
            )
        self._metrics = {
            "allocation_count": len(outputs),
            "copy_count": 0,
            "copy_count_scope": "complete synthetic adapter",
            "transient_live_bytes": sum(value.nbytes for value in outputs.values()),
            "cache_supported": True,
        }
        return outputs

    def metrics(self) -> Mapping[str, Any]:
        return self._metrics


AdapterFactory = Callable[[AdapterContext], BenchmarkAdapter]


def load_adapter(spec: str, context: AdapterContext) -> BenchmarkAdapter:
    if spec == "official":
        return OfficialAdapter(context)
    if spec == "synthetic":
        return SyntheticAdapter(context)
    if ":" not in spec:
        raise BenchmarkProtocolError("adapter spec must be official, synthetic, or module:factory")
    module_name, attribute_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute_name)
    adapter = factory(context)
    if not hasattr(adapter, "run") or not hasattr(adapter, "metrics"):
        raise BenchmarkProtocolError(f"adapter factory {spec} returned an invalid adapter")
    return adapter


def configure_threads(
    thread_budget: int,
    *,
    configure_torch: bool = True,
    environment: Mapping[str, str] | None = None,
    torch_thread_budget: int | None = None,
) -> dict[str, Any]:
    if thread_budget <= 0:
        raise BenchmarkProtocolError("thread budget must be positive")
    expected_environment = (
        {name: str(thread_budget) for name in THREAD_ENVIRONMENT_NAMES}
        if environment is None
        else dict(environment)
    )
    if set(expected_environment) != set(THREAD_ENVIRONMENT_NAMES) or any(
        not isinstance(value, str) or not value.isdigit() or int(value) <= 0
        for value in expected_environment.values()
    ):
        raise BenchmarkProtocolError("thread environment mapping is invalid")
    for name, value in expected_environment.items():
        os.environ[name] = value
    torch_threads: dict[str, int] = {}
    if configure_torch:
        try:
            import torch
        except ImportError:
            pass
        else:
            torch.set_num_threads(torch_thread_budget or thread_budget)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError as error:
                if torch.get_num_interop_threads() != 1:
                    raise BenchmarkProtocolError(
                        "Torch inter-op pool was already initialized above one thread"
                    ) from error
            torch_threads["num_threads"] = torch.get_num_threads()
            torch_threads["num_interop_threads"] = torch.get_num_interop_threads()
    return {
        "budget": thread_budget,
        "environment": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT_NAMES},
        "torch": torch_threads,
    }


def total_thread_budget_model(adapter_spec: str, thread_budget: int) -> dict[str, Any]:
    if thread_budget <= 0:
        raise BenchmarkProtocolError("thread budget must be positive")
    inactive = {name: "1" for name in THREAD_ENVIRONMENT_NAMES}
    if adapter_spec == "official":
        environment = {**inactive, "OMP_NUM_THREADS": str(thread_budget)}
        owner = "official_torch_intraop"
        torch_budget = thread_budget
    elif adapter_spec == "qwen_mm.benchmark:create_adapter":
        environment = inactive
        owner = "qwen_mm_processor_pool"
        torch_budget = 1
    else:
        environment = inactive
        owner = "adapter_context"
        torch_budget = 1
    return {
        "total_budget": thread_budget,
        "owner": owner,
        "inactive_runtime_budget": 1,
        "environment": environment,
        "torch_thread_budget": torch_budget,
    }


def architecture_family(machine: str | None = None) -> str:
    normalized = (machine or platform.machine()).lower()
    if normalized in {"arm64", "aarch64"}:
        return "arm64"
    if normalized in {"x86_64", "amd64"}:
        return "x86_64"
    return "other"


def environment_metadata() -> dict[str, Any]:
    package_names = (
        "numpy",
        "Pillow",
        "qwen-vl-utils",
        "tokenizers",
        "torch",
        "torchvision",
        "transformers",
    )
    packages: dict[str, str] = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "architecture_family": architecture_family(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "packages": packages,
    }


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise BenchmarkProtocolError("cannot summarize an empty sample set")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_samples(
    samples: Sequence[Mapping[str, Any]], resource_census: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    wall = [float(sample["wall_ms"]) for sample in samples]
    cpu = [float(sample["cpu_ms"]) for sample in samples]
    throughput = [float(sample["throughput_per_s"]) for sample in samples]
    summary: dict[str, Any] = {
        "sample_count": len(samples),
        "wall_ms": {
            "p50": percentile(wall, 0.50),
            "p90": percentile(wall, 0.90),
            "p99": percentile(wall, 0.99) if len(wall) >= 100 else None,
            "p99_qualified": len(wall) >= 100,
            "minimum": min(wall),
            "maximum": max(wall),
        },
        "cpu_ms": {"p50": percentile(cpu, 0.50), "p90": percentile(cpu, 0.90)},
        "throughput_per_s": {
            "p50": percentile(throughput, 0.50),
            "p90": percentile(throughput, 0.90),
        },
    }
    if resource_census is not None:
        rss = resource_census["rss"]
        adapter = resource_census["adapter_metrics"]
        summary.update(
            peak_rss_bytes=rss["peak_rss_bytes"],
            transient_live_bytes=rss["external_transient_rss_bytes"],
            allocation_count=adapter["allocation_count"],
            copy_count=adapter["copy_count"],
        )
    return summary


def _measure_once(
    adapter: BenchmarkAdapter, payload: CasePayload
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    gc.collect()
    return _measure_operation(adapter, payload)


def _measure_operation(
    adapter: BenchmarkAdapter, payload: CasePayload
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    wall_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    raw_outputs = adapter.run(payload)
    outputs = normalize_outputs(raw_outputs, payload)
    cpu_end = time.process_time_ns()
    wall_end = time.perf_counter_ns()
    wall_ms = (wall_end - wall_start) / 1_000_000
    cpu_ms = (cpu_end - cpu_start) / 1_000_000
    sample = {
        "wall_ms": wall_ms,
        "cpu_ms": cpu_ms,
        "throughput_per_s": payload.work_units / (wall_ms / 1_000),
        "core_utilization": cpu_ms / wall_ms if wall_ms else 0.0,
    }
    return outputs, sample


def _measure_supplemental_floor(
    adapter: BenchmarkAdapter,
    payload: CasePayload,
    baseline: Mapping[str, np.ndarray],
    *,
    required_elapsed_ms: float,
) -> dict[str, Any]:
    """Accumulate the time floor after retaining the raw one-operation samples.

    Every supplemental operation is individually clocked and checked for exact
    output stability, but only aggregate counters are retained.  Supplemental
    operations satisfy the five-second process floor and never contribute to
    latency percentiles, so fast candidates and slower references keep the same
    30 genuine one-operation samples without unbounded JSON growth.
    """

    gc.collect()
    iteration_count = 0
    elapsed_wall_ms = 0.0
    elapsed_cpu_ms = 0.0
    while elapsed_wall_ms < required_elapsed_ms:
        outputs, operation = _measure_operation(adapter, payload)
        compare_outputs(baseline, outputs, payload, exact_float=True)
        elapsed_wall_ms += operation["wall_ms"]
        elapsed_cpu_ms += operation["cpu_ms"]
        iteration_count += 1
        if iteration_count >= 1_000_000:
            raise BenchmarkProtocolError("supplemental timing-floor safety limit reached")
    return {
        "supplemental_iteration_count": iteration_count,
        "supplemental_elapsed_wall_ms": elapsed_wall_ms,
        "supplemental_elapsed_cpu_ms": elapsed_cpu_ms,
    }


def _nonnegative_integer(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkProtocolError("adapter resource counter must be a non-negative integer")
    return value


def _adapter_resource_metrics(adapter: BenchmarkAdapter) -> dict[str, Any]:
    raw = dict(adapter.metrics())
    return {
        "allocation_count": _nonnegative_integer(raw.get("allocation_count")),
        "copy_count": _nonnegative_integer(raw.get("copy_count")),
        "copied_bytes": _nonnegative_integer(raw.get("copied_bytes")),
        "retained_final_output_bytes": _nonnegative_integer(raw.get("retained_final_output_bytes")),
        "peak_transient_live_bytes": _nonnegative_integer(raw.get("transient_live_bytes")),
        "copy_count_scope": raw.get("copy_count_scope"),
        "cache_supported": raw.get("cache_supported"),
    }


def _resource_scope(value: Any) -> dict[str, int | None]:
    fields = {
        "request_index",
        "message_index",
        "content_item_index",
        "media_index",
        "input_index",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise BenchmarkProtocolError("native resource census scope is invalid")
    result: dict[str, int | None] = {}
    for field in sorted(fields):
        index = value[field]
        if index is not None and (
            isinstance(index, bool) or not isinstance(index, int) or index < 0
        ):
            raise BenchmarkProtocolError("native resource census scope index is invalid")
        result[field] = index
    return result


def _native_observation_census(adapter: BenchmarkAdapter) -> dict[str, Any] | None:
    observation_method = getattr(adapter, "observation_report", None)
    if not callable(observation_method):
        return None
    observation = observation_method()
    if not isinstance(observation, Mapping):
        raise BenchmarkProtocolError("native observation report must be an object")
    buffers = observation.get("buffers")
    copies = observation.get("copies")
    allocations = observation.get("allocations")
    calls = observation.get("calls")
    if (
        not isinstance(buffers, list)
        or not isinstance(copies, list)
        or not isinstance(allocations, Mapping)
        or not isinstance(calls, Mapping)
    ):
        raise BenchmarkProtocolError("native observation lacks buffer/allocation census")
    dropped_events = _nonnegative_integer(observation.get("dropped_events"))
    duration_ns = _nonnegative_integer(observation.get("duration_ns"))
    if observation.get("outcome") != "success" or dropped_events != 0:
        raise BenchmarkProtocolError("native resource census did not complete without event loss")
    if duration_ns is None:
        raise BenchmarkProtocolError("native resource census lacks operation duration")
    total_buffer_bytes = 0
    buffer_bytes_by_class: dict[str, int] = {}
    buffer_records: list[dict[str, Any]] = []
    for sequence, buffer in enumerate(buffers):
        if (
            not isinstance(buffer, Mapping)
            or buffer.get("sequence") != sequence
            or not isinstance(buffer.get("name"), str)
            or not buffer["name"]
            or not isinstance(buffer.get("class"), str)
            or not isinstance(buffer.get("scope"), Mapping)
        ):
            raise BenchmarkProtocolError("native resource census contains an invalid buffer")
        size = _nonnegative_integer(buffer.get("bytes"))
        if size is None:
            raise BenchmarkProtocolError("native resource census buffer lacks byte size")
        allocated_at = _nonnegative_integer(buffer.get("allocated_at_ns"))
        released_at = _nonnegative_integer(buffer.get("released_at_ns"))
        if (
            allocated_at is None
            or allocated_at > duration_ns
            or (released_at is not None and not allocated_at <= released_at <= duration_ns)
        ):
            raise BenchmarkProtocolError("native resource census buffer lifetime is invalid")
        total_buffer_bytes += size
        buffer_class = str(buffer["class"])
        buffer_bytes_by_class[buffer_class] = buffer_bytes_by_class.get(buffer_class, 0) + size
        buffer_records.append(
            {
                "sequence": sequence,
                "name": buffer["name"],
                "class": buffer_class,
                "scope": _resource_scope(buffer["scope"]),
                "bytes": size,
                "allocated_at_ns": allocated_at,
                "released_at_ns": released_at,
            }
        )
    allocation_census = {
        field: _nonnegative_integer(allocations.get(field))
        for field in (
            "allocation_count",
            "allocated_bytes",
            "copy_count",
            "copied_bytes",
            "transient_live_bytes",
            "peak_transient_live_bytes",
            "retained_final_output_bytes",
        )
    }
    if allocation_census["allocation_count"] != len(buffers):
        raise BenchmarkProtocolError("native allocation count differs from buffer census")
    if allocation_census["allocated_bytes"] != total_buffer_bytes:
        raise BenchmarkProtocolError("native allocated bytes differ from buffer census")
    copy_records: list[dict[str, Any]] = []
    for sequence, copy_event in enumerate(copies):
        if (
            not isinstance(copy_event, Mapping)
            or copy_event.get("sequence") != sequence
            or not isinstance(copy_event.get("name"), str)
            or not copy_event["name"]
            or not isinstance(copy_event.get("scope"), Mapping)
        ):
            raise BenchmarkProtocolError("native resource census contains an invalid copy")
        size = _nonnegative_integer(copy_event.get("bytes"))
        if size is None:
            raise BenchmarkProtocolError("native resource census copy lacks byte size")
        copy_records.append(
            {
                "sequence": sequence,
                "name": copy_event["name"],
                "scope": _resource_scope(copy_event["scope"]),
                "bytes": size,
            }
        )
    if allocation_census["copy_count"] != len(copy_records) or allocation_census[
        "copied_bytes"
    ] != sum(record["bytes"] for record in copy_records):
        raise BenchmarkProtocolError("native copy counters differ from copy census")
    return {
        "schema_version": observation.get("schema_version"),
        "outcome": "success",
        "dropped_events": 0,
        "duration_ns": duration_ns,
        "counter_scope": observation.get("counter_scope"),
        "allocations": allocation_census,
        "buffers": {
            "count": len(buffers),
            "total_bytes": total_buffer_bytes,
            "bytes_by_class": dict(sorted(buffer_bytes_by_class.items())),
            "records": buffer_records,
        },
        "copies": copy_records,
        "calls": {str(name): _nonnegative_integer(count) for name, count in sorted(calls.items())},
    }


def _resource_adapter_spec(config: Mapping[str, Any]) -> str:
    configured = config.get("resource_adapter_spec")
    if configured is not None:
        if not isinstance(configured, str) or not configured:
            raise BenchmarkProtocolError("resource adapter spec must be a non-empty string")
        return configured
    primary = str(config["adapter_spec"])
    if primary == "qwen_mm.benchmark:create_adapter":
        return "qwen_mm.benchmark:create_observed_adapter"
    return primary


def _darwin_current_rss_bytes() -> int:
    class MachTaskBasicInfo(ctypes.Structure):
        _fields_ = (
            ("virtual_size", ctypes.c_uint64),
            ("resident_size", ctypes.c_uint64),
            ("resident_size_max", ctypes.c_uint64),
            ("user_time_seconds", ctypes.c_int32),
            ("user_time_microseconds", ctypes.c_int32),
            ("system_time_seconds", ctypes.c_int32),
            ("system_time_microseconds", ctypes.c_int32),
            ("policy", ctypes.c_int32),
            ("suspend_count", ctypes.c_int32),
        )

    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    library.mach_task_self.restype = ctypes.c_uint
    library.task_info.argtypes = (
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint),
    )
    info = MachTaskBasicInfo()
    count = ctypes.c_uint(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_int))
    status = library.task_info(
        library.mach_task_self(),
        20,  # MACH_TASK_BASIC_INFO
        ctypes.byref(info),
        ctypes.byref(count),
    )
    if status != 0:
        raise BenchmarkProtocolError(f"Mach task_info failed with status {status}")
    return int(info.resident_size)


def _current_rss_bytes() -> tuple[int, str]:
    system = platform.system()
    if system == "Linux":
        fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
        return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE")), "linux-procfs-statm"
    if system == "Darwin":
        return _darwin_current_rss_bytes(), "darwin-mach-task-info"
    raise BenchmarkProtocolError(f"resource RSS census is unsupported on {system}")


class _RssSampler:
    def __init__(self) -> None:
        self.source: str | None = None
        self.baseline: int | None = None
        self.peak: int | None = None
        self.retained: int | None = None
        self.sample_count = 0
        self.error: BaseException | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._sample, name="qwen-mm-rss-census")

    def _sample(self) -> None:
        try:
            baseline, self.source = _current_rss_bytes()
            self.baseline = baseline
            self.peak = baseline
            self.retained = baseline
            self.sample_count = 1
            self._ready.set()
            while True:
                stopping = self._stop.wait(0.001)
                current, source = _current_rss_bytes()
                if source != self.source:
                    raise BenchmarkProtocolError("resource RSS source changed during census")
                self.peak = max(self.peak, current)
                self.retained = current
                self.sample_count += 1
                if stopping:
                    return
        except BaseException as error:
            self.error = error
            self._ready.set()

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(timeout=10):
            self._stop.set()
            self._thread.join(timeout=10)
            raise BenchmarkProtocolError("resource RSS sampler did not become ready")
        if self.error is not None:
            self._thread.join(timeout=10)
            raise BenchmarkProtocolError(
                "resource RSS sampler failed during startup"
            ) from self.error

    def finish(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise BenchmarkProtocolError("resource RSS sampler did not stop")
        if self.error is not None:
            raise BenchmarkProtocolError("resource RSS sampler failed") from self.error
        if (
            self.baseline is None
            or self.peak is None
            or self.retained is None
            or self.source is None
        ):
            raise BenchmarkProtocolError("resource RSS sampler result is incomplete")
        return {
            "baseline_rss_bytes": self.baseline,
            "peak_rss_bytes": self.peak,
            "retained_rss_bytes": self.retained,
            "sample_count": self.sample_count,
        }


def _resource_census(
    adapter: BenchmarkAdapter,
    payload: CasePayload,
    expected: Mapping[str, np.ndarray],
    *,
    adapter_spec: str,
) -> dict[str, Any]:
    gc.collect()
    sampler = _RssSampler()
    sampler.start()
    try:
        raw_outputs = adapter.run(payload)
        outputs = normalize_outputs(raw_outputs, payload)
    except BaseException:
        sampler.finish()
        raise
    rss_result = sampler.finish()
    compare_outputs(expected, outputs, payload, exact_float=True)
    baseline = _nonnegative_integer(rss_result.get("baseline_rss_bytes"))
    if baseline is None:
        raise BenchmarkProtocolError("resource RSS sampler baseline is missing")
    peak = _nonnegative_integer(rss_result.get("peak_rss_bytes"))
    retained = _nonnegative_integer(rss_result.get("retained_rss_bytes"))
    sample_count = _nonnegative_integer(rss_result.get("sample_count"))
    if peak is None or retained is None or sample_count is None or sample_count < 2:
        raise BenchmarkProtocolError("resource RSS sampler counters are incomplete")
    if peak < max(baseline, retained):
        raise BenchmarkProtocolError("resource RSS peak is below its envelope endpoints")
    output_bytes = sum(array.nbytes for array in outputs.values())
    adapter_metrics = _adapter_resource_metrics(adapter)
    native_observed = _native_observation_census(adapter)
    return {
        "timing_separation": "after_all_timed_samples",
        "adapter_spec": adapter_spec,
        "output_conformance": "exact_pass",
        "output_bytes": output_bytes,
        "rss": {
            "source": sampler.source,
            "sampler": "os_current_rss_sampler_thread",
            "baseline_rss_bytes": baseline,
            "peak_rss_bytes": peak,
            "retained_rss_bytes": retained,
            "rss_after_bytes": retained,
            "retained_rss_delta_bytes": max(0, retained - baseline),
            "transient_rss_bytes": max(0, peak - baseline - output_bytes),
            "external_transient_rss_bytes": max(0, peak - baseline - output_bytes),
            "sample_count": sample_count,
        },
        "adapter_metrics": adapter_metrics,
        "native_observed": native_observed,
    }


def _worker_affinity(config: Mapping[str, Any]) -> dict[str, Any]:
    requested = config.get("affinity_cpus")
    if requested is not None and (
        not isinstance(requested, list)
        or not requested
        or any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in requested)
        or len(requested) != len(set(requested))
    ):
        raise BenchmarkProtocolError("worker affinity CPUs must be unique non-negative integers")
    requested_cpus = None if requested is None else sorted(requested)
    system = platform.system()
    if system == "Darwin":
        return {
            "requested_cpus": requested_cpus,
            "status": "unavailable",
            "mechanism": None,
            "observed_cpus": None,
            "reason": "macOS does not expose a supported process CPU-affinity API",
        }
    if system != "Linux":
        return {
            "requested_cpus": requested_cpus,
            "status": "unavailable",
            "mechanism": None,
            "observed_cpus": None,
            "reason": f"CPU affinity is unsupported on {system}",
        }
    try:
        observed = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError) as error:
        raise BenchmarkProtocolError("Linux worker cannot attest CPU affinity") from error
    if requested_cpus is not None and observed != requested_cpus:
        raise BenchmarkProtocolError(
            f"Linux worker CPU affinity {observed} differs from requested {requested_cpus}"
        )
    return {
        "requested_cpus": requested_cpus,
        "status": "attested" if requested_cpus is not None else "observed",
        "mechanism": "taskset+sched_getaffinity"
        if requested_cpus is not None
        else "sched_getaffinity",
        "observed_cpus": observed,
        "reason": None,
    }


def _valid_finite_number(value: Any, *, minimum: float, strictly_positive: bool) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    if not math.isfinite(number):
        return False
    return number > minimum if strictly_positive else number >= minimum


def run_worker(config: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    warmups = config.get("warmups")
    minimum_samples_value = config.get("minimum_samples")
    minimum_seconds_value = config.get("minimum_seconds")
    if (
        isinstance(warmups, bool)
        or not isinstance(warmups, int)
        or warmups < 0
        or isinstance(minimum_samples_value, bool)
        or not isinstance(minimum_samples_value, int)
        or minimum_samples_value <= 0
        or not _valid_finite_number(minimum_seconds_value, minimum=0.0, strictly_positive=False)
    ):
        raise BenchmarkProtocolError("invalid benchmark worker sample protocol")
    primary_spec = str(config["adapter_spec"])
    specs = {primary_spec, str(config["oracle_adapter_spec"])}
    total_budget_model = total_thread_budget_model(primary_spec, int(config["thread_budget"]))
    thread_settings = configure_threads(
        int(config["thread_budget"]),
        configure_torch="official" in specs,
        environment=total_budget_model["environment"],
        torch_thread_budget=total_budget_model["torch_thread_budget"],
    )
    thread_settings["total_budget_model"] = total_budget_model
    payload = materialize_case(case)
    context = AdapterContext(
        profile_alias=str(config["profile_alias"]),
        build_label=str(config["build_label"]),
        thread_budget=int(config["thread_budget"]),
    )
    primary = load_adapter(primary_spec, context)
    oracle = load_adapter(str(config["oracle_adapter_spec"]), context)
    affinity = _worker_affinity(config)

    oracle_pre = normalize_outputs(oracle.run(payload), payload)
    primary_pre = normalize_outputs(primary.run(payload), payload)
    require_resize_v2 = bool(payload.buffers) and primary_spec != config["oracle_adapter_spec"]
    pre_measurement_witness = _compare_oracle_outputs(
        oracle_pre,
        primary_pre,
        payload,
        require_resize_v2=require_resize_v2,
    )
    baseline = {name: value.copy() for name, value in primary_pre.items()}

    for _ in range(warmups):
        warmed = normalize_outputs(primary.run(payload), payload)
        compare_outputs(baseline, warmed, payload, exact_float=True)

    minimum_samples = minimum_samples_value
    minimum_seconds = float(minimum_seconds_value)
    samples: list[dict[str, Any]] = []
    for sequence in range(minimum_samples):
        outputs, sample = _measure_once(primary, payload)
        compare_outputs(baseline, outputs, payload, exact_float=True)
        samples.append({"sequence": sequence, **sample})
    raw_sample_elapsed_ms = sum(sample["wall_ms"] for sample in samples)
    supplemental = _measure_supplemental_floor(
        primary,
        payload,
        baseline,
        required_elapsed_ms=max(0.0, minimum_seconds * 1_000 - raw_sample_elapsed_ms),
    )
    timing_floor = {
        "required_seconds": minimum_seconds,
        "raw_sample_iteration_count": len(samples),
        "raw_sample_elapsed_wall_ms": raw_sample_elapsed_ms,
        **supplemental,
    }
    timing_floor["total_iteration_count"] = (
        timing_floor["raw_sample_iteration_count"] + timing_floor["supplemental_iteration_count"]
    )
    timing_floor["total_elapsed_wall_ms"] = (
        timing_floor["raw_sample_elapsed_wall_ms"] + timing_floor["supplemental_elapsed_wall_ms"]
    )

    primary_post = normalize_outputs(primary.run(payload), payload)
    oracle_post = normalize_outputs(oracle.run(payload), payload)
    compare_outputs(baseline, primary_post, payload, exact_float=True)
    post_measurement_witness = _compare_oracle_outputs(
        oracle_post,
        primary_post,
        payload,
        require_resize_v2=require_resize_v2,
    )
    compare_outputs(oracle_pre, oracle_post, payload, exact_float=True)

    census_spec = _resource_adapter_spec(config)
    census_adapter = load_adapter(census_spec, context)
    resource_census = _resource_census(
        census_adapter,
        payload,
        baseline,
        adapter_spec=census_spec,
    )

    worker_nonce = config.get("worker_nonce")
    if worker_nonce is None:
        worker_nonce = os.urandom(32).hex()
    if (
        not isinstance(worker_nonce, str)
        or len(worker_nonce) != 64
        or any(character not in "0123456789abcdef" for character in worker_nonce)
    ):
        raise BenchmarkProtocolError("worker nonce must be a 256-bit lowercase hex value")

    return {
        "implementation": config["implementation"],
        "adapter_spec": config["adapter_spec"],
        "adapter_name": primary.name,
        "oracle_adapter_spec": config["oracle_adapter_spec"],
        "profile_alias": context.profile_alias,
        "build_label": context.build_label,
        "thread_settings": thread_settings,
        "process_id": os.getpid(),
        "worker_nonce": worker_nonce,
        "affinity": affinity,
        "environment": environment_metadata(),
        "case_id": payload.case_id,
        "work_units": payload.work_units,
        "release_name": payload.release_name,
        "boundary": payload.boundary,
        "cache_mode": payload.cache_mode,
        "input_fingerprint": payload.input_fingerprint,
        "logical_input_fingerprint": payload.logical_input_fingerprint,
        "messages_fingerprint": payload.messages_fingerprint,
        "output_signature": output_signature(baseline),
        "conformance": {
            "pre_measurement": "pass",
            "pre_measurement_witness": pre_measurement_witness,
            "all_measured_iterations_stable": True,
            "post_measurement": "pass",
            "post_measurement_witness": post_measurement_witness,
            "float_atol": payload.float_atol,
        },
        "timing_scope": {
            "clocks": ["perf_counter_ns", "process_time_ns"],
            "boundary": "adapter.run+required_output_normalization",
            "instrumentation": "none",
            "resource_census": "separate_after_all_timed_samples",
        },
        "warmups": warmups,
        "minimum_samples": minimum_samples,
        "minimum_seconds": minimum_seconds,
        "samples": samples,
        "timing_floor": timing_floor,
        "resource_census": resource_census,
        "summary": summarize_samples(samples, resource_census),
    }


def physical_cpu_count() -> int:
    try:
        value = os.sysconf("SC_NPROCESSORS_ONLN")
    except (AttributeError, OSError, ValueError):
        value = None
    return max(1, int(value or os.cpu_count() or 1))


def workload_provenance(path: Path) -> dict[str, Any]:
    schema_path = repository_root() / "benchmarks" / "workload-schema-v2.json"
    return {
        "path": str(path.resolve().relative_to(repository_root())),
        "sha256": _sha256(path.read_bytes()),
        "schema_path": str(schema_path.relative_to(repository_root())),
        "schema_sha256": _sha256(schema_path.read_bytes()),
    }
