from __future__ import annotations

import copy
import gc
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import platform
import resource
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .fixtures import fixture_directory, repository_root

WORKLOAD_SCHEMA_VERSION = 2
WORKLOAD_SCHEMA_ID = "qwen-mm-benchmark-workload-v2"
RESULT_SCHEMA_VERSION = 2
RESULT_SCHEMA_ID = "qwen-mm-benchmark-result-v2"
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


def configure_threads(thread_budget: int, *, configure_torch: bool = True) -> dict[str, Any]:
    if thread_budget <= 0:
        raise BenchmarkProtocolError("thread budget must be positive")
    for name in THREAD_ENVIRONMENT_NAMES:
        os.environ[name] = str(thread_budget)
    torch_threads: dict[str, int] = {}
    if configure_torch:
        try:
            import torch

            torch.set_num_threads(thread_budget)
            torch_threads["num_threads"] = torch.get_num_threads()
            torch_threads["num_interop_threads"] = torch.get_num_interop_threads()
        except (ImportError, RuntimeError):
            pass
    return {
        "budget": thread_budget,
        "environment": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT_NAMES},
        "torch": torch_threads,
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


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise BenchmarkProtocolError("cannot summarize an empty sample set")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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
        "peak_rss_bytes": max(int(sample["peak_rss_bytes"]) for sample in samples),
        "transient_live_bytes": max(int(sample["transient_live_bytes"]) for sample in samples),
        "allocation_count": sum(int(sample["allocation_count"]) for sample in samples),
        "copy_count": (
            sum(int(sample["copy_count"]) for sample in samples)
            if all(sample["copy_count"] is not None for sample in samples)
            else None
        ),
    }
    return summary


def _measure_once(
    adapter: BenchmarkAdapter, payload: CasePayload
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    gc.collect()
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    current_before, _ = tracemalloc.get_traced_memory()
    cpu_start = time.process_time_ns()
    wall_start = time.perf_counter_ns()
    raw_outputs = adapter.run(payload)
    wall_end = time.perf_counter_ns()
    cpu_end = time.process_time_ns()
    current_after, peak = tracemalloc.get_traced_memory()
    after = tracemalloc.take_snapshot()
    traced_allocations = sum(
        max(0, statistic.count_diff) for statistic in after.compare_to(before, "lineno")
    )
    tracemalloc.stop()
    outputs = normalize_outputs(raw_outputs, payload)
    adapter_metrics = dict(adapter.metrics())
    wall_ms = (wall_end - wall_start) / 1_000_000
    cpu_ms = (cpu_end - cpu_start) / 1_000_000
    transient = adapter_metrics.get("transient_live_bytes")
    if transient is None:
        transient = max(0, peak - min(current_before, current_after))
    allocations = adapter_metrics.get("allocation_count")
    if allocations is None:
        allocations = traced_allocations
    sample = {
        "wall_ms": wall_ms,
        "cpu_ms": cpu_ms,
        "throughput_per_s": payload.work_units / (wall_ms / 1_000),
        "core_utilization": cpu_ms / wall_ms if wall_ms else 0.0,
        "peak_rss_bytes": _peak_rss_bytes(),
        "transient_live_bytes": int(transient),
        "tracemalloc_live_delta_bytes": int(current_after - current_before),
        "allocation_count": int(allocations),
        "copy_count": adapter_metrics.get("copy_count"),
        "copy_count_scope": adapter_metrics.get("copy_count_scope"),
        "cache_supported": adapter_metrics.get("cache_supported"),
    }
    return outputs, sample


def run_worker(config: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    specs = {str(config["adapter_spec"]), str(config["oracle_adapter_spec"])}
    thread_settings = configure_threads(
        int(config["thread_budget"]), configure_torch="official" in specs
    )
    payload = materialize_case(case)
    context = AdapterContext(
        profile_alias=str(config["profile_alias"]),
        build_label=str(config["build_label"]),
        thread_budget=int(config["thread_budget"]),
    )
    primary = load_adapter(str(config["adapter_spec"]), context)
    oracle = load_adapter(str(config["oracle_adapter_spec"]), context)

    oracle_pre = normalize_outputs(oracle.run(payload), payload)
    primary_pre = normalize_outputs(primary.run(payload), payload)
    compare_outputs(oracle_pre, primary_pre, payload)
    baseline = {name: value.copy() for name, value in primary_pre.items()}

    for _ in range(int(config["warmups"])):
        warmed = normalize_outputs(primary.run(payload), payload)
        compare_outputs(baseline, warmed, payload, exact_float=True)

    samples: list[dict[str, Any]] = []
    elapsed_seconds = 0.0
    minimum_samples = int(config["minimum_samples"])
    minimum_seconds = float(config["minimum_seconds"])
    while len(samples) < minimum_samples or elapsed_seconds < minimum_seconds:
        outputs, sample = _measure_once(primary, payload)
        compare_outputs(baseline, outputs, payload, exact_float=True)
        samples.append(sample)
        elapsed_seconds += sample["wall_ms"] / 1_000
        if len(samples) >= 100_000:
            raise BenchmarkProtocolError("sample safety limit reached")

    primary_post = normalize_outputs(primary.run(payload), payload)
    oracle_post = normalize_outputs(oracle.run(payload), payload)
    compare_outputs(baseline, primary_post, payload, exact_float=True)
    compare_outputs(oracle_post, primary_post, payload)
    compare_outputs(oracle_pre, oracle_post, payload, exact_float=True)

    return {
        "implementation": config["implementation"],
        "adapter_spec": config["adapter_spec"],
        "adapter_name": primary.name,
        "oracle_adapter_spec": config["oracle_adapter_spec"],
        "profile_alias": context.profile_alias,
        "build_label": context.build_label,
        "thread_settings": thread_settings,
        "process_id": os.getpid(),
        "environment": environment_metadata(),
        "case_id": payload.case_id,
        "release_name": payload.release_name,
        "boundary": payload.boundary,
        "cache_mode": payload.cache_mode,
        "input_fingerprint": payload.input_fingerprint,
        "output_signature": output_signature(baseline),
        "conformance": {
            "pre_measurement": "pass",
            "all_measured_iterations_stable": True,
            "post_measurement": "pass",
            "float_atol": payload.float_atol,
        },
        "warmups": int(config["warmups"]),
        "minimum_samples": minimum_samples,
        "minimum_seconds": minimum_seconds,
        "samples": samples,
        "summary": summarize_samples(samples),
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
