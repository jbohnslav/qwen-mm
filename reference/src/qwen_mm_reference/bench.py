from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

from .fixtures import fixture_directory, manifest_path, repository_root, sha256_file, verify


PACKAGE_NAMES = (
    "transformers",
    "tokenizers",
    "qwen-vl-utils",
    "Pillow",
    "numpy",
    "torch",
    "torchvision",
    "av",
)
ARTIFACT_NAMES = (
    "chat_template.json",
    "chat_template.jinja",
    "config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)
THREAD_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def reference_root() -> Path:
    return Path(__file__).resolve().parents[2]


def hugging_face_cache() -> Path:
    return reference_root() / ".cache" / "huggingface"


def load_models() -> dict[str, dict[str, str]]:
    path = reference_root() / "models.json"
    return json.loads(path.read_text(encoding="utf-8"))


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: Iterable[float]) -> dict[str, float]:
    materialized = list(values)
    return {
        "min_ms": round(min(materialized), 3),
        "median_ms": round(statistics.median(materialized), 3),
        "p90_ms": round(percentile(materialized, 0.90), 3),
        "max_ms": round(max(materialized), 3),
    }


def sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def array_metadata(value: Any, include_hash: bool = True) -> dict[str, Any]:
    array = np.asarray(value)
    metadata: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "nbytes": int(array.nbytes),
    }
    if include_hash:
        metadata["sha256"] = sha256_array(array)
    return metadata


def output_signature(outputs: Mapping[str, Any]) -> dict[str, Any]:
    signature: dict[str, Any] = {}
    for name in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
        if name in outputs:
            signature[name] = array_metadata(outputs[name])

    if "input_ids" in outputs:
        signature["token_count"] = int(np.asarray(outputs["input_ids"]).size)
    if "image_grid_thw" in outputs:
        signature["image_grid_thw_values"] = np.asarray(
            outputs["image_grid_thw"]
        ).tolist()
    return signature


def split_video_metadata(videos: Any) -> tuple[Any, Any]:
    if videos is None:
        return None, None
    if len(videos) == 0:
        return videos, None
    first = videos[0]
    if isinstance(first, tuple) and len(first) == 2:
        video_values, video_metadata = zip(*videos)
        return list(video_values), list(video_metadata)
    return videos, None


def bind_messages(encoded_images: list[bytes]) -> list[dict[str, Any]]:
    images = [Image.open(io.BytesIO(data)) for data in encoded_images]
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": "Describe each image briefly."})
    return [{"role": "user", "content": content}]


def run_once(
    processor: Any,
    encoded_images: list[bytes],
) -> tuple[Mapping[str, Any], dict[str, float]]:
    total_start = time.perf_counter_ns()

    stage_start = total_start
    messages = bind_messages(encoded_images)
    after_bind = time.perf_counter_ns()

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    after_template = time.perf_counter_ns()

    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    after_vision = time.perf_counter_ns()

    video_values, video_metadata = split_video_metadata(videos)
    processor_kwargs: dict[str, Any] = {
        "text": [text],
        "images": images,
        "videos": video_values,
        "padding": True,
        "do_resize": False,
        "return_tensors": "np",
    }
    if video_metadata is not None:
        processor_kwargs["video_metadata"] = video_metadata
    if video_kwargs:
        processor_kwargs.update(video_kwargs)
    outputs = processor(**processor_kwargs)
    total_end = time.perf_counter_ns()

    to_ms = 1.0 / 1_000_000.0
    timings = {
        "bind": (after_bind - stage_start) * to_ms,
        "chat_template": (after_template - after_bind) * to_ms,
        "process_vision_info": (after_vision - after_template) * to_ms,
        "hf_processor": (total_end - after_vision) * to_ms,
        "total": (total_end - total_start) * to_ms,
    }
    return outputs, timings


def load_fixture_bytes(count: int) -> list[bytes]:
    manifest = verify()
    selected = manifest["images"][:count]
    return [(fixture_directory() / entry["filename"]).read_bytes() for entry in selected]


def artifact_hashes(model_id: str, revision: str) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for filename in ARTIFACT_NAMES:
        try:
            path = Path(
                hf_hub_download(
                    model_id,
                    filename,
                    revision=revision,
                    cache_dir=hugging_face_cache(),
                )
            )
        except Exception as error:
            artifacts[filename] = {"available": False, "error": type(error).__name__}
            continue
        artifacts[filename] = {
            "available": True,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return artifacts


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def mac_machine_metadata() -> dict[str, Any]:
    if platform.system() != "Darwin":
        return {}
    try:
        completed = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "SPSoftwareDataType", "-json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        data = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return {}

    hardware = (data.get("SPHardwareDataType") or [{}])[0]
    software = (data.get("SPSoftwareDataType") or [{}])[0]
    return {
        "machine_name": hardware.get("machine_name"),
        "machine_model": hardware.get("machine_model"),
        "chip": hardware.get("chip_type"),
        "cores": hardware.get("number_processors"),
        "memory": hardware.get("physical_memory"),
        "os_version": software.get("os_version"),
        "kernel_version": software.get("kernel_version"),
    }


def environment_metadata() -> dict[str, Any]:
    return {
        "machine": {
            **mac_machine_metadata(),
            "system": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "packages": package_versions(),
        "threads": {
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "environment": {
                name: os.environ.get(name) for name in THREAD_ENVIRONMENT_NAMES
            },
        },
    }


def benchmark_case(
    processor: Any,
    encoded_images: list[bytes],
    warmups: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        outputs, _ = run_once(processor, encoded_images)
        del outputs
        gc.collect()

    samples: list[dict[str, float]] = []
    signature: dict[str, Any] | None = None
    for _ in range(iterations):
        outputs, timings = run_once(processor, encoded_images)
        samples.append(timings)
        if signature is None:
            signature = output_signature(outputs)
        del outputs
        gc.collect()

    stage_names = samples[0].keys()
    return {
        "image_count": len(encoded_images),
        "input_bytes": sum(len(value) for value in encoded_images),
        "warmups": warmups,
        "iterations": iterations,
        "timing": {
            stage: summarize(sample[stage] for sample in samples)
            for stage in stage_names
        },
        "samples_ms": [
            {name: round(value, 3) for name, value in sample.items()}
            for sample in samples
        ],
        "output": signature,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="qwen3-vl-8b,qwen3.5-9b",
        help="Comma-separated aliases from reference/models.json",
    )
    parser.add_argument(
        "--cases",
        default="image1,image24",
        help="Comma-separated cases: image1,image24",
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.warmups < 0 or args.iterations <= 0:
        parser.error("warmups must be non-negative and iterations must be positive")

    models = load_models()
    selected_models = parse_csv(args.models)
    selected_cases = parse_csv(args.cases)
    case_counts = {"image1": 1, "image24": 24}

    unknown_models = sorted(set(selected_models) - models.keys())
    unknown_cases = sorted(set(selected_cases) - case_counts.keys())
    if unknown_models:
        parser.error(f"unknown model aliases: {', '.join(unknown_models)}")
    if unknown_cases:
        parser.error(f"unknown cases: {', '.join(unknown_cases)}")

    fixture_manifest = verify()
    result: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "boundary": "in-memory encoded JPEG bytes and messages to materialized CPU NumPy arrays",
        "excluded": [
            "network",
            "filesystem reads",
            "imports",
            "processor initialization",
            "model weights",
            "model execution",
            "GPU transfer",
            "output hashing",
        ],
        "environment": environment_metadata(),
        "fixture": {
            "manifest": str(manifest_path().relative_to(repository_root())),
            "manifest_sha256": sha256_file(manifest_path()),
            "count": fixture_manifest["count"],
            "width": fixture_manifest["width"],
            "height": fixture_manifest["height"],
        },
        "models": {},
    }

    for alias in selected_models:
        model = models[alias]
        model_id = model["model_id"]
        revision = model["revision"]
        print(f"loading {alias}: {model_id}@{revision}", flush=True)
        processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=hugging_face_cache(),
        )
        model_result: dict[str, Any] = {
            "model_id": model_id,
            "revision": revision,
            "processor_class": type(processor).__name__,
            "image_processor_class": type(processor.image_processor).__name__,
            "image_processor_backend": getattr(
                processor.image_processor, "backend", None
            ),
            "image_patch_size": processor.image_processor.patch_size,
            "artifacts": artifact_hashes(model_id, revision),
            "cases": {},
        }
        for case in selected_cases:
            print(f"benchmarking {alias}/{case}", flush=True)
            encoded_images = load_fixture_bytes(case_counts[case])
            model_result["cases"][case] = benchmark_case(
                processor,
                encoded_images,
                warmups=args.warmups,
                iterations=args.iterations,
            )
            total = model_result["cases"][case]["timing"]["total"]
            print(
                f"  total median={total['median_ms']:.3f} ms "
                f"p90={total['p90_ms']:.3f} ms",
                flush=True,
            )
        result["models"][alias] = model_result
        del processor
        gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
