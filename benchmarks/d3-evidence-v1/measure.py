from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import qwen_mm._native as native_module
import qwen_mm.benchmark as native_benchmark_module
import qwen_mm_reference.benchmark_protocol as reference_protocol_module
from qwen_mm.benchmark import _binding_requests, create_adapter, create_observed_adapter
from qwen_mm_reference.benchmark_protocol import load_workload, materialize_case

IMPLEMENTATION_FILES = (
    "crates/qwen-mm-core/src/media.rs",
    "crates/qwen-mm-core/src/patchify.rs",
    "crates/qwen-mm-core/src/processor.rs",
    "crates/qwen-mm-core/src/resize.rs",
)
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
)
WARMUP_COUNT = 2
SAMPLE_COUNT = 7


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity() -> dict[str, object]:
    files = {name: sha256_path(Path(name)) for name in IMPLEMENTATION_FILES}
    aggregate = hashlib.sha256()
    for name, digest in files.items():
        aggregate.update(name.encode())
        aggregate.update(b"\0")
        aggregate.update(digest.encode())
        aggregate.update(b"\0")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()
    diff = subprocess.check_output(["git", "diff", "--binary", "--", *IMPLEMENTATION_FILES])
    return {
        "revision": revision,
        "implementation_files": files,
        "implementation_digest": aggregate.hexdigest(),
        "implementation_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def harness_identity() -> dict[str, object]:
    files = {
        "benchmarks/d3-evidence-v1/measure.py": sha256_path(Path(__file__).resolve()),
        "benchmarks/workloads-v2.json": sha256_path(Path("benchmarks/workloads-v2.json")),
        "qwen_mm.benchmark": sha256_path(Path(native_benchmark_module.__file__).resolve()),
        "qwen_mm_reference.benchmark_protocol": sha256_path(
            Path(reference_protocol_module.__file__).resolve()
        ),
    }
    aggregate = hashlib.sha256()
    for name, digest in files.items():
        aggregate.update(name.encode())
        aggregate.update(b"\0")
        aggregate.update(digest.encode())
        aggregate.update(b"\0")
    return {
        "files": files,
        "module_origins": {
            "qwen_mm.benchmark": str(Path(native_benchmark_module.__file__).resolve()),
            "qwen_mm_reference.benchmark_protocol": str(
                Path(reference_protocol_module.__file__).resolve()
            ),
        },
        "aggregate_sha256": aggregate.hexdigest(),
    }


def asset_identity(profile: str) -> dict[str, object]:
    compatibility_path = Path("reference/compatibility/v1.json")
    compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
    profile_record = compatibility["profiles"][profile]
    model_id = profile_record["model_id"]
    revision = profile_record["revision"]
    assets_root = Path(os.environ["QWEN_MM_ASSETS_ROOT"]).resolve()
    directory = assets_root / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    artifacts = profile_record["artifacts"]
    template_names = sorted(name for name in artifacts if name.startswith("chat_template."))
    names = ["tokenizer.json", "tokenizer_config.json", *template_names]
    observed = {name: sha256_path(directory / name) for name in names}
    expected = {name: artifacts[name] for name in names}
    if observed != expected:
        raise RuntimeError(f"profile assets do not match the pinned manifest: {profile}")
    return {
        "compatibility_manifest_sha256": sha256_path(compatibility_path),
        "profile_fingerprint": profile_record["fingerprint"],
        "model_id": model_id,
        "revision": revision,
        "directory": str(directory),
        "artifacts": observed,
    }


def toolchain_identity() -> dict[str, object]:
    packages = {}
    for name in ("maturin", "numpy", "qwen-mm", "qwen-mm-reference"):
        packages[name] = importlib.metadata.version(name)
    cargo_wrapper = "./scripts/with-cargo.sh"
    maturin_path = Path(sys.executable).parent / "maturin"
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "rustc": subprocess.check_output([cargo_wrapper, "rustc", "-Vv"], text=True).strip(),
        "cargo": subprocess.check_output([cargo_wrapper, "cargo", "-V"], text=True).strip(),
        "maturin_executable": str(maturin_path.resolve()),
        "maturin_executable_sha256": sha256_path(maturin_path),
        "packages": packages,
        "rustflags": os.environ.get("RUSTFLAGS"),
        "cargo_build_target": os.environ.get("CARGO_BUILD_TARGET"),
    }


def official_output_signature(outputs: object) -> dict[str, object]:
    digest = hashlib.sha256()
    arrays: dict[str, object] = {}
    for name in sorted(outputs):
        array = outputs[name]
        shape = [int(value) for value in array.shape]
        strides = [int(value) for value in array.strides]
        dtype = str(array.dtype)
        data = array.tobytes(order="C")
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(dtype.encode())
        digest.update(b"\0")
        digest.update(json.dumps(shape, separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(json.dumps(strides, separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(data)
        arrays[name] = {
            "shape": shape,
            "strides": strides,
            "dtype": dtype,
            "nbytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    return {"sha256": digest.hexdigest(), "arrays": arrays}


def official_metadata_sidecar(metadata: object) -> dict[str, object]:
    canonical = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "value": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("baseline", "copy-only", "no-lut", "candidate"))
    parser.add_argument("profile")
    parser.add_argument("case")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    workload_path = Path("benchmarks/workloads-v2.json")
    workload = load_workload(workload_path)
    case = next(item for item in workload["cases"] if item["case_id"] == args.case)
    payload = materialize_case(case)
    context = SimpleNamespace(
        profile_alias=args.profile,
        build_label="d3-release",
        thread_budget=1,
    )
    adapter = create_adapter(context)

    for _ in range(WARMUP_COUNT):
        output = adapter.run(payload)
        del output
        gc.collect()

    samples = []
    timed_output = None
    for sample_index in range(SAMPLE_COUNT):
        gc.collect()
        cpu_start = time.process_time_ns()
        wall_start = time.perf_counter_ns()
        output = adapter.run(payload)
        wall_end = time.perf_counter_ns()
        cpu_end = time.process_time_ns()
        if sample_index == SAMPLE_COUNT - 1:
            timed_output = output
        else:
            del output
        samples.append(
            {
                "wall_ms": (wall_end - wall_start) / 1_000_000,
                "cpu_ms": (cpu_end - cpu_start) / 1_000_000,
            }
        )

    if timed_output is None:
        raise RuntimeError("measurement did not retain a final timed output")
    output_signature = official_output_signature(timed_output)
    del timed_output
    gc.collect()

    observed = create_observed_adapter(context)
    output = observed.run(payload)
    report = observed.observation_report()
    observed_output_signature = official_output_signature(output)
    del output
    if observed_output_signature != output_signature:
        raise RuntimeError("timed and observed output signatures differ")
    measured_peak_rss_bytes = peak_rss_bytes()

    # The benchmark protocol deliberately returns only official arrays. Run one
    # untimed public Processor call to bind the complete adapter metadata and
    # sidecar contract to the same payload and verify its arrays too.
    authenticated = adapter._processor.prepare_batch(_binding_requests(payload))
    authenticated_output_signature = official_output_signature(authenticated.arrays)
    if authenticated_output_signature != output_signature:
        raise RuntimeError("timed and metadata-authentication output signatures differ")
    metadata_sidecar = official_metadata_sidecar(authenticated.metadata)

    stage_ms: dict[str, float] = {}
    for span in report["spans"]:
        stage_ms[span["name"]] = stage_ms.get(span["name"], 0.0) + (
            span["exclusive_duration_ns"] / 1_000_000
        )
    buffer_bytes: dict[str, int] = {}
    for buffer in report["buffers"]:
        buffer_bytes[buffer["name"]] = buffer_bytes.get(buffer["name"], 0) + int(buffer["bytes"])
    copy_bytes: dict[str, int] = {}
    for copy in report["copies"]:
        copy_bytes[copy["name"]] = copy_bytes.get(copy["name"], 0) + int(copy["bytes"])

    result = {
        "variant": args.variant,
        "profile_alias": args.profile,
        "case_id": args.case,
        "thread_budget": context.thread_budget,
        "thread_environment": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT},
        "sample_count": len(samples),
        "wall_ms_p50": statistics.median(sample["wall_ms"] for sample in samples),
        "cpu_ms_p50": statistics.median(sample["cpu_ms"] for sample in samples),
        "peak_rss_bytes": measured_peak_rss_bytes,
        "samples": samples,
        "protocol": {
            "measurement_script_sha256": sha256_path(Path(__file__).resolve()),
            "argv": sys.argv,
            "warmup_count": WARMUP_COUNT,
            "sample_count": SAMPLE_COUNT,
            "wall_clock": "time.perf_counter_ns",
            "cpu_clock": "time.process_time_ns",
            "workload_path": str(workload_path),
            "workload_sha256": sha256_path(workload_path),
            "input_fingerprint": payload.input_fingerprint,
            "logical_input_fingerprint": payload.logical_input_fingerprint,
        },
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "source": source_identity(),
        "harness": harness_identity(),
        "assets": asset_identity(args.profile),
        "toolchain": toolchain_identity(),
        "native_module": {
            "path": str(Path(native_module.__file__).resolve()),
            "sha256": sha256_path(Path(native_module.__file__).resolve()),
        },
        "official_output_signature": output_signature,
        "observed_output_signature": observed_output_signature,
        "authenticated_output_signature": authenticated_output_signature,
        "official_metadata_sidecar": metadata_sidecar,
        "observation": {
            "duration_ms": report["duration_ns"] / 1_000_000,
            "allocations": report["allocations"],
            "stage_exclusive_ms": stage_ms,
            "buffer_bytes": buffer_bytes,
            "copy_bytes": copy_bytes,
            "dropped_events": report["dropped_events"],
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
