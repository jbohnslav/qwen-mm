"""Measure installed-wheel video preprocessing and MP4 preparation separately.

The official route uses pinned qwen-vl-utils list handling or its decoded-frame
geometry plus TorchVision and AutoProcessor. File decode is independent PyAV:
TorchVision 0.28 removed the file reader used by qwen-vl-utils 0.0.14.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from PIL import Image
from video_oracle import (
    PROFILES,
    ROOT,
    Case,
    compare_arrays,
    conversation,
    decode_mp4,
    load_processors,
    make_frames,
    official_inputs,
    request,
    runtime_identity,
    torch,
    video,
    write_mp4,
)


def measure_pair(official: Any, native: Any, *, repeats: int, warmup: int) -> dict:
    calls = {"official": official, "qwen_mm": native}
    for _ in range(warmup):
        for call in calls.values():
            call()
    samples: dict[str, list[float]] = {key: [] for key in calls}
    for repetition in range(repeats):
        # Alternate order to avoid assigning warm-cache/thermal effects to one lane.
        order = tuple(calls) if repetition % 2 == 0 else tuple(reversed(calls))
        for name in order:
            started = time.perf_counter_ns()
            output = calls[name]()
            samples[name].append((time.perf_counter_ns() - started) / 1e6)
            del output
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return {
        "samples_ms": samples,
        "median_ms": medians,
        "speedup": medians["official"] / medians["qwen_mm"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "reference/.cache/huggingface")
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--thread-budget", type=int, default=2)
    parser.add_argument("--no-files", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/tmp/qwen-mm-video-benchmark.json"))
    args = parser.parse_args()
    if min(args.frames, args.height, args.width, args.repetitions, args.thread_budget) <= 0:
        parser.error("frames, dimensions, repetitions and thread budget must be positive")
    if args.warmup < 0 or args.height % 64 or args.width % 64 or args.frames % 2:
        parser.error("use nonnegative warmup, even frame count and dimensions divisible by 64")
    profiles = args.profiles.split(",")
    if not set(profiles) <= set(PROFILES):
        parser.error(f"profiles must be selected from {PROFILES}")
    identity = runtime_identity()
    torch.set_num_threads(args.thread_budget)
    torch.set_num_interop_threads(1)
    rgb = make_frames(args.frames, args.height, args.width)
    pixels = args.height * args.width
    geometry = {"min_pixels": pixels, "max_pixels": pixels}
    workloads = [
        Case("decoded_no_resize", [request(conversation(video(rgb, **geometry)))]),
        Case(
            "decoded_resize",
            [
                request(
                    conversation(
                        video(rgb, resized_height=args.height + 32, resized_width=args.width + 64)
                    )
                )
            ],
        ),
        Case(
            "pillow_frame_list",
            [request(conversation(video([Image.fromarray(frame) for frame in rgb], **geometry)))],
        ),
    ]
    report = {
        "schema_id": "qwen-mm-video-benchmark-v1",
        "runtime": identity,
        "protocol": {
            "frames": args.frames,
            "height": args.height,
            "width": args.width,
            "repetitions": args.repetitions,
            "warmup": args.warmup,
            "thread_budget": args.thread_budget,
            "torch_threads": torch.get_num_threads(),
            "timing": "perf_counter_ns; alternating lane order; all outputs materialized",
            "scope": "whole prepare_batch including media adaptation, prompt rendering and tokenization",
            "file_reference": "independent PyAV full RGB decode with its default FFmpeg threading, pinned Qwen sampling/sizing, then official TorchVision/Transformers; TorchCodec performance is unmeasured",
            "file_native": "PyAV sampled-frame RGB conversion and native preprocessing with the configured processor thread budget",
            "claims": "diagnostic timings for these generated CPU workloads on this host; no inference or general video speed guarantee",
        },
        "results": [],
    }
    with tempfile.TemporaryDirectory(prefix="qwen-mm-video-benchmark-") as directory:
        if not args.no_files:
            path = Path(directory) / "generated.mp4"
            file_rgb = make_frames(max(36, args.frames * 4), args.height, args.width)
            write_mp4(path, file_rgb, fps=12)
            report["fixture"] = {
                "codec": "libx264rgb lossless",
                "fps": 12,
                "frames": len(file_rgb),
                "bytes": path.stat().st_size,
                "rgb_sha256": hashlib.sha256(file_rgb.tobytes()).hexdigest(),
            }
            workloads.append(
                Case(
                    "mp4_prepare_including_decode",
                    [
                        request(
                            conversation(
                                video(path, video_backend="pyav", nframes=args.frames, **geometry)
                            )
                        )
                    ],
                )
            )
            # Decoder cost is a diagnostic, measured independently and never
            # subtracted from end-to-end timings or presented as native decode.
            for _ in range(args.warmup):
                decode_mp4(path)
            decode_samples = []
            for _ in range(args.repetitions):
                started = time.perf_counter_ns()
                decode_mp4(path)
                decode_samples.append((time.perf_counter_ns() - started) / 1e6)
            report["pyav_full_decode_only"] = {
                "samples_ms": decode_samples,
                "median_ms": statistics.median(decode_samples),
                "scope": "full RGB decode; reference diagnostic, no resizing or tokenization",
            }
        for profile in profiles:
            native, official = load_processors(profile, args.cache_dir, args.thread_budget)
            for case in workloads:
                expected, routes = official_inputs(official, case)
                prepared = native.prepare_batch(case.requests)
                compare_arrays(expected, prepared)
                elapsed = measure_pair(
                    lambda c=case, p=official: official_inputs(p, c)[0],
                    lambda c=case, p=native: p.prepare_batch(c.requests),
                    repeats=args.repetitions,
                    warmup=args.warmup,
                )
                result = {
                    "profile": profile,
                    "case": case.name,
                    "reference_routes": routes,
                    "parity_passed": True,
                    "output_bytes": sum(value.nbytes for value in prepared.values()),
                    **elapsed,
                }
                report["results"].append(result)
                print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Benchmark report: {args.output}")


if __name__ == "__main__":
    main()
