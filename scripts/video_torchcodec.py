"""Verify an installed wheel with real TorchCodec decoders and FrameBatch objects.

Install the pinned reference dependencies and a TorchCodec version compatible
with Torch in a separate environment before running this optional witness.
Generated lossless MP4 frames are checked independently before processor parity.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torchcodec.decoders import VideoDecoder
from video_oracle import (
    PROFILES,
    ROOT,
    Case,
    compare_arrays,
    conversation,
    load_processors,
    make_frames,
    official_inputs,
    request,
    runtime_identity,
    video,
    write_mp4,
)


def source_witness(decoder: Any, rgb: np.ndarray, indices: list[int], order: str) -> Any:
    batch = decoder.get_frames_at(indices=indices)
    pixels = batch.data.numpy()
    if order == "NCHW":
        pixels = pixels.transpose(0, 2, 3, 1)
    np.testing.assert_array_equal(pixels, rgb[indices])
    np.testing.assert_allclose(
        batch.pts_seconds.numpy(), np.asarray(indices) / 12, rtol=0, atol=1e-12
    )
    return batch


def small_layout_cases(
    native: Any, official: Any, profile: str, directory: Path, thread_budget: int
) -> list[dict]:
    from qwen_mm._media import _ReadBudget
    from qwen_mm._video import normalize_video

    cases = []
    sizing = {"resized_height": 64, "resized_width": 64}
    indices = [0, 2, 3, 5]
    batch_indices = [0, 2, 5]
    for width in (32, 3):
        rgb = make_frames(6, 3, width)
        path = directory / f"height3-width{width}.mp4"
        write_mp4(path, rgb, fps=12)
        fixture = {
            "source_shape": list(rgb.shape),
            "source_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "fixture_mp4_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        metadata = {"fps": 12, "frames_indices": indices, "total_num_frames": 6}
        expected_source = (torch.from_numpy(rgb[indices]).permute(0, 3, 1, 2), metadata)
        reference = Case(
            "small_layout_reference", [request(conversation(video(expected_source, **sizing)))]
        )
        expected_decoder, _ = official_inputs(official, reference)
        for order in ("NCHW", "NHWC"):
            decoder = VideoDecoder(
                str(path), dimension_order=order, num_ffmpeg_threads=thread_budget
            )
            batch = source_witness(decoder, rgb, batch_indices, order)
            for kind, source, expected_rgb in (
                ("decoder", decoder, rgb[indices]),
                ("framebatch", batch, rgb[batch_indices]),
            ):
                options = {**sizing, **({"nframes": 4} if kind == "decoder" else {})}
                normalized, _ = normalize_video(
                    source,
                    options=options,
                    limits={},
                    budget=_ReadBudget(1 << 26, 1 << 30),
                    threads=thread_budget,
                )
                np.testing.assert_array_equal(np.stack(normalized["frames"]), expected_rgb)
                if kind == "decoder":
                    expected = expected_decoder
                    timestamps = [1 / 12, 4 / 12]
                else:
                    expected_batch = SimpleNamespace(
                        data=torch.from_numpy(expected_rgb).permute(0, 3, 1, 2),
                        pts_seconds=batch.pts_seconds,
                    )
                    reference = Case(
                        "small_framebatch_reference",
                        [request(conversation(video(expected_batch, **sizing)))],
                    )
                    expected, _ = official_inputs(official, reference)
                    timestamps = [1 / 12, 5 / 12]
                actual = native.prepare_batch([request(conversation(video(source, **options)))])
                differences = compare_arrays(expected, actual)
                prepared = actual.metadata["videos"][0]
                np.testing.assert_allclose(prepared["timestamps"], timestamps, rtol=0, atol=1e-12)
                cases.append(
                    {
                        "profile": profile,
                        "case": f"height3_width{width}_{kind}_{order}",
                        "passed": True,
                        "source_rgb_exact": True,
                        **fixture,
                        **differences,
                        "metadata": prepared,
                    }
                )
        options = {**sizing, "nframes": 4, "video_backend": "torchcodec"}
        normalized, _ = normalize_video(
            str(path),
            options=options,
            limits={},
            budget=_ReadBudget(1 << 26, 1 << 30),
            threads=thread_budget,
        )
        np.testing.assert_array_equal(np.stack(normalized["frames"]), rgb[indices])
        actual = native.prepare_batch([request(conversation(video(str(path), **options)))])
        differences = compare_arrays(expected_decoder, actual)
        cases.append(
            {
                "profile": profile,
                "case": f"height3_width{width}_explicit_torchcodec_file",
                "passed": True,
                "source_rgb_exact": True,
                **fixture,
                **differences,
                "metadata": actual.metadata["videos"][0],
            }
        )
    return cases


def run_suite(cache: Path, *, thread_budget: int = 2) -> dict:
    from qwen_mm._media import _ReadBudget
    from qwen_mm._video import normalize_video

    identity = runtime_identity()
    identity["versions"]["torchcodec"] = importlib.metadata.version("torchcodec")
    identity["versions"]["av"] = importlib.metadata.version("av")
    rgb = make_frames(36, 128, 192)
    indices = [6, 10, 13, 17, 20, 24]
    metadata = {"fps": 12, "frames_indices": indices, "total_num_frames": 19}
    sizing = {"resized_height": 128, "resized_width": 192}
    sampling = {"nframes": 6, "video_start": 0.5, "video_end": 2.0}
    options = {**sizing, **sampling}
    expected_source = (torch.from_numpy(rgb[indices]).permute(0, 3, 1, 2), metadata)
    results = []
    with tempfile.TemporaryDirectory(prefix="qwen-mm-torchcodec-") as temporary:
        path = Path(temporary) / "lossless clip.mp4"
        write_mp4(path, rgb, fps=12)
        file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        for profile in PROFILES:
            native, official = load_processors(profile, cache, thread_budget)
            case = Case(
                "reference_clip",
                [request(conversation(video(expected_source, **sizing)))],
            )
            expected_clip, _ = official_inputs(official, case)
            for order in ("NCHW", "NHWC"):
                decoder = VideoDecoder(
                    str(path), dimension_order=order, num_ffmpeg_threads=thread_budget
                )
                source_witness(decoder, rgb, indices, order)
                normalized, _ = normalize_video(
                    decoder,
                    options=options,
                    limits={},
                    budget=_ReadBudget(1 << 26, 1 << 30),
                    threads=thread_budget,
                )
                np.testing.assert_array_equal(np.stack(normalized["frames"]), rgb[indices])
                assert normalized["frames_indices"] == indices
                assert normalized["total_num_frames"] == 19
                actual = native.prepare_batch([request(conversation(video(decoder, **options)))])
                differences = compare_arrays(expected_clip, actual)
                prepared = actual.metadata["videos"][0]
                np.testing.assert_allclose(
                    prepared["timestamps"], [8 / 12, 15 / 12, 22 / 12], rtol=0, atol=1e-12
                )
                assert prepared["fps"] == 12 and prepared["total_num_frames"] == 19
                assert prepared["frames_indices"] == indices
                np.testing.assert_allclose(prepared["sample_fps"], 6 / 19 * 12, rtol=0)
                results.append(
                    {
                        "profile": profile,
                        "case": f"decoder_{order}",
                        "passed": True,
                        "source_rgb_exact": True,
                        **differences,
                        "metadata": prepared,
                    }
                )

                batch = source_witness(decoder, rgb, [3, 8, 35], order)
                expected_batch = SimpleNamespace(
                    data=torch.from_numpy(rgb[[3, 8, 35]]).permute(0, 3, 1, 2),
                    pts_seconds=batch.pts_seconds,
                )
                batch_case = Case(
                    "reference_framebatch",
                    [request(conversation(video(expected_batch, **sizing)))],
                )
                expected, _ = official_inputs(official, batch_case)
                actual = native.prepare_batch([request(conversation(video(batch, **sizing)))])
                differences = compare_arrays(expected, actual)
                prepared = actual.metadata["videos"][0]
                np.testing.assert_allclose(
                    prepared["timestamps"], [11 / 24, 35 / 12], rtol=0, atol=1e-12
                )
                results.append(
                    {
                        "profile": profile,
                        "case": f"odd_framebatch_{order}",
                        "passed": True,
                        "source_rgb_exact": True,
                        **differences,
                        "metadata": prepared,
                    }
                )

            actual = native.prepare_batch(
                [request(conversation(video(str(path), video_backend="torchcodec", **options)))]
            )
            differences = compare_arrays(expected_clip, actual)
            prepared = actual.metadata["videos"][0]
            assert prepared["frames_indices"] == indices and prepared["total_num_frames"] == 19
            np.testing.assert_allclose(
                prepared["timestamps"], [8 / 12, 15 / 12, 22 / 12], rtol=0, atol=1e-12
            )
            results.append(
                {
                    "profile": profile,
                    "case": "explicit_torchcodec_file",
                    "passed": True,
                    **differences,
                    "metadata": prepared,
                }
            )
            results.extend(
                small_layout_cases(native, official, profile, Path(temporary), thread_budget)
            )
    return {
        "schema_id": "qwen-mm-video-torchcodec-v1",
        "passed": True,
        "runtime": identity,
        "protocol": {
            "source_frames": 36,
            "source_height": 128,
            "source_width": 192,
            "source_fps": 12,
            "source_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "fixture_mp4_sha256": file_sha256,
            "sampling": sampling,
            "sizing": sizing,
            "sampled_absolute_indices": indices,
            "clip_total_num_frames": 19,
            "native_thread_budget": thread_budget,
            "ffmpeg_threads_per_decoder": thread_budget,
            "torch_threads": torch.get_num_threads(),
            "timestamp_atol": 1e-12,
            "ambiguous_layout_sources": "6 RGB frames of size 3x32 and 3x3; resized to 64x64",
            "scope": "real CPU TorchCodec decoder, FrameBatch and file backend; both profiles",
        },
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "reference/.cache/huggingface")
    parser.add_argument("--output", type=Path, default=Path("/tmp/qwen-mm-torchcodec.json"))
    parser.add_argument("--thread-budget", type=int, default=2)
    args = parser.parse_args()
    if args.thread_budget <= 0:
        parser.error("thread budget must be positive")
    torch.set_num_threads(args.thread_budget)
    torch.set_num_interop_threads(1)
    report = run_suite(args.cache_dir, thread_budget=args.thread_budget)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Passed {len(report['cases'])} real TorchCodec profile/case checks; report: {args.output}"
    )


if __name__ == "__main__":
    main()
