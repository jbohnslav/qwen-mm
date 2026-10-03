"""Generated-fixture, installed-wheel video parity against the pinned upstream route.

Run with a wheel installed in a Python 3.11 environment containing the reference
dependencies. No model weights, downloads, checked-in movies, or GPU are needed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.request import url2pathname

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils.vision_process import (
    calculate_video_frame_range,
    fetch_image,
    fetch_video,
    smart_nframes,
    smart_resize,
)
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("qwen3-vl-8b", "qwen3.5-9b")
PINNED = {"transformers": "5.14.1", "qwen-vl-utils": "0.0.14"}
SCHEMA_ID = "qwen-mm-video-oracle-v1"
VIDEO_ATOL = 2.0e-6


@dataclass
class Case:
    name: str
    requests: list[dict[str, Any]]
    padding_side: str = "right"
    image_resize: bool = False


class FixtureDecoder:
    """Deterministic CPU duck of TorchCodec's selective VideoDecoder contract."""

    def __init__(
        self, frames: np.ndarray, fps: float = 12.0, total_frames: int | None = None
    ) -> None:
        self.frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        self.metadata = SimpleNamespace(
            average_fps=fps,
            num_frames=len(frames) if total_frames is None else total_frames,
            height=frames.shape[1],
            width=frames.shape[2],
        )

    def get_frames_at(self, *, indices: list[int]) -> SimpleNamespace:
        return SimpleNamespace(
            data=self.frames[np.asarray(indices) % len(self.frames)],
            pts_seconds=torch.tensor(indices, dtype=torch.float64) / self.metadata.average_fps,
        )


def make_frames(count: int, height: int = 128, width: int = 192) -> np.ndarray:
    """Spatial/channel/temporal signatures make reordering mistakes observable."""
    y, x = np.indices((height, width), dtype=np.int32)
    frames = []
    for frame in range(count):
        rgb = np.stack(
            (
                (3 * x + y + frame * 17) % 256,
                (x + 5 * y + frame * 43) % 256,
                (7 * x + 2 * y + frame * 79) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        rgb[:8, :8] = ((frame * 31) % 256, 101, 233)
        frames.append(rgb)
    return np.stack(frames)


def conversation(*content: dict[str, Any], text: str = "Describe the motion.") -> list[dict]:
    return [{"role": "user", "content": [*content, {"type": "text", "text": text}]}]


def request(messages: list[dict], **sources: Any) -> dict:
    return {
        "messages": messages,
        "options": {"add_generation_prompt": True},
        **sources,
    }


def video(source: Any, **options: Any) -> dict:
    return {"type": "video", "video": source, **options}


def write_mp4(path: Path, frames: np.ndarray, fps: int = 12) -> None:
    """Encode deterministic RGB frames losslessly in an ordinary MP4 container."""
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264rgb", rate=fps)
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "rgb24"
        stream.options = {"crf": "0", "preset": "ultrafast", "bf": "0"}
        for rgb in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def decode_mp4(path: Any) -> tuple[torch.Tensor, dict]:
    """Independent PyAV oracle decode; never calls qwen_mm's media adapter."""
    import av

    source = (
        io.BytesIO(bytes(path)) if isinstance(path, (bytes, bytearray, memoryview)) else str(path)
    )
    with av.open(source) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    data = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
    return data, {"fps": fps, "total_num_frames": len(frames)}


def _video_resize(data: torch.Tensor, item: dict) -> torch.Tensor:
    # These are the pinned qwen-vl-utils fetch_video budgets; invoke its official
    # smart_resize and TorchVision kernel instead of the native implementation.
    count, _, height, width = data.shape
    minimum = item.get("min_pixels", 128 * 32**2)
    total = item.get("total_pixels", 128000 * 32**2 * 0.9)
    ceiling = max(min(768 * 32**2, total / count * 2), int(minimum * 1.05))
    maximum = min(item.get("max_pixels", ceiling), ceiling)
    if "resized_height" in item and "resized_width" in item:
        shape = smart_resize(item["resized_height"], item["resized_width"], factor=32)
    else:
        shape = smart_resize(height, width, factor=32, min_pixels=minimum, max_pixels=maximum)
    return resize(data, list(shape), InterpolationMode.BICUBIC, antialias=True).float()


def reference_video(source: Any, item: dict) -> tuple[torch.Tensor, dict, str]:
    """Use the official list route, or official operations on decoded sources."""
    if isinstance(source, (Path, str, bytes, bytearray, memoryview)):
        path = (
            url2pathname(source[7:])
            if isinstance(source, str) and source.startswith("file://")
            else source
        )
        data, metadata = decode_mp4(path)
        start, end, count = calculate_video_frame_range(item, len(data), metadata["fps"])
        nframes = (
            1
            if count == 1 and "nframes" not in item
            else smart_nframes(item, count, metadata["fps"])
        )
        indices = torch.linspace(start, end, nframes).round().long()
        metadata.update(frames_indices=indices.tolist(), total_num_frames=count)
        return _video_resize(data[indices], item), metadata, "pyav+pinned-qwen-sampling"

    if hasattr(source, "get_frames_at") and hasattr(source, "metadata"):
        info = source.metadata
        start, end, count = calculate_video_frame_range(item, info.num_frames, info.average_fps)
        nframes = smart_nframes(item, count, info.average_fps)
        indices = torch.linspace(start, end, nframes).round().long().tolist()
        data = source.get_frames_at(indices=indices).data
        if data.shape[-1] == 3:
            data = data.permute(0, 3, 1, 2)
        metadata = {"fps": info.average_fps, "frames_indices": indices, "total_num_frames": count}
        return _video_resize(data, item), metadata, "torchcodec-decoder-contract"

    metadata = item.get("video_metadata", item.get("metadata"))
    if isinstance(source, tuple) and len(source) == 2 and isinstance(source[1], dict):
        source, metadata = source
    if isinstance(source, dict):
        metadata = {
            **(source.get("metadata", source.get("video_metadata")) or {}),
            **{
                key: source[key]
                for key in ("fps", "frames_indices", "total_num_frames")
                if key in source
            },
        }
        item = {**item, **{key: source[key] for key in ("timestamps",) if key in source}}
        source = source["frames"]
    pts = None
    if hasattr(source, "data") and hasattr(source, "pts_seconds"):
        pts = np.asarray(source.pts_seconds).tolist()
        source = source.data
    if (
        isinstance(source, (list, tuple))
        and metadata is None
        and pts is None
        and "timestamps" not in item
    ):
        output, _ = fetch_video(
            {**item, "video": source},
            image_patch_size=16,
            return_video_sample_fps=True,
            return_video_metadata=True,
        )
        data, official_metadata = output
        if metadata is not None:
            official_metadata = copy.deepcopy(metadata)
        return data, official_metadata, "qwen-vl-utils-list"

    if isinstance(source, (list, tuple)):
        source = np.stack([np.asarray(frame) for frame in source])
    if isinstance(source, np.ndarray):
        data = torch.from_numpy(np.ascontiguousarray(source)).permute(0, 3, 1, 2)
    else:
        data = source.detach().cpu()
    if pts is None:
        pts = item.get("timestamps")
    if pts is not None:
        metadata = {"fps": 1.0, "frames_indices": list(pts), "total_num_frames": len(data)}
    elif metadata is None:
        fps = item.get("raw_fps", item.get("sample_fps", 2.0))
        sample_fps = item.get("sample_fps", 2.0)
        metadata = {
            "fps": fps,
            "frames_indices": list(range(len(data))),
            "total_num_frames": len(data) / sample_fps * fps,
        }
    else:
        metadata = copy.deepcopy(metadata)
        metadata.setdefault("fps", item.get("raw_fps", item.get("sample_fps", 2.0)))
        metadata.setdefault("frames_indices", list(range(len(data))))
        metadata.setdefault("total_num_frames", len(data))
    # qwen-utils emits quantized float32 0..255; pass those already sized values
    # straight through, preserving metadata and avoiding a second resize/sample.
    prepared_tuple = data.dtype.is_floating_point
    data = data if prepared_tuple else _video_resize(data, item)
    return (
        data,
        metadata,
        "transformers-presampled" if prepared_tuple else "decoded+pinned-qwen-resize",
    )


def official_inputs(processor: Any, case: Case) -> tuple[dict[str, np.ndarray], list[str]]:
    texts, images, videos, metadata, routes = [], [], [], [], []
    for entry in case.requests:
        messages = []
        for message in entry["messages"]:
            contents = message.get("content")
            if not isinstance(contents, list):
                messages.append(message)
                continue
            normalized = []
            for content in contents:
                item = dict(content)
                kind = item.get("type")
                if kind in ("image", "video"):
                    field = "images" if kind == "image" else "videos"
                    source = item.get(kind)
                    index = item.get("input_index")
                    if isinstance(source, int):
                        index = source
                    if index is not None:
                        source = entry[field][index]
                    item[kind] = source
                    if kind == "image":
                        images.append(fetch_image(item, image_patch_size=16))
                    else:
                        data, info, route = reference_video(source, item)
                        videos.append(data)
                        metadata.append(info)
                        routes.append(route)
                normalized.append(item)
            messages.append({**message, "content": normalized})
        texts.append(processor.apply_chat_template(messages, tokenize=False, **entry["options"]))
    processor.tokenizer.padding_side = case.padding_side
    result = processor(
        text=texts,
        images=images or None,
        videos=videos or None,
        video_metadata=metadata or None,
        do_sample_frames=False,
        do_resize=False,
        padding=True,
        truncation=False,
        return_tensors="np",
    )
    return dict(result), routes


def cases(directory: Path, *, include_files: bool = True) -> list[Case]:
    rgb = make_frames(5)
    pillows = [Image.fromarray(frame) for frame in rgb]
    tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2)
    sized = {"min_pixels": 128 * 192, "max_pixels": 128 * 192}
    small = [Image.fromarray(frame) for frame in make_frames(3, 63, 95)]
    timed = {"fps": 29.97, "frames_indices": [0, 7, 19, 31, 54], "total_num_frames": 60}
    frame_batch = SimpleNamespace(
        data=tensor,
        pts_seconds=torch.tensor([0.0, 0.08, 0.23, 0.41, 0.76]),
        duration_seconds=torch.tensor([0.04] * 5),
    )
    movie = video(pillows, **sized)
    still = {"type": "image", "image": pillows[0], **sized}
    out = [
        Case("pillow_even", [request(conversation(video(pillows[:4], **sized)))]),
        Case("pillow_odd", [request(conversation(movie))]),
        Case("pillow_single", [request(conversation(video(pillows[:1], **sized)))]),
        Case("pillow_small_inner_resize", [request(conversation(video(small)))], image_resize=True),
        Case(
            "pillow_explicit_dimensions",
            [request(conversation(video(pillows, resized_height=95, resized_width=159)))],
            image_resize=True,
        ),
        Case(
            "pillow_raw_sample_rates",
            [request(conversation(video(pillows, raw_fps=30.0, sample_fps=4.0, **sized)))],
        ),
        Case("numpy_thwc_odd", [request(conversation(video(rgb, **sized)))]),
        Case("numpy_noncontiguous", [request(conversation(video(rgb[:, :, ::-1], **sized)))]),
        Case("torch_tchw_odd", [request(conversation(video(tensor, **sized)))]),
        Case("numpy_single", [request(conversation(video(rgb[:1], **sized)))]),
        Case("decoded_metadata_indices", [request(conversation(video((tensor, timed), **sized)))]),
        Case(
            "decoded_dictionary", [request(conversation(video({"frames": rgb, **timed}, **sized)))]
        ),
        Case(
            "pillow_list_explicit_metadata",
            [request(conversation(video((pillows, timed), **sized)))],
        ),
        Case("framebatch_pts", [request(conversation(video(frame_batch, **sized)))]),
        Case(
            "decoded_explicit_timestamps",
            [request(conversation(video(rgb, timestamps=[0.0, 0.15, 0.9, 1.33, 2.81], **sized)))],
        ),
        Case(
            "decoded_spatial_resize",
            [
                request(
                    conversation(
                        video(make_frames(4, 177, 259), resized_height=96, resized_width=160)
                    )
                )
            ],
        ),
        Case(
            "decoded_total_pixel_budget",
            [
                request(
                    conversation(
                        video(make_frames(6, 192, 256), min_pixels=4096, total_pixels=61440)
                    )
                )
            ],
        ),
        Case("mixed_media_order", [request(conversation(still, movie, still, movie))]),
        Case(
            "indexed_repeated_video",
            [
                request(
                    conversation(
                        {"type": "video", "input_index": 0, **sized},
                        still,
                        {"type": "video", "video": 0, **sized},
                    ),
                    videos=[tensor],
                )
            ],
        ),
    ]
    out.extend(
        [
            Case(
                "decoded_raw_sample_rates",
                [request(conversation(video(rgb, raw_fps=30.0, sample_fps=4.0, **sized)))],
            ),
            Case(
                "decoded_fps_only_metadata",
                [request(conversation(video((tensor, {"fps": 12.0}), **sized)))],
            ),
        ]
    )
    long_decoder = FixtureDecoder(make_frames(4, 32, 32), fps=30.0, total_frames=15001)
    out.append(
        Case(
            "decoder_long_float32_sampling",
            [request(conversation(video(long_decoder, min_pixels=1024, max_pixels=1024)))],
        )
    )
    out.append(
        Case(
            "video_timestamp_cache_boundary",
            [
                request(
                    conversation(
                        {
                            "type": "video",
                            "input_index": 0,
                            "timestamps": [0.0, 0.3 - 1e-10],
                            **sized,
                        },
                        {
                            "type": "video",
                            "input_index": 0,
                            "timestamps": [0.0, 0.3 + 1e-10],
                            **sized,
                        },
                    ),
                    videos=[rgb[:2]],
                )
            ],
        )
    )
    decoder = FixtureDecoder(make_frames(36))
    out.extend(
        [
            Case(
                "decoder_selective_sampling",
                [request(conversation(video(decoder, nframes=8, **sized)))],
            ),
            Case(
                "decoder_clip_sampling",
                [
                    request(
                        conversation(
                            video(decoder, video_start=0.5, video_end=2.0, nframes=6, **sized)
                        )
                    )
                ],
            ),
        ]
    )
    # A tuple exactly as emitted by process_vision_info(return_video_metadata=True).
    ready, _ = fetch_video(
        video(pillows),
        image_patch_size=16,
        return_video_metadata=True,
        return_video_sample_fps=True,
    )
    out.append(Case("qwen_utils_prepared_tuple", [request(conversation(video(ready)))]))
    custom_ready, _ = fetch_video(
        video(pillows, resized_height=64, resized_width=96),
        image_patch_size=16,
        return_video_metadata=True,
        return_video_sample_fps=True,
    )
    out.append(Case("qwen_utils_custom_sized_tuple", [request(conversation(video(custom_ready)))]))
    tiny_ready = torch.from_numpy(make_frames(4, 32, 32)).permute(0, 3, 1, 2).float()
    tiny_metadata = {"fps": 12.0, "frames_indices": [0, 3, 6, 9], "total_num_frames": 12}
    out.append(
        Case(
            "ready_tuple_32pixel_frames",
            [request(conversation(video((tiny_ready, tiny_metadata))))],
        )
    )
    batch = [
        request(conversation(movie)),
        request(conversation(still, movie, text="Longer prompt.")),
        request([{"role": "user", "content": "Short text only"}]),
    ]
    out.extend(Case(f"heterogeneous_{side}_padding", batch, side) for side in ("left", "right"))
    if include_files:
        path = directory / "generated lossless.mp4"
        write_mp4(path, make_frames(36), fps=12)
        out.extend(
            [
                Case("mp4_default_fps", [request(conversation(video(path, **sized)))]),
                Case(
                    "mp4_file_uri",
                    [request(conversation(video(path.as_uri(), nframes=8, **sized)))],
                ),
                Case(
                    "mp4_clip_absolute_timing",
                    [
                        request(
                            conversation(
                                video(path, video_start=0.5, video_end=2.0, nframes=6, **sized)
                            )
                        )
                    ],
                ),
                Case(
                    "mp4_fps_minmax",
                    [
                        request(
                            conversation(video(path, fps=1.5, min_frames=2, max_frames=4, **sized))
                        )
                    ],
                ),
                Case(
                    "mp4_rounded_nframes", [request(conversation(video(path, nframes=5, **sized)))]
                ),
            ]
        )
        out.append(
            Case(
                "mp4_encoded_bytes",
                [request(conversation(video(path.read_bytes(), nframes=6, **sized)))],
            )
        )
        for count in (1, 3):
            boundary_path = directory / f"generated-{count}-frames.mp4"
            write_mp4(boundary_path, make_frames(count), fps=12)
            out.append(
                Case(f"mp4_{count}_frames", [request(conversation(video(boundary_path, **sized)))])
            )
        frame_paths = []
        for index, frame in enumerate(rgb[:3]):
            frame_path = directory / f"frame-{index}.png"
            Image.fromarray(frame).save(frame_path)
            frame_paths.append(str(frame_path))
        out.append(Case("frame_png_paths", [request(conversation(video(frame_paths, **sized)))]))
    return out


def unpatchify_video(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    t, h, w = map(int, grid)
    blocks = values.reshape(t, h // 2, w // 2, 2, 2, 3, 2, 16, 16)
    frames = blocks.transpose(0, 6, 5, 1, 3, 7, 2, 4, 8).reshape(t * 2, 3, h * 16, w * 16)
    return np.clip(np.rint(frames * np.float32(127.5) + np.float32(127.5)), 0, 255).astype(np.uint8)


def canonical_video_frames(frames: np.ndarray) -> np.ndarray:
    """Canonical normalized model representation of a recovered RGB8 witness."""
    count, channels, height, width = frames.shape
    values = (frames.astype(np.float32) - np.float32(127.5)) / np.float32(127.5)
    blocks = values.reshape(count // 2, 2, channels, height // 32, 2, 16, width // 32, 2, 16)
    return blocks.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(-1, 1536)


def check_metadata(prepared: Any, case: Case) -> list[dict]:
    """Check full timestamps and original sampling metadata, including clips."""
    expected = []
    for entry in case.requests:
        for message in entry["messages"]:
            if not isinstance(message.get("content"), list):
                continue
            for item in message["content"]:
                if item.get("type") != "video":
                    continue
                source = item.get("video")
                index = item.get("input_index")
                if isinstance(source, int):
                    index = source
                if index is not None:
                    source = entry["videos"][index]
                data, info, _ = reference_video(source, item)
                count = len(data)
                indices = list(info["frames_indices"])
                while len(indices) % 2:
                    indices.append(indices[-1])
                frame_times = [index / info["fps"] for index in indices]
                times = [
                    (frame_times[i] + frame_times[i + 1]) / 2 for i in range(0, len(indices), 2)
                ]
                supplied_pts = (
                    "timestamps" in item
                    or hasattr(source, "pts_seconds")
                    or (isinstance(source, dict) and "timestamps" in source)
                )
                expected.append(
                    {
                        "count": count,
                        "info": info,
                        "indices": indices,
                        "times": times,
                        "supplied_pts": supplied_pts,
                    }
                )
    occurrences = prepared.metadata["videos"]
    sidecars = prepared.metadata["sidecar"]["videos"]
    assert len(occurrences) == len(expected) == len(sidecars)
    evidence = []
    for position, (wanted, actual, sidecar) in enumerate(
        zip(expected, occurrences, sidecars, strict=True)
    ):
        np.testing.assert_allclose(actual["timestamps"], wanted["times"], rtol=0, atol=1e-12)
        assert actual["grid_row"] == sidecar["grid_row"] == position
        for key in ("fps", "frames_indices", "total_num_frames", "sample_fps"):
            assert actual[key] == sidecar[key], (position, key)
        if not wanted["supplied_pts"]:
            assert actual["frames_indices"] == wanted["indices"]
            assert actual["fps"] == wanted["info"]["fps"]
            assert actual["total_num_frames"] == wanted["info"]["total_num_frames"]
            sample_fps = (
                wanted["count"] / wanted["info"]["total_num_frames"] * wanted["info"]["fps"]
            )
            np.testing.assert_allclose(actual["sample_fps"], sample_fps, rtol=1e-14, atol=1e-14)
        evidence.append(
            {
                "occurrence": position,
                "fps": actual["fps"],
                "sample_fps": actual["sample_fps"],
                "frames_indices": actual["frames_indices"],
                "total_num_frames": actual["total_num_frames"],
                "timestamps": actual["timestamps"],
            }
        )
    return evidence


def compare_arrays(reference: dict, candidate: Any, *, resized_images: bool = False) -> dict:
    assert list(candidate) == list(reference), (list(candidate), list(reference))
    differences, witnesses = {}, []
    for key, expected in reference.items():
        actual = candidate[key]
        assert actual.shape == expected.shape, (key, actual.shape, expected.shape)
        assert actual.dtype == expected.dtype, (key, actual.dtype, expected.dtype)
        assert actual.flags.c_contiguous and np.isfinite(actual).all(), key
        differences[key] = (
            float(np.max(np.abs(actual.astype(np.float64) - expected))) if actual.size else 0
        )
        if key == "pixel_values_videos" and resized_images:
            # The list path begins with the repository's documented still-image
            # resize-v2 stage. Apply its frozen per-channel gates to every frame.
            sys.path.insert(0, str(ROOT / "reference/src"))
            from qwen_mm_reference.resize_conformance_v2 import compare_prepared_occurrence

            offset = 0
            for occurrence, grid in enumerate(reference["video_grid_thw"]):
                rows = int(np.prod(grid))
                wanted = unpatchify_video(expected[offset : offset + rows], grid)
                got = unpatchify_video(actual[offset : offset + rows], grid)
                np.testing.assert_array_equal(
                    actual[offset : offset + rows],
                    canonical_video_frames(got),
                    err_msg="video normalization/patch packing must be canonical for its RGB8 witness",
                )
                for frame, (wanted_frame, got_frame) in enumerate(zip(wanted, got, strict=True)):
                    result = compare_prepared_occurrence(
                        wanted_frame.transpose(1, 2, 0),
                        got_frame.transpose(1, 2, 0),
                        occurrence=occurrence,
                    )
                    assert result["passed"], (key, occurrence, frame, result)
                    witnesses.append({"occurrence": occurrence, "frame": frame, **result})
                offset += rows
        elif key.startswith("pixel_values"):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=VIDEO_ATOL, err_msg=key)
        else:
            np.testing.assert_array_equal(actual, expected, err_msg=key)
    return {"max_absolute_difference": differences, "resize_v2_witnesses": witnesses}


def check_errors(processor: Any, directory: Path, cache: Path) -> list[dict]:
    from qwen_mm import (
        InvalidRequestError,
        MediaDecodeError,
        MediaGeometryError,
        ResourceLimitError,
        UnsupportedMediaError,
    )

    rgb = make_frames(4)
    malformed = [
        ("empty_frames", [], {}),
        ("empty_numpy", rgb[:0], {}),
        ("wrong_channels", np.zeros((4, 64, 64, 2), dtype=np.uint8), {}),
        ("wrong_rank", np.zeros((64, 64, 3), dtype=np.uint8), {}),
        ("float_unquantized", rgb.astype(np.float32) + 0.125, {}),
        ("nonpositive_rate", rgb, {"raw_fps": 0}),
        ("nan_rate", rgb, {"raw_fps": float("nan")}),
        ("both_sampling_options", rgb, {"fps": 2, "nframes": 4}),
        ("timestamp_length", rgb, {"timestamps": [0.0]}),
        ("timestamp_order", rgb, {"timestamps": [0.0, 2.0, 1.0, 3.0]}),
        (
            "metadata_length",
            (
                torch.from_numpy(rgb).permute(0, 3, 1, 2),
                {"fps": 30, "frames_indices": [0], "total_num_frames": 9},
            ),
            {},
        ),
        ("missing_file", directory / "missing.mp4", {}),
    ]
    result = []
    expected_errors = {
        "float_unquantized": UnsupportedMediaError,
        "both_sampling_options": InvalidRequestError,
        "missing_file": MediaDecodeError,
    }
    for name, source, options in malformed:
        try:
            processor.prepare(conversation(video(source, **options)))
        except (
            InvalidRequestError,
            MediaGeometryError,
            MediaDecodeError,
            ResourceLimitError,
            UnsupportedMediaError,
        ) as error:
            assert type(error) is expected_errors.get(name, MediaGeometryError), (
                name,
                type(error).__name__,
            )
            assert str(error), name
            result.append({"case": name, "passed": True, "exception": type(error).__name__})
        else:
            raise AssertionError(f"{name}: malformed input accepted")
    corrupt = directory / "corrupt.mp4"
    corrupt.write_bytes(b"this is not a video")
    try:
        processor.prepare(conversation(video(corrupt)))
    except (UnsupportedMediaError, MediaDecodeError) as error:
        result.append({"case": "corrupt_file", "passed": True, "exception": type(error).__name__})
    else:
        raise AssertionError("corrupt file accepted")
    for name, limits in (
        ("frame_limit", {"raw_frames_per_video": 3}),
        ("decoded_pixel_limit", {"decoded_pixels_per_image_or_frame": 128 * 192 - 1}),
        ("output_limit", {"materialized_output_bytes_per_batch": 1024}),
    ):
        limited, _ = load_processors(processor.profile, cache, 1, limits=limits)
        try:
            limited.prepare(conversation(video(rgb, min_pixels=128 * 192, max_pixels=128 * 192)))
        except ResourceLimitError as error:
            resource = error.context.get("limit_name", error.context.get("resource"))
            assert resource in limits, (name, error.context)
            result.append({"case": name, "passed": True, "exception": type(error).__name__})
        else:
            raise AssertionError(f"{name}: resource ceiling ignored")
    valid = request(conversation(video(rgb, min_pixels=128 * 192, max_pixels=128 * 192)))
    before = processor.prepare_batch([valid])
    try:
        processor.prepare_batch([valid, request(conversation(video([])))])
    except MediaGeometryError:
        after = processor.prepare_batch([valid])
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
        result.append({"case": "failed_batch_recovery", "passed": True})
    else:
        raise AssertionError("malformed batch accepted")
    return result


def runtime_identity(*, require_wheel: bool = True) -> dict:
    import qwen_mm
    from qwen_mm import _native

    module = Path(qwen_mm.__file__).resolve()
    if require_wheel and module.is_relative_to(ROOT / "crates"):
        raise RuntimeError(
            "video verification requires an installed wheel, not the editable checkout"
        )
    for name, version in PINNED.items():
        actual = importlib.metadata.version(name)
        if actual != version:
            raise RuntimeError(f"{name} must be {version}; got {actual}")
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "package": str(module),
        "native_module": str(Path(_native.__file__).resolve()),
        "native_sha256": hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(),
        "facade_sha256": {
            name: hashlib.sha256((module.parent / name).read_bytes()).hexdigest()
            for name in ("_processor.py", "_media.py", "_video.py", "_tensors.py")
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "qwen-mm",
                "transformers",
                "qwen-vl-utils",
                "numpy",
                "torch",
                "torchvision",
                "pillow",
            )
        },
        "optional_versions": {
            name: importlib.metadata.version(name)
            if importlib.util.find_spec(name) is not None
            else None
            for name in ("av", "torchcodec")
        },
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "logical_cpus": os.cpu_count(),
        },
    }


def load_processors(
    profile: str, cache: Path, thread_budget: int, *, limits: dict[str, int] | None = None
) -> tuple[Any, Any]:
    from qwen_mm import Processor

    record = next(p for p in Processor.supported_profiles() if p["profile"] == profile)
    snapshot = (
        cache
        / ("models--" + record["model_id"].replace("/", "--"))
        / "snapshots"
        / record["revision"]
    )
    return Processor(
        profile, snapshot, thread_budget=thread_budget, limits=limits
    ), AutoProcessor.from_pretrained(snapshot, local_files_only=True)


def run_suite(
    cache: Path,
    *,
    include_files: bool = True,
    thread_budget: int = 2,
    selected: set[str] | None = None,
    require_wheel: bool = True,
) -> dict:
    identity = runtime_identity(require_wheel=require_wheel)
    torch.set_num_threads(thread_budget)
    results = []
    with tempfile.TemporaryDirectory(prefix="qwen-mm-video-oracle-") as temporary:
        directory = Path(temporary)
        generated = cases(directory, include_files=include_files)
        if selected is not None and selected - {case.name for case in generated}:
            raise ValueError(
                f"unknown selected cases: {sorted(selected - {case.name for case in generated})}"
            )
        for profile in PROFILES:
            native, official = load_processors(profile, cache, thread_budget)
            serial, _ = load_processors(profile, cache, 1)
            for case in generated:
                if selected is not None and case.name not in selected:
                    continue
                expected, routes = official_inputs(official, case)
                prepared = native.prepare_batch(case.requests, padding_side=case.padding_side)
                evidence = compare_arrays(expected, prepared, resized_images=case.image_resize)
                evidence["video_metadata"] = check_metadata(prepared, case)
                again = native.prepare_batch(case.requests, padding_side=case.padding_side)
                for key in prepared:
                    np.testing.assert_array_equal(
                        prepared[key], again[key], err_msg=f"repeat/{key}"
                    )
                serial_output = serial.prepare_batch(case.requests, padding_side=case.padding_side)
                for key in prepared:
                    np.testing.assert_array_equal(
                        prepared[key], serial_output[key], err_msg=f"serial/{key}"
                    )
                # Torch output is model-ready and preserves the NumPy-owned native storage.
                tensor_output = native.prepare_batch(
                    case.requests, padding_side=case.padding_side, return_tensors="pt"
                )
                for key in prepared:
                    np.testing.assert_array_equal(tensor_output[key].numpy(), prepared[key])
                result = {
                    "profile": profile,
                    "case": case.name,
                    "passed": True,
                    "reference_routes": routes,
                    "shapes": {k: list(v.shape) for k, v in prepared.items()},
                    **evidence,
                }
                results.append(result)
                print(
                    json.dumps({k: v for k, v in result.items() if k != "resize_v2_witnesses"}),
                    flush=True,
                )
            if selected is None:
                shared_case = next(case for case in generated if case.name == "mixed_media_order")
                with ThreadPoolExecutor(max_workers=2) as pool:
                    concurrent = list(
                        pool.map(
                            lambda _, p=native, c=shared_case: p.prepare_batch(c.requests), range(2)
                        )
                    )
                for key in concurrent[0]:
                    np.testing.assert_array_equal(concurrent[0][key], concurrent[1][key])
                results.append(
                    {"profile": profile, "case": "shared_processor_concurrency", "passed": True}
                )
            if selected is None:
                results.extend(
                    {"profile": profile, **error}
                    for error in check_errors(native, directory, cache)
                )
    return {
        "schema_id": SCHEMA_ID,
        "passed": True,
        "runtime": identity,
        "numeric_contract": {
            "video_final_atol": VIDEO_ATOL,
            "pillow_inner_resize": "frozen still-image resize-v2 per-channel gates",
        },
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "reference/.cache/huggingface")
    parser.add_argument("--output", type=Path, default=Path("/tmp/qwen-mm-video-oracle.json"))
    parser.add_argument("--no-files", action="store_true", help="Skip optional PyAV MP4 cases")
    parser.add_argument("--cases", help="Comma-separated generated cases")
    parser.add_argument("--thread-budget", type=int, default=2)
    args = parser.parse_args()
    report = run_suite(
        args.cache_dir,
        include_files=not args.no_files,
        thread_budget=args.thread_budget,
        selected=set(args.cases.split(",")) if args.cases else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Passed {len(report['cases'])} profile/case checks; report: {args.output}")


if __name__ == "__main__":
    main()
