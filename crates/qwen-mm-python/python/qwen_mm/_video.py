"""Video source adapters. Resize, normalization and packing stay in Rust."""

from __future__ import annotations

import io
import math
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import url2pathname

import numpy as np

from ._native import (
    InvalidRequestError,
    MediaDecodeError,
    MediaGeometryError,
    ResourceLimitError,
    UnsupportedMediaError,
)

VIDEO_OPTIONS = {
    "sample_fps",
    "raw_fps",
    "min_pixels",
    "max_pixels",
    "total_pixels",
    "resized_height",
    "resized_width",
}
SOURCE_OPTIONS = {
    "fps",
    "nframes",
    "min_frames",
    "max_frames",
    "video_start",
    "video_end",
    "video_backend",
    "timestamps",
    "video_metadata",
}


def _error(kind: type[Exception], message: str, **context: Any) -> Exception:
    error = kind(message)
    error.context = context
    return error


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(MediaGeometryError, f"{name} must be a finite positive number", field=name)
    if not math.isfinite(value) or value <= 0:
        raise _error(MediaGeometryError, f"{name} must be a finite positive number", field=name)
    return float(value)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _error(MediaGeometryError, f"{name} must be a positive integer", field=name)
    return value


def _check(name: str, actual: int, limit: int) -> None:
    if actual > limit:
        raise _error(
            ResourceLimitError,
            f"video exceeds {name}",
            limit_name=name,
            actual=actual,
            limit=limit,
        )


def _check_shape(count: int, height: int, width: int, limits: dict[str, int]) -> None:
    if min(count, height, width) <= 0:
        raise _error(MediaGeometryError, "video must contain nonempty RGB frames")
    _check("raw_frames_per_video", count, limits.get("raw_frames_per_video", 768))
    _check("decoded_edge_length", max(height, width), limits.get("decoded_edge_length", 32768))
    _check(
        "decoded_pixels_per_image_or_frame",
        height * width,
        limits.get("decoded_pixels_per_image_or_frame", 67108864),
    )


def pillow_rgb(value: Any, *, limits: dict[str, int]) -> np.ndarray:
    """Convert a Pillow object, checking dimensions before allocating RGB."""
    from PIL import Image

    _check_shape(1, value.height, value.width, limits)
    if value.mode == "RGBA":
        rgb = Image.new("RGB", value.size, (255, 255, 255))
        rgb.paste(value, mask=value.getchannel("A"))
    else:
        rgb = value.convert("RGB")
    return np.asarray(rgb)


def is_pillow(value: Any) -> bool:
    return type(value).__module__.startswith("PIL.") and hasattr(value, "convert")


def _numpy(value: Any) -> np.ndarray:
    # Importing qwen-mm or processing NumPy media never imports Torch.
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        try:
            value = value.detach().cpu().numpy()
        except (TypeError, RuntimeError) as cause:
            raise _error(
                UnsupportedMediaError, f"could not convert decoded RGB tensor: {cause}"
            ) from cause
    if not isinstance(value, np.ndarray):
        raise _error(UnsupportedMediaError, "decoded video must be a NumPy array or Torch tensor")
    return value


def _array_frames(
    value: Any, limits: dict[str, int], layout: str | None = None
) -> list[np.ndarray]:
    is_tensor = hasattr(value, "detach")
    shape = getattr(value, "shape", None)
    if shape is None:
        raise _error(UnsupportedMediaError, "decoded video must be a NumPy array or Torch tensor")
    if len(shape) != 4:
        raise _error(
            MediaGeometryError, "decoded video requires shape THWC or TCHW", shape=tuple(shape)
        )
    if layout is None:
        layout = "TCHW" if is_tensor and shape[1] == 3 else "THWC"
        if layout == "THWC" and shape[-1] != 3 and shape[1] == 3:
            layout = "TCHW"
    if layout == "TCHW":
        count, channels, height, width = shape
    elif layout == "THWC":
        count, height, width, channels = shape
    else:
        raise _error(InvalidRequestError, "video layout must be 'THWC' or 'TCHW'", layout=layout)
    if channels != 3:
        raise _error(MediaGeometryError, "decoded video requires exactly three RGB channels")
    # Validate before a CUDA -> CPU transfer or contiguous/float conversion.
    _check_shape(count, height, width, limits)
    array = _numpy(value)
    if tuple(array.shape) != tuple(shape):
        raise _error(MediaGeometryError, "decoded video shape changed during conversion")
    if layout == "TCHW":
        array = array.transpose(0, 2, 3, 1)
    if array.dtype != np.uint8:
        # qwen-vl-utils returns quantized uint8 pixels cast to float32.
        if not np.issubdtype(array.dtype, np.floating) or not np.all(
            np.isfinite(array) & (array >= 0) & (array <= 255) & (array == np.floor(array))
        ):
            raise _error(
                UnsupportedMediaError,
                "decoded video requires uint8 RGB or integral floating RGB in [0, 255]",
                dtype=str(array.dtype),
            )
        array = array.astype(np.uint8)
    # One bulk contiguous conversion, rather than a copy and transpose per frame.
    array = np.ascontiguousarray(array)
    return list(array)


def _metadata(value: Any, count: int) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _error(InvalidRequestError, "video_metadata must be a dictionary")
    result = {}
    if "fps" in value:
        result["fps"] = _positive(value["fps"], "video_metadata.fps")
    if "total_num_frames" in value:
        result["total_num_frames"] = _positive(value["total_num_frames"], "total_num_frames")
    if "frames_indices" in value:
        indices = value["frames_indices"]
        if hasattr(indices, "tolist"):
            indices = indices.tolist()
        if not isinstance(indices, (list, tuple)) or len(indices) != count:
            raise _error(MediaGeometryError, "frames_indices must have one entry per video frame")
        if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices):
            raise _error(MediaGeometryError, "frames_indices must contain nonnegative integers")
        if any(b < a for a, b in zip(indices, indices[1:], strict=False)):
            raise _error(MediaGeometryError, "frames_indices must be in presentation order")
        result["frames_indices"] = list(indices)
    return result


def _times(value: Any, count: int) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise _error(MediaGeometryError, "timestamps must have one entry per video frame")
    if any(
        isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0
        for t in value
    ):
        raise _error(MediaGeometryError, "timestamps must be finite nonnegative seconds")
    if any(b < a for a, b in zip(value, value[1:], strict=False)):
        raise _error(MediaGeometryError, "timestamps must be in presentation order")
    return [float(t) for t in value]


def _clip_range(total: int, fps: float, options: dict[str, Any]) -> tuple[int, int]:
    start, end = 0, total - 1
    for key in ("video_start", "video_end"):
        if key in options:
            value = options[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise _error(MediaGeometryError, f"{key} must be finite seconds")
            value = max(0.0, min(float(value), total / fps))
            if key == "video_start":
                start = math.ceil(value * fps)
            else:
                end = min(math.floor(value * fps), total - 1)
    if start > end or (
        start == end and total > 1 and ("video_start" in options or "video_end" in options)
    ):
        raise _error(MediaGeometryError, "video clip time range is empty or reversed")
    return start, end


def _sample_indices(
    total: int, fps: float, options: dict[str, Any], limits: dict[str, int]
) -> list[int]:
    """Pinned utils rounded-linspace sampling, with an extension for single frames."""
    total = _integer(total, "source frame count")
    fps = _positive(fps, "source fps")
    start, end = _clip_range(total, fps, options)
    available = end - start + 1
    if "fps" in options and "nframes" in options:
        raise _error(InvalidRequestError, "specify either fps or nframes, not both")
    if "nframes" in options:
        count = round(_integer(options["nframes"], "nframes") / 2) * 2
        if not 2 <= count <= available:
            raise _error(MediaGeometryError, "nframes must round to an even count within the clip")
    elif available == 1:
        count = 1
    else:
        target = _positive(options.get("fps", 2.0), "fps")
        minimum = math.ceil(_integer(options.get("min_frames", 4), "min_frames") / 2) * 2
        maximum = (_integer(options.get("max_frames", min(768, available)), "max_frames") // 2) * 2
        if maximum < 2 or (
            minimum > maximum and "min_frames" in options and "max_frames" in options
        ):
            raise _error(MediaGeometryError, "min_frames and max_frames are inconsistent")
        count = (
            math.floor(min(min(max(available / fps * target, minimum), maximum), available) / 2) * 2
        )
    _check("raw_frames_per_video", count, limits.get("raw_frames_per_video", 768))
    if count == 1:
        return [start]
    # Match ATen CPU linspace's float32 step, symmetric half construction and
    # fused multiply-add rounding; NumPy's float64 linspace picks other frames
    # on long videos. Float64 intermediates exactly represent these f32 products.
    first, last = np.float32(start), np.float32(end)
    step = np.float32((last - first) / np.float32(count - 1))
    offsets = np.arange(count, dtype=np.float64)
    values = np.where(
        offsets < count // 2,
        np.float64(first) + np.float64(step) * offsets,
        np.float64(last) - np.float64(step) * (count - 1 - offsets),
    ).astype(np.float32)
    return values.round().astype(np.int64).tolist()


def _frame_batch_layout(data: Any) -> str | None:
    shape = tuple(data.shape)
    if len(shape) == 4 and shape[1] == shape[-1] == 3:
        if hasattr(data, "stride"):
            strides = data.stride()
        else:
            strides = tuple(stride // data.itemsize for stride in data.strides)
        # TorchCodec's NCHW view keeps the underlying RGB channel stride of 1.
        if strides[1] == 1:
            return "TCHW"
        if strides[-1] == 1:
            return "THWC"
    return None


def _decoder_frames(
    decoder: Any, options: dict[str, Any], limits: dict[str, int], layout: str | None = None
) -> dict[str, Any]:
    metadata = decoder.metadata
    fps = _positive(metadata.average_fps, "decoder average_fps")
    indices = _sample_indices(metadata.num_frames, fps, options, limits)
    height, width = getattr(metadata, "height", None), getattr(metadata, "width", None)
    if height and width:
        _check_shape(len(indices), height, width, limits)
    try:
        batch = decoder.get_frames_at(indices=indices)
    except Exception as cause:
        raise _error(
            MediaDecodeError, f"video decoder could not retrieve frames: {cause}"
        ) from cause
    if layout is None and height and width:
        shape = tuple(batch.data.shape)
        if shape[1:] == (height, width, 3) and shape[1:] != (3, height, width):
            layout = "THWC"
        elif shape[1:] == (3, height, width) and shape[1:] != (height, width, 3):
            layout = "TCHW"
    frames = _array_frames(batch.data, limits, layout or _frame_batch_layout(batch.data))
    if len(frames) != len(indices):
        raise _error(MediaDecodeError, "decoder returned the wrong number of sampled frames")
    # Index/fps timestamps match the pinned Qwen metadata contract. Callers can
    # pass the FrameBatch itself to preserve presentation timestamps for VFR.
    return {
        "frames": frames,
        "fps": fps,
        "frames_indices": indices,
        "total_num_frames": float(
            _clip_range(metadata.num_frames, fps, options)[1]
            - _clip_range(metadata.num_frames, fps, options)[0]
            + 1
        ),
    }


def _pyav_frames(
    source: Any, options: dict[str, Any], limits: dict[str, int], threads: int
) -> dict[str, Any]:
    try:
        import av
    except ImportError as cause:
        raise _error(
            UnsupportedMediaError,
            "video files require PyAV; install qwen-mm[video], or select an installed TorchCodec backend",
        ) from cause
    try:
        with av.open(source) as container:
            if not container.streams.video:
                raise _error(MediaDecodeError, "video source has no video stream")
            stream = container.streams.video[0]
            stream.codec_context.thread_count = threads
            stream.codec_context.thread_type = "AUTO"
            fps = _positive(float(stream.average_rate or 0), "source fps")
            _check_shape(1, stream.height, stream.width, limits)
            total = stream.frames
            if not total:
                # Some containers omit frame counts. Count without allocating RGB.
                total = sum(1 for _ in container.decode(stream))
                container.seek(0)
            indices = _sample_indices(total, fps, options, limits)
            _check_shape(len(indices), stream.height, stream.width, limits)
            wanted = set(indices)
            selected = {}
            for index, frame in enumerate(container.decode(stream)):
                if index in wanted:
                    _check_shape(1, frame.height, frame.width, limits)
                    selected[index] = frame.to_ndarray(format="rgb24")
                if index >= indices[-1]:
                    break
            if len(selected) != len(wanted):
                raise _error(MediaDecodeError, "video ended before its declared sampled frames")
            return {
                "frames": [selected[i] for i in indices],
                "fps": fps,
                "frames_indices": indices,
                "total_num_frames": float(
                    _clip_range(total, fps, options)[1] - _clip_range(total, fps, options)[0] + 1
                ),
            }
    except (ResourceLimitError, MediaGeometryError, MediaDecodeError, InvalidRequestError):
        raise
    except Exception as cause:
        raise _error(
            MediaDecodeError, f"could not decode video: {cause}", detail=str(cause)
        ) from cause


def _file_frames(
    value: Any, options: dict[str, Any], limits: dict[str, int], budget: Any, threads: int
) -> dict[str, Any]:
    from ._media import _read_url

    source = value
    if isinstance(value, (str, os.PathLike)):
        value = os.fspath(value)
        if value.lower().startswith(("http://", "https://")):
            source = io.BytesIO(_read_url(value, budget=budget))
        else:
            if value.lower().startswith("file:"):
                parsed = urlsplit(value)
                if parsed.netloc not in ("", "localhost"):
                    raise _error(UnsupportedMediaError, "video file URLs require a local host")
                value = url2pathname(parsed.path)
            elif "://" in value:
                raise _error(UnsupportedMediaError, "unsupported video URL scheme", source=value)
            path = Path(value)
            try:
                budget.consume(path.stat().st_size, source=str(path))
            except OSError as cause:
                raise _error(
                    MediaDecodeError, f"could not read video path {path}: {cause}"
                ) from cause
            source = str(path)
    else:
        try:
            view = memoryview(value)
            budget.check(view.nbytes, source="encoded video")
            data = bytes(view)
        except TypeError as cause:
            raise _error(UnsupportedMediaError, "unsupported video source") from cause
        budget.consume(len(data), source="encoded video")
        source = io.BytesIO(data)
    backend = options.get("video_backend", "auto")
    if backend not in ("auto", "torchcodec", "pyav"):
        raise _error(InvalidRequestError, "video_backend must be 'auto', 'torchcodec', or 'pyav'")
    if backend != "pyav":
        try:
            from torchcodec.decoders import VideoDecoder
        except (ImportError, OSError, RuntimeError) as cause:
            if backend == "torchcodec":
                raise _error(
                    UnsupportedMediaError, f"TorchCodec is unavailable: {cause}"
                ) from cause
        else:
            try:
                # NHWC avoids per-frame Torch-to-RGB transposes/copies.
                decoder = VideoDecoder(source, dimension_order="NHWC", num_ffmpeg_threads=threads)
                return _decoder_frames(decoder, options, limits, "THWC")
            except (ResourceLimitError, MediaGeometryError, InvalidRequestError):
                raise
            except Exception as cause:
                raise _error(
                    MediaDecodeError, f"TorchCodec could not decode video: {cause}"
                ) from cause
    return _pyav_frames(source, options, limits, threads)


def normalize_video(
    value: Any, *, options: dict[str, Any], limits: dict[str, int], budget: Any, threads: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Produce owned-source schema and native per-occurrence options."""
    from ._media import _load_source

    if "fps" in options and "nframes" in options:
        raise _error(InvalidRequestError, "specify either fps or nframes, not both")
    if options.get("video_backend", "auto") not in ("auto", "pyav", "torchcodec"):
        raise _error(InvalidRequestError, "video_backend must be 'auto', 'torchcodec', or 'pyav'")
    for name in ("sample_fps", "raw_fps", "fps"):
        if name in options:
            _positive(options[name], name)
    native_options = {key: value for key, value in options.items() if key in VIDEO_OPTIONS}
    metadata = options.get("video_metadata")
    timestamps = options.get("timestamps")
    image_frames = False
    preprocessed = False
    plain_list = isinstance(value, (list, tuple))
    layout = None
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict):
        value, metadata = value
        plain_list = False
        dtype = getattr(value, "dtype", None)
        preprocessed = dtype is not None and "float" in str(dtype)
    if isinstance(value, dict):
        allowed = {
            "frames",
            "metadata",
            "video_metadata",
            "fps",
            "frames_indices",
            "total_num_frames",
            "timestamps",
            "image_frames",
            "preprocessed",
            "layout",
        }
        unknown = value.keys() - allowed
        if unknown or "frames" not in value:
            raise _error(
                InvalidRequestError,
                "video dictionary requires frames and supported metadata",
                fields=sorted(unknown),
            )
        metadata = value.get("metadata", value.get("video_metadata", metadata))
        if metadata is not None and not isinstance(metadata, dict):
            raise _error(InvalidRequestError, "video metadata must be a dictionary")
        metadata = {
            **(metadata or {}),
            **{k: value[k] for k in ("fps", "frames_indices", "total_num_frames") if k in value},
        }
        timestamps = value.get("timestamps", timestamps)
        image_frames = value.get("image_frames", False)
        if not isinstance(image_frames, bool):
            raise _error(InvalidRequestError, "image_frames must be a boolean")
        preprocessed = value.get("preprocessed", preprocessed)
        if not isinstance(preprocessed, bool):
            raise _error(InvalidRequestError, "preprocessed must be a boolean")
        layout = value.get("layout")
        value = value["frames"]
    if hasattr(value, "get_frames_at") and hasattr(value, "metadata"):
        normalized = _decoder_frames(value, options, limits, layout)
    elif hasattr(value, "pts_seconds") and hasattr(value, "data"):
        normalized = {
            "frames": _array_frames(value.data, limits, layout or _frame_batch_layout(value.data))
        }
        timestamps = value.pts_seconds if timestamps is None else timestamps
    elif isinstance(value, (str, os.PathLike, bytes, bytearray, memoryview)):
        normalized = _file_frames(value, options, limits, budget, threads)
    elif isinstance(value, (list, tuple)):
        if not value:
            raise _error(MediaGeometryError, "video frame list must not be empty")
        _check("raw_frames_per_video", len(value), limits.get("raw_frames_per_video", 768))
        frames = []
        for index, frame in enumerate(value):
            frame = _load_source(
                frame, source=f"video frame[{index}]", budget=budget, limits=limits
            )
            if isinstance(frame, (bytes, bytearray, memoryview)):
                try:
                    from PIL import Image

                    with Image.open(io.BytesIO(frame)) as image:
                        frame = pillow_rgb(image, limits=limits)
                except ResourceLimitError:
                    raise
                except Exception as cause:
                    raise _error(
                        MediaDecodeError, f"could not decode video frame {index}: {cause}"
                    ) from cause
            shape = getattr(frame, "shape", None)
            if shape is not None:
                if len(shape) != 3 or shape[-1] != 3:
                    raise _error(
                        MediaGeometryError, "video frame requires HWC RGB", frame_index=index
                    )
                _check_shape(1, *shape[:2], limits)
            frame = _numpy(frame)
            if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                raise _error(
                    MediaGeometryError, "video frame requires uint8 HWC RGB", frame_index=index
                )
            _check_shape(1, *frame.shape[:2], limits)
            frames.append(np.ascontiguousarray(frame))
        # Plain frame lists use the pinned utils image-list preprocessing path.
        # Explicit metadata describes already decoded/prepared clips instead.
        image_frames = image_frames or (plain_list and metadata is None and timestamps is None)
        normalized = {"frames": frames}
    else:
        normalized = {"frames": _array_frames(value, limits, layout)}
    count = len(normalized["frames"])
    normalized.update(_metadata(metadata, count))
    if timestamps is not None:
        normalized["timestamps"] = _times(timestamps, count)
    # Decoded clips are already sampled. Rate controls describe their timing;
    # source sampling controls only apply to files and decoder objects.
    if not isinstance(value, (str, os.PathLike, bytes, bytearray, memoryview)) and not hasattr(
        value, "get_frames_at"
    ):
        if any(
            key in options
            for key in ("fps", "nframes", "min_frames", "max_frames", "video_start", "video_end")
        ):
            raise _error(
                InvalidRequestError,
                "decoded clips use sample_fps/raw_fps; sampling options require a file or decoder",
            )
    if image_frames and preprocessed:
        raise _error(InvalidRequestError, "image_frames and preprocessed cannot both be true")
    if image_frames:
        normalized["image_frames"] = True
    if preprocessed and not any(
        key in options
        for key in ("min_pixels", "max_pixels", "total_pixels", "resized_height", "resized_width")
    ):
        normalized["preprocessed"] = True
    if "sample_fps" not in native_options:
        if "fps" in normalized and not image_frames:
            total = normalized.get("total_num_frames", count)
            native_options["sample_fps"] = count / total * normalized["fps"]
        else:
            native_options["sample_fps"] = 2.0
    return normalized, native_options
