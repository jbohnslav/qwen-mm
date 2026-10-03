"""Normalize convenient Python media sources into native inputs."""

from __future__ import annotations

import base64
import binascii
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote_to_bytes, urlsplit
from urllib.request import Request, url2pathname, urlopen

from ._native import InvalidRequestError, ResourceLimitError, UnsupportedMediaError

_DEFAULT_ENCODED_BYTES_PER_ITEM = 64 * 1024 * 1024
_DEFAULT_ENCODED_BYTES_PER_BATCH = 1024 * 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024
_URL_TIMEOUT_SECONDS = 30.0


def _error(exception: type[Exception], message: str, **context: Any) -> Exception:
    error = exception(message)
    error.context = context  # type: ignore[attr-defined]
    return error


def _resource_limit(source: str, name: str, actual: int, limit: int) -> Exception:
    return _error(
        ResourceLimitError,
        f"resource limit exceeded while reading image source {source!r}",
        limit_name=name,
        actual=actual,
        limit=limit,
        source=source,
    )


@dataclass
class _ReadBudget:
    item_limit: int
    batch_limit: int
    total: int = 0

    def check(self, size: int, *, source: str) -> None:
        if size > self.item_limit:
            raise _resource_limit(
                source,
                "encoded_bytes_per_item",
                size,
                self.item_limit,
            )
        batch_size = self.total + size
        if batch_size > self.batch_limit:
            raise _resource_limit(
                source,
                "encoded_bytes_per_batch",
                batch_size,
                self.batch_limit,
            )

    def consume(self, size: int, *, source: str) -> None:
        self.check(size, source=source)
        self.total += size

    def read_limit(self) -> int:
        return min(self.item_limit, max(0, self.batch_limit - self.total))


def _read_limited(stream: Any, *, source: str, budget: _ReadBudget) -> bytes:
    content_length = stream.headers.get("Content-Length") if hasattr(stream, "headers") else None
    if content_length is not None:
        try:
            expected = int(content_length)
        except ValueError:
            expected = None
        if expected is not None and expected >= 0:
            budget.check(expected, source=source)

    data = bytearray()
    limit = budget.read_limit()
    while True:
        remaining = limit - len(data)
        chunk = stream.read(min(_READ_CHUNK_BYTES, remaining + 1))
        if not chunk:
            budget.consume(len(data), source=source)
            return bytes(data)
        data.extend(chunk)
        if len(data) > limit:
            budget.check(len(data), source=source)
            raise AssertionError("read budget check accepted data beyond its effective limit")


def _read_path(path: Path, *, source: str, budget: _ReadBudget) -> bytes:
    try:
        budget.check(path.stat().st_size, source=source)
        with path.open("rb") as stream:
            return _read_limited(stream, source=source, budget=budget)
    except ResourceLimitError:
        raise
    except OSError as cause:
        raise _error(
            UnsupportedMediaError,
            f"could not read image path {source!r}: {cause}",
            source=source,
            detail=str(cause),
        ) from cause


def _read_url(url: str, *, budget: _ReadBudget) -> bytes:
    request = Request(url, headers={"User-Agent": "qwen-mm/0.1"})
    try:
        with urlopen(request, timeout=_URL_TIMEOUT_SECONDS) as response:
            return _read_limited(response, source=url, budget=budget)
    except ResourceLimitError:
        raise
    except HTTPError as cause:
        raise _error(
            UnsupportedMediaError,
            f"image URL {url!r} returned HTTP {cause.code}: {cause.reason}",
            source=url,
            status=cause.code,
            detail=str(cause.reason),
        ) from cause
    except (URLError, TimeoutError, OSError) as cause:
        detail = getattr(cause, "reason", cause)
        raise _error(
            UnsupportedMediaError,
            f"could not fetch image URL {url!r}: {detail}",
            source=url,
            detail=str(detail),
        ) from cause


def _decode_data_uri(uri: str, *, budget: _ReadBudget) -> bytes:
    metadata, separator, payload = uri.partition(",")
    if not separator:
        raise _error(
            InvalidRequestError,
            "image data URI is missing its comma separator",
            source="data URI",
        )
    media_type, *parameters = metadata[5:].split(";")
    if not media_type.lower().startswith("image/"):
        raise _error(
            UnsupportedMediaError,
            f"data URI media type {media_type!r} is not an image",
            media_type=media_type,
        )
    try:
        is_base64 = any(parameter.lower() == "base64" for parameter in parameters)
        if is_base64 and "%" not in payload:
            compact = "".join(payload.split())
            if len(compact) % 4 == 0:
                padding = len(compact) - len(compact.rstrip("="))
                budget.check((len(compact) // 4) * 3 - padding, source="data URI")
        encoded = unquote_to_bytes(payload)
        if is_base64:
            encoded = b"".join(encoded.split())
            data = base64.b64decode(encoded, validate=True)
        else:
            data = encoded
    except (binascii.Error, ValueError) as cause:
        raise _error(
            InvalidRequestError,
            f"image data URI payload is invalid: {cause}",
            source="data URI",
            detail=str(cause),
        ) from cause
    budget.consume(len(data), source="data URI")
    return data


def _consume_existing_image(value: Any, *, source: str, budget: _ReadBudget) -> None:
    candidate = value
    if isinstance(value, dict):
        if "rgb" in value or "data" not in value:
            return
        candidate = value["data"]
    try:
        buffer = memoryview(candidate)
    except TypeError:
        return
    if buffer.ndim == 1:
        budget.consume(buffer.nbytes, source=source)


def _load_source(
    value: Any, *, source: str, budget: _ReadBudget, limits: dict[str, int] | None = None
) -> Any:
    from ._video import is_pillow, pillow_rgb

    if is_pillow(value):
        return pillow_rgb(value, limits=limits or {})
    if isinstance(value, os.PathLike):
        path = Path(os.fspath(value))
        return _read_path(path, source=str(path), budget=budget)
    if not isinstance(value, str):
        _consume_existing_image(value, source=source, budget=budget)
        return value

    lowered = value.lower()
    if lowered.startswith("data:"):
        return _decode_data_uri(value, budget=budget)
    if lowered.startswith(("http://", "https://")):
        return _read_url(value, budget=budget)
    if lowered.startswith("file:"):
        parsed = urlsplit(value)
        path_text = url2pathname(parsed.path)
        if parsed.netloc and parsed.netloc != "localhost":
            path_text = f"//{parsed.netloc}{path_text}"
        return _read_path(Path(path_text), source=value, budget=budget)
    if "://" in value:
        scheme = urlsplit(value).scheme
        raise _error(
            UnsupportedMediaError,
            f"image URL scheme {scheme!r} is not supported",
            source=value,
            scheme=scheme,
        )
    return _read_path(Path(value), source=value, budget=budget)


def _is_numeric_reference(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _normalize_image_url(item: dict[str, Any]) -> Any:
    if "image_url" not in item:
        raise _error(
            InvalidRequestError,
            "image_url content requires an image_url field",
            field="image_url",
        )
    value = item.get("image_url")
    if isinstance(value, dict):
        if "url" not in value:
            raise _error(
                InvalidRequestError,
                "image_url content requires a url field",
                field="image_url.url",
            )
        return value["url"]
    return value


def _video_option_key(value: Any) -> Any:
    """Snapshot options losslessly; NumPy repr rounds values and is not identity."""
    if isinstance(value, dict):
        return (
            "dict",
            frozenset(
                (_video_option_key(key), _video_option_key(item)) for key, item in value.items()
            ),
        )
    if hasattr(value, "tolist"):
        return _video_option_key(value.tolist())
    if isinstance(value, (list, tuple)):
        return ("list", tuple(_video_option_key(item) for item in value))
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, (str, int, bool)) or value is None:
        return (type(value).__name__, value)
    # Let source validation reject unsupported values without hashing them.
    return ("unsupported", id(value))


def _normalize_request(
    request: Any, *, budget: _ReadBudget, limits: dict[str, int], threads: int
) -> Any:
    from ._video import SOURCE_OPTIONS, VIDEO_OPTIONS, normalize_video

    if not isinstance(request, dict):
        return request
    existing_images = request.get("images", [])
    existing_videos = request.get("videos", [])
    messages = request.get("messages")
    if not isinstance(existing_images, list) or not isinstance(existing_videos, list):
        return request
    if not isinstance(messages, list):
        return request
    images = [
        _load_source(image, source=f"request.images[{index}]", budget=budget, limits=limits)
        for index, image in enumerate(existing_images)
    ]
    videos: list[Any] = []
    video_cache: dict[Any, tuple[int, dict[str, Any]]] = {}
    referenced_videos: set[int] = set()
    normalized_messages: list[Any] = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            normalized_messages.append(message)
            continue
        normalized_content: list[Any] = []
        for item in message["content"]:
            if not isinstance(item, dict):
                normalized_content.append(item)
                continue
            kind = item.get("type")
            if kind == "image_url":
                source = _normalize_image_url(item)
                images.append(
                    _load_source(
                        source,
                        source=f"message image_url[{len(images)}]",
                        budget=budget,
                        limits=limits,
                    )
                )
                normalized = {k: v for k, v in item.items() if k not in {"type", "image_url"}}
                normalized.update({"type": "image", "image": len(images) - 1})
                normalized_content.append(normalized)
            elif kind == "image" and "image" in item and not _is_numeric_reference(item["image"]):
                images.append(
                    _load_source(
                        item["image"],
                        source=f"message image[{len(images)}]",
                        budget=budget,
                        limits=limits,
                    )
                )
                normalized_content.append({**item, "image": len(images) - 1})
            elif kind in ("video", "video_url"):
                aliases = [
                    key
                    for key in ("video", "video_url", "input_index", "buffer_index")
                    if key in item
                ]
                if len(aliases) != 1:
                    raise _error(
                        InvalidRequestError, "video content requires exactly one source field"
                    )
                source = item[aliases[0]]
                if kind == "video_url" and isinstance(source, dict):
                    source = source.get("url")
                inline = {k: v for k, v in item.items() if k not in {"type", *aliases, "options"}}
                nested = item.get("options", {})
                if not isinstance(nested, dict):
                    raise _error(InvalidRequestError, "video options must be a dictionary")
                if inline.keys() & nested.keys():
                    raise _error(InvalidRequestError, "video options cannot be specified twice")
                options = {**inline, **nested}
                unknown = options.keys() - VIDEO_OPTIONS - SOURCE_OPTIONS
                if unknown:
                    raise _error(
                        InvalidRequestError, "unsupported video options", fields=sorted(unknown)
                    )
                cache_key = None
                if _is_numeric_reference(source):
                    if not 0 <= source < len(existing_videos):
                        raise _error(
                            InvalidRequestError,
                            "video reference index is out of range",
                            input_index=source,
                        )
                    referenced_videos.add(source)
                    cache_key = (source, _video_option_key(options))
                    source = existing_videos[source]
                if cache_key is not None and cache_key in video_cache:
                    index, native_options = video_cache[cache_key]
                else:
                    video, native_options = normalize_video(
                        source,
                        options=options,
                        limits=limits,
                        budget=budget,
                        threads=threads,
                    )
                    index = len(videos)
                    videos.append(video)
                    if cache_key is not None:
                        video_cache[cache_key] = (index, native_options)
                normalized_content.append(
                    {"type": "video", "input_index": index, "options": native_options}
                )
            else:
                normalized_content.append(item)
        normalized_messages.append({**message, "content": normalized_content})
    if len(referenced_videos) != len(existing_videos):
        missing = next(
            index for index in range(len(existing_videos)) if index not in referenced_videos
        )
        raise _error(
            InvalidRequestError,
            "every supplied video input must be referenced",
            input_index=missing,
        )
    normalized_request = {**request, "messages": normalized_messages}
    if images or "images" in request:
        normalized_request["images"] = images
    if videos or "videos" in request:
        normalized_request["videos"] = videos
    return normalized_request


def normalize_requests(
    requests: Any,
    *,
    limits: dict[str, int],
    image_defaults: dict[str, int] | None = None,
    thread_budget: int = 1,
) -> Any:
    """Return the native request shape, resolving images and videos once."""
    if not isinstance(requests, list):
        return requests
    if image_defaults:

        def with_defaults(item: Any) -> Any:
            if not isinstance(item, dict) or item.get("type") not in ("image", "image_url"):
                return item
            if "options" in item:
                if not isinstance(item["options"], dict):
                    return item  # Preserve native validation of malformed options.
                return {**item, "options": {**image_defaults, **item["options"]}}
            return {**image_defaults, **item}

        updated = []
        for request in requests:
            if not isinstance(request, dict) or not isinstance(request.get("messages"), list):
                updated.append(request)
                continue
            messages = []
            for message in request["messages"]:
                if isinstance(message, dict) and isinstance(message.get("content"), list):
                    content = [with_defaults(item) for item in message["content"]]
                    message = {**message, "content": content}
                messages.append(message)
            updated.append({**request, "messages": messages})
        requests = updated
    budget = _ReadBudget(
        item_limit=limits.get("encoded_bytes_per_item", _DEFAULT_ENCODED_BYTES_PER_ITEM),
        batch_limit=limits.get("encoded_bytes_per_batch", _DEFAULT_ENCODED_BYTES_PER_BATCH),
    )
    return [
        _normalize_request(request, budget=budget, limits=limits, threads=thread_budget)
        for request in requests
    ]
