"""Installed-wheel tests for direct local and remote image sources."""

from __future__ import annotations

import base64
import contextlib
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import qwen_mm

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ASSETS_ROOT = REPOSITORY_ROOT / "reference" / ".cache" / "huggingface"
SNAPSHOT = ASSETS_ROOT / (
    "models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
FIXTURE_ROOT = REPOSITORY_ROOT / "fixtures" / "baseline" / "image24"


def _processor(*, limits: dict[str, int] | None = None) -> qwen_mm.Processor:
    return qwen_mm.Processor("qwen3-vl-8b", SNAPSHOT, limits=limits, thread_budget=1)


def _request(*images: object) -> list[dict[str, object]]:
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": "Describe these images."})
    return [{"messages": [{"role": "user", "content": content}]}]


class _ImageHandler(BaseHTTPRequestHandler):
    image = b""

    def do_GET(self) -> None:
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/image")
            self.end_headers()
        elif self.path == "/image":
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(self.image)))
            self.end_headers()
            self.wfile.write(self.image)
        elif self.path == "/large":
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.end_headers()
            self.wfile.write(self.image)
        else:
            self.send_error(404, "fixture not found")

    def log_message(self, _format: str, *_args: object) -> None:
        pass


@contextlib.contextmanager
def _image_server(image: bytes) -> Iterator[str]:
    _ImageHandler.image = image
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ImageHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_paths_file_urls_data_uris_and_http_redirects() -> None:
    path = FIXTURE_ROOT / "image-00.jpg"
    encoded = path.read_bytes()
    data_uri = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
    with _image_server(encoded) as server:
        output = _processor().prepare_batch(
            _request(path, str(path), path.as_uri(), data_uri, f"{server}/redirect")
        )
    assert output.arrays["image_grid_thw"].shape == (5, 3)
    cache_keys = [image["cache_key"] for image in output.metadata["images"]]
    assert len(set(cache_keys)) == 1


def test_openai_image_url_forms_preserve_order() -> None:
    first = FIXTURE_ROOT / "image-00.jpg"
    second = FIXTURE_ROOT / "image-01.jpg"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": first.as_uri()},
                {
                    "type": "image_url",
                    "image_url": {"url": second.as_uri(), "detail": "auto"},
                },
                {"type": "image", "image": first},
                {"type": "text", "text": "Compare them in order."},
            ],
        }
    ]
    output = _processor().prepare_batch([{"messages": messages}])
    assert output.arrays["image_grid_thw"].shape == (3, 3)
    cache_keys = [image["cache_key"] for image in output.metadata["images"]]
    assert cache_keys[0] != cache_keys[1]
    assert cache_keys[0] == cache_keys[2]


def test_existing_media_and_reference_forms_remain_compatible() -> None:
    encoded = (FIXTURE_ROOT / "image-00.jpg").read_bytes()
    raw = np.zeros((28, 28, 3), dtype=np.uint8)
    direct = _processor().prepare_batch(_request(encoded, raw))
    assert direct.arrays["image_grid_thw"].shape == (2, 3)

    referenced = _processor().prepare_batch(
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "input_index": 0},
                            {
                                "type": "image",
                                "image": 0,
                                "options": {"min_pixels": 28 * 28},
                            },
                            {"type": "text", "text": "Describe it twice."},
                        ],
                    }
                ],
                "images": [encoded],
            }
        ]
    )
    assert referenced.arrays["image_grid_thw"].shape == (2, 3)
    assert referenced.metadata["images"][0]["input_index"] == 0
    assert referenced.metadata["images"][1]["input_index"] == 0


def test_source_failures_are_typed_and_actionable() -> None:
    missing = FIXTURE_ROOT / "does-not-exist.jpg"
    try:
        _processor().prepare_batch(_request(missing))
    except qwen_mm.UnsupportedMediaError as error:
        assert str(missing) in str(error)
        assert error.context["source"] == str(missing)
    else:
        raise AssertionError("missing image path unexpectedly succeeded")

    try:
        _processor().prepare_batch(_request("data:image/jpeg;base64,%%%"))
    except qwen_mm.InvalidRequestError as error:
        assert "data URI" in str(error)
        assert error.context["source"] == "data URI"
    else:
        raise AssertionError("malformed image data URI unexpectedly succeeded")

    encoded = (FIXTURE_ROOT / "image-00.jpg").read_bytes()
    with _image_server(encoded) as server:
        try:
            _processor().prepare_batch(_request(f"{server}/missing"))
        except qwen_mm.UnsupportedMediaError as error:
            assert "HTTP 404" in str(error)
            assert error.context["status"] == 404
            assert error.context["source"] == f"{server}/missing"
        else:
            raise AssertionError("missing image URL unexpectedly succeeded")


def test_direct_reads_respect_item_and_batch_encoded_byte_limits() -> None:
    path = FIXTURE_ROOT / "image-00.jpg"
    encoded = path.read_bytes()
    item_limit = len(encoded) - 1
    limited = _processor(limits={"encoded_bytes_per_item": item_limit})
    try:
        limited.prepare_batch(_request(path))
    except qwen_mm.ResourceLimitError as error:
        assert error.context["limit_name"] == "encoded_bytes_per_item"
        assert error.context["limit"] == item_limit
        assert error.context["source"] == str(path)
    else:
        raise AssertionError("oversized local image unexpectedly succeeded")

    oversized_data_uri = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
    try:
        limited.prepare_batch(_request(oversized_data_uri))
    except qwen_mm.ResourceLimitError as error:
        assert error.context["limit_name"] == "encoded_bytes_per_item"
        assert error.context["actual"] == len(encoded)
        assert error.context["source"] == "data URI"
    else:
        raise AssertionError("oversized image data URI unexpectedly succeeded")

    with _image_server(encoded) as server:
        try:
            limited.prepare_batch(_request(f"{server}/large"))
        except qwen_mm.ResourceLimitError as error:
            assert error.context["limit_name"] == "encoded_bytes_per_item"
            assert error.context["actual"] > item_limit
            assert error.context["source"] == f"{server}/large"
        else:
            raise AssertionError("oversized streamed image unexpectedly succeeded")

    batch_limit = len(encoded) * 2 - 1
    batch_limited = _processor(limits={"encoded_bytes_per_batch": batch_limit})
    try:
        batch_limited.prepare_batch(_request(path, path))
    except qwen_mm.ResourceLimitError as error:
        assert error.context["limit_name"] == "encoded_bytes_per_batch"
        assert error.context["limit"] == batch_limit
    else:
        raise AssertionError("oversized encoded image batch unexpectedly succeeded")


def main() -> None:
    if not SNAPSHOT.is_dir():
        raise SystemExit(f"hash-pinned snapshot is missing at {SNAPSHOT}")
    test_paths_file_urls_data_uris_and_http_redirects()
    test_openai_image_url_forms_preserve_order()
    test_existing_media_and_reference_forms_remain_compatible()
    test_source_failures_are_typed_and_actionable()
    test_direct_reads_respect_item_and_batch_encoded_byte_limits()
    print("qwen-mm installed media-source tests passed")


if __name__ == "__main__":
    main()
