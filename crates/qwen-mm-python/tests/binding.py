"""Focused installed-wheel ownership, batching, and failure regression suite."""

from __future__ import annotations

import gc
import hashlib
import json
import threading
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import qwen_mm
import qwen_mm._native as qwen_mm_native

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ASSETS_ROOT = REPOSITORY_ROOT / "reference" / ".cache" / "huggingface"
FIXTURE_ROOT = REPOSITORY_ROOT / "fixtures" / "baseline" / "image24"
PROFILE_CASES = {
    "qwen3-vl-8b": (
        "models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        783,
    ),
    "qwen3.5-9b": (
        "models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        785,
    ),
}
OFFICIAL_TEXT_KEYS = ["input_ids", "attention_mask", "mm_token_type_ids"]
OFFICIAL_IMAGE_KEYS = [*OFFICIAL_TEXT_KEYS, "pixel_values", "image_grid_thw"]


def _processor(profile: str, *, limits: dict[str, int] | None = None) -> qwen_mm.Processor:
    relative, _ = PROFILE_CASES[profile]
    return qwen_mm.Processor(profile, ASSETS_ROOT / relative, limits=limits)


def _request(images: list[Any], *, generation: bool = True) -> list[dict[str, Any]]:
    content = [{"type": "image", "input_index": index} for index in range(len(images))]
    content.append({"type": "text", "text": "Describe each image briefly."})
    return [
        {
            "messages": [{"role": "user", "content": content}],
            "images": images,
            "options": {"add_generation_prompt": generation},
        }
    ]


def _assert_arrays(output: qwen_mm.PreparedBatch, token_columns: int, image_count: int) -> None:
    arrays = output.arrays
    assert list(arrays) == OFFICIAL_IMAGE_KEYS
    assert output.official_keys() == OFFICIAL_IMAGE_KEYS
    assert arrays["input_ids"].shape == (1, token_columns)
    assert arrays["attention_mask"].shape == (1, token_columns)
    assert arrays["mm_token_type_ids"].shape == (1, token_columns)
    assert arrays["pixel_values"].shape == (3072 * image_count, 1536)
    assert arrays["image_grid_thw"].shape == (image_count, 3)
    for key in OFFICIAL_TEXT_KEYS + ["image_grid_thw"]:
        assert arrays[key].dtype == np.dtype(np.int64)
        assert arrays[key].flags.c_contiguous
    assert arrays["pixel_values"].dtype == np.dtype(np.float32)
    assert arrays["pixel_values"].flags.c_contiguous
    for value in arrays.values():
        assert value.strides == (value.shape[1] * value.itemsize, value.itemsize)
    assert not arrays["pixel_values"].flags.owndata
    assert arrays["pixel_values"].base is not None
    assert "_pixel_values_allocation_ptr" not in output.metadata
    assert len(output.metadata["images"]) == image_count
    assert len(output.metadata["sidecar"]["images"]) == image_count
    assert [item["grid_row"] for item in output.metadata["images"]] == list(range(image_count))


def test_one_image_and_conditional_outputs() -> None:
    encoded = (FIXTURE_ROOT / "image-00.jpg").read_bytes()
    for profile, (_, token_columns) in PROFILE_CASES.items():
        processor = _processor(profile)
        output = processor.prepare_batch(_request([encoded]))
        _assert_arrays(output, token_columns, 1)
        text = processor.prepare_batch([{"messages": [{"role": "user", "content": "hello"}]}])
        assert list(text.arrays) == OFFICIAL_TEXT_KEYS
        assert text.official_keys() == OFFICIAL_TEXT_KEYS
        assert "pixel_values" not in text.arrays
        assert "image_grid_thw" not in text.arrays

    processor = _processor("qwen3-vl-8b")
    mixed = processor.prepare_batch(
        [
            {"messages": [{"role": "user", "content": "short"}]},
            _request([encoded])[0],
        ]
    )
    assert list(mixed.arrays) == OFFICIAL_IMAGE_KEYS
    assert mixed.arrays["input_ids"].shape[0] == 2
    assert mixed.metadata["images"][0]["request_index"] == 1
    for row, layout in enumerate(mixed.metadata["request_layouts"]):
        token_count = layout["token_count"]
        np.testing.assert_array_equal(
            mixed.arrays["attention_mask"][row],
            np.r_[
                np.ones(token_count, dtype=np.int64),
                np.zeros(layout["right_padding"], dtype=np.int64),
            ],
        )
        assert np.all(mixed.arrays["mm_token_type_ids"][row, token_count:] == 0)
        assert np.all(mixed.arrays["input_ids"][row, token_count:] == 151_643)


def test_24_independent_request_order() -> None:
    processor = _processor("qwen3-vl-8b")
    images = [np.full((28, 28, 3), index, dtype=np.uint8) for index in range(24)]
    requests = [_request([image], generation=False)[0] for image in images]
    output = processor.prepare_batch(requests)
    assert output.arrays["input_ids"].shape[0] == 24
    assert output.arrays["image_grid_thw"].shape == (24, 3)
    assert [item["request_index"] for item in output.metadata["images"]] == list(range(24))
    assert [item["grid_row"] for item in output.metadata["images"]] == list(range(24))
    for request_index, layout in enumerate(output.metadata["request_layouts"]):
        assert layout["request_index"] == request_index
        assert layout["image_grid_rows"] == [request_index, request_index + 1]


def test_exact_24_image_shape_and_gil_release() -> None:
    processor = _processor("qwen3-vl-8b")
    encoded = [(FIXTURE_ROOT / f"image-{index:02d}.jpg").read_bytes() for index in range(24)]
    request = _request(encoded)
    ready = threading.Event()
    stop = threading.Event()
    observed_native_window = threading.Event()

    def worker() -> None:
        ready.set()
        while not stop.is_set():
            if qwen_mm_native._test_native_batch_active():
                observed_native_window.set()

    thread = threading.Thread(target=worker)
    thread.start()
    ready.wait(timeout=5)
    output = processor.prepare_batch(request)
    stop.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert observed_native_window.is_set(), (
        "Python worker never executed while native batch processing was active"
    )
    _assert_arrays(output, 18_493, 24)
    assert output.arrays["pixel_values"].nbytes == 452_984_832
    golden = json.loads(
        (REPOSITORY_ROOT / "reference/goldens/v1/qwen3-vl-8b/image24/manifest.json").read_text()
    )["output"]["arrays"]
    for key in OFFICIAL_TEXT_KEYS + ["image_grid_thw"]:
        np.testing.assert_array_equal(output.arrays[key], np.asarray(golden[key]["data"]))
    pixel_hash = hashlib.sha256(memoryview(output.arrays["pixel_values"]).cast("B")).hexdigest()
    assert pixel_hash == golden["pixel_values"]["sha256"]
    del output, request, encoded
    gc.collect()


def test_raw_aliasing_validation_and_output_lifetimes() -> None:
    processor = _processor("qwen3-vl-8b")
    raw = np.arange(767 * 1023 * 3, dtype=np.uint8).reshape(767, 1023, 3)
    raw_reference = weakref.ref(raw)
    request = _request([raw, raw])
    output = processor.prepare_batch(request)
    arrays = output.arrays
    pixel_view = arrays["pixel_values"][:2, :8]
    grid_view = arrays["image_grid_thw"][:, :]
    assert arrays["pixel_values"].shape == (6144, 1536)
    assert arrays["image_grid_thw"].tolist() == [[1, 48, 64], [1, 48, 64]]
    assert output.metadata["images"][0]["cache_key"] == output.metadata["images"][1]["cache_key"]
    before = (
        arrays["input_ids"].copy(),
        arrays["image_grid_thw"].copy(),
        pixel_view.copy(),
    )
    del raw, request, processor, output, arrays
    gc.collect()
    assert raw_reference() is None
    np.testing.assert_array_equal(grid_view, before[1])
    np.testing.assert_array_equal(pixel_view, before[2])

    encoded_owner = np.frombuffer(
        (FIXTURE_ROOT / "image-00.jpg").read_bytes(), dtype=np.uint8
    ).copy()
    encoded_reference = weakref.ref(encoded_owner)
    encoded_output = _processor("qwen3-vl-8b").prepare_batch(_request([encoded_owner]))
    encoded_view = encoded_output.arrays["pixel_values"][:2, :8]
    encoded_before = encoded_view.copy()
    del encoded_owner, encoded_output
    gc.collect()
    assert encoded_reference() is None
    np.testing.assert_array_equal(encoded_view, encoded_before)

    invalid = [
        np.zeros((8, 8, 3), dtype=np.float32),
        np.zeros((8, 8), dtype=np.uint8),
        np.zeros((8, 8, 4), dtype=np.uint8),
        np.zeros((0, 8, 3), dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.uint8)[:, ::2, :],
    ]
    processor = _processor("qwen3-vl-8b")
    expected_categories = [
        qwen_mm.UnsupportedMediaError,
        qwen_mm.UnsupportedMediaError,
        qwen_mm.UnsupportedMediaError,
        qwen_mm.MediaGeometryError,
        qwen_mm.MediaGeometryError,
    ]
    for value, expected in zip(invalid, expected_categories, strict=True):
        try:
            processor.prepare_batch(_request([value]))
        except expected as error:
            assert error.category == expected.category
            assert isinstance(error.context, dict)
        else:
            raise AssertionError(f"invalid raw input was accepted: {value.shape} {value.dtype}")
        assert value.flags.writeable


def _assert_reusable_after_failure(
    processor: qwen_mm.Processor,
    exception: type[BaseException],
    request: list[dict[str, Any]],
) -> None:
    try:
        processor.prepare_batch(request)
    except exception as error:
        assert isinstance(error, qwen_mm.QwenMMError)
        assert error.category == exception.category
        assert isinstance(error.context, dict)
    else:
        raise AssertionError(f"{exception.__name__} was not raised")
    success = processor.prepare_batch([{"messages": [{"role": "user", "content": "ok"}]}])
    assert list(success.arrays) == OFFICIAL_TEXT_KEYS


def test_typed_failures_have_no_partial_outputs() -> None:
    processor = _processor("qwen3-vl-8b")
    cases: list[tuple[type[BaseException], list[dict[str, Any]]]] = [
        (qwen_mm.InvalidRequestError, []),
        (
            qwen_mm.UnsupportedOptionError,
            [
                {
                    "messages": [{"role": "user", "content": "hello"}],
                    "options": {"left_padding": True},
                }
            ],
        ),
        (
            qwen_mm.UnsupportedMediaError,
            [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "video", "input_index": 0}],
                        }
                    ],
                    "videos": [object()],
                }
            ],
        ),
        (
            qwen_mm.MediaDecodeError,
            _request([{"data": b"\xff\xd8\xff", "format": "jpeg"}]),
        ),
        (
            qwen_mm.MediaGeometryError,
            [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "input_index": 0,
                                    "options": {"resized_height": 32},
                                }
                            ],
                        }
                    ],
                    "images": [np.zeros((32, 32, 3), dtype=np.uint8)],
                }
            ],
        ),
    ]
    for exception, request in cases:
        _assert_reusable_after_failure(processor, exception, request)

    encoded = (FIXTURE_ROOT / "image-00.jpg").read_bytes()
    unsupported_media = [
        _request([{"data": encoded, "format": "gif"}]),
        _request([b"GIF89a" + bytes(64)]),
        _request([Path("image.jpg")]),
    ]
    for request in unsupported_media:
        _assert_reusable_after_failure(processor, qwen_mm.UnsupportedMediaError, request)

    malformed = [
        _request([encoded])[:-1],
        [{"messages": [{"role": "user", "content": [{"type": "text", "text": "x", "extra": 1}]}]}],
        [
            {
                "messages": [
                    {"role": "user", "content": [{"type": "image", "input_index": 0, "image": 0}]}
                ],
                "images": [encoded],
            }
        ],
        [
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "x",
                        "tool_calls": [{"name": "f", "arguments": "{"}],
                    }
                ]
            }
        ],
        [
            {
                "messages": [{"role": "user", "content": [{"type": "image", "input_index": True}]}],
                "images": [encoded],
            }
        ],
        [{"messages": [{"role": "user", "content": "x"}], "options": {"pad_token_id": True}}],
    ]
    for request in malformed:
        _assert_reusable_after_failure(processor, qwen_mm.InvalidRequestError, request)

    limited = _processor("qwen3-vl-8b", limits={"media_per_batch": 1})
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    _assert_reusable_after_failure(limited, qwen_mm.ResourceLimitError, _request([image, image]))

    encoded_limited = _processor("qwen3-vl-8b", limits={"encoded_bytes_per_item": 2})
    encoded_owner = bytearray(encoded)
    _assert_reusable_after_failure(
        encoded_limited, qwen_mm.ResourceLimitError, _request([encoded_owner])
    )
    assert encoded_owner == encoded
    raw_limited = _processor("qwen3-vl-8b", limits={"decoded_pixels_per_image_or_frame": 4})
    raw_owner = np.zeros((3, 3, 3), dtype=np.uint8)
    _assert_reusable_after_failure(raw_limited, qwen_mm.ResourceLimitError, _request([raw_owner]))
    assert raw_owner.flags.writeable

    _assert_reusable_after_failure(
        processor,
        qwen_mm.UnsupportedOptionError,
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "input_index": 0}],
                    }
                ],
                "options": {"left_padding": True},
                "images": [{"data": encoded, "format": "gif"}],
            }
        ],
    )
    _assert_reusable_after_failure(
        processor,
        qwen_mm.InvalidRequestError,
        [
            _request([{"data": encoded, "format": "gif"}])[0],
            {
                "messages": [
                    {"role": "system", "content": "late"},
                    {"role": "system", "content": "again"},
                ]
            },
        ],
    )

    all_categories = {
        qwen_mm.InvalidRequestError: "invalid_request",
        qwen_mm.UnsupportedOptionError: "unsupported_option",
        qwen_mm.UnsupportedMediaError: "unsupported_media",
        qwen_mm.ProfileMismatchError: "profile_mismatch",
        qwen_mm.MediaDecodeError: "media_decode",
        qwen_mm.MediaGeometryError: "media_geometry",
        qwen_mm.ResourceLimitError: "resource_limit",
        qwen_mm.ArithmeticOverflowError: "arithmetic_overflow",
        qwen_mm.DestinationTooSmallError: "destination_too_small",
        qwen_mm.InternalInvariantError: "internal_invariant",
    }
    for exception, category in all_categories.items():
        assert issubclass(exception, qwen_mm.QwenMMError)
        assert exception.category == category
        assert exception.__module__ == "qwen_mm._native"


def main() -> None:
    if not all((ASSETS_ROOT / relative).is_dir() for relative, _ in PROFILE_CASES.values()):
        raise SystemExit(f"hash-pinned snapshots are missing below {ASSETS_ROOT}")
    test_one_image_and_conditional_outputs()
    test_24_independent_request_order()
    test_exact_24_image_shape_and_gil_release()
    test_raw_aliasing_validation_and_output_lifetimes()
    test_typed_failures_have_no_partial_outputs()
    print("qwen-mm installed binding tests passed")


if __name__ == "__main__":
    main()
