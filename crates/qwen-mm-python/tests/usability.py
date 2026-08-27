"""Installed-wheel coverage for concise preparation and generation-ready padding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from qwen_mm import PreparedBatch, Processor

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ASSETS_ROOT = REPOSITORY_ROOT / "reference" / ".cache" / "huggingface"
QWEN3_SNAPSHOT = ASSETS_ROOT / (
    "models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
QWEN35_SNAPSHOT = ASSETS_ROOT / (
    "models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
IMAGE_PATH = REPOSITORY_ROOT / "fixtures" / "baseline" / "image24" / "image-00.jpg"


def _processor(profile: str = "qwen3-vl-8b") -> Processor:
    snapshot = QWEN3_SNAPSHOT if profile == "qwen3-vl-8b" else QWEN35_SNAPSHOT
    return Processor(profile, snapshot)


def _assert_arrays_equal(first: PreparedBatch, second: PreparedBatch) -> None:
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_prepare_is_the_single_request_equivalent() -> None:
    processor = _processor()
    messages = [{"role": "user", "content": "Who are you?"}]

    concise = processor.prepare(messages, add_generation_prompt=True)
    explicit = processor.prepare_batch(
        [
            {
                "messages": messages,
                "options": {"add_generation_prompt": True},
            }
        ]
    )
    _assert_arrays_equal(concise, explicit)
    assert concise.metadata == explicit.metadata

    image_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "input_index": 0},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    image = IMAGE_PATH.read_bytes()
    concise_image = processor.prepare(
        image_messages,
        images=[image],
        add_generation_prompt=True,
    )
    explicit_image = processor.prepare_batch(
        [
            {
                "messages": image_messages,
                "images": [image],
                "options": {"add_generation_prompt": True},
            }
        ]
    )
    _assert_arrays_equal(concise_image, explicit_image)
    assert concise_image.metadata == explicit_image.metadata

    qwen35 = _processor("qwen3.5-9b")
    non_thinking = qwen35.prepare(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert non_thinking.metadata["text"][0]["rendered_prompt"]


def test_prepared_batch_is_a_model_input_mapping() -> None:
    prepared = _processor().prepare([{"role": "user", "content": "Hello"}])

    assert prepared.keys() == list(prepared.arrays)
    assert list(prepared) == prepared.keys()
    assert len(prepared) == len(prepared.arrays)
    assert "input_ids" in prepared
    assert "metadata" not in prepared
    assert prepared["input_ids"] is prepared.arrays["input_ids"]
    assert all(
        value is prepared.arrays[key]
        for key, value in zip(prepared.keys(), prepared.values(), strict=True)
    )
    assert all(
        key == expected_key and value is prepared.arrays[key]
        for (key, value), expected_key in zip(prepared.items(), prepared.keys(), strict=True)
    )
    assert prepared.get("input_ids") is prepared["input_ids"]
    assert prepared.get("missing") is None
    sentinel = object()
    assert prepared.get("missing", sentinel) is sentinel

    def accept_model_inputs(**values: Any) -> dict[str, Any]:
        return values

    unpacked = accept_model_inputs(**prepared)
    assert list(unpacked) == prepared.keys()
    assert unpacked["input_ids"] is prepared["input_ids"]
    try:
        prepared["missing"]
    except KeyError as error:
        assert error.args == ("missing",)
    else:
        raise AssertionError("missing model input key did not raise KeyError")


def test_left_padding_matches_generation_layout() -> None:
    processor = _processor()
    requests = [
        {"messages": [{"role": "user", "content": "Short"}]},
        {
            "messages": [
                {
                    "role": "user",
                    "content": "A deliberately much longer prompt for batch padding coverage.",
                }
            ]
        },
    ]

    default = processor.prepare_batch(requests)
    left = processor.prepare_batch(requests, padding_side="left")
    observed, _report = processor.prepare_batch_observed(requests, padding_side="left")

    assert default.metadata["padding_side"] == "right"
    assert left.metadata["padding_side"] == "left"
    assert default.metadata["request_layouts"][0]["right_padding"] > 0
    assert default.metadata["request_layouts"][0]["left_padding"] == 0
    assert left.metadata["request_layouts"][0]["right_padding"] == 0
    assert left.metadata["request_layouts"][0]["left_padding"] > 0

    for row, layout in enumerate(left.metadata["request_layouts"]):
        padding = layout["left_padding"]
        tokens = layout["token_count"]
        for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
            expected = np.concatenate((default[key][row, tokens:], default[key][row, :tokens]))
            np.testing.assert_array_equal(left[key][row], expected)
            np.testing.assert_array_equal(observed[key][row], expected)
        assert np.all(left["attention_mask"][row, :padding] == 0)
        assert np.all(left["attention_mask"][row, padding:] == 1)

    try:
        processor.prepare_batch(requests, padding_side="center")
    except ValueError as error:
        assert "'left' or 'right'" in str(error)
    else:
        raise AssertionError("invalid padding side was accepted")


def main() -> None:
    if not QWEN3_SNAPSHOT.is_dir() or not QWEN35_SNAPSHOT.is_dir():
        raise SystemExit(f"hash-pinned snapshots are missing below {ASSETS_ROOT}")
    test_prepare_is_the_single_request_equivalent()
    test_prepared_batch_is_a_model_input_mapping()
    test_left_padding_matches_generation_layout()
    print("qwen-mm installed usability tests passed")


if __name__ == "__main__":
    main()
