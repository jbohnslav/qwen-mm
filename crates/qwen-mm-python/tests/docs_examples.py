"""Runnable Rosetta Stone examples executed against a freshly installed wheel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qwen_mm import Processor, UnsupportedMediaError

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ASSETS_ROOT = REPOSITORY_ROOT / "reference" / ".cache" / "huggingface"
FIXTURE_ROOT = REPOSITORY_ROOT / "fixtures" / "baseline" / "image24"
TEXT_KEYS = ["input_ids", "attention_mask", "mm_token_type_ids"]
IMAGE_KEYS = [*TEXT_KEYS, "pixel_values", "image_grid_thw"]


def main() -> None:
    profiles = Processor.supported_profiles()
    assert [profile["profile"] for profile in profiles] == ["qwen3-vl-8b", "qwen3.5-9b"]

    # Rosetta 0: familiar construction, resolved from the pinned local cache in tests.
    processor = Processor.from_pretrained(
        "Qwen3",
        cache_dir=ASSETS_ROOT,
        local_files_only=True,
        thread_budget=2,
    )
    assert processor.model_id == "Qwen/Qwen3-VL-8B-Instruct"
    assert processor.thread_budget == 2

    # Rosetta 1: text-only chat keeps the official messages object.
    text_messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {"role": "user", "content": [{"type": "text", "text": "Who are you?"}]},
    ]
    text = processor.prepare(text_messages, add_generation_prompt=True)
    assert list(text) == TEXT_KEYS
    assert list(text.arrays) == TEXT_KEYS

    first = FIXTURE_ROOT / "image-00.jpg"
    second = FIXTURE_ROOT / "image-01.jpg"

    # Rosetta 2: direct image URL. The HTTP/no-filter/redirect form is exercised
    # by media_sources.py; this deterministic docs test uses the same URL field.
    one_image_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": first.as_uri()},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    one_image = processor.prepare(one_image_messages, add_generation_prompt=True)
    assert list(one_image) == IMAGE_KEYS
    assert one_image["image_grid_thw"].shape == (1, 3)

    # Rosetta 3: multiple direct file sources retain their message order.
    multiple_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": first.as_uri()},
                {"type": "image", "image": second.as_uri()},
                {
                    "type": "text",
                    "text": "Identify the similarities between these images.",
                },
            ],
        }
    ]
    multiple_images = processor.prepare(multiple_messages, add_generation_prompt=True)
    assert multiple_images["image_grid_thw"].shape == (2, 3)

    # Rosetta 4: heterogeneous generation batch with explicit left padding.
    batch_image_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": first.as_uri()},
                {"type": "image", "image": second.as_uri()},
                {
                    "type": "text",
                    "text": "What are the common elements in these pictures?",
                },
            ],
        }
    ]
    heterogeneous = processor.prepare_batch(
        [
            {
                "messages": batch_image_messages,
                "options": {"add_generation_prompt": True},
            },
            {
                "messages": text_messages,
                "options": {"add_generation_prompt": True},
            },
        ],
        padding_side="left",
    )
    assert heterogeneous["input_ids"].shape[0] == 2
    assert heterogeneous["pixel_values"].dtype == np.float32
    assert heterogeneous.metadata["padding_side"] == "left"
    assert heterogeneous.metadata["request_layouts"][1]["left_padding"] > 0

    # Rosetta 5: qwen-vl-utils image sizing fields stay in the message.
    explicit_size_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": first.as_uri(),
                    "resized_height": 280,
                    "resized_width": 420,
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    explicit_size = processor.prepare(explicit_size_messages, add_generation_prompt=True)
    np.testing.assert_array_equal(explicit_size["image_grid_thw"], np.array([[1, 18, 26]]))

    # Rosetta 6: Qwen3.5 accepts the official nested OpenAI image_url message.
    qwen35 = Processor.from_pretrained(
        "Qwen3.5",
        cache_dir=ASSETS_ROOT,
        local_files_only=True,
    )
    openai_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": first.as_uri()}},
                {"type": "text", "text": "Where is this?"},
            ],
        }
    ]
    non_thinking = qwen35.prepare(
        openai_messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert list(non_thinking) == IMAGE_KEYS

    # Rosetta 7: a dataset adapter may keep its ordered path list preloaded.
    llama_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": 0},
                {"type": "text", "text": "Who are they?"},
            ],
        },
        {
            "role": "assistant",
            "content": "They're Kane and Gretzka from Bayern Munich.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What are they doing?"},
                {"type": "image", "image": 1},
            ],
        },
        {
            "role": "assistant",
            "content": "They are celebrating on the soccer field.",
        },
    ]
    llama_factory = processor.prepare(llama_messages, images=[first, first])
    assert llama_factory["image_grid_thw"].shape == (2, 3)

    # Rosetta 8: video remains a stable, explicit v0.1 boundary.
    try:
        processor.prepare(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": 0},
                        {"type": "text", "text": "Describe this video."},
                    ],
                }
            ],
            videos=[b"not decoded because video is rejected structurally"],
        )
    except UnsupportedMediaError as error:
        assert error.category == "unsupported_media"
    else:
        raise AssertionError("the documented v0.1 video boundary was not enforced")

    print("qwen-mm installed documentation examples passed")


if __name__ == "__main__":
    main()
