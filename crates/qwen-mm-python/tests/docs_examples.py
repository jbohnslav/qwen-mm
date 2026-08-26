"""Runnable public examples executed against a freshly installed wheel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qwen_mm import Processor, UnsupportedMediaError

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ASSETS_ROOT = REPOSITORY_ROOT / "reference" / ".cache" / "huggingface"
FIXTURE_ROOT = REPOSITORY_ROOT / "fixtures" / "baseline" / "image24"
TEXT_KEYS = ["input_ids", "attention_mask", "mm_token_type_ids"]
IMAGE_KEYS = [*TEXT_KEYS, "pixel_values", "image_grid_thw"]


def image_item(index: int, **options: int) -> dict[str, object]:
    return {"type": "image", "image": index, **options}


def main() -> None:
    profiles = Processor.supported_profiles()
    assert [profile["profile"] for profile in profiles] == ["qwen3-vl-8b", "qwen3.5-9b"]

    processor = Processor.from_huggingface_cache(
        "qwen3-vl-8b",
        cache_directory=ASSETS_ROOT,
        thread_budget=2,
    )
    assert processor.model_id == "Qwen/Qwen3-VL-8B-Instruct"
    assert processor.thread_budget == 2

    text_messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {"role": "user", "content": [{"type": "text", "text": "Who are you?"}]},
    ]
    text = processor.prepare_batch(
        [
            {
                "messages": text_messages,
                "options": {"add_generation_prompt": True},
            }
        ]
    )
    assert list(text.arrays) == TEXT_KEYS

    first = (FIXTURE_ROOT / "image-00.jpg").read_bytes()
    second = (FIXTURE_ROOT / "image-01.jpg").read_bytes()
    one_image = processor.prepare_batch(
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            image_item(0),
                            {"type": "text", "text": "Describe this image."},
                        ],
                    }
                ],
                "images": [first],
                "options": {"add_generation_prompt": True},
            }
        ]
    )
    assert list(one_image.arrays) == IMAGE_KEYS
    assert one_image.arrays["image_grid_thw"].shape == (1, 3)

    multiple_images = processor.prepare_batch(
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            image_item(0),
                            image_item(1),
                            {
                                "type": "text",
                                "text": "Identify the similarities between these images.",
                            },
                        ],
                    }
                ],
                "images": [first, second],
                "options": {"add_generation_prompt": True},
            }
        ]
    )
    assert multiple_images.arrays["image_grid_thw"].shape == (2, 3)

    heterogeneous = processor.prepare_batch(
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            image_item(0),
                            image_item(1),
                            {
                                "type": "text",
                                "text": "What are the common elements in these pictures?",
                            },
                        ],
                    }
                ],
                "images": [first, second],
                "options": {"add_generation_prompt": True},
            },
            {
                "messages": text_messages,
                "options": {"add_generation_prompt": True},
            },
        ]
    )
    assert heterogeneous.arrays["input_ids"].shape[0] == 2
    assert heterogeneous.arrays["pixel_values"].dtype == np.float32

    explicit_size = processor.prepare_batch(
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            image_item(0, resized_height=280, resized_width=420),
                            {"type": "text", "text": "Describe this image."},
                        ],
                    }
                ],
                "images": [first],
                "options": {"add_generation_prompt": True},
            }
        ]
    )
    np.testing.assert_array_equal(explicit_size.arrays["image_grid_thw"], np.array([[1, 18, 26]]))

    qwen35 = Processor.from_huggingface_cache("qwen3.5-9b", cache_directory=ASSETS_ROOT)
    non_thinking = qwen35.prepare_batch(
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            image_item(0),
                            {"type": "text", "text": "Where is this?"},
                        ],
                    }
                ],
                "images": [first],
                "options": {"add_generation_prompt": True, "enable_thinking": False},
            }
        ]
    )
    assert list(non_thinking.arrays) == IMAGE_KEYS

    llama_factory_row = processor.prepare_batch(
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            image_item(0),
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
                            image_item(1),
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": "They are celebrating on the soccer field.",
                    },
                ],
                "images": [first, first],
            }
        ]
    )
    assert llama_factory_row.arrays["image_grid_thw"].shape == (2, 3)

    try:
        processor.prepare_batch(
            [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "video", "video": 0}],
                        }
                    ],
                    "videos": [b"not decoded because video is rejected structurally"],
                }
            ]
        )
    except UnsupportedMediaError as error:
        assert error.category == "unsupported_media"
    else:
        raise AssertionError("the documented v0.1 video boundary was not enforced")

    print("qwen-mm installed documentation examples passed")


if __name__ == "__main__":
    main()
