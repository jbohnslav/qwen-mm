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

    text = processor.prepare_batch(
        [{"messages": [{"role": "user", "content": "Describe the color blue."}]}]
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

    interleaved = processor.prepare_batch(
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Before: "},
                            image_item(0),
                            {"type": "text", "text": " After: "},
                            image_item(1),
                            {"type": "text", "text": " What changed?"},
                        ],
                    }
                ],
                "images": [first, second],
                "options": {"add_generation_prompt": True, "add_vision_id": True},
            }
        ]
    )
    assert interleaved.arrays["image_grid_thw"].shape == (2, 3)

    rgb = np.zeros((64, 96, 3), dtype=np.uint8)
    heterogeneous = processor.prepare_batch(
        [
            {"messages": [{"role": "user", "content": "Text-only row"}]},
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            image_item(0, min_pixels=50_176, max_pixels=200_704),
                            {"type": "text", "text": "Image row"},
                        ],
                    }
                ],
                "images": [rgb],
            },
        ]
    )
    assert heterogeneous.arrays["input_ids"].shape[0] == 2
    assert heterogeneous.arrays["pixel_values"].dtype == np.float32

    qwen35 = Processor.from_huggingface_cache("qwen3.5-9b", cache_directory=ASSETS_ROOT)
    non_thinking = qwen35.prepare_batch(
        [
            {
                "messages": [{"role": "user", "content": "Answer concisely."}],
                "options": {"add_generation_prompt": True, "enable_thinking": False},
            }
        ]
    )
    assert list(non_thinking.arrays) == TEXT_KEYS

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
                    "videos": [object()],
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
