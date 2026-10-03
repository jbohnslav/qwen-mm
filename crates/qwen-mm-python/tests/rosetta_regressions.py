"""Regression coverage for the original Rosetta message and option shapes."""

import unittest
from pathlib import Path

import numpy as np
from qwen_mm import Processor

CACHE = Path(__file__).resolve().parents[3] / "reference/.cache/huggingface"


class RosettaRegressions(unittest.TestCase):
    def test_original_video_frame_list_mixes_with_images(self):
        for profile in ("Qwen3", "Qwen3.5"):
            processor = Processor.from_pretrained(profile, cache_dir=CACHE, local_files_only=True)
            frame = np.zeros((128, 128, 3), dtype=np.uint8)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame},
                        {
                            "type": "video",
                            "video": [frame, frame],
                            "resized_height": 128,
                            "resized_width": 128,
                        },
                    ],
                }
            ]
            prepared = processor.prepare(messages)
            self.assertEqual(prepared["image_grid_thw"].tolist(), [[1, 8, 8]])
            self.assertEqual(prepared["video_grid_thw"].tolist(), [[1, 8, 8]])
            self.assertIs(messages[0]["content"][1]["video"][0], frame)

    def test_direct_transformers_minimum_is_explicit_and_does_not_mutate_messages(self):
        for profile in ("Qwen3", "Qwen3.5"):
            processor = Processor.from_pretrained(profile, cache_dir=CACHE, local_files_only=True)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": np.zeros((128, 128, 3), dtype=np.uint8)},
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ]
            normal = processor.prepare(messages)
            direct = processor.prepare(messages, min_pixels=65536)
            np.testing.assert_array_equal(normal["image_grid_thw"], [[1, 8, 8]])
            np.testing.assert_array_equal(direct["image_grid_thw"], [[1, 16, 16]])
            self.assertNotIn("min_pixels", messages[0]["content"][0])
            messages[0]["content"][0]["min_pixels"] = 4096
            override = processor.prepare(messages, min_pixels=65536)
            np.testing.assert_array_equal(override["image_grid_thw"], [[1, 8, 8]])
            item = messages[0]["content"][0]
            item.pop("min_pixels")
            item["options"] = {"min_pixels": 4096}
            nested = processor.prepare(messages, min_pixels=65536)
            np.testing.assert_array_equal(nested["image_grid_thw"], [[1, 8, 8]])
            self.assertEqual(item["options"], {"min_pixels": 4096})


if __name__ == "__main__":
    unittest.main()
