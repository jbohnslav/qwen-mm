"""Installed-wheel Torch, batching, and decoding checks against Transformers."""

import gc
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from qwen_mm import Processor
from qwen_mm._tensors import TorchBatch
from transformers import AutoProcessor

CACHE = Path(__file__).resolve().parents[3] / "reference/.cache/huggingface"


class Ergonomics(unittest.TestCase):
    def test_shared_batch_options_and_row_overrides(self):
        processor = Processor.from_pretrained("Qwen3.5", cache_dir=CACHE, local_files_only=True)
        first = [{"role": "user", "content": "Hello"}]
        second = [{"role": "user", "content": "A longer prompt to require left padding."}]
        shorthand = processor.prepare_batch(
            [first, second], add_generation_prompt=True, enable_thinking=False, padding_side="left"
        )
        explicit = processor.prepare_batch(
            [
                {
                    "messages": row,
                    "options": {"add_generation_prompt": True, "enable_thinking": False},
                }
                for row in (first, second)
            ],
            padding_side="left",
        )
        for key in explicit:
            np.testing.assert_array_equal(shorthand[key], explicit[key])
        override = processor.prepare_batch(
            [{"messages": first, "options": {"enable_thinking": False}}],
            add_generation_prompt=True,
            enable_thinking=True,
        )
        expected = processor.prepare(first, add_generation_prompt=True, enable_thinking=False)
        np.testing.assert_array_equal(override["input_ids"], expected["input_ids"])

    def test_torch_outputs_share_storage_and_survive_numpy_owner(self):
        processor = Processor.from_pretrained("Qwen3", cache_dir=CACHE, local_files_only=True)
        messages = [{"role": "user", "content": "Hello"}]
        prepared = processor.prepare(messages)
        converted = TorchBatch(prepared, torch)
        for key in prepared:
            self.assertEqual(converted[key].data_ptr(), prepared[key].ctypes.data)
        expected = converted.input_ids.clone()
        del prepared
        gc.collect()
        torch.testing.assert_close(converted.input_ids, expected)
        actual = processor.prepare(messages, return_tensors="pt")
        self.assertIs(actual.to("cpu"), actual)
        self.assertIsInstance(actual.input_ids, torch.Tensor)
        self.assertEqual(actual.input_ids.dtype, torch.int64)
        self.assertNotIn("metadata", actual)
        self.assertEqual(actual.metadata["padding_side"], "right")
        torch.testing.assert_close(actual.input_ids, expected)

    def test_decode_matches_pinned_official_tokenizers(self):
        for profile in Processor.supported_profiles():
            snapshot = (
                CACHE
                / f"models--{profile['model_id'].replace('/', '--')}/snapshots/{profile['revision']}"
            )
            native = Processor(profile["profile"], snapshot)
            official = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
            for text in (
                "Hello, world!",
                "你好 🌍 café",
                "I 'm happy . Are you ?",
                "<|im_start|>assistant\nHello<|im_end|>",
            ):
                ids = official.tokenizer.encode(text, add_special_tokens=False)
                for skip in (True, False):
                    for cleanup in (True, False):
                        kwargs = dict(
                            skip_special_tokens=skip, clean_up_tokenization_spaces=cleanup
                        )
                        self.assertEqual(
                            native.decode(ids, **kwargs), official.decode(ids, **kwargs)
                        )
                        self.assertEqual(
                            native.batch_decode(torch.tensor([ids]), **kwargs),
                            official.batch_decode([ids], **kwargs),
                        )
                        self.assertEqual(
                            native.decode(np.array(ids), **kwargs), official.decode(ids, **kwargs)
                        )

    def test_invalid_output_option_fails_before_media_io(self):
        native = Processor.from_pretrained("Qwen3", cache_dir=CACHE, local_files_only=True)
        messages = [
            {"role": "user", "content": [{"type": "image", "image": "https://example.com/no.jpg"}]}
        ]
        with patch("qwen_mm._media._load_source", side_effect=AssertionError("unexpected I/O")):
            with self.assertRaisesRegex(ValueError, "return_tensors"):
                native.prepare(messages, return_tensors="invalid")
            with self.assertRaisesRegex(ValueError, "positive integer"):
                native.prepare(messages, min_pixels=True)


if __name__ == "__main__":
    unittest.main()
