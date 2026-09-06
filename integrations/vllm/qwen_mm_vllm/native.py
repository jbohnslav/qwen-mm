"""Native still-image processing for ordinary vLLM image requests.

vLLM owns HTTP media loading and chat rendering. Its decoded RGB images enter
qwen-mm once as raw arrays; qwen-mm owns resize, normalization and patch layout.
The server's tokenizer and placeholder/cache machinery remain authoritative.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from functools import cached_property
from typing import Any

import numpy as np
import torch
from qwen_mm import Processor
from transformers.feature_extraction_utils import BatchFeature
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import ImageSize, MultiModalDataParser

from .adapter import (
    PreparedQwen3VLInfo,
    PreparedQwen35Info,
    QwenMMPreparedMultiModalProcessor,
    _image_field_config,
)


def pixel_budgets(kwargs: Mapping[str, Any]) -> tuple[int, int]:
    unexpected = set(kwargs) - {"min_pixels", "max_pixels", "size"}
    if unexpected:
        raise ValueError(f"qwen-mm native image processor does not support {sorted(unexpected)}")
    size = kwargs.get("size", {})
    if not isinstance(size, Mapping) or set(size) - {"shortest_edge", "longest_edge"}:
        raise ValueError("size must contain only shortest_edge/longest_edge pixel budgets")
    minimum = kwargs.get("min_pixels", size.get("shortest_edge", 65536))
    maximum = kwargs.get("max_pixels", size.get("longest_edge", 16777216))
    if any(type(value) is not int or value <= 0 for value in (minimum, maximum)):
        raise ValueError("pixel budgets must be positive integers")
    if maximum > 16777216 or minimum < 4096:
        raise ValueError("supported pixel budgets are 4096 through 16777216")
    if minimum > maximum:
        raise ValueError("min_pixels must not exceed max_pixels")
    return minimum, maximum


class NativeImageParser(MultiModalDataParser):
    def _parse_image_data(self, data):
        if isinstance(data, dict):
            raise ValueError(
                "native image mode expects decoded images, not embeddings/prepared data"
            )
        return super()._parse_image_data(data)

    def _parse_video_data(self, data):
        if data is not None:
            raise ValueError("qwen-mm native server supports still images only")
        return None


class _NativeInfoMixin:
    @cached_property
    def native_processor(self):
        return Processor.from_pretrained(
            self.expected_model_id,
            cache_dir=os.environ.get("QWEN_MM_CACHE_DIR"),
            thread_budget=int(os.environ.get("QWEN_MM_THREADS", "1")),
        )

    def get_data_parser(self):
        return NativeImageParser()

    def get_image_size_with_most_features(self, max_pixels=None):
        _, maximum = pixel_budgets(self.ctx.get_merged_mm_kwargs({}))
        if max_pixels is not None:
            maximum = min(maximum, max_pixels)
        # Maximize attainable factor-aligned area under the budget while
        # respecting the native 200:1 aspect-ratio guard. This also covers
        # non-square budgets instead of underprofiling with floor(sqrt()).
        factor = self.get_hf_config().vision_config.patch_size * 2
        cells = maximum // factor**2
        width, height = max(
            ((min(cells // h, 200 * h), h) for h in range(1, math.isqrt(cells) + 1)),
            key=lambda pair: (pair[0] * pair[1], -abs(pair[0] - pair[1])),
        )
        return ImageSize(width=width * factor, height=height * factor)

    def get_max_image_tokens(self):
        _, maximum = pixel_budgets(self.ctx.get_merged_mm_kwargs({}))
        return maximum // (self.get_hf_config().vision_config.patch_size * 2) ** 2


class NativeQwen3VLInfo(_NativeInfoMixin, PreparedQwen3VLInfo):
    pass


class NativeQwen35Info(_NativeInfoMixin, PreparedQwen35Info):
    pass


class NativeDummyInputsBuilder(Qwen3VLDummyInputsBuilder):
    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        width, height = self.info.get_image_size_with_most_features()
        return {
            "image": self._get_dummy_images(
                width=width,
                height=height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            )
        }


class NativeImageProcessor(Qwen3VLMultiModalProcessor):
    _get_prompt_updates = QwenMMPreparedMultiModalProcessor._get_prompt_updates

    def _hf_processor_applies_updates(self, *args, **kwargs):
        return False

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        return _image_field_config(hf_inputs)

    def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        minimum, maximum = pixel_budgets(mm_kwargs)
        if set(mm_data) - {"images"}:
            raise ValueError("qwen-mm native server accepts only still-image processor data")
        result = {
            "input_ids": [
                self.info.get_tokenizer().encode(
                    prompt, **{"add_special_tokens": False, **tok_kwargs}
                )
            ]
        }
        images = mm_data.get("images", [])
        if images:
            arrays = [np.asarray(image) for image in images]
            if any(
                array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3
                for array in arrays
            ):
                raise ValueError("vLLM must supply uint8 RGB images to qwen-mm")
            # The public batch API supplies the canonical image pipeline. Its
            # synthetic image-only prompt is discarded; actual chat/token
            # semantics above remain vLLM's, including its cache replacement.
            prepared = self.info.native_processor.prepare(
                [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": i} for i in range(len(arrays))],
                    }
                ],
                images=arrays,
                min_pixels=minimum,
                max_pixels=maximum,
            )
            result.update(
                pixel_values=torch.from_numpy(prepared["pixel_values"]),
                image_grid_thw=torch.from_numpy(prepared["image_grid_thw"]),
            )
        return BatchFeature(result)


def register_native() -> None:
    if getattr(Qwen3VLForConditionalGeneration, "_qwen_mm_native_images", False):
        return
    for model, info in (
        (Qwen3VLForConditionalGeneration, NativeQwen3VLInfo),
        (Qwen3_5ForConditionalGeneration, NativeQwen35Info),
    ):
        MULTIMODAL_REGISTRY.register_processor(
            NativeImageProcessor, info=info, dummy_inputs=NativeDummyInputsBuilder
        )(model)
        model._qwen_mm_native_images = True
