"""vLLM v0.23.0 prepared-image processor prototype.

This module deliberately accepts qwen-mm *prepared pixels*, not post-encoder
``image_embeds``. It replaces the registered Qwen processor and data parser so
vLLM can retain its normal field slicing, prompt replacement, hashing, cache,
IPC, and vision-encoder paths without calling Hugging Face preprocessing.

The prototype is image-only. Video needs the same parser shape plus Qwen3-VL's
timestamp-bearing replacement metadata and is intentionally left to the
production adapter after the image seam has been proven.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ProcessingInfo,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import (
    ImageSize,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import PromptReplacement

CONTRACT_ID = "qwen-mm-compat-v1"
QWEN3_VL_PROFILE_FINGERPRINT = "9e2e515f166fdad60e68528aadd7ef2a410724b74855f1b90dfbe1eadcaa7ae1"
QWEN35_PROFILE_FINGERPRINT = "4f870d0c41812a7f5fe318d9ab7db64a805bac569c9fa4570e957752fd69edb5"

_PIXEL_VALUES = "pixel_values"
_GRID = "image_grid_thw"
_CONTRACT = "qwen_mm_contract_id"
_PROFILE = "qwen_mm_profile_fingerprint"
_CACHE_KEYS = "qwen_mm_cache_keys"
_FORBIDDEN_EMBED_FIELDS = frozenset({"image_embeds", "video_embeds"})
_MIN_IMAGE_PATCHES = 16
_MAX_IMAGE_PATCHES = 65_536


def _image_field_config(
    data: Mapping[str, torch.Tensor],
) -> Mapping[str, MultiModalFieldConfig]:
    grid = data.get(_GRID, torch.empty((0, 3), dtype=torch.int64))
    return {
        _PIXEL_VALUES: MultiModalFieldConfig.flat_from_sizes("image", grid.prod(dim=-1)),
        _GRID: MultiModalFieldConfig.batched("image", keep_on_cpu=True),
    }


def _as_cpu_tensor(value: object, *, name: str) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        if not value.flags.writeable:
            raise ValueError(f"{name} must be writable for safe torch.from_numpy use")
        return torch.from_numpy(value)
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be on CPU before vLLM IPC")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        return value
    raise TypeError(f"{name} must be a NumPy array or torch.Tensor")


class QwenMMPreparedImageItems(ModalityDataItems[Mapping[str, object], Mapping[str, torch.Tensor]]):
    """Validated prepared pixels split into vLLM's per-image field items."""

    def __init__(
        self,
        data: Mapping[str, object] | Sequence[Mapping[str, object]],
        *,
        expected_profile_fingerprint: str,
        patch_width: int,
        spatial_merge_size: int,
    ) -> None:
        supplied_items = [data] if isinstance(data, Mapping) else data
        for item in supplied_items:
            forbidden = _FORBIDDEN_EMBED_FIELDS.intersection(item)
            if forbidden:
                raise ValueError(
                    "prepared-pixel input must not contain post-encoder fields: "
                    f"{sorted(forbidden)}"
                )

        if isinstance(data, Sequence) and not isinstance(data, Mapping) and not data:
            normalized: Mapping[str, object] = {
                _PIXEL_VALUES: torch.empty((0, patch_width), dtype=torch.float32),
                _GRID: torch.empty((0, 3), dtype=torch.int64),
                _CONTRACT: CONTRACT_ID,
                _PROFILE: expected_profile_fingerprint,
                _CACHE_KEYS: [],
            }
        else:
            normalized = self._normalize_input(data)

        contract_id = normalized.get(_CONTRACT)
        if contract_id != CONTRACT_ID:
            raise ValueError(f"{_CONTRACT} must be {CONTRACT_ID!r}, got {contract_id!r}")

        profile_fingerprint = normalized.get(_PROFILE)
        if profile_fingerprint != expected_profile_fingerprint:
            raise ValueError(f"{_PROFILE} does not match the registered model profile")

        pixels = _as_cpu_tensor(normalized.get(_PIXEL_VALUES), name=_PIXEL_VALUES)
        grid = _as_cpu_tensor(normalized.get(_GRID), name=_GRID)
        if pixels.dtype != torch.float32:
            raise ValueError(f"{_PIXEL_VALUES} must have dtype float32")
        if grid.dtype != torch.int64:
            raise ValueError(f"{_GRID} must have dtype int64")
        if pixels.ndim != 2 or pixels.shape[1] != patch_width:
            raise ValueError(f"{_PIXEL_VALUES} must have shape [patches, {patch_width}]")
        if grid.ndim != 2 or grid.shape[1] != 3:
            raise ValueError(f"{_GRID} must have shape [images, 3]")
        if bool((grid <= 0).any()):
            raise ValueError(f"{_GRID} entries must be positive")
        if grid.numel() and not bool((grid[:, 0] == 1).all()):
            raise ValueError("prepared still images must have grid_t == 1")
        if grid.numel() and not bool(((grid[:, 1:] % spatial_merge_size) == 0).all()):
            raise ValueError("image grid_h/grid_w must be divisible by spatial_merge_size")
        patches_per_image = grid.prod(dim=-1)
        if patches_per_image.numel() and bool((patches_per_image < _MIN_IMAGE_PATCHES).any()):
            raise ValueError(f"each image must contain at least {_MIN_IMAGE_PATCHES} patches")
        if patches_per_image.numel() and bool((patches_per_image > _MAX_IMAGE_PATCHES).any()):
            raise ValueError(f"each image must contain at most {_MAX_IMAGE_PATCHES} patches")
        expected_rows = int(patches_per_image.sum())
        if pixels.shape[0] != expected_rows:
            raise ValueError(
                f"{_PIXEL_VALUES} has {pixels.shape[0]} rows; grid requires {expected_rows}"
            )

        cache_keys = normalized.get(_CACHE_KEYS)
        if not isinstance(cache_keys, Sequence) or isinstance(cache_keys, str):
            raise TypeError(f"{_CACHE_KEYS} must be a sequence of strings")
        if len(cache_keys) != grid.shape[0]:
            raise ValueError(f"{_CACHE_KEYS} must contain one key per image")
        if not all(isinstance(key, str) and key for key in cache_keys):
            raise ValueError(f"{_CACHE_KEYS} entries must be non-empty strings")

        # Own a stable snapshot before hashing or caching. The processor cache
        # can outlive the caller's request buffers.
        pixels = pixels.clone()
        grid = grid.clone()
        tensor_data = {_PIXEL_VALUES: pixels, _GRID: grid}
        super().__init__(tensor_data, "image")
        kwargs_by_modality = MultiModalKwargsItems.from_hf_inputs(
            BatchFeature(tensor_data), _image_field_config(tensor_data)
        )
        self._kwargs = list(kwargs_by_modality.get("image", []))
        self._pixels = pixels
        self._grid = grid
        self._cache_keys = list(cache_keys)
        self._profile_fingerprint = expected_profile_fingerprint

    @staticmethod
    def _normalize_input(
        data: Mapping[str, object] | Sequence[Mapping[str, object]],
    ) -> Mapping[str, object]:
        if isinstance(data, Mapping):
            return data
        if not isinstance(data, Sequence):
            raise TypeError("prepared image data must be a mapping or sequence")
        first = data[0]
        for item in data:
            if item.get(_CONTRACT) != first.get(_CONTRACT):
                raise ValueError("prepared items have different contract IDs")
            if item.get(_PROFILE) != first.get(_PROFILE):
                raise ValueError("prepared items have different profile fingerprints")

        pixels = torch.cat(
            [_as_cpu_tensor(item.get(_PIXEL_VALUES), name=_PIXEL_VALUES) for item in data]
        )
        grids = torch.cat([_as_cpu_tensor(item.get(_GRID), name=_GRID) for item in data])
        cache_keys: list[str] = []
        for item in data:
            item_keys = item.get(_CACHE_KEYS)
            if not isinstance(item_keys, Sequence) or isinstance(item_keys, str):
                raise TypeError(f"{_CACHE_KEYS} must be a sequence of strings")
            cache_keys.extend(item_keys)
        return {
            _PIXEL_VALUES: pixels,
            _GRID: grids,
            _CONTRACT: first.get(_CONTRACT),
            _PROFILE: first.get(_PROFILE),
            _CACHE_KEYS: cache_keys,
        }

    def get_count(self) -> int:
        return len(self._kwargs)

    def get(self, index: int) -> Mapping[str, object]:
        item = self._kwargs[index].get_data()
        return {
            _PIXEL_VALUES: item[_PIXEL_VALUES],
            _GRID: item[_GRID].unsqueeze(0),
            _CONTRACT: CONTRACT_ID,
            _PROFILE: self._profile_fingerprint,
            _CACHE_KEYS: [self._cache_keys[index]],
        }

    def get_item_for_hash(self, index: int) -> object:
        item = self._kwargs[index].get_data()
        return {
            "contract_id": CONTRACT_ID,
            "profile_fingerprint": self._profile_fingerprint,
            "prepared_cache_key": self._cache_keys[index],
            "pixel_values": item[_PIXEL_VALUES],
            "image_grid_thw": item[_GRID],
        }

    def get_processor_data(self) -> Mapping[str, object]:
        return {_PIXEL_VALUES: self._pixels, _GRID: self._grid}

    def get_passthrough_data(self) -> Mapping[str, object]:
        return {}


class QwenMMPreparedPixelParser(MultiModalDataParser):
    """Parse only qwen-mm prepared image dictionaries for the registered path."""

    def __init__(
        self,
        *,
        expected_profile_fingerprint: str,
        patch_width: int,
        spatial_merge_size: int,
    ) -> None:
        super().__init__()
        self.expected_profile_fingerprint = expected_profile_fingerprint
        self.patch_width = patch_width
        self.spatial_merge_size = spatial_merge_size

    def _parse_image_data(self, data: Any):
        if data is None:
            return None
        return QwenMMPreparedImageItems(
            data,
            expected_profile_fingerprint=self.expected_profile_fingerprint,
            patch_width=self.patch_width,
            spatial_merge_size=self.spatial_merge_size,
        )

    def _parse_video_data(self, data: Any):
        if data is None:
            return None
        raise ValueError("the A6 prepared-pixel prototype is image-only")


class _PreparedInfoMixin:
    expected_profile_fingerprint: str
    expected_model_id: str
    expected_revision: str

    def __init__(self, ctx: object) -> None:
        super().__init__(ctx)
        model_config = self.ctx.model_config
        if model_config.model != self.expected_model_id:
            raise ValueError(
                "qwen-mm prepared-pixel plugin only supports the frozen model "
                f"{self.expected_model_id!r}, got {model_config.model!r}"
            )
        if model_config.revision != self.expected_revision:
            raise ValueError(
                "qwen-mm prepared-pixel plugin requires frozen revision "
                f"{self.expected_revision!r}, got {model_config.revision!r}"
            )

    def get_data_parser(self) -> QwenMMPreparedPixelParser:
        config = self.get_hf_config().vision_config
        patch_width = config.in_channels * config.temporal_patch_size * config.patch_size**2
        return QwenMMPreparedPixelParser(
            expected_profile_fingerprint=self.expected_profile_fingerprint,
            patch_width=patch_width,
            spatial_merge_size=config.spatial_merge_size,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_image_size_with_most_features(self, max_pixels: int | None = None) -> ImageSize:
        if max_pixels is not None and max_pixels != 16_777_216:
            raise ValueError("prepared image profiling does not accept max_pixels overrides")
        return ImageSize(width=4096, height=4096)

    def get_max_image_tokens(self) -> int:
        return _MAX_IMAGE_PATCHES // 4

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        del seq_len, mm_counts
        return {"image": self.get_max_image_tokens()}


class QwenMMPreparedMultiModalProcessor:
    """Mixin applied to vLLM's Qwen3 processor at import time below."""

    def _apply_hf_processor_main(self, *args: object, **kwargs: object):
        prompt_ids, mm_processed_data, _ = super()._apply_hf_processor_main(*args, **kwargs)
        return prompt_ids, mm_processed_data, True

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        del mm_kwargs, tok_kwargs
        tokenizer = self.info.get_tokenizer()
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        return BatchFeature({"input_ids": [prompt_ids], **mm_data})

    def _hf_processor_applies_updates(self, *args: object, **kwargs: object) -> bool:
        del args, kwargs
        # qwen-mm supplies final IDs with visual runs already expanded. vLLM
        # must discover their ranges, not expand a marker a second time.
        return True

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        del hf_processor_mm_kwargs
        return _image_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptReplacement]:
        del mm_items, hf_processor_mm_kwargs
        config = self.info.get_hf_config()
        image_token_id = config.image_token_id
        merge_length = config.vision_config.spatial_merge_size**2

        def replacement(item_idx: int) -> list[int]:
            grid = out_mm_kwargs["image"][item_idx][_GRID].data
            return [image_token_id] * (int(grid.prod()) // merge_length)

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=replacement,
            )
        ]


class PreparedQwen3VLInfo(_PreparedInfoMixin, Qwen3VLProcessingInfo):
    """Importable processing-info class for Qwen3-VL worker processes."""

    expected_profile_fingerprint = QWEN3_VL_PROFILE_FINGERPRINT
    expected_model_id = "Qwen/Qwen3-VL-8B-Instruct"
    expected_revision = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


class PreparedQwen35Info(_PreparedInfoMixin, Qwen3_5ProcessingInfo):
    """Importable processing-info class for dense Qwen3.5 worker processes."""

    expected_profile_fingerprint = QWEN35_PROFILE_FINGERPRINT
    expected_model_id = "Qwen/Qwen3.5-9B"
    expected_revision = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


class PreparedQwenProcessor(QwenMMPreparedMultiModalProcessor, Qwen3VLMultiModalProcessor):
    """Prepared processor shared by the two models, matching pinned vLLM's MRO."""


class PreparedDummyInputsBuilder(Qwen3VLDummyInputsBuilder):
    """Construct maximum-size prepared inputs for vLLM's profiling pass."""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, object],
    ) -> Mapping[str, object]:
        del seq_len
        if mm_options:
            raise ValueError("prepared image profiling does not support multimodal overrides")
        num_images = mm_counts.get("image", 0)
        config = self.info.get_hf_config().vision_config
        width, height = self.info.get_image_size_with_most_features()
        grid = torch.tensor(
            [1, height // config.patch_size, width // config.patch_size],
            dtype=torch.int64,
        ).repeat(num_images, 1)
        patch_width = config.in_channels * config.temporal_patch_size * config.patch_size**2
        num_patches = int(grid.prod(dim=-1).sum())
        return {
            "image": {
                _PIXEL_VALUES: torch.zeros((num_patches, patch_width), dtype=torch.float32),
                _GRID: grid,
                _CONTRACT: CONTRACT_ID,
                _PROFILE: self.info.expected_profile_fingerprint,
                _CACHE_KEYS: [f"qwen-mm-dummy-{i}" for i in range(num_images)],
            }
        }


def register() -> None:
    """Register the prepared-pixel processor for the two frozen profiles."""
    if getattr(Qwen3VLForConditionalGeneration, "_qwen_mm_prepared_pixels", False):
        return

    MULTIMODAL_REGISTRY.register_processor(
        PreparedQwenProcessor,
        info=PreparedQwen3VLInfo,
        dummy_inputs=PreparedDummyInputsBuilder,
    )(Qwen3VLForConditionalGeneration)
    MULTIMODAL_REGISTRY.register_processor(
        PreparedQwenProcessor,
        info=PreparedQwen35Info,
        dummy_inputs=PreparedDummyInputsBuilder,
    )(Qwen3_5ForConditionalGeneration)

    Qwen3VLForConditionalGeneration._qwen_mm_prepared_pixels = True
    Qwen3_5ForConditionalGeneration._qwen_mm_prepared_pixels = True
