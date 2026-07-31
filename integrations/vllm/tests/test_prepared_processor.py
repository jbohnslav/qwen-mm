from __future__ import annotations

import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from qwen_mm_vllm import (
    CONTRACT_ID,
    QWEN3_VL_PROFILE_FINGERPRINT,
    QWEN35_PROFILE_FINGERPRINT,
    PreparedDummyInputsBuilder,
    PreparedQwen3VLInfo,
    PreparedQwen35Info,
    PreparedQwenProcessor,
    QwenMMPreparedMultiModalProcessor,
    QwenMMPreparedPixelParser,
    register,
)
from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor
from vllm.multimodal.cache import (
    MultiModalProcessorOnlyCache,
    MultiModalProcessorSenderCache,
    MultiModalReceiverCache,
)
from vllm.multimodal.processing import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs

QWEN3_VL_IMAGE_TOKEN_ID = 151655
QWEN35_IMAGE_TOKEN_ID = 248056


class SpyTokenizer:
    def __init__(self, image_token_id: int) -> None:
        self.image_token_id = image_token_id

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        if text == "":
            return []
        if text and text == "<image>" * text.count("<image>"):
            return [self.image_token_id] * text.count("<image>")
        raise AssertionError(f"unexpected tokenization request: {text!r}")

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        del kwargs
        return "".join(
            "<image>" if item == self.image_token_id else str(item) for item in token_ids
        )


class SpyInfo:
    model_id = "Qwen/Qwen3-VL-8B-Instruct"

    def __init__(
        self,
        *,
        profile_fingerprint: str = QWEN3_VL_PROFILE_FINGERPRINT,
        image_token_id: int = QWEN3_VL_IMAGE_TOKEN_ID,
    ) -> None:
        self.tokenizer = SpyTokenizer(image_token_id)
        self.config = SimpleNamespace(
            image_token_id=image_token_id,
            vision_config=SimpleNamespace(
                in_channels=3,
                temporal_patch_size=2,
                patch_size=16,
                spatial_merge_size=2,
            ),
        )
        self.data_parser = QwenMMPreparedPixelParser(
            expected_profile_fingerprint=profile_fingerprint,
            patch_width=1536,
            spatial_merge_size=2,
        )

    def get_data_parser(self) -> QwenMMPreparedPixelParser:
        return self.data_parser

    def parse_mm_data(self, data, *, validate: bool = True):
        del validate
        return self.data_parser.parse_mm_data(data)

    def get_tokenizer(self) -> SpyTokenizer:
        return self.tokenizer

    def get_hf_config(self):
        return self.config

    def get_hf_processor(self, **kwargs: object):
        del kwargs
        raise AssertionError("Hugging Face processor must not be invoked")


class DummyInputs:
    def get_dummy_text(self, mm_counts) -> str:
        return "<image>" * mm_counts.get("image", 0)


class PreparedProcessor(QwenMMPreparedMultiModalProcessor, Qwen3VLMultiModalProcessor):
    pass


class CacheModelConfig:
    def get_multimodal_config(self):
        return SimpleNamespace(mm_processor_cache_gb=0.01)


def prepared_data(
    *,
    cache_key: str = "source-sha256:abc",
    profile_fingerprint: str = QWEN3_VL_PROFILE_FINGERPRINT,
    grid: list[list[int]] | None = None,
) -> dict[str, object]:
    grid = grid or [[1, 4, 4]]
    num_patches = sum(np.prod(item) for item in grid)
    pixels = np.arange(num_patches * 1536, dtype=np.float32).reshape(num_patches, 1536)
    return {
        "pixel_values": pixels,
        "image_grid_thw": np.array(grid, dtype=np.int64),
        "qwen_mm_contract_id": CONTRACT_ID,
        "qwen_mm_profile_fingerprint": profile_fingerprint,
        "qwen_mm_cache_keys": [f"{cache_key}:{i}" for i in range(len(grid))],
    }


def apply_prepared(
    processor: PreparedProcessor,
    data: dict[str, object],
    *,
    image_token_id: int = QWEN3_VL_IMAGE_TOKEN_ID,
    prompt: list[int] | None = None,
):
    mm_items = processor.info.parse_mm_data({"image": data})
    inputs = ProcessorInputs(
        prompt=prompt or [11, *([image_token_id] * 4), 12],
        mm_data_items=mm_items,
    )
    return processor.apply(inputs, TimingContext(enabled=False)), mm_items


@pytest.mark.parametrize(
    ("profile_fingerprint", "image_token_id"),
    [
        (QWEN3_VL_PROFILE_FINGERPRINT, QWEN3_VL_IMAGE_TOKEN_ID),
        (QWEN35_PROFILE_FINGERPRINT, QWEN35_IMAGE_TOKEN_ID),
    ],
)
def test_prepared_pixels_follow_qwen_vllm_path_without_hf_processor(
    monkeypatch, profile_fingerprint, image_token_id
):
    def fail_auto_processor(*args: object, **kwargs: object):
        del args, kwargs
        raise AssertionError("AutoProcessor must not be invoked")

    monkeypatch.setattr("transformers.AutoProcessor.from_pretrained", fail_auto_processor)
    monkeypatch.setitem(
        sys.modules,
        "qwen_vl_utils",
        SimpleNamespace(process_vision_info=fail_auto_processor),
    )
    processor = PreparedProcessor(
        SpyInfo(
            profile_fingerprint=profile_fingerprint,
            image_token_id=image_token_id,
        ),
        DummyInputs(),
    )
    data = prepared_data(profile_fingerprint=profile_fingerprint)
    pixel_pointer = data["pixel_values"].__array_interface__["data"][0]

    output, mm_items = apply_prepared(processor, data, image_token_id=image_token_id)
    image_item = output["mm_kwargs"]["image"][0]

    assert output["prompt_token_ids"] == [11, *([image_token_id] * 4), 12]
    assert set(image_item) == {"pixel_values", "image_grid_thw"}
    assert "image_embeds" not in image_item
    assert image_item["pixel_values"].data.shape == (16, 1536)
    assert image_item["image_grid_thw"].data.tolist() == [1, 4, 4]
    assert image_item["pixel_values"].data.data_ptr() != pixel_pointer
    assert mm_items["image"][0]["pixel_values"].data_ptr() != pixel_pointer
    assert output["mm_placeholders"]["image"][0].offset == 1
    assert output["mm_placeholders"]["image"][0].length == 4
    assert len(output["mm_hashes"]["image"]) == 1


def test_processor_only_cache_reuses_normal_vllm_item_and_prompt_metadata():
    cache = MultiModalProcessorOnlyCache(CacheModelConfig())
    processor = PreparedProcessor(SpyInfo(), DummyInputs(), cache=cache)

    first, _ = apply_prepared(processor, prepared_data())
    second, _ = apply_prepared(processor, prepared_data())

    assert first["mm_hashes"] == second["mm_hashes"]
    assert first["mm_placeholders"] == second["mm_placeholders"]
    assert second["mm_kwargs"]["image"][0]["pixel_values"].data.shape == (16, 1536)


def test_sender_and_receiver_caches_omit_and_restore_warm_item():
    sender = MultiModalProcessorSenderCache(CacheModelConfig())
    receiver = MultiModalReceiverCache(CacheModelConfig())
    processor = PreparedProcessor(SpyInfo(), DummyInputs(), cache=sender)

    first, _ = apply_prepared(processor, prepared_data())
    second, _ = apply_prepared(processor, prepared_data())
    item_hash = first["mm_hashes"]["image"][0]
    cold_item = first["mm_kwargs"]["image"][0]

    assert second["mm_kwargs"]["image"][0] is None
    assert receiver.get_and_update_item(cold_item, item_hash) is cold_item
    assert receiver.get_and_update_item(None, item_hash) is cold_item
    assert first["mm_placeholders"] == second["mm_placeholders"]


def test_cache_identity_binds_profile_key_pixels_and_grid():
    processor = PreparedProcessor(SpyInfo(), DummyInputs())
    first, _ = apply_prepared(processor, prepared_data(cache_key="one"))
    second, _ = apply_prepared(processor, prepared_data(cache_key="two"))
    changed = prepared_data(cache_key="one")
    changed["pixel_values"][0, 0] = -1
    third, _ = apply_prepared(processor, changed)
    assert first["mm_hashes"]["image"] != second["mm_hashes"]["image"]
    assert first["mm_hashes"]["image"] != third["mm_hashes"]["image"]


def test_cache_owns_snapshot_against_caller_mutation():
    processor = PreparedProcessor(SpyInfo(), DummyInputs())
    data = prepared_data()
    output, _ = apply_prepared(processor, data)
    cached_pixels = output["mm_kwargs"]["image"][0]["pixel_values"].data

    data["pixel_values"].fill(-1)
    assert not bool((cached_pixels == -1).any())


def test_final_ids_find_unequal_multi_image_ranges_without_expansion():
    processor = PreparedProcessor(SpyInfo(), DummyInputs())
    data = prepared_data(grid=[[1, 4, 4], [1, 4, 8]])
    prompt = [11, *([QWEN3_VL_IMAGE_TOKEN_ID] * 4), 12, *([QWEN3_VL_IMAGE_TOKEN_ID] * 8), 13]

    output, _ = apply_prepared(processor, data, prompt=prompt)

    assert output["prompt_token_ids"] == prompt
    assert [(item.offset, item.length) for item in output["mm_placeholders"]["image"]] == [
        (1, 4),
        (6, 8),
    ]
    assert [item["pixel_values"].data.shape[0] for item in output["mm_kwargs"]["image"]] == [16, 32]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("image_embeds", torch.empty((4, 8)), "post-encoder"),
        ("qwen_mm_contract_id", "wrong", "qwen-mm-compat-v1"),
        ("qwen_mm_profile_fingerprint", "wrong", "registered model profile"),
        ("pixel_values", np.empty((15, 1536), dtype=np.float32), "grid requires 16"),
        ("image_grid_thw", np.array([[1, 256, 512]]), "at most 65536 patches"),
    ],
)
def test_invalid_or_embedding_shaped_input_fails_fast(field, value, message):
    processor = PreparedProcessor(SpyInfo(), DummyInputs())
    data = prepared_data()
    data[field] = value
    with pytest.raises(ValueError, match=message):
        apply_prepared(processor, data)


def test_general_plugin_registration_is_reentrant_and_replaces_qwen_factories():
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

    register()
    vl_factory = Qwen3VLForConditionalGeneration._processor_factory
    qwen35_factory = Qwen3_5ForConditionalGeneration._processor_factory
    register()

    assert vl_factory is Qwen3VLForConditionalGeneration._processor_factory
    assert qwen35_factory is Qwen3_5ForConditionalGeneration._processor_factory
    assert issubclass(vl_factory.processor, Qwen3VLMultiModalProcessor)
    assert issubclass(qwen35_factory.processor, Qwen3VLMultiModalProcessor)
    assert "PreparedQwen3VLInfo" in vl_factory.info.__qualname__
    assert "PreparedQwen35Info" in qwen35_factory.info.__qualname__


@pytest.mark.parametrize(
    ("info_class", "model_id", "revision"),
    [
        (
            PreparedQwen3VLInfo,
            "Qwen/Qwen3-VL-8B-Instruct",
            "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        ),
        (
            PreparedQwen35Info,
            "Qwen/Qwen3.5-9B",
            "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        ),
    ],
)
def test_registered_info_refuses_unfrozen_model_or_revision(info_class, model_id, revision):
    context = SimpleNamespace(model_config=SimpleNamespace(model=model_id, revision=revision))
    assert info_class(context).model_id == model_id

    wrong_model = SimpleNamespace(
        model_config=SimpleNamespace(model="Qwen/other", revision=revision)
    )
    with pytest.raises(ValueError, match="only supports the frozen model"):
        info_class(wrong_model)

    wrong_revision = SimpleNamespace(model_config=SimpleNamespace(model=model_id, revision=None))
    with pytest.raises(ValueError, match="requires frozen revision"):
        info_class(wrong_revision)


def test_dummy_profile_uses_frozen_limits_without_hf_processor():
    class ProfilingInfo:
        expected_profile_fingerprint = QWEN3_VL_PROFILE_FINGERPRINT

        def get_hf_config(self):
            return SimpleNamespace(
                vision_config=SimpleNamespace(
                    in_channels=3,
                    temporal_patch_size=2,
                    patch_size=16,
                )
            )

        def get_image_size_with_most_features(self):
            return 64, 64

        def get_hf_processor(self, **kwargs: object):
            del kwargs
            raise AssertionError("profiling must not load the Hugging Face processor")

    builder = PreparedDummyInputsBuilder(ProfilingInfo())
    image = builder.get_dummy_mm_data(1024, {"image": 1}, {})["image"]

    assert image["pixel_values"].shape == (16, 1536)
    assert bool((image["pixel_values"] == 0).all())
    assert image["image_grid_thw"].tolist() == [[1, 4, 4]]


@pytest.mark.parametrize(
    "registered_class",
    [
        PreparedQwen3VLInfo,
        PreparedQwen35Info,
        PreparedQwenProcessor,
        PreparedDummyInputsBuilder,
    ],
)
def test_general_plugin_classes_are_importable_and_pickle_safe(registered_class):
    assert "<locals>" not in registered_class.__qualname__
    assert pickle.loads(pickle.dumps(registered_class)) is registered_class
