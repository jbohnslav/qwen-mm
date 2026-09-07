"""Exercise stock vLLM expansion/cache with real native pixels, without a GPU."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from PIL import Image
from qwen_mm import Processor
from qwen_mm_vllm.native import NativeImageParser, NativeImageProcessor, pixel_budgets
from test_prepared_processor import CacheModelConfig, DummyInputs, SpyInfo
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.processing import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs


class CountingNative:
    def __init__(self, model):
        self.processor = Processor.from_pretrained(model)
        self.calls = 0

    def prepare(self, *args, **kwargs):
        self.calls += 1
        return self.processor.prepare(*args, **kwargs)


@pytest.fixture(params=[("Qwen/Qwen3-VL-8B-Instruct", 151655), ("Qwen/Qwen3.5-9B", 248056)])
def processor(request):
    model, token = request.param
    info = SpyInfo(image_token_id=token)
    from types import SimpleNamespace

    info.ctx = SimpleNamespace(get_merged_mm_kwargs=lambda kwargs: kwargs)
    info.model_id = model
    info.data_parser = NativeImageParser()
    info.native_processor = CountingNative(model)
    return NativeImageProcessor(
        info, DummyInputs(), cache=MultiModalProcessorOnlyCache(CacheModelConfig())
    )


def apply(processor, images, kwargs=None):
    token = processor.info.config.image_token_id
    prompt = [11]
    for i in range(len(images)):
        prompt += [token, 12 + i]
    items = processor.info.parse_mm_data({"image": images} if images else {})
    return processor.apply(
        ProcessorInputs(prompt=prompt, mm_data_items=items, hf_processor_mm_kwargs=kwargs or {}),
        TimingContext(enabled=False),
    )


def test_native_pixels_expansion_cache_and_ownership(processor):
    arrays = [np.full((256, 256, 3), 33, np.uint8), np.full((256, 512, 3), 199, np.uint8)]
    images = [Image.fromarray(array) for array in arrays]
    first = apply(processor, images)
    assert processor.info.native_processor.calls == 1
    second = apply(processor, images)
    assert processor.info.native_processor.calls == 1
    assert first["mm_hashes"] == second["mm_hashes"]
    assert [(x.offset, x.length) for x in first["mm_placeholders"]["image"]] == [(1, 64), (66, 128)]
    token = processor.info.config.image_token_id
    assert first["prompt_token_ids"] == [11, *([token] * 64), 12, *([token] * 128), 13]
    for item, value, grid in zip(
        first["mm_kwargs"]["image"], [33, 199], [[1, 16, 16], [1, 16, 32]], strict=True
    ):
        assert set(item) == {"pixel_values", "image_grid_thw"}
        assert item["image_grid_thw"].data.tolist() == grid
        expected = torch.tensor(value / 255 * 2 - 1, dtype=torch.float32)
        torch.testing.assert_close(
            item["pixel_values"].data,
            expected.expand_as(item["pixel_values"].data),
            atol=2e-7,
            rtol=0,
        )
    snapshot = first["mm_kwargs"]["image"][0]["pixel_values"].data.clone()
    images[0].paste((0, 0, 0), (0, 0, 256, 256))
    changed = apply(processor, images)
    assert changed["mm_hashes"]["image"][0] != first["mm_hashes"]["image"][0]
    torch.testing.assert_close(first["mm_kwargs"]["image"][0]["pixel_values"].data, snapshot)
    resized = apply(processor, images, {"max_pixels": 65536})
    assert resized["mm_placeholders"]["image"][1].length < 128
    assert resized["mm_hashes"] != changed["mm_hashes"]


def test_text_only_and_concurrent(processor):
    assert apply(processor, [])["prompt_token_ids"] == [11]
    assert processor.info.native_processor.calls == 0
    image = Image.new("RGB", (256, 256), "red")
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(lambda _: apply(processor, [image]), range(4)))
    assert all(output["mm_placeholders"]["image"][0].length == 64 for output in outputs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"do_resize": False},
        {"max_pixels": True},
        {"min_pixels": 100, "max_pixels": 50},
        {"size": {"bad": 1}},
    ],
)
def test_reject_unsupported_options(kwargs):
    with pytest.raises(ValueError):
        pixel_budgets(kwargs)


def test_no_embeddings_or_video():
    parser = NativeImageParser()
    with pytest.raises(ValueError, match="decoded images"):
        parser.parse_mm_data({"image": {"image_embeds": torch.zeros(1)}})
    with pytest.raises(ValueError, match="still images"):
        parser.parse_mm_data({"video": np.zeros((2, 32, 32, 3), np.uint8)})


def test_aligned_pixels_match_pinned_transformers(processor):
    from qwen_mm_vllm.adapter import PreparedQwen3VLInfo, PreparedQwen35Info
    from transformers import AutoProcessor

    revision = (
        PreparedQwen35Info if "3.5" in processor.info.model_id else PreparedQwen3VLInfo
    ).expected_revision
    reference = AutoProcessor.from_pretrained(processor.info.model_id, revision=revision)
    images = [
        Image.fromarray(np.random.default_rng(6).integers(0, 256, (256, 512, 3), dtype=np.uint8))
    ]
    native = apply(processor, images)
    expected = reference.image_processor(images=images, return_tensors="pt")
    actual = native["mm_kwargs"]["image"][0]
    torch.testing.assert_close(
        actual["pixel_values"].data, expected["pixel_values"], atol=2e-7, rtol=0
    )
    assert actual["image_grid_thw"].data.tolist() == expected["image_grid_thw"][0].tolist()


def test_profiling_avoids_hf_processor_and_covers_non_square_budget():
    from types import SimpleNamespace

    from qwen_mm_vllm.native import NativeDummyInputsBuilder, NativeQwen35Info

    class Context:
        model_config = SimpleNamespace(
            model=NativeQwen35Info.expected_model_id, revision=NativeQwen35Info.expected_revision
        )

        def get_merged_mm_kwargs(self, kwargs):
            return {"max_pixels": 768 * 1024}

        def get_hf_config(self, *args):
            return SimpleNamespace(vision_config=SimpleNamespace(patch_size=16))

    info = NativeQwen35Info(Context())
    # Fail-fast: no HF processor may be constructed even during profiling.
    info.get_hf_config = lambda: Context().get_hf_config()
    info.get_hf_processor = lambda **kwargs: pytest.fail("HF processor during profiling")
    width, height = info.get_image_size_with_most_features()
    assert width * height == 768 * 1024
    images = NativeDummyInputsBuilder(info).get_dummy_mm_data(4096, {"image": 2}, {})["image"]
    assert len(images) == 2 and images[0].size == (width, height)


def test_repeated_images_keep_distinct_placeholder_occurrences(processor):
    image = Image.new("RGB", (256, 256), "blue")
    result = apply(processor, [image, image])
    assert [(item.offset, item.length) for item in result["mm_placeholders"]["image"]] == [
        (1, 64),
        (66, 64),
    ]
    assert result["mm_hashes"]["image"][0] == result["mm_hashes"]["image"][1]
    torch.testing.assert_close(
        result["mm_kwargs"]["image"][0]["pixel_values"].data,
        result["mm_kwargs"]["image"][1]["pixel_values"].data,
    )


def test_server_pixel_limits_apply_and_request_options_override(processor):
    from types import SimpleNamespace

    processor.info.ctx = SimpleNamespace(
        get_merged_mm_kwargs=lambda kwargs: {"max_pixels": 65536, **kwargs}
    )
    image = Image.new("RGB", (512, 512), "red")
    server_default = apply(processor, [image])
    assert server_default["mm_kwargs"]["image"][0]["image_grid_thw"].data.tolist() == [1, 16, 16]
    override = apply(processor, [image], {"max_pixels": 262144})
    assert override["mm_kwargs"]["image"][0]["image_grid_thw"].data.tolist() == [1, 32, 32]
