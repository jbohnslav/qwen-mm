# v0.1 official-example comparison

This document compares qwen-mm `0.1.0` with the public Qwen examples that a
new user is most likely to copy. The comparison is about preprocessing only:
model loading, generation, decoding generated tokens, HTTP transport, and
agent orchestration are not features of this wheel.

For the focused code-first view, start with the
[official-example Rosetta Stone](official-example-rosetta-stone-v0.1.md). This
document is the exhaustive source and boundary appendix.

## Frozen sources

The compatibility pins are immutable. SGLang, Qwen-MM-Plugins,
LLaMA-Factory, and ModelScope SWIFT are comparison-only sources: they inform
input ergonomics but do not change the v0.1 parity oracle.

| Source | Version or revision | Relevant examples |
| --- | --- | --- |
| [Qwen3-VL-8B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/README.md) | model revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | Transformers single-image chat |
| [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/README.md) | model revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | text, image, video, non-thinking mode, serving, and agent examples |
| [Qwen3-VL repository](https://github.com/QwenLM/Qwen3-VL/tree/96588727e44c78b25ba03ea03b8e12f7e64fd0da) | commit `96588727e44c78b25ba03ea03b8e12f7e64fd0da` | single image, multi-image, batch, pixel controls, video, visual IDs, deployment, and cookbooks |
| [`qwen-vl-utils`](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/qwen-vl-utils/README.md) | `0.0.14` at the Qwen3-VL commit above | file, URL, base64, PIL, image sizing, video files, and frame lists |
| [Transformers Qwen3-VL docs and tests](https://github.com/huggingface/transformers/tree/a08ace4bbd97e721c98751deec37d87b026acadc/tests/models/qwen3_vl) | `5.14.1`, commit `a08ace4bbd97e721c98751deec37d87b026acadc` | processor outputs, batching/padding, overrides, decoded video, and frame-list video |
| [vLLM multimodal examples](https://github.com/vllm-project/vllm/tree/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/examples/generate/multimodal) | `0.23.0`, commit `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` | text, single/multi-image, video, image-plus-video, offline prompts, and OpenAI-compatible payloads |
| [SGLang Qwen3-VL cookbook](https://github.com/sgl-project/sglang/blob/2935bb8e79e669b71aa4fef3b412fa25bc656c25/docs/cookbook/autoregressive/Qwen/Qwen3-VL.mdx) and [Qwen3.5 cookbook](https://github.com/sgl-project/sglang/blob/2935bb8e79e669b71aa4fef3b412fa25bc656c25/docs/cookbook/autoregressive/Qwen/Qwen3.5.mdx) | comparison commit `2935bb8e79e669b71aa4fef3b412fa25bc656c25` | text, single/multi-image, video, reasoning, and tool-serving payloads |
| [Qwen-MM-Plugins](https://github.com/QwenLM/Qwen-MM-Plugins/tree/ab339d2016ed1da8e9a96c477b067ecc74a6ce59) | `1.0.8`, comparison commit `ab339d2016ed1da8e9a96c477b067ecc74a6ce59` | separate image/video lists, local paths, URLs, dry runs, and agent-facing errors |
| [LLaMA-Factory multimodal data guide](https://github.com/hiyouga/LLaMA-Factory/blob/a18110d2f064b1518ac313eeb2ba980946467b4d/data/README.md) | comparison commit `a18110d2f064b1518ac313eeb2ba980946467b4d` | `<image>`/`<video>` placeholders, ordered media lists, repeated images, multi-turn data, and Qwen3-VL fine-tuning |
| [ModelScope SWIFT Qwen3-VL guide](https://github.com/modelscope/ms-swift/blob/5e198be8a7b4078ff86ade1b7ca3cc018ff971bf/docs/source_en/BestPractices/Qwen3-VL-Best-Practice.md) and [custom dataset guide](https://github.com/modelscope/ms-swift/blob/5e198be8a7b4078ff86ade1b7ca3cc018ff971bf/docs/source_en/Customization/Custom-dataset.md) | comparison commit `5e198be8a7b4078ff86ade1b7ca3cc018ff971bf` | text, multi-image, mixed image/video, frame-list video, tools, agents, and separate media columns |

The two processor revisions, Transformers `5.14.1`, and qwen-vl-utils
`0.0.14` are also recorded in
[`reference/compatibility/v1.json`](../reference/compatibility/v1.json).

### File-level inventory

The source families above were expanded to the following concrete examples so
that a grouped status row does not hide an omitted flow:

- Both pinned Hugging Face model-card READMEs were reviewed end to end. The
  Qwen3-VL repository review includes its root README, qwen-vl-utils README,
  and all thirteen notebooks in `cookbooks/`: 2D grounding, 3D grounding,
  computer use, document parsing, German document OCR, long-document
  understanding, multimodal coding, mobile agent, OCR, omni recognition,
  spatial understanding, think-with-images, and video understanding. The
  task-specific notebooks use the same single-image boundary; long-document
  understanding contributes many ordered page images, while spatial and video
  understanding contribute extracted-frame and true-video cases.
- The Transformers review covers
  `docs/source/en/chat_templating_multimodal.md`,
  `docs/source/en/model_doc/qwen3_vl.md`, and
  `tests/models/qwen3_vl/test_processing_qwen3_vl.py`. Those examples add
  processor-output, batch/padding, override, decoded-video, and frame-list
  coverage beyond the model cards.
- The vLLM review covers
  `vision_language_offline.py`,
  `vision_language_multi_image_offline.py`, and
  `openai_chat_completion_client_for_multimodal.py` under
  `examples/generate/multimodal/`. Other files in that directory target audio,
  omni, encoder-decoder, or non-Qwen model families and add no Qwen v0.1
  preprocessing behavior.
- Both SGLang Qwen cookbooks were reviewed. The Qwen-MM-Plugins review covers
  its root README plus `cookbooks/core/usage.md` and
  `cookbooks/api/usage.md`; capability-specific agent cookbooks build on that
  same media boundary.
- The LLaMA-Factory review covers `data/README.md`, `data/mllm_demo.json`,
  `data/mllm_video_demo.json`, and
  `examples/train_lora/qwen3vl_lora_sft.yaml`. The SWIFT review covers the
  Qwen3-VL best-practice and custom-dataset guides linked above. These sources
  add downstream dataset/media ordering, not a second preprocessing oracle.

## Coverage matrix

“Supported after caller I/O” means qwen-mm accepts the resulting encoded bytes
or `uint8` RGB array but deliberately does not open a path or URL. “Deferred”
means the example stays visible here but is not part of v0.1.

| Official example family | Input behavior being compared | v0.1 status | qwen-mm equivalent or boundary |
| --- | --- | --- | --- |
| Both pinned model cards | Text-only chat | Supported | [Text-only](#text-only) |
| Qwen3-VL model card and Transformers model doc | One remote image inside a user message | Supported after caller I/O | [One image](#one-image) |
| Qwen3-VL README and SGLang cookbook | Multiple images in one message | Supported after caller I/O | [Multiple and interleaved images](#multiple-and-interleaved-images) |
| Qwen3-VL README | Heterogeneous batch: multi-image request plus text-only request | Supported preprocessing with right padding | [Heterogeneous batches and raw RGB](#heterogeneous-batches-and-raw-rgb). Direct Hugging Face generation's documented left-padding mutation is intentionally not emulated. |
| Qwen3-VL README and qwen-vl-utils | Per-image `min_pixels`, `max_pixels`, `resized_height`, and `resized_width` | Supported | [Image and chat options](#image-and-chat-options) |
| qwen-vl-utils and Transformers chat guide | Local path, HTTP URL, data URI, base64 wrapper, or PIL image | Encoded JPEG/PNG/WebP bytes and raw RGB are supported after caller I/O; path/URL/base64-wrapper/PIL loading is intentionally unsupported | [One image](#one-image) |
| Qwen3-VL README | Visual IDs across interleaved image/video conversations | Image IDs supported; video occurrence deferred | `options.add_vision_id=True`; see [Image and chat options](#image-and-chat-options) |
| Transformers docs/tests and qwen-vl-utils | Video URL/path, decoded video array, sampled frame list, FPS or frame-count sampling | Deferred to the video phase | No v0.1 equivalent; the request raises `UnsupportedMediaError` before execution. |
| Qwen3-VL and video-understanding cookbooks | Video preprocessing, temporal metadata, grounding, and frame-list workflows | Deferred to the video phase | Page/frame extraction remains caller-owned; still frames can be submitted as independent images, but are not relabeled as video. |
| Qwen long-document cookbook | PDF download and conversion to many page images | Page-image preprocessing supported after caller conversion; PDF/network handling intentionally unsupported | Use the [multiple-image](#multiple-and-interleaved-images) form after converting pages to encoded bytes or RGB arrays. |
| Qwen image/OCR/grounding/computer-use cookbooks | Task-specific single-image and iterative image inputs | Preprocessing supported; task execution and tool loops out of scope | [One image](#one-image) or [heterogeneous batches](#heterogeneous-batches-and-raw-rgb) |
| Qwen3.5 model card | Thinking and non-thinking chat rendering | Supported for `qwen3.5-9b` | `options.enable_thinking`; see [Image and chat options](#image-and-chat-options) |
| Qwen3.5 model card and SGLang | Tools, tool calls, tool responses, and agent orchestration | Frozen chat rendering is supported; orchestration/model execution out of scope | Supply `options.tools` and normal assistant/tool messages. |
| Transformers processor API | PyTorch tensors returned directly | NumPy output supported; direct Torch construction intentionally unsupported | Convert downstream with `torch.from_numpy` when desired. [Outputs and failures](#outputs-and-failures) |
| vLLM offline examples | Raw image, raw video, or image-plus-video with manual prompt placeholders | Still-image preprocessing supported; video deferred | qwen-mm renders the pinned chat template and returns prepared image arrays, avoiding manual placeholder construction. |
| vLLM OpenAI client | Text, URL/local/base64 image, multi-image, and URL/base64 video payloads | Local still-image preprocessing supported after caller I/O; HTTP transport and video deferred | Use the text/single/multi-image forms below. |
| qwen-mm vLLM plugin prototype | Prepared still pixels handed to pinned vLLM without HF preprocessing | Prototype exists, but production integration is deferred beyond v0.1 | See [`integrations/vllm/README.md`](../integrations/vllm/README.md). |
| SGLang Qwen3-VL/Qwen3.5 cookbooks | OpenAI-compatible text, single/multi-image, video, and reasoning requests | Local text/still-image preprocessing supported after caller I/O; SGLang transport and video deferred | The message ordering maps directly; media transport does not. |
| Qwen-MM-Plugins core/API cookbooks | Agent tools with separate `images`/`videos`, local paths/URLs, dry-run routing, OCR, and grounding | Comparison-only integration inspiration | qwen-mm likewise keeps media buffers separate from message references, but performs no I/O, API call, routing, or model task. |
| LLaMA-Factory multimodal datasets | Multi-turn text with ordered `<image>` placeholders and a separate `images` list, including repeated images | Supported after caller I/O | Replace placeholders with ordered numeric image content items and pass the corresponding encoded buffers or RGB arrays in `images`; training labels and dataset loading remain caller-owned. |
| LLaMA-Factory video datasets | `<video>` placeholders with a separate `videos` list | Deferred to the video phase | Dataset parsing and video decoding are outside v0.1. |
| ModelScope SWIFT datasets and requests | Text, multi-image, mixed image/video, frame-list video, tools, and agent messages with separate media lists | Text/still-image chat rendering supported after caller I/O; video and training integration deferred | The separate-media-list design maps to qwen-mm's request boundary; dataset loading, label construction, orchestration, and video sampling stay outside the wheel. |

## Runnable qwen-mm equivalents

All examples below use the installed wheel and the exact snapshot already in a
local Hugging Face hub cache. `Processor.from_huggingface_cache` never accesses
the network.

### Text-only

```python
from qwen_mm import Processor

processor = Processor.from_huggingface_cache("qwen3-vl-8b")
prepared = processor.prepare_batch(
    [{"messages": [{"role": "user", "content": "Describe the color blue."}]}]
)
assert list(prepared.arrays) == ["input_ids", "attention_mask", "mm_token_type_ids"]
```

### One image

```python
from pathlib import Path

from qwen_mm import Processor

processor = Processor.from_huggingface_cache("qwen3-vl-8b", thread_budget=4)
image_bytes = Path("image.jpg").read_bytes()
prepared = processor.prepare_batch(
    [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": 0},
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ],
            "images": [image_bytes],
            "options": {"add_generation_prompt": True},
        }
    ]
)
pixel_values = prepared.arrays["pixel_values"]
image_grid_thw = prepared.arrays["image_grid_thw"]
```

Bare encoded buffers are signature-detected. Use
`{"data": image_bytes, "format": "jpeg"}` when an explicit format is clearer.
PNG and WebP are also accepted.

### Multiple and interleaved images

```python
from pathlib import Path

images = [Path("before.png").read_bytes(), Path("after.png").read_bytes()]
prepared = processor.prepare_batch(
    [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Before: "},
                        {"type": "image", "image": 0},
                        {"type": "text", "text": " After: "},
                        {"type": "image", "image": 1},
                        {"type": "text", "text": " What changed?"},
                    ],
                }
            ],
            "images": images,
            "options": {"add_generation_prompt": True, "add_vision_id": True},
        }
    ]
)
assert prepared.arrays["image_grid_thw"].shape[0] == 2
```

Every supplied image must be referenced. Reusing an index intentionally reuses
the same supplied buffer at another prompt location.

### Heterogeneous batches and raw RGB

```python
import numpy as np

rgb = np.zeros((64, 96, 3), dtype=np.uint8)
prepared = processor.prepare_batch(
    [
        {"messages": [{"role": "user", "content": "Text-only row"}]},
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": 0},
                        {"type": "text", "text": "Image row"},
                    ],
                }
            ],
            "images": [rgb],
        },
    ]
)
assert prepared.arrays["input_ids"].shape[0] == 2
```

Raw arrays must be `uint8` HWC RGB. Batched text arrays are right-padded under
compatibility contract v1; qwen-mm rejects requests for mutable left-padding
or truncation behavior.

### Image and chat options

Image controls belong on the corresponding content item:

```python
content = [
    {
        "type": "image",
        "image": 0,
        "min_pixels": 50_176,
        "max_pixels": 200_704,
    },
    {"type": "text", "text": "Read this image."},
]
```

`resized_height` and `resized_width` are the exact-size alternative. Request
options support `add_generation_prompt`, `add_vision_id`, and tools. The
`qwen3.5-9b` profile additionally supports `enable_thinking`; setting it
requires `add_generation_prompt=True`.

### Outputs and failures

`PreparedBatch.arrays` contains only official model-input arrays, in frozen
processor order. Conditional image keys are absent from text-only results.
`PreparedBatch.metadata` contains integration-only occurrence, range, cache,
profile, and request-layout metadata.

```python
from qwen_mm import QwenMMError

try:
    processor.prepare_batch(requests)
except QwenMMError as error:
    print(error.category, error.context)
```

Input and compatibility failures use stable qwen-mm exception subclasses.
Missing cache snapshots raise `FileNotFoundError` with the exact pinned
`hf download` command. Processing is all-or-nothing: failed batches do not
return partial arrays.

## Deliberate v0.1 boundary

v0.1 does not fetch URLs, open paths, decode video, accept arbitrary PIL or
Torch objects, run a model, decode generated tokens, or provide a production
vLLM/SGLang adapter. Those omissions avoid hidden I/O and ambiguous ownership.
The official examples remain listed above so later phases can add capabilities
without rewriting release history.
