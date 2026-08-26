# Official Qwen examples → qwen-mm Rosetta Stone

This is the short code-comparison guide. Each pair stops at the preprocessing
boundary: the official side produces model inputs, and the qwen-mm side
produces the corresponding NumPy arrays. Model loading, generation, and
decoding are omitted.

The official excerpts are condensed to the relevant lines while preserving
their messages, options, and media order. Links point to immutable upstream
revisions. The exhaustive source audit and support classifications remain in
the [comparison appendix](official-example-comparison-v0.1.md).

If you read only one pair, read [one image from a URL](#2-one-image-from-a-url).
Run one constructor from section 0, then use any numbered pair below.

## 0. Construct the processor

Source: [Qwen3-VL-8B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/README.md#using--transformers-to-chat).

Official Transformers examples:

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
```

qwen-mm:

```python
from qwen_mm import Processor

processor = Processor.from_huggingface_cache("qwen3-vl-8b")
```

Both select Qwen3-VL-8B-Instruct. qwen-mm opens only revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` from the local Hugging Face cache;
it never downloads implicitly. Use `"qwen3.5-9b"` for the pinned Qwen3.5-9B
profile.

## 1. Text-only chat

Source: [Qwen3-VL batch example](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/README.md#batch-inference).

Official:

```python
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}],
    },
    {"role": "user", "content": [{"type": "text", "text": "Who are you?"}]},
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
```

qwen-mm:

```python
prepared = processor.prepare_batch(
    [
        {
            "messages": messages,
            "options": {"add_generation_prompt": True},
        }
    ]
)
inputs = prepared.arrays
```

The qwen-mm output contains `input_ids`, `attention_mask`, and
`mm_token_type_ids`. It contains no placeholder image arrays for a text-only
request.

## 2. One image from a URL

Source: [Qwen3-VL-8B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/README.md#using--transformers-to-chat).

Official:

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
```

qwen-mm:

```python
from urllib.request import urlopen

image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
with urlopen(image_url) as response:
    image_bytes = response.read()

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
inputs = prepared.arrays
```

The only ownership difference is explicit: Transformers may fetch the URL;
qwen-mm requires the caller to supply encoded JPEG/PNG/WebP bytes or a
`uint8` HWC RGB array.

## 3. Multiple images

Source: [Qwen3-VL multi-image example](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/README.md#multi-image-inference).

Official:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///path/to/image1.jpg"},
            {"type": "image", "image": "file:///path/to/image2.jpg"},
            {
                "type": "text",
                "text": "Identify the similarities between these images.",
            },
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
```

qwen-mm:

```python
from pathlib import Path

images = [Path("image1.jpg").read_bytes(), Path("image2.jpg").read_bytes()]
prepared = processor.prepare_batch(
    [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": 0},
                        {"type": "image", "image": 1},
                        {
                            "type": "text",
                            "text": "Identify the similarities between these images.",
                        },
                    ],
                }
            ],
            "images": images,
            "options": {"add_generation_prompt": True},
        }
    ]
)
inputs = prepared.arrays
```

The integer in each image content item is the position in that request's
`images` list. Text and image items may be interleaved in any order. Repeating
an integer repeats the same supplied image at another prompt position.

## 4. Heterogeneous batch

Source: [Qwen3-VL batch example](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/README.md#batch-inference).

Official:

```python
processor.tokenizer.padding_side = "left"

messages1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///path/to/image1.jpg"},
            {"type": "image", "image": "file:///path/to/image2.jpg"},
            {
                "type": "text",
                "text": "What are the common elements in these pictures?",
            },
        ],
    }
]
messages2 = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}],
    },
    {"role": "user", "content": [{"type": "text", "text": "Who are you?"}]},
]

inputs = processor.apply_chat_template(
    [messages1, messages2],
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    padding=True,
)
```

qwen-mm:

```python
from pathlib import Path

image1_bytes = Path("/path/to/image1.jpg").read_bytes()
image2_bytes = Path("/path/to/image2.jpg").read_bytes()

prepared = processor.prepare_batch(
    [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": 0},
                        {"type": "image", "image": 1},
                        {
                            "type": "text",
                            "text": "What are the common elements in these pictures?",
                        },
                    ],
                }
            ],
            "images": [image1_bytes, image2_bytes],
            "options": {"add_generation_prompt": True},
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Who are you?"}],
                },
            ],
            "options": {"add_generation_prompt": True},
        },
    ]
)
inputs = prepared.arrays
```

Both preprocess a multi-image row and a text-only row together. qwen-mm always
uses the frozen compatibility contract's right padding. It does not reproduce
the official example's mutable left-padding step, so direct batched generation
must account for that downstream.

## 5. qwen-vl-utils path loading and explicit image size

Source: [`qwen-vl-utils` Qwen3-VL example](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/qwen-vl-utils/README.md#qwen3vl).

Official:

```python
from qwen_vl_utils import process_vision_info

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "file:///path/to/your/image.jpg",
                "resized_height": 280,
                "resized_width": 420,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos, video_kwargs = process_vision_info(
    messages,
    image_patch_size=16,
    return_video_kwargs=True,
    return_video_metadata=True,
)
inputs = processor(
    text=text,
    images=images,
    videos=videos,
    return_tensors="pt",
    do_resize=False,
    **video_kwargs,
)
```

qwen-mm:

```python
from pathlib import Path

image_bytes = Path("/path/to/your/image.jpg").read_bytes()
prepared = processor.prepare_batch(
    [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": 0,
                            "resized_height": 280,
                            "resized_width": 420,
                        },
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ],
            "images": [image_bytes],
            "options": {"add_generation_prompt": True},
        }
    ]
)
inputs = prepared.arrays
```

One qwen-mm call replaces chat rendering, `process_vision_info`, image
resizing, tokenization, and tensor assembly. `min_pixels` and `max_pixels` use
the same per-image position.

## 6. Qwen3.5 non-thinking mode

Source: [Qwen3.5-9B non-thinking example](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/README.md#disable-thinking).

Official serving request:

```python
from openai import OpenAI

client = OpenAI()
image_url = (
    "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/RealWorld/RealWorld-04.png"
)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": "Where is this?"},
        ],
    }
]
chat_response = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B",
    messages=messages,
    max_tokens=32768,
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
```

qwen-mm local preprocessing:

```python
from urllib.request import urlopen

from qwen_mm import Processor

image_url = (
    "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/RealWorld/RealWorld-04.png"
)
with urlopen(image_url) as response:
    image_bytes = response.read()

processor = Processor.from_huggingface_cache("qwen3.5-9b")
prepared = processor.prepare_batch(
    [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": 0},
                        {"type": "text", "text": "Where is this?"},
                    ],
                }
            ],
            "images": [image_bytes],
            "options": {
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        }
    ]
)
inputs = prepared.arrays
```

Sampling options such as `top_k` belong to generation and are not processor
inputs. `enable_thinking` changes chat rendering, so qwen-mm accepts it.

## 7. LLaMA-Factory multimodal dataset row

Source: [LLaMA-Factory `mllm_demo.json`](https://github.com/hiyouga/LLaMA-Factory/blob/a18110d2f064b1518ac313eeb2ba980946467b4d/data/mllm_demo.json).

Official dataset shape:

```python
row = {
    "messages": [
        {"role": "user", "content": "<image>Who are they?"},
        {
            "role": "assistant",
            "content": "They're Kane and Gretzka from Bayern Munich.",
        },
        {"role": "user", "content": "What are they doing?<image>"},
        {
            "role": "assistant",
            "content": "They are celebrating on the soccer field.",
        },
    ],
    "images": ["mllm_demo_data/1.jpg", "mllm_demo_data/1.jpg"],
}
```

qwen-mm request after caller-owned dataset/path loading:

```python
from pathlib import Path

image_bytes = Path("mllm_demo_data/1.jpg").read_bytes()
prepared = processor.prepare_batch(
    [
        {
            "messages": [
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
            ],
            "images": [image_bytes, image_bytes],
        }
    ]
)
```

The `<image>` placeholders become ordered numeric content items. Dataset
loading and training-label construction stay in LLaMA-Factory or the caller.
ModelScope SWIFT's separate `images`/`videos` columns map the same way.

## 8. Video

Source: [Qwen3-VL video example](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/README.md#video-inference).

Official:

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/"
                "Qwen2-VL/space_woaudio.mp4",
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
```

qwen-mm v0.1:

```python
from pathlib import Path

from qwen_mm import UnsupportedMediaError

video_bytes = Path("/path/to/video.mp4").read_bytes()
try:
    processor.prepare_batch(
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": 0},
                            {"type": "text", "text": "Describe this video."},
                        ],
                    }
                ],
                "videos": [video_bytes],
            }
        ]
    )
except UnsupportedMediaError:
    pass  # Video preprocessing is deferred beyond v0.1.
```

There is deliberately no equivalent video preprocessing path in v0.1. Frame
sampling, temporal metadata, and video decoding remain deferred; extracted
still frames may be submitted as independent images but are not treated as a
video.

## 9. vLLM and SGLang HTTP examples

Sources: [vLLM multi-image example](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/examples/generate/multimodal/vision_language_multi_image_offline.py)
and [SGLang Qwen3-VL cookbook](https://github.com/sgl-project/sglang/blob/2935bb8e79e669b71aa4fef3b412fa25bc656c25/docs/cookbook/autoregressive/Qwen/Qwen3-VL.mdx).

These examples send OpenAI-compatible messages to a running model server or
pass media through a framework-owned model runner. qwen-mm is a local
preprocessor, not a server, HTTP client, or model runner, so there is no honest
one-line substitution for those transport and generation calls. Their text
and still-image message order maps to sections 1–3 above; a production
prepared-pixel adapter is deferred beyond v0.1.

## Field and call cheat sheet

| Official example | qwen-mm |
| --- | --- |
| `AutoProcessor.from_pretrained(model_id)` | `Processor.from_huggingface_cache(profile)` |
| URL, path, data URI, or PIL object in the message | Caller loads bytes/RGB; message holds an integer image reference |
| `processor.apply_chat_template(...)` | `processor.prepare_batch([request])` |
| `process_vision_info(messages)` | Included in `prepare_batch` for still images |
| `processor(..., images=..., return_tensors="pt")` | Included in `prepare_batch`; arrays are NumPy |
| `min_pixels`, `max_pixels`, or explicit dimensions on an image | Same option names on the numeric image content item |
| `padding=True` plus mutable `padding_side` | Automatic fixed right padding |
| `chat_template_kwargs={"enable_thinking": ...}` | `options={"enable_thinking": ...}` |
| Video URL/path/frames | `UnsupportedMediaError` in v0.1 |
| vLLM/SGLang request or generation | Out of scope; qwen-mm only prepares local arrays |
