# Official Qwen examples → qwen-mm Rosetta Stone

This is the code-first comparison. Each official excerpt stops when it has
produced model inputs; the qwen-mm excerpt immediately below produces model inputs. NumPy is the default; the examples request PyTorch
tensors to match the official calls. Model loading and generation remain in
Transformers; qwen-mm also decodes output tokens.

The official snippets are shortened only around the preprocessing boundary.
Their message objects, media order, and processor options are preserved. Links
point to immutable upstream revisions; the broader source inventory is in the
[comparison appendix](official-example-comparison-v0.1.md).

The short version is: keep the official `messages`, replace the processor
constructor, and make one `prepare` call.

## 0. Construct the processor

Source: [Qwen3-VL-8B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/README.md#using--transformers-to-chat).

Official Transformers:

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
```

qwen-mm:

```python
from qwen_mm import Processor

processor = Processor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
```

The short name works too:

```python
processor = Processor.from_pretrained("Qwen3")
```

For Qwen3.5, use `"Qwen3.5"` or `"Qwen/Qwen3.5-9B"`. qwen-mm resolves each
name to its tested immutable revision and downloads only the pinned
processor/tokenizer artifacts through the normal Hugging Face cache—not model
weights. `cache_dir=` selects a cache and `local_files_only=True` makes the call
offline/cache-only.

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

qwen-mm, using the same `messages`:

```python
inputs = processor.prepare(messages, add_generation_prompt=True, return_tensors="pt")
```

`inputs` is a mapping over `input_ids`, `attention_mask`, and
`mm_token_type_ids`, so normal `consumer(**inputs)` keyword expansion works.
With `return_tensors="pt"`, values are Torch tensors and `inputs.to(model.device)`
moves them to the model device. Omit that option for NumPy arrays. Adapter-only
details remain separate in `inputs.metadata`.

For the next step, the [local Transformers consumer example](transformers-consumer-verification.md)
shows the NumPy-to-Torch handoff, generation, and output decoding. Its opt-in
test runs our Qwen3.5 arrays through pinned 0.8B weights on CPU.

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

qwen-mm, using the same `messages`:

```python
inputs = processor.prepare(
    messages, add_generation_prompt=True, min_pixels=65536, return_tensors="pt"
)
```

That URL is the caller's explicit image input, so qwen-mm fetches it directly.
There is no permission flag, private-network filter, security-policy mode, or
warning ceremony. Redirects, a 30-second timeout, byte ceilings, and typed
network failures are ordinary reliability behavior.

Plain paths, `pathlib.Path` values, `file:` URLs, and image data URIs work in
the same `image` field.

The composed `qwen-vl-utils` default used by qwen-mm has a 4,096-pixel minimum.
The direct Transformers call above uses 65,536. The explicit `min_pixels=65536`
matches that call, including small images; it does not mutate `messages`.
Per-image pixel options override shared defaults.

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

The pinned Transformers 5.14.1 loader rejects the upstream `file:` URIs above.
For that official call, use plain filesystem paths instead. qwen-mm accepts
the original `file:` form directly; the suite records this upstream issue and
compares against the official call with only that path adjustment.

qwen-mm, using the same `messages`:

```python
inputs = processor.prepare(
    messages, add_generation_prompt=True, min_pixels=65536, return_tensors="pt"
)
```

Image and text items may be interleaved in any order. Existing callers may
still preload encoded bytes or RGB arrays in a separate `images` list and use
integer image references; direct sources and the deterministic preloaded form
share the same native preprocessing path.

## 4. Heterogeneous batch with left padding

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

The pinned Transformers 5.14.1 loader rejects the upstream `file:` URIs above.
For that official call, use plain filesystem paths instead. qwen-mm accepts
the original `file:` form directly; the suite records this upstream issue and
compares against the official call with only that path adjustment.

qwen-mm, using the same `messages1` and `messages2`:

```python
inputs = processor.prepare_batch(
    [messages1, messages2],
    add_generation_prompt=True,
    padding_side="left",
    min_pixels=65536,
    return_tensors="pt",
)
```

Request wrappers remain available when rows need separate media lists or options;
row options override shared options.
The default remains `padding_side="right"`; `inputs.metadata["padding_side"]`
and each request layout record the selected side and count.

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

qwen-mm, using the same `messages`:

```python
inputs = processor.prepare(messages, add_generation_prompt=True, return_tensors="pt")
```

One call performs chat rendering, still-image loading, resizing, tokenization,
and array assembly. `min_pixels`, `max_pixels`, `resized_height`, and
`resized_width` stay on the same image content item.

## 6. Qwen3.5 OpenAI image content and non-thinking mode

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

qwen-mm local preprocessing, using the same `messages`:

```python
processor = Processor.from_pretrained("Qwen3.5")
inputs = processor.prepare(
    messages,
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt",
)
```

Both the nested `{"url": ...}` object above and a direct `image_url` string are
accepted. Sampling settings such as `temperature` and `top_k` belong to the
model runner, not preprocessing.

## 7. LLaMA-Factory and separate-media dataset rows

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

qwen-mm after the dataset adapter expands the two ordered placeholders:

```python
messages = [
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
]

inputs = processor.prepare(messages, images=row["images"], return_tensors="pt")
```

The paths in `row["images"]` are accepted directly; the adapter does not read
them into bytes. Placeholder expansion, training labels, and dataset loading
remain the training framework's responsibility. ModelScope SWIFT's separate
image columns and Qwen-MM-Plugins' explicit image lists map to the same
preloaded-media form.

## 8. Video remains an explicit boundary

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

qwen-mm v0.1 rejects the original video message explicitly, without reading
media. The same applies to local video paths and frame lists:

```python
from qwen_mm import UnsupportedMediaError

processor.prepare(messages)  # raises UnsupportedMediaError: video is not supported
```

Video decoding, frame sampling, and temporal metadata are deferred. Extracted
still frames may be submitted as images, but are not treated as video.

## 9. vLLM, SGLang, and Qwen-MM-Plugins boundaries

Sources: [vLLM multimodal examples](https://github.com/vllm-project/vllm/tree/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/examples/generate/multimodal),
[SGLang Qwen cookbooks](https://github.com/sgl-project/sglang/tree/2935bb8e79e669b71aa4fef3b412fa25bc656c25/docs/cookbook/autoregressive/Qwen),
and [Qwen-MM-Plugins](https://github.com/QwenLM/Qwen-MM-Plugins/tree/ab339d2016ed1da8e9a96c477b067ecc74a6ce59).

Their OpenAI-compatible text and still-image message objects can be passed
directly to `prepare`, including URL and nested `image_url` content as shown in
sections 2 and 6. Their HTTP transport, model execution, generation settings,
agent routing, and video handling are outside qwen-mm. The prepared-pixel vLLM
adapter remains a separate prototype rather than part of the v0.1 wheel.

## 10. Thinking and tool conversations

These chat options and message fields are preserved. Tool execution remains the
caller's responsibility.

Official:

```python
messages = [
    {"role": "user", "content": "What is the temperature in Boston?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "temperature", "arguments": {"city": "Boston"}},
            }
        ],
    },
    {"role": "tool", "content": "18 degrees Celsius"},
    {"role": "user", "content": "Summarize the result."},
]
tools = [
    {
        "type": "function",
        "function": {
            "name": "temperature",
            "description": "Read the temperature in a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
inputs = processor.apply_chat_template(
    messages,
    tools=tools,
    enable_thinking=True,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
```

qwen-mm, using the same messages and tools:

```python
inputs = processor.prepare(
    messages,
    tools=tools,
    enable_thinking=True,
    add_generation_prompt=True,
    return_tensors="pt",
)
```

This thinking example uses Qwen3.5. Omitting `enable_thinking` keeps the pinned
profile's default; it is not inferred from the consumer model's size.

## 11. Raw RGB arrays

Here `rgb` is a caller-owned `uint8` array with shape `(height, width, 3)`.
The official path converts it to PIL; qwen-mm accepts it directly.

Official composed path:

```python
from PIL import Image
from qwen_vl_utils import process_vision_info

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": Image.fromarray(rgb)},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, _ = process_vision_info(messages, image_patch_size=16)
inputs = processor(text=[text], images=images, do_resize=False, return_tensors="pt")
```

qwen-mm:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": rgb},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
inputs = processor.prepare(messages, add_generation_prompt=True, return_tensors="pt")
```

## 12. Generation and output decoding

Given an already-loaded compatible Transformers `model` and the Torch `inputs`
from a supported example, both processors use the same downstream code:

Official:

```python
inputs = inputs.to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
new_tokens = generated_ids[:, inputs["input_ids"].shape[1] :]
output_text = processor.batch_decode(new_tokens, skip_special_tokens=True)
```

qwen-mm:

```python
inputs = inputs.to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
new_tokens = generated_ids[:, inputs["input_ids"].shape[1] :]
output_text = processor.batch_decode(new_tokens, skip_special_tokens=True)
```

The local verification consumer uses pinned Qwen3.5-0.8B weights with explicit
thinking options and the verified 9B processor contract. It does not change
which model IDs `Processor.from_pretrained` accepts.

## Call translation

| Official sample | qwen-mm |
| --- | --- |
| `AutoProcessor.from_pretrained(model_id)` | `Processor.from_pretrained("Qwen3")` or the same supported model ID |
| `messages` with a path, URL, or image data URI | The same `messages` |
| OpenAI `image_url` string or nested `{"url": ...}` | The same `messages` |
| `processor.apply_chat_template(...)` for one chat | `processor.prepare(messages, ...)` |
| `process_vision_info(messages)` plus `processor(...)` | Included in `prepare` for still images |
| `min_pixels`, `max_pixels`, or explicit dimensions | The same fields on the image content item |
| `padding=True` plus `padding_side = "left"` | `prepare_batch(..., padding_side="left")` |
| NumPy or Torch model inputs | NumPy by default; `return_tensors="pt"` and `.to(device)` for Torch |
| `processor.batch_decode(...)` | `processor.batch_decode(...)` |
| `chat_template_kwargs={"enable_thinking": ...}` | `prepare(..., enable_thinking=...)` |
| Preloaded ordered media lists | `prepare(messages, images=...)` or per-request batch lists |
| Video URL/path/frames | `UnsupportedMediaError`; deferred beyond v0.1 |
| vLLM/SGLang generation or agent calls | Out of scope; qwen-mm stops at local model inputs |
