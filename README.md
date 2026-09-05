# qwen-mm

`qwen-mm` is a native Rust text and still-image preprocessor for pinned
Qwen3-VL and Qwen3.5 model snapshots. One Python call renders chat, tokenizes,
decodes and resizes images, and returns the NumPy arrays needed by the model.
It can download the small, pinned processor snapshot and read an image source
the caller explicitly supplies; it does not download, load, or run model
weights.

## Install

qwen-mm 0.1.0 is a release candidate, awaiting approval and publication. It
supports CPython 3.11 on native macOS ARM64 and Linux x86_64 with glibc.
Install the matching candidate wheel from the release artifact bundle:

```shell
python3.11 -m pip install /path/to/qwen_mm-0.1.0-cp311-abi3-<platform>.whl
```

Replace `<platform>` with the actual filename. After publication, the equivalent
index install will be `python3.11 -m pip install qwen-mm==0.1.0`.
See the [install and support guide](docs/install-v0.1.md),
[release notes](CHANGELOG.md), and [release procedure](docs/releasing-v0.1.md).

## Quickstart

```python
from qwen_mm import Processor

processor = Processor.from_pretrained("Qwen3", thread_budget=4)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "image.jpg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

prepared = processor.prepare(messages, add_generation_prompt=True)

input_ids = prepared["input_ids"]
pixel_values = prepared["pixel_values"]
image_grid_thw = prepared["image_grid_thw"]
```

`Processor.from_pretrained` accepts `Qwen3`, `Qwen3.5`,
`Qwen/Qwen3-VL-8B-Instruct`, or `Qwen/Qwen3.5-9B`. It uses the normal Hugging
Face cache and downloads only qwen-mm's hash-pinned processor/tokenizer files,
never model weights. Pass `cache_dir=` to select another cache or
`local_files_only=True` for an offline/cache-only call.

`Processor.from_huggingface_cache(profile, cache_directory=...)` remains the
lower-level no-download path for explicitly managed services and reproducible
runs. `Processor.supported_profiles()` returns every accepted profile with its
model ID, revision, and compatibility fingerprint. The direct
`Processor(profile, assets_directory)` constructor accepts an already-resolved
snapshot directory.

For one conversation, `prepare(messages, ...)` exposes the common chat options
directly. Ordinary batches accept `[messages1, messages2]` with shared chat
options. Request dictionaries also support per-row options and separate media:

```python
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
            "images": [open("image.jpg", "rb").read()],
            "options": {"add_generation_prompt": True},
        }
    ],
    padding_side="left",
)
```

## v0.1 scope

| Flow | Status |
| --- | --- |
| Text-only chat | Supported |
| One image, multiple images, or interleaved text and images | Supported |
| Heterogeneous batches and repeated image references | Supported |
| JPEG, PNG, or WebP encoded buffers | Supported |
| `uint8` HWC RGB NumPy arrays | Supported |
| Per-image pixel budgets or explicit resize dimensions | Supported |
| Qwen3.5 thinking/non-thinking rendering and tool messages | Supported |
| Filesystem paths, file/HTTP(S) URLs, or image data URIs | Supported directly |
| PIL images or Torch tensors as image inputs | Caller conversion required |
| Torch model-input outputs and token decoding | Supported; Torch is optional |
| Video files, decoded video, frame lists, or video sampling | Deferred |
| Model loading, generation, serving, and agent orchestration | Out of scope |

An image content item may hold a `pathlib.Path`, a plain filesystem path, a
`file:` URL, an `http:` or `https:` URL, an image data URI, encoded bytes, an
RGB array, or an integer position in the request's separate `images` list.
Qwen3.5's OpenAI-style `{"type": "image_url", "image_url": ...}` string and
nested `{"url": ...}` forms are also accepted. If the caller supplies an
HTTP(S) URL, qwen-mm fetches it directly: there is no permission flag,
private-network filter, security-policy mode, or warning ceremony. Timeouts,
resource ceilings, and actionable network failures are reliability behavior.

The preloaded form is `{"type": "image", "image": 0}`; `input_index` and
`buffer_index` are accepted aliases. Every supplied preloaded image must be
referenced. Image options stay inline and work with either source form:

```python
{"type": "image", "image": 0, "min_pixels": 50_176, "max_pixels": 200_704}
{"type": "image", "image": 1, "resized_height": 336, "resized_width": 336}
```

Encoded input may be any bytes-like object. Use
`{"data": image_bytes, "format": "jpeg"}` to declare `jpeg`, `png`, or
`webp` explicitly. Raw arrays must have dtype `uint8`, shape `(height, width,
3)`, and packed RGB channels; row padding is allowed, but transposed, reversed,
or channel-strided views are rejected.

## Batches, outputs, and threads

`prepare_batch` accepts a non-empty list of conversations or request dictionaries.
Shared `add_generation_prompt`, `add_vision_id`, `tools`, and `enable_thinking`
options apply to every row; a dictionary's `options` overrides them. Rows may contain
different text lengths and image counts. Text arrays are right-padded by
default; pass `padding_side="left"` for generation batches that follow the
official Qwen examples. `PreparedBatch.metadata["padding_side"]` records the
choice, and each request layout records its left and right padding.

`PreparedBatch.arrays` contains the official model-input keys in stable order:
`input_ids`, `attention_mask`, `mm_token_type_ids`, and, when the batch has an
image, `pixel_values` and `image_grid_thw`. Integer arrays are `int64`; pixels
are `float32`. The prepared result is also a mapping over exactly those arrays,
so `prepared["input_ids"]` and normal `consumer(**prepared)` keyword expansion
work directly. NumPy remains the default. With Torch installed, pass
`return_tensors="pt"` to either preparation method for tensors sharing the CPU
array storage and a `.to(device)` method. `decode` and `batch_decode` use the
same pinned native tokenizer, so generation needs no second processor.
`PreparedBatch.metadata` contains qwen-mm integration metadata, including
request layouts and image occurrences, and is not a model input.

Shared `min_pixels` and `max_pixels` keyword arguments provide image defaults;
values on an individual image override them. The default minimum is 4096 pixels,
matching the composed qwen-vl-utils contract. Use `min_pixels=65536` to match
direct Transformers image examples. These distinct upstream defaults are
explicit in the [executable Rosetta Stone](docs/official-example-rosetta-stone-v0.1.md).

`thread_budget` is the processor-owned native worker budget, from 1 through
256. It defaults to 1. Reuse a processor across calls and choose a budget that
fits the host and the number of processors running concurrently. Batch work is
executed without holding Python's global interpreter lock.

## Errors and compatibility

Request failures are all-or-nothing and raise stable subclasses of
`QwenMMError`, including `InvalidRequestError`, `UnsupportedOptionError`,
`UnsupportedMediaError`, `MediaDecodeError`, `MediaGeometryError`, and
`ResourceLimitError`. Each exposes a stable `category` and structured
`context`. The lower-level cache-only constructor raises `FileNotFoundError`
with the exact pinned `hf download` command when its snapshot is missing;
`from_pretrained(..., local_files_only=True)` preserves Hugging Face's ordinary
offline cache-miss error.

The [official-example Rosetta Stone](docs/official-example-rosetta-stone-v0.1.md)
puts each upstream preprocessing sample immediately before its qwen-mm
equivalent. The [full comparison appendix](docs/official-example-comparison-v0.1.md)
records the wider Hugging Face, qwen-vl-utils, vLLM, SGLang, Qwen-MM-Plugins,
LLaMA-Factory, and ModelScope SWIFT source audit and deliberate v0.1
boundaries. The implementation contract and immutable pins are in
[ADR 0001](docs/compatibility-v1.md) and
[`reference/compatibility/v1.json`](reference/compatibility/v1.json).

The opt-in [local Transformers consumer test](docs/transformers-consumer-verification.md)
feeds prepared arrays through a pinned Qwen3.5-0.8B model on CPU and compares
them with the official processor. It verifies the model handoff without making
inference latency a release gate.

Performance claims are limited to the controlled cases, hosts, and comparison
paths in [BENCHMARK.md](BENCHMARK.md) and the
[performance certification](docs/performance-certification-v1.md); they are not
general model-throughput claims. See [DESIGN.md](DESIGN.md) for architecture
and [ROADMAP.md](ROADMAP.md) for release gates.

## License

qwen-mm is licensed under [Apache-2.0](LICENSE). Downloaded Qwen assets retain
their upstream licenses; processor construction does not download model weights.
