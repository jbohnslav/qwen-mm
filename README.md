# qwen-mm

`qwen-mm` is a native Rust text and still-image preprocessor for pinned
Qwen3-VL and Qwen3.5 model snapshots. One Python call renders chat, tokenizes,
decodes and resizes images, and returns the NumPy arrays needed by the model.
It performs no network I/O and does not load or run model weights.

## Install

qwen-mm 0.1 requires Python 3.11:

```shell
pip install qwen-mm
```

Download one of the exact compatible Hugging Face snapshots before creating a
processor:

```shell
hf download Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
hf download Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a
```

## Quickstart

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

input_ids = prepared.arrays["input_ids"]
pixel_values = prepared.arrays["pixel_values"]
image_grid_thw = prepared.arrays["image_grid_thw"]
```

`from_huggingface_cache` resolves `HF_HUB_CACHE`, `HF_HOME`,
`XDG_CACHE_HOME`, or the standard user cache in that order. It only opens the
profile's hash-pinned local snapshot. Pass `cache_directory=` to use another
Hugging Face hub cache root. `Processor.supported_profiles()` returns every
accepted profile with its model ID, revision, and compatibility fingerprint.
The direct `Processor(profile, assets_directory)` constructor is available for
an explicitly managed snapshot directory.

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
| Filesystem paths, URLs, data-URI wrappers, PIL, or Torch inputs | Caller I/O/conversion required |
| Video files, decoded video, frame lists, or video sampling | Deferred |
| Model loading, generation, serving, and agent orchestration | Out of scope |

Media references use an integer position in the request's separate `images`
list. The canonical content form is `{"type": "image", "image": 0}`;
`input_index` and `buffer_index` are accepted aliases. Every supplied image
must be referenced. Image options can be inline on the content item:

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

`prepare_batch` accepts a non-empty list of requests. Requests may contain
different text lengths and image counts. Text arrays are right-padded under the
frozen v0.1 contract.

`PreparedBatch.arrays` contains the official model-input keys in stable order:
`input_ids`, `attention_mask`, `mm_token_type_ids`, and, when the batch has an
image, `pixel_values` and `image_grid_thw`. Integer arrays are `int64`; pixels
are `float32`. `PreparedBatch.metadata` contains qwen-mm integration metadata,
including request layouts and image occurrences, and is not a model input.

`thread_budget` is the processor-owned native worker budget, from 1 through
256. It defaults to 1. Reuse a processor across calls and choose a budget that
fits the host and the number of processors running concurrently. Batch work is
executed without holding Python's global interpreter lock.

## Errors and compatibility

Request failures are all-or-nothing and raise stable subclasses of
`QwenMMError`, including `InvalidRequestError`, `UnsupportedOptionError`,
`UnsupportedMediaError`, `MediaDecodeError`, `MediaGeometryError`, and
`ResourceLimitError`. Each exposes a stable `category` and structured
`context`. A missing cached snapshot raises `FileNotFoundError` with the exact
pinned `hf download` command.

The [official-example Rosetta Stone](docs/official-example-rosetta-stone-v0.1.md)
puts each upstream preprocessing sample immediately before its qwen-mm
equivalent. The [full comparison appendix](docs/official-example-comparison-v0.1.md)
records the wider Hugging Face, qwen-vl-utils, vLLM, SGLang, Qwen-MM-Plugins,
LLaMA-Factory, and ModelScope SWIFT source audit and deliberate v0.1
boundaries. The implementation contract and immutable pins are in
[ADR 0001](docs/compatibility-v1.md) and
[`reference/compatibility/v1.json`](reference/compatibility/v1.json).

Performance claims are limited to the controlled cases, hosts, and comparison
paths in [BENCHMARK.md](BENCHMARK.md) and the
[performance certification](docs/performance-certification-v1.md); they are not
general model-throughput claims. See [DESIGN.md](DESIGN.md) for architecture
and [ROADMAP.md](ROADMAP.md) for release gates.
