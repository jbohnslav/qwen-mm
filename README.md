# qwen-mm

[![CI](https://github.com/jbohnslav/qwen-mm/actions/workflows/ci.yml/badge.svg)](https://github.com/jbohnslav/qwen-mm/actions/workflows/ci.yml)

**Qwen's image preprocessing pipeline, consolidated and accelerated in Rust.**

`qwen-mm` puts the preprocessing best practices for pinned Qwen3-VL and Qwen3.5
model snapshots behind one Python API. It brings together the relevant
still-image behavior from `qwen-vl-utils` and the downstream processing normally
performed by Transformers: image loading, model-specific sizing, normalization,
patch layout, chat formatting, tokenization, and batching.

The goal is to make correct model inputs easy to produce. A pipeline can run
without errors while using different image budgets, placing image tokens
incorrectly, or applying the wrong chat or padding conventions. qwen-mm makes
those choices explicit and checks supported flows against the official
workflows, so callers have fewer details to assemble themselves.

Use a familiar `from_pretrained`, then one `prepare` or `prepare_batch` call
for paths, URLs, encoded images, or RGB arrays. Get NumPy arrays or optional
Torch tensors ready for the model, with native batching that releases Python's
GIL. The optional [vLLM plugin](integrations/vllm/README.md) handles still-image
preprocessing inside a prebuilt vLLM server using ordinary image requests.
Model weights, training, and generation remain with the caller's framework.

## Correctness and compatibility

qwen-mm reimplements the supported still-image pipeline in Rust; it does not
require callers to run `qwen-vl-utils` first. An implementation can also be
correct without that package: what matters is preserving the relevant
model-specific behavior. The
[executable Rosetta Stone](docs/official-example-rosetta-stone-v0.1.md) compares
official examples with qwen-mm equivalents, and the
[verification reports](docs/rosetta-verification.md) cover both pinned profiles,
message/image alignment, model-input layouts, batching, and resize fidelity.
Differences between direct Transformers and composed `qwen-vl-utils` defaults
are documented explicitly, including their different minimum image budgets.

Correctness has a defined scope. Resizing follows the tested resize-v2 fidelity
contract rather than bit-for-bit Pillow equality; resized pixels can change
generated answers. v0.1 supports text and still images, with video and frame
lists deferred. It is not a complete replacement for every `qwen-vl-utils`
feature or support for arbitrary Qwen model revisions.

## What to expect from performance

The clearest gains are in CPU preprocessing. End-to-end gains depend on how
much that work limits the surrounding training or serving pipeline.

| Workload | Measured result | What it establishes |
| --- | --- | --- |
| Standalone 24-image preprocessing, eight native threads | About **3.5–3.6× faster on x86** and **6.4× on Apple Silicon** in earlier controlled benchmarks | Faster preprocessing for the measured hosts, profiles, and comparison paths |
| Qwen3.5-9B serving on an L40S with prebuilt vLLM | Median image-processing time **46 ms → 28 ms**, about **38% lower** | Less CPU processing time in the measured mix of image requests |
| End-to-end serving in that experiment | About **3–7% lower median time to first token** for heavy/concurrent requests | A small diagnostic latency improvement; no convincing sustained throughput gain |

The [standalone benchmark report](benchmarks/performance-certification-v1/report.md)
and [vLLM measurements](integrations/vllm/evidence/45e5/gpu-20260907/README.md)
retain their exact source and workload provenance. The serving sample is small,
and the four-image latency gain is comparable to variation between stock runs.
These measurements are not fresh performance certification of every release
wheel. The broader strict D4 certificate remains a **MISS**, and no general
speed or memory guarantee is made.

For Transformers training, faster preparation may help keep GPUs fed when CPU
image processing is the bottleneck. If data preparation already overlaps with
GPU work, training throughput may change little. **End-to-end training
throughput has not been benchmarked.**

## Install

The optional vLLM plugin is not part of the initial PyPI publication: its pinned
vLLM dependency needs a security upgrade. Installing core `qwen-mm` does not install vLLM.

qwen-mm is public on [GitHub](https://github.com/jbohnslav/qwen-mm).
Version 0.1.0 is undergoing native release verification before PyPI publication. It
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
| Still-image preprocessing inside prebuilt vLLM | Optional plugin; see its pinned support and verification scope |
| Video files, decoded video, frame lists, or video sampling | Deferred |
| Model loading, training, generation, and server orchestration | Owned by the caller's framework |

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

For ordinary image requests to a prebuilt vLLM server, see the
[external vLLM plugin](integrations/vllm/README.md).
