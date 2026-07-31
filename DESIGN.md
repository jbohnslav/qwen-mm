# qwen-mm design

Status: Draft

## Summary

`qwen-mm` is a Qwen-specific Rust implementation of image and text preprocessing for Qwen3-VL and Qwen3.5-VL. It is not intended to become a generic replacement for Hugging Face processors. Its purpose is to reproduce the exact preprocessing contract for these model families while removing Python per-item work, duplicated transforms, and unnecessary intermediate copies.

Given encoded images and structured Qwen conversations, the processor should produce byte-for-byte identical token metadata and numerically equivalent visual tensors in one native batch call.

## Goals

- Support only Qwen3-VL and Qwen3.5-VL preprocessing.
- Load and execute the Hugging Face `tokenizer.json` implementation through the Rust `tokenizers` crate.
- Own the exact Qwen chat rendering and visual-placeholder expansion rules.
- Decode, orient, resize, normalize, and patchify images in Rust.
- Parallelize independent preprocessing work with Rayon, without the Python global interpreter lock.
- Fuse visual operations and avoid unnecessary full-image intermediate buffers.
- Match pinned reference versions of Transformers, Tokenizers, Qwen VL Utils, and each model's processor configuration.
- Provide thin Python, Arrow/Ray, and vLLM integration layers around a framework-independent Rust core.
- Act as a drop-in replacement for the corresponding preprocessing components in Hugging Face Transformers and vLLM.

## Non-goals

- General Hugging Face processor or chat-template compatibility.
- Broad computer-vision functionality.
- A dependency on OpenCV, PyTorch, or vLLM in the core library.
- Initial support for URLs, PIL objects, arbitrary tensors, loosely shaped dictionaries, or many interchangeable video backends.
- Defining one timeless version of "canonical Qwen behavior." Compatibility is tied to explicitly pinned upstream versions and model configuration.

## Upstream motivation and evidence

The performance problem is visible in upstream implementation work and user reports:

- [Transformers PR #45719](https://github.com/huggingface/transformers/pull/45719) found that `Qwen2VLImageProcessor` spent a significant amount of preprocessing time copying data. Reducing copies improved an eight-image, 1024×1024 benchmark from 33 ms to 25.7 ms, a 22% end-to-end improvement. The PR author confirmed that Qwen3-VL reuses this processing path.
- The official [Qwen3.5-9B processor configuration](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/preprocessor_config.json) declares `Qwen3VLProcessor` and `Qwen2VLImageProcessorFast`, connecting that image path directly to Qwen3.5.
- [Qwen issue #2060](https://github.com/QwenLM/Qwen3-VL/issues/2060) reports `build_inputs` taking 8.04 seconds, or 97.4% of the measured forward phase, in a Qwen training pipeline. The reporter later describes roughly four seconds of image preprocessing for a batch of 16 versus 225 ms for the model forward pass. This is a user benchmark from a custom stack, not a controlled upstream microbenchmark.
- [vLLM issue #22444](https://github.com/vllm-project/vllm/issues/22444) reports Qwen2-VL image-plus-text throughput of 17.89 requests per second versus 86.92 for text-only requests, alongside unstable 0–50% GPU utilization instead of approximately 95%. This is an end-to-end serving result and does not isolate preprocessing from the vision encoder or request transport.
- The [Qwen3-VL README](https://github.com/QwenLM/Qwen3-VL#new-qwen-vl-utils-usage) warns callers to set `do_resize=False` because Qwen VL Utils has already resized the visuals. It also documents the speed and reliability tradeoffs among TorchVision, Decord, and TorchCodec video backends.
- [vLLM's optimization documentation](https://github.com/vllm-project/vllm/blob/main/docs/configuration/optimization.md#input-processing) treats input processing as a possible bottleneck relative to model execution, uses Qwen2.5-VL in its API-process scale-out examples, and automatically caches multimodal processor results to avoid repeated work.

These sources support the existence of avoidable processor cost. They do not establish one universal speedup for every model, input distribution, or integration path; `qwen-mm` must supply its own reproducible end-to-end benchmarks.

### Canonical Hugging Face integration examples

The official model-card examples are reference integration paths for this project:

- The pinned [Qwen3-VL-30B-A3B-Instruct-FP8 model card](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8/blob/d9748a51ae66354c4dad665aab2c71f26cf2c8cd/README.md) includes image and video message forms. Its offline vLLM helper renders the prompt with `AutoProcessor.apply_chat_template(..., tokenize=False)`, calls `qwen_vl_utils.process_vision_info`, and forwards the returned image, video, and video-processor values to vLLM.
- The pinned [Qwen3.5-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/34d7bb1c2fe64851378eee0753dbbf227899ba80/README.md) adds the corresponding offline multimodal vLLM example and explicitly installs `qwen-vl-utils==0.0.14`. Its helper has the same split between chat rendering and visual processing, supports both image and video outputs, and passes video sampling metadata through `mm_processor_kwargs`.

Both examples pass `processor.image_processor.patch_size` to `process_vision_info` and request `return_video_kwargs=True` plus `return_video_metadata=True`. The resulting supported path is:

```text
structured image/video messages
    ├── Transformers chat rendering ───────────────→ prompt
    └── Qwen VL Utils media loading/processing ───→ images, videos, video kwargs
                                                        ↓
                                     vLLM multi_modal_data/mm_processor_kwargs
```

These model cards are compatibility and integration evidence, not performance measurements. They identify the real multi-call Python boundary that `qwen-mm` intends to collapse into one native batch operation. Their exact image and video message shapes should be the first smoke cases in the reference fixture generator, with the model-card revision, package versions, processor assets, and media checksums recorded alongside the expected outputs. The Qwen3.5 link is intentionally revision-pinned because the current model card no longer contains that offline example.

### Direct AsyncLLM admission bottleneck

As of 2026-07-31, [vLLM issue #49317](https://github.com/vllm-project/vllm/issues/49317) describes an especially close match to the motivating workload:

- direct in-process `AsyncLLM.generate()` calls with raw Qwen3.5 multimodal prompts;
- high external concurrency and many distinct images per request;
- only 2–6 requests running in EngineCore while `num_requests_waiting` remains zero;
- near-idle GPU utilization; and
- no change after increasing `max_num_batched_tokens`.

The reported request shape contains one reusable query image and roughly 200 distinct document images per request. The shared image can hit vLLM's multimodal cache, while the distinct images cannot.

The source-level cause is upstream of EngineCore admission. `AsyncLLM.add_request()` is asynchronous, but it calls `InputProcessor.process_inputs()` synchronously on the asyncio event-loop thread. For raw multimodal prompts, that call performs tokenization and Hugging Face image resize and patchification on the CPU. Concurrent `generate()` coroutines therefore serialize in preprocessing before the scheduler can classify them as running or waiting.

The diagnostic signature is:

```text
many external generate() calls in flight
        +
EngineCore waiting ≈ 0
        +
EngineCore running unexpectedly low
        +
low GPU utilization
        ↓
frontend preprocessing/admission bottleneck
```

This signature does not by itself prove that an upstream data system such as Ray Data, object storage, or dataset parsing is the bottleneck. It also does not make `max_num_seqs` a measure of expected occupancy: that setting is an upper bound, and effective engine batches can be constrained by token, encoder, KV-cache, or request availability limits.

[vLLM PR #49608](https://github.com/vllm-project/vllm/pull/49608) proposes moving raw-prompt processing onto the existing renderer thread pool. Its in-process H100 benchmark used Qwen2-VL-2B with 32 concurrent `AsyncLLM.generate()` calls, four distinct 1024×1024 images per request, and no multimodal cache hits. The patch changed:

```text
mean end-to-end latency     6.17 s  → 3.50 s
maximum event-loop stall   6.14 s  → 0.99 s
mean heartbeat gap          323 ms  → 2 ms
total wall time            6.32 s  → 5.79 s
```

The approximately 8% wall-time improvement is an important limitation. Moving work off the event-loop thread restores concurrent admission and overlap, but the Hugging Face preprocessing itself remains CPU- and GIL-bound. The PR's later repeated benchmark found 4.56–5.00-second event-loop stalls on the original direct `AsyncLLM` path and no measurable stalls with the patch.

The PR is scoped specifically to direct `AsyncLLM` callers supplying raw prompts. The OpenAI server path already runs multimodal rendering through an asynchronous renderer pool, and a `vllm serve` comparison showed no throughput effect from this patch. Benchmarks and adapters must keep those two paths separate.

This issue is direct evidence for two parts of the `qwen-mm` design: preprocessing must be a genuinely parallel native batch operation, and integrations must hand its results across the engine-admission boundary without falling back to the synchronous Hugging Face processor.

## Compatibility contract

The difficult part of this project is reproducing the complete processor semantics, not implementing a resize kernel. Behavior may differ between Qwen VL Utils and Transformers, and it may change between upstream versions. The implementation therefore becomes authoritative only after differential conformance testing against pinned references.

The compatibility contract includes:

- Token IDs and their ordering.
- Chat-template rendering and special-token insertion.
- Visual-placeholder count and placement.
- `image_grid_thw` and other visual metadata.
- Pixel conversion, interpolation, antialiasing, normalization, patchification, and output layout.
- Model-specific differences between Qwen3-VL and Qwen3.5-VL.

Exact reference versions must be recorded with every fixture corpus:

- `transformers`
- `tokenizers`
- `qwen-vl-utils`
- model processor configuration
- tokenizer and chat-template artifacts

## Workspace layout

The project should be split along framework boundaries:

```text
qwen-mm-core       Pure Rust preprocessing
qwen-mm-python     PyO3 and NumPy/buffer-protocol bindings
qwen-mm-arrow      Arrow arrays for Ray and data pipelines
qwen-mm-vllm       Thin vLLM adapter, potentially upstreamable
```

`qwen-mm-core` returns owned or borrowed typed buffers. It does not return `torch::Tensor` values and does not link against PyTorch. Python bindings expose NumPy-compatible storage so callers can use `torch.from_numpy` without a copy where the output dtype and layout allow it.

## Processing pipeline

### Text

The tokenizer is loaded directly from `tokenizer.json`:

```rust
use tokenizers::Tokenizer;

let tokenizer = Tokenizer::from_file("tokenizer.json")?;
let encodings = tokenizer.encode_batch(texts, true)?;
```

This avoids a Python tokenizer wrapper or a C ABI shim while using Hugging Face's native tokenizer implementation.

The processor accepts structured conversations and renders the pinned Qwen template directly. It does not initially attempt to support arbitrary Jinja chat templates. For example, structured content such as an image followed by `Describe the scene` maps to the Qwen form:

```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Describe the scene<|im_end|>
```

After visual preprocessing determines the grid, the processor expands the visual placeholders according to `grid_thw` and the model's merge size before tokenization is finalized.

### Images

The visual path performs only the operations required by the supported Qwen processors:

1. Decode JPEG, PNG, or WebP input.
2. Apply EXIF orientation.
3. Convert grayscale, alpha, CMYK, and other supported inputs to RGB.
4. Apply the model-specific `smart_resize` rule.
5. Resize once using the reference-compatible interpolation and antialias behavior.
6. Normalize channels.
7. Apply temporal duplication or padding where required.
8. Patchify, reorder, and flatten into the final Qwen layout.
9. Produce `image_grid_thw`, visual-token counts, and placeholder metadata.

The `smart_resize` implementation must reproduce the reference's exact order of factor alignment, minimum and maximum pixel-budget checks, aspect-ratio preservation, and floor/ceil/round operations.

### Videos

Temporal preprocessing is part of the compatibility contract, including frame sampling, odd frame counts, temporal-factor padding, and the distinction between image and video inputs. Initial API work should remain limited to encoded images and raw RGB frames. Video decoding backends and convenience input types should be added only after the image path and temporal tensor semantics conform to the pinned references.

The core should not require Decord or choose among a large set of video libraries. Any decoder integration belongs behind a narrow adapter that supplies frames to the core representation.

## Input API

The initial visual API supports only encoded image bytes and caller-owned RGB data:

```rust
pub enum VisualInput<'a> {
    EncodedImage(&'a [u8]),
    Rgb8 {
        data: &'a [u8],
        height: usize,
        width: usize,
        stride: usize,
    },
}

pub struct QwenRequest<'a> {
    pub messages: &'a [Message<'a>],
    pub visuals: &'a [VisualInput<'a>],
}
```

Encoded bytes are convenient for stored JPEGs and similar assets. Raw RGB input avoids a decode/re-encode cycle when an upstream system already owns image or camera-frame buffers.

The structured message API owns the exact mapping from Qwen content to rendered text:

```rust
processor.prepare_batch(&[
    QwenRequest {
        messages: &[
            Message::user(&[
                Content::Image { index: 0 },
                Content::Text("Describe the scene"),
            ]),
        ],
        visuals: &[image],
    },
])?;
```

## Output API

The batch result exposes typed contiguous buffers suitable for NumPy, Arrow, Ray, and vLLM adapters:

```text
PreparedBatch
├── input_ids:       int32
├── input_offsets:   int32
├── pixel_values:    bf16 or f32
├── visual_offsets:  int64
└── image_grid_thw:  int32
```

In addition to an allocating API, the processor should support caller-owned destinations:

```rust
processor.prepare_batch_into(
    requests,
    input_ids_buffer,
    pixel_values_buffer,
    metadata_buffer,
)?;
```

This allows Ray actors and inference workers to reuse arenas or pinned-memory pools rather than moving the bottleneck into allocation and serialization.

## Fusion and memory layout

The target visual path is:

```text
decode to RGB u8
        ↓
resize to final dimensions
        ↓
normalize + transpose + patchify
        ↓
write into final pixel_values layout
```

The common Python path can materialize several PIL, NumPy, and Torch representations of the same image and may resize more than once. `qwen-mm` should instead use one intermediate RGB buffer and one final output buffer where full fusion is not practical.

The eventual optimized resize-output path may write directly into the Qwen patch-major destination layout. That optimization follows conformance: it must not change interpolation, normalization, channel ordering, or patch ordering.

Rayon parallelizes independent requests and visuals. The design should make batches of many images native work items rather than invoking Rust separately for every Python object.

## Dependencies

The expected Rust stack is narrow:

```toml
[dependencies]
tokenizers = "..."
image = "..."
fast_image_resize = "..."
rayon = "..."
ndarray = "..."
half = "..."
safetensors = "..."

# qwen-mm-python only
pyo3 = { version = "...", features = ["extension-module"] }
numpy = "..."
```

OpenCV is not required for this operation set. `fast_image_resize` supplies SIMD resize kernels without introducing OpenCV's broader native surface. Dedicated decoders such as `zune-jpeg` or a libjpeg-turbo binding should be considered only if profiling shows that decoding is a material bottleneck.

## Differential conformance tests

Each candidate build is compared with the pinned Python processor:

```python
reference = hf_processor(
    text=text,
    images=images,
    videos=videos,
    return_tensors="np",
)
candidate = rust_processor.prepare(
    messages=messages,
    images=images,
    videos=videos,
)

assert reference["input_ids"].tolist() == candidate.input_ids
assert reference["image_grid_thw"].tolist() == candidate.image_grid_thw
np.testing.assert_allclose(
    reference["pixel_values"],
    candidate.pixel_values,
    rtol=0,
    atol=chosen_tolerance,
)
```

Expected outputs should be saved as fixtures so Rust tests do not require Python at runtime. Fixture metadata records all pinned package, tokenizer, template, and processor-config versions.

The corpus includes thousands of generated and hand-selected edge cases:

- Dimensions immediately below, at, and above every resize-factor boundary.
- Pixel counts around minimum and maximum thresholds.
- Extreme aspect ratios.
- Grayscale, RGBA, CMYK, and EXIF-rotated inputs.
- One-frame videos and odd frame counts.
- Multiple visuals interleaved with text.
- Explicit resized dimensions.
- Repeated image references.
- Empty and unusual chat turns.
- Qwen3-VL and Qwen3.5-VL model-specific configurations.
- Image and video message shapes mirrored from the pinned official Hugging Face model-card examples.

Particular attention is required for known areas of disagreement or accidental duplication in upstream paths, including temporal-factor handling in `smart_resize` and possible double resizing. The pinned output contract, rather than an assumed universal implementation, decides compatibility for each supported configuration.

## First baseline: the current Python path

Before implementing Rust processing, the repository needs one small, reproducible answer to: "How long does the current path take on this machine?" This is a directional baseline, not the final benchmark suite.

The first completed run and its limitations are recorded in [BENCHMARK.md](BENCHMARK.md).

### Reference configurations

Run the same image-only benchmark against two processor configurations:

- `Qwen/Qwen3-VL-8B-Instruct`
- `Qwen/Qwen3.5-9B`

Only processor, tokenizer, and template artifacts are loaded; model weights and model execution are not part of this benchmark. Each model repository is pinned to an immutable Hugging Face revision. The Python environment locks exact versions of Transformers, Tokenizers, Qwen VL Utils, Pillow, NumPy, Torch, and TorchVision and writes those versions into every result.

### Inputs and timing boundary

The headline case, `image24`, is one user conversation containing 24 distinct images followed by one short text instruction. This matches the admission-bound shape of many distinct visuals in one raw prompt. The companion `image1` case uses the same shape with one image so fixed overhead is visible without creating a larger matrix; 24 separate one-image requests belong in the later batch-scaling suite.

The 24 inputs are deterministic 1023×767 RGB JPEG fixtures at quality 90. Each image contains a smooth generated pattern, low-amplitude seeded noise, and geometric overlays. The dimensions are deliberately not aligned to Qwen's spatial factor, so the reference path must decode and resize rather than receiving an already aligned no-op case. The exact JPEG bytes are committed under `fixtures/baseline/image24/` and recorded in a manifest with byte length and SHA-256; those committed bytes, rather than whatever a local JPEG encoder happens to produce, are the benchmark inputs. A generator remains in the repository for provenance and controlled updates.

Encoded bytes are loaded into memory before timing. Every measured iteration creates fresh `BytesIO` and PIL image objects from those bytes so lazy image decoding still occurs inside the measured operation. Network access, filesystem reads, Python imports, `AutoProcessor.from_pretrained`, model loading, GPU transfer, and generation are outside the timing boundary.

The measured operation begins with encoded image bytes plus a structured conversation and ends when CPU-resident NumPy outputs have been materialized:

```python
pil_images = [Image.open(BytesIO(data)) for data in encoded_images]
messages = bind_visuals(conversation, pil_images)

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
images, videos, video_kwargs = process_vision_info(
    messages,
    image_patch_size=processor.image_processor.patch_size,
    return_video_kwargs=True,
    return_video_metadata=True,
)

video_values, video_metadata = split_video_metadata(videos)
outputs = processor(
    text=[text],
    images=images,
    videos=video_values,
    video_metadata=video_metadata,
    padding=True,
    do_resize=False,
    return_tensors="np",
    **video_kwargs,
)
```

`do_resize=False` is part of this canonical reference path because Qwen VL Utils has already resized the visual inputs. A separate integration benchmark can later measure paths that accidentally resize twice; that behavior must not contaminate the initial correctness baseline.

### Measurement protocol

Use a small self-contained Python runner based on `time.perf_counter_ns`, rather than adding a full benchmark framework initially:

1. Initialize the processor and load fixture bytes.
2. Run three unreported warm-up iterations.
3. Run ten measured iterations in the same process.
4. Recreate PIL objects and messages for every iteration.
5. Collect garbage between iterations, outside the timed region.
6. Report median, minimum, and p90 latency for `image1` and `image24`.

The runner records both total time and directional stage times:

- PIL/message binding;
- chat-template rendering;
- `process_vision_info`, including decode, orientation, and resize; and
- final Hugging Face processing, including tokenization, normalization, grid construction, and patchification.

The total timer surrounds the complete operation and is authoritative. Nested stage timers are diagnostic only. Library thread defaults are not artificially restricted for the first run; the result records CPU model, physical and logical core counts, operating system, Python version, Torch thread counts, and relevant thread environment variables.

Every result also records output shapes, `image_grid_thw`, token count, and hashes of the token and pixel buffers. That prevents a fast result produced by skipped work or a changed processor configuration from being compared with a complete run.

The intended command is:

```bash
uv run --project reference python -m qwen_mm_reference.bench \
    --models qwen3-vl-8b,qwen3.5-9b \
    --cases image1,image24
```

The first headline number is the median end-to-end `image24` latency for each processor configuration. Video, concurrency, memory profiling, cold starts, URL loading, vLLM admission, and GPU utilization are intentionally deferred.

## Performance validation

Performance comparisons should measure the complete batch operation, not an isolated resize kernel. The relevant comparison begins with encoded bytes and structured conversations and ends with token metadata and final visual tensors.

Measurements should include:

- End-to-end latency and throughput across batch sizes.
- Scaling when a request contains many images.
- Decode, resize, tokenization, and patchification time.
- Bytes allocated and number of full-image intermediate buffers.
- Python calls and serialization or copy overhead in each adapter.
- External requests ready and in flight versus EngineCore requests running and waiting.
- Time from raw input availability to preprocessing start, preprocessing completion, engine submission, first output, and completion.
- Event-loop heartbeat delay and maximum stall for direct `AsyncLLM` integrations.
- Distinct-image, repeated-image, cold-cache, and warm-cache behavior.

Optimization work follows profiles. The expected primary gain is eliminating whole stages and copies; a faster individual resize kernel is secondary.

## Implementation plan

Each milestone has a runnable artifact and an exit condition. Correctness gates precede optimization, and the first vertical slice is encoded JPEG plus text to final arrays for one Qwen3-VL configuration.

### Milestone 0: lock and measure the reference

Deliverables:

- `reference/pyproject.toml` and `reference/uv.lock` with pinned Python dependencies;
- immutable Hugging Face revisions and downloaded processor-artifact hashes;
- the deterministic `image1` and `image24` fixture generator and manifest;
- the current-path timing runner described above; and
- one JSON result file per processor configuration containing environment, timing, and output metadata.

Exit condition: a clean checkout verifies the committed fixture hashes, regenerates equivalent reference outputs, and produces the first local baseline with one command.

### Milestone 1: define contracts and golden fixtures

Pin the Rust toolchain in `rust-toolchain.toml`, then create the Cargo workspace and the `qwen-mm-core` crate with formatting, linting, and test commands. Finalize typed messages, visual inputs, batch outputs, error categories, model-family configuration, and fixture metadata before processing code spreads those assumptions through the implementation.

Build the Python fixture exporter around the exact path used by the baseline. Start with a small golden set covering text-only, one image, 24 images, interleaved images and text, and the pinned model-card message shapes. Save arrays in a language-neutral representation plus JSON metadata. Expand into the adversarial corpus as each processing stage lands.

Exit condition: Rust tests can load the fixtures, and every fixture states exactly which upstream artifacts produced it.

### Milestone 2: text-only vertical slice

Implement structured Qwen messages, direct rendering of each pinned Qwen template, `tokenizer.json` loading through `tokenizers`, batch tokenization, special tokens, offsets, and attention metadata. Keep Qwen3-VL and Qwen3.5-VL differences explicit in configuration rather than inferred from arbitrary model files.

Exit condition: rendered strings and token arrays are exact for both model families across the text and unusual-chat fixtures.

### Milestone 3: conformant single-image path

Implement encoded JPEG/PNG/WebP decoding, EXIF orientation, RGB conversion, exact `smart_resize`, reference-compatible resampling, normalization, temporal duplication for still images, patchification, flattening, `image_grid_thw`, visual-token counts, and placeholder expansion. Use straightforward intermediate buffers and a serial implementation first.

Bring up Qwen3-VL first, then add Qwen3.5-VL configuration before doing performance work. When upstream implementations disagree, create a targeted fixture and resolve behavior through the declared pinned compatibility contract.

Exit condition: metadata is exact and pixel tensors meet the documented tolerance for both model families' image corpus.

### Milestone 4: native batch API and first comparison

Implement `prepare_batch` and the minimal PyO3/NumPy binding accepting encoded byte buffers and structured requests. One Python call must process the entire `image1` or `image24` case; there is no Python callback per visual.

Run the Rust path through the exact same fixture bytes, conversations, output checks, warm-up count, and iteration count as the Python baseline. Publish correctness results beside latency rather than reporting speed in isolation.

Exit condition: the binding is conformant and the first apples-to-apples `image1`/`image24` comparison is reproducible. If `image24` is not faster, profile before expanding the API or adding more integrations.

### Milestone 5: parallelize and fuse from profiles

Add deterministic Rayon parallelism across independent requests and visuals. Record output positions before dispatch so scheduling cannot change result order. Profile allocation, decode, resize, normalization, patchification, tokenization, and Python-boundary costs.

Then remove demonstrated bottlenecks: reuse scratch storage, eliminate redundant format conversions, combine normalization with layout conversion, and add caller-owned destination buffers. Direct resize into patch-major output is optional and only proceeds if a profile justifies its complexity.

Exit condition: all conformance tests remain unchanged, `image1` has no material regression, and `image24` demonstrates a repeatable improvement on the recorded host.

### Milestone 6: temporal semantics and video adapters

First support caller-supplied RGB frame sequences so frame sampling, temporal padding, metadata, grids, and patchification can be conformed without coupling the core to a decoder. Add one encoded-video adapter only after profiling and evaluating maintained Rust or native-backed decoders.

Exit condition: canonical model-card video inputs and adversarial frame-count cases match the pinned reference, with decoder-specific behavior isolated from `qwen-mm-core`.

### Milestone 7: production integrations

Add Arrow/Ray buffers, reusable arenas, and the thin vLLM adapter. Measure direct `AsyncLLM` and server paths separately, including preprocessing-to-admission time, running/waiting counts, event-loop stalls, cache behavior, and GPU utilization. The adapter must pass already prepared results across the admission boundary without silently invoking the Hugging Face processor again.

Exit condition: an end-to-end integration demonstrates that native preprocessing reaches vLLM without duplicate work and improves the originally observed admission-bound workload.

## Acceptance criterion

For every declared model and pinned reference configuration, `qwen-mm` accepts encoded images and structured Qwen conversations and produces:

- identical token IDs and visual metadata;
- numerically equivalent visual tensors within a documented tolerance;
- one native batch call with no Python per-item processing; and
- no unnecessary full-image intermediate copies.
