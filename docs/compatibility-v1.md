# ADR 0001: qwen-mm compatibility contract v1

- Status: accepted for implementation
- Date: 2026-07-31
- Contract ID: `qwen-mm-compat-v1`
- Machine-readable profiles: [`reference/compatibility/v1.json`](../reference/compatibility/v1.json)
- Supersedes conflicting compatibility statements in `DESIGN.md`

## Decision

`qwen-mm-compat-v1` is compatible with the exact composed Python path and the
two immutable profiles defined below. It is not an attempt to preserve every
behavior accepted by current or future versions of Transformers, Qwen VL
Utils, Pillow, TorchVision, or the model repositories.

The implementation must reject an unknown profile, changed artifact, changed
source fingerprint, unsupported option, or input outside the declared envelope.
It must not silently fall back to a nearby upstream version or a more permissive
interpretation.

## Release oracle

For each request, the oracle is this composition, in this order:

1. Load `AutoProcessor` from the pinned model repository revision.
2. Render the structured messages with the repository's pinned template via
   `processor.apply_chat_template` using `tokenize=False` and the supported
   options in this ADR.
3. Call `qwen_vl_utils.process_vision_info` on the same messages with
   `image_patch_size=processor.image_processor.patch_size`,
   `return_video_kwargs=True`, and `return_video_metadata=True`.
4. Separate `(video, metadata)` pairs while preserving encounter order.
5. Call the resolved `Qwen3VLProcessor` with:

   ```python
   processor(
       text=rendered_texts,
       images=images_or_none,
       videos=video_values_or_none,
       video_metadata=video_metadata,  # only when videos are present
       padding=True,
       truncation=False,
       do_resize=False,
       do_sample_frames=False,
       return_tensors="np",
   )
   ```

6. Materialize CPU NumPy arrays. Network, model weights, model execution, GPU
   transfer, and post-generation decoding are not part of the oracle.

`do_resize=False` is mandatory because Qwen VL Utils has already resized image
and video inputs. A direct Transformers call with `do_resize=True` is a
diagnostic path, not a parity oracle.

For a batch, chat rendering is per conversation. Media are flattened in
conversation order, then message order, then content-item order. The processor
receives the rendered strings as one list and pads them together. Output media
rows and grid entries retain that traversal order.

## Immutable profiles

The profile manifest contains the package lock hash, relevant installed-source
hashes, model artifact hashes, resolved classes, preprocessing constants,
special token IDs, and behavior-changing environment. Its profile fingerprint
is SHA-256 of a JSON object containing `contract_id`, `python`, `lock`,
`packages`, `source_files`, `environment`, `oracle_kwargs`, and `profile` (with
the profile's `fingerprint` member removed), serialized as UTF-8 canonical JSON
with sorted keys and separators `(',', ':')`.

The two accepted profiles are:

| Alias | Repository revision | Profile fingerprint |
| --- | --- | --- |
| `qwen3-vl-8b` | `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | `9e2e515f166fdad60e68528aadd7ef2a410724b74855f1b90dfbe1eadcaa7ae1` |
| `qwen3.5-9b` | `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a` | `4f870d0c41812a7f5fe318d9ab7db64a805bac569c9fa4570e957752fd69edb5` |

Both resolve under the lock to `Qwen3VLProcessor`, `Qwen2VLImageProcessor`
with the TorchVision backend, `Qwen3VLVideoProcessor`, and `Qwen2Tokenizer`.
Both use spatial patch size 16, temporal patch size 2, merge size 2, RGB
mean/std `[0.5, 0.5, 0.5]`, and a flattened patch width of
`3 * 2 * 16 * 16 = 1536`.

The image processor assets advertise 65,536 through 16,777,216 pixels, but the
composed image path bypasses that resize policy. Qwen VL Utils uses factor 32
and defaults to 4,096 through 16,777,216 pixels. The composed values win.

An upstream change creates a new contract/profile and conformance corpus. It
never mutates this profile in place.

### Environment

The semantic profile requires:

- `MODEL_SEQ_LEN` unset, which selects Qwen VL Utils' value `128000`;
- `FORCE_QWENVL_VIDEO_READER` unset; encoded-video backend behavior is not
  certified by v1 and will be pinned by E3;
- `TORCHCODEC_NUM_THREADS` unset, which selects `8` if that adapter is used;
- Python 3.11 with exact package versions and source hashes from the manifest.

Thread-count variables (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and
`TOKENIZERS_PARALLELISM`) affect performance scheduling, not accepted values.
They must still be recorded in conformance and benchmark provenance. The v1
reference capture leaves them unset.

## Request and message envelope

One request contains a non-empty ordered message list. A batch contains at
least one request. All strings are valid UTF-8.

Supported roles and content are:

- `system`: zero or one, first only, with string content or text items only;
- `user`: string content or an ordered list of text, image, or video items;
- `assistant`: string content or text items, with optional tool calls;
- `tool`: string content or an ordered list of text, image, or video items.

System visuals are rejected. Although one pinned template can ignore such an
item while Qwen VL Utils still extracts it, that creates an inconsistent
placeholder/media count and is not a supported behavior.

Content items use exactly one recognized kind. Text requires `text`. Image
requires one supported image payload. Video requires one supported video
payload. Unknown roles, item kinds, missing payloads, conflicting kinds, and a
rendered placeholder count that differs from supplied media are errors.

Media occurrences are not deduplicated in parity mode. Reusing the same image
object or bytes in two content items produces two ordered media occurrences,
two grid rows, and two sets of pixel rows. Caching may reuse computation only
when the resulting observable output and occurrence accounting are unchanged.

### Template options

Supported for both profiles:

- `add_generation_prompt: bool`;
- `add_vision_id: bool`;
- a JSON-compatible list of tool definitions;
- assistant tool calls in direct or `{ "function": ... }` form;
- consecutive tool responses.

Supported only by `qwen3.5-9b`:

- `enable_thinking: bool` for a generation prompt;
- assistant `reasoning_content: string`;
- the pinned template's extraction of reasoning from `<think>...</think>` when
  `reasoning_content` is absent.

The default is `add_generation_prompt=false`, `add_vision_id=false`, and for
Qwen3.5 thinking enabled as encoded by the pinned template. Option values and
rendered UTF-8 bytes are part of exact parity.

The following are rejected rather than approximated:

- arbitrary chat-template overrides;
- `documents`;
- `continue_final_message`;
- assistant-token masks;
- `load_audio_from_video` and audio inputs;
- Qwen3.5 thinking/reasoning options on the Qwen3-VL profile;
- tokenizer truncation, left padding, or a caller-selected pad token.

`tokenize=False` and `return_dict=False` are internal rendering choices, not
public options. Tokenization occurs exactly once in the composed processor.

### Padding and sequence length

Text arrays are right padded to the longest rendered sequence in the batch.
The profile-specific pad token is used, `attention_mask` is zero on padding,
and `mm_token_type_ids` follows the official processor exactly. No truncation
is performed. A request that would exceed the configured token limit fails
before any destination is written.

## Media envelope

### Images

The public v1 image API supports:

- encoded JPEG, PNG, and WebP bytes;
- caller-owned 8-bit RGB with explicit height, width, and non-negative row
  stride; and
- grayscale, RGBA, and CMYK only when supplied as encoded JPEG/PNG/WebP and
  converted by the pinned parity color rules.

Alpha is composited onto white, matching Qwen VL Utils 0.0.14 for RGBA. Other
Pillow-convertible encoded modes are not accepted until they have fixtures.

URLs, filesystem paths, data URLs/base64, PIL objects, arbitrary NumPy/Torch
tensors, and mutable Python dictionaries are accepted by parts of the upstream
stack but are outside the Rust public contract. An outer adapter may bind them
to encoded bytes or RGB before calling the core; that adapter is not parity
certified by v1.

Per-item `min_pixels`, `max_pixels`, and paired `resized_height`/
`resized_width` are supported integer options. The effective resize factor is
32. `max_pixels >= min_pixels` is required, and an aspect ratio greater than
200 is rejected. Python's ties-to-even `round` behavior is exact contract
behavior.

EXIF orientation is intentionally ignored. Qwen VL Utils 0.0.14 opens the
stored raster, converts it to RGB, and resizes without EXIF transpose. A named
future non-parity mode may offer orientation correction; parity mode must not.

### Video

The core v1 surface supports caller-owned sequences of 8-bit RGB frames with
common dimensions. Encoded video is handled only by the single adapter to be
pinned in E3; URLs and paths are not core inputs.

Raw-frame items support `sample_fps`, `raw_fps`, `min_pixels`, `max_pixels`,
`total_pixels`, and paired `resized_height`/`resized_width`. Frames remain in
input order. An odd frame count repeats the final frame to temporal factor 2.
The composed-path quirk is preserved: Qwen VL Utils passes the already doubled
image factor into `fetch_image`, whose own factor calculation doubles it again
for the per-frame image resize. The later video budget and bicubic-antialias
resize then operate on that result.

Encoded-video sampling options `fps`, `nframes`, `min_frames`, `max_frames`,
`video_start`, and `video_end` belong to the E3 adapter. When that adapter is
added, `fps` and `nframes` remain mutually exclusive, sampled indices and
timestamps are exact, and `nframes` must be in `[2, total_frames]` after
factor-2 rounding.

## Official outputs

The parity adapter returns a C-contiguous map with the same conditional key set
as the pinned `Qwen3VLProcessor`:

| Key | Dtype | Shape | Presence |
| --- | --- | --- | --- |
| `input_ids` | `int64` | `[batch, sequence]` | always |
| `attention_mask` | `int64` | `[batch, sequence]` | always |
| `mm_token_type_ids` | `int64` | `[batch, sequence]` | always |
| `pixel_values` | `float32` | `[image_patches, 1536]` | images present |
| `image_grid_thw` | `int64` | `[image_occurrences, 3]` | images present |
| `pixel_values_videos` | `float32` | `[video_patches, 1536]` | videos present |
| `video_grid_thw` | `int64` | `[video_occurrences, 3]` | videos present |

There are no zero-length placeholder arrays for absent modalities. Pixel rows
have strides `[6144, 4]`; grid rows have `[24, 8]`; text rows have
`[sequence * 8, 8]`. Shapes, dtypes, strides, conditional presence, row order,
and values are parity requirements.

The Rust core may use compact integers, ragged text, or lower-precision pixels
internally. Those representations must be materialized as the table above by
the parity adapter. `int32`, `bf16`, and ragged outputs are opt-in non-parity
modes and cannot be used for conformance or headline benchmarks.

### Integration sidecar

Video processing also retains, per occurrence and in order:

- `fps: float64`;
- `frames_indices: int64[]`;
- `total_num_frames: float64`;
- effective `sample_fps: float64`; and
- the exact prompt replacement range in code-point and token coordinates.

Images retain their prompt replacement ranges and grid-row index. These values
are an integration sidecar, not extra keys in the model-input map. A6 may add
vLLM-specific identifiers, but may not change the official arrays or the
occurrence/range semantics frozen here.

## Comparison policy

Tolerances are fixed before Rust kernels are evaluated. They may be tightened
by a new contract after evidence, but never widened for a candidate.

| Boundary | Rule |
| --- | --- |
| Rendered/expanded prompt | exact UTF-8 bytes |
| IDs, masks, grids, occurrence order, placeholder counts, frame indices, ranges | exact |
| Key presence, shape, dtype, strides | exact |
| Caller-owned RGB and lossless decode before resize | exact bytes |
| PNG and lossless WebP prepared RGB | exact bytes |
| JPEG and lossy WebP prepared RGB | same dimensions; per-channel absolute byte difference at most 1 |
| Pillow image resize result | same dimensions; per-channel absolute byte difference at most 1 |
| TorchVision video resize in its 0..255 `float32` domain | `rtol=0`, `atol=1e-4` |
| Raw-frame `fps`, `sample_fps`, and `total_num_frames` | `rtol=0`, `atol=1e-9`; frame indices and one-decimal prompt timestamps remain exact |
| Pinned encoded-video adapter FPS reported by E3 | `rtol=0`, `atol=1e-6` Hz; sampling count/indices remain exact |
| Normalize/patchify from identical prepared RGB or frames | `rtol=0`, `atol=1e-6` |
| JPEG/lossy-WebP final normalized tensor | `rtol=0`, `atol=(2/255 + 8*float32_epsilon)` |
| Raw-frame video final tensor | `rtol=0`, `atol=1e-6` after a passing resize-stage comparison |

For every numeric comparison, report maximum absolute error, RMSE, absolute
error p50/p90/p99, count over the bound, maximum ULP distance, and the first
differing request/media/grid/patch/channel coordinate. ULP distance is
diagnostic because values at or near zero make it a poor standalone gate.

Invalid cases compare by stable error category, not Python exception text.

## Resource policy

Upstream acceptance alone does not make an input safe. The parity API applies
checked arithmetic and the following configurable defaults before allocating
or decoding. A caller may lower them. Raising them creates a distinct named
runtime policy but does not change output semantics for inputs common to both
policies.

| Limit | Default |
| --- | ---: |
| requests per batch | 256 |
| messages per request | 256 |
| content items per request | 1,024 |
| UTF-8 text/tool JSON bytes per request | 4 MiB |
| media occurrences per request / batch | 64 / 256 |
| encoded bytes per item / batch | 64 MiB / 1 GiB |
| decoded source pixels per image or frame | 67,108,864 |
| decoded edge length | 32,768 |
| raw frames per video | 768 |
| rendered tokens per request | 262,144 |
| rendered tokens per batch | 1,048,576 |
| materialized output bytes per batch | 4 GiB |

The semantic image prepared-pixel maximum remains 16,777,216 per occurrence.
Video frame budgets follow the pinned Qwen VL Utils constants and
`MODEL_SEQ_LEN=128000`. All products, offsets, strides, and capacities use
checked arithmetic. An overflow fails even if a configured numeric limit would
otherwise permit the input.

Planning is atomic. `plan_batch` validates inputs and exact capacities without
writing caller memory. `execute_plan_into` rejects stale plans and undersized
or overlapping destinations before writing anything.

## Stable error categories

Public errors have a stable category plus structured context. Python exception
strings are diagnostic only.

| Category | Conditions |
| --- | --- |
| `invalid_request` | empty batch/messages, invalid role order, malformed content/tool call, placeholder/media mismatch, invalid UTF-8 at a foreign-function boundary |
| `unsupported_option` | template override, documents, truncation, audio, unsupported thinking option, or other excluded surface |
| `unsupported_media` | unsupported codec, mode, object kind, URL/path, encoded video without the pinned adapter |
| `profile_mismatch` | unknown alias, revision, asset/source/lock fingerprint, or preprocessing constant |
| `media_decode` | recognized encoded media is corrupt, incomplete, or cannot be decoded |
| `media_geometry` | zero dimension, aspect ratio greater than 200, inconsistent frame sizes/stride, or invalid resize/frame option combination |
| `resource_limit` | any configured count, byte, pixel, token, frame, or output limit is exceeded |
| `arithmetic_overflow` | checked size/offset/stride/capacity arithmetic overflows |
| `destination_too_small` | caller-owned output storage is absent, overlapping, misaligned, or undersized |
| `internal_invariant` | a validated plan becomes inconsistent or an unreachable processor state occurs |

Known upstream assertions (`max_pixels < min_pixels`, both `fps` and `nframes`)
map to `media_geometry`. Unknown profile files never fall through to
`media_decode`. Failure order is deterministic: request structure, profile,
options, resource preflight, decode, geometry/stage processing, destination
validation, then execution invariants.

## Non-goals

This contract does not promise:

- compatibility with unpinned upstream versions or model repository heads;
- arbitrary Hugging Face `ProcessorMixin` inputs and kwargs;
- URL fetching, filesystem access, base64 decoding, audio, or model execution;
- EXIF-corrected output in parity mode;
- multiple encoded-video backends;
- implicit truncation or best-effort recovery from malformed media;
- compact dtypes as parity output; or
- post-vision-encoder embeddings (`image_embeds`).

## Consequences for downstream tickets

- A2 exports every official key, rendered bytes, prepared media, sidecar
  metadata, profile fingerprint, environment, and categorized errors.
- A3 uses the fixed comparison table and error categories above.
- B1 implements the envelope, limits, outputs, profiles, and errors without new
  product choices.
- A6 may discover vLLM-specific field names and cache identifiers, but the
  prepared-pixel, ordering, replacement-range, and no-`image_embeds` decisions
  are already fixed.
