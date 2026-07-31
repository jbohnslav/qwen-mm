# ADR 0002: vLLM prepared-pixel seam

- Status: accepted for the A6 spike
- Date: 2026-07-31
- Scope: image inputs for the two `qwen-mm-compat-v1` profiles
- vLLM pin: `v0.23.0` at
  `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`

## Decision

Use an out-of-tree vLLM general plugin that replaces both the registered Qwen
multimodal processor and its data parser. The parser accepts qwen-mm-prepared
`pixel_values` and `image_grid_thw`; the processor emits vLLM's normal
`MultiModalKwargsItem` fields, content-bound per-item hashes, and placeholder
ranges discovered from qwen-mm's final prompt IDs. It never maps prepared pixels
to `image_embeds`, expands final IDs a second time, or calls the Hugging Face
processor on the prepared route.

The existing vLLM extension points are sufficient at the pinned revision. No
upstream patch is required for the image seam. The prototype is deliberately
image-only: video needs an equivalent prepared-input type plus the pinned
Qwen3-VL timestamp replacement metadata before it is safe to register.

The immutable environment and source hashes are in
[`integrations/vllm/vllm-pin.json`](../integrations/vllm/vllm-pin.json). The pin
verifier checks the commit, source hashes, and the exact assumptions used below.

## Why the stock dictionary path is not the seam

The stock Qwen parser treats dictionary image input as post-encoder data and
requires `image_embeds` plus `image_grid_thw`. By contrast, the model's normal
processor-output field configuration already understands `pixel_values` and
slices its first dimension by `image_grid_thw.prod(-1)`. Relabeling prepared
pixels as `image_embeds` would skip the vision encoder and is semantically wrong.

The plugin therefore installs:

1. `QwenMMPreparedPixelParser`, which admits a validated prepared-pixel
   dictionary without enabling vLLM's embedding-input mode;
2. `QwenMMPreparedMultiModalProcessor`, mixed into the pinned
   `Qwen3VLMultiModalProcessor`, which tokenizes only when needed and otherwise
   echoes the already prepared fields into vLLM's base processor pipeline; and
3. a `vllm.general_plugins` entry point that replaces the processor factories
   for `Qwen3VLForConditionalGeneration` and the frozen dense
   `Qwen3_5ForConditionalGeneration` profile.

The installed plugin is deliberately fail-closed: it accepts only
`Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` and
`Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Because general
plugins replace factories by architecture class, any other model ID or missing
or different revision raises during processor-info construction instead of
silently claiming a frozen compatibility profile.

The registered processor API and overwrite behavior are visible in the pinned
[`MultiModalRegistry`](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/multimodal/registry.py#L142-L174).

## Prepared input contract

`multi_modal_data["image"]` is one mapping with these fields:

| Field | Contract |
| --- | --- |
| `pixel_values` | CPU, C-contiguous `float32`, shape `[sum(grid_t*grid_h*grid_w), 1536]` |
| `image_grid_thw` | CPU, C-contiguous `int64`, shape `[images, 3]`; positive; `grid_t == 1`; H/W divisible by merge size 2; 16–65,536 patches per image |
| `qwen_mm_contract_id` | exactly `qwen-mm-compat-v1` |
| `qwen_mm_profile_fingerprint` | the exact fingerprint registered for the selected model |
| `qwen_mm_cache_keys` | one non-empty deterministic key per image occurrence |

The accompanying `TokensPrompt.prompt_token_ids` are the final qwen-mm IDs: each
image's image-token run is already expanded to
`grid.prod() / spatial_merge_size**2`. The adapter uses vLLM's prompt-update
metadata to discover and validate those runs; it does not alter the IDs.

The adapter rejects `image_embeds` and `video_embeds`. It initially wraps NumPy
with `torch.from_numpy`, validates the complete mapping, then clones pixels and
grids into adapter-owned CPU storage before hashing or cache insertion. This
intentional admission copy prevents caller mutation from changing cached data
under an existing identity.

The supported profile fingerprints are:

| Profile | Fingerprint |
| --- | --- |
| `qwen3-vl-8b` | `9e2e515f166fdad60e68528aadd7ef2a410724b74855f1b90dfbe1eadcaa7ae1` |
| `qwen3.5-9b` | `4f870d0c41812a7f5fe318d9ab7db64a805bac569c9fa4570e957752fd69edb5` |

## Pinned execution path

1. A caller supplies already rendered prompt IDs and the prepared mapping in a
   vLLM `TokensPrompt`.
2. The async renderer runs multimodal parsing and processing in its renderer
   thread pool.
3. The custom parser returns one item per grid row and supplies prepared fields
   as processor data, not passthrough embedding data. This preserves vLLM's
   processor-cache path.
4. vLLM's base processor constructs normal field items. `pixel_values` uses a
   flat slice sized by grid product; `image_grid_thw` is batched and retained on
   CPU.
5. The custom prompt metadata describes each already-expanded image-token run.
   vLLM finds each unequal per-image run in occurrence order, leaves the final
   IDs unchanged, and stores its exact range as `mm_placeholders`.
6. `InputProcessor` converts those values to `MultiModalFeatureSpec`, retaining
   the processed item, modality, hash/identifier, and placeholder position.
7. The Qwen model's pinned `_parse_and_validate_image_input` selects the
   `pixel_values` branch, converts pixels to the vision tower dtype, and calls
   the vision encoder. It does not select the `image_embeds` branch. See the
   pinned [Qwen3-VL model boundary](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/model_executor/models/qwen3_vl.py#L2039-L2112).

## Hashing and cache identity

vLLM computes one multimodal hash per item from model ID, modality item, and
processor kwargs. The prepared item contributes:

- `qwen-mm-compat-v1`;
- the full profile fingerprint; and
- the caller's prepared cache key;
- the owned per-image pixel tensor; and
- its grid row.

The qwen-mm cache key must itself cover source media bytes or an immutable media
digest, every resize/sampling/template option that affects the prepared result,
and the ordered media occurrence's relevant metadata. Repeated references may
reuse the same prepared key; occurrence order and replacement ranges remain
request-local. Future video keys must additionally cover sampled frame indices,
true/sample FPS, timestamps, and backend/profile identity.

The prototype deliberately hashes owned tensor content as well as the caller's
key. This makes a reused or incorrect key fail safe rather than replay the wrong
pixels or replacement length. B1/C1 may later replace the full scan with a
trusted native content digest, but only if the producer owns immutable storage
and binds that digest to all output-affecting inputs.

Because prepared items are processor data and return no passthrough fields, the
normal vLLM processor cache remains active. A miss stores the
`MultiModalKwargsItem` and resolved prompt update; a hit can omit the item from
API-process-to-engine IPC and replay the cached data/range metadata. vLLM's
P0/P1 cache mirroring and shared-memory variants are mapped in the pinned
[`cache.py`](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/multimodal/cache.py).

The dummy-input builder uses synthetic keys only during vLLM's startup profiling
pass. At the pinned revision, `InputProcessor` explicitly resets that processor
cache after `MultiModalBudget` construction, before live admission. The pin
verifier locks this reset assumption; removing it upstream would make profiling
keys unsafe to share with the live cache and must fail the verifier.

The plugin's processor, info, and dummy-builder classes are module-level so
their qualified names are importable and pickle-safe in multiprocess engine
deployments. Pinned vLLM itself uses `Qwen3VLMultiModalProcessor` for dense
Qwen3.5, with `Qwen3_5ProcessingInfo` supplying that model's distinct config.

## Slicing, ownership, IPC, and dtype conversions

- NumPy `float32`/`int64` first wraps without a copy, then one deliberate clone
  creates the adapter-owned cache snapshot. Field splitting creates views of
  that snapshot.
- On a processor-cache cold miss, pinned vLLM reparses the missing-item sequence;
  its `torch.cat` path can add another CPU copy even for one item. Cross-request
  or model batching can allocate another concatenated tensor.
- Grid metadata remains on CPU. Prepared pixel tensors cross the API-to-engine
  boundary on the first miss; process separation therefore entails IPC
  serialization or shared-memory publication. A cache hit may carry `data=None`
  and skip that transfer.
- The model runner performs the required CPU-to-GPU transfer. The Qwen model
  then converts public parity `float32` pixels to the vision-tower dtype, usually
  BF16/FP16, which can allocate another device tensor. A future explicitly
  non-parity output mode may avoid that conversion; the v1 parity API may not.
- Prepared pixels are still pre-encoder inputs. The vision encoder runs exactly
  once in the normal Qwen model path.

## Async admission requirement

At this pin, `BaseRenderer` owns a `ThreadPoolExecutor` sized by
`renderer_num_workers`, and `_process_multimodal_async` submits the complete
multimodal processor call to that executor. This is the scheduling mechanism
that keeps preprocessing off the asyncio event-loop thread; releasing the GIL
inside qwen-mm enables useful worker concurrency but is not a substitute for
offloading the call. See the pinned
[`BaseRenderer`](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/renderers/base.py#L74-L106).

Direct callers must not pass a raw prompt straight to the deprecated
`AsyncLLM.generate()` compatibility path, because that branch still calls the
synchronous `InputProcessor.process_inputs()`. They must await
`renderer.render_cmpl_async()` or `renderer.render_chat_async()` first and pass
the resulting engine input to `generate()`. The OpenAI server already uses the
async renderer.

```python
prepared_prompt = {
    "prompt_token_ids": rendered_ids,
    "multi_modal_data": {"image": prepared_images},
}
engine_input = (await llm.renderer.render_cmpl_async([prepared_prompt]))[0]
async for output in llm.generate(engine_input, sampling_params, request_id):
    ...
```

## Executable proof

The proof runs against a real empty-device build of the pinned vLLM source. Its
fail-fast spy makes `AutoProcessor.from_pretrained` and
`info.get_hf_processor()` raise. A successful result proves the prepared route
does not use the official processor. Tests also cover:

- unchanged final prompt IDs plus exact unequal multi-image placeholder ranges;
- functional prepared routing for both frozen profile fingerprints and their
  distinct image-token IDs;
- output field names and absence of `image_embeds`;
- caller-buffer mutation isolation;
- real processor-only and sender/receiver LRU cold/warm replay, including
  sender-side `data=None` on a warm hit;
- cache separation by key, tensor content, grid, and profile;
- fail-closed model ID and revision validation;
- deterministic HF-free dummy profiling and frozen maximum bounds;
- re-entrant plugin registration for both frozen model classes; and
- import/pickle round trips for every class stored in the registered factory.

Commands and environment construction are documented in
[`integrations/vllm/README.md`](../integrations/vllm/README.md).

## Downstream contracts

### B1: profiles, types, and limits

- Expose contract ID, full profile fingerprint, ordered per-occurrence cache
  key, `float32` pixels, `int64` grids, and patch/grid invariants in the Python
  result type.
- Expose final expanded prompt IDs and ordered image ranges; do not hand the
  adapter pre-expansion template markers.
- Permit the adapter to take one owned snapshot at admission. A future borrowed
  mode needs an enforceable immutable owner, not a prose-only mutation rule.
- Keep image prepared inputs distinct from post-encoder embeddings at the type
  level. Do not offer a generic dictionary that can contain both.
- Reserve a versioned video sidecar for frames, FPS, sampled indices,
  timestamps, and replacement coordinates.

### C1: batch executor and ownership

- Return one content-bound stable cache identity per media occurrence before
  results are flattened into batch tensors.
- Keep per-item pixel row spans derivable exactly from `grid_thw.prod()` and
  preserve occurrence order during parallel execution.
- Release the GIL around the native batch operation, but measure admission only
  through vLLM's renderer executor boundary.
- Make storage ownership explicit. The safe baseline transfers an owned snapshot;
  borrowed views require a lifetime-enforced immutable native owner.

### F1: integration and copy accounting

- Count NumPy wrapping, multi-item concatenation, API/engine IPC, H2D, and
  vision-dtype conversion separately.
- Test cache-off, LRU, and shared-memory modes with both cold misses and warm
  hits. Cache correctness must be keyed by the full prepared identity above.
- Add the equivalent video parser only after exact timestamp replacements and
  cache metadata have conformance fixtures.
