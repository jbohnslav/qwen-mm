# Executable Rosetta verification

Run `sh scripts/test-rosetta.sh --output /tmp/rosetta.json` to build a fresh
production wheel, install it in an isolated environment with hash-locked
reference dependencies, fetch only the two pinned processor snapshots, and
execute the actual Python fences from the
[Rosetta Stone](official-example-rosetta-stone-v0.1.md). No API key or model
weights are needed for this default suite. CI runs it on native macOS ARM and
Linux x86_64. Set `ROSETTA_WHEEL_DIR` to retain the tested wheel.

The runner binds placeholder file paths to generated PNGs and public image URLs
to a local HTTP fixture server. It preserves the code, messages, and options.
`--live-urls` uses the two original public URLs instead. The upstream file-URI
bug in sections 3 and 4 is recorded: pinned Transformers rejects those sources,
so its comparison rerun uses plain paths; qwen-mm accepts the original file URIs.
Direct Transformers examples explicitly request `min_pixels=65536` from
qwen-mm. Composed qwen-vl-utils examples retain the frozen 4096-pixel default.

Every supported section compares keys, shapes, dtypes, integer tensors, and
finite pixels for both profiles. Pixels also face the frozen maximum-error
bound; this small suite does not replace the full resize-v2 fidelity corpus.
The inventory includes constructors, text, URLs, multiple images, heterogeneous
batches, image sizing, OpenAI image content, dataset placeholder expansion,
thinking and tool history, raw RGB, data URIs, and original video rejection.
Video is an expected unsupported-media error. Section 9 documents the serving
adapter boundary; it is not a vLLM or SGLang integration test.

With the [pinned local consumer](transformers-consumer-verification.md) already
downloaded, add `--local-model`. Both literal generation/decoding fences then run
for each supported Qwen3.5 preprocessing section on real Qwen3.5-0.8B weights.
They must produce finite logits; exact inputs require exact first-token logits,
greedy tokens, and decoded text. The native decoder is also compared with the
pinned official tokenizer. The consumer uses the supported 9B processor profile;
this does not add a public 0.8B profile or claim live Qwen3-VL inference.

For hosted verification, explicitly invoke:

```sh
uv run --locked --package qwen-mm-reference python scripts/verify_openrouter.py \
  --key-file /path/outside/repository/openrouter.key \
  --output /tmp/openrouter.json
```

This spends a small amount on seven bounded requests: text, single and ordered
multiple synthetic images, the literal section 6 public image request, a tool
call and result, and thinking enabled. The script records provider usage/cost
and never writes the credential to its report. The section 6 transport changes
are explicit: OpenRouter model/endpoint, a 128-token response cap, flattened
HTTP JSON, and `reasoning.enabled` instead of server-specific chat-template
kwargs. See OpenRouter's [reasoning API](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
and [tool-call protocol](https://openrouter.ai/docs/guides/features/tool-calling).
Hosted answers establish request behavior, not local tensor parity or image
answer accuracy. CI never runs paid calls.

The local tool-history example uses the processor message schema. A complete
hosted response is not a processor message: a serving adapter must handle
transport fields such as tool-call indexes and `tool_call_id`, and represent
an empty assistant body as `""` instead of JSON null. This suite does not claim
a general OpenRouter response adapter.

Set `ROSETTA_RESIZE_OUTPUT=/tmp/resize.zip` on the wheel runner to also capture
all 43 public geometries from the frozen resize-v2 holdout. The JSON and ZIP
retain actual installed-wheel pixels and all fidelity metrics. The two holdout
geometries that cannot be expressed through the public Qwen planner retain
historical core bytes and are identified separately. Existing Phase C overlays
and Phase D performance results are not rewritten or recertified.

Reports are retained under `reference/rosetta/v1/`; each records the document
and runner hashes, and local suites record the installed wheel identity and
dependency versions. `reference/consumer/v1/` is the earlier six-case NumPy
handoff witness. These are correctness witnesses, not new performance
certification or approval to publish.

## Retained results, 2026-09-05

| Evidence | Result |
| --- | --- |
| [macOS ARM](../reference/rosetta/v1/macos-arm64.json) | Both profiles pass; nine Qwen3.5 flows execute the actual generation fences with exact fixture inputs, first logits, tokens, and decoded text. |
| [Linux x86_64](../reference/rosetta/v1/linux-x86_64.json) | Both profiles pass from the production wheel; generation remains the separate ARM consumer witness. |
| [Original public URLs](../reference/rosetta/v1/macos-arm64-live-urls.json) | Both profiles pass exact integer/shape checks and the pixel maximum-error bound. |
| [ARM resize](../reference/rosetta/v1/macos-arm64-resize.json), [Linux resize](../reference/rosetta/v1/linux-x86_64-resize.json) | All 43 fresh public geometries pass the full frozen quality metrics on both wheels; sibling ZIPs contain captured arrays. |
| [OpenRouter](../reference/rosetta/v1/openrouter.json) | Seven requests pass; provider-reported cost $0.00036235. |
| [Legacy comparison](../reference/rosetta/v1/legacy-conformance-comparison.json) | All 290 result records are identical between the pre-change and current wheels. |

The legacy Phase C v1 comparison is **270 pass / 20 fail on both wheels**,
not an all-green v1 claim. Its unchanged failures are the resize-v2 pixel
differences and additive `left_padding` metadata in the old goldens. The
sibling comparison ZIP retains both full reports. No golden or threshold was
weakened. Full `make check`, installed binding/ownership tests, and validation
of the historical v2 overlay also pass; the fresh resize captures above are
the evidence for the current wheels.
