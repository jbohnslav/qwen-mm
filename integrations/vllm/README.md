# qwen-mm for a prebuilt vLLM server

**Status:** CPU integration is verified; real GPU serving is pending Modal billing.
See [the evidence and remaining work](evidence/45e5/README.md).

This external plugin replaces still-image resize, normalization and patch layout
inside vLLM. Clients send ordinary OpenAI-compatible `image_url` requests. vLLM
retains HTTP/Pillow media decoding, chat rendering, prompt tokenization, caches,
GPU transfers, the vision encoder and generation. No vLLM fork or source build
is needed.

The stock HTTP interface at vLLM 0.23.0 has no prepared-pixel payload.
`--enable-mm-embeds` accepts features **after** the vision encoder; qwen-mm
produces pixels **before** that encoder. Do not enable it for this integration.
See `scripts/audit_server_inputs.py` for the executable schema rejection.

## Install and launch

Use Linux x86_64, Python 3.11 and an NVIDIA GPU supported by the published vLLM
wheel. Build qwen-mm itself from this checkout, or supply its matching Linux
wheel. The earlier release-candidate wheel pins NumPy 2.4.6 and cannot coexist
with vLLM 0.23.0; this checkout supports NumPy 2.3.5 through 2.x.

```bash
# In the repository; builds qwen-mm, not vLLM.
uv build --wheel --out-dir dist/native
uv venv --python 3.11 .venv-vllm
uv pip install --python .venv-vllm/bin/python --only-binary=:all: \
  vllm==0.23.0 transformers==5.14.1 numpy==2.3.5
uv pip install --python .venv-vllm/bin/python \
  dist/native/qwen_mm-*.whl ./integrations/vllm

VLLM_PLUGINS=qwen_mm_native_images QWEN_MM_THREADS=1 \
.venv-vllm/bin/vllm serve Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --dtype bfloat16 --max-model-len 4096 --max-num-seqs 4 \
  --gpu-memory-utilization 0.8 --enforce-eager \
  --limit-mm-per-prompt '{"image":4,"video":0}' \
  --mm-processor-kwargs '{"min_pixels":65536,"max_pixels":262144}'

python integrations/vllm/scripts/client.py example.png
```

The other supported profile is `Qwen/Qwen3-VL-8B-Instruct` at revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. Model IDs and revisions are checked
strictly. Preprocessing tests cover both profiles; real serving evidence is
scoped to the model named in the experiment report.

Installing the package exposes `qwen_mm_native_images`; vLLM loads all plugins
by default, so explicitly set `VLLM_PLUGINS` on shared installations. For the
stock baseline, use `VLLM_PLUGINS=''`. The optional `qwen_mm_serving_audit`
entry point is inactive unless `QWEN_MM_AUDIT_PATH` names an output file.

## Contract and limitations

- Still RGB images only; HTTP image decoding/conversion belongs to vLLM.
  Embeddings, prepared dictionaries, video and arbitrary model revisions fail.
- Supported processor options: `min_pixels`, `max_pixels`, and their
  `size.shortest_edge` / `size.longest_edge` aliases. Budgets are areas, not
  edge lengths. Defaults match the pinned HF processor: 65,536–16,777,216.
  Explicit limits must stay within 4,096–16,777,216. Other options fail instead
  of silently falling back to Hugging Face.
- qwen-mm loads hash-pinned lightweight processor assets. `QWEN_MM_CACHE_DIR`
  selects its HF cache directory; `QWEN_MM_THREADS` defaults to one.
- vLLM's decoded-image hashes and processor-option hashes drive its normal
  caches. Repeated images may skip preprocessing and the vision encoder.
  Prompt placeholders are expanded by vLLM exactly once.
- PIL-to-NumPy materialization and native input ownership snapshots copy RGB
  data. Native output is shared with Torch through `from_numpy`; downstream
  item assembly/concatenation, IPC, device transfer and bf16 conversion can
  copy. This is not an end-to-end zero-copy route.
- The current public native API also renders/tokenizes a small synthetic
  image-only prompt, which is discarded. vLLM owns the real conversation.
  This overhead is included in measurements.
- No new D4/F3 production certification or video support is claimed.

## Verify and measure

```bash
uv pip install --python .venv-vllm/bin/python pytest
.venv-vllm/bin/python integrations/vllm/scripts/audit_server_inputs.py --output /tmp/vllm-input-audit.json
.venv-vllm/bin/python -m pytest integrations/vllm/tests
# Requires a locally built Linux wheel at the path documented by the script.
# Allocates one L40S, 8 CPU, 32 GiB RAM, maximum 2400 seconds.
modal run integrations/vllm/scripts/modal_serve.py
```

`serve_experiment.py` runs stock/native/native/stock on the same allocated GPU,
with equal thread, image, context and generation budgets. It retains server
logs and raw request, health-probe, process memory and processor/vision events.
Health latency is a responsiveness proxy, not direct event-loop scheduling lag.
The experiment uses local HTTP inside the allocation, so it includes JSON and
base64 transport but does not represent WAN latency.

The earlier in-process prepared-pixel spike remains in `adapter.py` and its
20 regression tests. It can be registered explicitly by Python callers via
`qwen_mm_vllm.register()`; it is not the default HTTP plugin. Historical source
pin and design evidence remain in `vllm-pin.json` and
[the A6 ADR](../../docs/vllm-prepared-pixel-seam.md).
