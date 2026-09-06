# vLLM integration evidence — 45e5

Status: **CPU integration verified; real GPU serving and benefit unverified.**
The ticket remains open. This is not a production certification or an adoption
recommendation based on serving performance.

## What passed

- Stock OpenAI chat schema/content-parser probe rejects `image_pixels`;
  the Qwen dictionary parser requires post-encoder `image_embeds`.
  `stock-input-audit.json` records exact errors and upstream source hashes.
- Published CPU vLLM 0.23.0 wheel, Python 3.11.15, Torch 2.11.0+cpu,
  Transformers 5.14.1, NumPy 2.3.5, on an Intel i5-1240P (AVX2).
- Actual native pixels for both pinned model profiles, aligned equality with
  Transformers, unequal and repeated-image placeholder ranges, text-only,
  concurrent calls, cache reuse/invalidation and output ownership.
  Tests fail if the HF processor is invoked on the native path, including
  profiling. Final counts are in `cpu-tests.txt`.
- Installed Linux qwen-mm wheel with NumPy 2.3.5: Rosetta regressions,
  documentation examples, usability and media sources (`numpy23-api.txt`).
- Full `make check`: Rust checks/tests, 133 reference tests, wheel smoke on
  NumPy 2.4.6 (`repository-check.txt`). All 105 script tests pass
  (`script-tests.txt`). The reference/development NumPy lock is unchanged.

## Binary provenance

The CPU image was
`vllm/vllm-openai-cpu:v0.23.0-x86_64` at
`sha256:6240a6bba604e607300e47490e3477211f968bdf125211bb877d19a70b8fe844`.
A separate Python 3.11 environment used the published CPU wheel:

```
https://github.com/vllm-project/vllm/releases/download/v0.23.0/vllm-0.23.0%2Bcpu-cp38-abi3-manylinux_2_34_x86_64.whl
sha256:571a4ac24629d6100479cdf20d67b4c5cd1dd9bab1b485cd4f3bd631c961187d
```

The source audit is vLLM commit
`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`. No vLLM source was built or
patched. Only qwen-mm's own Rust extension was built.

## GPU attempt and remaining work

The user authorized a bounded Modal allocation. The GPU image built from
published wheels, including `vllm==0.23.0`, with `--only-binary=:all:`:
Modal image `im-DM3PGleaseVewa41EscTwW`, app
[ap-IGNdg3IBdvUo5Kmpg7jZC3](https://modal.com/apps/jbohnslav/main/ap-IGNdg3IBdvUo5Kmpg7jZC3).
Allocation then failed:

> Please add a payment method to use L40S GPU functions.

No GPU was allocated and no real model inference or serving timing was
observed. The GPU image predates the final harness measurement guards;
rebuild from the final snapshot when billing is available.

The checked-in `modal_serve.py` / `serve_experiment.py` harness bounds the run
to one L40S, 8 CPU, 32 GiB RAM, 2400 seconds, with Qwen3.5-9B revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. It runs ABBA stock/native with
ordinary local HTTP requests, fixed generation settings and empty caches at
each server start. Text, one image, unequal images, repeated images, four-image
requests and concurrency four are included. A failing regression first exposed
shared fixture content falsely labeled cold; final fixtures isolate cold inputs
and deliberately repeat each body for the warm request.

The harness records raw TTFT, completion latency, payload bytes, token counts,
process-tree RSS and health-response delays. Opt-in processor and vision-tower
instrumentation rejects missing native dispatch, HF fallback, event-loop CPU
execution or mismatched tower calls. Those runtime assertions have unit coverage
but **have not run on a real serving engine**. Once GPU testing succeeds, inspect
per-request cache behavior, compare output/token semantics, summarize paired
latency and throughput, and record the adoption decision. Health-response delay
is a proxy for responsiveness; local HTTP does not measure WAN latency.

Historical release candidate `5746358`, D4 MISS, and legacy Phase C outcomes
are unchanged. These integration changes require fresh release artifacts before
publication; the old wheels remain evidence for their original source commit.
