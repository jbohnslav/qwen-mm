# Python reference benchmark

This directory contains the pinned Python reference used to time and export the
current Qwen multimodal preprocessing path. It loads processor artifacts only;
it does not download model weights or run generation.

The immutable v1 oracle, package/source hashes, model artifact hashes, resolved
classes, and special-token/configuration values are recorded in
[`compatibility/v1.json`](compatibility/v1.json). The normative request,
output, tolerance, limit, and error contract is
[`docs/compatibility-v1.md`](../docs/compatibility-v1.md).

From the repository root, generate or verify the committed baseline images in
the shared locked uv workspace:

```bash
uv sync --locked --inexact --package qwen-mm-reference
uv run --locked --no-sync --package qwen-mm-reference python -m qwen_mm_reference.fixtures generate
uv run --locked --no-sync --package qwen-mm-reference python -m qwen_mm_reference.fixtures verify
```

Run the small baseline on both processor configurations:

```bash
uv run --locked --no-sync --package qwen-mm-reference python -m qwen_mm_reference.bench \
  --models qwen3-vl-8b,qwen3.5-9b \
  --cases image1,image24 \
  --output results/local.json
```

The timed boundary begins with in-memory encoded JPEG bytes and a structured
conversation. It ends with materialized CPU NumPy arrays. Network and disk I/O,
processor initialization, model execution, and output hashing are excluded.

## Golden exporter

Golden cases are versioned JSON request descriptions in `cases/v1/`. The
exporter binds their repository-relative encoded media, then runs the exact
composed v1 oracle and records every stage in a canonical, self-hashed manifest.
It always loads processor artifacts from the local cache with
`local_files_only=True`.

After the locked environment and model artifacts have been cached once, this
single command exports a complete smoke case without network access:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --locked --no-sync --package qwen-mm-reference \
  python -m qwen_mm_reference.golden export \
  --case reference/cases/v1/multimodal-smoke.json \
  --profile qwen3-vl-8b \
  --output reference/goldens/v1/qwen3-vl-8b/multimodal-smoke/manifest.json
```

The manifest contains the exact logical messages and template options, encoded
media signatures/properties, rendered and expanded UTF-8 prompts, replacement
ranges, prepared RGB/frame arrays, video kwargs/metadata, every official output
array, the invocation graph, fixed comparison policy, and full environment,
source, artifact, model, platform, and codec provenance. Every array records its
shape, dtype, strides, byte order, C-order SHA-256, and storage mode.

Arrays up to 1 MiB are stored as full nested JSON values by default. Larger
arrays remain authenticated `signature_only` entries, keeping the 432 MiB
`image24` tensor out of Git while making regeneration byte-verifiable. Pass
`--write-arrays` to materialize larger values as `.npy` files beside the
manifest, or change `--inline-max-bytes` for a different versioned capture
policy.

Export the large signature-only case on demand:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --locked --no-sync --package qwen-mm-reference \
  python -m qwen_mm_reference.golden export \
  --case reference/cases/v1/image24.json \
  --profile qwen3-vl-8b \
  --output reference/goldens/v1/qwen3-vl-8b/image24/manifest.json
```

Validate one manifest or a whole golden tree, including schema requirements,
conditional output keys, inline array data, prompt bytes, provenance, and the
canonical manifest hash:

```bash
uv run --locked --no-sync --package qwen-mm-reference \
  python -m qwen_mm_reference.golden validate reference/goldens/v1
```

## Paired benchmark v2

The historical `qwen_mm_reference.bench` command is a directional official-path
baseline. Reference-versus-candidate measurements use
`qwen_mm_reference.benchmark_v2`, whose workload, result schemas, adapter
contract, CI smoke mode, and dedicated-host protocol are documented in
[`benchmarks/README.md`](../benchmarks/README.md).

Run the protocol without model downloads through its deterministic subprocess
self-test:

```console
make benchmark-v2-self-test
```

## Conformance corpus and comparator

The versioned three-tier corpus, structured comparator, seeded live runner, and
regression minimizer are documented in
[`docs/conformance-v1.md`](../docs/conformance-v1.md). Run the fast committed
gate with:

```bash
make conformance-smoke
```

This executes the compact pinned rules, validates that every required coverage
tag remains represented, and checks all committed goldens and their cross-stage
grid/patch/placeholder invariants. The full seeded matrix takes a candidate
command and runs both pinned profiles against freshly exported oracle results:

```bash
make conformance-full \
  CANDIDATE_COMMAND='qwen-mm-candidate --case {case} --profile {profile} --output {actual}'
```
