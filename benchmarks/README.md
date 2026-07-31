# Paired benchmark v2

This directory owns the apples-to-apples performance protocol. The original
`qwen_mm_reference.bench` module remains the historical one-host baseline; it
is not evidence for a reference-versus-candidate speedup claim.

## Contract

[`workloads-v2.json`](workloads-v2.json) is the single input contract for both
implementations. A worker materializes its structured messages and immutable
encoded or caller-owned RGB buffers before timing, records a SHA-256 fingerprint
over both, then gives the same payload to the reference oracle and selected
implementation. Paired subprocess results are rejected if their fingerprints
differ.

Every worker:

1. loads one primary adapter and an untimed reference oracle in a fresh process;
2. verifies full output key order, dtype, shape, stride, and values before timing;
3. performs configured warmups;
4. times only the adapter call, checking each materialized result against the
   primary implementation's pre-measurement output after the timer stops; and
5. reruns full candidate-versus-oracle and stability checks after measurement.

The required parity output contract is `float32` pixels plus `int64` IDs, masks,
token types, and grids. Integer values and all structural properties are exact.
Float tolerances are frozen per workload from the compatibility-v1 contract;
timed-iteration stability is always exact.

The two public boundaries are:

- `encoded_to_numpy`: in-memory encoded buffers plus messages to fully
  materialized CPU NumPy arrays;
- `rgb_to_vllm_ready`: caller-owned HWC RGB buffers plus messages to those same
  prepared arrays, suitable for the validated vLLM seam.

Text-only cases use `structured_messages_to_numpy` as a regression guard.

## Workload matrix

The versioned workload includes short/long text, the historical `image1` and
`image24` cases, 24 independent requests, mixed JPEG/PNG/lossless-WebP shapes,
aligned inputs, factor/pixel-boundary shapes, raw RGB, repeated inputs with
cache disabled/enabled plus cache-separated distinct inputs, and 1/4/16/32/64
image scaling.
`image1` and `image24` retain their historical IDs and carry the release names
`jpeg1_offgrid` and `jpeg24_one_request`.

Generated media is deterministic from the recorded seed. It is materialized
outside the timer, and the exact bytes or RGB values are covered by each
worker's input fingerprint. Package versions and host data remain result
provenance.

## Adapters

`official` is the built-in pinned Transformers/Qwen VL Utils implementation and
requires the model artifacts to exist in `reference/.cache/huggingface`.
`synthetic` exercises the protocol itself and marks the resulting artifact
`self_test_only`; it is never valid performance evidence.

A candidate is named as `python.module:factory`. The factory receives an
`AdapterContext(profile_alias, build_label, thread_budget)` and returns an
object with:

```python
name: str


def run(payload: CasePayload) -> Mapping[str, array_like]: ...
def metrics() -> Mapping[str, object]: ...
```

`run` must honor `payload.boundary`, `payload.cache_mode`, exact messages, and
buffers. `metrics` may report `allocation_count`, `copy_count`,
`copy_count_scope`, `transient_live_bytes`, and `cache_supported`. When an
upstream implementation cannot expose allocation or live-byte counters, the
harness records tracemalloc observations and explicitly scopes controlled copy
counts rather than presenting them as complete native measurements.

## Modes and commands

Protocol self-test, including real fresh subprocesses and report generation:

```console
make benchmark-v2-self-test
```

Pull-request smoke against a candidate factory:

```console
make benchmark-v2-smoke \
  BENCHMARK_CANDIDATE=qwen_mm.benchmark:create_adapter \
  BENCHMARK_OUTPUT=/tmp/qwen-mm-smoke.json \
  BENCHMARK_REPORT=/tmp/qwen-mm-smoke.md
```

Smoke uses the tagged correctness/headline subset, one fresh process per
implementation, one warmup, at least three samples, the shipping build label,
and a one-thread budget. It verifies correctness but never fails on timing.

Dedicated-host collection uses five fresh processes per implementation,
randomized AB/BA process order, at least 30 samples and five measured seconds
per process, one-thread and equal production-thread regimes, and shipping plus
native builds:

```console
uv run --locked --no-sync --package qwen-mm-reference \
  python -m qwen_mm_reference.benchmark_v2 run \
  --mode dedicated \
  --candidate-adapter qwen_mm.benchmark:create_adapter \
  --output /var/tmp/qwen-mm-dedicated-arm64.json \
  --report /var/tmp/qwen-mm-dedicated-arm64.md
```

Raw samples record wall and CPU time, throughput, core utilization, process peak
RSS, transient live bytes, allocation/copy observations, cache support, and
thread settings. Summaries contain p50/p90, p99 only at 100 or more samples,
paired per-process speedups, and a seeded bootstrap 95% confidence interval.

[`result-schema-v2.json`](result-schema-v2.json) is the versioned artifact
schema. A result contains exactly one `architecture_family`; ARM and x86 data
must remain in separate files and reports. The harness records performance but
contains no `2x` or regression threshold. Ticket D4 owns release enforcement on
dedicated hosts; noisy pull-request workers do not.
