# Paired benchmark v2

This directory owns the apples-to-apples performance protocol. The original
`qwen_mm_reference.bench` module remains the historical one-host baseline; it
is not evidence for a reference-versus-candidate speedup claim.

## Contract

[`workloads-v2.json`](workloads-v2.json) is the single input contract for both
implementations. A worker materializes its structured messages and immutable
encoded or caller-owned RGB buffers before timing, records a SHA-256 fingerprint
over both, then gives the same payload to the reference oracle and selected
implementation. `input_fingerprint` binds the exact bytes delivered on that
host. `logical_input_fingerprint` is identical for ordinary fixtures, raw RGB,
and text; for `generated_encoded` media it instead binds the generator
declaration, messages, ordered formats, and deterministic pre-encode RGB bytes.
Paired subprocess results are rejected if either fingerprint differs.

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
worker's `input_fingerprint`. Lossless and lossy codec output can differ across
architectures even with the same pinned Pillow version, so the separate logical
fingerprint makes generated encoded inputs comparable without claiming their
compressed bytes are identical. Live validation recomputes both fingerprints.
Foreign-architecture portable validation recomputes the logical fingerprint and
requires the exact fingerprints to be valid SHA-256 values that agree between
the paired workers. Package versions and host data remain result provenance.

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

## Correctness prerequisite and release status

Every benchmark result is `DIAGNOSTIC ONLY` and `releasable: false` until D4
implements and passes the controlled-host performance thresholds on an eligible
authenticated host, which may be an appropriately selected Modal CPU class.
For image workloads the harness also evaluates the current Phase C correctness
report at `reference/phase-c/v1/report.json`. This is a necessary correctness
prerequisite, never a performance certification.

The harness computes eligibility rather than accepting a pass boolean. A Phase
C prerequisite passes only when the versioned report covers both pinned
profiles and its complete declared text/image case inventory without skips,
contains no performance claims, still matches every content-addressed input,
and identifies the exact qwen-mm package and native runtime imported by the
benchmark. Result validation also re-hashes the selected benchmark adapter, so
changing or removing either the adapter wrapper or runtime invalidates the
recorded gate.

Case and boundary labels are derived again from the authenticated workload path,
workload/schema hashes, and deterministic input fingerprints. Protocol, pair,
worker, and summary labels must all agree with those current definitions; an
image measurement cannot be relabeled as text to bypass the Phase C prerequisite.

Missing, malformed, stale, incomplete, synthetic, or artifact-mismatched
evidence remains diagnostic. Use `--phase-c-report PATH` only to select another
report location and `--phase-c-assets-root PATH` to locate its authenticated
profile assets; neither option overrides validation.

Image benchmarks default to the current
[`phase-c/v2`](../reference/phase-c/v2/summary.md) correctness overlay. It
authenticates the immutable 290-case v1 report and binds the selected production
resizer to the frozen resize-v2 holdout and installed wheel. The v1 report is
still available as historical exact-Pillow evidence, but it is stale for the
pic-scale production wheel.

[`result-schema-v3.json`](result-schema-v3.json) is the current artifact
schema; [`result-schema-v2.json`](result-schema-v2.json) remains immutable
historical evidence. A result contains exactly one `architecture_family`; ARM and x86 data
must remain in separate files and reports. The harness records performance but
contains no `2x` or regression threshold. Ticket D4 owns release enforcement on
controlled hosts, including an eligible authenticated Modal CPU class; noisy
pull-request workers do not.

The v3 JSON schema fixes the public envelope and requires the named comparison
policy. The program validator deliberately owns the detailed per-occurrence
resize witness checks, including comparison identity, frozen gates, required
channel diagnostics, exact downstream transformation, and matching pre/post
witnesses.

Phase D1 whole-operation capture and the native ARM/Modal x86 publication
workflow are documented in
[`docs/profile-evidence-v1.md`](../docs/profile-evidence-v1.md).
