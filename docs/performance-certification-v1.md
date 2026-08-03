# Image performance certification v1

Phase D4 certifies the CPU preprocessing boundary from in-memory encoded media
and structured messages through fully materialized NumPy arrays. It compares
the official and candidate processors on native macOS ARM and native Linux x86
without aggregating architectures. The portable `shipping` wheel is the release
gate; an otherwise identical `-C target-cpu=native` wheel is retained as
supplemental evidence only.

The raw ZIPs are the source of truth. `scripts/d4_evidence.py` authenticates
them, derives the versioned JSON artifact and readable report, and can later
rebuild the derived result byte-for-byte. A successful evidence command means
the evidence is internally valid; it does not by itself mean every performance
gate passed.

## Preconditions

Start from the exact clean commit to be measured and do not change source,
benchmark contracts, build inputs, or non-evidence documentation until both
host captures finish. The two pinned profile snapshots must exist under
`reference/.cache/huggingface`, and the local Modal CLI must already be
authenticated for the x86 capture.

Run the focused protocol, archive, and evaluator tests before allocating the
hosts:

```console
make d4-test
```

The runners also have non-measuring plan modes. These check local inputs and
show the requested build/capture plan without producing certification data:

```console
./scripts/with-cargo.sh uv run --locked --no-sync --package qwen-mm-reference \
  python scripts/d4_local.py
modal run scripts/modal_d4.py --dry-run
```

The local capture rejects anything except native macOS ARM on the current M4
baseline. The Modal capture requests one non-preemptible, single-use native
Linux x86 container with 16 CPUs, 32 GiB of memory, and a 12-hour timeout. It
rejects emulation, inadequate cgroup resources, or fewer than eight distinct
physical cores. The Modal image is pinned by digest and installs the locked uv
and Rust toolchains; no long-running app or service remains after `modal run`
returns.

## Capture both hosts

The ARM and x86 captures are independent and may run at the same time from two
shells as long as the checkout stays unchanged. Each host builds and measures
both variants serially on that one host so their toolchain, environment, and
host attestations remain comparable:

```console
make d4-arm-capture
make d4-modal-capture
```

The default raw archive paths are `/tmp/qwen-mm-d4-arm64.zip` and
`/tmp/qwen-mm-d4-x86_64.zip`. Override `D4_ARM_ARCHIVE` or `D4_X86_ARCHIVE` as
needed. `D4_HOST_LABEL` changes only the descriptive local host label. Archive
writes are validated before atomically replacing the destination.

Every build/capture lane enforces the frozen controls below:

- separate clean virtual environments and retained wheel files for portable
  `shipping` and `-C target-cpu=native` builds, with the installed native module
  reconciled to the actual archived wheel bytes;
- a complete fresh Phase C run before and after each build's timed matrix, with
  both pinned profiles, no failures or skips, and distinct pre/post reports;
- the same input, messages, output keys/order, values, dtypes, profile, warm
  state, tolerance policy, and total thread budget for each official/candidate
  pair, with no fallback or cache work;
- five fresh process pairs per coordinate, fixed seed `20260731`, deterministic
  randomized AB/BA order, three warmups, and exactly 30 one-operation latency
  samples per process;
- a separate sequence of individually clocked, exact-output-checked operations
  when needed to reach five measured seconds; these supplemental operations are
  summarized but never enter p50, p90, or p99 distributions;
- full release cases at thread budgets one and eight, plus `image24` at two and
  four, with memory/allocation observation isolated from timing;
- one declared owner for the total thread budget, Torch inter-op fixed at one,
  and measured CPU use bounded by the requested budget plus 0.25 cores;
- fixed, nested physical-core masks on Linux x86, recomputed from the archived
  `lscpu` topology and visible affinity. macOS records that fixed affinity is
  unavailable instead of inventing placement evidence; and
- no sample pruning and a maximum 5% coefficient of variation across the five
  process medians for every measured implementation coordinate.

`repeat24_cached` is preserved as explicitly unsupported and non-gating because
neither adapter implements that cache mode. It is not timed or replaced with an
uncached measurement. Profilers are also excluded from timed certification.
The locked `py-spy==0.4.1` tool may be used for a separate diagnostic when a
miss needs investigation, but its output cannot replace a controlled capture.

## Build and validate evidence

Once both raw archives exist, derive the default
`benchmarks/performance-certification-v1/result.json` and `report.md`:

```console
make d4-certify
make d4-validate
```

Override `D4_ARTIFACT` or `D4_REPORT` for other destinations. Generation writes
both files atomically. Validation reopens both bounded ZIPs, verifies their
manifests and member hashes, extracts and re-hashes the archived wheels,
reconciles build and runtime identities, validates the archived Phase C reports
against the pinned assets, and revalidates every raw benchmark and no-pruning
noise assessment. It then rebuilds the canonical artifact from the raw archives
using the artifact's original timestamp and requires exact canonical JSON and
rendered-report equality.

The provenance check binds the result to the exact 40-character source
revision; source-tree, workload, schema, model registry, profile asset, wheel,
native module, toolchain, environment, host, cgroup, topology, affinity, raw
archive, raw manifest, and command identities; and the exact input and messages
for both implementations. Exact input bytes must match between builds on one
architecture. Deterministic logical input and message identities must also
match across ARM and x86. Validation rejects source or protocol changes after
capture; only generated evidence and Kingdom metadata may be added on top of
the captured commit.

## Frozen release gates

All gates apply to `shipping`; native-build results are supplemental:

- for `image24` and `ragged24`, the lower bound of the deterministic 20,000-
  resample paired process-median bootstrap 95% speedup interval is at least
  `2.0x`, for both profiles on each architecture at eight threads;
- on the M4 ARM baseline, the `image24` candidate p50 is below 111.0 ms for
  Qwen3-VL and below 111.6 ms for Qwen3.5 at eight threads;
- `image1`, `text_short`, and `text_long` have a candidate/reference p50 ratio
  no greater than 1.05 at one and eight threads, for both profiles and
  architectures;
- candidate `image24` parallel efficiency `E_N = t1 / (N * tN)` is at least
  0.60 at two, four, and eight threads, for both profiles and architectures;
  and
- for every image coordinate and process pair, the candidate's larger of
  external transient RSS and exact native peak transient bytes is no more than
  50% of the official external transient RSS. Every official denominator must
  be positive, the candidate allocation/copy census must be complete with no
  dropped events, and no full `float32` HWC or CHW intermediate image is
  permitted.

The published JSON/report pair includes the raw samples, p50/p90 and qualified
p99, throughput, CPU time, core utilization, paired speedups and confidence
intervals, scoped RSS, native allocation/copy/lifetime evidence, conformance
witnesses, and separate ARM/x86 summaries. It makes no video or vLLM
production claim.

## When a target misses

An authentic miss is a valid D4 measurement result. Evidence generation still
writes `result.json` with `certification_status: "miss"` and `releasable: false`;
the report lists every failed unchanged gate and its measured reason. `make
d4-validate` can therefore succeed for a truthful miss—validation proves the
artifact was derived correctly, not that it is releasable.

Do not change a threshold, tolerance, workload, build label, or host claim after
seeing the data. Publish the miss, reopen the relevant optimization ticket, and
escalate the exact gate, observed point estimate or confidence interval,
measured gap, ranked bottleneck, and safest follow-up options. If no safe change
is justified, retain the explicit miss rather than manufacturing a pass.
