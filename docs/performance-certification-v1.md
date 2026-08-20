# Image performance certification v1

> Historical protocol notice: v1 evidence retains the gates and meaning
> recorded in this document. New D3.5/D4 captures use the still-image quality
> and per-coordinate speed gates in
> [`qwen-mm-still-image-resize-v2`](image-resize-contract-v2.md); all other
> capture, provenance, resource, stability, memory, scaling, and miss-reporting
> controls below remain in force.

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
`reference/.cache/huggingface`. For the x86 capture, either authenticate the
local Modal CLI or prepare a controlled local Linux container as described
below.

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
./scripts/with-cargo.sh uv run --locked --no-sync --package qwen-mm-reference \
  python scripts/d4_linux.py
modal run scripts/modal_d4.py --dry-run
modal run scripts/modal_d4.py --short-probe
```

The local capture rejects anything except native macOS ARM on the current M4
baseline. The Modal path uses a CPU-only [VM Sandbox][modal-vm], not a Modal
Function or gVisor container. It supplies `(request, hard limit)` tuples of
`(16.0, 16.0)` physical CPU cores and `(32768, 32768)` MiB, enables
`experimental_options={"vm_runtime": True}`, and sets Modal's supported
24-hour maximum lifetime plus a 10-minute idle timeout. CPU-only
[Modal Sandboxes are not subject to
preemption][modal-preemption]. Each invocation resolves the source-built image,
re-opens it by immutable `im-...` identity, creates exactly one named Sandbox,
and records the `im-...` and `sb-...` IDs. The controller always calls
`terminate(wait=True)`, verifies a terminal status, and detaches in a nested
`finally`; it accepts the downloaded artifact only after this cleanup and writes
a sibling `*.modal-lifecycle.json` record.

The worker rejects anything except KVM, an exact `0-15` process affinity and
effective cgroup cpuset, exactly 16 online logical CPUs mapping one-to-one to 16
physical cores, and nested t1/t2/t4/t8 masks. Root `cpu.max` and `memory.max`
files are not invented when the VM omits them: the fixed limits are instead
bound to the authenticated `Sandbox.create` call and recorded alongside the
observed cpuset. The unchanged native-thread t1 enforcement probes run both
before and after the capture. `--short-probe` runs just these topology and
pre/post affinity controls and still guarantees termination; it cannot produce
performance evidence.

The dry-run includes the explicit resource-price calculation. At the repository
price snapshot it is about $3.04 per requested-resource hour (about $72.94 for
the 24-hour ceiling), before image-build
or provider adjustments; unlike non-preemptible Functions, CPU-only Sandboxes
do not add the Function 3x non-preemptibility multiplier. The base image remains
pinned by digest and installs the locked uv and Rust toolchains. The earlier
Modal/gVisor runtime is permanently ineligible: it reported a one-CPU affinity
mask while two and four native threads consumed 1.885 and 3.775 CPUs.

[modal-vm]: https://modal.com/docs/guide/vm-sandboxes
[modal-preemption]: https://modal.com/docs/guide/preemption

As an alternative x86 provider, `scripts/d4_linux.py` runs inside a native
Linux x86 container on a controlled local host. Its image must contain the
exact clean source commit and Git object database, the pinned assets, the
locked Linux `.venv`, and the build dependencies used by the Modal image. The
container must use cgroup v2 with an exact CPU cpuset, a finite CPU quota at
least as large as the number of physical cores represented by that cpuset, and
a finite memory limit no greater than host `MemTotal`. Set the following values
from the host's `lscpu` topology and use an output directory outside the
ephemeral container:

```console
docker run --detach \
  --name="$D4_ALLOCATION_ID" \
  --cpuset-cpus="$D4_CPUSET" \
  --cpus="$D4_PHYSICAL_CORES" \
  --memory="$D4_MEMORY_BYTES" \
  --hostname="$D4_ALLOCATION_ID" \
  --mount "type=bind,src=$D4_OUTPUT_DIR,dst=/evidence" \
  "qwen-mm-d4:$D4_SOURCE_REVISION" \
  make d4-linux-capture \
    D4_X86_ARCHIVE=/evidence/qwen-mm-d4-x86_64.zip \
    D4_LINUX_HOST_LABEL="$D4_LINUX_HOST_LABEL" \
    D4_LINUX_ALLOCATION_ID="$D4_ALLOCATION_ID"
```

Invoking `d4-linux-capture` attests that this is the only D4 capture workload
in the container and that its cgroup allocation and host power/performance
policy will remain stable for the whole run. It does not claim host physical-
core exclusivity: Docker cpusets do not prevent unrelated host processes from
using those CPUs. The runner records that limitation, authenticates available
governor/frequency/boost policy endpoints, records explicit unavailability,
and requires exact power-policy, cgroup, topology, and affinity equality before
and after capture. Before and after the full matrix it also runs one, two, and
four native `pbkdf2_hmac` worker threads for at least one second under the t1
`taskset` mask. Every thread must report that exact affinity and observed CPU,
and aggregate process CPU/wall must remain at or below 1.25. The unchanged no-
pruning 5% CV gate remains the control for observed contention and
repeatability.

## Capture both hosts

The ARM and x86 captures are independent and may run at the same time from two
shells as long as the checkout stays unchanged. Each host builds and measures
both variants serially on that one host so their toolchain, environment, and
host attestations remain comparable:

```console
make d4-arm-capture
make d4-linux-capture
# A future Modal runtime is an alternative only after its enforcement
# preflight passes:
make d4-modal-capture
```

Run exactly one of `d4-modal-capture` and `d4-linux-capture` for the x86 input.

The default raw archive paths are `/tmp/qwen-mm-d4-arm64.zip` and
`/tmp/qwen-mm-d4-x86_64.zip`. Override `D4_ARM_ARCHIVE` or `D4_X86_ARCHIVE` as
needed. `D4_HOST_LABEL` changes only the descriptive ARM host label.
`D4_LINUX_HOST_LABEL` and `D4_LINUX_ALLOCATION_ID` identify a local Linux
capture; an empty allocation ID defaults to the container hostname. Archive
writes are validated before atomically replacing the destination. Every raw
archive includes the complete Phase-C `outputs/` trees referenced by its four
pre/post reports, including the reconstructed resize bytes and exact NumPy
arrays used for semantic validation.

The Modal launcher first atomically saves a retrieved full-run payload beside
the requested destination as `<stem>.unvalidated<suffix>`. It removes that
diagnostic copy only after local archive, source, asset, Phase-C, and benchmark
validation succeeds. If validation fails after the VM has terminated, the
launcher prints and retains the unvalidated path so the completed remote run
can be inspected or repackaged without launching another worker.

A measured gate miss does not truncate the matrix. In particular, a failed
5% no-pruning CV assessment is written to the build's structured
`failures.json`; the runner continues through every remaining thread budget,
the other build variant, and post-capture Phase C. The final certification
report recomputes those noise gates together with speed, regression, scaling,
and memory gates and lists every miss. A command failure, malformed result,
conformance failure, input/provenance inconsistency, private-environment drift,
or unsafe host-control change still aborts immediately because the remaining
measurements could not be authenticated as one complete capture window.

On a local Linux failure, the runner prints and retains its exact
`/tmp/qwen-mm-d4-linux-*` workspace, including partial result, noise, build,
and log evidence. That workspace is diagnostic and not a certifiable raw ZIP.
Successful runs remove their scratch workspace after the validated archive is
written. Do not use Docker `--rm`: after a failure, read the retained path from
`docker logs "$D4_ALLOCATION_ID"` and extract it with `docker cp` before
removing the stopped container. A successful stopped container can be removed
after the bound archive is verified.

Every build/capture lane enforces the frozen controls below:

- separate clean virtual environments and retained wheel files for portable
  `shipping` and `-C target-cpu=native` builds, with the installed native module
  reconciled to the actual archived wheel bytes;
- the native override is authenticated as a build input; byte-identical native
  and shipping outputs are valid when the compiler finds no target-specific
  code-generation difference, and their equal hashes remain explicit evidence;
- a complete fresh Phase C run before and after each build's timed matrix, with
  both pinned profiles, no failures or skips, and distinct pre/post reports;
- architecture-local Phase C resize-v2 evidence reconstructed from the exact
  installed wheel's archived `pixel_values` and grid arrays for all 43 legal
  processor geometries. Each host's reconstructed RGB8 bytes must pass the
  unchanged candidate-independent max-error, RMSE, p99, bias, SSIM, and exact
  invariant gates; the two direct-core-only slices remain authenticated from
  the committed selection evidence. Cross-architecture SIMD output is not
  required to be byte-identical inside that frozen envelope, while repeated
  execution of one build/input/thread budget remains byte-exact;
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
  `lscpu` topology and visible affinity. Modal authenticates its allocated
  resources; local Linux authenticates its cgroup and power-policy snapshots
  without claiming physical-core exclusivity. macOS records that fixed
  affinity is unavailable instead of inventing placement evidence; and
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
noise assessment. It also requires each structured failure list to equal the
misses recomputed from the unpruned raw medians. It then rebuilds the canonical
artifact from the raw archives using the artifact's original timestamp and
requires exact canonical JSON and rendered-report equality.

The provenance check binds the result to the exact 40-character source
revision; source-tree, workload, schema, model registry, profile asset, wheel,
native module, toolchain, environment, provider, host, cgroup, power policy,
topology, affinity, raw archive, raw manifest, and command identities; and the
exact input and messages for both implementations. Exact input bytes must
match between builds on one architecture. Deterministic logical input and
message identities must also match across ARM and x86. Validation rejects
source or protocol changes after capture; only generated evidence and Kingdom
metadata may be added on top of the captured commit.

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
