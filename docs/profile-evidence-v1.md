# Phase D1 native profile evidence

Phase D1 captures the complete two-profile, six-case matrix on native macOS
ARM64 and native Linux x86-64. Each host uses thread budgets 1 and 4, three
bounded observations per coordinate, and one OS-sampled whole-operation profile
per coordinate. The same single `profiled-release` wheel on a host must pass a
fresh Phase C run and the paired real A5 benchmark before profiling.

The x86 lane is one ephemeral Modal function with 8 physical CPU cores and 32
GiB of memory. It is not a service and stops when the capture returns. Modal is
a real benchmark host for this project; D4 decides whether a selected Modal CPU
class is suitable for the final performance claim.

## Capture and publish

Start from the clean implementation commit and leave the checkout clean until
**both** archives exist. Captures use scratch space and embed their eventual
repository paths, so neither command publishes evidence yet:

```console
make profile-d1-arm-archive
make profile-d1-modal-archive
```

The ARM command requires native Apple silicon and `/usr/bin/sample`. The Modal
CLI must already be authenticated. The x86 runner performs a short native
`py-spy==0.4.1` direct-child preflight before starting the 24 sampled
coordinates. For each x86 coordinate, py-spy launches the authenticated worker
it profiles at 99 Hz. The worker performs unmarked setup, records a monotonic
two-second dispatch deadline, and starts marked whole-operation iterations only
before that deadline. A final operation started before the deadline completes
at its whole-operation boundary; then the worker performs an unmarked output
post-check and exits naturally. Raw setup and post-check stacks are retained as
sampler evidence but excluded from the canonical collapse and rankings. The
bundled worker result authenticates the window, runtime identity, and pre/post
signatures.

Each command writes an integrity-validated ZIP under `/tmp` by default. Archive
validation enforces a 90 MiB compressed return cap, an explicit durable-member
allowlist, and hashes every member. Phase C scratch output is never archived.

Only after both archives were captured from the same revision, atomically ingest
them and run the canonical Phase C, benchmark, and profile validators against
the installed final paths:

```console
make profile-d1-ingest
make profile-d1-merge
make profile-d1-validate
```

Override `PROFILE_D1_ARM_ARCHIVE` or `PROFILE_D1_X86_ARCHIVE` when the archives
are elsewhere. Ingest refuses to overwrite an existing host tree and rolls back
publication if canonical semantic validation fails. A regenerated ZIP manifest
cannot turn modified Phase C, benchmark, or profile content into accepted
evidence. Installed benchmark validation is authenticated-portable for both
hosts: it re-hashes the architecture-independent adapter wrapper, binds the
recorded native identity to the archived wheel/runtime authentication, and
re-hashes the durable Phase C path without comparing against whatever native
extension happens to be installed in the validation environment. Capture-time
validation remains live against the exact profiled wheel and scratch Phase C
report.

The committed layout is:

```text
benchmarks/profile-evidence-v1/
  arm64/       # ARM bundle, raw samples, one wheel, gates, logs, provenance
  x86_64/      # x86 bundle, raw samples, one wheel, gates, logs, provenance
  bundle.json  # canonical merged two-architecture bundle
  report.md    # final machine-derived bottleneck report
```

Run the orchestration unit tests with `make profile-d1-test`. Performance misses
do not invalidate D1 evidence; record them explicitly and escalate target
trouble during D2–D4 rather than weakening or relabeling the measured protocol.
