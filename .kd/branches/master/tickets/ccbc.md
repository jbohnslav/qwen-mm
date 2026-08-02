---
id: "ccbc"
status: open
deps: [8b14, 72e4]
links: []
created: 2026-08-02T22:54:31Z
type: task
priority: 2
parent: 9516
---
# D0: Prove a minimal Modal benchmark runner

## Goal

Prove that the existing qwen-mm build and benchmark protocol can run in one
ephemeral native Linux x86 Modal container, with enough reproducible provenance
for D1 to collect profiles later. Keep this an infrastructure spike: one local
runner and one command, not a deployed application or benchmark platform.

## Scope

- Add the smallest local Modal entrypoint that packages the checked-out source,
  creates a pinned Linux build/runtime image, requests explicit CPU and memory,
  and runs one command to completion without deploying a persistent service.
- Build and smoke-test the qwen-mm wheel in the container so the run exercises
  native x86 code rather than an ARM artifact, emulator, or synthetic-only
  package path.
- Run the existing A5 `benchmark-v2-self-test` in its real fresh-subprocess
  mode and retain its JSON/report artifacts as proof that the benchmark
  orchestration works inside Modal.
- Record `uname`, architecture, CPU model/topology, requested/visible CPU and
  memory, container/image identity, source revision and dirty-state fingerprint,
  Rust/Python/package versions, environment, command, and timestamps.
- Define a simple artifact-return/download path and document the exact hook D1
  will use for a real `qwen_mm.benchmark:create_adapter` capture after that
  adapter exists.
- State explicitly that Modal evidence is native Linux x86 diagnostic evidence,
  not proof of an exclusive dedicated host for D4.

## Non-goals

Do not deploy or serve a Modal app, create queues/volumes unless the minimal run
proves they are necessary, implement D1 observability or the candidate adapter,
run the full dedicated suite, or make a performance/certification claim.

## Acceptance Criteria

- [ ] A single documented `modal run ...` command creates an ephemeral pinned
      environment, builds the native x86 wheel, runs binding smoke, executes the
      A5 benchmark self-test, and exits without a persistent deployment.
- [ ] Returned artifacts include the validated benchmark JSON/report, build and
      binding-smoke results, command/stdout logs, and complete source/image/
      toolchain/host/resource provenance.
- [ ] The runner fails closed on wrong architecture, emulation, missing source
      or assets, build/test failure, invalid benchmark artifacts, or artifact
      download failure.
- [ ] No credentials or secrets are committed, and the runner uses the existing
      configured Modal profile without printing token material.
- [ ] Documentation gives D1 a narrow follow-on command/interface for real
      candidate profiling and labels Modal correctly as native x86 diagnostic
      infrastructure rather than a dedicated-host D4 result.
- [ ] Focused local tests, Modal artifact validation, `git diff --check`, and
      the repository checks affected by the runner pass.

## Worklog

### 2026-08-02 — ticket created ahead of D1

The King paused active D1 work and requested one minimal Modal benchmark ticket
at the top of the Phase D queue. This ticket depends only on the already-closed
A5 harness and C3 correctness gate; D1 now depends on this ticket. Existing
unfinished D1 instrumentation remains uncommitted paused WIP and is outside D0.
