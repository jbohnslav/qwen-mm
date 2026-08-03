---
id: "ccbc"
status: closed
deps: [8b14, 72e4]
links: []
created: 2026-08-02T22:54:31Z
type: task
priority: 2
closed_at: 2026-08-03T00:37:30Z
assignee: hand
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

- [x] A single documented `modal run ...` command creates an ephemeral pinned
      environment, builds the native x86 wheel, runs binding smoke, executes the
      A5 benchmark self-test, and exits without a persistent deployment.
- [x] Returned artifacts include the validated benchmark JSON/report, build and
      binding-smoke results, command/stdout logs, and complete source/image/
      toolchain/host/resource provenance.
- [x] The runner fails closed on wrong architecture, emulation, missing source
      or assets, build/test failure, invalid benchmark artifacts, or artifact
      download failure.
- [x] No credentials or secrets are committed, and the runner uses the existing
      configured Modal profile without printing token material.
- [x] Documentation gives D1 a narrow follow-on command/interface for real
      candidate profiling and labels Modal correctly as native x86 diagnostic
      infrastructure rather than a dedicated-host D4 result.
- [x] Focused local tests, Modal artifact validation, `git diff --check`, and
      the repository checks affected by the runner pass.

## Worklog

### 2026-08-02 — implementation started in an isolated worktree

Started after A5 and C3 were confirmed closed. Because the main checkout holds
explicitly paused, uncommitted D1 observability WIP, D0 implementation and Modal
execution are isolated at `/private/tmp/qwen-mm-d0-ccbc` on branch
`codex/d0-modal-ccbc` from commit `c92002b`. A collaboration sub-agent was
initially assigned the bounded runner, then implementation returned to the Lord;
an independent sub-agent reviewed the finished code and artifact before closure.

### 2026-08-02 — ticket created ahead of D1

The King paused active D1 work and requested one minimal Modal benchmark ticket
at the top of the Phase D queue. This ticket depends only on the already-closed
A5 harness and C3 correctness gate; D1 now depends on this ticket. Existing
unfinished D1 instrumentation remains uncommitted paused WIP and is outside D0.
- [20:37] — Closed: Completed D0 with commit 655f4dd. Final ephemeral Modal run ap-5rMMt2wqwIsXumD1LyjL9R built the pinned amd64 image, installed the 78-package locked workspace, built and clean-installed an x86-64 ELF wheel, passed binding smoke, ran and validated benchmark v2 self-test, and exited. Integrity-validated artifact: /private/tmp/qwen-mm-modal-d0.zip, SHA-256 0b72ed39fb40dd53f53a51607549fa25b6b724e033931e103c961203cec3a9fc; image im-cPTYvC7wIs0iUbIx4xVYKA; source fingerprint 7b028c597fb7b70d726b279cc9099c6c871c908ce348fd5d60a96f8840fd5e9e/147/19312998. Evidence is explicitly self-test-only, non-releasable, non-dedicated Modal diagnostic data. Focused tests, independent review, exact source/artifact match, local benchmark self-test, git diff --check, and make check passed; 22 asset-dependent Rust tests remained intentionally ignored by the existing suite.
