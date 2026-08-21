---
id: "1bef"
status: closed
deps: []
links: []
created: 2026-08-21T16:10:13Z
type: bug
priority: 1
closed_at: 2026-08-21T16:19:19Z
parent: cd33
---
# Stream Modal D4 progress and validate in the locked local environment

The paid compact x86 capture completed successfully, but scripts/modal_d4.py waited before reading a silent worker stream and make d4-modal-capture ran final archive validation inside the isolated Modal CLI environment, where qwen_mm_reference was unavailable. Preserve live, durable stage telemetry and ensure post-capture validation executes in the repository locked environment without another paid allocation.

## Acceptance Criteria

- [ ] The Sandbox worker emits flushed structured progress for validation, build, conformance, each timed thread budget, archive, and failure/completion stages.
- [ ] The local controller streams stdout and stderr while the worker runs, persists a durable sidecar log immediately, and emits bounded heartbeats with the last observed stage.
- [ ] The make workflow validates a retrieved payload in the locked repository environment where qwen_mm_reference is installed, while retaining raw evidence on failure.
- [ ] Focused tests cover live streaming, failure retention, and the local-environment boundary without paid compute.

## Worklog

- [2026-08-21 12:19] — Reproduced the paid-run defects before implementation: the controller used process.wait() before consuming either stream, the worker emitted no progress, and the standalone Modal CLI environment could not import qwen_mm_reference during final archive validation. Implemented flushed structured worker stages, concurrent stdout/stderr teeing to console and a durable JSONL sidecar, 30-second last-stage heartbeats, raw-payload retention, and a Make workflow that performs final promotion under locked uv. Focused regressions pass; make d4-test passes 109 tests; Ruff and format checks pass; make -n confirms paid retrieval precedes locked finalization.
- [12:19] — Closed: Modal D4 observability and locked-environment finalization are implemented and fully tested without allocating paid compute.
