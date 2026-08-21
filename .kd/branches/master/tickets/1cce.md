---
id: "1cce"
status: closed
deps: []
links: []
created: 2026-08-21T20:54:26Z
type: bug
priority: 1
closed_at: 2026-08-21T20:56:49Z
assignee: hand
parent: cd33
tags: [d4, certification, integrity]
---
# Accept pic-scale scratch buffer in D4 certification semantics

Fresh same-revision ARM and x86 compact archives both contain the legitimate candidate transient buffer resize.pic_scale.scratch emitted by crates/qwen-mm-core/src/resize.rs. D4 certification aborts before gate evaluation because BUFFER_SEMANTICS in performance_certification_v1.py does not recognize it. Fix the frozen validator and add an end-to-end regression that transforms a real/native capture carrying this semantic. Preserve archive provenance: do not mutate captured archives or weaken currentness checks.

## Acceptance Criteria

- [x] Validator assigns resize.pic_scale.scratch the exact static uint8/vector/non-full-image/transient semantics used by the native recorder.
- [x] A regression test fails before the fix and proves certification accepts captured native buffer records while still rejecting unknown names.
- [x] Integrity/currentness checks and all D4 validation tests remain strict; no threshold or archive provenance is changed.

## Worklog

- [2026-08-21 16:56] — Fixed the stale D4 semantic inventory by registering resize.pic_scale.scratch as uint8/vector/non-full-image/transient, matching the Rust pic-scale allocation. Added an end-to-end certification regression with a complete transient lifetime and reconciled allocation/peak counters. Verified 111 D4 tests plus Ruff check/format. Existing same-revision ARM/x86 archives are preserved unchanged; strict currentness means they cannot be relabeled as this fix revision.
