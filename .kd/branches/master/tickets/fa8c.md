---
id: "fa8c"
status: closed
deps: [8eb9, 832f, 2896, 6b73, 015f, 7b4b, 5e62, c3b2]
links: [e48a]
created: 2026-08-01T00:29:33Z
type: epic
priority: 2
closed_at: 2026-08-01T04:08:30Z
assignee: hand
---
# Phase B: Serial Rust vertical slice

## Goal

Implement the first end-to-end Rust image processor as a correctness-first,
single-request serial path. The slice must load the two frozen profiles,
reproduce the declared text and image semantics, and run without Python at
runtime before batching, bindings, parallelism, or performance work begins.

## Child Tickets

- B1 / `8eb9` — define profiles, public/core types, limits, and errors.
- B2 / `832f` — implement the pinned tokenizer and chat-template surfaces.
- B2.1 / `c3b2` — close the independently discovered A3 chat-corpus gap.
- B3 / `2896` — implement geometry and `smart_resize` rules.
- B4 / `6b73` — implement raw-RGB normalization and patchification.
- B5 / `015f` — prove and select parity-safe resize kernels.
- B6 / `7b4b` — implement decoding, color conversion, and prepared-RGB semantics.
- B7 / `5e62` — integrate the serial single-request image path.

## Execution Shape

B1 is the first executable ticket. B2 and B3 can proceed independently after
B1. B4 follows B3, while B5 independently proves the resize boundary after B3;
B6 follows B5. B7 integrates B2, B4, and B6. The individual Phase A
dependencies remain on the child tickets so each worker can see its actual
oracle or corpus prerequisite.

## Scope Boundaries

This phase is correctness-first and serial. Native batching, the two-phase
caller-buffer API, PyO3/NumPy ownership, parallel execution, performance claims,
raw-frame video semantics, encoded video, and vLLM integration belong to later
phases. B5 may evaluate the future video resize kernel because that decision is
part of the frozen stage boundary, but Phase B does not expose a video API.

## Acceptance Criteria

- [x] B1 through B7 are closed with implementation and verification evidence in
      their worklogs.
- [x] A single native Rust request supports text plus one or more encoded or
      caller-owned RGB images for both frozen profiles without importing or
      invoking Python, Hugging Face, Qwen VL Utils, Pillow, or TorchVision at
      runtime.
- [x] Rendered text, IDs, masks, modality IDs, occurrence order, grids, shapes,
      dtypes, strides, errors, and resource-limit behavior satisfy the frozen
      exact rules; every numeric image stage satisfies its immutable tolerance.
- [x] One-image, interleaved-image, and repeated-image cases pass committed and
      differential conformance for both Qwen3-VL and Qwen3.5 profiles.
- [x] No resize dependency or custom kernel is accepted before B5 records stage
      evidence against pinned Pillow and TorchVision behavior.
- [x] The repository's full correctness suite passes from a clean checkout, and
      `ROADMAP.md` is updated only if Phase B evidence changes downstream scope
      or dependencies.

## Worklog

### 2026-08-01 — Phase B accepted

All dependency tickets are closed: B1 profiles/types/limits, B2 tokenizer/chat,
B3 geometry, B4 normalize/patchify, B5 resize evidence, B6 decode/color, B7
serial composition, and the audit-created B2.1 chat-corpus remediation. The
Lord reviewed each integration, ran independent and adversarial audits, required
corrections before acceptance, and used collaboration subagents only; no
Kingdom peasant worker was invoked.

The final Rust-only Phase B corpus passed all 10 profile/case combinations and
62 comparisons with zero values over bound on native macOS ARM. The same
authenticated command also passed 10/10 with zero values over bound under
Linux x86_64 Docker/QEMU using Rust 1.97.1. Two independent official-oracle
regenerations matched the committed Phase B manifest, sources, and expected
blobs byte-for-byte.

`make check` passed Ruff, formatting, strict workspace Clippy, 71 normal Rust
tests with 14 intentional local-asset tests skipped in the normal tier, example
and doc tests, 31 reference tests, corpus/golden/conformance validation, the
offline core build, and wheel smoke. All 14 local-asset Rust tests passed in a
separate run, including both exact tokenizer profiles and six composed-processor
regressions. Fresh B5/B6 native and Linux-emulated reports remained
semantically identical across architectures after host metadata was removed.

No downstream Phase B scope or dependency changed, so the Lord did not edit the
user-owned `ROADMAP.md`. The pre-existing user changes to `DESIGN.md`, root
`README.md`, and untracked `ROADMAP.md` were preserved and excluded from every
Phase B commit.

### 2026-08-01 — epic created

Created the Phase B epic and seven child tickets from the dependency-ordered
roadmap. The graph preserves the frozen Phase A prerequisites and keeps the
epic itself as a completion roll-up rather than executable implementation work.

### 2026-08-01 — execution model selected

The King asked the Lord to orchestrate Phase B with collaboration subagents,
not Kingdom peasants. Work will follow the dependency graph: each child starts
only after its prerequisites close, independent ready tickets may run in
parallel, and the Lord will personally review code, acceptance criteria, and
test evidence before updating the worklog, closing, and committing each ticket.
