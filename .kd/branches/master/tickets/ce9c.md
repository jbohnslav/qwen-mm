---
id: "ce9c"
status: closed
deps: []
links: []
created: 2026-09-23T13:06:53Z
type: task
priority: 2
closed_at: 2026-09-23T14:50:06Z
resolution: completed
closed_context: codex:09044ae762a8387c
assignee: codex:09044ae762a8387c
---
# Upgrade to tokenizers v1 and rerun representative benchmarks

Upgrade the Rust tokenizers dependency from pinned 0.22.2 to the new v1 release candidate (or stable v1 if available at implementation time), verify tokenization parity, and measure the impact with our existing benchmark harness.

## Acceptance Criteria

- [x] Pin the selected v1 version in Cargo.toml, update Cargo.lock, and adapt affected APIs/features while preserving required tokenizer loading and encoding behavior.
- [x] Existing relevant build and conformance checks pass, including exact token-ID parity for representative Qwen text/chat and multimodal prompts.
- [x] Capture a baseline with tokenizers 0.22.2 and rerun a representative subset of the existing benchmark-v2/performance harness with v1 on matching hardware, inputs, thread counts, and build settings.
- [x] Cover text tokenization and end-to-end multimodal processing, including single requests and batches; report latency/throughput and relevant memory changes without requiring a full recertification sweep.
- [x] Save reproducible commands, dependency versions, hardware/provenance, and before/after results; summarize measured gains, regressions, and any remaining compatibility limitations.

## Context and approach

The implementation pins `tokenizers = "=1.0.0-rc.2"` and uses the read-only pipeline API. Legacy hash-validated model assets are canonicalized in memory. Because v1 no longer emits offsets, visual spans are located using byte lengths from the pinned byte-level vocabularies and NFC-normalized span boundaries. A decomposed-Unicode image/video regression test covers this change.

rc.2's JSON reader always constructs regex splits with no backend, breaking Qwen3.5's unrecognized Unicode-mark pattern. `vendor/tk-serialize` contains the published 0.1.0-rc.2 crate with a narrow fallback initialization patch; the reader's `fancy-regex` feature is explicitly enabled. See its PATCH.md. Qwen3-VL retains its recognized native grammar.

## Results and validation

The matched release-wheel comparison is complete on the local Apple M4 (24 GiB),
with both profiles and `text_short`, `text_long`, `image1`, and `jpeg24_requests`.
Settings: t1, two fresh-process repetitions, two warmups, at least ten samples
and 0.1 seconds per timing loop. Python's official tokenizer stays at 0.22.2.

- Qwen3-VL long text: 2.298 ms → 0.270 ms (8.50x).
- Qwen3.5 long text: 2.334 ms → 0.471 ms (4.96x).
- Short text: 1.12x / 1.06x; image cases within approximately 1.3% of baseline.
- All 16 matched process/case pairs have identical before/after output signatures.
  Both builds passed their paired oracle checks and report validation against
  their exact measured wheels.
- Process peak RSS in text cases increased by approximately 21–32 MiB for
  Qwen3-VL and 72–83 MiB for Qwen3.5. Image-case RSS varied; the Qwen3.5 batch
  increased approximately 323 MiB. These include oracle allocations and allocator
  retention, not just the tokenizer. This upgrade does not certify a memory budget.

Evidence, raw timings, memory census, wheel hashes, commands, and reproduction
scripts: `benchmarks/tokenizers-v1/README.md` and `comparison.md`.

`make rust-check`, all 15 local-asset text tests, and lint pass. All installed Python binding suites pass (construction, docs, media sources,
usability, Rosetta regressions, batching, thread budgets, GIL release, and errors). The source distribution explicitly includes the
vendored reader; its extracted locked dependency graph resolves offline.

The old Phase B fixture expects pre-resize-v2 pixels: all ten case reports are
identical on the baseline and upgraded code, including these existing pixel
failures. The Phase C resize-only overlay rejects tokenizer source changes as
outside its scope. Neither gate was weakened. Results remain local diagnostics,
not full release certification, x86 results, or multicore claims.

Hugging Face announced the v1 Rust release candidate on September 21, 2026:
https://huggingface.co/blog/tokenizers-v1

The upstream report claims 3–30x faster single-thread encoding versus v0.23 on an Apple M4 Max, with workload-dependent gains. Those figures exclude Python binding overhead and are not a baseline for this repository's pinned 0.22.2 or end-to-end multimodal pipeline. Measure our actual workload before changing performance claims.

Reuse the existing benchmark tooling (`reference/src/qwen_mm_reference/benchmark_v2.py`, `crates/qwen-mm-python/python/qwen_mm/benchmark.py`, and Makefile benchmark/D4 targets). Keep old evidence intact and store the new comparison separately. Select a focused set of representative cases and document the selection; expand only if parity failures or regressions need investigation.

## Worklog

- 2026-09-23: Created at the user's request after identifying the upstream v1 announcement. Confirmed the current Rust dependency pin is 0.22.2. Upgrade and benchmark execution have not started.

- 2026-09-23: Pulled onto master and started. Found missing offsets and Qwen3.5 regex-loader bug in rc.2; adapted spans and patched the reader. Rust checks and 15 text tests passed. Matched wheel comparison started; full certification gates retain the scope limitations described above.

- 2026-09-23: Completed matched wheel measurements and exact cross-build output comparison. Confirmed old Phase B reports identical before/after. Added source-distribution include after packaging inspection exposed the omitted patched reader; verified the extracted archive resolves offline.

- 2026-09-23: Completed installed-wheel binding suite, lint, paired artifact validation, exact output-signature comparison, and source-distribution inspection. Closed with local benchmark results and documented prerelease patch, RSS tradeoff, and certification limitations.

## Lifecycle

- 2026-09-23T14:50:06Z [codex:09044ae762a8387c] — closed (completed)
