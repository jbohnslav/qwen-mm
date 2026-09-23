# tokenizers v1 comparison

Long-text end-to-end preprocessing improved **8.50x for Qwen3-VL** and
**4.96x for Qwen3.5**. Short text improved 1.12x and 1.06x respectively.
Image and 24-request image-batch timings stayed within about 1.3% of baseline;
this small difference is not evidence of a meaningful speedup or regression.
All 16 matched process/case pairs produced identical output signatures.

Memory is a tradeoff: text-case process peak RSS rose by roughly 21–32 MiB for
Qwen3-VL and 72–83 MiB for Qwen3.5 in this census. Image-case RSS was variable,
including a roughly 323 MiB increase for the Qwen3.5 batch. These process-wide
measurements include both the oracle and candidate and allocator retention;
they do not isolate the tokenizer or certify a memory budget.

This is a local diagnostic comparison of qwen-mm's Rust tokenizers 0.22.2
against 1.0.0-rc.2 with the Qwen3.5 reader patch documented in
`../../vendor/tk-serialize/PATCH.md`. It is not a new release certification.

Results are generated in [comparison.md](comparison.md) and the underlying
[comparison.json](comparison.json). Run `summarize.py` after reproducing both runs
to require exact before/after output signatures and regenerate the tables.

## Method

Both wheels use release builds on the same Apple M4 (24 GiB RAM). The baseline
is built from the commit in `builds.json`; the candidate uses this change.
`provenance.json` records wheel hashes and the exact benchmark commands.
The Python reference environment, including tokenizers 0.22.2, stays unchanged.
The environment's Python package version is not the embedded Rust crate version.

The existing paired benchmark harness measures structured messages through
returned NumPy arrays. Cases cover short/long text, one image, and a batch of
24 single-image requests for Qwen3-VL-8B and Qwen3.5-9B. Each uses one thread,
two independent process repetitions, two warmups, at least ten samples and
0.1 seconds per timing loop. Each workload receives pre/post oracle comparisons
and measured-output stability checks. Resource census runs outside timed samples.

The two builds run sequentially (baseline first), so host drift is still possible.
Two process repetitions are a focused regression check, not a confidence study.
The results do not establish x86 performance, multicore scaling, cold loading
latency, or full release eligibility. RSS is process-wide, includes the Python
oracle and allocator retention, and is not tokenizer-only memory.

## Reproduction

Build the old checkout and this checkout with the same release settings:

```sh
# In a detached checkout of the baseline commit from builds.json:
./scripts/with-cargo.sh /path/to/reference-env/bin/maturin build --release --locked \
  --interpreter /path/to/reference-env/bin/python --out /tmp/tokenizers-before

# In the upgraded checkout:
./scripts/with-cargo.sh .venv/bin/maturin build --release --locked \
  --out /tmp/tokenizers-after
.venv/bin/python benchmarks/tokenizers-v1/reproduce.py \
  --baseline-wheel /tmp/tokenizers-before/qwen_mm-0.1.0-cp311-abi3-macosx_11_0_arm64.whl \
  --candidate-wheel /tmp/tokenizers-after/qwen_mm-0.1.0-cp311-abi3-macosx_11_0_arm64.whl \
  --output /tmp/tokenizers-comparison
```

The script unpacks each wheel into its own temporary import directory. The same
reference Python executable and dependencies run both comparisons; existing
installations are not changed. The pinned model assets must already exist under
`reference/.cache/huggingface`.

## Validation boundaries

`validation/rust-check.log` records formatting, Clippy, workspace tests,
doctests, and an offline core build. `validation/text-tests.log` includes all
15 text tests, including the local-asset tests and a new regression for NFC
composition, emoji, image spans, and video spans. Existing tests check exact
committed token IDs, padding, masks, special tokens, and chat templates.

The historical Phase B fixture still expects the pre-resize-v2 pixel values.
Its report is retained separately; all ten case reports are identical to the same run on the baseline checkout,
including pixel failures, so these are not a tokenizer regression. The Phase C resize overlay deliberately rejects changes outside its
resize-only source allowlist (`validation/resize-overlay-scope.log`). These
legacy gates are not relaxed by this change. Benchmark reports remain diagnostic
and do not inherit release certification from the old wheel.

## Upstream integration

The [upstream announcement](https://huggingface.co/blog/tokenizers-v1) describes
large workload-dependent improvements. This integration uses the actual rc.2
pipeline API, which differs from the earlier `Tokenizer` API. It canonicalizes
the pinned tokenizer JSON in memory and derives visual span boundaries from
byte-level vocabulary lengths, accounting for NFC composition. It never edits
the model snapshots. Qwen3.5 currently uses the regex fallback; Qwen3-VL uses
the recognized native splitting grammar.

The source-distribution check found and fixed omission of the vendored reader:
`pyproject.toml` explicitly includes it. The extracted archive contains the patch
and resolves its locked Cargo dependency graph offline. To validate saved benchmark
results later, use `reproduce.py --validate-only` with the same two wheels; the
validator binds each report to its own native binary, not whichever wheel is
currently installed in the development environment.

All installed-wheel Python suites passed: pretrained construction, documentation
examples, media sources, usability, Rosetta regressions, and binding tests
(including thread budgets, 24-request ordering, output lifetimes, and GIL release).
See `validation/python-binding.log`.
