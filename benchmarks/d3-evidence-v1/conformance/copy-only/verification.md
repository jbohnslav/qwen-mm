# D3 copy-only Phase C conformance verification

- Result: PASS
- Base revision: `f92a3ab8860445af1f6abeb31f996024aef1b64f`
- Temporary clean candidate revision: `34ae4b0df63bebd0a1ffe0039fdd44b4563debec`
- Requested/applied diff SHA-256: `fac134ba9b52a2d8a33fff29b97e1a8ad5b7431c4dcc9cf8d520de9b79d0b124`
- Candidate source fingerprint: `f075d27b1b80835159e991f498d34d76bc9a0a990aa2d3b0c7389d1d1a1b4625`
- Git gate inputs: clean (`gate_input_status=[]`)
- Scope: `text_image`; profiles `qwen3-vl-8b`, `qwen3.5-9b`
- Cases: 290 declared, 290 executed, 290 passed, 0 failed, 0 skipped
- Built wheel SHA-256: `8d336b7778567b4a0af9168d63d76e223b048b494c0460114f8fc7be96796b18`
- Native extension SHA-256: `aae345294c7850688a434cf36eef47f393bc3036851ebc57ea0077cd609a3499`
- Installed package artifact SHA-256: `f947311a8f7b462cf26d26395f1d09445955fb5fc5d2897648754b966bfc4a16`
- Platform: Darwin arm64, Python 3.11.15, NumPy 2.4.6
- Artifact set: 1,039 files, 1,053,186,892 bytes
- Artifact manifest SHA-256: `334c64118980ae75f114da5acc61b647e2fa4165eaf291b891b15758382ef3a4`
- Linked asset manifest before/after SHA-256: `deca9e8f4bea1bf5de1ddfc4422a6f0b4fc4712bba74a3f350533681ccc01dd4` (identical)

## Evidence hashes

- `report.json`: `dc6d1d80a4f395e604f9ba956d813468487dd065c07fe7572405f2b9d0c111e4`
- `summary.md`: `83b227056b6b6edcb436345db4c5e77e09d100bac84c107dcde1ec106e74b951`
- `full-phase-c.log`: `096f9dce3eeb455e4a7d4c4435b4520b36678f66de637277cd8c74442ec34d86`
- `post-validation.log`: `ccbed5993ff9284a926982adf6ebfc07a29adc3fb658c5c9bd393b23d0d0c0f3`
- `candidate.patch`: `fac134ba9b52a2d8a33fff29b97e1a8ad5b7431c4dcc9cf8d520de9b79d0b124`
- `assets-before.sha256`: `deca9e8f4bea1bf5de1ddfc4422a6f0b4fc4712bba74a3f350533681ccc01dd4`
- `assets-after.sha256`: `deca9e8f4bea1bf5de1ddfc4422a6f0b4fc4712bba74a3f350533681ccc01dd4`

## Commands

The exact conformance invocation and isolated cache/output paths are recorded in
`run-command.txt`. It invokes `make phase-c-conformance` with the report,
summary, and artifact outputs outside the clone. The harness itself reran the
authoritative semantic validator. A second post-run invocation was also run:

```text
./scripts/with-cargo.sh uv run --locked --package qwen-mm-reference \
  python -m qwen_mm_reference.phase_c_conformance validate \
  --assets-root reference/.cache/huggingface \
  --report /private/tmp/qwen-mm-d3-copy-only-conformance.owRVUz/evidence/report.json
```

The first fresh-cache attempt was unable to resolve PyPI inside the sandbox;
that log is retained as `full-phase-c-initial-sandbox-failure.log`. The identical
rerun with network access completed and is the `full-phase-c.log` named above.
The candidate clone remained clean after the run, and the linked asset inventory
was unchanged.
