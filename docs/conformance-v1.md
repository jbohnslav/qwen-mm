# Conformance corpus and comparator v1

This is the executable evidence gate for compatibility contract v1. It decides
whether a candidate processor matches the pinned composed Python oracle and
reports the first processing stage at which they diverge.

The candidate emits the same staged manifest shape as the golden exporter. It
does not need to reproduce oracle provenance or Python exception text, but it
must provide the logical input/media occurrence records, rendered and expanded
prompts, prepared media, video metadata, replacement ranges, stable error
category, and official output descriptors needed by the comparison policy.

## Three tiers

1. `reference/goldens/v1/` contains the small committed end-to-end cases. These
   cover both templates, text, image, raw-frame video, conditional output keys,
   one expected decode error, and the authenticated signature of the large
   `image24` output.
2. `reference/conformance/v1/rules.json` contains compact executable fixtures
   for factor-32 ties-to-even rounding, minimum/maximum pixel transitions,
   aspect ratio 200, paired explicit dimensions, frame-count rounding, FPS
   sampling, timestamps, placeholder arithmetic, and checked-u64 boundaries.
3. `reference/conformance/v1/corpus.json` is the coverage index for the full
   matrix. Its live recipes cover both profiles; template roles/tools/options;
   Unicode, padding, and literal visual tokens; raw strided RGB; JPEG, PNG, and
   WebP; grayscale, alpha, and CMYK conversion; EXIF orientations; malformed
   and truncated inputs; mixed/repeated/interleaved visuals; batch permutations;
   stable failures; and every frozen resource limit.

`python -m qwen_mm_reference.corpus validate` executes tier 2 and rejects tier
3 if a required coverage tag, profile, source case, manifest, or rule silently
disappears.

## Comparator

Compare one candidate result with one golden:

```bash
uv run --locked --no-sync --package qwen-mm-reference \
  python -m qwen_mm_reference.conformance compare \
  --expected reference/goldens/v1/qwen3-vl-8b/multimodal-smoke/manifest.json \
  --actual /path/to/candidate/manifest.json
```

The comparator enforces:

- exact case/profile/status and stable error category;
- exact output key presence and order, shapes, dtypes, strides, byte order, and
  contiguity;
- exact prompt UTF-8, integer arrays, occurrence order, frame indices,
  replacement ranges, and lossless prepared RGB;
- the fixed A1 tolerance selected for prepared lossy images, video resize,
  normalize/patchify, and final image/video pixels; and
- batch/text shape, conditional modality, grid-row/occurrence,
  grid-product/patch-row, placeholder/occurrence, and placeholder/merged-patch
  invariants.

A numeric failure includes maximum absolute and ULP error, RMSE, p50/p90/p99
absolute error, count beyond the fixed bound, and the first differing array
index. Final pixel failures also decode that index into request/media occurrence,
patch, temporal offset, channel, and patch x/y coordinates.

An authenticated `signature_only` array can prove byte identity. If its hash
differs, tolerant diagnostics require both sides to be re-emitted as inline
values or `.npy`; the report says so instead of collapsing the failure into an
unexplained hash mismatch.

## Pull-request and full gates

The fast pull-request gate is:

```bash
make conformance-smoke
```

It executes all compact rules and validates every committed manifest without
loading either model processor. It is included in `make reference-smoke` and
therefore in `make check`.

The nightly/reference-change entry point is:

```bash
make conformance-full \
  CONFORMANCE_SEED=1364677966 \
  CONFORMANCE_CASES=16 \
  CONFORMANCE_OUTPUT=reference/results/conformance-local \
  CANDIDATE_COMMAND='qwen-mm-candidate --case {case} --profile {profile} --output {actual}'
```

The command deterministically generates the same ordered cases, exports a fresh
staged oracle with locally cached pinned artifacts, runs every case on both
profiles, and writes per-case expected/actual manifests plus `report.json` and a
matrix `summary.json`. The command template supports `{case}`, `{profile}`, and
`{actual}`. `{expected}` is also available for comparator-adapter development,
but a real candidate must not read it.

## Reproduce, minimize, and promote

Generate a replayable case set:

```bash
python -m qwen_mm_reference.corpus generate \
  --seed 1364677966 --count 16 --output-directory /tmp/qwen-mm-live-cases
```

Run one live differential case:

```bash
python -m qwen_mm_reference.corpus differential \
  --case /tmp/qwen-mm-live-cases/seed-5157454e-0000.json \
  --profile qwen3-vl-8b \
  --candidate-command 'qwen-mm-candidate --case {case} --profile {profile} --output {actual}' \
  --output-directory /tmp/qwen-mm-failure
```

Minimize any failure with a command whose nonzero exit means the failure still
reproduces. The reducer deterministically removes requests, messages, and
content items while preserving that predicate:

```bash
python -m qwen_mm_reference.corpus minimize \
  --case /tmp/qwen-mm-live-cases/seed-5157454e-0000.json \
  --predicate-command 'qwen-mm-repro --case {case}' \
  --output /tmp/minimized.json
```

Promote the minimized input to a committed regression case. Promotion refuses
to overwrite an existing ID:

```bash
python -m qwen_mm_reference.corpus promote \
  --case /tmp/minimized.json --id regression-description
```
