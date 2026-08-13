# Phase C installed-wheel text/image conformance v1

> Historical gate notice: authenticated Phase C v1 reports preserve the exact
> contract recorded here. D3.5 and later conformance runs apply
> [`qwen-mm-still-image-resize-v2`](image-resize-contract-v2.md) only to the
> still-image RGB8 values and their consequent final tensors; structure,
> geometry, errors, and all other v1 checks remain exact.

This gate certifies correctness of the production Python wheel for the frozen
text/image scope and both pinned profiles. It is the dependency required before
image measurement work can be considered for release.

The driver and candidate are deliberately separate. The driver runs from the
locked reference environment and may invoke the pinned official oracle. The
candidate runs with Python isolated mode in a temporary environment containing
the production `qwen-mm` wheel and its exact NumPy dependency. It receives only
the case, media, profile alias, and authenticated local assets. Audit hooks fail
candidate access to expected artifacts, and the result records that no Pillow,
Torch, TorchVision, Transformers, or Qwen VL Utils module was imported.

## Coverage

The 290-case canonical inventory is derived from the committed manifests and
gate definitions, not from a report-provided count. It includes:

- all five Phase B cases for both profiles;
- 22 applicable exact chat/template cases;
- all 54 committed decode/color/codec cases for both profiles;
- text, corrupt-image, and `image24` golden cases for both profiles;
- contiguous and positive row-padded RGB success plus two invalid layouts;
- every image-applicable resource boundary one below, at, and one above;
- heterogeneous text/image batches in identity and two deterministic orders;
- ten malformed public-request cases and public checked-arithmetic overflow for
  both profiles; and
- twelve installed geometry cases per profile covering min/max budgets, exact
  budgets, explicit dimensions, aspect-ratio limits, and observable
  ties-to-even rounding.

The compact rule document is authenticated separately. Helper-only rules are
not mislabeled as candidate executions; each installed-boundary rule maps to
the exact geometry or arithmetic candidate case that exercises it. Video-bearing
cases, temporal rules, and the raw-frame resource axis are named as Phase E
exclusions. Candidate skips are forbidden.

Every success checks official key order, dtype, byte order, shape, C strides,
contiguity, exact integer arrays, prompts, padding, modality IDs, grids,
replacement ranges, occurrences, sidecar and layout ranges, prepared RGB, and
final pixels. Final failures retain maximum absolute and ULP error, RMSE,
p50/p90/p99 absolute error, count over the immutable bound, and the first
request/media/patch/temporal/channel/pixel coordinate beyond that bound.

## Run and validate

With both pinned snapshots under `reference/.cache/huggingface`:

```bash
make phase-c-conformance
```

This builds a production wheel, installs it into a sanitized temporary
environment, writes `reference/phase-c/v1/report.json` and
`reference/phase-c/v1/summary.md`, and validates their currentness. Override
`PHASE_C_ASSETS_ROOT`, `PHASE_C_REPORT_OUTPUT`, `PHASE_C_SUMMARY_OUTPUT`, or
`PHASE_C_ARTIFACT_OUTPUT` to use other locations.

Validate already-committed evidence without running the matrix:

```bash
make phase-c-conformance-validate
```

Validation recomputes the exact case inventory, every repository input digest,
the source fingerprint, and every pinned profile artifact digest. It rejects a
dropped case, any skip, stale input, candidate identity drift, a failed result,
an undeclared exclusion, or correctness evidence containing measurement claim
fields.

For a clean reproduction, check out the source revision recorded in the report,
provide the authenticated external profile cache, run `make
phase-c-conformance`, and compare the report's source fingerprint and candidate
runtime identity. The report itself is excluded from that source fingerprint,
so committing evidence does not create a circular hash.
