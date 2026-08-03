# Phase C text/image conformance v1

Status: **pass**

Candidate cases: 290; skipped: 0.

Both pinned profiles passed the installed production-wheel text/image gate.

## Evidence counts

- `committed-golden`: 6
- `committed-phase-b`: 10
- `installed-geometry-rule`: 24
- `live-arithmetic`: 2
- `live-chat`: 22
- `live-heterogeneous-permutation`: 6
- `live-media`: 108
- `live-raw-layout`: 8
- `live-resource-boundary`: 84
- `live-schema-error`: 20

## Phase E exclusions

- `golden:qwen3-vl-8b/multimodal-smoke:raw-video-component`
- `chat:qwen3-vl-8b-visual-padding:mixed-video-case`
- `chat:qwen3.5-9b-visual-padding:mixed-video-case`
- `live:visual-occurrence-order:image-video-interleave-leg`
- `live:resource-and-overflow-matrix:raw_frames`
- `rule:nframes-half-up-to-even`
- `rule:nframes-half-down-to-even`
- `rule:fps-default-min`
- `rule:fps-clamped-max`
- `rule:fps-and-nframes-exclusive`
- `rule:nframes-below-factor`
- `rule:sample-indices-five`
- `rule:video-placeholder-count`

The candidate process used isolated Python mode, imported the installed wheel, and loaded no forbidden oracle module.
All source inputs, local profile assets, the wheel package, and the native extension are authenticated in `report.json`.
