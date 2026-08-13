# Phase C still-image resize conformance overlay v2

The immutable Phase C v1 report remains the exact structural and non-resize
evidence base. The v2 overlay is current only when all of these bindings hold:

- the authenticated v1 report contains all 290 declared, executed, passing
  cases with no skips;
- the current revision descends from the v1 tested revision, and every changed
  package-tree file is in the explicit resize-backend allowlist;
- the current source and `Cargo.lock` select pic-scale 0.7.11 Bicubic with
  `PreferQuality` and single-thread execution;
- the fresh wheel's native-module identity matches the supplied wheel and its
  CycloneDX SBOM contains that exact pic-scale dependency; and
- production RGB8 passes all 45 frozen resize-v2 holdout cases. The 43 cases
  whose destinations are legal 32-pixel Qwen patch geometry are additionally
  rerun through the isolated installed wheel and must byte-match the direct
  production-core blob. The remaining two one-axis holdout destinations are
  deliberately low-level resizer cases; `Processor` would round their geometry.

Validation recomputes file hashes, git ancestry and production deltas, holdout
authentication, all numerical and exact-invariant gates, installed-wheel
isolation, runtime identity, and the wheel SBOM binding. Historical v1 files
are not rewritten or reinterpreted.

Generate the overlay with:

```text
python -m qwen_mm_reference.phase_c_overlay_v2 capture \
  --candidate-python /path/to/fresh-wheel-venv/bin/python \
  --wheel /path/to/fresh-wheel.whl \
  --production-blob /path/to/selected-production.rgb8.bin
```
