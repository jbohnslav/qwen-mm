# Still-image resize compatibility and performance amendment v2

- Status: accepted for D3.5 implementation
- Date: 2026-08-13
- Contract ID: `qwen-mm-still-image-resize-v2`
- Amends: [`compatibility-v1.md`](compatibility-v1.md),
  [`resize-parity-v1.md`](resize-parity-v1.md),
  [`media-parity-v1.md`](media-parity-v1.md),
  [`phase-b-image-parity-v1.md`](phase-b-image-parity-v1.md),
  [`phase-c-conformance-v1.md`](phase-c-conformance-v1.md), and
  [`performance-certification-v1.md`](performance-certification-v1.md)

## Decision and scope

The production still-image path may use an optimized Rust RGB8 resizer whose
pixel values are not identical to Pillow 12.3.0. The numerical rules below
replace the v1 one-byte Pillow comparison only at the boundary between the
official Pillow-resized RGB8 occurrence and the candidate-resized RGB8
occurrence, and for final still-image tensors whose only difference is caused
by those accepted RGB8 values.

All other v1 semantics remain unchanged. In particular, this amendment does
not relax:

- resize planning, destination dimensions, grids, patch counts, occurrence
  order, keys, dtypes, shapes, strides, offsets, or replacement ranges;
- no-op behavior, decode and color conversion before resize, normalization,
  temporal duplication, layout transformation, and patchification after
  resize;
- input validation, resource limits, checked arithmetic, atomic destination
  behavior, stable error categories, or deterministic failure order;
- any raw-frame or encoded-video rule, including the TorchVision-compatible
  video resizer and its v1 tolerances; or
- exact repeatability for one candidate build, input, processor configuration,
  and thread budget.

Downstream still-image operations are compared exactly as before after both
implementations are given the same candidate-resized RGB8 buffer. An
end-to-end official-versus-candidate tensor comparison may differ only at
locations and by magnitudes implied by a passing RGB8 comparison; it is not an
independent opportunity to add error.

## Frozen RGB8 quality gates

Compare each resized media occurrence independently, and compare each R, G,
and B channel independently. Let `R` be Pillow's RGB8 output, `C` the candidate
RGB8 output, `e = C - R` evaluated in signed real arithmetic, and `N` the
number of pixels in one occurrence/channel. Every occurrence/channel must pass
all of these gates:

| Metric | Required value |
| --- | ---: |
| maximum absolute error, `max(abs(e))` | `<= 32` |
| root mean squared error, `sqrt(sum(e^2) / N)` | `<= 5` |
| 99th percentile absolute error | `<= 16` |
| absolute signed mean bias, `abs(sum(e) / N)` | `<= 2` |
| canonical windowed SSIM | `>= 0.98` |

The absolute-error percentile uses nearest-rank order statistics: sort the `N`
absolute errors and select the value at one-based rank `ceil(0.99 * N)`. No
aggregation across channels, media occurrences, requests, cases, or profiles
may hide a failure.

Reports must also include MAE, PSNR, absolute-error p50 and p90, and the gated
metrics above for every occurrence/channel. MAE, PSNR, p50, and p90 are
diagnostic, not release gates. PSNR uses peak value 255 and is positive
infinity when MSE is zero.

### Canonical windowed SSIM

SSIM is computed separately for each occurrence/channel in `float64`:

1. Extend each image by five pixels on all sides using reflection without
   repeating the edge pixel (reflect-101). The supported resized-image envelope
   is large enough for this padding; a result too small to define it is a
   conformance failure rather than a request to change the statistic.
2. Form an `11 x 11` Gaussian kernel with `sigma = 1.5`, centered at `(5, 5)`,
   `w[i,j] = exp(-((i-5)^2 + (j-5)^2) / (2 * 1.5^2))`, normalized in
   `float64` so its 121 weights sum to one.
3. At every original-image pixel, use that kernel to compute population local
   means `mu_R`, `mu_C`, population variances
   `sigma_R^2 = sum(w * (R - mu_R)^2)` and
   `sigma_C^2 = sum(w * (C - mu_C)^2)`, and population covariance
   `sigma_RC = sum(w * (R - mu_R) * (C - mu_C))`.
4. Compute the local value
   `((2*mu_R*mu_C + C1) * (2*sigma_RC + C2)) /
   ((mu_R^2 + mu_C^2 + C1) * (sigma_R^2 + sigma_C^2 + C2))`, where
   `C1 = (0.01 * 255)^2` and `C2 = (0.03 * 255)^2`.
5. The canonical windowed SSIM is the arithmetic mean of all local values over
   the original image extent.

Implementations and reports must use this definition, not a library default
whose padding, covariance normalization, channel aggregation, or constants may
differ.

## Exact invariants

The following remain byte-exact even though general resized pixels use the
quality envelope:

- A no-op plan returns the source RGB bytes exactly, including correct handling
  of a supported positive row stride.
- A spatially constant input remains exactly the same per-channel constant at
  every output pixel. This includes black, white, and the solid RGB primaries
  red `(255,0,0)`, green `(0,255,0)`, and blue `(0,0,255)`.
- Channels are neither mixed nor reordered: identical source channels produce
  identical output channels, and permuting source channels produces the same
  permutation of independently resized output channels.
- Repeated execution with the same candidate build, input, plan, options, and
  thread budget is byte-identical. Thread-count sweeps must also produce
  byte-identical outputs unless a future explicitly versioned contract says
  otherwise.

Geometry and all non-pixel observables remain exact against the official path
for both no-op and resizing cases.

## Corpus and selection-bias control

The thresholds in this document are broad neural-preprocessing guardrails, not
an attempt to visually or bitwise clone Pillow. They are fixed before any
`pic-scale` result is inspected and must not be widened to select a faster
candidate.

The existing 17-case resize-v1 corpus remains a regression and adversarial
suite, but it is not an unbiased selection set: results from
`fast_image_resize` were already observed when this amendment was adopted.
Before comparing additional candidates, D3.5 must therefore commit and
authenticate a new fixed-seed holdout corpus and its Pillow outputs. The
holdout must use seed `20260813`, include representative natural-image crops
and scales, and include independent ramps, impulses, edges, checkerboards,
noise, chroma-separated patterns, upsampling, downsampling, one-axis resizing,
and factor/min/max boundary plans. Candidate selection requires every gated
occurrence/channel in both the holdout and existing adversarial coverage to
pass. The holdout inputs and expected outputs may not be regenerated after a
candidate is evaluated.

## Performance release gates

All v1 capture, provenance, warm-state, equal-thread-budget, no-pruning,
memory, allocation, parallel-efficiency, and controlled-host requirements stay
in force. The shipping wheel must satisfy both of these image requirements:

1. For every applicable shipping image coordinate in the D4 release matrix,
   the lower bound of the deterministic paired process-median bootstrap 95%
   confidence interval for official/candidate speedup is strictly greater than
   `1.0`. A coordinate is the full combination of profile, controlled
   architecture, release image workload, and required thread budget.
   Unsupported `repeat24_cached` remains excluded rather than being relabeled.
2. The existing headline batch gate remains: the lower 95% confidence bound is
   at least `2.0x` for both `image24` and `ragged24`, both profiles, each
   controlled architecture, and the production thread regime.

The existing text-only regression guard also remains: candidate/reference p50
is no greater than `1.05` for `text_short` and `text_long` at every required
profile, architecture, and thread-budget coordinate.

An authentic miss remains publishable diagnostic evidence but is not
releasable. Neither a quality threshold nor a performance threshold may be
changed after inspecting a candidate or controlled capture; a miss is handled
by reporting and escalation under the v1 procedure.

## Versioning consequences

Authenticated v1 manifests, binary fixtures, reports, schemas, and their
recorded meanings are immutable historical evidence. D3.5 must create new
versioned conformance evidence or an explicit versioned overlay that names
`qwen-mm-still-image-resize-v2`; it must not rewrite a v1 artifact and present
it as though the original contract had permitted these differences.
