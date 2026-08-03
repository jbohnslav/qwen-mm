# D3 controlled Modal selection summary

## Result

Select direct-stride resize and decoder-owned no-resize transfer. Reject the
normalization lookup from the production writer.

The integrity-validated artifact is `modal-selection.zip`, 10,857,914 bytes,
SHA-256
`addffbb99886b84057421b6af2bf09034ed6abba9f2d27cabe4e9f08e59daaa0`.
It contains all 24 raw coordinates, exact patches, four release ELF modules,
four build logs, the measurement log, full provenance, summary, and a manifest
covering every retained member.

## Controlled worker

- Linux x86-64 gVisor; masked AMD family 175 model 17.
- Modal request: 16 physical CPU cores, 32 GiB memory, nonpreemptible,
  single-use container.
- Observed capacity: 32 logical/affinity/lscpu CPUs and 404,603,887,616 bytes
  from `/proc/meminfo`; gVisor exposed no cgroup-v2 limit files.
- Every coordinate used a fresh wrapper and child process pinned by `taskset`
  to one actual CPU, with all native thread environments set to one.
- Two warmups and seven retained wall/CPU samples per coordinate.
- Function-worker elapsed time: 369.005867 seconds. Recorded elapsed-worker
  estimate: $0.310673. This excludes image construction and provider billing
  adjustments and is not an all-in bill.

The four-variant order was rotated across the six profile/case workloads, but
each workload received only one order. This is D3 selection evidence, not the
replicated D4 certification protocol.

## Timing and isolated LUT decision

All values below are milliseconds. `patch no-LUT` and `patch LUT` are the
observed exclusive `native.media.normalize_patchify_layout` stages.

| Profile / case | Baseline wall | Copy-only wall | No-LUT wall | LUT wall | Patch no-LUT | Patch LUT | LUT patch delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3-vl-8b / image24 | 1322.529 | 1366.390 | 1339.331 | 1422.842 | 99.127 | 135.482 | +36.7% |
| qwen3-vl-8b / ragged24 | 463.752 | 356.375 | 352.522 | 381.795 | 44.946 | 60.892 | +35.5% |
| qwen3-vl-8b / rgb24 | 483.314 | 478.000 | 680.928 | 507.726 | 63.284 | 87.251 | +37.9% |
| qwen3.5-9b / image24 | 996.939 | 1006.114 | 1050.755 | 1085.056 | 98.777 | 135.250 | +36.9% |
| qwen3.5-9b / ragged24 | 354.018 | 354.399 | 355.088 | 369.979 | 45.003 | 60.910 | +35.3% |
| qwen3.5-9b / rgb24 | 480.864 | 440.265 | 484.276 | 464.604 | 63.197 | 87.834 | +39.0% |

The LUT makes its targeted stage about 35-39% slower in every workload and its
whole operation regresses in four of six comparisons against no-LUT. It is
therefore rejected.

Whole-operation timing does not cleanly isolate the copy changes. For example,
qwen3-vl `rgb24` is unaffected by decoder ownership, yet its copy-only and
no-LUT observed destination-allocation stages were 160.785 and 341.059 ms,
driving wall medians of 478.000 and 680.928 ms. The deterministic copy counters,
not these confounded whole-operation differences, justify retaining the copy
changes.

## Deterministic memory and copy selection

The selected path has the following exact baseline-to-no-LUT changes in both
profiles:

| Case | Allocations removed | Bytes removed | Copies removed | Copied bytes removed | Peak transient bytes removed |
| --- | ---: | ---: | ---: | ---: | ---: |
| image24 | 24 | 56,494,152 | 24 | 56,494,152 | 2,353,923 |
| ragged24 | 29 | 35,557,053 | 29 | 35,557,053 | 2,353,923 |
| rgb24 | 24 | 36,132,888 | 24 | 36,132,888 | 2,353,923 |

Direct positive-stride reads account for 24 removed allocations/copies per
operation and 25,726,653-56,494,152 bytes. Decoder-owned no-resize transfer
removes another five allocations/copies and 9,830,400 bytes in `ragged24`.
All unaffected counter coordinates are exactly unchanged.

## Correctness and identity

All four measured variants produced bit-identical official arrays and
metadata sidecars for every profile/case coordinate. The matrix shares harness
SHA-256
`6dcc88628ad37981de52748041cc5603085aa24d5196789af584014e50b8360c`.

| Variant | Exact diff SHA-256 | Linux x86 release module SHA-256 |
| --- | --- | --- |
| baseline | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `273e8cec02c40bb27dbeea9348587d26a02f5ced99bf73da5b6702760e2fd109` |
| copy-only | `fac134ba9b52a2d8a33fff29b97e1a8ad5b7431c4dcc9cf8d520de9b79d0b124` | `a54c037f295785bc38f918be071d6fb7a7e2f820f97d1c09d8e93e5a7e9b0ab3` |
| measured no-LUT | `cb3f6496bee26a026cdf6478c48f6b34052639ea25b8d8170023f377afbc94fc` | `9ec182fbeb309afe94f7ce86b604546546d40dee6e76bfd6d9563b0f894a96b5` |
| rejected LUT | `f939bb402383c39327238d95b3663ba2021fddad12c9a160c3e1fc93839e1923` | `5cd3401cb72d6646cd0fe5c91e09c9a4b1d3426bd284cf92f25ac405f7ba7b98` |

The measured no-LUT patch retained an unreachable lookup helper and its test.
Clean final source deletes only that dormant code. `selected-cleanup.patch`
records this delta; `selected.patch` is the complete final diff from the base,
SHA-256
`14c2a6888efad7fdc1a8fd98c6a34bd7fc5597ebb56914832eed58bf50533bc8`.
The production `execute_image_patchify_plan_into` extract is byte-identical in
measured no-LUT and selected source, SHA-256
`74794d63d401d9e6deaeef0c3cea7a6ca80d6fbedcdda4ec86e9efd6abb92a84`.
Separate local release builds produced module SHA-256 `6cc7562a...` for
measured no-LUT and `378d4b5e...` for clean selected source. They do not
byte-match, so no binary-equivalence claim is made; the exact source-level
production-path comparison and selected source's own conformance/build
identities are retained instead.

## Full two-profile conformance

Each accepted stage was committed in an isolated clean clone and passed the
authoritative installed-wheel Phase C gate:

| Source | Cases | Skips | Report SHA-256 | Installed native SHA-256 |
| --- | ---: | ---: | --- | --- |
| exact copy-only | 290/290 | 0 | `dc6d1d80a4f395e604f9ba956d813468487dd065c07fe7572405f2b9d0c111e4` | `aae345294c7850688a434cf36eef47f393bc3036851ebc57ea0077cd609a3499` |
| exact measured no-LUT | 290/290 | 0 | `a67b42c70bae4c0a2237cdfb3b1495597ef6fd05ab30a2da45e03b25aa31cbae` | `acae2fff3c903d3edecb67b3c6951bef75fd4cb1dff58b9c25c006c6f3065c71` |
| clean selected source | 290/290 | 0 | `0b4b1e00815e7b8b7bc7548bc3c6b896c8a8bbeb1a3a38627e77e3dabc04f7fe` | `94422cf4e8792773777ae615089ae6593f5a2113dc97ab2622297770d9e0016a` |

Reports, readable summaries, full logs, validation records, and retained-file
hash manifests are under `conformance/`.
