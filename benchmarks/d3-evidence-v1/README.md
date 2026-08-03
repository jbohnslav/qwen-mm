# Phase D3 isolated optimization evidence

This directory preserves the authenticated same-host evidence used to evaluate
three D3 experiments. It is optimization evidence only. It is not D4 release
certification and makes no D4 throughput, scaling, or release-performance
claim.

## D1 motivation

The dual-host D1 report ranked `native.media.resize` first at 53.3% of
exclusive ARM time and 55.2% of x86 time. It measured
`resize.packed_source` as the largest removable copy: 1,448,491,392 bytes
and 876 calls per host. It also measured `resize.noop.source_copy` at
325,582,848 bytes and `native.media.normalize_patchify_layout` at 19.0% ARM
and 21.6% x86. The D1 sampling tool, `py-spy==0.4.1`, is already pinned in
the repository's development dependencies and lockfile.

Those ranks selected the experiments below. D3 does not reinterpret or replace
D1's dual-host profile.

## Final selection

The controlled Modal run selected direct-stride resize and decoder-owned
no-resize transfer, and rejected the normalization lookup. The lookup made its
targeted normalize/patchify stage 35-39% slower across all six workloads and
regressed whole-operation time in four of six no-LUT comparisons. The complete
validated artifact, readable tables, identities, limitations, and decision are
in `modal-selection.zip` and `modal-summary.md`.

Clean final source also deletes the measured no-LUT patch's unreachable lookup
helper and its test. `selected.patch` is the exact final diff; the production
writer is byte-identical to measured no-LUT. Exact copy-only, measured no-LUT,
and clean selected source each passed the full 290-case installed-wheel Phase C
gate with both profiles and zero skips; retained evidence is under
`conformance/`.

## Exact protocol and authentication

All four variants used the same Apple M4 host, one native thread, two warmups,
seven unobserved complete-operation samples, and one observed operation for
each frozen profile crossed with `image24`, `ragged24`, and `rgb24`.
Wall and CPU values are medians. The release extension for each variant was
built with:

```bash
PYO3_PYTHON=.venv/bin/python \
  ./scripts/with-cargo.sh .venv/bin/maturin develop \
  --release --locked --skip-install
```

A representative measurement command was:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  QWEN_MM_ASSETS_ROOT=/absolute/hash-pinned/cache \
  .venv/bin/python benchmarks/d3-evidence-v1/measure.py \
  candidate qwen3-vl-8b image24 /tmp/d3-coordinate.json
```

The checked-in `measure.py` hashes the implementation files and exact binary,
its own code, `workloads-v2.json`, the imported native adapter and reference
protocol, input fingerprints, selected hash-pinned assets, Rust/Cargo/Python
toolchains, installed package versions, and all thread settings. The final
timed output, observed output, and a separate untimed metadata-authentication
call must have identical official arrays before a coordinate can be written.
The complete metadata/sidecar is canonicalized and signed too.

Every intermediate implementation diff is preserved verbatim. Each patch file
SHA-256 equals the `implementation_diff_sha256` embedded in its raw JSON.

| Variant | Implementation digest | Exact diff / patch SHA-256 | Native module SHA-256 |
| --- | --- | --- | --- |
| baseline | `6fdc3c643e2267db41fdeb06de871a8bd04f05d5442b033f885f0e40dfc2975d` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty diff at `f92a3ab8860445af1f6abeb31f996024aef1b64f`) | `6c0460c72b323d0772a3d4306f2c0e78aa3d5efd86609f5b665634d978c6a3cf` |
| copy-only | `fa34ce748d0152ba942f911d67106c7cf75d38c918b5ac8d2ba82ce09d96377a` | `fac134ba9b52a2d8a33fff29b97e1a8ad5b7431c4dcc9cf8d520de9b79d0b124` (`copy-only.patch`) | `8ce0ab3a9b32f6ea9bf4e42b6e83f9949b86d103875c1b41b0b74c3b24bab6d5` |
| no-lut | `2c5a996c4ca75c619b7c08d4eccb6c953912fced350010425c53204bf07355cf` | `cb3f6496bee26a026cdf6478c48f6b34052639ea25b8d8170023f377afbc94fc` (`no-lut.patch`) | `6cc7562a3bca5fa06a59bfefe31f5f8adab530ac8c00441133f4030254e1f601` |
| candidate | `a1b61e502a479814629e951f9fe9a0070d5995eea719c96eff533fbc211635fb` | `f939bb402383c39327238d95b3663ba2021fddad12c9a160c3e1fc93839e1923` (`candidate.patch`) | `e86e4f73bbc72e375bba5781e908104b7ef209f9b258462a5c2d3aac5538bfce` |

The table above records the original local experiment builds. `candidate` is
the subsequently rejected LUT variant. Clean selected source has exact diff
SHA-256 `14c2a6888efad7fdc1a8fd98c6a34bd7fc5597ebb56914832eed58bf50533bc8`.

All 24 coordinates share harness SHA-256
`6dcc88628ad37981de52748041cc5603085aa24d5196789af584014e50b8360c`.

## Exact output and sidecar equality

For each row below, all four variants produced one and only one official
signature and one metadata/sidecar signature. The official signature covers
every array's name, dtype, shape, byte strides, byte length, and contents.
Equality is exact, not tolerance-based.

| Profile / case | Official arrays SHA-256 | Metadata/sidecar SHA-256 |
| --- | --- | --- |
| qwen3-vl-8b / image24 | `995119cbb63fe0b4a17b296555489ef60acbc7f45437f3f206f56ad25fce9e5e` | `7746837ce611b56a00af29d32818515b0456ba9707cba41b705184effab1109b` |
| qwen3-vl-8b / ragged24 | `e37b43151a88491515b15a2d6e22f9ae9cd09993742868ad07908990f4e496e4` | `4b1c00ba38069ede09b0d86fd108e8683964a9fe6c64c60e42c404f2f1944640` |
| qwen3-vl-8b / rgb24 | `173684219f8149e55d659f138af714125aed1b51f89b3ac18d9dec3fb952508f` | `e49ec8f66b32fd0169e059e768c1374ff3cd97b547cd269c7a753b5ee653b2d8` |
| qwen3.5-9b / image24 | `4d885822255feb810a6b0068d2b461aafd64a1885afb17bb059e121676f87641` | `b1647144420ad7efebfc0fe9687a663dfaa2136e4d34ea7781e21d5f36ee6faf` |
| qwen3.5-9b / ragged24 | `b5d30c8de5b70136a0e34381191340134f2ccb48d06e87d3760502dead6dea69` | `3fa0b5806ab6741bf265379310568dfeb7d79b4ec757f8007d0f741527059df1` |
| qwen3.5-9b / rgb24 | `39665a39ef1ad7055792e06c672f9c72aa30cb04ee494896d71d08a9e7035e28` | `293fe6b5eea36d2255162730acaf1ffa5fca81be1af6eb9f08cf27c5b121feb6` |

## Local timing limitation

The authenticated local pass is not decision-quality performance evidence.
It followed a long D2 validation run, and a full Phase C attempt occurred
between candidate and later variants. Several seven-sample series show obvious
host interference: candidate qwen3.5 `image24` ranged from 386.5 to
776.0 ms, candidate `rgb24` reached 351.6 ms, and no-LUT qwen3-vl
`ragged24` ranged from 140.5 to 294.7 ms. Copy-only also trailed baseline in
five medians despite deterministically removing 25.7-56.5 MB of work per
operation.

The raw timings are retained below and are not discarded or massaged, but they
must not be used to claim that D3 hit or missed a release target. The
performance escalation is to rerun these exact four patches and authenticated
harness on the reserved Modal runner or another quiet host. Deterministic
allocation/copy counts and exact-output evidence remain valid.

The controlled Modal escalation is a D3 experiment-selection run, not the D4
certification protocol. Each of the six profile/case workloads receives one
four-variant treatment order; the order is rotated across workloads to balance
position globally, but is not replicated within a workload. That limits causal
timing interpretation even on a quiet worker. D4 retains the replicated release
performance protocol and is the only phase that can certify the target.
The artifact's dollar figure is an estimate for elapsed function-worker time
from the recorded CPU, memory, and nonpreemptible rates. It excludes image
construction and any provider billing adjustments, so it is not an invoice or
an all-in run-cost claim.

Modal's gVisor runtime does not expose cgroup-v2 limit files. The resource
validator records which attestation mode was used. When cgroup-v2 files exist,
it requires their CPU quota, effective cpuset, and memory limit to cover the
16-CPU/32-GiB request. The gVisor fallback is accepted only with empty cgroup
provenance, an explicit gVisor platform and uname, at least 16 CPUs reported
independently by process affinity, `os.cpu_count()`, and `lscpu`, and at least
32 GiB from `/proc/meminfo`. Both modes also retain the source-authenticated
Modal decorator request, nonpreemptible setting, and single-use-container flag.

## Experiment 1: direct strided still-image resize

Horizontal convolution now indexes validated positive-stride input directly.
Vertical-only resize does the same. This removes the unconditional packed
source allocation while retaining the required owned output for raw no-op
input.

| Profile / case | Wall before | Wall after | Wall delta | CPU delta | Allocations removed | Allocated bytes removed | Copies removed | Copied bytes removed | Peak transient removed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3-vl-8b / image24 | 362.971 ms | 371.609 ms | +2.4% | +2.4% | 24 | 56,494,152 | 24 | 56,494,152 | 2,353,923 |
| qwen3-vl-8b / ragged24 | 117.436 ms | 118.993 ms | +1.3% | +1.3% | 24 | 25,726,653 | 24 | 25,726,653 | 2,353,923 |
| qwen3-vl-8b / rgb24 | 178.181 ms | 180.611 ms | +1.4% | +1.4% | 24 | 36,132,888 | 24 | 36,132,888 | 2,353,923 |
| qwen3.5-9b / image24 | 367.925 ms | 372.464 ms | +1.2% | +1.2% | 24 | 56,494,152 | 24 | 56,494,152 | 2,353,923 |
| qwen3.5-9b / ragged24 | 118.690 ms | 117.256 ms | -1.2% | -1.2% | 24 | 25,726,653 | 24 | 25,726,653 | 2,353,923 |
| qwen3.5-9b / rgb24 | 179.489 ms | 179.925 ms | +0.2% | +0.2% | 24 | 36,132,888 | 24 | 36,132,888 | 2,353,923 |

The allocation/copy removal is exact and matches the former
`resize.packed_source` capacity. The local timing delta is inconclusive.

## Experiment 2: encoded no-resize ownership transfer

When a decoder already returns exact packed final geometry, its owned
`Vec<u8>` becomes `prepared_rgb`. Only `ragged24` contains the five
affected encoded occurrences.

| Profile / case | Wall before | Wall after | Wall delta | CPU delta | Allocations removed | Allocated bytes removed | Copies removed | Copied bytes removed | Peak transient removed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3-vl-8b / image24 | 371.609 ms | 365.024 ms | -1.8% | -1.8% | 0 | 0 | 0 | 0 | 0 |
| qwen3-vl-8b / ragged24 | 118.993 ms | 254.705 ms | +114.1% | +81.4% | 5 | 9,830,400 | 5 | 9,830,400 | 0 |
| qwen3-vl-8b / rgb24 | 180.611 ms | 191.216 ms | +5.9% | +5.7% | 0 | 0 | 0 | 0 | 0 |
| qwen3.5-9b / image24 | 372.464 ms | 371.453 ms | -0.3% | -0.3% | 0 | 0 | 0 | 0 | 0 |
| qwen3.5-9b / ragged24 | 117.256 ms | 120.467 ms | +2.7% | +2.7% | 5 | 9,830,400 | 5 | 9,830,400 | 0 |
| qwen3.5-9b / rgb24 | 179.925 ms | 181.156 ms | +0.7% | +0.7% | 0 | 0 | 0 | 0 | 0 |

The qwen3-vl ragged series contains the documented host-interference spike.
The deterministic 9.83 MB allocation/copy removal is the retained result;
unaffected suites correctly show zero counter changes.

## Experiment 3: bounded normalization lookup — rejected

The fused final patch writer replaces per-output division with a
`3 x 256` `float32` lookup on the stack. It is 3,072 bytes, bounded per
call, contains no caller data, and is regenerated from the immutable profile.
Tests compare every value bit-for-bit with the frozen expression for both
profiles.

| Profile / case | Wall before | Wall after | Wall delta | CPU delta | Allocations removed | Allocated bytes removed | Copies removed | Copied bytes removed | Peak transient removed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3-vl-8b / image24 | 365.024 ms | 338.683 ms | -7.2% | -7.2% | 0 | 0 | 0 | 0 | 0 |
| qwen3-vl-8b / ragged24 | 254.705 ms | 108.054 ms | -57.6% | -49.9% | 0 | 0 | 0 | 0 | 0 |
| qwen3-vl-8b / rgb24 | 191.216 ms | 214.341 ms | +12.1% | +11.2% | 0 | 0 | 0 | 0 | 0 |
| qwen3.5-9b / image24 | 371.453 ms | 460.247 ms | +23.9% | +16.9% | 0 | 0 | 0 | 0 | 0 |
| qwen3.5-9b / ragged24 | 120.467 ms | 110.927 ms | -7.9% | -7.9% | 0 | 0 | 0 | 0 | 0 |
| qwen3.5-9b / rgb24 | 181.156 ms | 298.751 ms | +64.9% | +42.2% | 0 | 0 | 0 | 0 | 0 |

The local timing result was inconclusive. Exact array and sidecar parity proved
the lookup preserved the output contract, but the controlled Modal rerun found
a consistent 35-39% regression in the targeted stage. The lookup is not in
selected production source.

## Historical rejected-LUT stack and remaining hotspots

| Profile / case | Wall baseline | Wall candidate | Wall delta | CPU delta | Allocations removed | Allocated bytes removed | Copies removed | Copied bytes removed | Peak transient removed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3-vl-8b / image24 | 362.971 ms | 338.683 ms | -6.7% | -6.7% | 24 | 56,494,152 | 24 | 56,494,152 | 2,353,923 |
| qwen3-vl-8b / ragged24 | 117.436 ms | 108.054 ms | -8.0% | -8.0% | 29 | 35,557,053 | 29 | 35,557,053 | 2,353,923 |
| qwen3-vl-8b / rgb24 | 178.181 ms | 214.341 ms | +20.3% | +19.1% | 24 | 36,132,888 | 24 | 36,132,888 | 2,353,923 |
| qwen3.5-9b / image24 | 367.925 ms | 460.247 ms | +25.1% | +18.0% | 24 | 56,494,152 | 24 | 56,494,152 | 2,353,923 |
| qwen3.5-9b / ragged24 | 118.690 ms | 110.927 ms | -6.5% | -6.5% | 29 | 35,557,053 | 29 | 35,557,053 | 2,353,923 |
| qwen3.5-9b / rgb24 | 179.489 ms | 298.751 ms | +66.4% | +43.5% | 24 | 36,132,888 | 24 | 36,132,888 | 2,353,923 |

Again, the wall/CPU columns are diagnostic only. The exact counters show no
allocation, copy, or owned-transient regression. Aggregating the candidate's
six observed operations leaves these material stage totals:

| Remaining stage | Aggregate exclusive time |
| --- | ---: |
| `native.media.resize` | 901.385 ms |
| `native.batch.plan` | 252.182 ms |
| `native.media.normalize_patchify_layout` | 227.476 ms |
| `native.media.decode_color` | 159.853 ms |
| `binding.destination.allocate` | 79.381 ms |

Resize remains the next profile target. These observed times were collected on
the same interfered host and rank work; they are not a D4 target claim.

## Final buffer census

| Lifetime | Buffer | Element type | Bound / ownership |
| --- | --- | --- | --- |
| Python parse through detached native call | `binding.owned_media` | `u8` | One immutable owned snapshot per supplied input; released after native execution. |
| Batch plan through execution | `prepared_rgb` | packed HWC `u8` | Exact planned RGB capacity per occurrence; retained so plans remain independently reusable. |
| Resize only | horizontal destination | packed HWC `u8` | One full `source_height x destination_width x 3` intermediate only when both axes resize; released before plan return. |
| Resize only | floating weights, bounds, fixed coefficients | `f64` / `(usize, usize)` / `i32` | Per-axis destination dimensions and bounded bicubic support; released after that convolution. |
| Returned batch | `pixel_values` | patch-major `f32` | Exact final/caller-owned destination allocated before execution. |

`execute_image_patchify_plan_into` reads `prepared_rgb` and writes
normalized channel/temporal/merge/patch order directly into the exact
`pixel_values` slice. There is no full `float32` HWC or CHW image,
normalized-image allocation, or temporal-duplication allocation.

There is no persistent arena. Direct-stride access removes the dominant copy
without retaining workload-sized memory. Selected source adds no normalization
scratch, so scratch high-water cannot scale with media size, request history,
failures, or processor count.

## Rejected experiments

- **Zero-copy `binding.owned_media`: rejected.** Native execution detaches
  the GIL and accepts writable NumPy arrays and arbitrary buffer providers.
  Borrowing a raw pointer would permit concurrent mutation and weaken the
  immutable-snapshot and failure-lifetime contract.
- **Consume caller-owned raw input on no-resize: rejected.** The public Rust
  surface borrows raw RGB and must return owned prepared storage. Only storage
  owned by the encoded decoder can transfer safely.
- **Retained per-processor media arena: rejected.** C1 plans must remain
  independently reusable and retain exact prepared media. A shared arena would
  alias plans or retain workload-sized storage across calls.
- **Full normalized image scratch: rejected.** It would recreate the
  `float32` intermediate D3 is required to avoid.
- **Normalization lookup: rejected.** The bounded 3 KiB table was exact, but a
  controlled same-worker run measured a 35-39% regression in its targeted
  stage. Its dormant helper and test were also removed from clean final source.

## Verification and gate status

Passed in the reviewed final source:

- `./scripts/cargo.sh test -p qwen-mm-core --locked`: 92 normal tests plus
  doctests.
- `./scripts/cargo.sh test -p qwen-mm-core --locked -- --ignored`: all 23
  asset-backed ignored tests.
- `./scripts/cargo.sh test -p qwen-mm-core --locked
  height_only_resize_matches_packed_oracle_for_padded_rows`: 1 passed.
- `./scripts/cargo.sh clippy -p qwen-mm-core --all-targets --locked --
  -D warnings`: passed.
- `make check`: final selected source passed workspace lint/format/Clippy,
  workspace tests/docs, 94 reference tests, fixtures, corpus, and goldens. Its
  last wheel-smoke step could not fetch NumPy because sandbox DNS was blocked;
  no test failed before that external fetch.
- `./scripts/test-python-binding.sh` and the full Phase B corpus passed on the
  measured production path before final cleanup; the selected production
  writer is source-identical and the clean selected installed-wheel Phase C
  gate below is the final-source boundary proof.
- `git diff --check`, Ruff check, and Ruff format check: passed.

Full Phase C evidence now passes from isolated clean commits for exact
copy-only, exact measured no-LUT, and clean selected source: 290/290 cases for
both profiles with zero skips in every run. The reports authenticate clean gate
inputs, source fingerprints, wheels, installed native modules, profile assets,
and isolated candidate execution. See `conformance/*/verification.md` and
`modal-summary.md` for exact hashes. No D4 certification is claimed.
