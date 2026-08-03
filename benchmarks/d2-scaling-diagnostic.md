# Phase D2 bounded-parallelism diagnostic

- Status: **diagnostic only**; D4 owns release certification.
- Source base: `f92a3ab8860445af1f6abeb31f996024aef1b64f` plus implementation diff
  `3aece7208a5d3931dee3b9d7506a4ba33c6afa00b9396ef795a638f1b71dc21f`.
- Wheel SHA-256: `2b082d49222b2eaecf2d28437a9b76002fdb4a1e98edde9243e9d6ea2ff8f243`.
- Host: Mac mini, Apple M4 (4 performance + 6 efficiency cores), 24 GiB,
  arm64, macOS 26.5.2 / Darwin 25.5.0, Python 3.11.15.
- Workload: `qwen3-vl-8b/image24`, release build, one fresh process per
  budget, executed sequentially t1 through t4 with two warmups and eight
  minimum measured samples plus a one-second minimum measurement window.
- Boundary: installed release wheel, structured messages + encoded in-memory
  images through fully materialized NumPy arrays.

The implementation digest is reproducible with this exact wheel-source scope:

```text
git diff --binary -- Cargo.toml Cargo.lock \
  crates/qwen-mm-core/Cargo.toml crates/qwen-mm-core/src \
  crates/qwen-mm-python/Cargo.toml crates/qwen-mm-python/src \
  crates/qwen-mm-python/python | shasum -a 256
```

The same benchmark worker command was used for every run, changing only the
config file's explicit `thread_budget` and output path:

```text
env PYTHONPATH=/tmp/qwen-mm-d2-final-site:/tmp/qwen-mm-d2-6841/reference/src \
  QWEN_MM_ASSETS_ROOT=/tmp/qwen-mm-d2-6841/reference/.cache/huggingface \
  /Users/jim/code/qwen-mm/.venv/bin/python \
  -m qwen_mm_reference.benchmark_v2 _worker \
  --config /tmp/qwen-mm-d2-config-t{thread}.json \
  --workload /tmp/qwen-mm-d2-6841/benchmarks/workloads-v2.json \
  --case image24 --output /tmp/qwen-mm-d2-final-result-t{thread}.json
```

The shared benchmark worker's `configure_threads` function exported
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `RAYON_NUM_THREADS` equal
to each t1–t4 budget. That differs from the production guidance to set active
nested runtimes to one while qwen-mm owns the visual budget. In this candidate
stack those settings were dormant: qwen-mm never initialized global Rayon,
Torch was not configured or imported, and the native image/tokenizer path does
not invoke BLAS or NumExpr kernels. Median process CPU remained below the
configured budget at every point (100.0%, 193.4%, 282.4%, and 367.3%). The
settings are disclosed here rather than treated as additional active workers;
hosts that actually invoke those runtimes must still apply the documented
one-total-budget controls.

| Budget | Samples | Wall p50 | p90 | Min–max | CPU p50 | Median CPU utilization | Throughput p50 | Speedup vs t1 | Peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 356.678 ms | 396.267 ms | 352.318–476.820 ms | 356.692 ms | 100.0% | 2.804/s | 1.000x | 4,900,044,800 B |
| 2 | 8 | 187.720 ms | 194.589 ms | 185.725–205.245 ms | 363.481 ms | 193.4% | 5.327/s | 1.900x | 4,901,535,744 B |
| 3 | 8 | 131.199 ms | 137.267 ms | 130.327–150.406 ms | 369.883 ms | 282.4% | 7.622/s | 2.719x | 4,909,694,976 B |
| 4 | 10 | 102.450 ms | 105.632 ms | 102.244–118.867 ms | 376.762 ms | 367.3% | 9.761/s | 3.481x | 4,915,986,432 B |

All pre-measurement, measured-iteration, and post-measurement output checks
passed. Every budget produced the same dtype, shape, stride, and SHA-256 for
all five official arrays, including the 452,984,832-byte `pixel_values`
result. The complete raw sample arrays, per-sample CPU utilization and RSS,
output signatures, commands, and provenance are in
[`d2-scaling-diagnostic.json`](d2-scaling-diagnostic.json).

The frozen paired benchmark adapter intentionally returns only official arrays,
so metadata was authenticated in a separate pass through the same final release
wheel. Canonical sorted compact JSON of `PreparedBatch.metadata` was 272,244
bytes and had SHA-256
`7746837ce611b56a00af29d32818515b0456ba9707cba41b705184effab1109b` at
every budget t1–t4. Its complete `sidecar` subtree was 2,310 bytes with SHA-256
`e99b46fd1cc5eb02a1fc1202762dd5a2456a7a4f3de8bd15c5a6bff31cdef530`
at every budget. Metadata/sidecar equality also passed exact
`PreparedImageBatch` comparison in the t1–t4 mixed/ragged/repeated core stress
sweep for both profiles and the t1–t4 installed-binding metadata test.

t2, t3, and t4 scale monotonically on the D1-directed many-image boundary;
t4 reaches 3.481x with 367.3% median process CPU utilization. One t1 sample was
a large wall-time outlier, but its CPU time remained close to the other t1
samples. This short local diagnostic does not promote the result into a
controlled-host release claim.

Native-inclusive peak whole-process RSS increased by 15,941,632 bytes
(15.2 MiB, 0.33%) from t1 to t4. This high-water value includes two candidate
adapter instances, pre/post verification outputs, baseline copies, the 432 MiB
retained pixel matrix, Python, and Rust-native allocations. It honestly bounds
the same-command native-plus-Python scaling envelope, but it cannot isolate
native transient allocations from the much larger retained result. The
unobserved worker's approximately 429 KiB `transient_live_bytes` value is its
Python `tracemalloc` fallback and explicitly excludes Rust-native transient
buffers. Native allocation/lifetime attribution remains on the serial observed
lane; claiming per-thread native transient peaks from that fallback would be
incorrect.
