# qwen-mm whole-operation profile v1

- Source revision: `7434eef6d81a71ec479f1246f6a5b1ab29a3fc2c`
- Profiles: `qwen3-vl-8b, qwen3.5-9b`
- Cases: `text_short, text_long, image1, image24, ragged24, rgb24`
- Equal thread budgets: `1, 4`
- Timing ranking: exclusive stage duration; inclusive spans are retained but not summed.
- Build: locked Maturin `profiled-release` evidence wheel.

## arm64

### Provenance

- Host: `Darwin 25.5.0`; CPU `Apple M4`.
- Build command: `env CARGO_PROFILE_RELEASE_DEBUG=1 CARGO_PROFILE_RELEASE_OPT_LEVEL=3 CARGO_PROFILE_RELEASE_STRIP=none RUSTFLAGS=-Cforce-frame-pointers=yes uv run --locked --no-sync maturin build --release --locked --interpreter /Users/jim/code/qwen-mm/.venv/bin/python --out /var/folders/4k/33n4pf5d33z_vd4yk3phjpxc0000gn/T/qwen-mm-d1-arm-capture-mqk9t1kd/benchmarks/profile-evidence-v1/arm64/artifacts`
- Build evidence: wheel `da3932744e07e087d74cdd1115f233406d1db02c2113ec3382c20c2633d72577`; native `a9d19ed6e97c14fbd8bec4283378ed3950732429ad8e2063f366505f8de39fde`; log `555b2a29c33bca4e8c58a747863b9c983af6d805863357ae89f81004512eeeba`.
- Capture command: `/Users/jim/code/qwen-mm/reference/src/qwen_mm_reference/profile_v1.py capture --workload /Users/jim/code/qwen-mm/benchmarks/workloads-v2.json --benchmark-result /var/folders/4k/33n4pf5d33z_vd4yk3phjpxc0000gn/T/qwen-mm-d1-arm-capture-mqk9t1kd/benchmarks/profile-evidence-v1/arm64/benchmark/result.json --profiles qwen3-vl-8b,qwen3.5-9b --cases text_short,text_long,image1,image24,ragged24,rgb24 --thread-budgets 1,4 --repetitions 3 --event-capacity 4096 --build-command 'env CARGO_PROFILE_RELEASE_DEBUG=1 CARGO_PROFILE_RELEASE_OPT_LEVEL=3 CARGO_PROFILE_RELEASE_STRIP=none RUSTFLAGS=-Cforce-frame-pointers=yes uv run --locked --no-sync maturin build --release --locked --interpreter /Users/jim/code/qwen-mm/.venv/bin/python --out /var/folders/4k/33n4pf5d33z_vd4yk3phjpxc0000gn/T/qwen-mm-d1-arm-capture-mqk9t1kd/benchmarks/profile-evidence-v1/arm64/artifacts' --build-log /var/folders/4k/33n4pf5d33z_vd4yk3phjpxc0000gn/T/qwen-mm-d1-arm-capture-mqk9t1kd/benchmarks/profile-evidence-v1/arm64/build/build.log --wheel /var/folders/4k/33n4pf5d33z_vd4yk3phjpxc0000gn/T/qwen-mm-d1-arm-capture-mqk9t1kd/benchmarks/profile-evidence-v1/arm64/artifacts/qwen_mm-0.1.0-cp311-abi3-macosx_11_0_arm64.whl --phase-c-report /var/folders/4k/33n4pf5d33z_vd4yk3phjpxc0000gn/T/qwen-mm-d1-arm-capture-mqk9t1kd/benchmarks/profile-evidence-v1/arm64/phase-c/report.json --artifact-directory /var/folders/4k/33n4pf5d33z_vd4yk3phjpxc0000gn/T/qwen-mm-d1-arm-capture-mqk9t1kd/benchmarks/profile-evidence-v1/arm64/artifacts --artifact-publish-directory benchmarks/profile-evidence-v1/arm64/artifacts --py-spy py-spy --sampler-rate-hz 99 --sampler-duration-seconds 2 --source-revision 7434eef6d81a71ec479f1246f6a5b1ab29a3fc2c --source-digest 8a0e3379baa338cd2648d9270249b6f2ba8e46107e3c2e10d23258ac2b222367 --source-clean --output /var/folders/4k/33n4pf5d33z_vd4yk3phjpxc0000gn/T/qwen-mm-d1-arm-capture-mqk9t1kd/benchmarks/profile-evidence-v1/arm64/profile/bundle.json --benchmark-phase-c-source-report /var/folders/4k/33n4pf5d33z_vd4yk3phjpxc0000gn/T/qwen-mm-d1-arm-capture-mqk9t1kd/benchmarks/profile-evidence-v1/arm64/phase-c/report.json`
- Sampler: `PROGRAM:sample  PROJECT:SamplingTools-64575.39.1`; mode `native`; binary `569d1e8f29e32e0557fdfffb7620e20126fab5df90f9e1bce4c6930ccded7363`; representative full command `/usr/bin/sample 62640 2 1 -mayDie -file /var/folders/4k/33n4pf5d33z_vd4yk3phjpxc0000gn/T/qwen-mm-profile-worker-gn6cqjzc/profile.raw`. Per-coordinate commands are retained in the bundle.
- Phase C: `pass`; report `2209c3eddbe0953afd9891ff70519c790cf294c72fb97ccc4e7478a41f640db5`; gate `1c3484e051ec50614d2caa19982b2cf3e92b9fae35a26e6a0651056995d9e0ae`; runtime native `a9d19ed6e97c14fbd8bec4283378ed3950732429ad8e2063f366505f8de39fde`.

### Observed exclusive-time bottlenecks

| Rank | Stage | Exclusive time | Share |
| ---: | --- | ---: | ---: |
| 1 | `native.media.resize` | 4485.061 ms | 53.3% |
| 2 | `native.media.normalize_patchify_layout` | 1599.802 ms | 19.0% |
| 3 | `native.media.decode_color` | 956.351 ms | 11.4% |
| 4 | `native.batch.plan` | 900.574 ms | 10.7% |
| 5 | `binding.destination.allocate` | 291.938 ms | 3.5% |
| 6 | `native.chat.tokenize` | 132.592 ms | 1.6% |
| 7 | `binding.parse_requests` | 20.643 ms | 0.2% |
| 8 | `native.media.plan` | 7.812 ms | 0.1% |
| 9 | `binding.prepare_batch` | 6.310 ms | 0.1% |
| 10 | `binding.numpy.materialize` | 4.371 ms | 0.1% |
| 11 | `native.destination.execute` | 4.079 ms | 0.0% |
| 12 | `native.chat.render` | 0.300 ms | 0.0% |

### OS-sampled stack hotspots by coordinate

| Profile/case/threads | Top inclusive frame | Top self frame | Samples |
| --- | --- | --- | ---: |
| `qwen3-vl-8b/image1/1` | `PyEval_EvalCode  (in libpython3.11.dylib)` (99.9%) | `_RNvNtCslYd21UUycFy_12qwen_mm_core6resize21resize_fixed_point_u8  (in _native.abi3.so)` (62.9%) | 1535 |
| `qwen3-vl-8b/image1/4` | `PyEval_EvalCode  (in libpython3.11.dylib)` (99.9%) | `_RNvNtCslYd21UUycFy_12qwen_mm_core6resize21resize_fixed_point_u8  (in _native.abi3.so)` (62.8%) | 1534 |
| `qwen3-vl-8b/image24/1` | `PyEval_EvalCode  (in libpython3.11.dylib)` (100.0%) | `_RNvNtCslYd21UUycFy_12qwen_mm_core6resize21resize_fixed_point_u8  (in _native.abi3.so)` (62.5%) | 1534 |
| `qwen3-vl-8b/image24/4` | `PyEval_EvalCode  (in libpython3.11.dylib)` (99.9%) | `_RNvNtCslYd21UUycFy_12qwen_mm_core6resize21resize_fixed_point_u8  (in _native.abi3.so)` (62.7%) | 1526 |
| `qwen3-vl-8b/ragged24/1` | `PyEval_EvalCode  (in libpython3.11.dylib)` (99.9%) | `_RNvNtCslYd21UUycFy_12qwen_mm_core6resize21resize_fixed_point_u8  (in _native.abi3.so)` (50.8%) | 1531 |
| `qwen3-vl-8b/ragged24/4` | `PyEval_EvalCode  (in libpython3.11.dylib)` (99.9%) | `_RNvNtCslYd21UUycFy_12qwen_mm_core6resize21resize_fixed_point_u8  (in _native.abi3.so)` (51.3%) | 1533 |
| `qwen3-vl-8b/rgb24/1` | `PyEval_EvalCode  (in libpython3.11.dylib)` (99.9%) | `_RNvNtCslYd21UUycFy_12qwen_mm_core6resize21resize_fixed_point_u8  (in _native.abi3.so)` (40.5%) | 1537 |
| `qwen3-vl-8b/rgb24/4` | `PyEval_EvalCode  (in libpython3.11.dylib)` (99.9%) | `_RNvNtCslYd21UUycFy_12qwen_mm_core6resize21resize_fixed_point_u8  (in _native.abi3.so)` (40.0%) | 1531 |
| `qwen3-vl-8b/text_long/1` | `PyEval_EvalCode  (in libpython3.11.dylib)` (95.5%) | `_xzm_free  (in libsystem_malloc.dylib)` (12.5%) | 1528 |
| `qwen3-vl-8b/text_long/4` | `PyEval_EvalCode  (in libpython3.11.dylib)` (95.5%) | `_xzm_free  (in libsystem_malloc.dylib)` (12.9%) | 1527 |
| `qwen3-vl-8b/text_short/1` | `PyEval_EvalCode  (in libpython3.11.dylib)` (98.8%) | `_RNvNtCs4Yb0IwGNUY2_4sha26sha25611compress256  (in _native.abi3.so)` (20.5%) | 1528 |
| `qwen3-vl-8b/text_short/4` | `PyEval_EvalCode  (in libpython3.11.dylib)` (99.2%) | `_RNvNtCs4Yb0IwGNUY2_4sha26sha25611compress256  (in _native.abi3.so)` (21.0%) | 1527 |
| `qwen3.5-9b/image1/1` | `PyEval_EvalCode  (in libpython3.11.dylib)` (99.9%) | `_RNvNtCslYd21UUycFy_12qwen_mm_core6resize21resize_fixed_point_u8  (in _native.abi3.so)` (62.6%) | 1525 |
| `qwen3.5-9b/image1/4` | `PyEval_EvalCode  (in libpython3.11.dylib)` (99.9%) | `_RNvNtCslYd21UUycFy_12qwen_mm_core6resize21resize_fixed_point_u8  (in _native.abi3.so)` (62.0%) | 1526 |
| `qwen3.5-9b/image24/1` | `PyEval_EvalCode  (in libpython3.11.dylib)` (99.9%) | `_RNvNtCslYd21UUycFy_12qwen_mm_core6resize21resize_fixed_point_u8  (in _native.abi3.so)` (62.6%) | 1537 |
| `qwen3.5-9b/image24/4` | `PyEval_EvalCode  (in libpython3.11.dylib)` (99.9%) | `_RNvNtCslYd21UUycFy_12qwen_mm_core6resize21resize_fixed_point_u8  (in _native.abi3.so)` (62.3%) | 1533 |
| `qwen3.5-9b/ragged24/1` | `PyEval_EvalCode  (in libpython3.11.dylib)` (99.9%) | `_RNvNtCslYd21UUycFy_12qwen_mm_core6resize21resize_fixed_point_u8  (in _native.abi3.so)` (50.7%) | 1524 |
| `qwen3.5-9b/ragged24/4` | `PyEval_EvalCode  (in libpython3.11.dylib)` (99.9%) | `_RNvNtCslYd21UUycFy_12qwen_mm_core6resize21resize_fixed_point_u8  (in _native.abi3.so)` (52.0%) | 1534 |
| `qwen3.5-9b/rgb24/1` | `PyEval_EvalCode  (in libpython3.11.dylib)` (99.9%) | `_RNvNtCslYd21UUycFy_12qwen_mm_core6resize21resize_fixed_point_u8  (in _native.abi3.so)` (39.9%) | 1524 |
| `qwen3.5-9b/rgb24/4` | `PyEval_EvalCode  (in libpython3.11.dylib)` (99.9%) | `_RNvNtCslYd21UUycFy_12qwen_mm_core6resize21resize_fixed_point_u8  (in _native.abi3.so)` (40.3%) | 1533 |
| `qwen3.5-9b/text_long/1` | `PyEval_EvalCode  (in libpython3.11.dylib)` (95.8%) | `_xzm_free  (in libsystem_malloc.dylib)` (10.3%) | 1514 |
| `qwen3.5-9b/text_long/4` | `PyEval_EvalCode  (in libpython3.11.dylib)` (94.6%) | `_xzm_free  (in libsystem_malloc.dylib)` (12.6%) | 1528 |
| `qwen3.5-9b/text_short/1` | `PyEval_EvalCode  (in libpython3.11.dylib)` (99.0%) | `_RNvNtCs4Yb0IwGNUY2_4sha26sha25611compress256  (in _native.abi3.so)` (19.2%) | 1532 |
| `qwen3.5-9b/text_short/4` | `PyEval_EvalCode  (in libpython3.11.dylib)` (99.4%) | `_RNvNtCs4Yb0IwGNUY2_4sha26sha25611compress256  (in _native.abi3.so)` (19.6%) | 1532 |

### Allocation bottlenecks

| Rank | Buffer | Class | Allocated | Count |
| ---: | --- | --- | ---: | ---: |
| 1 | `pixel_values` | `retained_output` | 11,605,377,024 B | 48 |
| 2 | `prepared_rgb` | `transient` | 1,450,672,128 B | 876 |
| 3 | `resize.packed_source` | `transient` | 1,448,491,392 B | 876 |
| 4 | `resize.horizontal.destination` | `transient` | 1,123,667,712 B | 624 |
| 5 | `decoded_rgb` | `transient` | 1,014,896,736 B | 588 |
| 6 | `binding.owned_media` | `transient` | 526,338,360 B | 876 |
| 7 | `resize.vertical.weights_f64` | `transient` | 20,287,488 B | 624 |
| 8 | `resize.horizontal.weights_f64` | `transient` | 19,826,688 B | 624 |
| 9 | `resize.vertical.coefficients_i32` | `transient` | 10,143,744 B | 624 |
| 10 | `resize.horizontal.coefficients_i32` | `transient` | 9,913,344 B | 624 |
| 11 | `resize.horizontal.bounds` | `transient` | 7,729,152 B | 624 |
| 12 | `resize.vertical.bounds` | `transient` | 7,643,136 B | 624 |

### Copy bottlenecks

| Rank | Copy | Copied | Copies |
| ---: | --- | ---: | ---: |
| 1 | `resize.packed_source` | 1,448,491,392 B | 876 |
| 2 | `binding.owned_media` | 526,338,360 B | 876 |
| 3 | `resize.noop.source_copy` | 325,582,848 B | 252 |

### Buffer-lifetime bottlenecks

| Rank | Buffer | Class | Total lifetime |
| ---: | --- | --- | ---: |
| 1 | `binding.owned_media` | `transient` | 196551.596 ms |
| 2 | `prepared_rgb` | `transient` | 118227.898 ms |
| 3 | `resize.packed_source` | `transient` | 4452.894 ms |
| 4 | `resize.horizontal.destination` | `transient` | 4433.490 ms |
| 5 | `decoded_rgb` | `transient` | 3589.053 ms |
| 6 | `resize.horizontal.bounds` | `transient` | 2539.511 ms |
| 7 | `resize.horizontal.coefficients_i32` | `transient` | 2535.241 ms |
| 8 | `input_ids` | `retained_output` | 1906.642 ms |
| 9 | `attention_mask` | `retained_output` | 1906.610 ms |
| 10 | `mm_token_type_ids` | `retained_output` | 1906.569 ms |
| 11 | `resize.vertical.bounds` | `transient` | 1901.836 ms |
| 12 | `resize.vertical.coefficients_i32` | `transient` | 1897.501 ms |
