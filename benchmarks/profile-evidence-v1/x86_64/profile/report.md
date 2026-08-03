# qwen-mm whole-operation profile v1

- Source revision: `7434eef6d81a71ec479f1246f6a5b1ab29a3fc2c`
- Profiles: `qwen3-vl-8b, qwen3.5-9b`
- Cases: `text_short, text_long, image1, image24, ragged24, rgb24`
- Equal thread budgets: `1, 4`
- Timing ranking: exclusive stage duration; inclusive spans are retained but not summed.
- Build: locked Maturin `profiled-release` evidence wheel.

## x86_64

### Provenance

- Host: `Linux 4.19.0-gvisor`; CPU `Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Address sizes:       46 bits physical, 48 bits virtual
Byte Order:          Little Endian
CPU(s):              24
On-line CPU(s) list: 0-23
Vendor ID:           AuthenticAMD
Model name:          unknown
CPU family:          175
Model:               17
Thread(s) per core:  1
Core(s) per socket:  24
Socket(s):           1
Stepping:            unknown
BogoMIPS:            3700.58
Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy svm cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext perfctr_core fsgsbase bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves avx512vbmi umip avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid fsrm
Virtualization:      AMD-V
Hypervisor vendor:   Microsoft
Virtualization type: full`.
- Build command: `env CARGO_PROFILE_RELEASE_DEBUG=1 CARGO_PROFILE_RELEASE_OPT_LEVEL=3 CARGO_PROFILE_RELEASE_STRIP=none RUSTFLAGS=-Cforce-frame-pointers=yes uv run --locked --no-sync maturin build --release --locked --interpreter /workspace/qwen-mm/.venv/bin/python --out /tmp/qwen-mm-d1-x86-capture-cbwcn9ib/benchmarks/profile-evidence-v1/x86_64/artifacts`
- Build evidence: wheel `ec1ae587e38526c42370201dc924b4bf11e063b3a8dc7bee5e6735fcd908f52c`; native `7140f91b06ea41d9612160fe213c4338d1ff7b064eadef848ca633fcdd1f719c`; log `efd362eb290682ed640439f9cab97d4c02406f85d3c9f8a52397f33451c7997a`.
- Capture command: `/workspace/qwen-mm/reference/src/qwen_mm_reference/profile_v1.py capture --workload /workspace/qwen-mm/benchmarks/workloads-v2.json --benchmark-result /tmp/qwen-mm-d1-x86-capture-cbwcn9ib/benchmarks/profile-evidence-v1/x86_64/benchmark/result.json --profiles qwen3-vl-8b,qwen3.5-9b --cases text_short,text_long,image1,image24,ragged24,rgb24 --thread-budgets 1,4 --repetitions 3 --event-capacity 4096 --build-command 'env CARGO_PROFILE_RELEASE_DEBUG=1 CARGO_PROFILE_RELEASE_OPT_LEVEL=3 CARGO_PROFILE_RELEASE_STRIP=none RUSTFLAGS=-Cforce-frame-pointers=yes uv run --locked --no-sync maturin build --release --locked --interpreter /workspace/qwen-mm/.venv/bin/python --out /tmp/qwen-mm-d1-x86-capture-cbwcn9ib/benchmarks/profile-evidence-v1/x86_64/artifacts' --build-log /tmp/qwen-mm-d1-x86-capture-cbwcn9ib/benchmarks/profile-evidence-v1/x86_64/build/build.log --wheel /tmp/qwen-mm-d1-x86-capture-cbwcn9ib/benchmarks/profile-evidence-v1/x86_64/artifacts/qwen_mm-0.1.0-cp311-abi3-manylinux_2_34_x86_64.whl --phase-c-report /tmp/qwen-mm-d1-x86-capture-cbwcn9ib/benchmarks/profile-evidence-v1/x86_64/phase-c/report.json --artifact-directory /tmp/qwen-mm-d1-x86-capture-cbwcn9ib/benchmarks/profile-evidence-v1/x86_64/artifacts --artifact-publish-directory benchmarks/profile-evidence-v1/x86_64/artifacts --py-spy /workspace/qwen-mm/.venv/bin/py-spy --sampler-rate-hz 99 --sampler-duration-seconds 2 --source-revision 7434eef6d81a71ec479f1246f6a5b1ab29a3fc2c --source-digest 8a0e3379baa338cd2648d9270249b6f2ba8e46107e3c2e10d23258ac2b222367 --source-clean --output /tmp/qwen-mm-d1-x86-capture-cbwcn9ib/benchmarks/profile-evidence-v1/x86_64/profile/bundle.json --benchmark-phase-c-source-report /tmp/qwen-mm-d1-x86-capture-cbwcn9ib/benchmarks/profile-evidence-v1/x86_64/phase-c/report.json`
- Sampler: `py-spy 0.4.1`; mode `Python-only`; binary `e7c2de2dc54449ec88c086f1859555b4e34e63ccdcf3f8804496f9306cd44de6`; representative full command `/workspace/qwen-mm/.venv/bin/py-spy record --rate 99 --format raw -o /tmp/qwen-mm-profile-worker-yo8wqkeu/profile.raw -- /workspace/qwen-mm/.venv/bin/python -m qwen_mm_reference.profile_v1 _sample_worker --config /tmp/qwen-mm-profile-worker-yo8wqkeu/config.json`. Per-coordinate commands are retained in the bundle.
- Phase C: `pass`; report `75c0ba3bf5aac386b69d36360651055fa201e113f8b3b3273472a11703523635`; gate `2858f30aceae4e2f7088250cce4ae11b40012a23fb9c16d01ef7b589582cd5d8`; runtime native `7140f91b06ea41d9612160fe213c4338d1ff7b064eadef848ca633fcdd1f719c`.

### Observed exclusive-time bottlenecks

| Rank | Stage | Exclusive time | Share |
| ---: | --- | ---: | ---: |
| 1 | `native.media.resize` | 7687.537 ms | 55.2% |
| 2 | `native.media.normalize_patchify_layout` | 3002.756 ms | 21.6% |
| 3 | `native.media.decode_color` | 1293.162 ms | 9.3% |
| 4 | `binding.destination.allocate` | 1165.870 ms | 8.4% |
| 5 | `native.chat.tokenize` | 367.813 ms | 2.6% |
| 6 | `native.batch.plan` | 313.898 ms | 2.3% |
| 7 | `binding.parse_requests` | 55.069 ms | 0.4% |
| 8 | `native.destination.execute` | 16.710 ms | 0.1% |
| 9 | `binding.numpy.materialize` | 13.451 ms | 0.1% |
| 10 | `binding.prepare_batch` | 7.062 ms | 0.1% |
| 11 | `native.media.plan` | 6.953 ms | 0.0% |
| 12 | `native.chat.render` | 0.969 ms | 0.0% |

### OS-sampled stack hotspots by coordinate

| Profile/case/threads | Top inclusive frame | Top self frame | Samples |
| --- | --- | --- | ---: |
| `qwen3-vl-8b/image1/1` | `<module> (qwen_mm_reference/profile_v1.py:2704)` (100.0%) | `run (qwen_mm/benchmark.py:152)` (100.0%) | 183 |
| `qwen3-vl-8b/image1/4` | `<module> (qwen_mm_reference/profile_v1.py:2704)` (100.0%) | `run (qwen_mm/benchmark.py:152)` (100.0%) | 171 |
| `qwen3-vl-8b/image24/1` | `<module> (qwen_mm_reference/profile_v1.py:2704)` (100.0%) | `run (qwen_mm/benchmark.py:152)` (100.0%) | 231 |
| `qwen3-vl-8b/image24/4` | `<module> (qwen_mm_reference/profile_v1.py:2704)` (100.0%) | `run (qwen_mm/benchmark.py:152)` (100.0%) | 216 |
| `qwen3-vl-8b/ragged24/1` | `<module> (qwen_mm_reference/profile_v1.py:2704)` (100.0%) | `run (qwen_mm/benchmark.py:152)` (100.0%) | 220 |
| `qwen3-vl-8b/ragged24/4` | `<module> (qwen_mm_reference/profile_v1.py:2704)` (100.0%) | `run (qwen_mm/benchmark.py:152)` (100.0%) | 210 |
| `qwen3-vl-8b/rgb24/1` | `<module> (qwen_mm_reference/profile_v1.py:2704)` (100.0%) | `run (qwen_mm/benchmark.py:152)` (99.6%) | 238 |
| `qwen3-vl-8b/rgb24/4` | `<module> (qwen_mm_reference/profile_v1.py:2704)` (100.0%) | `run (qwen_mm/benchmark.py:152)` (100.0%) | 224 |
| `qwen3-vl-8b/text_long/1` | `<module> (qwen_mm_reference/profile_v1.py:2704)` (100.0%) | `run (qwen_mm/benchmark.py:152)` (99.4%) | 167 |
| `qwen3-vl-8b/text_long/4` | `<module> (qwen_mm_reference/profile_v1.py:2704)` (100.0%) | `run (qwen_mm/benchmark.py:152)` (100.0%) | 203 |
| `qwen3-vl-8b/text_short/1` | `<module> (qwen_mm_reference/profile_v1.py:2704)` (100.0%) | `run (qwen_mm/benchmark.py:152)` (95.6%) | 205 |
| `qwen3-vl-8b/text_short/4` | `<module> (qwen_mm_reference/profile_v1.py:2704)` (100.0%) | `run (qwen_mm/benchmark.py:152)` (98.4%) | 193 |
| `qwen3.5-9b/image1/1` | `<module> (qwen_mm_reference/profile_v1.py:2704)` (100.0%) | `run (qwen_mm/benchmark.py:152)` (100.0%) | 200 |
| `qwen3.5-9b/image1/4` | `<module> (qwen_mm_reference/profile_v1.py:2704)` (100.0%) | `run (qwen_mm/benchmark.py:152)` (100.0%) | 226 |
| `qwen3.5-9b/image24/1` | `<module> (qwen_mm_reference/profile_v1.py:2704)` (100.0%) | `run (qwen_mm/benchmark.py:152)` (100.0%) | 198 |
| `qwen3.5-9b/image24/4` | `<module> (qwen_mm_reference/profile_v1.py:2704)` (100.0%) | `run (qwen_mm/benchmark.py:152)` (100.0%) | 222 |
| `qwen3.5-9b/ragged24/1` | `<module> (qwen_mm_reference/profile_v1.py:2704)` (100.0%) | `run (qwen_mm/benchmark.py:152)` (99.6%) | 225 |
| `qwen3.5-9b/ragged24/4` | `<module> (qwen_mm_reference/profile_v1.py:2704)` (100.0%) | `run (qwen_mm/benchmark.py:152)` (100.0%) | 220 |
| `qwen3.5-9b/rgb24/1` | `<module> (qwen_mm_reference/profile_v1.py:2704)` (100.0%) | `run (qwen_mm/benchmark.py:152)` (100.0%) | 205 |
| `qwen3.5-9b/rgb24/4` | `<module> (qwen_mm_reference/profile_v1.py:2704)` (100.0%) | `run (qwen_mm/benchmark.py:152)` (99.5%) | 213 |
| `qwen3.5-9b/text_long/1` | `<module> (qwen_mm_reference/profile_v1.py:2704)` (100.0%) | `run (qwen_mm/benchmark.py:152)` (100.0%) | 212 |
| `qwen3.5-9b/text_long/4` | `<module> (qwen_mm_reference/profile_v1.py:2704)` (100.0%) | `run (qwen_mm/benchmark.py:152)` (100.0%) | 183 |
| `qwen3.5-9b/text_short/1` | `<module> (qwen_mm_reference/profile_v1.py:2704)` (100.0%) | `run (qwen_mm/benchmark.py:152)` (95.9%) | 193 |
| `qwen3.5-9b/text_short/4` | `<module> (qwen_mm_reference/profile_v1.py:2704)` (100.0%) | `run (qwen_mm/benchmark.py:152)` (94.1%) | 186 |

### Allocation bottlenecks

| Rank | Buffer | Class | Allocated | Count |
| ---: | --- | --- | ---: | ---: |
| 1 | `pixel_values` | `retained_output` | 11,605,377,024 B | 48 |
| 2 | `prepared_rgb` | `transient` | 1,450,672,128 B | 876 |
| 3 | `resize.packed_source` | `transient` | 1,448,491,392 B | 876 |
| 4 | `resize.horizontal.destination` | `transient` | 1,123,667,712 B | 624 |
| 5 | `decoded_rgb` | `transient` | 1,014,896,736 B | 588 |
| 6 | `binding.owned_media` | `transient` | 526,280,688 B | 876 |
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
| 2 | `binding.owned_media` | 526,280,688 B | 876 |
| 3 | `resize.noop.source_copy` | 325,582,848 B | 252 |

### Buffer-lifetime bottlenecks

| Rank | Buffer | Class | Total lifetime |
| ---: | --- | --- | ---: |
| 1 | `binding.owned_media` | `transient` | 325305.185 ms |
| 2 | `prepared_rgb` | `transient` | 206496.806 ms |
| 3 | `resize.packed_source` | `transient` | 7615.330 ms |
| 4 | `resize.horizontal.destination` | `transient` | 7559.659 ms |
| 5 | `decoded_rgb` | `transient` | 6106.078 ms |
| 6 | `input_ids` | `retained_output` | 4206.723 ms |
| 7 | `attention_mask` | `retained_output` | 4206.485 ms |
| 8 | `mm_token_type_ids` | `retained_output` | 4206.277 ms |
| 9 | `resize.horizontal.bounds` | `transient` | 3944.223 ms |
| 10 | `resize.horizontal.coefficients_i32` | `transient` | 3932.643 ms |
| 11 | `resize.vertical.bounds` | `transient` | 3653.576 ms |
| 12 | `resize.vertical.coefficients_i32` | `transient` | 3641.100 ms |
