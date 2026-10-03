# Video v0.2 verification and timings

The installed qwen-mm 0.2.0 wheel passed 120 video profile/case checks against
Transformers 5.14.1 and qwen-vl-utils 0.0.14. The final oracle, benchmark and real
TorchCodec witness use the same native binary and Python facades; their SHA-256
values are recorded in each report. [validation.json](validation.json) ties those
hashes to the local production wheel, source files and completed validation gates.

The [v0.2.0 release](https://github.com/jbohnslav/qwen-mm/releases/tag/v0.2.0)
includes native verification manifests and checksums. Both release jobs pass
120 video checks; their artifacts are authenticated separately from the local
timing and model witnesses below.

## Evidence

| Report | What it establishes |
| --- | --- |
| [oracle.json](oracle.json) | 120 checks across `qwen3-vl-8b` and `qwen3.5-9b`: 42 generated success cases, shared-processor concurrency and 17 error/resource/recovery checks per profile. |
| [oracle-linux-docker.json](oracle-linux-docker.json) | The same 120 checks on the final Linux x86_64 wheel under Docker emulation on macOS ARM64; native CI is recorded separately. |
| [rounding-linux-before.json](rounding-linux-before.json), [rounding-linux-after.json](rounding-linux-after.json) | Independent float/impulse witnesses: four strict-tolerance mismatches before the correction, zero after, with the same diagnostic source and reference environment. |
| [torchcodec.json](torchcodec.json) | 40 checks with real TorchCodec 0.17.0: decoded FrameBatch, selective VideoDecoder and file routes, including resized clips and timestamp metadata. |
| [benchmark.json](benchmark.json) | Generated CPU preprocessing timings with output parity checked before timing. |
| [consumer.json](consumer.json) | Eight Qwen3.5-0.8B CPU model-consumer cases using the 9B processor assets, including odd-frame video and mixed image/video input, with input, first-token-logit and generation comparisons. |
| [rosetta.json](rosetta.json) | Executable published Rosetta recipes, with only placeholder paths, image URLs and cache locations bound to fixtures; every required recipe passed. This is a documentation recipe report. |
| [public-install.json](public-install.json) | Downloaded public macOS wheel installed with its declared video extra and no Torch: six mixed/Pillow/real-MP4 calls pass on both profiles, plus package smoke and 13 source regressions. |
| [validation.json](validation.json) | Local wheel/source provenance and the native, binding, scripts, installed-wheel, formatting and packaging gates. Remote CI and release results are tracked separately by their GitHub manifests. |

The consumer and recipe reports identify their installed production wheel;
validation records the final wheel's native, facade and source hashes. The 0.8B
consumer is a witness for the matching processor assets, rather than an
additional supported profile or an inference-speed measurement.

## Contract coverage

The generated fixtures distinguish time, channels and both spatial axes. Coverage
includes Pillow frame lists and frame paths, NumPy and Torch decoded clips,
qwen-vl-utils prepared float tuples, explicit frame indices and presentation
timestamps, single and odd frame counts, clipping and sampling, spatial and total
pixel budgets, repeated media references, mixed image/video ordering,
heterogeneous batches, both padding sides, Torch outputs and bounded concurrent
calls. Long-source sampling cases compare the actual pinned Torch linspace
indices, including nonzero clip starts. A timestamp boundary case checks that
distinct subnanosecond source times survive the media cache key.

Keys, shapes, dtypes, contiguity, IDs, masks, token types, grids and occurrence
order are checked exactly. Normalized video tensors use absolute tolerance
`2e-6`; the final ordinary video comparisons were exact. Pillow frame lists that
perform an inner still-image resize use the existing frozen resize-v2
per-channel gates, followed by an exact canonical normalization/packing witness.
The tests also check original frame indices, effective sampling rates, total
frame counts and temporal-pair timestamps, including the distinct upstream
padding rules for plain image lists and decoded clips.

The Linux gate exposed four one-level uint8 differences when resizing the
generated 177×259 clip to 96×160. The pinned Linux Torch CPU build's GCC-generated
loop rounds grouped products separately before accumulating them, then contracts
the scalar tail; the ARM build contracts the products throughout. Those
different float32
accumulation orders can place the same pixel on opposite sides of a half-integer
rounding boundary. Native video resize follows the corresponding pinned CPU
contraction schedule for standalone frames and clips. This regression was fixed
in the resize implementation with the
video tolerance unchanged at `2e-6`.

[`video_resize_diagnostic.py`](../../scripts/video_resize_diagnostic.py) reports
the source/target geometry, uint8 mismatch coordinates, official float32 values
before rounding, horizontal intermediates and independent impulse-derived
coefficients. It always emits the original rounding-boundary witnesses, even on
a host where output parity passes. Its two-dimensional impulse probes avoid the
upstream width-one antialias edge path. The diagnostic is separate from the
acceptance gate.

Pillow lists run through actual pinned `fetch_video`. Decoded clips use the
pinned geometry and TorchVision resize followed by the actual Transformers
processor with resizing and sampling disabled. For MP4 parity, the oracle
independently decodes all RGB frames with PyAV, then applies the pinned Qwen
sampling equations and TorchVision/Transformers. TorchVision 0.28 removed the
file reader used by qwen-vl-utils 0.0.14, so this MP4 reference is explicitly
identified in the report. The separate real TorchCodec report covers its native
decoder route.

## Measured preprocessing

Captured on 2026-10-03 on an Apple M4, macOS ARM64, Python 3.11.15, with two
processor worker threads and two Torch threads. Each lane has two warmups and
five timed repetitions; call order alternates. Times cover complete
`prepare_batch` adaptation, preprocessing, prompt rendering and tokenization,
with all outputs materialized. They exclude model loading and inference.

| Profile | Workload | Reference median ms | qwen-mm median ms | Speedup |
| --- | --- | ---: | ---: | ---: |
| qwen3-vl-8b | Decoded, no resize | 2.400 | 1.039 | 2.31× |
| qwen3-vl-8b | Decoded, resize | 6.266 | 3.212 | 1.95× |
| qwen3-vl-8b | Pillow frame list | 2.127 | 1.244 | 1.71× |
| qwen3-vl-8b | MP4 including decode | 19.730 | 12.034 | 1.64× |
| qwen3.5-9b | Decoded, no resize | 2.475 | 1.031 | 2.40× |
| qwen3.5-9b | Decoded, resize | 6.356 | 3.198 | 1.99× |
| qwen3.5-9b | Pillow frame list | 2.137 | 1.246 | 1.72× |
| qwen3.5-9b | MP4 including decode | 19.862 | 12.077 | 1.64× |

Decoded/list workloads have eight 192×256 RGB frames. The resize case targets
224×320. The lossless MP4 contains 36 such frames at 12 fps and selects eight.
The file reference converts every decoded frame to RGB using PyAV's default
FFmpeg threading; qwen-mm converts the selected frames to RGB and uses the
configured native worker budget. Thus the MP4 comparison includes both media
adaptation and preprocessing. It does not measure a TorchCodec speedup.

An independent full-RGB PyAV decode measured 16.877 ms median. That diagnostic
is recorded separately and is never subtracted from the end-to-end times.
These measurements describe the generated workloads on this host. They make no
general video, inference-speed or peak-RSS claim.

## Reproduce

From the repository root, the fresh production-wheel parity gate builds a wheel,
installs the locked reference dependencies in an isolated environment and runs
the source regressions plus the public oracle:

```sh
make video-test
```

For retained reports or timing, use Python from a separate environment containing
the installed production wheel and the locked `qwen-mm-reference` dependencies.
The scripts reject the checkout's editable qwen-mm package. Processor assets
must be present in `reference/.cache/huggingface`, or provide `--cache-dir`.
Model weights and a GPU are unnecessary for the oracle and benchmark.

```sh
/path/to/wheel-venv/bin/python scripts/video_oracle.py --output /tmp/video-oracle.json
/path/to/wheel-venv/bin/python -m unittest discover -s scripts/tests -p test_video_oracle.py
/path/to/wheel-venv/bin/python scripts/video_benchmark.py --output /tmp/video-benchmark.json
/path/to/torchcodec-wheel-venv/bin/python scripts/video_torchcodec.py --output /tmp/video-torchcodec.json
```

The benchmark defaults match the captured report. PyAV 18.0.0 is needed for the
generated MP4 cases; `--no-files` runs only decoded/list workloads. The real
TorchCodec witness additionally requires TorchCodec and a compatible FFmpeg
runtime. Run timings after the other validation jobs finish.
