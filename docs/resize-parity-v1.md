# Resize parity v1 decision record

Status: accepted for B5 on 2026-08-01.

## Decision

Use the in-tree `qwen-mm-tolerance-image-and-torchvision-video-v1` kernels in
`qwen-mm-core`:

- Image: a custom separable, fixed-point RGB8 Keys bicubic-antialias kernel.
  It uses half-pixel coordinates, `a=-0.5`, normalized double-precision
  weights, dynamic precision at most 22 bits, i16 coefficients, clamping, and
  an independently quantized horizontal and vertical pass. It is
  tolerance-compatible with Pillow 12.3.0 under the frozen one-byte bound. It
  is **not** an arithmetic port of Pillow's constant-22-bit/i32 `Resample.c`
  path and must not be described as exact Pillow arithmetic.
- Video: an in-tree source-faithful implementation of the pinned TorchVision
  CPU uint8 tensor path. It casts RGB8 to float32, performs the PyTorch
  separable antialiased Keys cubic (`a=-0.5`) convolution using float32 and the
  compiled multiply-add shape, clamps to `0..255`, rounds ties to even as the
  uint8 wrapper does, and exposes the integer-valued result as float32.

Both entry points consume the destination dimensions and checked packed RGB
layout from a B3 `ImageGeometryPlan`. They accept caller-owned strided RGB8,
return packed HWC output, and have no Python or Torch runtime dependency.

`fast_image_resize = 6.1.0` remains exact-pinned as a **dev dependency only**
so the rejected candidate is reproducible. It is not linked into normal
`qwen-mm-core` consumers.

## Frozen oracle and evidence

The 17-case oracle is
[`reference/resize/v1/manifest.json`](../reference/resize/v1/manifest.json).
It was captured on macOS arm64 with Python 3.11.15 and exact packages Pillow
12.3.0, NumPy 2.4.6, qwen-vl-utils 0.0.14, Torch 2.13.0, and TorchVision
0.28.0. The manifest authenticates the generator, compatibility manifest,
qwen-vl-utils source, profile fingerprints, every case slice, and these
artifacts:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `sources.rgb8.bin` | 582,565 | `b2bdeece59fa3711a4f073e9aa6f6cc5c60ffff7ee9a511859c5f76366f8a15a` |
| `pillow-image-rgb8.bin` | 359,424 | `8a7d8f79f1a0eabd5257a7e35b086ef9a5bbad72e5780eaf5fc7ae88dc24b5b2` |
| `torchvision-video-f32le.bin` | 1,437,696 | `dc6f0b22d600dc76be485cf58f8db2b68ba8a0e977d63d403211dc60fafca710` |

The manifest SHA-256 is
`fb846bd7487dafa6bb91da1f9e18b7719f086a08057f284b42f9d11361dfe1ea`.
Coverage includes no-op, up/downsampling, aligned and off-grid sizes,
factor/min/max boundaries, portrait and landscape, horizontal-only and
vertical-only resizing, padded non-noop input stride, impulse, ramp,
checkerboard, edges, deterministic noise, and independent RGB channels.

The source semantics were checked against Pillow 12.3.0
[`Resample.c`](https://github.com/python-pillow/Pillow/blob/12.3.0/src/libImaging/Resample.c),
TorchVision 0.28.0
[`_functional_tensor.py`](https://github.com/pytorch/vision/blob/v0.28.0/torchvision/transforms/_functional_tensor.py),
and PyTorch 2.13.0
[`UpSampleKernel.cpp`](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/cpu/UpSampleKernel.cpp).
The local qwen-vl-utils source is authenticated by the compatibility and resize
manifests rather than an unpinned branch URL.

## Candidate results

The full per-case maximum error, RMSE, p50/p90/p99 absolute error, nonzero
count, count beyond tolerance, maximum ULP distance, and first differing
element/pixel/channel are stored in the host reports.

| Candidate | Exact version | Image result | Video result | Decision |
| --- | --- | --- | --- | --- |
| In-tree custom kernels | `qwen-mm-core 0.1.0`; implementation SHA-256 `bf05ef4166f5d9fc8d7f2e76df10f814c607d027ea53a435eb7b778d61fd3dd7` | 0/17 cases failed; max error 1; 825/359,424 bytes differ; none beyond tolerance. Worst RMSE 0.0967404980 in `factor-above-independent-rgb`, first difference element 3457: 247 vs 246. | 0/17 cases failed; bit-identical on the corpus (max error and ULP distance 0). | Selected |
| `fast_image_resize` Catmull-Rom | `fast_image_resize=6.1.0` | 6/17 cases failed; 10,049 values beyond tolerance. Worst case `off-grid-landscape-checkerboard`: max error 21, RMSE 2.812596449, p90 3, p99 13, first difference element 6: 1 vs 0. | 3/17 cases failed; 114 values beyond `1e-4`. Worst RMSE case `min-above-checkerboard`: max error 1, RMSE 0.0592927061, 108 beyond tolerance, first difference element 38: 64 vs 63. | Rejected |

The selected image differences are expected from its dynamic-i16 arithmetic;
they are evidence of tolerance compatibility, not exact Pillow reproduction.
The v1 tolerance remains unchanged. The selected video result is exact for the
frozen weights and samples, but this record does not claim a general
bit-for-bit guarantee beyond the supported corpus and platforms.

## Platform evidence

| Report | Execution | Result |
| --- | --- | --- |
| [`macos-arm64-native.json`](../reference/resize/v1/results/macos-arm64-native.json) | Native macOS arm64, rustc 1.97.1 | Selected image and video pass |
| [`linux-x86_64-emulated.json`](../reference/resize/v1/results/linux-x86_64-emulated.json) | Linux x86_64 binary under Docker Desktop QEMU emulation on macOS arm64, rustc 1.97.1 | Selected image and video pass |

Both reports authenticate the fixture manifest, selected implementation, and
report-generator source. The Linux result is separate x86_64 code-generation
and execution evidence, but it is explicitly **emulated**: the oracle was not
regenerated on Linux, this is not native Linux hardware evidence, and it makes
no performance claim. Native Linux CI should continue to run the committed
corpus as an additional gate.

## Consequences for B6 and Phase E

B6 must decode supported encoded images into non-linear RGB8 first, preserve
the v1 mode/alpha policy, obtain the B3 plan from decoded dimensions and
options, then call `resize_image_rgb8`. Decode, orientation, color conversion,
and metadata policy do not belong inside the resize kernel. The packed resize
output can flow directly to B4 patchification.

Phase E must reuse `resize_video_rgb8_to_f32` for the final frame resize and
preserve qwen-vl-utils 0.0.14 raw-frame ordering. A caller-owned raw-frame list
first goes through the per-frame image path (including its factor-56 quirk), is
stacked/padded temporally, then receives the final video smart-resize plan at
factor 28 and this TorchVision-compatible kernel. Encoded-video decoding,
sampling, temporal padding, metadata, and FPS behavior remain Phase E work.

## Reproduction

From the repository root:

```console
make resize-conformance
make resize-conformance-regenerate
make resize-report-macos
```

The Linux x86_64 emulated evidence was produced with:

```console
docker run --rm --platform linux/amd64 \
  -e CARGO_TARGET_DIR=/tmp/qwen-mm-target \
  -v "$PWD:/work" -w /work rust:1.97.1-bookworm \
  cargo run --locked -p qwen-mm-core --example resize_stage_report -- \
  --output reference/resize/v1/results/linux-x86_64-emulated.json \
  --host-id linux-x86_64-emulated \
  --execution docker-desktop-qemu-emulation-on-macos-arm64
```

Regeneration requires the already-synced locked reference environment. Normal
Rust execution consumes only the committed authenticated artifacts.
