# Encoded image and prepared-RGB parity v1 decision record

Status: accepted implementation evidence for B6 on 2026-08-01.

## Decision

`qwen-mm-core::prepare_image_rgb8` is the Rust-only still-image preparation
boundary. It accepts encoded JPEG, PNG, or WebP bytes, or a caller-owned
strided RGB8 view, and returns the exact B3 geometry plus packed non-linear HWC
RGB8 bytes after the B5 resize. Production code does not invoke Python, Pillow,
Torch, OpenCV, a shell command, the network, or the filesystem.

The exact-pinned Rust codec stack is:

| Boundary | Version and features | Rationale |
| --- | --- | --- |
| JPEG | `libjpeg-turbo-rs=0.8.0`, `default-features=false`, `simd,std` | Its baseline, progressive, grayscale, and CMYK output was exact against the pinned Pillow/libjpeg-turbo oracle on the committed corpus and independent decoder probes. Strict warning handling plus a marker-aware terminal-EOI check rejects the truncations accepted by the codec's default surface. |
| PNG/WebP dispatch | `image=0.25.10`, `default-features=false`, `png,webp` | Restricts the general image crate to the two selected decoders; its JPEG feature is deliberately disabled. |
| PNG | `png=0.18.1` through the committed lockfile | Exact RGB/L/RGBA decode and exact prepared RGB on the corpus. |
| WebP | `image-webp=0.2.4` through the committed lockfile | Exact lossless RGB/RGBA decode and prepared RGB; lossy output meets the frozen one-byte bound. |

The rejected `image` JPEG path used zune-jpeg and produced maximum byte errors
of 2–3 on representative pinned Pillow JPEGs, with values outside the allowed
one-byte bound, and accepted tested truncations. Enabling it would therefore
weaken both parity and corruption handling. The selected dependency manifests
and `Cargo.lock` are authenticated in each host report so transitive decoder
versions cannot drift behind hard-coded report labels.

## Frozen semantics

- PNG accepts 8-bit `RGB`, `L`, and `RGBA`; JPEG accepts 8-bit `RGB`, `L`, and
  `CMYK`; WebP accepts `RGB` and `RGBA`. Palette, 16-bit, and `LA` PNG are
  `unsupported_media` under the accepted v1 envelope.
- `L` replicates into all three RGB channels. Direct `RGBA` is composited onto
  white with exact integer alpha arithmetic. PNG `tRNS` attached to source
  `L` or `RGB` is ignored during conversion, matching Pillow's source-mode
  behavior even when the Rust decoder expands it to alpha. CMYK and YCCK use
  the selected JPEG decoder's direct RGB conversion, gated against Pillow's
  pinned libjpeg-turbo output.
- EXIF orientation is ignored for values 1 through 8: preparation uses stored
  raster dimensions and pixels and never transposes.
- Animated WebP and APNG are rejected as `unsupported_media`. Recognized
  excluded codecs are also rejected without falling through to another
  decoder.
- A raw RGB view may include row padding. Its readable extent is
  `(height - 1) * stride + width * 3`; padding after the final row is not
  required and trailing bytes are ignored. Narrow stride, short extent, and
  zero dimensions are stable `media_geometry` failures.
- Checked byte, pixel, edge, stride, capacity, prepared-pixel, and materialized
  output limits run before allocation at their frozen precedence. Arithmetic
  overflow remains distinct from numeric resource excess. Encoded geometry is
  deferred until a recognized supported image has decoded successfully, so a
  corrupt image cannot masquerade as an aspect-ratio failure.

## Oracle and coverage

The self-authenticating fixture is
[`reference/media/v1/manifest.json`](../reference/media/v1/manifest.json). Its
generator validates Pillow 12.3.0, NumPy 2.4.6, qwen-vl-utils 0.0.14,
libjpeg-turbo 3.1.4.1, and libwebp 1.6.0 before writing output. The manifest
records the capture platform, Python version, immutable compatibility-manifest
hash, both profile fingerprints, generator hash, blob hashes, every case-slice
hash, and a canonical JSON self-hash.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `manifest.json` | 30,866 | `5c4a260ee611feafd8cf23f3bcf0622e479172dc15f928267e63c71fceea7627` |
| `encoded.bin` | 196,089 | `efaec0c1995bc84a48fdc5f462266c3d3f0e0f116fcf61d13581396bf56c67a8` |
| `prepared-rgb8.bin` | 1,038,336 | `8fdea5c4ca27959cd79543948287d3ae7b6cc55a3cbb3099af7c90220fd20aaa` |

Coverage includes RGB, grayscale, direct alpha boundary values, ignored PNG
`tRNS`, CMYK APP14 transform 0, YCCK APP14 transform 2, baseline and progressive
JPEG, EXIF orientations 1–8, PNG, lossy/lossless WebP, lossy VP8X alpha, no-op
and off-grid resizing, exact lossless WebP resize, accepted aspect-ratio
boundary 200 and rejected aspect ratio 201, malformed/truncated data, invalid
PNG/JPEG/WebP structural fields and CRC/extent collisions, missing JPEG EOI,
malicious edge/pixel headers, format mismatch, unsupported modes/codecs, APNG,
and animated WebP. Every case runs under both supported profiles. Lossless
cases carry a zero-byte bound, and both the always-on test and report runner
reject a manifest that attempts to weaken that bound.

The host reports also execute packed and padded raw RGB no-op cases with an
exactly sized final row plus packed and padded 65x97 off-grid resize cases. The
off-grid raw views are compared to the authenticated Pillow luma/RGB oracle,
not merely to each other.

All successful cases currently match the pinned Pillow prepared RGB bytes
exactly on both profiles: maximum absolute error, RMSE, p50/p90/p99, nonzero
count, count over the bound, maximum byte ULP distance, and first difference
are all zero/absent. JPEG and lossy WebP retain the accepted one-byte contract
allowance even though this corpus is exact.

## Platform evidence

| Report | Execution | Result |
| --- | --- | --- |
| [`macos-arm64-native.json`](../reference/media/v1/results/macos-arm64-native.json) | Native macOS arm64, rustc 1.97.1 | Both profiles pass every output and categorized-error case |
| [`linux-x86_64-emulated.json`](../reference/media/v1/results/linux-x86_64-emulated.json) | Linux x86_64 binary under Docker Desktop QEMU emulation on macOS arm64, rustc 1.97.1 | Both profiles pass every output and categorized-error case |

The Linux report is separate x86_64 code-generation and execution evidence,
but it is explicitly emulated. The oracle was not regenerated on Linux, this
is not native Linux hardware evidence, and it makes no performance claim.

## Reproduction

From the repository root:

```console
make media-conformance
make media-conformance-regenerate
make media-report-macos
```

The Linux x86_64 emulated evidence is produced with:

```console
docker run --rm --platform linux/amd64 \
  -e CARGO_TARGET_DIR=/tmp/qwen-mm-target \
  -v "$PWD:/work" -w /work rust:1.97.1-bookworm \
  cargo run --locked -p qwen-mm-core --example media_stage_report -- \
  --output reference/media/v1/results/linux-x86_64-emulated.json \
  --host-id linux-x86_64-emulated \
  --execution docker-desktop-qemu-emulation-on-macos-arm64
```

Regeneration is reference tooling only. The committed Rust test consumes the
authenticated blobs directly and requires no Python runtime.
