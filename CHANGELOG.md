# Changelog

## 0.1.0 — release candidate (unreleased)

First Python release candidate for CPU text and still-image preprocessing with
`Qwen/Qwen3-VL-8B-Instruct` and `Qwen/Qwen3.5-9B`, at the immutable revisions
listed in the [support matrix](docs/install-v0.1.md).

- One `Processor.prepare` or `prepare_batch` call renders chat, tokenizes,
  decodes JPEG/PNG/WebP or raw RGB, resizes, normalizes, and packs model arrays.
- Familiar `from_pretrained` construction, pinned processor downloads, paths,
  URLs, data URIs, heterogeneous batches, pixel controls, thinking, and tools.
- NumPy output by default; optional Torch views/device transfer and native
  `decode`/`batch_decode`. Model weights and inference are caller-owned.
- Optional `qwen-mm-vllm` package for still-image preprocessing inside prebuilt
  vLLM 0.23.0 with Transformers 5.14.1, using ordinary image URL requests.
  Native preprocessing covers both profiles; Qwen3.5-9B GPU serving was verified
  on Modal. See the [measurements and limitations](integrations/vllm/evidence/45e5/gpu-20260907/README.md).
- Core NumPy requirement widened to `>=2.3.5,<3` for vLLM compatibility;
  standalone reference tests retain NumPy 2.4.6 and vLLM uses 2.3.5.
- Native macOS ARM64 and Linux x86_64 wheels for CPython 3.11; Apache-2.0 license.

The composed qwen-vl-utils default minimum is 4096 pixels. Direct Transformers
examples require `min_pixels=65536`. Resizing follows the accepted resize-v2
quality contract, not bitwise Pillow equality. The historical Phase C v1
comparison remains 270 pass / 20 fail; the strict D4 performance certificate
remains a MISS. These results are not relabeled by the release.

Video (including frame lists), SGLang integration, general production certification, Windows,
other architectures/Python versions, model loading, and generation are outside
the v0.1 support envelope. No renewed speed or memory claim accompanies these
artifacts. See [executable verification](docs/rosetta-verification.md) and the
[release procedure](docs/releasing-v0.1.md).
