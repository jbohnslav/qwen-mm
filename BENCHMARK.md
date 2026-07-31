# Initial Python preprocessing baseline

This is the first directional measurement of the current Qwen multimodal preprocessing path. It is deliberately small and machine-specific; it is not the final performance suite.

## Headline result

| Processor configuration | `image1` median | `image1` p90 | `image24` median | `image24` p90 | `image24` throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-VL-8B-Instruct | 9.025 ms | 9.146 ms | 221.969 ms | 223.244 ms | 108.1 images/s |
| Qwen3.5-9B | 8.978 ms | 9.100 ms | 223.082 ms | 224.719 ms | 107.6 images/s |

On this machine, the warm synchronous Python path takes about **223 ms for 24 images**, or 9.25–9.30 ms per image in this case.

## Machine and environment

The benchmark ran on a Mac mini:

- Mac mini `Mac16,10`
- Apple M4, 10 CPU cores: 4 performance and 6 efficiency
- 24 GB unified memory
- arm64 macOS 26.5.2, build 25F84
- CPython 3.11.15
- Transformers 5.14.1
- Tokenizers 0.22.2
- Qwen VL Utils 0.0.14
- Torch 2.13.0 and TorchVision 0.28.0
- Pillow 12.3.0 and NumPy 2.4.6

Both models resolved to `Qwen3VLProcessor` with `Qwen2VLImageProcessor` using the TorchVision backend. Torch reported four intra-op threads and ten inter-op threads. No thread-count environment variables were set.

The model repositories were pinned to:

- `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`

Only processor and tokenizer artifacts were downloaded. Model weights were not loaded.

## Timed work

The headline case is one conversation containing 24 distinct, in-memory 1023×767 RGB JPEGs and a short text instruction. The images are deliberately off the model's spatial alignment boundary and become 1024×768 processor inputs. The single-image case uses the first image from the same corpus.

Timing starts with encoded bytes and a structured conversation. Each iteration creates fresh `BytesIO` and PIL objects, renders the chat template, calls Qwen VL Utils, and calls the Hugging Face processor with `do_resize=False`. Timing stops after CPU NumPy arrays are fully materialized.

Processor initialization, imports, fixture filesystem reads, network access, model execution, GPU transfer, garbage collection, and output hashing are excluded. Each result follows three warm-up iterations with ten measured iterations.

The 24-image case consumes 6,376,486 encoded input bytes and materializes a 432 MiB `float32` pixel buffer with shape `[73728, 1536]`. Every image produces `image_grid_thw = [1, 48, 64]`. Qwen3-VL produces 18,493 input tokens and Qwen3.5 produces 18,495 because their chat templates differ.

## Stage breakdown

Median latency for `image24`:

| Stage | Qwen3-VL-8B | Qwen3.5-9B |
| --- | ---: | ---: |
| Bind fresh PIL images and messages | 0.250 ms | 0.248 ms |
| Render chat template | 0.096 ms | 0.122 ms |
| Qwen VL Utils decode, orient, and resize | 142.377 ms | 142.731 ms |
| Hugging Face tokenize, normalize, grid, and patchify | 79.356 ms | 79.687 ms |
| Complete operation | 221.969 ms | 223.082 ms |

Qwen VL Utils accounts for about 64% of the measured latency, while final Hugging Face processing accounts for about 36%. Warm chat rendering and message binding are negligible in this case. The two processor configurations produce byte-identical pixel buffers and grids; only their token arrays differ.

## Interpretation and limits

This establishes the first comparison point for the Rust implementation: it must consume the same bytes and messages and produce conformant outputs before its time is compared with roughly 223 ms.

The result does not measure vLLM admission, event-loop blocking, GIL contention among concurrent requests, Ray, storage, URLs, video decoding, GPU utilization, memory high-water marks, cold starts, or model inference. The inputs are deterministic synthetic JPEGs rather than a representative image corpus. Consequently, this number describes this exact warm CPU path on this Mac mini and should not be treated as a universal Qwen preprocessing latency.

## Reproduce

From the repository root:

```bash
uv sync --locked --inexact --package qwen-mm-reference
uv run --locked --no-sync --package qwen-mm-reference python -m qwen_mm_reference.fixtures verify
uv run --locked --no-sync --package qwen-mm-reference python -m qwen_mm_reference.bench \
  --models qwen3-vl-8b,qwen3.5-9b \
  --cases image1,image24 \
  --warmups 3 \
  --iterations 10 \
  --output results/local.json
```

The exact run is preserved in [reference/results/2026-07-31-mac-mini-m4.json](reference/results/2026-07-31-mac-mini-m4.json). Fixture provenance and hashes are in [fixtures/baseline/manifest.json](fixtures/baseline/manifest.json).
