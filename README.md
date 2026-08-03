# qwen-mm

`qwen-mm` is a Rust implementation of image and text preprocessing for Qwen3-VL and Qwen3.5-VL.

Current preprocessing pipelines can spread work across a Hugging Face processor, Qwen VL Utils, image resizing, video sampling, video decoders such as Decord or TorchCodec, and Python-based tokenization. This can duplicate work and introduce unnecessary overhead.

The goal of this project is to consolidate that pipeline in Rust, deduplicate and fuse operations where possible, and parallelize workloads such as processing many images without being limited by Python's global interpreter lock. It is intended to be a drop-in replacement for the corresponding Qwen3-VL and Qwen3.5-VL preprocessing components in Hugging Face Transformers and vLLM.

See [DESIGN.md](DESIGN.md) for the proposed architecture, compatibility contract, and conformance strategy.

The implementation-level compatibility source of truth is
[ADR 0001](docs/compatibility-v1.md), with machine-readable pinned profiles in
[`reference/compatibility/v1.json`](reference/compatibility/v1.json).

The first current-path measurement is documented in [BENCHMARK.md](BENCHMARK.md).

The dependency-ordered implementation plan and release gates are in [ROADMAP.md](ROADMAP.md).
