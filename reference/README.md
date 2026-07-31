# Python reference benchmark

This directory contains the pinned Python reference used to time and export the
current Qwen multimodal preprocessing path. It loads processor artifacts only;
it does not download model weights or run generation.

Generate or verify the committed baseline images:

```bash
uv run python -m qwen_mm_reference.fixtures generate
uv run python -m qwen_mm_reference.fixtures verify
```

Run the small baseline on both processor configurations:

```bash
uv run python -m qwen_mm_reference.bench \
  --models qwen3-vl-8b,qwen3.5-9b \
  --cases image1,image24 \
  --output results/local.json
```

The timed boundary begins with in-memory encoded JPEG bytes and a structured
conversation. It ends with materialized CPU NumPy arrays. Network and disk I/O,
processor initialization, model execution, and output hashing are excluded.
