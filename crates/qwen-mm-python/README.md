# qwen-mm Python binding crate

This crate is the thin PyO3 boundary around `qwen-mm-core`. The publishable
Python project metadata and Maturin configuration live in the repository-root
`pyproject.toml`; Python-specific conversion and native module code live here.
Processor logic belongs in the core crate.

Build and verify a development wheel from the repository root with:

```bash
make wheel-smoke
```

The public API is deliberately small:

```python
from qwen_mm import Processor

processor = Processor("qwen3-vl-8b", "/path/to/pinned/snapshot")
prepared = processor.prepare_batch(
    [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "input_index": 0},
                        {"type": "text", "text": "Describe the image."},
                    ],
                }
            ],
            "images": [{"data": jpeg_bytes, "format": "jpeg"}],
            "options": {"add_generation_prompt": True},
        }
    ]
)
model_inputs = prepared.arrays
adapter_metadata = prepared.metadata
```

`images` entries may also be aligned `numpy.uint8[H, W, 3]` arrays with packed
RGB pixels and an optional positive row-padding stride. Negative, overlapping,
and channel-sliced layouts are rejected. Inputs are validated and copied into
GIL-independent native ownership before processing. Returned arrays are exact
C-contiguous `int64`/`float32` outputs. Their Rust allocations are transferred
into NumPy without copying, so they remain valid after the request, its media,
and the processor are dropped. Adapter metadata is intentionally separate from
`.arrays`; iteration over `.arrays` yields only the official conditional
processor keys.

Run the installed-wheel image1/image24, ownership, GIL-release, and failure
suite (requires both pinned snapshots under `reference/.cache`) with:

```bash
make python-binding-test
```
