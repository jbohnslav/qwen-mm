# qwen-mm Python package

This package is the thin PyO3 boundary around `qwen-mm-core`. Processor logic
belongs in the core crate; Python-specific conversion and packaging code belongs
here.

Build and verify a development wheel from the repository root with:

```bash
make wheel-smoke
```
