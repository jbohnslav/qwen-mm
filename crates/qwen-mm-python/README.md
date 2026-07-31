# qwen-mm Python binding crate

This crate is the thin PyO3 boundary around `qwen-mm-core`. The publishable
Python project metadata and Maturin configuration live in the repository-root
`pyproject.toml`; Python-specific conversion and native module code live here.
Processor logic belongs in the core crate.

Build and verify a development wheel from the repository root with:

```bash
make wheel-smoke
```
