# Development workspace

The workspace is deliberately small while the compatibility contract and
golden reference are being frozen. It establishes ownership boundaries and
repeatable checks without selecting image, tokenizer, or parallel-runtime
kernels ahead of their conformance work.

## Pinned baseline

| Component | Pin | Reason |
| --- | --- | --- |
| Rust | `1.97.1` | Current stable patch on 2026-07-31; includes Cargo, rustfmt, and Clippy as one reproducible toolchain. |
| Edition | `2024` | Current Rust edition; the toolchain is new enough to support it directly. |
| PyO3 | `0.29.0` | Current stable release in the live Cargo index when bootstrapped; exact-pinned in `Cargo.lock`, with the Python 3.11 stable ABI. |
| Maturin | `1.14.1` | Current stable wheel builder when bootstrapped; exact-pinned in `pyproject.toml` and the smoke command. |

`qwen-mm-core` has no third-party dependencies. In particular it builds without
Python, Torch, OpenCV, vLLM, or network access. `qwen-mm-python` is a separate
PyO3 extension package; its `extension-module` feature is enabled only by
Maturin so ordinary Rust tests can link against the host Python normally.

## Commands

From the repository root:

```bash
make core-check       # build qwen-mm-core locked and offline, with no Python dependency
make rust-check       # format, Clippy -D warnings, unit tests, doc tests, offline core build
make reference-smoke  # sync the existing locked reference environment and verify fixtures
make wheel-smoke      # build, install, import, and exercise a fresh development wheel
make check            # run all of the above
```

The first Cargo command installs the exact toolchain declared in
`rust-toolchain.toml` when rustup is available. Cargo commands use the committed
lock file; the final core build is explicitly offline. Workspace tests resolve
Python 3.11 through `uv` and pass that interpreter to PyO3 explicitly, so they
do not accidentally bind to an older system Python. Set `PYO3_PYTHON` to an
explicit Python 3.11 executable to override that discovery in managed CI or a
custom development environment.

## Ownership and future layout

- `crates/qwen-mm-core/`: processor semantics and dependency-light Rust APIs.
- `crates/qwen-mm-python/`: PyO3 conversions, Python package metadata, and wheel
  tests. Processor algorithms do not belong here.
- `profiles/`: future versioned compatibility manifests and pinned processor
  assets. Large or licensed upstream assets should be addressed by their hash,
  not copied casually.
- `fixtures/`: deterministic checked-in conformance inputs and manifests.
- `reference/`: the pinned Python oracle, exporters, and comparison tooling.
- `integrations/`: future external-system adapters such as the validated vLLM
  prepared-pixel seam.
- `benchmarks/`: future end-to-end paired benchmark scenarios and reports.
  Narrow crate-level microbenchmarks may live under the owning crate's
  `benches/` directory once a ticket identifies a meaningful kernel boundary.

CI is a correctness gate. Shared runners do not enforce performance thresholds;
the roadmap reserves those gates for dedicated benchmark hosts.
