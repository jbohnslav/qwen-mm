# Contributing

Use CPython 3.11 and the Rust toolchain in `rust-toolchain.toml`:

```sh
make sync
make hooks
make check
```

Native Linux x86_64 and macOS ARM64 CI also runs installed-wheel examples.
Model processor assets may be downloaded from the pinned Hugging Face snapshots;
tests do not require model weights or paid inference services.

Keep token IDs and supported output contracts exact. Add a regression test for
behavior changes and run the relevant installed-wheel checks. Reproduce benchmark
claims with the existing harness and state the host, thread budget, and scope.
Update third-party notices after dependency changes with
`python scripts/third_party_licenses.py`.

Open a pull request against `master`. Contributions are licensed under the
project's Apache-2.0 license. See SECURITY.md for private vulnerability reports.
