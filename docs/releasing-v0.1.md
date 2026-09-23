# Reproduce and rehearse the v0.1.0 release

The source repository is public at [jbohnslav/qwen-mm](https://github.com/jbohnslav/qwen-mm).
The maintainer authorized initial publication on September 23, 2026 (ticket `769d`).
GitHub Actions builds and verifies both native platforms before publishing; candidate
manifests bind every wheel to its source commit. Changes to shipping source require
fresh verification. No model weights or paid benchmark compute are needed.

## Native build and verification

Use CPython 3.11, uv 0.11.29, Rust 1.97.1 from `rust-toolchain.toml`, and the
committed Cargo/uv lockfiles. On native macOS ARM64 and native Linux x86_64:

```sh
uv sync --locked --all-packages
uv run --locked python scripts/release.py --output dist/candidate
```

Run from the exact candidate commit with no uncommitted shipping files or
untracked files. Ticket/worklog-only edits are allowed. The output directory
must be new. Linux may run in a native x86_64 Debian 12 container with these
tools; set `RELEASE_BUILD_ENVIRONMENT` to the image digest to retain its identity.
Do not use emulation. The runner's `--compatibility pypi` enforces an audited
PyPI-compatible platform tag; a generic `linux_x86_64` wheel is not accepted.

The runner builds twice with independent Cargo target directories, fixed
`SOURCE_DATE_EPOCH`, and source-path remapping, and requires identical wheel
SHA-256 values on that host. This establishes repeatability under the recorded
build environment; it does not promise identical wheels across different
compilers or operating systems. The shipping wheel contains no test hooks.

A fresh Python environment installs the production wheel and its declared
runtime dependencies first. Smoke, public examples, facade regressions, and
binding/ownership tests run there. The one instrumentation-only GIL test runs
in the separate full binding suite against a test-hooks build of the same
source. The oracle's hash-locked dependencies are then installed for exact
Markdown Rosetta comparisons on both profiles and all 43 public resize-v2
geometries. Full `make check`, all script unit tests, and the full hooked
binding suite also run. No model weights or paid hosted APIs are required.

The runner also builds the optional `qwen-mm-vllm` wheel twice using separate
build environments and the hash-locked `integrations/vllm/build-requirements.txt`.
It verifies matching bytes, exact Python sources, Apache-2.0 license, dependency
pins, and both vLLM entry points. Its manifest includes `plugin_artifact`.
Select just one plugin wheel for publication; its hash must match on both hosts.

On Linux, additionally verify both selected wheels in a fresh vLLM environment:

```sh
uv run --locked python scripts/release_vllm.py \
  --core-wheel dist/candidate/build-1/qwen_mm-0.1.0-cp311-abi3-manylinux_2_34_x86_64.whl \
  --plugin-wheel dist/candidate/plugin-build-1/qwen_mm_vllm-0.1.0-py3-none-any.whl \
  --output dist/candidate-vllm
```

Use the actual audited core filename if its glibc tag differs. This check
resolves the normal published GPU dependency set without installing it, then
installs both candidate wheels with prebuilt CPU vLLM and NumPy 2.3.5 in a
fresh environment. CPU versions are constrained to the retained 45e5 runtime
snapshot; installation and dependency checks must succeed without bypassing
requirements. Tests exercise installed plugin entry points, both-profile native
preprocessing/caching, public API behavior, and the HTTP input schema. A CPU
wheel is used only for this verification; the plugin dependency remains the
normal `vllm==0.23.0` requirement. Previous L40S generation evidence retains its
original artifact provenance. No new GPU run or performance claim is implied.
The verification environment stays under ignored `dist/` for inspection; retain
its logs, resolved dependencies and manifest, not the environment itself.

Each output directory retains both core/plugin build copies, command logs, exported oracle
requirements, Rosetta JSON, resize JSON/ZIP, and `manifest.json`. The manifest
records the exact candidate Git commit/tree, tools, host, every command result,
and SHA-256 of every retained file. `status: passed` is written only after all
checks and a final source-currentness check succeed. Keep the complete output
bundle; the compact checked-in evidence points to the retained local bundle.

Currentness applies to this fresh functional evidence. Historical Phase C/D
reports, the ARM model-consumer witness, and hosted reports retain their
original provenance. In particular the strict D4 MISS, legacy v1's unchanged
20 failures, and the two non-public resize geometries are not fresh PASS claims.
The v0.1 artifact makes no new performance claim. See
[the changelog](../CHANGELOG.md) and [support matrix](install-v0.1.md).

## GitHub Actions and PyPI

`ci.yml` runs repository hooks, Rust tests, Python smoke checks, and installed
Rosetta examples on pushes and pull requests. `release-candidate.yml` is a manual
and reusable workflow that runs the complete native checks above. It also runs
on master when release workflows or publishing scripts change. Its Linux job
also runs the installed vLLM verification. Artifacts are retained for 30 days.

Before the first upload, register a pending PyPI trusted publisher for each of
`qwen-mm` and `qwen-mm-vllm` with these exact values:

- GitHub owner: `jbohnslav`
- Repository: `qwen-mm`
- Workflow: `release.yml`
- Environment: `pypi`

Create the corresponding GitHub environment. Publishing uses short-lived OIDC
credentials; no stored PyPI API token is required. The publishing job alone has
`id-token: write`. The GitHub release job alone has `contents: write`.

Run the release-candidate workflow on the intended commit and inspect both native
jobs before tagging. Then publish the same verified commit:

```sh
git tag -a v0.1.0 CANDIDATE_COMMIT -m 'qwen-mm 0.1.0'
git push origin v0.1.0
```

The tag triggers `release.yml`, which repeats native verification, checks manifest
commit/version/status and wheel digests, and requires identical plugin wheels on
both hosts. It selects two core wheels and one plugin wheel. The Linux vLLM check
must refer to exactly those selected artifacts. A mismatched tag/version fails.

The workflow creates a GitHub release containing wheels, manifests, and SHA256SUMS,
then publishes the wheels to PyPI using trusted publishing. Finally, both native
platforms install the core package from PyPI and run its smoke test, and all three
index artifact hashes are compared with the verified wheels. Manually dispatching
`release.yml` on an ordinary branch verifies candidates without publishing.

The release runner also rehearses uploads with `uv publish --dry-run` against a
closed loopback endpoint. This validates package upload plans without credentials
or public writes; it does not prove PyPI publisher configuration.

## Failed or partial publication

Before upload, discard failed candidates and correct the source. Do not move an
already public tag or replace an uploaded version. For a defective published
release, yank it with a reason and issue a corrected version. If publication is
partially complete, rerun using the same verified artifacts: existing GitHub
checksums must match, and PyPI skips existing files. The final index hash check
rejects mismatches. A rebuild with different hashes needs investigation, not a
checksum override. The native wheel filename declares its actual glibc floor.
