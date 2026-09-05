# Reproduce and rehearse the v0.1.0 release

This procedure prepares artifacts without publishing, tagging, or spending on
benchmark compute. Product approval remains ticket `7544`; final release
approval remains `57e6`. The candidate source commit and artifact hashes in the
retained release manifests identify the bits to approve. A later evidence-only
commit may record those manifests; it is not a replacement candidate source.
Any implementation, metadata, runner, or release-document change requires a new
candidate commit and a complete rerun on both platforms.

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

Each output directory retains both wheels, command logs, exported oracle
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

## Publishing rehearsal and approval commands

The runner uses uv's built-in offline dry run, which validates the upload plan
without sending package files or requiring credentials. Repeat it for both
selected wheels (one per platform, not their duplicate build copies):

```sh
uv publish --dry-run --offline --no-config --trusted-publishing never \
  dist/macos-arm64/build-1/*.whl dist/linux-x86_64/build-1/*.whl
```

This does not establish ownership of the PyPI project name, validate a token,
or test a live index. No remote is configured in this checkout; configure the
intended Git remote and package-index account before the actual cut.

Only after `7544`, `e8d1`, and the release-epic gates are satisfied and Jim
approves the recorded candidate, verify hashes against both manifests, then
execute the following with `CANDIDATE_COMMIT` set to their common source commit:

```sh
kd status --check
git tag -a v0.1.0 "$CANDIDATE_COMMIT" -m 'qwen-mm 0.1.0'
git push origin v0.1.0
uv publish --no-config --trusted-publishing never \
  dist/macos-arm64/build-1/*.whl dist/linux-x86_64/build-1/*.whl
```

Supply the PyPI token through `UV_PUBLISH_TOKEN` in the environment or use a
separately configured trusted publisher; never put credentials in evidence.
After publication, verify a new CPython 3.11 install of `qwen-mm==0.1.0` on both
native platforms and execute the public smoke/examples again.

Before upload, rollback is simply discarding the candidate bundle and deleting
an unpushed local tag with `git tag -d v0.1.0`. After upload, treat the release
as immutable: use the PyPI project's version controls to yank a defective
release with a reason, communicate the issue, and issue a corrected version.
Do not delete and try to reuse the same version or silently move a public tag.
If only one platform upload succeeds, preserve its hash and upload the already
verified missing wheel after diagnosing the error. Rehearsal does not upload,
so it needs no remote rollback.

The manual `release-candidate.yml` workflow mirrors the native runner and
retains artifacts. It cannot publish. Its Linux runner may require a newer
glibc floor than the Debian 12 local build; its filename and manifest are
authoritative for that bundle, and it must be separately approved if selected.
