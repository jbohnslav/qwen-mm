# qwen-mm 0.1.0 candidate verification

Verified 2026-09-05. Ticket `e8d1` prepares artifacts; public release remains
subject to the separate usability/release approvals in `7544` and `57e6`.
Nothing was tagged, uploaded to a package index, or publicly released.

## Exact candidate

Both native platforms built Git commit
`57463587b335cbf54915162adcfc7d0238d170fd`,
tree `c32e11074aafa1961be6fe998c2075dbb9eeac79`.
This report and its evidence were committed afterward as an evidence-only
update. The shipping source remains that candidate commit, not the later
recordkeeping commit. Both runs rechecked source currentness at completion.

The release has Apache-2.0 licensing, package metadata, an install/support
guide, release notes, a native manual CI workflow, and a reproducible runner.
See the [release procedure](../../../docs/releasing-v0.1.md),
[install guide](../../../docs/install-v0.1.md), and
[changelog](../../../CHANGELOG.md).

## Artifacts and hosts

| Native host | Wheel | SHA-256 |
| --- | --- | --- |
| macOS 26.5.2 ARM64 | `qwen_mm-0.1.0-cp311-abi3-macosx_11_0_arm64.whl` | `0a552b3376a8d02e43a546ce99bfba01d48d71783fe1fc32413a5393a5e5b918` |
| Debian 12 / glibc 2.36 x86_64 | `qwen_mm-0.1.0-cp311-abi3-manylinux_2_34_x86_64.whl` | `7a6ff3a7b97bd5435f5add33f7f1d1b82e30c382d2d489f675df4b24f9cdbaf8` |

Both use CPython 3.11.15, Cargo/Rust 1.97.1, uv 0.11.29, and locked Maturin
1.14.1. Each host built twice in independent Cargo target directories with
fixed `SOURCE_DATE_EPOCH=1788644851` and source-path
remapping; both wheel SHA-256 values matched on each host. The Linux wheel's
binary glibc floor is 2.34; the exercised host is glibc 2.36. macOS has a binary
deployment floor of 11.0, with the tested OS recorded above. These floors do
not imply a test on every OS version.

Linux ran natively on the existing NUC in the locally retained prepared image
`sha256:cb56a019011ab40da62bcf321c23932e18ab99d68058dc39682b5cf7f5bed754`,
derived from the existing pinned-tool Debian image
`sha256:35b8b35192ffdabf7940c2cc42b2c4016a54784f4f6a6b94594e724ef57b5a98`.
No paid benchmark compute was allocated. The GitHub workflow is provided for
reproduction but was not dispatched: this checkout has no configured remote.

Selected wheel files and complete original run directories remain locally at:

- [macOS wheel](../../../dist/v0.1.0-macos-arm64/build-1/qwen_mm-0.1.0-cp311-abi3-macosx_11_0_arm64.whl)
- [Linux wheel](../../../dist/v0.1.0-linux-x86_64/build-1/qwen_mm-0.1.0-cp311-abi3-manylinux_2_34_x86_64.whl)

Wheels are intentionally retained under ignored `dist/`, not committed as
source. Preserve them when transferring this release candidate. Historical
absolute paths in the Linux manifest describe `/release` inside its build
container; the local copy above has been verified against every manifest hash.

## Verification

Both manifests are `passed`: all 21 recorded command steps succeeded.

| Check | macOS ARM64 | Linux x86_64 |
| --- | --- | --- |
| Two independent release builds, identical bytes | PASS | PASS |
| Metadata, license, Python facade contents, audited platform tag | PASS | PASS |
| Fresh production-wheel install with declared runtime dependencies | PASS | PASS |
| Smoke, public docs, construction/media/usability regressions | PASS | PASS |
| Nine production binding/ownership tests; private test hook absent | PASS | PASS |
| Exact Markdown Rosetta suite, both supported profiles | PASS | PASS |
| All 43 public installed-wheel resize-v2 geometries | PASS | PASS |
| Full `make check` (Rust, reference tests, fixtures, conformance, wheel smoke) | PASS | PASS |
| All 105 script unit tests | PASS | PASS |
| Separate full binding suite including instrumented GIL-release check | PASS | PASS |
| Loopback-only `uv publish --dry-run` | PASS | PASS |

Rosetta retains 25 inventory entries: every required entry passes; optional
local-model generation is explicitly unexecuted in this release run. The
previous local model-consumer and hosted witnesses retain their original
provenance. The resize report also identifies two non-public direct-core
geometries whose bytes are historical; only 43 public cases are fresh wheel
captures. Their frozen thresholds were not changed.

The historical legacy Phase C v1 comparison remains **270 pass / 20 fail** on
both compared wheels. The strict D4 performance certificate remains **MISS**.
These artifacts establish the narrowed still-image/text functional envelope;
they make no fresh speed, memory, Qwen3-VL inference, video, Windows, additional
Python/model, or production vLLM/SGLang claim.

## Retained evidence and approval commands

- [macOS manifest](macos-arm64/manifest.json), [Rosetta](macos-arm64/rosetta.json),
  [resize](macos-arm64/resize.json), [captured arrays](macos-arm64/resize.zip),
  [all command logs and oracle requirements](macos-arm64/verification-logs.zip).
- [Linux manifest](linux-x86_64/manifest.json), [Rosetta](linux-x86_64/rosetta.json),
  [resize](linux-x86_64/resize.json), [captured arrays](linux-x86_64/resize.zip),
  [all command logs and oracle requirements](linux-x86_64/verification-logs.zip).

Every original retained file was rehashed against its manifest, and each
Rosetta witness was checked against its wheel hash and current document/runner
hashes. Each resize capture is bound to the same shipping wheel. The combined
publish rehearsal is retained in [publish-dry-run.log](publish-dry-run.log).
uv still prints `Uploading` during dry-run; the command includes `--dry-run`
and succeeds against a closed loopback port. It contacts no public index and does not validate PyPI project ownership or
credentials. uv rejects `publish --offline`, so dry-run uses a closed loopback
endpoint instead.

From the repository root, repeat the exact safe rehearsal:

```sh
uv publish --dry-run --no-config --trusted-publishing never \
  --publish-url http://127.0.0.1:9/legacy/ \
  dist/v0.1.0-macos-arm64/build-1/qwen_mm-0.1.0-cp311-abi3-macosx_11_0_arm64.whl \
  dist/v0.1.0-linux-x86_64/build-1/qwen_mm-0.1.0-cp311-abi3-manylinux_2_34_x86_64.whl
```

After the separate release approvals and remote/index setup, the exact cut
commands are:

```sh
git tag -a v0.1.0 57463587b335cbf54915162adcfc7d0238d170fd -m 'qwen-mm 0.1.0'
git push origin v0.1.0
uv publish --no-config --trusted-publishing never \
  dist/v0.1.0-macos-arm64/build-1/qwen_mm-0.1.0-cp311-abi3-macosx_11_0_arm64.whl \
  dist/v0.1.0-linux-x86_64/build-1/qwen_mm-0.1.0-cp311-abi3-manylinux_2_34_x86_64.whl
```

Supply credentials outside the command/log and verify wheel hashes first.
The [release procedure](../../../docs/releasing-v0.1.md) includes fresh index
installs and rollback: discard an unpublished candidate, or yank a defective
published release and issue a new version without moving a public tag.
[uv publishing documentation](https://docs.astral.sh/uv/guides/package/) and
[PyPI yanking documentation](https://docs.pypi.org/project-management/yanking/)
were checked while preparing the procedure. No publication approval is inferred
from completing this mechanical ticket.
