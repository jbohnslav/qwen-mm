# qwen-mm 0.1.0 candidate verification

Refreshed and verified 2026-09-07 after the optional vLLM integration.
The selected source commit is **`4c9f72a7123cd077421c20cc97ac815d13f0dcb9`**, tree
`bade9da7b12547832848761a382f04fcaefa1b87`. This report is an evidence-only update after verification;
it does not replace the selected source commit. Nothing was tagged or uploaded.

Download the [candidate bundle](../../../dist/qwen-mm-0.1.0-release-candidate.zip)
(SHA-256 `dec53092fb40b9b97c65685cf527dab6dd7337b82ef647cd29cb3825344df991`). It contains all three selected wheels, SHA256SUMS,
license, installation notes, and the three verification manifests.
The [September 5 candidate](previous-candidate-20260905.md) is historical:
its core wheels pin NumPy 2.4.6 and are not selected for this cut.

## Selected artifacts

| Wheel | SHA-256 |
| --- | --- |
| qwen_mm-0.1.0-cp311-abi3-macosx_11_0_arm64.whl | `fc7c47749efacbe549807dd49fd8632918c1de894a62648fb32ba9154b62646f` |
| qwen_mm-0.1.0-cp311-abi3-manylinux_2_34_x86_64.whl | `0d71e01364939ee3b0375510f86dd9a12c8b8afa0dfb793cfdc8843d8d59a242` |
| qwen_mm_vllm-0.1.0-py3-none-any.whl | `284f4f2a950534b30db157ab108bab45ba5a392385fd5d5d0bb055dbca152cce` |

Original build directories are `dist/v0.1.0-refresh-macos-arm64` and
`dist/v0.1.0-refresh-linux-x86_64`; supplemental plugin verification is in
`dist/v0.1.0-refresh-vllm`. The consolidated selection is
`dist/v0.1.0-refresh-selected`. Every retained original file was checked against
its manifest, and each wheel inside the download bundle was rehashed.
The optional plugin has identical bytes across both builds on both hosts.
No source distribution is selected.

## Verification and scope

Both native manifests report **passed**, with **29 successful steps each**:

- Two independent Cargo target directories produced identical core wheels.
- Two isolated, hash-locked Hatch environments produced identical plugin wheels.
- Exact source contents, license, Python/dependency metadata and entry points passed.
- Fresh standalone installs passed the public API, documentation, ownership and
  facade regressions; the production wheel contains no private GIL test hook.
- Exact Markdown Rosetta comparisons passed for both pinned profiles, and all
  43 public installed-wheel resize-v2 cases passed.
- Full `make check`, all **107 script tests**, and the separate full binding/GIL
  suite passed on both platforms.
- Core and optional-plugin loopback-only publishing rehearsals passed.

The supplemental Linux manifest reports **passed**, with **nine successful steps**.
Ordinary GPU vLLM dependencies resolve with both selected wheels. A separate
fresh environment then installed the selected wheels with prebuilt CPU vLLM
0.23.0+cpu, Transformers 5.14.1, Torch 2.11.0+cpu and NumPy 2.3.5. `uv pip check`,
both installed entry points, **all 40 integration tests**, public API checks and
the HTTP input audit passed. Dependencies were not bypassed. CPU runtime
versions are constrained in `integrations/vllm/release-cpu-constraints.txt`;
GPU resolution is recorded separately and is not a GPU execution test.

Hosts were native macOS macOS-26.6.2-arm64-arm-64bit ARM64 and
Debian 12/glibc 2.36 x86_64 on the existing NUC. Both used Python 3.11.15,
Rust 1.97.1, uv 0.11.29, Maturin 1.14.1 and Hatchling 1.32.0. Wheel floors are
macOS 11.0 and glibc 2.34; this does not claim tests on every compatible OS.
The NUC used prepared image
`sha256:cb56a019011ab40da62bcf321c23932e18ab99d68058dc39682b5cf7f5bed754`;
CPU vLLM additionally required Debian `libnuma1` 2.0.16-1. Install that package
before reproducing the supplemental CPU check in a minimal Debian container.
No paid compute or GPU allocation was used for this refresh. CI was not
dispatched because this checkout has no configured Git remote.

Core processing source and all four plugin module hashes match the retained
[Qwen3.5-9B L40S serving witness](../../../integrations/vllm/evidence/45e5/gpu-20260907/README.md).
That witness retains its earlier wheel provenance; no fresh GPU-generation
claim is made for these artifacts. The release makes no renewed speed or
memory claim. Approximate resized pixels can change generated text. Historical
legacy v1 remains 270 pass / 20 fail, strict D4 remains **MISS**, and video,
SGLang and general production certification remain outside this release.
Optional local generation and two non-public direct-core resize geometries
retain historical provenance; they are not relabeled as fresh wheel tests.

## Evidence and release commands

- [macOS manifest](refresh-20260907/macos-arm64/manifest.json),
  [Rosetta](refresh-20260907/macos-arm64/rosetta.json),
  [resize](refresh-20260907/macos-arm64/resize.json),
  [logs](refresh-20260907/macos-arm64/verification-logs.zip).
- [Linux manifest](refresh-20260907/linux-x86_64/manifest.json),
  [Rosetta](refresh-20260907/linux-x86_64/rosetta.json),
  [resize](refresh-20260907/linux-x86_64/resize.json),
  [logs](refresh-20260907/linux-x86_64/verification-logs.zip).
- [Packaged vLLM manifest](refresh-20260907/vllm/manifest.json),
  [install, dependency resolution and test logs](refresh-20260907/vllm/verification-logs.zip).
- [Processing source parity](refresh-20260907/source-parity.json),
  [CPU system dependency setup](refresh-20260907/cpu-system.log),
  [combined three-wheel publishing rehearsal](refresh-20260907/publish-dry-run.log).

Repeat the safe rehearsal from the repository root:

```sh
uv publish --dry-run --no-config --trusted-publishing never \
  --publish-url http://127.0.0.1:9/legacy/ dist/v0.1.0-refresh-selected/*.whl
```

The rehearsal validates upload plans without contacting a public index. It does
not verify ownership or credentials for either PyPI project. Before the actual
cut, configure the intended Git remote and both package-index destinations.
After the API approval in `7544` and final candidate approval in `57e6`, verify
SHA256SUMS and run:

```sh
kd status --check
git tag -a v0.1.0 4c9f72a7123cd077421c20cc97ac815d13f0dcb9 -m 'qwen-mm 0.1.0'
git push origin v0.1.0
uv publish --no-config --trusted-publishing never dist/v0.1.0-refresh-selected/*.whl
```

Supply credentials outside commands and logs. Follow the
[release procedure](../../../docs/releasing-v0.1.md) for post-publication fresh
installs and rollback, and the [support matrix](../../../docs/install-v0.1.md)
and [vLLM guide](../../../integrations/vllm/README.md) for supported use.
Human approval remains outstanding; the artifact ticket does not self-approve it.
