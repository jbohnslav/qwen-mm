# Install and support: v0.1.0

This candidate is not yet published. Obtain the native wheel and matching
`manifest.json` from the candidate bundle described in
[the release procedure](releasing-v0.1.md). Verify the wheel's SHA-256 against
`artifact.sha256` before installing into a fresh environment:

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python /path/to/qwen_mm-0.1.0-cp311-abi3-PLATFORM.whl
.venv/bin/python -c 'from qwen_mm import Processor; print(Processor.supported_profiles())'
```

Use the actual wheel filename in place of `PLATFORM`. Once the release is
approved and published, use `uv pip install --python .venv/bin/python
qwen-mm==0.1.0` or `python3.11 -m pip install qwen-mm==0.1.0`.

| Dimension | v0.1 contract |
| --- | --- |
| Python | CPython 3.11 only (`>=3.11,<3.12`); the `abi3` filename does not extend this support claim |
| macOS | Native Apple Silicon ARM64; wheel deployment floor 11.0; exercised on the macOS version recorded in the manifest |
| Linux | Native x86_64 with glibc; required glibc floor is encoded in the audited `manylinux` filename; tested Debian 12/glibc 2.36 |
| Dependencies | `numpy>=2.3.5,<3` (reference lock: 2.4.6; vLLM environment: 2.3.5), `huggingface-hub==1.26.0`, and their resolved transitive dependencies |
| Optional interoperability | Torch must be installed separately for `return_tensors="pt"`; reference tests use the versions in `uv.lock` |
| Qwen3 | `Qwen/Qwen3-VL-8B-Instruct`, revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| Qwen3.5 | `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Media | Text, JPEG/PNG/WebP bytes, uint8 HWC RGB arrays, paths, file/HTTP(S) URLs, image data URIs |
| Unsupported | Windows, Intel macOS, Linux ARM/musl, other Python versions/models, video/frame lists, SGLang adapters, general production certification |

A wheel's minimum OS tag describes binary compatibility, not a claim that every
OS release at or above that floor has been tested. ARM and Linux native-host
versions and exact tools are retained in each manifest. Wheels are the only
v0.1 distribution artifacts; no source distribution is selected for this cut.
Developers can build from the candidate Git commit with Rust 1.97.1,
Python 3.11, uv 0.11.29, the committed lockfiles, and the release runner.

The optional `qwen-mm-vllm==0.1.0` wheel is a separate Apache-2.0 package for
Linux x86_64, CPython 3.11, vLLM 0.23.0 and Transformers 5.14.1. Install it
alongside the matching core wheel; it is not required for standalone qwen-mm.
Its universal Python wheel tag does not extend the supported server platforms.
See the [installation and launch guide](../integrations/vllm/README.md).
Preprocessing tests cover both profiles; real GPU serving was verified with
Qwen3.5-9B on an L40S. Approximate resized pixels can change generated text.
The retained GPU measurements are scoped evidence, not general production
certification or a new throughput claim.

Follow the [README quickstart](../README.md#quickstart) or the
[executable Rosetta Stone](official-example-rosetta-stone-v0.1.md).
`from_pretrained` downloads only pinned processor/tokenizer files on first use;
use `local_files_only=True` after populating the cache for offline use. Image
URLs still require network access when used. Supply bytes or local paths for
fully offline processing. This package does not load model weights or execute
a model, and its Apache-2.0 license does not replace upstream asset licenses.
