# vLLM prepared-pixel seam prototype

This isolated package is the executable A6 proof for
[`docs/vllm-prepared-pixel-seam.md`](../../docs/vllm-prepared-pixel-seam.md).
It is intentionally not a root uv-workspace member: vLLM is platform-specific
and must not enter the framework-independent core lock.

## Pin

- vLLM `v0.23.0`
- commit `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`
- Python 3.11.15
- Torch 2.11.0
- Transformers 5.14.1

The Mac proof uses vLLM's documented empty-device development build. It imports
and runs Python input processing but cannot execute a model. The production
canary remains pinned to Linux x86_64 and one NVIDIA H100 80 GB.

## Reproduce on a non-Linux development host

```bash
git clone --depth 1 --branch v0.23.0 https://github.com/vllm-project/vllm.git /tmp/vllm-a6
git -C /tmp/vllm-a6 rev-parse HEAD

uv venv /tmp/vllm-a6-venv
VLLM_TARGET_DEVICE=empty uv pip install \
  --python /tmp/vllm-a6-venv/bin/python \
  -e /tmp/vllm-a6
uv pip install --python /tmp/vllm-a6-venv/bin/python \
  transformers==5.14.1 pytest
uv pip install --python /tmp/vllm-a6-venv/bin/python \
  --no-deps -e integrations/vllm

/tmp/vllm-a6-venv/bin/python \
  integrations/vllm/scripts/verify_vllm_pin.py /tmp/vllm-a6
/tmp/vllm-a6-venv/bin/python -m pytest -q integrations/vllm/tests
```

The vLLM build frontend may require its declared build dependencies when
`--no-build-isolation` is used. Normal build isolation installs them
automatically.

## Plugin activation

Installing this package creates the general-plugin entry point
`qwen_mm_prepared_pixels`. By default vLLM loads all general plugins. To select
only this plugin:

```bash
export VLLM_PLUGINS=qwen_mm_prepared_pixels
```

The prototype accepts prepared still images only. It is a seam proof, not the
final production adapter. It refuses any model/revision other than the two
frozen compatibility profiles, so vLLM must be launched with the exact
`--revision` recorded in the ADR.
