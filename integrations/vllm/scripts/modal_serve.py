"""Run the serving witness on one bounded L40S, using published vLLM wheels."""

import os
from pathlib import Path

import modal

app = modal.App("qwen-mm-vllm-45e5")
if modal.is_local():
    ROOT = Path(__file__).resolve().parents[3]
    WHEEL = ROOT / "dist/vllm-45e5/qwen_mm-0.1.0-cp311-abi3-linux_x86_64.whl"
    image = (
        modal.Image.from_registry(
            "nvidia/cuda@sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c",
            add_python="3.11",
        )
        .apt_install("libnuma1", "libgomp1")
        .pip_install(
            "vllm==0.23.0",
            "transformers==5.14.1",
            "numpy==2.3.5",
            extra_options="--only-binary=:all:",
        )
        .add_local_file(WHEEL, "/tmp/" + WHEEL.name, copy=True)
        .add_local_dir(
            Path(os.environ.get("QWEN_MM_VLLM_SOURCE", ROOT / "integrations/vllm")),
            "/integration",
            copy=True,
            ignore=["__pycache__", ".pytest_cache"],
        )
        .run_commands("pip install /tmp/" + WHEEL.name + " /integration")
        .env(
            {
                "HF_HOME": "/hf-cache",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                "CUDA_HOME": "/usr/local/cuda",
            }
        )
    )
else:
    image = None
cache = modal.Volume.from_name("qwen-mm-vllm-model-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="L40S",
    cpu=8,
    memory=32768,
    timeout=2400,
    volumes={"/hf-cache": cache},
    max_containers=1,
)
def experiment(probe_only: bool = False):
    import io
    import json
    import subprocess
    import tarfile

    Path("/results").mkdir(exist_ok=True)
    returncode = -1
    try:
        result = subprocess.run(
            ["python", "/integration/scripts/flashinfer_probe.py", "/results/flashinfer.json"],
            check=False,
        )
        returncode = result.returncode
        if returncode == 0 and not probe_only:
            result = subprocess.run(
                ["python", "/integration/scripts/serve_experiment.py", "/results"], check=False
            )
            returncode = result.returncode
    finally:
        Path("/results").mkdir(exist_ok=True)
        Path("/results/worker-exit.json").write_text(json.dumps({"returncode": returncode}))
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            archive.add("/results", arcname="results")
        cache.commit()
    return buffer.getvalue()


@app.local_entrypoint()
def main(output: str = "dist/vllm-45e5/serving.tar.gz", probe_only: bool = False):
    # Return logs on failure as well: the controller checks results.json.
    import io
    import json
    import tarfile

    data = experiment.remote(probe_only)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_bytes(data)
    print("Saved", output)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        status = json.load(archive.extractfile("results/worker-exit.json"))
    if status["returncode"] != 0:
        raise RuntimeError("Serving experiment failed; logs retained in " + output)
