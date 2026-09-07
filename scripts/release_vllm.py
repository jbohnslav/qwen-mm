"""Verify candidate wheels in a fresh Linux CPU vLLM environment; no model server."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from release import ROOT, UV, inspect_plugin_wheel, inspect_wheel, sha256, source_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-wheel", type=Path, required=True)
    parser.add_argument("--plugin-wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("requires native Linux x86_64")
    if output.exists() or not output.is_relative_to(ROOT / "dist"):
        parser.error("--output must be a new directory under dist/")
    source = source_identity()
    core, plugin = args.core_wheel.resolve(), args.plugin_wheel.resolve()
    report = {
        "source": source,
        "core_artifact": inspect_wheel(core),
        "plugin_artifact": inspect_plugin_wheel(plugin),
        "status": "running",
        "steps": [],
    }
    output.mkdir(parents=True)
    env = dict(os.environ, PYTHONNOUSERSITE="1", VLLM_PLUGINS="", OMP_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)

    def run(name, command):
        print(name, flush=True)
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        report["steps"].append({"name": name, "command": command, "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{name} failed: {(output / (name + '.log')).read_text()[-8000:]}")

    try:
        # Resolve ordinary published GPU dependencies as well as exercising CPU vLLM.
        requirements = output / "gpu-requirements.in"
        requirements.write_text(f"{core}\n{plugin}\n")
        run(
            "gpu-dependency-resolution",
            UV
            + [
                "pip",
                "compile",
                str(requirements),
                "--python-version",
                "3.11",
                "--only-binary",
                ":all:",
                "--output-file",
                str(output / "gpu-requirements.txt"),
            ],
        )
        run("create-environment", UV + ["venv", "--python", sys.executable, str(output / "venv")])
        python = str(output / "venv/bin/python")
        run(
            "install",
            UV
            + [
                "pip",
                "install",
                "--python",
                python,
                "--only-binary",
                ":all:",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cpu",
                "--index-strategy",
                "unsafe-best-match",
                "--constraint",
                "integrations/vllm/release-cpu-constraints.txt",
                str(core),
                str(plugin),
                "pytest",
            ],
        )
        run("dependency-check", UV + ["pip", "check", "--python", python])
        run("runtime-dependencies", UV + ["pip", "freeze", "--python", python])
        run(
            "installed-entry-points",
            [
                python,
                "-c",
                """
from importlib.metadata import distribution
from pathlib import Path
import qwen_mm, qwen_mm_vllm, numpy
assert numpy.__version__ == '2.3.5'
assert '/site-packages/' in str(Path(qwen_mm.__file__))
assert '/site-packages/' in str(Path(qwen_mm_vllm.__file__))
dist = distribution('qwen-mm-vllm')
entries = {ep.name: ep for ep in dist.entry_points if ep.group == 'vllm.general_plugins'}
assert set(entries) == {'qwen_mm_native_images', 'qwen_mm_serving_audit'}
for name, ep in entries.items():
    print(name, ep.value)
    ep.load()()
print('Installed wheel imports and plugin registration passed')
""",
            ],
        )
        run("integration-tests", [python, "-m", "pytest", "integrations/vllm/tests", "-q"])
        run("public-api", [python, "crates/qwen-mm-python/tests/usability.py"])
        run(
            "http-input-audit",
            [
                python,
                "integrations/vllm/scripts/audit_server_inputs.py",
                "--output",
                str(output / "http-input-audit.json"),
            ],
        )
        assert source_identity() == source
        report["status"] = "passed"
    finally:
        if report["status"] != "passed":
            report["status"] = "failed"
        report["files"] = {
            str(p.relative_to(output)): sha256(p) for p in output.iterdir() if p.is_file()
        }
        (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Plugin candidate verified: {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
