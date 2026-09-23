"""Select authenticated wheels from both native release jobs for publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {("Darwin", "arm64"): "macos-arm64", ("Linux", "x86_64"): "linux-x86_64"}


def verified_file(root: Path, subdirectory: str, artifact: dict) -> Path:
    name = artifact["name"]
    if Path(name).name != name or not name.endswith(".whl"):
        raise ValueError("invalid wheel filename")
    path = root / subdirectory / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
        raise ValueError(f"wheel hash mismatch: {name}")
    if path.stat().st_size != artifact["bytes"]:
        raise ValueError(f"wheel size mismatch: {name}")
    return path


def assemble(artifacts: Path, output: Path, *, commit: str, version: str, tag: str) -> None:
    if tag != f"v{version}":
        raise ValueError("release tag does not match package version")
    manifests = {}
    plugin_check = None
    for path in artifacts.rglob("manifest.json"):
        data = json.loads(path.read_text())
        if data.get("schema") == "qwen-mm-release-v1":
            target = TARGETS.get((data["host"]["system"], data["host"]["machine"]))
            if not target or target in manifests:
                raise ValueError("unexpected or duplicate native release target")
            manifests[target] = (path, data)
        elif "core_artifact" in data and "plugin_artifact" in data:
            if plugin_check is not None:
                raise ValueError("duplicate vLLM verification manifest")
            plugin_check = (path, data)
    if set(manifests) != set(TARGETS.values()) or plugin_check is None:
        raise ValueError("both native manifests and installed vLLM verification are required")
    selected = []
    plugin_hashes = set()
    for target, (path, data) in manifests.items():
        if (
            data["status"] != "passed"
            or data["source"]["commit"] != commit
            or data["version"] != version
        ):
            raise ValueError(f"failed, stale, or wrong-version release evidence: {target}")
        if not data.get("steps") or any(step["exit_code"] != 0 for step in data["steps"]):
            raise ValueError("release verification contains failed or missing steps")
        wheel = verified_file(path.parent, "build-1", data["artifact"])
        if not wheel.name.startswith(f"qwen_mm-{version}-cp311-abi3-"):
            raise ValueError("unexpected core wheel version or ABI")
        platform = "macosx_" if target == "macos-arm64" else "manylinux_"
        architecture = "arm64.whl" if target == "macos-arm64" else "x86_64.whl"
        if platform not in wheel.name or not wheel.name.endswith(architecture):
            raise ValueError("wheel platform does not match verified host")
        selected.append(wheel)
        plugin = verified_file(path.parent, "plugin-build-1", data["plugin_artifact"])
        if plugin.name != f"qwen_mm_vllm-{version}-py3-none-any.whl":
            raise ValueError("unexpected plugin wheel")
        plugin_hashes.add(data["plugin_artifact"]["sha256"])
        if target == "linux-x86_64":
            selected.append(plugin)
    if len(plugin_hashes) != 1:
        raise ValueError("plugin differs between hosts")
    check_path, check = plugin_check
    linux = manifests["linux-x86_64"][1]
    if (
        check["status"] != "passed"
        or check["source"]["commit"] != commit
        or check["core_artifact"] != linux["artifact"]
        or check["plugin_artifact"] != linux["plugin_artifact"]
    ):
        raise ValueError("vLLM check is not bound to the selected wheels")
    if not check.get("steps") or any(step["exit_code"] != 0 for step in check["steps"]):
        raise ValueError("vLLM verification contains failed or missing steps")
    (output / "dist").mkdir(parents=True, exist_ok=False)
    for source in selected:
        shutil.copyfile(source, output / "dist" / source.name)
    for target, (path, _) in manifests.items():
        shutil.copyfile(path, output / f"{target}-manifest.json")
    shutil.copyfile(check_path, output / "vllm-manifest.json")
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n"
            for p in sorted((output / "dist").iterdir())
        )
    )
    (output / "RELEASE_NOTES.md").write_text(
        f"qwen-mm {version}\n\nSource commit: `{commit}`.\n\n"
        "Apache-2.0. CPython 3.11 on native macOS ARM64 and manylinux x86_64. "
        "Supports the pinned Qwen3-VL and Qwen3.5 text/still-image profiles. "
        "The optional qwen-mm-vllm wheel targets the documented pinned Linux vLLM stack.\n\n"
        "Includes tokenizers 1.0.0-rc.2 with the documented Qwen3.5 reader patch. "
        "Native release jobs built each wheel twice and verified installed APIs, "
        "ownership, examples, and current resize conformance. "
        "Historical performance results are not fresh release certification.\n\n"
        "Install after PyPI publication: `python3.11 -m pip install qwen-mm=="
        f"{version}`. The attached core wheel also installs directly with pip.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assemble(
        args.artifacts,
        args.output,
        commit=os.environ["RELEASE_COMMIT"],
        version=version,
        tag=os.environ["RELEASE_TAG"],
    )


if __name__ == "__main__":
    main()
