"""Build and verify native v0.1 wheels; never publish or tag a release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UV = [str(ROOT / "scripts/with-cargo.sh"), "uv"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def source_identity() -> dict:
    # Ticket state is deliberately outside the shipping source contract.
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", ".", ":!.kd"], cwd=ROOT, check=True)
    untracked = capture("git", "ls-files", "--others", "--exclude-standard")
    if untracked:
        raise RuntimeError(f"Commit or ignore untracked files before building: {untracked}")
    return {
        "commit": capture("git", "rev-parse", "HEAD"),
        "tree": capture("git", "rev-parse", "HEAD^{tree}"),
        "source_date_epoch": capture("git", "show", "-s", "--format=%ct", "HEAD"),
    }


def inspect_wheel(wheel: Path) -> dict:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata = BytesParser().parsebytes(
            archive.read(next(name for name in names if name.endswith(".dist-info/METADATA")))
        )
        assert metadata["Name"] == "qwen-mm"
        assert metadata["Version"] == "0.1.0"
        assert set(metadata["Requires-Python"].split(",")) == {">=3.11", "<3.12"}
        assert metadata["License-Expression"] == "Apache-2.0"
        assert any(name.endswith("/licenses/LICENSE") for name in names)
        assert not any(".cache/" in name or ".kd/" in name for name in names)
        assert "linux_x86_64.whl" not in wheel.name or "manylinux" in wheel.name
        for path in (ROOT / "crates/qwen-mm-python/python/qwen_mm").glob("*.py"):
            assert archive.read(f"qwen_mm/{path.name}") == path.read_bytes()
        return {"name": wheel.name, "sha256": sha256(wheel), "bytes": wheel.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory under dist/")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "dist") or output.exists():
        parser.error("--output must be a new directory under the repository's ignored dist/")
    host = (platform.system(), platform.machine())
    if host not in {("Darwin", "arm64"), ("Linux", "x86_64")}:
        parser.error(f"unsupported native build host: {host}")
    if sys.version_info[:2] != (3, 11):
        parser.error("run with Python 3.11")
    source = source_identity()
    output.mkdir(parents=True)
    environment = dict(os.environ, SOURCE_DATE_EPOCH=source["source_date_epoch"])
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    # Do not inherit target/feature overrides into shipping artifacts.
    for key in ("CARGO_BUILD_TARGET", "CARGO_ENCODED_RUSTFLAGS", "RUSTFLAGS"):
        environment.pop(key, None)
    if host[0] == "Darwin":
        environment["MACOSX_DEPLOYMENT_TARGET"] = "11.0"
    steps = []

    def run(label: str, command: list[str], *, env: dict | None = None) -> None:
        print(f"[{label}] {' '.join(command)}", flush=True)
        with (output / f"{label}.log").open("w") as log:
            result = subprocess.run(
                command, cwd=ROOT, env=env or environment, stdout=log, stderr=subprocess.STDOUT
            )
        steps.append({"name": label, "command": command, "exit_code": result.returncode})
        if result.returncode:
            print((output / f"{label}.log").read_text()[-16000:], file=sys.stderr)
            raise RuntimeError(f"{label} failed; see {output}")

    report = {
        "schema": "qwen-mm-release-v1",
        "version": "0.1.0",
        "source": source,
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": sys.version,
            "rust": capture(str(ROOT / "scripts/cargo.sh"), "--version"),
            "uv": capture(*UV, "--version"),
            "libc": platform.libc_ver(),
            "build_environment": os.environ.get("RELEASE_BUILD_ENVIRONMENT", "native host"),
        },
        "steps": steps,
        "status": "running",
    }
    try:
        wheels = []
        with tempfile.TemporaryDirectory(prefix="qwen-mm-release-") as temporary:
            work = Path(temporary)
            for index in (1, 2):
                build_environment = dict(environment)
                build_environment["CARGO_TARGET_DIR"] = str(work / f"target-{index}")
                # Keep absolute build paths out of native panic/debug strings.
                build_environment["RUSTFLAGS"] = f"--remap-path-prefix={ROOT}=/qwen-mm"
                run(
                    f"build-{index}",
                    UV
                    + [
                        "run",
                        "--locked",
                        "maturin",
                        "build",
                        "--release",
                        "--locked",
                        "--compatibility",
                        "pypi",
                        "--interpreter",
                        sys.executable,
                        "--out",
                        str(output / f"build-{index}"),
                    ],
                    env=build_environment,
                )
                built = list((output / f"build-{index}").glob("*.whl"))
                assert len(built) == 1
                wheels.append(built[0])
            report["artifact"] = inspect_wheel(wheels[0])
            assert sha256(wheels[0]) == sha256(wheels[1]), "independent build hashes differ"
            report["reproducibility"] = (
                "two separate Cargo target directories; identical wheel SHA-256"
            )
            run("create-environment", UV + ["venv", "--python", sys.executable, str(work / "venv")])
            python = str(work / "venv/bin/python")
            run("install", UV + ["pip", "install", "--python", python, str(wheels[0])])
            run(
                "fetch-pinned-processors",
                [
                    python,
                    "-c",
                    (
                        "from qwen_mm import Processor; "
                        "[Processor.from_pretrained(p['model_id'], cache_dir='reference/.cache/huggingface') "
                        "for p in Processor.supported_profiles()]"
                    ),
                ],
            )
            tests = ROOT / "crates/qwen-mm-python/tests"
            for name in (
                "smoke",
                "pretrained",
                "docs_examples",
                "media_sources",
                "usability",
                "rosetta_regressions",
            ):
                # Minimal declared runtime dependencies only; Torch remains optional.
                run(name, [python, str(tests / f"{name}.py")])
            run(
                "ownership",
                [
                    python,
                    "-c",
                    (
                        "import runpy; import qwen_mm._native as n; "
                        "assert not hasattr(n, '_test_native_batch_active'); "
                        f"suite=runpy.run_path({str(tests / 'binding.py')!r}); "
                        "[(print(name, flush=True), fn()) for name, fn in suite.items() "
                        "if name.startswith('test_') and name != 'test_exact_24_image_shape_and_gil_release']"
                    ),
                ],
            )
            run(
                "export-oracle",
                UV
                + [
                    "export",
                    "--locked",
                    "--package",
                    "qwen-mm-reference",
                    "--no-emit-workspace",
                    "--no-dev",
                    "--format",
                    "requirements-txt",
                    "--output-file",
                    str(output / "oracle-requirements.txt"),
                ],
            )
            run(
                "install-oracle",
                UV
                + [
                    "pip",
                    "install",
                    "--python",
                    python,
                    "--require-hashes",
                    "-r",
                    str(output / "oracle-requirements.txt"),
                ],
            )
            run("rosetta-ergonomics", [python, str(tests / "rosetta_ergonomics.py")])
            run(
                "rosetta",
                [python, "scripts/rosetta_suite.py", "--output", str(output / "rosetta.json")],
            )
            run(
                "resize",
                UV
                + [
                    "run",
                    "--locked",
                    "--package",
                    "qwen-mm-reference",
                    "python",
                    "scripts/verify_rosetta_resize.py",
                    "--candidate-python",
                    python,
                    "--wheel",
                    str(wheels[0]),
                    "--output",
                    str(output / "resize.zip"),
                ],
            )
            run("repository-check", ["make", "check"])
            run(
                "script-tests",
                UV
                + [
                    "run",
                    "--locked",
                    "--package",
                    "qwen-mm-reference",
                    "python",
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "scripts/tests",
                ],
            )
            run("binding-with-gil-hooks", ["make", "python-binding-test"])
            run(
                "publish-dry-run",
                UV
                + [
                    "publish",
                    "--dry-run",
                    "--offline",
                    "--no-config",
                    "--trusted-publishing",
                    "never",
                    str(wheels[0]),
                ],
            )
            assert source_identity() == source, "source changed during verification"
            report["status"] = "passed"
    finally:
        report["files"] = {
            str(path.relative_to(output)): sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        }
        (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Release candidate verified: {output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
