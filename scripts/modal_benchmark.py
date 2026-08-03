"""Ephemeral native Linux x86 build and benchmark smoke on Modal."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/qwen-mm")
BASE_IMAGE = "python@sha256:28255a3ace7eb4c48bc1b57b90af29e1bc82b4fd6c60614a8e3dce61b87ff941"
BASE_IMAGE_TAG = "python:3.11.15-slim-bookworm"
MODAL_CPU = 4.0
MODAL_MEMORY_MIB = 8_192
RUST_VERSION = "1.97.1"
UV_VERSION = "0.11.29"
PYPI_INDEX = "https://pypi.org/simple"

app = modal.App("qwen-mm-d0-modal-benchmark")


def _source_ignore(relative: Path) -> bool:
    """Apply the fingerprint policy to Modal's relative upload paths."""

    support_directory = str(LOCAL_ROOT / "scripts")
    if support_directory not in sys.path:
        sys.path.insert(0, support_directory)
    from modal_benchmark_support import is_ignored_source_path

    return is_ignored_source_path(relative)


benchmark_image = (
    modal.Image.from_registry(BASE_IMAGE)
    .apt_install(
        "build-essential",
        "ca-certificates",
        "curl",
        "file",
        "git",
        "pkg-config",
        "util-linux",
    )
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        f"| sh -s -- -y --profile minimal --default-toolchain {RUST_VERSION}",
        f"/root/.cargo/bin/rustup component add --toolchain {RUST_VERSION} clippy rustfmt",
        f"curl -LsSf https://astral.sh/uv/{UV_VERSION}/install.sh "
        "| env UV_INSTALL_DIR=/usr/local/bin sh",
    )
    .env(
        {
            "PATH": "/root/.cargo/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin",
            "UV_PYTHON_DOWNLOADS": "never",
        }
    )
    .add_local_dir(str(LOCAL_ROOT), remote_path=str(REMOTE_ROOT), copy=True, ignore=_source_ignore)
    .run_commands(
        f"cd {REMOTE_ROOT} && uv sync --locked --all-packages --default-index {PYPI_INDEX}",
    )
    .workdir(str(REMOTE_ROOT))
)


def _run_logged(command: list[str], *, log_path: Path, environment: dict[str, str]) -> None:
    result = subprocess.run(
        command,
        cwd=REMOTE_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    rendered = "$ " + shlex.join(command) + "\n" + result.stdout
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(rendered, encoding="utf-8")
    if result.returncode != 0:
        print(rendered, file=sys.stderr, flush=True)
        raise RuntimeError(
            f"command failed with exit code {result.returncode}; see {log_path.name}: "
            f"{shlex.join(command)}"
        )


def _capture(command: list[str]) -> str:
    return subprocess.run(
        command,
        cwd=REMOTE_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    ).stdout.strip()


def _cpu_description(lscpu_fields: dict[str, Any]) -> str:
    cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    model_lines = [
        line.split(":", 1)[1].strip()
        for line in cpuinfo.splitlines()
        if line.startswith("model name")
    ]
    if model_lines and model_lines[0].lower() != "unknown":
        return model_lines[0]
    model_name = str(lscpu_fields.get("Model name", "")).strip()
    if model_name and model_name.lower() != "unknown":
        return model_name
    vendor = str(lscpu_fields.get("Vendor ID", "unknown vendor"))
    family = str(lscpu_fields.get("CPU family", "unknown"))
    model = str(lscpu_fields.get("Model", "unknown"))
    return f"{vendor} family {family} model {model} (provider masks model name)"


def _memory_total_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1_024
    raise RuntimeError("/proc/meminfo did not contain MemTotal")


def _cgroup_limits() -> dict[str, str]:
    paths = (
        "/sys/fs/cgroup/cpu.max",
        "/sys/fs/cgroup/cpuset.cpus.effective",
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
        "/sys/fs/cgroup/cpuset/cpuset.cpus",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    )
    values: dict[str, str] = {}
    for path in paths:
        try:
            value = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            values[path] = value
    return values


@app.function(image=benchmark_image, cpu=MODAL_CPU, memory=MODAL_MEMORY_MIB, timeout=1_800)
def run_d0(
    *,
    source_revision: str,
    expected_source_digest: str,
    expected_source_files: int,
    expected_source_bytes: int,
    local_dirty_state: str,
    modal_client_version: str,
) -> bytes:
    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    from modal_benchmark_support import (
        PROVENANCE_SCHEMA_ID,
        PROVENANCE_SCHEMA_VERSION,
        REQUIRED_ARTIFACT_FILES,
        assert_native_linux_x86,
        create_artifact,
        read_files,
        source_tree_digest,
    )

    started_at = datetime.now(UTC).isoformat()
    system = platform.system()
    machine = platform.machine()
    lscpu = json.loads(_capture(["lscpu", "--json"]))
    lscpu_fields = {
        str(item["field"]).removesuffix(":"): item.get("data")
        for item in lscpu.get("lscpu", [])
        if isinstance(item, dict) and "field" in item
    }
    cpu_description = _cpu_description(lscpu_fields)
    assert_native_linux_x86(system=system, machine=machine, cpu_description=cpu_description)
    actual_digest, actual_files, actual_bytes = source_tree_digest(REMOTE_ROOT)
    expected = (expected_source_digest, expected_source_files, expected_source_bytes)
    actual = (actual_digest, actual_files, actual_bytes)
    if actual != expected:
        raise RuntimeError(
            f"packaged source fingerprint mismatch: expected {expected}, got {actual}"
        )

    with tempfile.TemporaryDirectory(prefix="qwen-mm-modal-d0-") as temporary:
        artifact_root = Path(temporary)
        logs = artifact_root / "logs"
        benchmark = artifact_root / "benchmark"
        build = artifact_root / "build"
        logs.mkdir()
        benchmark.mkdir()
        build.mkdir()
        environment = dict(os.environ)
        for name in (
            "PIP_EXTRA_INDEX_URL",
            "PIP_INDEX_URL",
            "UV_EXTRA_INDEX_URL",
            "UV_INDEX",
            "UV_INDEX_URL",
        ):
            environment.pop(name, None)
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "UV_DEFAULT_INDEX": PYPI_INDEX,
                "BENCHMARK_OUTPUT": str(benchmark / "result.json"),
                "BENCHMARK_REPORT": str(benchmark / "report.md"),
            }
        )

        wheel_directory = build / "wheel"
        wheel_directory.mkdir()
        build_commands = [
            [
                "uv",
                "run",
                "--locked",
                "--default-index",
                PYPI_INDEX,
                "maturin",
                "build",
                "--locked",
                "--interpreter",
                sys.executable,
                "--out",
                str(wheel_directory),
            ],
        ]
        build_log = logs / "build.log"
        _run_logged(build_commands[0], log_path=build_log, environment=environment)
        wheels = sorted(wheel_directory.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected exactly one built wheel, got {len(wheels)}")
        smoke_venv = build / "smoke-venv"
        install_commands = [
            ["uv", "venv", "--python", sys.executable, str(smoke_venv)],
            [
                "uv",
                "pip",
                "install",
                "--default-index",
                PYPI_INDEX,
                "--python",
                str(smoke_venv / "bin/python"),
                str(wheels[0]),
            ],
            [str(smoke_venv / "bin/python"), "crates/qwen-mm-python/tests/smoke.py"],
        ]
        for command in install_commands:
            previous = build_log.read_text(encoding="utf-8")
            command_log = build / "command.log"
            _run_logged(command, log_path=command_log, environment=environment)
            build_log.write_text(
                previous + "\n" + command_log.read_text(encoding="utf-8"), encoding="utf-8"
            )
        native_modules = sorted((smoke_venv / "lib").rglob("qwen_mm/_native*.so"))
        if len(native_modules) != 1:
            raise RuntimeError(f"expected one installed native module, got {len(native_modules)}")
        elf_description = _capture(["file", "-b", str(native_modules[0])])
        if "ELF 64-bit" not in elf_description or "x86-64" not in elf_description:
            raise RuntimeError(f"installed native module is not x86-64 ELF: {elf_description}")
        build_log.write_text(
            build_log.read_text(encoding="utf-8")
            + f"\n$ file -b {native_modules[0]}\n{elf_description}\n",
            encoding="utf-8",
        )

        benchmark_command = ["make", "benchmark-v2-self-test"]
        _run_logged(
            benchmark_command,
            log_path=logs / "benchmark.log",
            environment=environment,
        )
        validation_command = [
            "uv",
            "run",
            "--locked",
            "--default-index",
            PYPI_INDEX,
            "--no-sync",
            "--package",
            "qwen-mm-reference",
            "python",
            "-m",
            "qwen_mm_reference.benchmark_v2",
            "validate",
            str(benchmark / "result.json"),
        ]
        _run_logged(
            validation_command,
            log_path=logs / "validation.log",
            environment=environment,
        )

        safe_modal_environment = {
            name: os.environ[name]
            for name in (
                "MODAL_CLOUD_PROVIDER",
                "MODAL_APP_ID",
                "MODAL_CONTAINER_ID",
                "MODAL_FUNCTION_ID",
                "MODAL_IMAGE_ID",
                "MODAL_REGION",
                "MODAL_TASK_ID",
            )
            if name in os.environ
        }
        provenance: dict[str, Any] = {
            "schema_id": PROVENANCE_SCHEMA_ID,
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "execution": "modal_native_linux_x86_diagnostic",
            "dedicated_host": False,
            "performance_claim": False,
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "system": system,
            "machine": machine,
            "uname": _capture(["uname", "-a"]),
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "cpu_description": cpu_description,
            "logical_cpu_count": os.cpu_count(),
            "visible_cpu_affinity": sorted(os.sched_getaffinity(0)),
            "proc_meminfo_total_bytes": _memory_total_bytes(),
            "cgroup_limits": _cgroup_limits(),
            "requested_cpu_physical_cores": MODAL_CPU,
            "requested_memory_mib": MODAL_MEMORY_MIB,
            "lscpu": lscpu,
            "base_image_tag": BASE_IMAGE_TAG,
            "base_image_reference": BASE_IMAGE,
            "python_default_index": PYPI_INDEX,
            "modal_client_version": modal_client_version,
            "modal_environment": safe_modal_environment,
            "python_version": platform.python_version(),
            "uv_version": _capture(["uv", "--version"]),
            "rustc_version": _capture(["rustc", "--version"]),
            "cargo_version": _capture(["cargo", "--version"]),
            "python_packages": _capture(
                [
                    "uv",
                    "pip",
                    "freeze",
                    "--python",
                    str(REMOTE_ROOT / ".venv/bin/python"),
                ]
            ).splitlines(),
            "build_system_packages": _capture(
                [
                    "dpkg-query",
                    "-W",
                    "-f=${binary:Package}=${Version}\\n",
                    "build-essential",
                    "ca-certificates",
                    "curl",
                    "file",
                    "git",
                    "pkg-config",
                    "util-linux",
                ]
            ).splitlines(),
            "execution_environment": {
                name: environment[name]
                for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "UV_DEFAULT_INDEX")
            },
            "source_revision": source_revision,
            "source_dirty_state": local_dirty_state,
            "source_digest": actual_digest,
            "source_file_count": actual_files,
            "source_bytes": actual_bytes,
            "wheel_name": wheels[0].name,
            "wheel_sha256": __import__("hashlib").sha256(wheels[0].read_bytes()).hexdigest(),
            "native_module_file": elf_description,
            "commands": [
                shlex.join(build_commands[0]),
                *(shlex.join(command) for command in install_commands),
                f"file -b {native_modules[0]}",
                shlex.join(benchmark_command),
                shlex.join(validation_command),
            ],
        }
        (artifact_root / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staged = read_files(artifact_root, REQUIRED_ARTIFACT_FILES)
        staged[f"build/{wheels[0].name}"] = wheels[0].read_bytes()
        return create_artifact(staged)


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=LOCAL_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    ).stdout.strip()


@app.local_entrypoint()
def main(output: str = "/tmp/qwen-mm-modal-d0.zip") -> None:
    sys.path.insert(0, str(LOCAL_ROOT / "scripts"))
    from modal_benchmark_support import source_tree_digest, write_artifact_atomically

    digest, file_count, source_bytes = source_tree_digest(LOCAL_ROOT)
    revision = _git_output("rev-parse", "HEAD")
    dirty_state = _git_output("status", "--short", "--untracked-files=all")
    modal_version = _capture_local_modal_version()
    artifact = run_d0.remote(
        source_revision=revision,
        expected_source_digest=digest,
        expected_source_files=file_count,
        expected_source_bytes=source_bytes,
        local_dirty_state=dirty_state,
        modal_client_version=modal_version,
    )
    destination = Path(output).expanduser().resolve()
    validated = write_artifact_atomically(artifact, destination)
    provenance = validated["provenance"]
    print(f"wrote validated Modal D0 artifact: {destination}")
    print(
        "native worker: "
        f"{provenance['system']} {provenance['machine']} / {provenance['cpu_description']}"
    )


def _capture_local_modal_version() -> str:
    try:
        return importlib.metadata.version("modal")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
