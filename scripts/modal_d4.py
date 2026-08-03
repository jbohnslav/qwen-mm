"""Build and capture both D4 variants on one controlled Modal x86 worker."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

LOCAL_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ASSETS_ROOT = (LOCAL_ROOT / "reference/.cache/huggingface").resolve()
REMOTE_ROOT = Path("/workspace/qwen-mm")
REMOTE_ASSETS_ROOT = REMOTE_ROOT / "reference/.cache/huggingface"
REMOTE_PYTHON = REMOTE_ROOT / ".venv/bin/python"
SUPPORT_ROOT = REMOTE_ROOT if REMOTE_ROOT.is_dir() else LOCAL_ROOT

sys.path.insert(0, str(SUPPORT_ROOT / "scripts"))
from d4_capture_support import (  # noqa: E402
    BASE_IMAGE,
    BUILD_LABELS,
    MODAL_CPU,
    MODAL_MEMORY_MIB,
    PYPI_INDEX,
    PYPI_OVERRIDE_ENVIRONMENT_NAMES,
    RUST_VERSION,
    THREAD_BUDGETS,
    UV_VERSION,
    D4CaptureError,
    assert_assets_identity,
    assert_build_invariants,
    assets_identity,
    build_environment_evidence,
    capture_input_identities,
    create_capture_archive,
    modal_resource_attestation,
    normalize_build_artifact_paths,
    normalized_capture_environment,
    physical_core_masks,
    reconcile_installed_runtime,
    reference_sync_command,
    reference_sync_evidence,
    source_payload_identity,
    source_upload_ignored,
    toolchain_pins,
    wheel_build_command,
    write_capture_archive,
)

app = modal.App("qwen-mm-d4-controlled-capture")
PYPI_UNSET_ARGUMENTS = " ".join(f"-u {name}" for name in PYPI_OVERRIDE_ENVIRONMENT_NAMES)


def _source_ignore(relative: Path) -> bool:
    return source_upload_ignored(relative)


d4_image = (
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
        f"cd {REMOTE_ROOT} && env {PYPI_UNSET_ARGUMENTS} UV_DEFAULT_INDEX={PYPI_INDEX} "
        f"uv sync --locked --all-packages --group dev --default-index {PYPI_INDEX}",
    )
    # Phase C authenticates committed inputs with git show/merge-base. Keep the
    # object database separate from the fingerprinted source payload.
    .add_local_dir(str(LOCAL_ROOT / ".git"), remote_path=str(REMOTE_ROOT / ".git"), copy=True)
    .add_local_dir(str(LOCAL_ASSETS_ROOT), remote_path=str(REMOTE_ASSETS_ROOT), copy=True)
    .workdir(str(REMOTE_ROOT))
)


def _capture(command: list[str], *, environment: dict[str, str]) -> str:
    return subprocess.run(
        command,
        cwd=REMOTE_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    ).stdout.strip()


def _cgroup_limits() -> dict[str, str]:
    values: dict[str, str] = {}
    for name in ("cpu.max", "cpuset.cpus.effective", "memory.max"):
        path = Path("/sys/fs/cgroup") / name
        try:
            values[str(path)] = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
    return values


def _memory_total_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    raise D4CaptureError("/proc/meminfo did not expose MemTotal")


@app.function(
    image=d4_image,
    cpu=MODAL_CPU,
    memory=MODAL_MEMORY_MIB,
    timeout=43_200,
    nonpreemptible=True,
    single_use_containers=True,
)
def run_d4(
    *,
    expected_source: dict[str, Any],
    expected_assets: dict[str, Any],
    source_revision: str,
    modal_client_version: str,
) -> bytes:
    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    from d4_capture_support import source_payload_identity
    from d4_linux import _affinity_enforcement_phase
    from d4_local import _capture as capture_command
    from d4_local import _collect_files as collect_files
    from d4_local import _run_logged, _runtime_identity
    from d4_worker import run_capture
    from modal_benchmark_support import assert_native_linux_x86
    from profile_capture_support import install_wheel_command

    observed_source = source_payload_identity(REMOTE_ROOT)
    if observed_source != expected_source:
        raise D4CaptureError(
            f"uploaded source identity mismatch: {observed_source} != {expected_source}"
        )
    assert_assets_identity(REMOTE_ASSETS_ROOT, expected_assets)
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise D4CaptureError("Modal D4 runner requires native Linux x86_64")
    lscpu_json = json.loads(_capture(["lscpu", "--json"], environment=dict(os.environ)))
    cpu_description = " ".join(
        str(item.get("data", ""))
        for item in lscpu_json.get("lscpu", [])
        if isinstance(item, dict) and item.get("field") == "Model name:"
    ).strip()
    if not cpu_description:
        cpu_description = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    assert_native_linux_x86(
        system=platform.system(), machine=platform.machine(), cpu_description=cpu_description
    )
    visible_affinity = sorted(os.sched_getaffinity(0))
    topology = _capture(["lscpu", "--parse=CPU,CORE,SOCKET,ONLINE"], environment=dict(os.environ))
    masks = physical_core_masks(topology, allowed_cpus=visible_affinity)
    cgroups = _cgroup_limits()
    uname = _capture(["uname", "-a"], environment=dict(os.environ))
    resource_attestation = modal_resource_attestation(
        cgroup_limits=cgroups,
        platform_text=platform.platform(),
        uname=uname,
        visible_affinity=visible_affinity,
        masks=masks,
    )

    environment = normalized_capture_environment(os.environ)
    # Reported taskset/sched_getaffinity state is insufficient under a virtual
    # kernel that does not actually constrain native worker threads.  Exercise
    # the frozen t1 budget before any long build or benchmark work.
    _affinity_enforcement_phase(masks[1], environment=environment)
    normalized_build_environment = build_environment_evidence(environment)
    build_host = {
        "os_release": {
            "command": ["cat", "/etc/os-release"],
            "output": Path("/etc/os-release").read_text(encoding="utf-8").strip(),
        },
        "uname": {"command": ["uname", "-a"], "output": uname},
    }
    started = datetime.now(UTC)
    with tempfile.TemporaryDirectory(prefix="qwen-mm-d4-modal-") as temporary:
        temporary_root = Path(temporary)
        working_root = temporary_root / "working"
        artifact_root = temporary_root / "artifact"
        build_invariants: dict[str, dict[str, Any]] = {}
        native_hashes: dict[str, str] = {}
        wheel_hashes: dict[str, str] = {}
        for label in BUILD_LABELS:
            venv = working_root / "venvs" / label
            wheel_directory = working_root / "wheels" / label
            cargo_target = working_root / "cargo-target" / label
            build_log = artifact_root / "logs" / label / "build.log"
            sync_log = artifact_root / "logs" / label / "sync.log"
            create_venv = [
                "uv",
                "venv",
                "--python",
                str(REMOTE_PYTHON),
                str(venv),
            ]
            sync_command = reference_sync_command(venv=venv)
            build_command = wheel_build_command(
                python=venv / "bin/python",
                output=wheel_directory,
                build_label=label,
                cargo_target_dir=cargo_target,
            )
            _run_logged(create_venv, log=build_log, environment=environment)
            _run_logged(
                sync_command,
                log=sync_log,
                environment=environment,
            )
            _run_logged(
                build_command,
                log=build_log,
                environment=environment,
            )
            wheels = sorted(wheel_directory.glob("*.whl"))
            if len(wheels) != 1:
                raise D4CaptureError(f"{label}: expected exactly one wheel, got {len(wheels)}")
            python = venv / "bin/python"
            retained_wheel = artifact_root / "builds" / label / wheels[0].name
            retained_wheel.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wheels[0], retained_wheel)
            install_command = install_wheel_command(python=python, wheel=retained_wheel)
            _run_logged(
                install_command,
                log=build_log,
                environment=environment,
            )
            runtime = _runtime_identity(python, environment=environment)
            runtime_reconciliation = reconcile_installed_runtime(
                wheel=retained_wheel, runtime=runtime
            )
            native_hashes[label] = runtime["native_sha256"]
            wheel_hashes[label] = runtime_reconciliation["wheel_contents"]["wheel"]["sha256"]
            run_capture(
                python=python,
                wheel=retained_wheel,
                build_label=label,
                assets_root=REMOTE_ASSETS_ROOT,
                output_root=artifact_root,
                affinity_masks=masks,
                execute=True,
            )
            build_path = artifact_root / "builds" / label / "build.json"
            build = json.loads(build_path.read_text(encoding="utf-8"))
            packages = capture_command(
                ["uv", "pip", "freeze", "--python", str(python)],
                environment=environment,
            ).splitlines()
            toolchain = {
                "python": {
                    "command": [str(python), "--version"],
                    "output": capture_command([str(python), "--version"], environment=environment),
                },
                "maturin": {
                    "command": ["uv", "run", "--locked", "--no-sync", "maturin", "--version"],
                    "output": capture_command(
                        ["uv", "run", "--locked", "--no-sync", "maturin", "--version"],
                        environment=environment,
                    ),
                },
                "rustc": {
                    "command": ["rustc", "--version", "--verbose"],
                    "output": capture_command(
                        ["rustc", "--version", "--verbose"], environment=environment
                    ),
                },
                "cargo": {
                    "command": ["cargo", "--version", "--verbose"],
                    "output": capture_command(
                        ["cargo", "--version", "--verbose"], environment=environment
                    ),
                },
            }
            build.update(
                {
                    "commands": {
                        "create_venv": create_venv,
                        "sync_reference": sync_command,
                        "build_wheel": build_command,
                        "install_retained_wheel": install_command,
                    },
                    "build_environment": normalized_build_environment,
                    "build_host": build_host,
                    "toolchain": toolchain,
                    "runtime": runtime,
                    "runtime_reconciliation": runtime_reconciliation,
                    "reference_sync": reference_sync_evidence(
                        venv=venv,
                        environment=environment,
                        log=sync_log,
                    ),
                    "packages": packages,
                }
            )
            build_path.write_text(
                json.dumps(build, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            build_invariants[label] = {
                "build_environment": normalized_build_environment,
                "build_host": build_host,
                "packages": [
                    normalize_build_artifact_paths(
                        package,
                        {
                            working_root / "venvs" / label: "<build-venv>",
                            artifact_root / "builds" / label: "<retained-build>",
                        },
                    )
                    for package in packages
                ],
                "toolchain": {name: value["output"] for name, value in toolchain.items()},
            }
        assert_build_invariants(build_invariants)
        if len(set(native_hashes.values())) != len(BUILD_LABELS):
            raise D4CaptureError("shipping and native builds produced the same native binary")
        if len(set(wheel_hashes.values())) != len(BUILD_LABELS):
            raise D4CaptureError("shipping and native builds produced the same wheel archive")
        assert_assets_identity(REMOTE_ASSETS_ROOT, expected_assets)

        modal_environment = {
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
        provenance = {
            "schema_id": "qwen-mm-d4-raw-capture-provenance-v1",
            "schema_version": 1,
            "claim": "raw controlled-host input for the separate D4 certification evaluator",
            "architecture_family": "x86_64",
            "provider": "modal",
            "started_at": started.isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
            "source_revision": source_revision,
            "source": observed_source,
            "assets": expected_assets,
            "capture_inputs": capture_input_identities(REMOTE_ROOT),
            "host": {
                "system": platform.system(),
                "machine": platform.machine(),
                "platform": platform.platform(),
                "uname": uname,
                "hostname": socket.gethostname(),
                "cpu_description": cpu_description,
                "lscpu": lscpu_json,
                "lscpu_parse": topology,
                "logical_cpu_count": os.cpu_count(),
                "proc_meminfo_total_bytes": _memory_total_bytes(),
                "resource_attestation": resource_attestation,
            },
            "modal": {"client_version": modal_client_version, "environment": modal_environment},
            "build_wheel_sha256": wheel_hashes,
            "build_native_sha256": native_hashes,
            "toolchain_pins": toolchain_pins(),
            "sample_pruning": "forbidden",
            "noise_cv_max": 0.05,
            "environment": {
                name: environment[name]
                for name in ("PYTHONHASHSEED", "LC_ALL", "TZ", "UV_DEFAULT_INDEX")
            }
            | {"removed_package_index_variables": list(PYPI_OVERRIDE_ENVIRONMENT_NAMES)},
        }
        (artifact_root / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        files = collect_files(artifact_root)
        index = {
            "schema_id": "qwen-mm-d4-raw-capture-index-v1",
            "schema_version": 1,
            "architecture_family": "x86_64",
            "build_labels": list(BUILD_LABELS),
            "thread_budgets": list(THREAD_BUDGETS),
            "files": sorted(files),
        }
        (artifact_root / "capture-index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        files["capture-index.json"] = (artifact_root / "capture-index.json").read_bytes()
        return create_capture_archive(files)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=LOCAL_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    ).stdout.strip()


@app.local_entrypoint()
def main(output: str = "/tmp/qwen-mm-d4-x86_64.zip", dry_run: bool = False) -> None:
    source = source_payload_identity(LOCAL_ROOT)
    assets = assets_identity(LOCAL_ASSETS_ROOT)
    revision = _git("rev-parse", "HEAD")
    dirty = _git("status", "--short", "--untracked-files=all")
    plan = {
        "function": "run_d4",
        "source": source,
        "assets": assets,
        "source_revision": revision,
        "resources": {
            "cpu": MODAL_CPU,
            "memory_mib": MODAL_MEMORY_MIB,
            "nonpreemptible": True,
            "single_use_container": True,
        },
        "build_labels": list(BUILD_LABELS),
        "thread_budgets": list(THREAD_BUDGETS),
    }
    if dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if dirty:
        raise D4CaptureError("D4 Modal capture requires a clean checkout")
    artifact = run_d4.remote(
        expected_source=source,
        expected_assets=assets,
        source_revision=revision,
        modal_client_version=importlib.metadata.version("modal"),
    )
    destination = Path(output).expanduser().resolve()
    write_capture_archive(
        artifact,
        destination,
        expected_source=source,
        expected_assets=assets,
        phase_c_assets_root=LOCAL_ASSETS_ROOT,
    )
    print(f"wrote validated D4 x86_64 raw capture: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
