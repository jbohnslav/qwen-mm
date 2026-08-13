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
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import modal
except ModuleNotFoundError:  # The in-Sandbox worker does not need the Modal client.
    modal = None  # type: ignore[assignment]

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
    assert_build_variant_artifacts,
    assets_identity,
    build_environment_evidence,
    capture_input_identities,
    create_capture_archive,
    normalize_build_artifact_paths,
    normalized_capture_environment,
    physical_core_masks,
    physical_core_representatives,
    reconcile_installed_runtime,
    reference_sync_command,
    reference_sync_evidence,
    source_payload_identity,
    source_upload_ignored,
    toolchain_pins,
    verify_private_environment_integrity,
    wheel_build_command,
    write_capture_archive,
)

APP_NAME = "qwen-mm-d4-controlled-capture"
SANDBOX_RUNTIME = "modal-vm-sandbox-v1"
SANDBOX_CPU = (MODAL_CPU, MODAL_CPU)
SANDBOX_MEMORY_MIB = (MODAL_MEMORY_MIB, MODAL_MEMORY_MIB)
SANDBOX_TIMEOUT_SECONDS = 43_200
SANDBOX_IDLE_TIMEOUT_SECONDS = 600
SANDBOX_CONTROL_PATH = "/tmp/qwen-mm-d4-control.json"
SANDBOX_OUTPUT_PATH = "/tmp/qwen-mm-d4-output.bin"
SANDBOX_PROBE_PATH = "/tmp/qwen-mm-d4-probe.json"
VM_EXPERIMENTAL_OPTIONS = {"vm_runtime": True}
MODAL_PRICING_URL = "https://modal.com/pricing"
SANDBOX_CPU_USD_PER_PHYSICAL_CORE_SECOND = 0.00003942
SANDBOX_MEMORY_USD_PER_GIB_SECOND = 0.00000667

if modal is not None:
    app = modal.App(APP_NAME)
else:
    app = None
PYPI_UNSET_ARGUMENTS = " ".join(f"-u {name}" for name in PYPI_OVERRIDE_ENVIRONMENT_NAMES)


def _source_ignore(relative: Path) -> bool:
    return source_upload_ignored(relative)


if modal is not None:
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
        .add_local_dir(
            str(LOCAL_ROOT), remote_path=str(REMOTE_ROOT), copy=True, ignore=_source_ignore
        )
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
else:
    d4_image = None


def _image_anchor() -> None:
    """Make Modal resolve the image during App initialization without running a Function."""


if app is not None:
    _image_anchor = app.function(image=d4_image)(_image_anchor)


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


def _pricing_snapshot() -> dict[str, Any]:
    hourly = 3600 * (
        MODAL_CPU * SANDBOX_CPU_USD_PER_PHYSICAL_CORE_SECOND
        + (MODAL_MEMORY_MIB / 1024) * SANDBOX_MEMORY_USD_PER_GIB_SECOND
    )
    return {
        "source": MODAL_PRICING_URL,
        "cpu_usd_per_physical_core_second": SANDBOX_CPU_USD_PER_PHYSICAL_CORE_SECOND,
        "memory_usd_per_gib_second": SANDBOX_MEMORY_USD_PER_GIB_SECOND,
        "requested_resource_usd_per_hour": hourly,
        "nonpreemptible_multiplier": 1.0,
        "note": "CPU-only Modal Sandboxes are not subject to preemption",
    }


def _sandbox_plan(
    *, source: dict[str, Any], assets: dict[str, Any], source_revision: str
) -> dict[str, Any]:
    return {
        "runner": SANDBOX_RUNTIME,
        "source": source,
        "assets": assets,
        "source_revision": source_revision,
        "image": {
            "base_image": BASE_IMAGE,
            "resolved_modal_image_id": "resolved immediately before Sandbox.create",
        },
        "resources": {
            "cpu_request_and_hard_limit": list(SANDBOX_CPU),
            "memory_request_and_hard_limit_mib": list(SANDBOX_MEMORY_MIB),
            "vm_runtime": True,
            "nonpreemptible": True,
            "nonpreemptible_basis": "Modal CPU-only Sandbox runtime semantics",
            "single_use": True,
            "timeout_seconds": SANDBOX_TIMEOUT_SECONDS,
            "idle_timeout_seconds": SANDBOX_IDLE_TIMEOUT_SECONDS,
        },
        "lifecycle": {
            "create_once": True,
            "terminate_wait": True,
            "poll_after_terminate": True,
            "detach_in_finally": True,
        },
        "pricing_snapshot": _pricing_snapshot(),
        "build_labels": list(BUILD_LABELS),
        "thread_budgets": list(THREAD_BUDGETS),
    }


def _vm_resource_attestation(
    *,
    cgroup_limits: dict[str, str],
    virtualization: str,
    visible_affinity: list[int],
    masks: dict[int, tuple[int, ...]],
    topology: str,
    control: dict[str, Any],
    observed_image_id: str | None,
) -> dict[str, Any]:
    """Authenticate the fixed VM allocation without inventing absent cgroup files."""

    expected_cpus = list(range(int(MODAL_CPU)))
    if control.get("runner") != SANDBOX_RUNTIME:
        raise D4CaptureError("Modal D4 worker did not receive a VM Sandbox control record")
    resources = control.get("resources")
    if not isinstance(resources, dict) or resources != {
        "cpu_request_and_hard_limit": list(SANDBOX_CPU),
        "memory_request_and_hard_limit_mib": list(SANDBOX_MEMORY_MIB),
        "vm_runtime": True,
        "nonpreemptible": True,
        "single_use": True,
    }:
        raise D4CaptureError("Modal VM Sandbox resource request changed")
    image_id = control.get("resolved_modal_image_id")
    if not isinstance(image_id, str) or not image_id.startswith("im-"):
        raise D4CaptureError("Modal VM Sandbox image identity is not pinned")
    if observed_image_id is not None and observed_image_id != image_id:
        raise D4CaptureError("Modal VM Sandbox runtime image differs from the pinned image ID")
    sandbox_id = control.get("sandbox_id")
    if not isinstance(sandbox_id, str) or not sandbox_id.startswith("sb-"):
        raise D4CaptureError("Modal VM Sandbox allocation identity is missing")
    if virtualization.strip().lower() != "kvm":
        raise D4CaptureError(f"Modal VM Sandbox did not report KVM: {virtualization!r}")
    if visible_affinity != expected_cpus:
        raise D4CaptureError("Modal VM Sandbox affinity is not exactly CPUs 0-15")
    cpuset_key = "/sys/fs/cgroup/cpuset.cpus.effective"
    if cgroup_limits.get(cpuset_key) != "0-15":
        raise D4CaptureError("Modal VM Sandbox effective cpuset is not exactly CPUs 0-15")
    expected_masks = {budget: tuple(range(budget)) for budget in THREAD_BUDGETS}
    if masks != expected_masks:
        raise D4CaptureError("Modal VM Sandbox physical-core topology is not exact or SMT-free")
    rows = [line for line in topology.splitlines() if line.strip() and not line.startswith("#")]
    if len(rows) != len(expected_cpus):
        raise D4CaptureError("Modal VM Sandbox topology does not contain exactly 16 online CPUs")
    representatives = physical_core_representatives(topology, allowed_cpus=visible_affinity)
    if representatives != tuple(expected_cpus):
        raise D4CaptureError("Modal VM Sandbox does not expose 16 distinct SMT-free physical cores")
    return {
        "mode": SANDBOX_RUNTIME,
        "requested_resources_bound_by": "Modal Sandbox.create request/limit tuples",
        "requested_physical_cores": MODAL_CPU,
        "requested_memory_mib": MODAL_MEMORY_MIB,
        "nonpreemptible": True,
        "nonpreemptible_basis": "Modal CPU-only Sandbox runtime semantics",
        "single_use_container": True,
        "sandbox_id": sandbox_id,
        "resolved_modal_image_id": image_id,
        "vm_runtime": True,
        "virtualization": "kvm",
        "visible_affinity": visible_affinity,
        "physical_core_masks": {f"t{budget}": list(masks[budget]) for budget in THREAD_BUDGETS},
        "cgroup_limits": cgroup_limits,
        "api_resource_request": resources,
    }


def _run_d4_capture(
    *,
    expected_source: dict[str, Any],
    expected_assets: dict[str, Any],
    source_revision: str,
    modal_client_version: str,
    sandbox_control: dict[str, Any],
    probe_only: bool = False,
) -> bytes:
    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    from d4_capture_support import source_payload_identity
    from d4_linux import _affinity_enforcement, _affinity_enforcement_phase
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
    virtualization = _capture(["systemd-detect-virt"], environment=dict(os.environ))
    resource_attestation = _vm_resource_attestation(
        cgroup_limits=cgroups,
        virtualization=virtualization,
        visible_affinity=visible_affinity,
        masks=masks,
        topology=topology,
        control=sandbox_control,
        observed_image_id=os.environ.get("MODAL_IMAGE_ID"),
    )

    environment = normalized_capture_environment(os.environ)
    # Reported taskset/sched_getaffinity state is insufficient under a virtual
    # kernel that does not actually constrain native worker threads.  Exercise
    # the frozen t1 budget before any long build or benchmark work.
    preflight = _affinity_enforcement_phase(masks[1], environment=environment)
    if probe_only:
        postflight = _affinity_enforcement_phase(masks[1], environment=environment)
        probe = {
            "schema_id": "qwen-mm-d4-modal-vm-probe-v1",
            "schema_version": 1,
            "host": {
                "system": platform.system(),
                "machine": platform.machine(),
                "platform": platform.platform(),
                "uname": uname,
                "cpu_description": cpu_description,
                "lscpu": lscpu_json,
                "lscpu_parse": topology,
                "logical_cpu_count": os.cpu_count(),
                "proc_meminfo_total_bytes": _memory_total_bytes(),
                "resource_attestation": resource_attestation
                | {
                    "affinity_enforcement": _affinity_enforcement(
                        expected_affinity=masks[1], pre=preflight, post=postflight
                    )
                },
            },
            "sandbox_control": sandbox_control,
        }
        return (json.dumps(probe, indent=2, sort_keys=True) + "\n").encode()
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
        variant_plan: dict[str, Any] = {"builds": {}}
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
            variant_plan["builds"][label] = {
                "venv": str(venv),
                "create_venv": create_venv,
                "build": build_command,
            }
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
        assert_build_variant_artifacts(
            variant_plan, native_hashes=native_hashes, wheel_hashes=wheel_hashes
        )
        assert_assets_identity(REMOTE_ASSETS_ROOT, expected_assets)
        for label in BUILD_LABELS:
            verify_private_environment_integrity(
                working_root / "venvs" / label,
                evidence_path=artifact_root / "builds" / label / "environment-integrity.json",
                checkpoint="before-archive",
            )
        postflight = _affinity_enforcement_phase(masks[1], environment=environment)
        final_affinity = sorted(os.sched_getaffinity(0))
        final_topology = _capture(
            ["lscpu", "--parse=CPU,CORE,SOCKET,ONLINE"], environment=dict(os.environ)
        )
        final_masks = physical_core_masks(final_topology, allowed_cpus=final_affinity)
        final_attestation = _vm_resource_attestation(
            cgroup_limits=_cgroup_limits(),
            virtualization=_capture(["systemd-detect-virt"], environment=dict(os.environ)),
            visible_affinity=final_affinity,
            masks=final_masks,
            topology=final_topology,
            control=sandbox_control,
            observed_image_id=os.environ.get("MODAL_IMAGE_ID"),
        )
        if (
            final_affinity != visible_affinity
            or final_topology != topology
            or final_masks != masks
            or final_attestation != resource_attestation
        ):
            raise D4CaptureError("Modal VM Sandbox allocation changed during D4 capture")
        resource_attestation["affinity_enforcement"] = _affinity_enforcement(
            expected_affinity=masks[1], pre=preflight, post=postflight
        )

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
            "modal": {
                "client_version": modal_client_version,
                "environment": modal_environment,
                "sandbox_control": sandbox_control,
                "pricing_snapshot": _pricing_snapshot(),
                "lifecycle": {
                    "single_use": True,
                    "termination_required_before_local_archive_acceptance": True,
                    "lifecycle_completion_recorded_in_local_sidecar": True,
                },
            },
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


def _worker_main(*, control_path: Path, output_path: Path, probe_only: bool) -> None:
    try:
        control = json.loads(control_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise D4CaptureError("Modal VM Sandbox control file is missing or invalid") from error
    required = {
        "runner",
        "expected_source",
        "expected_assets",
        "source_revision",
        "modal_client_version",
        "resolved_modal_image_id",
        "sandbox_id",
        "resources",
    }
    if not isinstance(control, dict) or set(control) != required:
        raise D4CaptureError("Modal VM Sandbox control record has an invalid shape")
    result = _run_d4_capture(
        expected_source=control["expected_source"],
        expected_assets=control["expected_assets"],
        source_revision=control["source_revision"],
        modal_client_version=control["modal_client_version"],
        sandbox_control=control,
        probe_only=probe_only,
    )
    output_path.write_bytes(result)


def _sandbox_control(
    *,
    source: dict[str, Any],
    assets: dict[str, Any],
    revision: str,
    image_id: str,
    sandbox_id: str,
) -> dict[str, Any]:
    return {
        "runner": SANDBOX_RUNTIME,
        "expected_source": source,
        "expected_assets": assets,
        "source_revision": revision,
        "modal_client_version": importlib.metadata.version("modal"),
        "resolved_modal_image_id": image_id,
        "sandbox_id": sandbox_id,
        "resources": {
            "cpu_request_and_hard_limit": list(SANDBOX_CPU),
            "memory_request_and_hard_limit_mib": list(SANDBOX_MEMORY_MIB),
            "vm_runtime": True,
            "nonpreemptible": True,
            "single_use": True,
        },
    }


def _execute_in_vm_sandbox(
    *,
    source: dict[str, Any],
    assets: dict[str, Any],
    revision: str,
    probe_only: bool,
    lifecycle_path: Path | None = None,
) -> tuple[bytes, dict[str, Any]]:
    if modal is None or app is None or d4_image is None:
        raise D4CaptureError("the Modal client is required to allocate the D4 VM Sandbox")
    hydrated_image = d4_image.hydrate()
    image_id = hydrated_image.object_id
    if not isinstance(image_id, str) or not image_id.startswith("im-"):
        raise D4CaptureError("Modal did not resolve the D4 image to an immutable image ID")
    pinned_image = modal.Image.from_id(image_id)
    sandbox = None
    created_at = datetime.now(UTC)
    lifecycle: dict[str, Any] = {
        "schema_id": "qwen-mm-d4-modal-vm-lifecycle-v1",
        "schema_version": 1,
        "runner": SANDBOX_RUNTIME,
        "resolved_modal_image_id": image_id,
        "probe_only": probe_only,
        "create_once": True,
        "terminate_wait": True,
        "detach_in_finally": True,
        "created_at": created_at.isoformat(),
    }
    try:
        sandbox = modal.Sandbox.create(
            "sleep",
            "infinity",
            app=app,
            name=f"d4-{revision[:12]}-{uuid.uuid4().hex[:12]}",
            image=pinned_image,
            cpu=SANDBOX_CPU,
            memory=SANDBOX_MEMORY_MIB,
            timeout=SANDBOX_TIMEOUT_SECONDS,
            idle_timeout=SANDBOX_IDLE_TIMEOUT_SECONDS,
            workdir=str(REMOTE_ROOT),
            experimental_options=VM_EXPERIMENTAL_OPTIONS,
        )
        sandbox_id = sandbox.object_id
        if not isinstance(sandbox_id, str) or not sandbox_id.startswith("sb-"):
            raise D4CaptureError("Modal did not return a Sandbox allocation ID")
        lifecycle["sandbox_id"] = sandbox_id
        control = _sandbox_control(
            source=source,
            assets=assets,
            revision=revision,
            image_id=image_id,
            sandbox_id=sandbox_id,
        )
        sandbox.filesystem.write_text(
            json.dumps(control, indent=2, sort_keys=True) + "\n", SANDBOX_CONTROL_PATH
        )
        output_path = SANDBOX_PROBE_PATH if probe_only else SANDBOX_OUTPUT_PATH
        command = [
            str(REMOTE_PYTHON),
            str(REMOTE_ROOT / "scripts/modal_d4.py"),
            "--sandbox-worker",
            "--control",
            SANDBOX_CONTROL_PATH,
            "--worker-output",
            output_path,
        ]
        if probe_only:
            command.append("--probe-only")
        process = sandbox.exec(*command, timeout=600 if probe_only else SANDBOX_TIMEOUT_SECONDS)
        exit_code = process.wait()
        stdout = process.stdout.read()
        stderr = process.stderr.read()
        lifecycle["worker_exit_code"] = exit_code
        lifecycle["worker_stdout"] = stdout
        lifecycle["worker_stderr"] = stderr
        if exit_code != 0:
            raise D4CaptureError(
                f"Modal VM Sandbox worker failed with exit {exit_code}: {stderr or stdout}"
            )
        payload = sandbox.filesystem.read_bytes(output_path)
    finally:
        if sandbox is not None:
            try:
                lifecycle["terminate_result"] = sandbox.terminate(wait=True)
                exit_code = sandbox.poll()
                lifecycle["poll_after_terminate"] = exit_code
                if exit_code is None:
                    raise D4CaptureError(
                        "Modal VM Sandbox remained live after terminate(wait=True)"
                    )
                lifecycle["terminated"] = True
            finally:
                sandbox.detach()
                lifecycle["detached"] = True
                lifecycle["completed_at"] = datetime.now(UTC).isoformat()
                if lifecycle_path is not None:
                    lifecycle_path.parent.mkdir(parents=True, exist_ok=True)
                    lifecycle_path.write_text(
                        json.dumps(lifecycle, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
    return payload, lifecycle


def main(
    output: str = "/tmp/qwen-mm-d4-x86_64.zip",
    dry_run: bool = False,
    short_probe: bool = False,
    probe_output: str = "/tmp/qwen-mm-d4-modal-vm-probe.json",
) -> None:
    source = source_payload_identity(LOCAL_ROOT)
    assets = assets_identity(LOCAL_ASSETS_ROOT)
    revision = _git("rev-parse", "HEAD")
    dirty = _git("status", "--short", "--untracked-files=all")
    plan = _sandbox_plan(source=source, assets=assets, source_revision=revision)
    if dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if dirty and not short_probe:
        raise D4CaptureError("D4 Modal capture requires a clean checkout")
    destination = Path(probe_output if short_probe else output).expanduser().resolve()
    lifecycle_path = destination.with_suffix(
        destination.suffix + (".lifecycle.json" if short_probe else ".modal-lifecycle.json")
    )
    payload, lifecycle = _execute_in_vm_sandbox(
        source=source,
        assets=assets,
        revision=revision,
        probe_only=short_probe,
        lifecycle_path=lifecycle_path,
    )
    if short_probe:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        print(f"wrote terminated Modal VM Sandbox probe: {destination}")
        return
    write_capture_archive(
        payload,
        destination,
        expected_source=source,
        expected_assets=assets,
        phase_c_assets_root=LOCAL_ASSETS_ROOT,
    )
    print(f"wrote validated D4 x86_64 raw capture: {destination}")


if app is not None:
    main = app.local_entrypoint()(main)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sandbox-worker", action="store_true")
    parser.add_argument("--control", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--probe-only", action="store_true")
    arguments = parser.parse_args()
    if not arguments.sandbox_worker or arguments.control is None or arguments.worker_output is None:
        parser.error("direct execution is reserved for the internal --sandbox-worker path")
    _worker_main(
        control_path=arguments.control,
        output_path=arguments.worker_output,
        probe_only=arguments.probe_only,
    )
