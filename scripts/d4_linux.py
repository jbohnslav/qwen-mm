"""Build and capture both D4 variants on a controlled local Linux x86 host."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from d4_capture_support import (  # noqa: E402
    BUILD_LABELS,
    LOCAL_LINUX_AFFINITY_PROBE_SECONDS,
    LOCAL_LINUX_AFFINITY_PROBE_THREADS,
    LOCAL_LINUX_CPU_WALL_RATIO_MAX,
    PYPI_OVERRIDE_ENVIRONMENT_NAMES,
    THREAD_BUDGETS,
    D4CaptureError,
    assert_build_invariants,
    assert_build_variant_artifacts,
    assert_source_payload_matches_revision,
    assets_identity,
    build_environment_evidence,
    capture_input_identities,
    create_capture_archive,
    local_linux_resource_attestation,
    local_linux_toolchain_pins,
    normalize_build_artifact_paths,
    normalized_capture_environment,
    parse_cpu_list,
    physical_core_masks,
    reconcile_installed_runtime,
    reference_sync_command,
    reference_sync_evidence,
    source_payload_identity,
    verify_private_environment_integrity,
    wheel_build_command,
    write_capture_archive,
)
from d4_local import (  # noqa: E402
    _capture,
    _collect_files,
    _git,
    _run_logged,
    _runtime_identity,
)
from d4_worker import run_capture  # noqa: E402
from profile_capture_support import install_wheel_command  # noqa: E402

REPOSITORY_ROOT = SCRIPT_DIRECTORY.parent
ASSETS_ROOT = REPOSITORY_ROOT / "reference/.cache/huggingface"
BASE_PYTHON = REPOSITORY_ROOT / ".venv/bin/python"
CGROUP_ROOT = Path("/sys/fs/cgroup")
CPU_SYSFS_ROOT = Path("/sys/devices/system/cpu")
CGROUP_NAMES = ("cpu.max", "cpuset.cpus.effective", "memory.max")
LOCAL_RESOURCE_MODE = "cgroup_v2_cpuset"
AFFINITY_PROBE_METHOD = "taskset-t1-python-native-threads-v1"
AFFINITY_PROBE_SCRIPT = r"""
import ctypes
import hashlib
import json
import os
import sys
import threading
import time

thread_count = int(sys.argv[1])
probe_seconds = float(sys.argv[2])
libc = ctypes.CDLL(None)
libc.sched_getcpu.argtypes = []
libc.sched_getcpu.restype = ctypes.c_int
barrier = threading.Barrier(thread_count + 1)
reports = [None] * thread_count

def worker(index):
    native_thread_id = threading.get_native_id()
    affinity = sorted(os.sched_getaffinity(native_thread_id))
    barrier.wait()
    deadline = time.perf_counter() + probe_seconds
    observed_cpus = set()
    iterations = 0
    while time.perf_counter() < deadline:
        observed_cpus.add(libc.sched_getcpu())
        hashlib.pbkdf2_hmac("sha256", b"qwen-mm-d4", b"affinity-probe", 100000)
        observed_cpus.add(libc.sched_getcpu())
        iterations += 1
    final_affinity = sorted(os.sched_getaffinity(native_thread_id))
    if final_affinity != affinity:
        raise RuntimeError("worker affinity changed during the probe")
    reports[index] = {
        "worker_index": index,
        "native_thread_id": native_thread_id,
        "affinity": affinity,
        "observed_cpus": sorted(observed_cpus),
        "iterations": iterations,
    }

threads = [threading.Thread(target=worker, args=(index,)) for index in range(thread_count)]
for thread in threads:
    thread.start()
wall_started = time.perf_counter()
cpu_started = time.process_time()
barrier.wait()
for thread in threads:
    thread.join()
wall_seconds = time.perf_counter() - wall_started
process_cpu_seconds = time.process_time() - cpu_started
print(json.dumps({
    "native_thread_count": thread_count,
    "wall_seconds": wall_seconds,
    "process_cpu_seconds": process_cpu_seconds,
    "process_cpu_wall_ratio": process_cpu_seconds / wall_seconds,
    "worker_reports": reports,
}, sort_keys=True))
"""


@contextmanager
def _capture_workspace() -> Iterator[Path]:
    """Retain exact partial evidence on failure, but remove successful scratch data."""

    root = Path(tempfile.mkdtemp(prefix="qwen-mm-d4-linux-"))
    try:
        yield root
    except BaseException:
        print(
            f"D4 local Linux capture failed; retained diagnostic workspace: {root}",
            file=sys.stderr,
        )
        raise
    else:
        shutil.rmtree(root)


def assert_linux_baseline(*, system: str, machine: str, cpu_description: str) -> None:
    """Reject non-Linux, non-x86, and explicitly emulated CPU environments."""

    if system != "Linux":
        raise D4CaptureError(f"local Linux D4 runner requires Linux, got {system!r}")
    if machine.lower() not in {"amd64", "x86_64"}:
        raise D4CaptureError(f"local Linux D4 runner requires native x86_64, got {machine!r}")
    normalized = cpu_description.lower()
    emulation_markers = ("qemu virtual cpu", "tcg cpu", "software emulation")
    marker = next((value for value in emulation_markers if value in normalized), None)
    if marker is not None:
        raise D4CaptureError(f"local Linux CPU description contains emulation marker {marker!r}")


def _read_cgroup_limits(root: Path = CGROUP_ROOT) -> dict[str, str]:
    values: dict[str, str] = {}
    for name in CGROUP_NAMES:
        path = root / name
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise D4CaptureError(f"local Linux capture requires cgroup v2 file {path}") from error
        if not value:
            raise D4CaptureError(f"local Linux cgroup value is empty: {path}")
        values[str(path)] = value
    return values


def _memory_total_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    raise D4CaptureError("/proc/meminfo did not expose MemTotal")


def _cpu_description(lscpu_json: Mapping[str, Any]) -> str:
    description = " ".join(
        str(item.get("data", ""))
        for item in lscpu_json.get("lscpu", [])
        if isinstance(item, Mapping) and item.get("field") == "Model name:"
    ).strip()
    if description:
        return description
    return Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")


def _physical_core_count(lscpu_parse: str, *, allowed_cpus: Sequence[int]) -> int:
    allowed = set(allowed_cpus)
    if not allowed:
        raise D4CaptureError("local Linux allocation exposes no allowed CPUs")
    seen: set[int] = set()
    physical: set[tuple[int, int]] = set()
    for raw_line in lscpu_parse.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        columns = line.split(",")
        if len(columns) != 4:
            raise D4CaptureError("lscpu topology row must contain CPU,CORE,SOCKET,ONLINE")
        try:
            cpu, core, socket_id = (int(columns[index]) for index in range(3))
        except ValueError as error:
            raise D4CaptureError(f"invalid lscpu topology row: {line!r}") from error
        if cpu in seen:
            raise D4CaptureError(f"lscpu topology repeats logical CPU {cpu}")
        seen.add(cpu)
        if cpu in allowed and columns[3].strip().lower() in {"y", "yes", "1", "true"}:
            physical.add((socket_id, core))
    if not allowed.issubset(seen):
        missing = sorted(allowed - seen)
        raise D4CaptureError(f"allowed CPUs are missing from lscpu topology: {missing}")
    if len(physical) < max(THREAD_BUDGETS):
        raise D4CaptureError(
            f"local Linux allocation exposes {len(physical)} physical cores; "
            f"{max(THREAD_BUDGETS)} required"
        )
    return len(physical)


def _power_policy_snapshot(
    visible_affinity: Sequence[int], *, root: Path = CPU_SYSFS_ROOT
) -> dict[str, dict[str, str]]:
    """Capture stable CPU policy knobs without sampling dynamic frequencies."""

    paths: dict[str, str] = {}
    unavailable: dict[str, str] = {}
    per_cpu_groups = (
        "scaling_governor",
        "energy_performance_preference",
        "scaling_min_freq",
        "scaling_max_freq",
    )
    for group in per_cpu_groups:
        found = False
        for cpu in visible_affinity:
            path = root / f"cpu{cpu}/cpufreq/{group}"
            try:
                value = path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if value:
                paths[str(path)] = value
                found = True
        if not found:
            unavailable[group] = "no nonempty readable endpoint for any visible CPU"
    global_groups = {
        "intel_pstate_no_turbo": root / "intel_pstate/no_turbo",
        "cpufreq_boost": root / "cpufreq/boost",
    }
    for group, path in global_groups.items():
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            value = ""
        if value:
            paths[str(path)] = value
        else:
            unavailable[group] = "missing, unreadable, or empty"
    return {"paths": paths, "unavailable": unavailable}


def _affinity_enforcement_phase(
    expected_affinity: Sequence[int],
    *,
    environment: Mapping[str, str],
    diagnostic_log: Path | None = None,
) -> dict[str, Any]:
    """Exercise native pthread affinity and one-core CPU usage under taskset."""

    expected = list(expected_affinity)
    if len(expected) != 1:
        raise D4CaptureError("affinity preflight requires the exact t1 mask")
    probes: list[dict[str, Any]] = []
    for thread_count in LOCAL_LINUX_AFFINITY_PROBE_THREADS:
        command = [
            "taskset",
            "--cpu-list",
            str(expected[0]),
            str(BASE_PYTHON),
            "-c",
            AFFINITY_PROBE_SCRIPT,
            str(thread_count),
            str(LOCAL_LINUX_AFFINITY_PROBE_SECONDS),
        ]
        try:
            raw_probe = _capture(command, environment=dict(environment))
        except subprocess.CalledProcessError as error:
            raw_probe = str(error.stdout or "")
            if diagnostic_log is not None:
                diagnostic_log.parent.mkdir(parents=True, exist_ok=True)
                with diagnostic_log.open("a", encoding="utf-8") as output:
                    output.write(f"native_thread_count={thread_count}\n{raw_probe}\n")
            raise D4CaptureError("local Linux affinity preflight command failed") from error
        try:
            if diagnostic_log is not None:
                diagnostic_log.parent.mkdir(parents=True, exist_ok=True)
                with diagnostic_log.open("a", encoding="utf-8") as output:
                    output.write(f"native_thread_count={thread_count}\n{raw_probe}\n")
            probe = json.loads(raw_probe)
        except (json.JSONDecodeError, OSError) as error:
            raise D4CaptureError("local Linux affinity preflight returned invalid JSON") from error
        reports = probe.get("worker_reports")
        try:
            wall_seconds = float(probe["wall_seconds"])
            process_cpu_seconds = float(probe["process_cpu_seconds"])
            ratio = float(probe["process_cpu_wall_ratio"])
        except (KeyError, TypeError, ValueError) as error:
            raise D4CaptureError("local Linux affinity preflight timings are invalid") from error
        if (
            probe.get("native_thread_count") != thread_count
            or not isinstance(reports, list)
            or len(reports) != thread_count
            or not all(math.isfinite(value) for value in (wall_seconds, process_cpu_seconds, ratio))
            or wall_seconds < LOCAL_LINUX_AFFINITY_PROBE_SECONDS
            or process_cpu_seconds <= 0
            or ratio <= 0
            or not math.isclose(
                ratio,
                process_cpu_seconds / wall_seconds,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            or ratio > LOCAL_LINUX_CPU_WALL_RATIO_MAX
        ):
            raise D4CaptureError("local Linux affinity preflight exceeded its frozen controls")
        native_ids: set[int] = set()
        for index, report in enumerate(reports):
            if not isinstance(report, Mapping):
                raise D4CaptureError("local Linux affinity worker report is invalid")
            native_id = report.get("native_thread_id")
            if (
                report.get("worker_index") != index
                or isinstance(native_id, bool)
                or not isinstance(native_id, int)
                or native_id <= 0
                or native_id in native_ids
                or report.get("affinity") != expected
                or report.get("observed_cpus") != expected
                or isinstance(report.get("iterations"), bool)
                or not isinstance(report.get("iterations"), int)
                or report["iterations"] <= 0
            ):
                raise D4CaptureError("local Linux affinity worker escaped the t1 mask")
            native_ids.add(native_id)
        probes.append(probe)
    return {"probes": probes}


def _affinity_enforcement(
    *, expected_affinity: Sequence[int], pre: Mapping[str, Any], post: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "method": AFFINITY_PROBE_METHOD,
        "expected_affinity": list(expected_affinity),
        "probe_seconds": LOCAL_LINUX_AFFINITY_PROBE_SECONDS,
        "cpu_wall_ratio_max": LOCAL_LINUX_CPU_WALL_RATIO_MAX,
        "pre": dict(pre),
        "post": dict(post),
    }


def linux_resource_attestation(
    *,
    cgroup_limits: Mapping[str, str],
    lscpu_parse: str,
    visible_affinity: Sequence[int],
    masks: Mapping[int, Sequence[int]],
    memory_total_bytes: int,
    power_policy: Mapping[str, Mapping[str, str]],
    affinity_enforcement: Mapping[str, Any] | None = None,
    dedicated_capture: bool,
    stable: bool,
) -> dict[str, Any]:
    """Validate and describe a controlled, cpuset-bound local allocation."""

    expected_paths = {str(CGROUP_ROOT / name) for name in CGROUP_NAMES}
    if set(cgroup_limits) != expected_paths:
        raise D4CaptureError("local Linux capture requires complete root cgroup v2 evidence")
    if not dedicated_capture or not stable:
        raise D4CaptureError(
            "local Linux capture requires explicit dedicated-capture and stable attestations"
        )
    visible = list(visible_affinity)
    if (
        not visible
        or visible != sorted(set(visible))
        or any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in visible)
    ):
        raise D4CaptureError("local Linux visible affinity must be sorted unique CPU IDs")
    expected_masks = set(THREAD_BUDGETS)
    if set(masks) != expected_masks:
        raise D4CaptureError("local Linux affinity masks do not cover t1/t2/t4/t8")
    recomputed_masks = physical_core_masks(lscpu_parse, allowed_cpus=visible)
    if {budget: tuple(mask) for budget, mask in masks.items()} != recomputed_masks:
        raise D4CaptureError("local Linux affinity masks differ from allowed CPU topology")
    allocated_physical_cores = _physical_core_count(lscpu_parse, allowed_cpus=visible)

    try:
        quota_text, period_text = cgroup_limits[str(CGROUP_ROOT / "cpu.max")].split()
        if quota_text == "max":
            raise ValueError
        quota = int(quota_text)
        period = int(period_text)
        memory_text = cgroup_limits[str(CGROUP_ROOT / "memory.max")]
        if memory_text == "max":
            raise ValueError
        allocated_memory_bytes = int(memory_text)
    except (TypeError, ValueError) as error:
        raise D4CaptureError(
            "local Linux capture requires finite cgroup CPU quota and memory limit"
        ) from error
    if period <= 0 or quota <= 0 or quota / period < allocated_physical_cores:
        raise D4CaptureError("cgroup CPU quota is smaller than the allocated physical-core cpuset")
    if (
        allocated_memory_bytes <= 0
        or memory_total_bytes <= 0
        or allocated_memory_bytes > memory_total_bytes
    ):
        raise D4CaptureError("cgroup memory limit exceeds available physical memory")
    cpuset = parse_cpu_list(cgroup_limits[str(CGROUP_ROOT / "cpuset.cpus.effective")])
    if cpuset != set(visible):
        raise D4CaptureError("cgroup effective cpuset differs from visible process affinity")

    attestation = local_linux_resource_attestation(
        cgroup_limits=cgroup_limits,
        allocated_physical_cores=allocated_physical_cores,
        allocated_memory_bytes=allocated_memory_bytes,
        exclusive_physical_cores=False,
        dedicated_capture=dedicated_capture,
        stable=stable,
        visible_affinity=visible,
        masks=recomputed_masks,
        power_policy=power_policy,
        affinity_enforcement=affinity_enforcement,
    )
    if attestation["mode"] != LOCAL_RESOURCE_MODE:
        raise D4CaptureError("local Linux support returned an unknown resource mode")
    return attestation


def build_plan(
    working_root: Path, *, affinity_masks: Mapping[int, Sequence[int]]
) -> dict[str, Any]:
    """Return the two isolated build plans and the complete affinity matrix."""

    if set(affinity_masks) != set(THREAD_BUDGETS):
        raise D4CaptureError("local Linux plan does not cover t1/t2/t4/t8")
    builds: dict[str, Any] = {}
    for label in BUILD_LABELS:
        venv = working_root / "venvs" / label
        wheel_directory = working_root / "wheels" / label
        cargo_target = working_root / "cargo-target" / label
        builds[label] = {
            "venv": str(venv),
            "create_venv": ["uv", "venv", "--python", str(BASE_PYTHON), str(venv)],
            "sync": reference_sync_command(venv=venv),
            "build": wheel_build_command(
                python=venv / "bin/python",
                output=wheel_directory,
                build_label=label,
                cargo_target_dir=cargo_target,
            ),
        }
    return {
        "architecture": "x86_64",
        "provider": "local_linux",
        "builds": builds,
        "affinity_masks": {f"t{budget}": list(affinity_masks[budget]) for budget in THREAD_BUDGETS},
        "thread_budgets": list(THREAD_BUDGETS),
    }


def _resource_snapshot(
    *, dedicated_capture: bool, stable: bool
) -> tuple[dict[str, Any], str, list[int], dict[int, tuple[int, ...]]]:
    environment = normalized_capture_environment(os.environ)
    lscpu_json = json.loads(_capture(["lscpu", "--json"], environment=environment))
    description = _cpu_description(lscpu_json)
    assert_linux_baseline(
        system=platform.system(), machine=platform.machine(), cpu_description=description
    )
    visible_affinity = sorted(os.sched_getaffinity(0))
    topology = _capture(["lscpu", "--parse=CPU,CORE,SOCKET,ONLINE"], environment=environment)
    masks = physical_core_masks(topology, allowed_cpus=visible_affinity)
    power_policy = _power_policy_snapshot(visible_affinity)
    attestation = linux_resource_attestation(
        cgroup_limits=_read_cgroup_limits(),
        lscpu_parse=topology,
        visible_affinity=visible_affinity,
        masks=masks,
        memory_total_bytes=_memory_total_bytes(),
        power_policy=power_policy,
        dedicated_capture=dedicated_capture,
        stable=stable,
    )
    host = {
        "system": platform.system(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "uname": _capture(["uname", "-a"], environment=environment),
        "hostname": socket.gethostname(),
        "cpu_description": description,
        "lscpu": lscpu_json,
        "lscpu_parse": topology,
        "logical_cpu_count": os.cpu_count(),
        "proc_meminfo_total_bytes": _memory_total_bytes(),
        "resource_attestation": attestation,
    }
    return host, topology, visible_affinity, masks


def execute_capture(
    *,
    output: Path,
    host_label: str,
    allocation_id: str,
    dedicated_capture: bool,
    stable: bool,
) -> None:
    """Build, measure, validate, and atomically write one local Linux raw ZIP."""

    if not host_label.strip() or not allocation_id.strip():
        raise D4CaptureError("local Linux host label and allocation ID must be non-empty")
    dirty = _git("status", "--short", "--untracked-files=all")
    if dirty:
        raise D4CaptureError("D4 capture requires a clean checkout")
    if not BASE_PYTHON.is_file():
        raise D4CaptureError(f"locked base environment is missing: {BASE_PYTHON}")

    source = source_payload_identity(REPOSITORY_ROOT)
    source_revision = _git("rev-parse", "HEAD")
    assert_source_payload_matches_revision(REPOSITORY_ROOT, source_revision, source)
    assets = assets_identity(ASSETS_ROOT)
    environment = normalized_capture_environment(os.environ)
    cargo_bin = _capture(
        [str(REPOSITORY_ROOT / "scripts/cargo.sh"), "--print-bin-dir"],
        environment=environment,
    )
    environment["PATH"] = f"{cargo_bin}{os.pathsep}{environment.get('PATH', '')}"
    normalized_build_environment = build_environment_evidence(environment)
    started = datetime.now(UTC)
    host, topology, visible_affinity, masks = _resource_snapshot(
        dedicated_capture=dedicated_capture, stable=stable
    )
    build_host = {
        "os_release": {
            "command": ["cat", "/etc/os-release"],
            "output": Path("/etc/os-release").read_text(encoding="utf-8").strip(),
        },
        "uname": {"command": ["uname", "-a"], "output": host["uname"]},
    }
    with _capture_workspace() as temporary_root:
        working_root = temporary_root / "working"
        artifact_root = temporary_root / "artifact"
        control_root = artifact_root / "logs" / "controls"
        control_root.mkdir(parents=True, exist_ok=True)
        preflight = _affinity_enforcement_phase(
            masks[1],
            environment=environment,
            diagnostic_log=control_root / "affinity-preflight.raw.log",
        )
        (control_root / "affinity-pre.json").write_text(
            json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        plan = build_plan(working_root, affinity_masks=masks)
        build_invariants: dict[str, dict[str, Any]] = {}
        native_hashes: dict[str, str] = {}
        wheel_hashes: dict[str, str] = {}
        for label in BUILD_LABELS:
            commands = plan["builds"][label]
            venv = Path(commands["venv"])
            wheel_directory = working_root / "wheels" / label
            build_log = artifact_root / "logs" / label / "build.log"
            sync_log = artifact_root / "logs" / label / "sync.log"
            _run_logged(commands["create_venv"], log=build_log, environment=environment)
            _run_logged(commands["sync"], log=sync_log, environment=environment)
            _run_logged(commands["build"], log=build_log, environment=environment)
            wheels = sorted(wheel_directory.glob("*.whl"))
            if len(wheels) != 1:
                raise D4CaptureError(f"{label}: expected exactly one wheel, got {len(wheels)}")
            python = venv / "bin/python"
            retained_wheel = artifact_root / "builds" / label / wheels[0].name
            retained_wheel.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wheels[0], retained_wheel)
            install_command = install_wheel_command(python=python, wheel=retained_wheel)
            _run_logged(install_command, log=build_log, environment=environment)
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
                assets_root=ASSETS_ROOT,
                output_root=artifact_root,
                affinity_masks=masks,
                execute=True,
            )
            build_path = artifact_root / "builds" / label / "build.json"
            build = json.loads(build_path.read_text(encoding="utf-8"))
            packages = _capture(
                ["uv", "pip", "freeze", "--python", str(python)], environment=environment
            ).splitlines()
            toolchain = {
                "python": {
                    "command": [str(python), "--version"],
                    "output": _capture([str(python), "--version"], environment=environment),
                },
                "maturin": {
                    "command": ["uv", "run", "--locked", "--no-sync", "maturin", "--version"],
                    "output": _capture(
                        ["uv", "run", "--locked", "--no-sync", "maturin", "--version"],
                        environment=environment,
                    ),
                },
                "rustc": {
                    "command": ["rustc", "--version", "--verbose"],
                    "output": _capture(
                        ["rustc", "--version", "--verbose"], environment=environment
                    ),
                },
                "cargo": {
                    "command": ["cargo", "--version", "--verbose"],
                    "output": _capture(
                        ["cargo", "--version", "--verbose"], environment=environment
                    ),
                },
            }
            build.update(
                {
                    "commands": {
                        "create_venv": commands["create_venv"],
                        "sync_reference": commands["sync"],
                        "build_wheel": commands["build"],
                        "install_retained_wheel": install_command,
                    },
                    "build_environment": normalized_build_environment,
                    "build_host": build_host,
                    "toolchain": toolchain,
                    "runtime": runtime,
                    "runtime_reconciliation": runtime_reconciliation,
                    "reference_sync": reference_sync_evidence(
                        venv=venv, environment=environment, log=sync_log
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
        assert_build_variant_artifacts(plan, native_hashes=native_hashes, wheel_hashes=wheel_hashes)

        postflight = _affinity_enforcement_phase(
            masks[1],
            environment=environment,
            diagnostic_log=control_root / "affinity-postflight.raw.log",
        )
        (control_root / "affinity-post.json").write_text(
            json.dumps(postflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        final_host, final_topology, final_affinity, final_masks = _resource_snapshot(
            dedicated_capture=dedicated_capture, stable=stable
        )
        if (
            final_topology != topology
            or final_affinity != visible_affinity
            or final_masks != masks
            or final_host["resource_attestation"] != host["resource_attestation"]
        ):
            raise D4CaptureError("local Linux allocation changed during D4 capture")
        controls = _affinity_enforcement(expected_affinity=masks[1], pre=preflight, post=postflight)
        initial_attestation = host["resource_attestation"]
        host["resource_attestation"] = linux_resource_attestation(
            cgroup_limits=initial_attestation["cgroup_limits"],
            lscpu_parse=topology,
            visible_affinity=visible_affinity,
            masks=masks,
            memory_total_bytes=host["proc_meminfo_total_bytes"],
            power_policy=initial_attestation["power_policy"],
            affinity_enforcement=controls,
            dedicated_capture=dedicated_capture,
            stable=stable,
        )
        for label in BUILD_LABELS:
            verify_private_environment_integrity(
                working_root / "venvs" / label,
                evidence_path=artifact_root / "builds" / label / "environment-integrity.json",
                checkpoint="before-archive",
            )

        completed = datetime.now(UTC)
        provenance = {
            "schema_id": "qwen-mm-d4-raw-capture-provenance-v1",
            "schema_version": 1,
            "claim": "raw controlled-host input for the separate D4 certification evaluator",
            "architecture_family": "x86_64",
            "provider": "local_linux",
            "started_at": started.isoformat(),
            "completed_at": completed.isoformat(),
            "source_revision": source_revision,
            "source": source,
            "assets": assets,
            "capture_inputs": capture_input_identities(REPOSITORY_ROOT),
            "host": host,
            "local_linux": {
                "host_label": host_label.strip(),
                "allocation_id": allocation_id.strip(),
            },
            "build_wheel_sha256": wheel_hashes,
            "build_native_sha256": native_hashes,
            "toolchain_pins": local_linux_toolchain_pins(),
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
        files = _collect_files(artifact_root)
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
        archive = create_capture_archive(files)
        write_capture_archive(
            archive,
            output,
            expected_source=source,
            expected_assets=assets,
            phase_c_assets_root=ASSETS_ROOT,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/qwen-mm-d4-x86_64.zip"))
    parser.add_argument("--host-label", default="local-linux-x86")
    parser.add_argument(
        "--allocation-id",
        default="",
        help="stable allocation/container identifier; defaults to the runtime hostname",
    )
    parser.add_argument(
        "--dedicated-capture",
        action="store_true",
        help="attest this is the only D4 capture workload in the controlled window",
    )
    parser.add_argument(
        "--stable",
        action="store_true",
        help="attest allocation and host performance policy remain stable for the full run",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        masks = {budget: tuple(range(budget)) for budget in THREAD_BUDGETS}
        print(
            json.dumps(
                build_plan(Path("/tmp/qwen-mm-d4-linux-plan"), affinity_masks=masks),
                indent=2,
                sort_keys=True,
            )
        )
        return
    execute_capture(
        output=args.output.expanduser().resolve(),
        host_label=args.host_label,
        allocation_id=args.allocation_id or socket.gethostname(),
        dedicated_capture=args.dedicated_capture,
        stable=args.stable,
    )
    print(f"wrote validated D4 local Linux x86_64 raw capture: {args.output.resolve()}")


if __name__ == "__main__":
    main()
