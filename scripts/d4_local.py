"""Build and capture the two D4 variants on the controlled local ARM host."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from d4_capture_support import (  # noqa: E402
    BUILD_LABELS,
    COMPACT_BUILD_LABELS,
    COMPACT_CASES_BY_THREAD_BUDGET,
    COMPACT_SUBPROCESS_COUNT,
    COMPACT_THREAD_BUDGETS,
    PYPI_OVERRIDE_ENVIRONMENT_NAMES,
    THREAD_BUDGETS,
    D4CaptureError,
    assert_build_variant_artifacts,
    assert_source_payload_matches_revision,
    assets_identity,
    build_environment_evidence,
    capture_input_identities,
    create_capture_archive,
    local_affinity_provenance,
    normalized_capture_environment,
    reconcile_installed_runtime,
    reference_sync_command,
    reference_sync_evidence,
    source_payload_identity,
    toolchain_pins,
    verify_private_environment_integrity,
    wheel_build_command,
    write_capture_archive,
)
from d4_worker import compact_capture_plan, run_compact_capture  # noqa: E402
from profile_capture_support import install_wheel_command  # noqa: E402

REPOSITORY_ROOT = SCRIPT_DIRECTORY.parent
ASSETS_ROOT = REPOSITORY_ROOT / "reference/.cache/huggingface"
BASE_PYTHON = REPOSITORY_ROOT / ".venv/bin/python"


@contextmanager
def _capture_workspace() -> Iterator[Path]:
    """Retain exact partial evidence on failure, but remove successful scratch data."""

    root = Path(tempfile.mkdtemp(prefix="qwen-mm-d4-local-"))
    try:
        yield root
    except BaseException:
        print(
            f"D4 local ARM capture failed; retained diagnostic workspace: {root}",
            file=sys.stderr,
        )
        raise
    else:
        shutil.rmtree(root)


def _run_logged(command: list[str], *, log: Path, environment: dict[str, str]) -> str:
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as output:
        output.write("$ " + shlex.join(command) + "\n" + completed.stdout)
        if not completed.stdout.endswith("\n"):
            output.write("\n")
    if completed.returncode != 0:
        raise RuntimeError(
            f"D4 build command failed ({completed.returncode}): {shlex.join(command)}\n"
            f"{completed.stdout[-4000:]}"
        )
    return completed.stdout.strip()


def _capture(command: list[str], *, environment: dict[str, str]) -> str:
    return subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    ).stdout.strip()


def _runtime_identity(python: Path, *, environment: dict[str, str]) -> dict[str, Any]:
    script = """
import hashlib
import importlib.metadata
import json
from pathlib import Path
import qwen_mm
import qwen_mm._native as native
path = Path(native.__file__).resolve()
print(json.dumps({
    "package_version": importlib.metadata.version("qwen-mm"),
    "package_origin": str(Path(qwen_mm.__file__).resolve()),
    "native_origin": str(path),
    "native_bytes": path.stat().st_size,
    "native_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
}))
    """
    value = json.loads(_capture([str(python), "-c", script], environment=environment))
    venv_root = python.parent.parent.resolve()
    try:
        Path(value["native_origin"]).resolve().relative_to(venv_root)
    except ValueError as error:
        raise D4CaptureError(
            "installed qwen-mm native module escaped its separate build venv"
        ) from error
    return value


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    ).stdout.strip()


def _sysctl(name: str, *, environment: dict[str, str]) -> str:
    return _capture(["sysctl", "-n", name], environment=environment)


def assert_local_baseline(*, system: str, machine: str, cpu_model: str) -> None:
    """Fail before building unless this is the declared current M4 baseline."""

    if system != "Darwin" or machine.lower() not in {"arm64", "aarch64"}:
        raise D4CaptureError("local D4 runner requires the controlled native macOS ARM host")
    if "M4" not in cpu_model:
        raise D4CaptureError("local D4 runner requires the current Apple M4 baseline")


def build_plan(working_root: Path) -> dict[str, Any]:
    builds: dict[str, Any] = {}
    for label in BUILD_LABELS:
        venv = working_root / "venvs" / label
        wheel_directory = working_root / "wheels" / label
        cargo_target = working_root / "cargo-target" / label
        builds[label] = {
            "venv": str(venv),
            "create_venv": [
                "uv",
                "venv",
                "--python",
                str(BASE_PYTHON),
                str(venv),
            ],
            "sync": reference_sync_command(venv=venv),
            "build": wheel_build_command(
                python=venv / "bin/python",
                output=wheel_directory,
                build_label=label,
                cargo_target_dir=cargo_target,
            ),
        }
    return {
        "architecture": "arm64",
        "builds": builds,
        "affinity": local_affinity_provenance(),
        "thread_budgets": list(THREAD_BUDGETS),
    }


def compact_build_plan(working_root: Path) -> dict[str, Any]:
    """Expose the complete shipping-only compact build and capture plan."""

    shipping = build_plan(working_root)["builds"]["shipping"]
    capture = compact_capture_plan(
        python=Path(shipping["venv"]) / "bin/python",
        wheel=working_root / "retained/shipping/qwen_mm.whl",
        build_label="shipping",
        assets_root=ASSETS_ROOT,
        output_root=working_root / "artifact",
        affinity_masks={budget: None for budget in COMPACT_THREAD_BUDGETS},
    )
    return {
        "suite": "compact",
        "architecture": "arm64",
        "builds": {"shipping": shipping},
        "capture": capture,
        "affinity": local_affinity_provenance(),
        "thread_budgets": list(COMPACT_THREAD_BUDGETS),
        "cases_by_thread_budget": {
            f"t{budget}": list(COMPACT_CASES_BY_THREAD_BUDGET[budget])
            for budget in COMPACT_THREAD_BUDGETS
        },
        "subprocess_count": COMPACT_SUBPROCESS_COUNT,
    }


def _collect_files(artifact_root: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for path in sorted(artifact_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(artifact_root)
        files[relative.as_posix()] = path.read_bytes()
    return files


def execute_capture(*, output: Path, host_label: str) -> None:
    system = platform.system()
    machine = platform.machine()
    if system != "Darwin" or machine.lower() not in {"arm64", "aarch64"}:
        raise D4CaptureError("local D4 runner requires the controlled native macOS ARM host")
    dirty = _git("status", "--short", "--untracked-files=all")
    if dirty:
        raise D4CaptureError("D4 capture requires a clean checkout")
    if not BASE_PYTHON.is_file():
        raise D4CaptureError(f"locked base environment is missing: {BASE_PYTHON}")

    source_revision = _git("rev-parse", "HEAD")
    source = source_payload_identity(REPOSITORY_ROOT)
    assert_source_payload_matches_revision(REPOSITORY_ROOT, source_revision, source)
    assets = assets_identity(ASSETS_ROOT)
    environment = normalized_capture_environment(os.environ)
    cpu_model = _sysctl("machdep.cpu.brand_string", environment=environment)
    assert_local_baseline(system=system, machine=machine, cpu_model=cpu_model)
    cargo_bin = _capture(
        [str(REPOSITORY_ROOT / "scripts/cargo.sh"), "--print-bin-dir"],
        environment=environment,
    )
    environment["PATH"] = f"{cargo_bin}{os.pathsep}{environment.get('PATH', '')}"
    normalized_build_environment = build_environment_evidence(environment)
    build_host = {
        "os_release": {
            "command": ["sw_vers"],
            "output": _capture(["sw_vers"], environment=environment),
        },
        "uname": {
            "command": ["uname", "-a"],
            "output": _capture(["uname", "-a"], environment=environment),
        },
    }
    started = datetime.now(UTC)
    with _capture_workspace() as temporary_root:
        working_root = temporary_root / "working"
        artifact_root = temporary_root / "artifact"
        plan = compact_build_plan(working_root)
        native_hashes: dict[str, str] = {}
        wheel_hashes: dict[str, str] = {}
        for label in COMPACT_BUILD_LABELS:
            commands = plan["builds"][label]
            build_log = artifact_root / "logs" / label / "build.log"
            sync_log = artifact_root / "logs" / label / "sync.log"
            _run_logged(commands["create_venv"], log=build_log, environment=environment)
            _run_logged(commands["sync"], log=sync_log, environment=environment)
            _run_logged(commands["build"], log=build_log, environment=environment)
            wheel_directory = working_root / "wheels" / label
            wheels = sorted(wheel_directory.glob("*.whl"))
            if len(wheels) != 1:
                raise D4CaptureError(f"{label}: expected exactly one wheel, got {len(wheels)}")
            python = working_root / "venvs" / label / "bin/python"
            retained_wheel = artifact_root / "builds" / label / wheels[0].name
            retained_wheel.parent.mkdir(parents=True, exist_ok=True)
            retained_wheel.write_bytes(wheels[0].read_bytes())
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
            run_compact_capture(
                python=python,
                wheel=retained_wheel,
                assets_root=ASSETS_ROOT,
                output_root=artifact_root,
                affinity_masks={budget: None for budget in COMPACT_THREAD_BUDGETS},
                execute=True,
            )
            build_path = artifact_root / "builds" / label / "build.json"
            build = json.loads(build_path.read_text(encoding="utf-8"))
            packages = _capture(
                ["uv", "pip", "freeze", "--python", str(python)],
                environment=environment,
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
                        venv=Path(commands["venv"]),
                        environment=environment,
                        log=sync_log,
                    ),
                    "packages": packages,
                }
            )
            build_path.write_text(
                json.dumps(build, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        assert_build_variant_artifacts(
            {"builds": plan["builds"]},
            native_hashes=native_hashes,
            wheel_hashes=wheel_hashes,
            build_labels=COMPACT_BUILD_LABELS,
        )
        for label in COMPACT_BUILD_LABELS:
            verify_private_environment_integrity(
                working_root / "venvs" / label,
                evidence_path=artifact_root / "builds" / label / "environment-integrity.json",
                checkpoint="before-archive",
            )

        completed = datetime.now(UTC)
        provenance = {
            "schema_id": "qwen-mm-d4-raw-capture-provenance-v1",
            "schema_version": 1,
            "suite": "compact",
            "claim": "raw controlled-host input for the separate D4 certification evaluator",
            "architecture_family": "arm64",
            "host_label": host_label,
            "started_at": started.isoformat(),
            "completed_at": completed.isoformat(),
            "source_revision": source_revision,
            "source": source,
            "assets": assets,
            "capture_inputs": capture_input_identities(REPOSITORY_ROOT),
            "host": {
                "system": system,
                "machine": machine,
                "platform": platform.platform(),
                "uname": " ".join(platform.uname()),
                "hostname": socket.gethostname(),
                "cpu_model": cpu_model,
                "physical_cpu_count": int(_sysctl("hw.physicalcpu", environment=environment)),
                "logical_cpu_count": int(_sysctl("hw.logicalcpu", environment=environment)),
                "memory_bytes": int(_sysctl("hw.memsize", environment=environment)),
                "affinity": local_affinity_provenance(),
            },
            "build_wheel_sha256": wheel_hashes,
            "build_native_sha256": native_hashes,
            "toolchain_pins": toolchain_pins(suite="compact"),
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
            "suite": "compact",
            "architecture_family": "arm64",
            "build_labels": list(COMPACT_BUILD_LABELS),
            "thread_budgets": list(COMPACT_THREAD_BUDGETS),
            "cases_by_thread_budget": {
                f"t{budget}": list(COMPACT_CASES_BY_THREAD_BUDGET[budget])
                for budget in COMPACT_THREAD_BUDGETS
            },
            "subprocess_count": COMPACT_SUBPROCESS_COUNT,
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
    parser.add_argument("--output", type=Path, default=Path("/tmp/qwen-mm-d4-arm64.zip"))
    parser.add_argument("--host-label", default="local-m4")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(compact_build_plan(Path("/tmp/qwen-mm-d4-plan")), indent=2, sort_keys=True)
        )
        return
    execute_capture(output=args.output.expanduser().resolve(), host_label=args.host_label)
    print(f"wrote validated D4 ARM raw capture: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
