"""Capture the native macOS ARM Phase D1 evidence archive from a clean commit."""

from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

LOCAL_ROOT = Path(__file__).resolve().parents[1]
ASSETS_ROOT = LOCAL_ROOT / "reference/.cache/huggingface"
PYTHON = LOCAL_ROOT / ".venv/bin/python"

sys.path.insert(0, str(LOCAL_ROOT / "scripts"))
import profile_capture_support as support  # noqa: E402
from modal_benchmark_support import committed_source_tree_digest  # noqa: E402


def _run_logged(command: list[str], *, log_path: Path, environment: dict[str, str]) -> None:
    result = subprocess.run(
        command,
        cwd=LOCAL_ROOT,
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
        raise RuntimeError(
            f"command failed with exit code {result.returncode}; see {log_path}: "
            f"{shlex.join(command)}"
        )


def _capture(command: list[str]) -> str:
    return subprocess.run(
        command,
        cwd=LOCAL_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    ).stdout.strip()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return value


def _git_output(*arguments: str, root: Path = LOCAL_ROOT) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    ).stdout.strip()


def clean_committed_source_identity(root: Path = LOCAL_ROOT) -> tuple[str, str, int, int]:
    status = _git_output("status", "--porcelain", "--untracked-files=all", root=root)
    if status:
        raise RuntimeError("local D1 capture requires the clean implementation commit")
    revision = _git_output("rev-parse", "HEAD", root=root)
    digest, source_files, source_bytes = committed_source_tree_digest(root, revision)
    return revision, digest, source_files, source_bytes


def _candidate_identity() -> dict[str, Any]:
    program = (
        "import json; "
        "from qwen_mm_reference.benchmark_v2 import candidate_artifact_identity; "
        "print(json.dumps(candidate_artifact_identity('qwen_mm.benchmark:create_adapter'), "
        "sort_keys=True))"
    )
    value = json.loads(_capture([str(PYTHON), "-c", program]))
    if not isinstance(value, dict):
        raise RuntimeError("installed candidate identity is not an object")
    return value


def _sample_version() -> str:
    output = _capture(["/usr/bin/what", "/usr/bin/sample"])
    version = next((line.strip() for line in output.splitlines() if "PROGRAM:sample" in line), "")
    if not version:
        raise RuntimeError("cannot authenticate /usr/bin/sample")
    return version


def _preflight_sample(*, log_path: Path, environment: dict[str, str]) -> None:
    with tempfile.TemporaryDirectory(prefix="qwen-mm-sample-preflight-") as directory:
        output = Path(directory) / "sample.txt"
        worker = subprocess.Popen(
            [str(PYTHON), "-c", "import qwen_mm._native, time; time.sleep(5)"],
            cwd=LOCAL_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        command = [
            "/usr/bin/sample",
            str(worker.pid),
            "1",
            "10",
            "-mayDie",
            "-file",
            str(output),
        ]
        result = subprocess.run(
            command,
            cwd=LOCAL_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        worker.terminate()
        worker_stdout, worker_stderr = worker.communicate(timeout=10)
        rendered = (
            "$ "
            + shlex.join(command)
            + "\n"
            + result.stdout
            + f"\nworker stdout: {worker_stdout}\nworker stderr: {worker_stderr}\n"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(rendered, encoding="utf-8")
        if result.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(
                f"native macOS sample preflight failed before the D1 matrix; see {log_path}"
            )


def capture_arm_archive(output: Path) -> dict[str, Any]:
    if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        raise RuntimeError("the local D1 lane requires a native macOS ARM64 host")
    revision, digest, source_files, source_bytes = clean_committed_source_identity()
    assets = support.directory_identity(ASSETS_ROOT)
    started_at = datetime.now(UTC).isoformat()
    paths = support.host_paths("arm64")

    capture_root = Path(tempfile.mkdtemp(prefix="qwen-mm-d1-arm-capture-"))
    phase_c_scratch = Path(tempfile.mkdtemp(prefix="qwen-mm-phase-c-scratch-"))

    def path(relative: PurePosixPath) -> Path:
        return capture_root / str(relative)

    environment = dict(os.environ)
    for name in (
        "PIP_EXTRA_INDEX_URL",
        "PIP_INDEX_URL",
        "PYTHONPATH",
        "UV_EXTRA_INDEX_URL",
        "UV_INDEX",
        "UV_INDEX_URL",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            "QWEN_MM_ASSETS_ROOT": str(ASSETS_ROOT),
        }
    )
    try:
        wheel_directory = path(paths["publish"])
        wheel_directory.mkdir(parents=True)
        build = support.profile_build_command(python=PYTHON, wheel_directory=wheel_directory)
        build_log = path(paths["build_log"])
        _run_logged(build, log_path=build_log, environment=environment)
        wheels = sorted(wheel_directory.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"profile build produced {len(wheels)} wheels; expected one")
        wheel = wheels[0]
        wheel_contents = support.wheel_contents_identity(wheel)

        install = support.install_wheel_command(python=PYTHON, wheel=wheel)
        _run_logged(install, log_path=path(paths["logs"] / "install.log"), environment=environment)
        _preflight_sample(
            log_path=path(paths["logs"] / "sampler-preflight.log"), environment=environment
        )
        installed_candidate = _candidate_identity()
        native_origin = _capture(
            [
                str(PYTHON),
                "-c",
                "import importlib.util; print(importlib.util.find_spec('qwen_mm._native').origin)",
            ]
        )
        native_module_file = _capture(["file", "-b", native_origin])

        phase_c = support.phase_c_command(
            python=PYTHON,
            wheel=wheel,
            assets_root=ASSETS_ROOT,
            output=phase_c_scratch / "evidence",
            report=path(paths["phase_c_report"]),
            summary=path(paths["phase_c_summary"]),
        )
        _run_logged(phase_c, log_path=path(paths["logs"] / "phase-c.log"), environment=environment)
        phase_c_validate = support.phase_c_validation_command(
            python=PYTHON, assets_root=ASSETS_ROOT, report=path(paths["phase_c_report"])
        )
        _run_logged(
            phase_c_validate,
            log_path=path(paths["logs"] / "phase-c-validation.log"),
            environment=environment,
        )
        benchmark = support.benchmark_command(
            python=PYTHON,
            workload=LOCAL_ROOT / "benchmarks/workloads-v2.json",
            assets_root=ASSETS_ROOT,
            phase_c_report=path(paths["phase_c_report"]),
            phase_c_publish_report=Path(str(paths["phase_c_report"])),
            output=path(paths["benchmark_result"]),
            report=path(paths["benchmark_report"]),
        )
        _run_logged(
            benchmark,
            log_path=path(paths["logs"] / "benchmark.log"),
            environment=environment,
        )
        benchmark_validate = support.benchmark_validation_command(
            python=PYTHON,
            result=path(paths["benchmark_result"]),
            phase_c_source_report=path(paths["phase_c_report"]),
        )
        _run_logged(
            benchmark_validate,
            log_path=path(paths["logs"] / "benchmark-validation.log"),
            environment=environment,
        )

        publish = path(paths["publish"])
        for source in (
            build_log,
            path(paths["phase_c_report"]),
            path(paths["benchmark_result"]),
        ):
            destination = publish / source.name
            if destination.exists():
                raise RuntimeError(f"duplicate profile publication basename: {destination.name}")
            shutil.copy2(source, destination)

        profile = support.profile_capture_command(
            python=PYTHON,
            workload=LOCAL_ROOT / "benchmarks/workloads-v2.json",
            benchmark_result=path(paths["benchmark_result"]),
            build_command=shlex.join(build),
            build_log=build_log,
            wheel=wheel,
            phase_c_report=path(paths["phase_c_report"]),
            artifact_directory=publish,
            artifact_publish_directory=Path(str(paths["publish"])),
            source_revision=revision,
            source_digest=digest,
            output=path(paths["profile_bundle"]),
            benchmark_phase_c_source_report=path(paths["phase_c_report"]),
        )
        _run_logged(profile, log_path=path(paths["logs"] / "profile.log"), environment=environment)

        after_revision, after_digest, after_files, after_bytes = clean_committed_source_identity()
        if (after_revision, after_digest, after_files, after_bytes) != (
            revision,
            digest,
            source_files,
            source_bytes,
        ):
            raise RuntimeError("source identity changed during local ARM capture")

        installed_host_root = LOCAL_ROOT / str(paths["root"])
        if installed_host_root.exists():
            raise RuntimeError("capture both clean-commit archives before ingesting evidence")
        installed_host_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(capture_root / str(paths["root"]), installed_host_root)
        try:
            profile_validate = support.profile_validation_command(
                python=PYTHON, bundle=installed_host_root / "profile/bundle.json"
            )
            _run_logged(
                profile_validate,
                log_path=path(paths["logs"] / "profile-validation.log"),
                environment=environment,
            )
            profile_report = support.profile_report_command(
                python=PYTHON,
                bundle=installed_host_root / "profile/bundle.json",
                output=path(paths["profile_report"]),
            )
            _run_logged(
                profile_report,
                log_path=path(paths["logs"] / "profile-report.log"),
                environment=environment,
            )
        finally:
            shutil.rmtree(installed_host_root)

        phase_c_result = _load_json(path(paths["phase_c_report"]))
        benchmark_result = _load_json(path(paths["benchmark_result"]))
        profile_result = _load_json(path(paths["profile_bundle"]))
        phase_c_runtime = phase_c_result["candidate"]["runtime_identity"]
        benchmark_runtime = benchmark_result["protocol"]["candidate_identity"]["runtime_identity"]
        profile_runtime = profile_result["captures"][0]["build"]["candidate_identity"][
            "runtime_identity"
        ]
        runtime_authentication = {
            "wheel_contents": wheel_contents,
            "installed_candidate": installed_candidate,
            "phase_c_runtime_identity": phase_c_runtime,
            "benchmark_runtime_identity": benchmark_runtime,
            "profile_runtime_identity": profile_runtime,
            "native_module_file": native_module_file,
            "all_runtime_identities_equal": (
                phase_c_runtime
                == benchmark_runtime
                == profile_runtime
                == installed_candidate.get("runtime_identity")
            ),
        }
        support.validate_runtime_authentication(runtime_authentication, architecture="arm64")
        path(paths["runtime_authentication"]).write_text(
            json.dumps(runtime_authentication, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        path(paths["asset_manifest"]).write_text(
            json.dumps(assets, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        sample_version = _sample_version()
        commands = [
            build,
            install,
            phase_c,
            phase_c_validate,
            benchmark,
            benchmark_validate,
            profile,
            profile_validate,
            profile_report,
        ]
        provenance = {
            "schema_id": support.PROVENANCE_SCHEMA_ID,
            "schema_version": support.PROVENANCE_SCHEMA_VERSION,
            "execution": "local_native_macos_arm_profile_v1",
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "source": {
                "revision": revision,
                "tree_sha256": digest,
                "file_count": source_files,
                "bytes": source_bytes,
                "clean": True,
                "git_status": "",
            },
            "assets": assets,
            "host": {
                "system": platform.system(),
                "machine": platform.machine(),
                "platform": platform.platform(),
                "uname": platform.uname()._asdict(),
                "hostname": socket.gethostname(),
                "cpu_description": _capture(["sysctl", "-n", "machdep.cpu.brand_string"]),
                "logical_cpu_count": os.cpu_count(),
            },
            "image": {
                "python_version": platform.python_version(),
                "sample_version": sample_version,
            },
            "protocol": {
                "profiles": list(support.PROFILE_ALIASES),
                "cases": list(support.PROFILE_CASES),
                "thread_budgets": list(support.THREAD_BUDGETS),
                "build_label": support.PROFILE_BUILD_LABEL,
                "observed_coordinates": len(support.matrix_coordinates()),
                "observation_repetitions": support.PROFILE_REPETITIONS,
                "sampler": {
                    "name": "sample",
                    "native": True,
                    "interval_ms": 1,
                    "duration_seconds": support.PY_SPY_DURATION_SECONDS,
                },
            },
            "build": {
                "command": shlex.join(build),
                "log": support.file_identity(build_log, relative_to=capture_root),
                "wheel": support.file_identity(wheel, relative_to=capture_root),
                "native_module_file": native_module_file,
            },
            "commands": [shlex.join(command) for command in commands],
        }
        support.validate_capture_provenance(provenance, architecture="arm64")
        path(paths["provenance"]).write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        artifact = support.create_integrity_archive(
            support.collect_evidence_files(capture_root, architecture="arm64"),
            architecture="arm64",
        )
        validated = support.write_artifact_atomically(artifact, output, architecture="arm64")
        return validated
    finally:
        shutil.rmtree(capture_root, ignore_errors=True)
        shutil.rmtree(phase_c_scratch, ignore_errors=True)


def main() -> None:
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/qwen-mm-arm-d1-profile.zip")
    validated = capture_arm_archive(output.expanduser().resolve())
    provenance = validated["provenance"]
    print(f"wrote integrity-validated local ARM D1 archive: {output}")
    print("canonical semantic validation runs after the separate ingest step")
    print(
        f"captured {provenance['protocol']['observed_coordinates']} coordinates from "
        f"{provenance['source']['revision']}"
    )


if __name__ == "__main__":
    main()
