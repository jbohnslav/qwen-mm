"""Native Linux x86 Phase D1 conformance, benchmark, and profile capture on Modal."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import modal

LOCAL_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ASSETS_ROOT = LOCAL_ROOT / "reference/.cache/huggingface"
REMOTE_ROOT = Path("/workspace/qwen-mm")
REMOTE_ASSETS_ROOT = REMOTE_ROOT / "reference/.cache/huggingface"
REMOTE_PYTHON = REMOTE_ROOT / ".venv/bin/python"
BASE_IMAGE = "python@sha256:28255a3ace7eb4c48bc1b57b90af29e1bc82b4fd6c60614a8e3dce61b87ff941"
BASE_IMAGE_TAG = "python:3.11.15-slim-bookworm"
MODAL_CPU = 8.0
MODAL_MEMORY_MIB = 32_768
RUST_VERSION = "1.97.1"
UV_VERSION = "0.11.29"
PYPI_INDEX = "https://pypi.org/simple"
COMMAND_OUTPUT_TAIL_CHARS = 4_000

app = modal.App("qwen-mm-d1-native-profile")


def _support() -> Any:
    repository_root = REMOTE_ROOT if REMOTE_ROOT.is_dir() else LOCAL_ROOT
    support_directory = str(repository_root / "scripts")
    if support_directory not in sys.path:
        sys.path.insert(0, support_directory)
    import profile_capture_support

    return profile_capture_support


def _source_ignore(relative: Path) -> bool:
    support_directory = str(LOCAL_ROOT / "scripts")
    if support_directory not in sys.path:
        sys.path.insert(0, support_directory)
    from modal_benchmark_support import is_ignored_source_path

    # The digest deliberately excludes Kingdom control files and previously
    # committed profile evidence, but the shipped .git database still expects
    # tracked copies of them. Include those paths in the upload so `git status`
    # authenticates a genuinely clean checkout.
    if relative.parts[:1] == (".kd",) or relative.parts[:2] == (
        "benchmarks",
        "profile-evidence-v1",
    ):
        return False
    return is_ignored_source_path(relative)


profile_image = (
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
        f"uv pip install --system --default-index {PYPI_INDEX} py-spy==0.4.1",
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
    # Phase C authenticates committed inputs with git show, so ship repository
    # metadata separately from the fingerprinted source and offline assets.
    .add_local_dir(str(LOCAL_ROOT / ".git"), remote_path=str(REMOTE_ROOT / ".git"), copy=True)
    .add_local_dir(
        str(LOCAL_ASSETS_ROOT),
        remote_path=str(REMOTE_ASSETS_ROOT),
        copy=True,
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
        output_tail = result.stdout[-COMMAND_OUTPUT_TAIL_CHARS:]
        raise RuntimeError(
            f"command failed with exit code {result.returncode}; see {log_path}: "
            f"{shlex.join(command)}\ncommand output tail:\n{output_tail}"
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


def _preflight_py_spy(*, python: Path, log_path: Path, environment: dict[str, str]) -> None:
    """Prove native py-spy can profile a child before the 24-cell run."""

    with tempfile.TemporaryDirectory(prefix="qwen-mm-py-spy-preflight-") as directory:
        temporary = Path(directory)
        raw = temporary / "preflight.raw"
        program = (
            "import qwen_mm._native, time; "
            "deadline=time.monotonic()+1.0; value=0; "
            "exec('while time.monotonic() < deadline:\\n value += 1')"
        )
        command = [
            "py-spy",
            "record",
            "--native",
            "--rate",
            "99",
            "--format",
            "raw",
            "-o",
            str(raw),
            "--",
            str(python),
            "-c",
            program,
        ]
        result = subprocess.run(
            command,
            cwd=REMOTE_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=30,
        )
        raw_text = raw.read_text(encoding="utf-8") if raw.is_file() else ""
        rendered = "$ " + shlex.join(command) + "\n" + result.stdout
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(rendered, encoding="utf-8")
        summary = re.findall(r"\bSamples:\s*(\d+)\s+Errors:\s*(\d+)\b", result.stdout)
        try:
            raw_samples = sum(
                int(line.rsplit(" ", 1)[1]) for line in raw_text.splitlines() if line.strip()
            )
        except (IndexError, ValueError):
            raw_samples = -1
        summary_valid = (
            len(summary) == 1
            and int(summary[0][0]) > 0
            and int(summary[0][1]) == 0
            and raw_samples == int(summary[0][0])
        )
        if result.returncode != 0 or not raw_text.strip() or not summary_valid:
            raise RuntimeError(
                "native py-spy child-launch preflight failed "
                f"before the full D1 matrix (see {log_path})"
            )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return value


def _candidate_identity() -> dict[str, Any]:
    program = (
        "import json; "
        "from qwen_mm_reference.benchmark_v2 import candidate_artifact_identity; "
        "print(json.dumps(candidate_artifact_identity('qwen_mm.benchmark:create_adapter'), "
        "sort_keys=True))"
    )
    value = json.loads(_capture([str(REMOTE_PYTHON), "-c", program]))
    if not isinstance(value, dict):
        raise RuntimeError("installed candidate identity is not an object")
    return value


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


def _cgroup_limits() -> dict[str, str]:
    values: dict[str, str] = {}
    for path in (
        "/sys/fs/cgroup/cpu.max",
        "/sys/fs/cgroup/cpuset.cpus.effective",
        "/sys/fs/cgroup/memory.max",
    ):
        try:
            value = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            values[path] = value
    return values


def _memory_total_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo does not contain MemTotal")


def _assert_remote_source(
    *,
    expected_revision: str,
    expected_digest: str,
    expected_files: int,
    expected_bytes: int,
) -> dict[str, Any]:
    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    from modal_benchmark_support import committed_source_tree_digest

    revision = _capture(["git", "rev-parse", "HEAD"])
    status = _capture(["git", "status", "--porcelain", "--untracked-files=all"])
    digest, files, source_bytes = committed_source_tree_digest(REMOTE_ROOT, revision)
    expected = (expected_revision, expected_digest, expected_files, expected_bytes, "")
    actual = (revision, digest, files, source_bytes, status)
    if actual != expected:
        raise RuntimeError(f"packaged source identity mismatch: expected {expected}, got {actual}")
    return {
        "revision": revision,
        "tree_sha256": digest,
        "file_count": files,
        "bytes": source_bytes,
        "clean": True,
        "git_status": status,
    }


@app.function(image=profile_image, cpu=MODAL_CPU, memory=MODAL_MEMORY_MIB, timeout=7_200)
def run_d1_profile(
    *,
    source_revision: str,
    expected_source_digest: str,
    expected_source_files: int,
    expected_source_bytes: int,
    expected_assets: dict[str, Any],
    modal_client_version: str,
) -> bytes:
    support = _support()
    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    from modal_benchmark_support import assert_native_linux_x86

    started_at = datetime.now(UTC).isoformat()
    lscpu = json.loads(_capture(["lscpu", "--json"]))
    lscpu_fields = {
        str(item["field"]).removesuffix(":"): item.get("data")
        for item in lscpu.get("lscpu", [])
        if isinstance(item, dict) and "field" in item
    }
    cpu_description = _cpu_description(lscpu_fields)
    assert_native_linux_x86(
        system=platform.system(), machine=platform.machine(), cpu_description=cpu_description
    )
    source = _assert_remote_source(
        expected_revision=source_revision,
        expected_digest=expected_source_digest,
        expected_files=expected_source_files,
        expected_bytes=expected_source_bytes,
    )
    support.assert_directory_identity(REMOTE_ASSETS_ROOT, expected_assets)

    capture_root = Path(tempfile.mkdtemp(prefix="qwen-mm-d1-x86-capture-"))
    evidence_root = capture_root / str(support.X86_EVIDENCE_ROOT)
    evidence_root.mkdir(parents=True)
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
            "QWEN_MM_ASSETS_ROOT": str(REMOTE_ASSETS_ROOT),
            "UV_DEFAULT_INDEX": PYPI_INDEX,
        }
    )

    def path(relative: PurePosixPath) -> Path:
        return capture_root / str(relative)

    wheel_directory = path(support.WHEEL_DIRECTORY)
    wheel_directory.mkdir(parents=True)
    build = support.profile_build_command(
        python=REMOTE_PYTHON,
        wheel_directory=wheel_directory,
    )
    build_log = path(support.BUILD_LOG)
    _run_logged(build, log_path=build_log, environment=environment)
    wheels = sorted(wheel_directory.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"profile build produced {len(wheels)} wheels; expected exactly one")
    wheel = wheels[0]
    wheel_contents = support.wheel_contents_identity(wheel)

    install = support.install_wheel_command(python=REMOTE_PYTHON, wheel=wheel)
    _run_logged(
        install,
        log_path=path(support.LOG_DIRECTORY / "install.log"),
        environment=environment,
    )
    _preflight_py_spy(
        python=REMOTE_PYTHON,
        log_path=path(support.LOG_DIRECTORY / "sampler-preflight.log"),
        environment=environment,
    )
    installed_candidate = _candidate_identity()
    native_origin = _capture(
        [
            str(REMOTE_PYTHON),
            "-c",
            "import importlib.util; print(importlib.util.find_spec('qwen_mm._native').origin)",
        ]
    )
    native_module_file = _capture(["file", "-b", native_origin])

    phase_c = support.phase_c_command(
        python=REMOTE_PYTHON,
        wheel=wheel,
        assets_root=REMOTE_ASSETS_ROOT,
        output=Path(tempfile.mkdtemp(prefix="qwen-mm-phase-c-scratch-")) / "evidence",
        report=path(support.PHASE_C_REPORT),
        summary=path(support.PHASE_C_SUMMARY),
    )
    _run_logged(
        phase_c,
        log_path=path(support.LOG_DIRECTORY / "phase-c.log"),
        environment=environment,
    )
    phase_c_validate = support.phase_c_validation_command(
        python=REMOTE_PYTHON,
        assets_root=REMOTE_ASSETS_ROOT,
        report=path(support.PHASE_C_REPORT),
    )
    _run_logged(
        phase_c_validate,
        log_path=path(support.LOG_DIRECTORY / "phase-c-validation.log"),
        environment=environment,
    )

    benchmark = support.benchmark_command(
        python=REMOTE_PYTHON,
        workload=REMOTE_ROOT / "benchmarks/workloads-v2.json",
        assets_root=REMOTE_ASSETS_ROOT,
        phase_c_report=path(support.PHASE_C_REPORT),
        phase_c_publish_report=Path(str(support.PHASE_C_REPORT)),
        output=path(support.BENCHMARK_RESULT),
        report=path(support.BENCHMARK_REPORT),
    )
    _run_logged(
        benchmark,
        log_path=path(support.LOG_DIRECTORY / "benchmark.log"),
        environment=environment,
    )
    benchmark_validate = support.benchmark_validation_command(
        python=REMOTE_PYTHON,
        result=path(support.BENCHMARK_RESULT),
        phase_c_source_report=path(support.PHASE_C_REPORT),
    )
    _run_logged(
        benchmark_validate,
        log_path=path(support.LOG_DIRECTORY / "benchmark-validation.log"),
        environment=environment,
    )

    publish_directory = path(support.PROFILE_PUBLISH)
    publish_directory.mkdir(parents=True, exist_ok=True)
    for source_path in (
        build_log,
        wheel,
        path(support.PHASE_C_REPORT),
        path(support.BENCHMARK_RESULT),
    ):
        destination = publish_directory / source_path.name
        if source_path.resolve() == destination.resolve():
            continue
        if destination.exists():
            raise RuntimeError(f"duplicate profile publication basename: {destination.name}")
        shutil.copy2(source_path, destination)

    profile = support.profile_capture_command(
        python=REMOTE_PYTHON,
        workload=REMOTE_ROOT / "benchmarks/workloads-v2.json",
        benchmark_result=path(support.BENCHMARK_RESULT),
        build_command=shlex.join(build),
        build_log=build_log,
        wheel=wheel,
        phase_c_report=path(support.PHASE_C_REPORT),
        artifact_directory=publish_directory,
        artifact_publish_directory=Path(str(support.PROFILE_PUBLISH)),
        source_revision=source_revision,
        source_digest=expected_source_digest,
        output=path(support.PROFILE_BUNDLE),
        benchmark_phase_c_source_report=path(support.PHASE_C_REPORT),
    )
    _run_logged(
        profile,
        log_path=path(support.LOG_DIRECTORY / "profile.log"),
        environment=environment,
    )
    phase_c_result = _load_json(path(support.PHASE_C_REPORT))
    benchmark_result = _load_json(path(support.BENCHMARK_RESULT))
    profile_result = _load_json(path(support.PROFILE_BUNDLE))
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
    support.validate_runtime_authentication(runtime_authentication)
    path(support.RUNTIME_AUTHENTICATION).write_text(
        json.dumps(runtime_authentication, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path(support.ASSET_MANIFEST).write_text(
        json.dumps(expected_assets, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    source_after = _assert_remote_source(
        expected_revision=source_revision,
        expected_digest=expected_source_digest,
        expected_files=expected_source_files,
        expected_bytes=expected_source_bytes,
    )
    support.assert_directory_identity(REMOTE_ASSETS_ROOT, expected_assets)
    installed_host_root = REMOTE_ROOT / str(support.X86_EVIDENCE_ROOT)
    if installed_host_root.exists():
        raise RuntimeError(
            "temporary canonical validation root already exists; capture from the clean "
            "implementation commit before evidence is committed"
        )
    installed_host_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(evidence_root, installed_host_root)
    try:
        profile_validate = support.profile_validation_command(
            python=REMOTE_PYTHON,
            bundle=installed_host_root / "profile/bundle.json",
        )
        _run_logged(
            profile_validate,
            log_path=path(support.LOG_DIRECTORY / "profile-validation.log"),
            environment=environment,
        )
        profile_report = support.profile_report_command(
            python=REMOTE_PYTHON,
            bundle=installed_host_root / "profile/bundle.json",
            output=path(support.PROFILE_REPORT),
        )
        _run_logged(
            profile_report,
            log_path=path(support.LOG_DIRECTORY / "profile-report.log"),
            environment=environment,
        )
    finally:
        shutil.rmtree(installed_host_root)
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
        "execution": "modal_native_linux_x86_profile_v1",
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "source": source_after,
        "assets": expected_assets,
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "uname": _capture(["uname", "-a"]),
            "hostname": socket.gethostname(),
            "cpu_description": cpu_description,
            "logical_cpu_count": os.cpu_count(),
            "visible_cpu_affinity": sorted(os.sched_getaffinity(0)),
            "proc_meminfo_total_bytes": _memory_total_bytes(),
            "cgroup_limits": _cgroup_limits(),
            "requested_cpu_physical_cores": MODAL_CPU,
            "requested_memory_mib": MODAL_MEMORY_MIB,
            "lscpu": lscpu,
        },
        "image": {
            "base_image_tag": BASE_IMAGE_TAG,
            "base_image_reference": BASE_IMAGE,
            "rust_version": RUST_VERSION,
            "uv_version": _capture(["uv", "--version"]),
            "py_spy_version": _capture(["py-spy", "--version"]),
            "python_version": platform.python_version(),
        },
        "modal": {
            "client_version": modal_client_version,
            "environment": {
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
            },
        },
        "protocol": {
            "profiles": list(support.PROFILE_ALIASES),
            "cases": list(support.PROFILE_CASES),
            "thread_budgets": list(support.THREAD_BUDGETS),
            "build_label": support.PROFILE_BUILD_LABEL,
            "observed_coordinates": len(support.matrix_coordinates()),
            "observation_repetitions": support.PROFILE_REPETITIONS,
            "sampler": {
                "name": "py-spy",
                "version": support.PY_SPY_VERSION,
                "native": True,
                "rate_hz": support.PY_SPY_RATE_HZ,
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
    support.validate_capture_provenance(provenance)
    path(support.PROVENANCE).write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact = support.create_integrity_archive(
        support.collect_evidence_files(capture_root, architecture="x86_64"),
        architecture="x86_64",
    )
    support.validate_profile_artifact(artifact)
    if source_after != source:
        raise RuntimeError("source identity changed during profile capture")
    return artifact


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=LOCAL_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    ).stdout.strip()


def _modal_version() -> str:
    try:
        return importlib.metadata.version("modal")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


@app.local_entrypoint()
def main(output: str = "/tmp/qwen-mm-modal-d1-profile.zip") -> None:
    support = _support()
    sys.path.insert(0, str(LOCAL_ROOT / "scripts"))
    from modal_benchmark_support import committed_source_tree_digest

    status = _git_output("status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(
            "D1 Modal profile evidence requires an exact clean source; commit the ticket first"
        )
    revision = _git_output("rev-parse", "HEAD")
    digest, file_count, source_bytes = committed_source_tree_digest(LOCAL_ROOT, revision)
    assets = support.directory_identity(LOCAL_ASSETS_ROOT)
    artifact = run_d1_profile.remote(
        source_revision=revision,
        expected_source_digest=digest,
        expected_source_files=file_count,
        expected_source_bytes=source_bytes,
        expected_assets=assets,
        modal_client_version=_modal_version(),
    )
    destination = Path(output).expanduser().resolve()
    validated = support.write_artifact_atomically(artifact, destination)
    provenance = validated["provenance"]
    print(f"wrote integrity-validated Modal D1 archive: {destination}")
    print("canonical semantic validation runs after the separate ingest step")
    print(
        "native worker: "
        f"{provenance['host']['system']} {provenance['host']['machine']} / "
        f"{provenance['host']['cpu_description']}"
    )
    print(
        f"captured {provenance['protocol']['observed_coordinates']} coordinates with "
        "fresh Phase C and paired real benchmark evidence"
    )
