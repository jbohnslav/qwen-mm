"""Controlled four-variant D3 benchmark matrix on one Modal x86 worker."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

LOCAL_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ASSETS_ROOT = (LOCAL_ROOT / "reference/.cache/huggingface").resolve()
REMOTE_ROOT = Path("/workspace/qwen-mm")
REMOTE_ASSETS_ROOT = REMOTE_ROOT / "reference/.cache/huggingface"
REMOTE_PYTHON = REMOTE_ROOT / ".venv/bin/python"
BASE_IMAGE = "python@sha256:28255a3ace7eb4c48bc1b57b90af29e1bc82b4fd6c60614a8e3dce61b87ff941"
RUST_VERSION = "1.97.1"
UV_VERSION = "0.11.29"
PYPI_INDEX = "https://pypi.org/simple"
MODAL_CPU = 16.0
MODAL_MEMORY_MIB = 32_768
COOLDOWN_SECONDS = 30
IMPLEMENTATION_FILES = (
    "crates/qwen-mm-core/src/media.rs",
    "crates/qwen-mm-core/src/patchify.rs",
    "crates/qwen-mm-core/src/processor.rs",
    "crates/qwen-mm-core/src/resize.rs",
)
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
}

app = modal.App("qwen-mm-d3-controlled-matrix")


def _source_ignore(relative: Path) -> bool:
    support_directory = str(LOCAL_ROOT / "scripts")
    if support_directory not in sys.path:
        sys.path.insert(0, support_directory)
    from modal_benchmark_support import is_ignored_source_path

    return is_ignored_source_path(relative)


d3_image = (
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
        f"cd {REMOTE_ROOT} && uv sync --locked --all-packages --group dev "
        f"--default-index {PYPI_INDEX}",
    )
    .add_local_dir(str(LOCAL_ASSETS_ROOT), remote_path=str(REMOTE_ASSETS_ROOT), copy=True)
    .workdir(str(REMOTE_ROOT))
)


def _capture(command: list[str], *, environment: dict[str, str] | None = None) -> str:
    return subprocess.run(
        command,
        cwd=REMOTE_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    ).stdout.strip()


def _run_logged(
    command: list[str], *, log_path: Path, environment: dict[str, str], append: bool = False
) -> None:
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
    mode = "a" if append else "w"
    with log_path.open(mode, encoding="utf-8") as output:
        output.write(rendered)
        if not rendered.endswith("\n"):
            output.write("\n")
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {shlex.join(command)}\n{result.stdout[-4000:]}"
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cpu_description(lscpu: dict[str, Any]) -> str:
    fields = {
        str(item.get("field", "")).removesuffix(":"): str(item.get("data", ""))
        for item in lscpu.get("lscpu", [])
        if isinstance(item, dict)
    }
    cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    names = [
        line.split(":", 1)[1].strip()
        for line in cpuinfo.splitlines()
        if line.startswith("model name")
    ]
    if names and names[0].lower() != "unknown":
        return names[0]
    model_name = fields.get("Model name", "").strip()
    if model_name and model_name.lower() != "unknown":
        return model_name
    vendor = fields.get("Vendor ID", "unknown vendor")
    family = fields.get("CPU family", "unknown")
    model = fields.get("Model", "unknown")
    return f"{vendor} family {family} model {model} (provider masks model name)"


def _memory_total_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo did not contain MemTotal")


def _cgroup_limits() -> dict[str, str]:
    values: dict[str, str] = {}
    for name in ("cpu.max", "cpuset.cpus.effective", "memory.max"):
        path = Path("/sys/fs/cgroup") / name
        try:
            values[str(path)] = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
    return values


@app.function(
    image=d3_image,
    cpu=MODAL_CPU,
    memory=MODAL_MEMORY_MIB,
    timeout=3_600,
    nonpreemptible=True,
    single_use_containers=True,
)
def run_d3_matrix(
    *,
    expected_source_digest: str,
    expected_source_files: int,
    expected_source_bytes: int,
    expected_diff_sha256: dict[str, str],
    expected_assets: dict[str, Any],
    modal_client_version: str,
) -> bytes:
    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))
    from modal_benchmark_support import assert_native_linux_x86, source_tree_digest
    from modal_d3_support import CASES, PROFILES, VARIANTS, create_archive, validate_matrix
    from profile_capture_support import assert_directory_identity

    started_at = datetime.now(UTC)
    observed_source = source_tree_digest(REMOTE_ROOT)
    expected_source = (expected_source_digest, expected_source_files, expected_source_bytes)
    if observed_source != expected_source:
        raise RuntimeError(
            f"uploaded source identity mismatch: expected {expected_source}, got {observed_source}"
        )
    assert_directory_identity(REMOTE_ASSETS_ROOT, expected_assets)
    patch_paths = {
        variant: REMOTE_ROOT / f"benchmarks/d3-evidence-v1/{variant}.patch"
        for variant in VARIANTS
        if variant != "baseline"
    }
    for variant, path in patch_paths.items():
        if _sha256(path) != expected_diff_sha256[variant]:
            raise RuntimeError(f"uploaded {variant} patch identity mismatch")

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
            **THREAD_ENVIRONMENT,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            "QWEN_MM_ASSETS_ROOT": str(REMOTE_ASSETS_ROOT),
            "UV_DEFAULT_INDEX": PYPI_INDEX,
            "PYO3_PYTHON": str(REMOTE_PYTHON),
        }
    )

    # Establish a synthetic clean Git baseline from the exact uploaded
    # candidate. This gives measure.py a real revision plus reconstructable
    # per-variant diffs without uploading mutable local worktree metadata.
    _capture(["git", "apply", "--check", "-R", str(patch_paths["candidate"])])
    _capture(["git", "apply", "-R", str(patch_paths["candidate"])])
    _capture(["git", "init", "--quiet"])
    _capture(["git", "add", "-A"])
    commit_environment = dict(environment)
    commit_environment.update(
        {
            "GIT_AUTHOR_DATE": "2026-08-03T00:00:00+00:00",
            "GIT_COMMITTER_DATE": "2026-08-03T00:00:00+00:00",
        }
    )
    _capture(
        [
            "git",
            "-c",
            "user.name=qwen-mm D3 evidence",
            "-c",
            "user.email=d3@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "synthetic authenticated D3 baseline",
        ],
        environment=commit_environment,
    )
    baseline_revision = _capture(["git", "rev-parse", "HEAD"])
    _capture(["git", "apply", str(patch_paths["candidate"])])
    current_variant = "candidate"

    def set_variant(target: str) -> None:
        nonlocal current_variant
        if current_variant == target:
            return
        if current_variant != "baseline":
            _capture(["git", "apply", "-R", str(patch_paths[current_variant])])
        if target != "baseline":
            _capture(["git", "apply", str(patch_paths[target])])
        current_variant = target
        diff = subprocess.run(
            ["git", "diff", "--binary", "--", *IMPLEMENTATION_FILES],
            cwd=REMOTE_ROOT,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        if hashlib.sha256(diff).hexdigest() != expected_diff_sha256[target]:
            raise RuntimeError(f"{target}: live source diff does not match its preserved patch")

    with tempfile.TemporaryDirectory(prefix="qwen-mm-d3-modal-") as temporary:
        artifact_root = Path(temporary)
        logs = artifact_root / "logs"
        binaries = artifact_root / "binaries"
        coordinates = artifact_root / "coordinates"
        logs.mkdir()
        binaries.mkdir()
        coordinates.mkdir()
        native_origin: Path | None = None
        built_native: dict[str, dict[str, str]] = {}
        build_command = [
            str(REMOTE_ROOT / ".venv/bin/maturin"),
            "develop",
            "--release",
            "--locked",
            "--skip-install",
        ]
        for variant in ("candidate", "baseline", "copy-only", "no-lut"):
            set_variant(variant)
            _run_logged(
                build_command,
                log_path=logs / f"build-{variant}.log",
                environment=environment,
            )
            origin = Path(
                _capture(
                    [
                        str(REMOTE_PYTHON),
                        "-c",
                        "import pathlib,qwen_mm._native as n; print(pathlib.Path(n.__file__).resolve())",
                    ],
                    environment=environment,
                )
            )
            if native_origin is None:
                native_origin = origin
            elif native_origin != origin:
                raise RuntimeError("native extension origin changed across builds")
            retained = binaries / f"{variant}.so"
            shutil.copy2(origin, retained)
            built_native[variant] = {
                "sha256": _sha256(retained),
                "file": _capture(["file", "-b", str(retained)], environment=environment),
            }
        if native_origin is None or len({item["sha256"] for item in built_native.values()}) != 4:
            raise RuntimeError("variant builds did not produce four distinct native modules")

        set_variant("candidate")
        time.sleep(COOLDOWN_SECONDS)
        affinity = sorted(os.sched_getaffinity(0))
        if not affinity:
            raise RuntimeError("Modal worker exposed no CPU affinity")
        pinned_cpu = affinity[0]
        latin_orders = (
            ("baseline", "copy-only", "no-lut", "candidate"),
            ("copy-only", "no-lut", "candidate", "baseline"),
            ("no-lut", "candidate", "baseline", "copy-only"),
            ("candidate", "baseline", "copy-only", "no-lut"),
        )
        records: list[dict[str, Any]] = []
        sequence = 0
        matrix_run_nonce = secrets.token_hex(16)
        measure_log = logs / "measure.log"
        matrix_coordinates = [(profile, case) for profile in PROFILES for case in CASES]
        for coordinate_index, (profile, case) in enumerate(matrix_coordinates):
            order = latin_orders[coordinate_index % len(latin_orders)]
            for order_index, variant in enumerate(order):
                set_variant(variant)
                shutil.copy2(binaries / f"{variant}.so", native_origin)
                if _sha256(native_origin) != built_native[variant]["sha256"]:
                    raise RuntimeError(f"{variant}: installed native module copy mismatch")
                output = coordinates / variant / f"{profile}-{case}.json"
                output.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    "taskset",
                    "-c",
                    str(pinned_cpu),
                    str(REMOTE_PYTHON),
                    "scripts/modal_d3_worker.py",
                    variant,
                    profile,
                    case,
                    str(output),
                    "--pinned-cpu",
                    str(pinned_cpu),
                    "--sequence",
                    str(sequence),
                    "--coordinate-index",
                    str(coordinate_index),
                    "--order-index",
                    str(order_index),
                    "--variant-order",
                    *order,
                    "--matrix-run-nonce",
                    matrix_run_nonce,
                    "--coordinate-run-nonce",
                    hashlib.sha256(
                        f"{matrix_run_nonce}:{sequence}:{variant}:{profile}:{case}".encode()
                    ).hexdigest(),
                ]
                imported_hash = _capture(
                    [
                        str(REMOTE_PYTHON),
                        "-c",
                        "import hashlib,pathlib,qwen_mm._native as n; "
                        "print(hashlib.sha256(pathlib.Path(n.__file__).read_bytes()).hexdigest())",
                    ],
                    environment=environment,
                )
                if imported_hash != built_native[variant]["sha256"]:
                    raise RuntimeError(f"{variant}: imported extension is not the retained build")
                _run_logged(
                    command,
                    log_path=measure_log,
                    environment=environment,
                    append=sequence > 0,
                )
                record = json.loads(output.read_text(encoding="utf-8"))
                records.append(record)
                sequence += 1
        set_variant("candidate")
        shutil.copy2(binaries / "candidate.so", native_origin)
        summary = validate_matrix(
            records,
            expected_diff_sha256=expected_diff_sha256,
            pinned_cpu=pinned_cpu,
        )

        lscpu = json.loads(_capture(["lscpu", "--json"], environment=environment))
        cgroup_limits = _cgroup_limits()
        resource_attestation_mode = (
            "cgroup_v2_limits" if cgroup_limits else "gvisor_observed_capacity"
        )
        cpu_description = _cpu_description(lscpu)
        assert_native_linux_x86(
            system=platform.system(),
            machine=platform.machine(),
            cpu_description=cpu_description,
        )
        completed_at = datetime.now(UTC)
        elapsed_seconds = (completed_at - started_at).total_seconds()
        elapsed_worker_usd = (
            elapsed_seconds * 3.0 * (MODAL_CPU * 0.0000131 + (MODAL_MEMORY_MIB / 1024) * 0.00000222)
        )
        assert_directory_identity(REMOTE_ASSETS_ROOT, expected_assets)
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
            "schema_id": "qwen-mm-d3-modal-evidence-v1",
            "schema_version": 1,
            "execution": "controlled_same_worker_relative_variant_matrix",
            "performance_claim_scope": "D3 same-worker relative variant selection",
            "d4_certification": False,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "cost_estimate": {
                "elapsed_worker_usd": elapsed_worker_usd,
                "basis_seconds": elapsed_seconds,
                "scope": "elapsed function worker only; excludes image construction and provider billing adjustments",
            },
            "uploaded_source": {
                "sha256": expected_source_digest,
                "file_count": expected_source_files,
                "bytes": expected_source_bytes,
            },
            "assets": expected_assets,
            "synthetic_baseline_revision": baseline_revision,
            "matrix_run_nonce": matrix_run_nonce,
            "expected_diff_sha256": expected_diff_sha256,
            "variant_native_modules": built_native,
            "host": {
                "system": platform.system(),
                "machine": platform.machine(),
                "platform": platform.platform(),
                "uname": _capture(["uname", "-a"], environment=environment),
                "hostname": socket.gethostname(),
                "cpu_description": cpu_description,
                "logical_cpu_count": os.cpu_count(),
                "visible_cpu_affinity": affinity,
                "pinned_cpu": pinned_cpu,
                "proc_meminfo_total_bytes": _memory_total_bytes(),
                "cgroup_limits": cgroup_limits,
                "resource_attestation": {
                    "mode": resource_attestation_mode,
                    "requested_resources_bound_by": "source-authenticated Modal function decorator",
                },
                "requested_physical_cores": MODAL_CPU,
                "requested_memory_mib": MODAL_MEMORY_MIB,
                "nonpreemptible": True,
                "single_use_container": True,
                "lscpu": lscpu,
            },
            "image": {
                "base_image": BASE_IMAGE,
                "rust_version": _capture(["rustc", "--version"], environment=environment),
                "cargo_version": _capture(["cargo", "--version"], environment=environment),
                "uv_version": _capture(["uv", "--version"], environment=environment),
                "python_version": platform.python_version(),
            },
            "modal": {"client_version": modal_client_version, "environment": modal_environment},
            "protocol": {
                "profiles": list(PROFILES),
                "cases": list(CASES),
                "variants": list(VARIANTS),
                "coordinates": len(records),
                "warmups": 2,
                "samples": 7,
                "thread_budget": 1,
                "cooldown_seconds": COOLDOWN_SECONDS,
                "latin_orders": [list(order) for order in latin_orders],
                "fresh_subprocess_per_coordinate": True,
                "taskset_cpu": pinned_cpu,
            },
            "pricing_snapshot": {
                "cpu_usd_per_physical_core_second": 0.0000131,
                "memory_usd_per_gib_second": 0.00000222,
                "nonpreemptible_multiplier": 3.0,
            },
        }
        files: dict[str, bytes] = {
            "provenance.json": (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode(),
            "summary.json": (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
        }
        for variant, path in patch_paths.items():
            files[f"patches/{variant}.patch"] = path.read_bytes()
        for record in records:
            name = (
                f"coordinates/{record['variant']}/"
                f"{record['profile_alias']}-{record['case_id']}.json"
            )
            files[name] = (json.dumps(record, indent=2) + "\n").encode()
        for variant in VARIANTS:
            files[f"binaries/{variant}.so"] = (binaries / f"{variant}.so").read_bytes()
        for log in sorted(logs.iterdir()):
            files[f"logs/{log.name}"] = log.read_bytes()
        return create_archive(files)


def _local_modal_version() -> str:
    try:
        return importlib.metadata.version("modal")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


@app.local_entrypoint()
def main(output: str = "/tmp/qwen-mm-d3-modal.zip") -> None:
    sys.path.insert(0, str(LOCAL_ROOT / "scripts"))
    from modal_benchmark_support import source_tree_digest
    from modal_d3_support import EMPTY_SHA256, sha256_path, write_archive_atomically
    from profile_capture_support import directory_identity

    source_digest, source_files, source_bytes = source_tree_digest(LOCAL_ROOT)
    expected_diff_sha256 = {
        "baseline": EMPTY_SHA256,
        "copy-only": sha256_path(LOCAL_ROOT / "benchmarks/d3-evidence-v1/copy-only.patch"),
        "no-lut": sha256_path(LOCAL_ROOT / "benchmarks/d3-evidence-v1/no-lut.patch"),
        "candidate": sha256_path(LOCAL_ROOT / "benchmarks/d3-evidence-v1/candidate.patch"),
    }
    assets = directory_identity(LOCAL_ASSETS_ROOT)
    artifact = run_d3_matrix.remote(
        expected_source_digest=source_digest,
        expected_source_files=source_files,
        expected_source_bytes=source_bytes,
        expected_diff_sha256=expected_diff_sha256,
        expected_assets=assets,
        modal_client_version=_local_modal_version(),
    )
    destination = Path(output).expanduser().resolve()
    validated = write_archive_atomically(
        artifact,
        destination,
        expected_source=(source_digest, source_files, source_bytes),
        expected_diff_sha256=expected_diff_sha256,
        expected_assets=assets,
    )
    provenance = validated["provenance"]
    summary = validated["summary"]
    print(f"wrote integrity-validated D3 Modal artifact: {destination}")
    print(
        f"worker: {provenance['host']['cpu_description']} / "
        f"pinned CPU {provenance['host']['pinned_cpu']}"
    )
    print(
        f"validated {summary['coordinate_count']} coordinates; harness {summary['harness_sha256']}"
    )
