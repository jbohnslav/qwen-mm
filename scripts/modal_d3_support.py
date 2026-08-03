"""Pure-stdlib validation and packaging for the Modal D3 matrix."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import statistics
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_ID = "qwen-mm-d3-modal-evidence-v1"
SCHEMA_VERSION = 1
ARTIFACT_SCHEMA_ID = "qwen-mm-d3-modal-artifact-v1"
ARTIFACT_SCHEMA_VERSION = 1
VARIANTS = ("baseline", "copy-only", "no-lut", "candidate")
PROFILES = ("qwen3-vl-8b", "qwen3.5-9b")
CASES = ("image24", "ragged24", "rgb24")
THREAD_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
)
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
BASE_IMAGE = "python@sha256:28255a3ace7eb4c48bc1b57b90af29e1bc82b4fd6c60614a8e3dce61b87ff941"
RUST_VERSION = "1.97.1"
UV_VERSION = "0.11.29"
MODAL_CPU = 16.0
MODAL_MEMORY_MIB = 32_768
COOLDOWN_SECONDS = 30
CPU_USD_PER_PHYSICAL_CORE_SECOND = 0.0000131
MEMORY_USD_PER_GIB_SECOND = 0.00000222
NONPREEMPTIBLE_MULTIPLIER = 3.0
CGROUP_ATTESTATION_MODE = "cgroup_v2_limits"
GVISOR_ATTESTATION_MODE = "gvisor_observed_capacity"
RESOURCE_BINDING = "source-authenticated Modal function decorator"
MAX_ARCHIVE_COMPRESSED_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 64
MAX_ARCHIVE_MEMBER_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
LATIN_ORDERS = (
    ("baseline", "copy-only", "no-lut", "candidate"),
    ("copy-only", "no-lut", "candidate", "baseline"),
    ("no-lut", "candidate", "baseline", "copy-only"),
    ("candidate", "baseline", "copy-only", "no-lut"),
)


class ModalD3EvidenceError(ValueError):
    """Raised when D3 evidence is incomplete or unauthenticated."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return sha256_bytes(_canonical(value))


def _safe_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or path.as_posix() != name:
        raise ModalD3EvidenceError(f"unsafe artifact path: {name!r}")
    return path


def _coordinate_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return str(record.get("profile_alias")), str(record.get("case_id"))


def _is_nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _validate_observation(record: Mapping[str, Any], variant: str) -> None:
    peak_rss_bytes = record.get("peak_rss_bytes")
    if not _is_nonnegative_integer(peak_rss_bytes) or peak_rss_bytes == 0:
        raise ModalD3EvidenceError(f"{variant}: peak RSS is not a positive byte count")
    observation = record.get("observation")
    if not isinstance(observation, Mapping):
        raise ModalD3EvidenceError(f"{variant}: observation report is missing")
    duration_ms = observation.get("duration_ms")
    if not _is_positive_finite_number(duration_ms):
        raise ModalD3EvidenceError(f"{variant}: observation duration is invalid")
    allocations = observation.get("allocations")
    allocation_names = (
        "allocation_count",
        "allocated_bytes",
        "copy_count",
        "copied_bytes",
        "transient_live_bytes",
        "peak_transient_live_bytes",
        "retained_final_output_bytes",
    )
    if not isinstance(allocations, Mapping) or any(
        not _is_nonnegative_integer(allocations.get(name)) for name in allocation_names
    ):
        raise ModalD3EvidenceError(f"{variant}: allocation counters are invalid")
    if allocations["transient_live_bytes"] != 0:
        raise ModalD3EvidenceError(f"{variant}: successful observation retained transient bytes")
    if (
        allocations["peak_transient_live_bytes"] > allocations["allocated_bytes"]
        or allocations["retained_final_output_bytes"] > allocations["allocated_bytes"]
    ):
        raise ModalD3EvidenceError(f"{variant}: allocation lifetime counters are inconsistent")

    named_bytes: dict[str, Mapping[str, int]] = {}
    for field in ("buffer_bytes", "copy_bytes"):
        values = observation.get(field)
        if (
            not isinstance(values, Mapping)
            or any(not isinstance(name, str) or not name for name in values)
            or any(not _is_nonnegative_integer(value) for value in values.values())
        ):
            raise ModalD3EvidenceError(f"{variant}: named {field} counters are invalid")
        named_bytes[field] = values
    if sum(named_bytes["buffer_bytes"].values()) != allocations["allocated_bytes"]:
        raise ModalD3EvidenceError(f"{variant}: allocated bytes do not reconcile with buffers")
    if sum(named_bytes["copy_bytes"].values()) != allocations["copied_bytes"]:
        raise ModalD3EvidenceError(f"{variant}: copied bytes do not reconcile with copies")
    if allocations["allocation_count"] < len(named_bytes["buffer_bytes"]):
        raise ModalD3EvidenceError(f"{variant}: allocation count is smaller than named buffers")
    if allocations["copy_count"] < len(named_bytes["copy_bytes"]):
        raise ModalD3EvidenceError(f"{variant}: copy count is smaller than named copies")

    stage_ms = observation.get("stage_exclusive_ms")
    if not isinstance(stage_ms, Mapping) or any(
        not isinstance(name, str)
        or not name
        or not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
        for name, value in stage_ms.items()
    ):
        raise ModalD3EvidenceError(f"{variant}: stage timing counters are invalid")


def _lscpu_count(value: Any) -> int | None:
    if not isinstance(value, Mapping) or not isinstance(value.get("lscpu"), list):
        return None
    for item in value["lscpu"]:
        if isinstance(item, Mapping) and item.get("field") == "CPU(s):":
            try:
                count = int(item.get("data"))
            except (TypeError, ValueError):
                return None
            return count if count > 0 else None
    return None


def validate_matrix(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_diff_sha256: Mapping[str, str],
    pinned_cpu: int,
) -> dict[str, Any]:
    expected_coordinates = {
        (variant, profile, case) for variant in VARIANTS for profile in PROFILES for case in CASES
    }
    observed_coordinates = {
        (str(record.get("variant")), *_coordinate_key(record)) for record in records
    }
    if len(records) != len(expected_coordinates) or observed_coordinates != expected_coordinates:
        raise ModalD3EvidenceError("D3 matrix is incomplete or contains duplicate coordinates")
    if set(expected_diff_sha256) != set(VARIANTS):
        raise ModalD3EvidenceError("expected diff map does not cover exactly four variants")
    if expected_diff_sha256["baseline"] != EMPTY_SHA256:
        raise ModalD3EvidenceError("baseline diff must be the SHA-256 of an empty diff")

    variant_identity: dict[str, dict[str, str]] = {}
    coordinate_equality: list[dict[str, Any]] = []
    harnesses: set[str] = set()
    native_hashes: set[str] = set()
    wrapper_pids: set[int] = set()
    measurement_pids: set[int] = set()
    matrix_run_nonces: set[str] = set()
    coordinate_run_nonces: set[str] = set()
    for variant in VARIANTS:
        subset = [record for record in records if record["variant"] == variant]
        diff_hashes = {record["source"]["implementation_diff_sha256"] for record in subset}
        implementation_hashes = {record["source"]["implementation_digest"] for record in subset}
        binaries = {record["native_module"]["sha256"] for record in subset}
        if diff_hashes != {expected_diff_sha256[variant]}:
            raise ModalD3EvidenceError(f"{variant}: implementation diff identity mismatch")
        if len(implementation_hashes) != 1 or len(binaries) != 1:
            raise ModalD3EvidenceError(f"{variant}: implementation or native identity changed")
        native_hash = next(iter(binaries))
        native_hashes.add(native_hash)
        variant_identity[variant] = {
            "implementation_digest": next(iter(implementation_hashes)),
            "implementation_diff_sha256": next(iter(diff_hashes)),
            "native_module_sha256": native_hash,
        }
        for record in subset:
            if record.get("thread_budget") != 1:
                raise ModalD3EvidenceError(f"{variant}: coordinate did not use thread budget 1")
            if record.get("sample_count") != 7:
                raise ModalD3EvidenceError(f"{variant}: coordinate did not retain seven samples")
            thread_environment = record.get("thread_environment")
            if not isinstance(thread_environment, Mapping) or any(
                thread_environment.get(name) != "1" for name in THREAD_NAMES
            ):
                raise ModalD3EvidenceError(f"{variant}: thread environment is not locked to one")
            protocol = record.get("modal_protocol")
            if not isinstance(protocol, Mapping) or protocol.get("pinned_cpu") != pinned_cpu:
                raise ModalD3EvidenceError(f"{variant}: taskset CPU binding is missing")
            if protocol.get("fresh_subprocess") is not True:
                raise ModalD3EvidenceError(
                    f"{variant}: fresh coordinate subprocess is not attested"
                )
            if protocol.get("wrapper_sched_getaffinity") != [pinned_cpu] or protocol.get(
                "measurement_sched_getaffinity"
            ) != [pinned_cpu]:
                raise ModalD3EvidenceError(f"{variant}: actual process affinity is not pinned")
            wrapper_pid = protocol.get("wrapper_pid")
            measurement_pid = protocol.get("measurement_pid")
            if not isinstance(wrapper_pid, int) or not isinstance(measurement_pid, int):
                raise ModalD3EvidenceError(f"{variant}: coordinate PIDs are missing")
            if wrapper_pid in wrapper_pids or measurement_pid in measurement_pids:
                raise ModalD3EvidenceError("coordinate processes did not have unique PIDs")
            wrapper_pids.add(wrapper_pid)
            measurement_pids.add(measurement_pid)
            matrix_run_nonce = protocol.get("matrix_run_nonce")
            coordinate_run_nonce = protocol.get("coordinate_run_nonce")
            if (
                not isinstance(matrix_run_nonce, str)
                or len(matrix_run_nonce) != 32
                or any(character not in "0123456789abcdef" for character in matrix_run_nonce)
            ):
                raise ModalD3EvidenceError("matrix run nonce is invalid")
            expected_coordinate_nonce = hashlib.sha256(
                (
                    f"{matrix_run_nonce}:{protocol.get('sequence')}:{variant}:"
                    f"{record.get('profile_alias')}:{record.get('case_id')}"
                ).encode()
            ).hexdigest()
            if (
                coordinate_run_nonce != expected_coordinate_nonce
                or coordinate_run_nonce in coordinate_run_nonces
            ):
                raise ModalD3EvidenceError("coordinate run nonce is invalid or duplicated")
            matrix_run_nonces.add(matrix_run_nonce)
            coordinate_run_nonces.add(coordinate_run_nonce)
            samples = record.get("samples")
            if not isinstance(samples, list) or len(samples) != 7:
                raise ModalD3EvidenceError(f"{variant}: samples are missing")
            walls = [sample.get("wall_ms") for sample in samples if isinstance(sample, Mapping)]
            cpus = [sample.get("cpu_ms") for sample in samples if isinstance(sample, Mapping)]
            if (
                len(walls) != 7
                or len(cpus) != 7
                or any(not _is_positive_finite_number(value) for value in (*walls, *cpus))
            ):
                raise ModalD3EvidenceError(f"{variant}: samples are not finite positive timings")
            wall_p50 = record.get("wall_ms_p50")
            cpu_p50 = record.get("cpu_ms_p50")
            if not math.isclose(
                statistics.median(walls),
                wall_p50 if _is_positive_finite_number(wall_p50) else -1,
                rel_tol=0.0,
                abs_tol=1e-9,
            ) or not math.isclose(
                statistics.median(cpus),
                cpu_p50 if _is_positive_finite_number(cpu_p50) else -1,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ModalD3EvidenceError(f"{variant}: retained medians do not match samples")
            _validate_observation(record, variant)
            if record["observation"].get("dropped_events") != 0:
                raise ModalD3EvidenceError(f"{variant}: observed events were dropped")
            signatures = (
                record.get("official_output_signature"),
                record.get("observed_output_signature"),
                record.get("authenticated_output_signature"),
            )
            if signatures[0] != signatures[1] or signatures[0] != signatures[2]:
                raise ModalD3EvidenceError(f"{variant}: timed/observed/auth arrays differ")
            sidecar = record.get("official_metadata_sidecar")
            if not isinstance(sidecar, Mapping) or sidecar.get("sha256") != _digest(
                sidecar.get("value")
            ):
                raise ModalD3EvidenceError(f"{variant}: metadata sidecar signature mismatch")
            harnesses.add(str(record["harness"]["aggregate_sha256"]))
    if len(native_hashes) != len(VARIANTS):
        raise ModalD3EvidenceError("four variants did not produce four distinct native modules")
    if len(harnesses) != 1:
        raise ModalD3EvidenceError("measurement harness changed across variants")
    if len(matrix_run_nonces) != 1:
        raise ModalD3EvidenceError("coordinates do not belong to one matrix run")

    expected_sequence = []
    matrix_coordinates = [(profile, case) for profile in PROFILES for case in CASES]
    for coordinate_index, (profile, case) in enumerate(matrix_coordinates):
        order = LATIN_ORDERS[coordinate_index % len(LATIN_ORDERS)]
        for order_index, variant in enumerate(order):
            expected_sequence.append(
                (variant, profile, case, coordinate_index, order_index, list(order))
            )
    sequenced = sorted(records, key=lambda record: record["modal_protocol"]["sequence"])
    if [record["modal_protocol"]["sequence"] for record in sequenced] != list(range(24)):
        raise ModalD3EvidenceError("coordinate sequence is incomplete or duplicated")
    for record, expected in zip(sequenced, expected_sequence, strict=True):
        protocol = record["modal_protocol"]
        actual = (
            record["variant"],
            record["profile_alias"],
            record["case_id"],
            protocol.get("coordinate_index"),
            protocol.get("order_index"),
            protocol.get("variant_order"),
        )
        if actual != expected:
            raise ModalD3EvidenceError("coordinate execution did not follow the Latin-square plan")

    for profile in PROFILES:
        for case in CASES:
            subset = [record for record in records if _coordinate_key(record) == (profile, case)]
            output_hashes = {record["official_output_signature"]["sha256"] for record in subset}
            sidecar_hashes = {record["official_metadata_sidecar"]["sha256"] for record in subset}
            input_hashes = {record["protocol"]["input_fingerprint"] for record in subset}
            logical_hashes = {record["protocol"]["logical_input_fingerprint"] for record in subset}
            asset_hashes = {record["assets"]["profile_fingerprint"] for record in subset}
            if any(
                len(values) != 1
                for values in (
                    output_hashes,
                    sidecar_hashes,
                    input_hashes,
                    logical_hashes,
                    asset_hashes,
                )
            ):
                raise ModalD3EvidenceError(
                    f"{profile}/{case}: input, asset, array, or sidecar identity differs"
                )
            coordinate_equality.append(
                {
                    "profile_alias": profile,
                    "case_id": case,
                    "official_output_sha256": next(iter(output_hashes)),
                    "metadata_sidecar_sha256": next(iter(sidecar_hashes)),
                    "input_fingerprint": next(iter(input_hashes)),
                    "logical_input_fingerprint": next(iter(logical_hashes)),
                    "profile_fingerprint": next(iter(asset_hashes)),
                }
            )
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "coordinate_count": len(records),
        "thread_budget": 1,
        "pinned_cpu": pinned_cpu,
        "matrix_run_nonce": next(iter(matrix_run_nonces)),
        "harness_sha256": next(iter(harnesses)),
        "variant_identity": variant_identity,
        "coordinate_equality": coordinate_equality,
    }


def create_archive(files: Mapping[str, bytes]) -> bytes:
    required = {"provenance.json", "summary.json"}
    required.update(f"patches/{variant}.patch" for variant in VARIANTS if variant != "baseline")
    required.update(
        f"coordinates/{variant}/{profile}-{case}.json"
        for variant in VARIANTS
        for profile in PROFILES
        for case in CASES
    )
    required.update(f"binaries/{variant}.so" for variant in VARIANTS)
    required.update(f"logs/build-{variant}.log" for variant in VARIANTS)
    required.add("logs/measure.log")
    missing = sorted(required - files.keys())
    if missing:
        raise ModalD3EvidenceError(f"artifact payload is incomplete: {missing}")
    manifest_files: dict[str, dict[str, Any]] = {}
    for name, value in sorted(files.items()):
        _safe_name(name)
        if not isinstance(value, bytes):
            raise TypeError(f"artifact member must be bytes: {name}")
        manifest_files[name] = {"bytes": len(value), "sha256": sha256_bytes(value)}
    manifest = {
        "schema_id": ARTIFACT_SCHEMA_ID,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "files": manifest_files,
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, value in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, value)
        info = zipfile.ZipInfo("artifact-manifest.json", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output.getvalue()


def validate_archive(value: bytes) -> dict[str, Any]:
    if len(value) > MAX_ARCHIVE_COMPRESSED_BYTES:
        raise ModalD3EvidenceError("artifact exceeds the compressed-size limit")
    try:
        with zipfile.ZipFile(io.BytesIO(value)) as archive:
            names = archive.namelist()
            if len(names) > MAX_ARCHIVE_MEMBERS:
                raise ModalD3EvidenceError("artifact contains too many members")
            if len(names) != len(set(names)):
                raise ModalD3EvidenceError("artifact contains duplicate paths")
            total_uncompressed = 0
            for info in archive.infolist():
                file_type = (info.external_attr >> 16) & 0o170000
                if file_type not in {0, 0o100000}:
                    raise ModalD3EvidenceError(
                        f"artifact member is not a regular file: {info.filename}"
                    )
                if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                    raise ModalD3EvidenceError(
                        f"artifact member exceeds the size limit: {info.filename}"
                    )
                total_uncompressed += info.file_size
                if total_uncompressed > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                    raise ModalD3EvidenceError("artifact exceeds the uncompressed-size limit")
            files = {name: archive.read(name) for name in names}
    except (OSError, zipfile.BadZipFile) as error:
        raise ModalD3EvidenceError("artifact is not a readable ZIP") from error
    for name in files:
        _safe_name(name)
    try:
        manifest = json.loads(files.pop("artifact-manifest.json"))
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ModalD3EvidenceError("artifact manifest is missing or invalid") from error
    if (
        manifest.get("schema_id") != ARTIFACT_SCHEMA_ID
        or manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or set(manifest.get("files", {})) != set(files)
    ):
        raise ModalD3EvidenceError("artifact manifest identity or inventory mismatch")
    for name, data in files.items():
        identity = manifest["files"].get(name)
        if identity != {"bytes": len(data), "sha256": sha256_bytes(data)}:
            raise ModalD3EvidenceError(f"artifact member identity mismatch: {name}")
    try:
        provenance = json.loads(files["provenance.json"])
        retained_summary = json.loads(files["summary.json"])
        records = [
            json.loads(files[f"coordinates/{variant}/{profile}-{case}.json"])
            for variant in VARIANTS
            for profile in PROFILES
            for case in CASES
        ]
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ModalD3EvidenceError("artifact JSON payload is missing or invalid") from error
    if provenance.get("schema_id") != SCHEMA_ID or provenance.get("schema_version") != 1:
        raise ModalD3EvidenceError("provenance schema mismatch")
    if (
        provenance.get("execution") != "controlled_same_worker_relative_variant_matrix"
        or provenance.get("performance_claim_scope") != "D3 same-worker relative variant selection"
        or provenance.get("d4_certification") is not False
    ):
        raise ModalD3EvidenceError("D3 execution or claim scope provenance is invalid")
    if provenance.get("host", {}).get("system") != "Linux" or provenance.get("host", {}).get(
        "machine"
    ) not in {"x86_64", "amd64"}:
        raise ModalD3EvidenceError("artifact was not captured on native Linux x86_64")
    host = provenance["host"]
    visible_affinity = host.get("visible_cpu_affinity")
    logical_cpu_count = host.get("logical_cpu_count")
    lscpu_count = _lscpu_count(host.get("lscpu"))
    memory_total = host.get("proc_meminfo_total_bytes")
    resource_attestation = host.get("resource_attestation")
    if (
        host.get("requested_physical_cores") != MODAL_CPU
        or host.get("requested_memory_mib") != MODAL_MEMORY_MIB
        or host.get("nonpreemptible") is not True
        or host.get("single_use_container") is not True
        or not isinstance(visible_affinity, list)
        or len(visible_affinity) < MODAL_CPU
        or any(
            not isinstance(cpu, int) or isinstance(cpu, bool) or cpu < 0 for cpu in visible_affinity
        )
        or len(visible_affinity) != len(set(visible_affinity))
        or not _is_nonnegative_integer(host.get("pinned_cpu"))
        or host.get("pinned_cpu") not in visible_affinity
        or not _is_nonnegative_integer(logical_cpu_count)
        or logical_cpu_count < MODAL_CPU
        or lscpu_count is None
        or lscpu_count < MODAL_CPU
        or not _is_nonnegative_integer(memory_total)
        or memory_total < MODAL_MEMORY_MIB * 1024 * 1024
        or not isinstance(host.get("cgroup_limits"), Mapping)
        or not isinstance(resource_attestation, Mapping)
        or resource_attestation.get("requested_resources_bound_by") != RESOURCE_BINDING
    ):
        raise ModalD3EvidenceError("Modal resource or affinity provenance is incomplete")
    cgroup_limits = host["cgroup_limits"]
    attestation_mode = resource_attestation.get("mode")
    if attestation_mode == CGROUP_ATTESTATION_MODE:
        cpu_max = cgroup_limits.get("/sys/fs/cgroup/cpu.max")
        cpuset = cgroup_limits.get("/sys/fs/cgroup/cpuset.cpus.effective")
        memory_max = cgroup_limits.get("/sys/fs/cgroup/memory.max")
        if (
            not isinstance(cpu_max, str)
            or not isinstance(cpuset, str)
            or not isinstance(memory_max, str)
        ):
            raise ModalD3EvidenceError("required cgroup v2 CPU, cpuset, or memory limit is missing")
        try:
            quota, period = cpu_max.split()
            if int(period) <= 0 or (quota != "max" and int(quota) / int(period) < MODAL_CPU):
                raise ModalD3EvidenceError(
                    "cgroup CPU quota is smaller than the requested reservation"
                )
            if memory_max != "max" and int(memory_max) < MODAL_MEMORY_MIB * 1024 * 1024:
                raise ModalD3EvidenceError(
                    "cgroup memory limit is smaller than the requested reservation"
                )
        except (TypeError, ValueError) as error:
            raise ModalD3EvidenceError("cgroup CPU or memory limit is invalid") from error
        cpuset_cpus: set[int] = set()
        try:
            for group in cpuset.split(","):
                start_text, separator, end_text = group.partition("-")
                start = int(start_text)
                end = int(end_text) if separator else start
                if start < 0 or end < start:
                    raise ValueError
                cpuset_cpus.update(range(start, end + 1))
        except ValueError as error:
            raise ModalD3EvidenceError("cgroup effective cpuset is invalid") from error
        if (
            len(cpuset_cpus) < MODAL_CPU
            or not set(visible_affinity).issubset(cpuset_cpus)
            or host["pinned_cpu"] not in cpuset_cpus
        ):
            raise ModalD3EvidenceError("cgroup effective cpuset is smaller than the reservation")
    elif attestation_mode == GVISOR_ATTESTATION_MODE:
        if (
            cgroup_limits != {}
            or "gvisor" not in str(host.get("platform", "")).lower()
            or "gvisor" not in str(host.get("uname", "")).lower()
        ):
            raise ModalD3EvidenceError("gVisor fallback resource attestation is inconsistent")
    else:
        raise ModalD3EvidenceError("resource attestation mode is invalid")
    uploaded_source = provenance.get("uploaded_source")
    if (
        not isinstance(uploaded_source, Mapping)
        or not isinstance(uploaded_source.get("sha256"), str)
        or len(uploaded_source["sha256"]) != 64
        or not isinstance(uploaded_source.get("file_count"), int)
        or uploaded_source["file_count"] <= 0
        or not isinstance(uploaded_source.get("bytes"), int)
        or uploaded_source["bytes"] <= 0
    ):
        raise ModalD3EvidenceError("uploaded source identity is incomplete")
    assets = provenance.get("assets")
    if (
        not isinstance(assets, Mapping)
        or assets.get("schema_id") != "qwen-mm-logical-directory-identity-v2"
        or not isinstance(assets.get("tree_sha256"), str)
        or len(assets["tree_sha256"]) != 64
        or not isinstance(assets.get("entry_count"), int)
        or assets["entry_count"] <= 0
        or not isinstance(assets.get("logical_bytes"), int)
        or assets["logical_bytes"] <= 0
    ):
        raise ModalD3EvidenceError("asset identity is incomplete")
    expected_diff_sha256 = provenance.get("expected_diff_sha256")
    if not isinstance(expected_diff_sha256, Mapping):
        raise ModalD3EvidenceError("provenance lacks expected variant diffs")
    for variant in VARIANTS:
        if variant == "baseline":
            continue
        patch_name = f"patches/{variant}.patch"
        if sha256_bytes(files[patch_name]) != expected_diff_sha256.get(variant):
            raise ModalD3EvidenceError(f"packaged patch identity mismatch: {variant}")
    native_modules = provenance.get("variant_native_modules")
    if not isinstance(native_modules, Mapping) or set(native_modules) != set(VARIANTS):
        raise ModalD3EvidenceError("native module provenance is incomplete")
    for variant in VARIANTS:
        binary = files[f"binaries/{variant}.so"]
        identity = native_modules[variant]
        if sha256_bytes(binary) != identity.get("sha256") or "ELF 64-bit" not in identity.get(
            "file", ""
        ):
            raise ModalD3EvidenceError(f"packaged native module mismatch: {variant}")
        build_log = files[f"logs/build-{variant}.log"]
        if b"--release" not in build_log or b"Finished `release` profile" not in build_log:
            raise ModalD3EvidenceError(f"release build log is invalid: {variant}")
    summary = validate_matrix(
        records,
        expected_diff_sha256=expected_diff_sha256,
        pinned_cpu=int(provenance["host"]["pinned_cpu"]),
    )
    if summary != retained_summary:
        raise ModalD3EvidenceError("retained summary does not match raw coordinates")
    if provenance.get("matrix_run_nonce") != summary["matrix_run_nonce"]:
        raise ModalD3EvidenceError("provenance does not bind the matrix run nonce")
    protocol = provenance.get("protocol")
    expected_protocol = {
        "profiles": list(PROFILES),
        "cases": list(CASES),
        "variants": list(VARIANTS),
        "coordinates": 24,
        "warmups": 2,
        "samples": 7,
        "thread_budget": 1,
        "cooldown_seconds": COOLDOWN_SECONDS,
        "latin_orders": [list(order) for order in LATIN_ORDERS],
        "fresh_subprocess_per_coordinate": True,
        "taskset_cpu": host["pinned_cpu"],
    }
    if protocol != expected_protocol:
        raise ModalD3EvidenceError("Modal measurement protocol provenance is invalid")
    image = provenance.get("image")
    rust_version = str(image.get("rust_version", "")).split() if isinstance(image, Mapping) else []
    uv_version = str(image.get("uv_version", "")).split() if isinstance(image, Mapping) else []
    if (
        not isinstance(image, Mapping)
        or image.get("base_image") != BASE_IMAGE
        or rust_version[:2] != ["rustc", RUST_VERSION]
        or uv_version[:2] != ["uv", UV_VERSION]
    ):
        raise ModalD3EvidenceError("Modal image or toolchain provenance is invalid")
    pricing = provenance.get("pricing_snapshot")
    expected_pricing = {
        "cpu_usd_per_physical_core_second": CPU_USD_PER_PHYSICAL_CORE_SECOND,
        "memory_usd_per_gib_second": MEMORY_USD_PER_GIB_SECOND,
        "nonpreemptible_multiplier": NONPREEMPTIBLE_MULTIPLIER,
    }
    if pricing != expected_pricing:
        raise ModalD3EvidenceError("Modal pricing snapshot is invalid")
    elapsed_seconds = provenance.get("elapsed_seconds")
    cost_estimate = provenance.get("cost_estimate")
    if (
        not _is_positive_finite_number(elapsed_seconds)
        or not isinstance(cost_estimate, Mapping)
        or cost_estimate.get("basis_seconds") != elapsed_seconds
        or cost_estimate.get("scope")
        != "elapsed function worker only; excludes image construction and provider billing adjustments"
    ):
        raise ModalD3EvidenceError("elapsed worker cost estimate provenance is invalid")
    expected_cost = (
        elapsed_seconds
        * NONPREEMPTIBLE_MULTIPLIER
        * (
            MODAL_CPU * CPU_USD_PER_PHYSICAL_CORE_SECOND
            + (MODAL_MEMORY_MIB / 1024) * MEMORY_USD_PER_GIB_SECOND
        )
    )
    observed_cost = cost_estimate.get("elapsed_worker_usd")
    if not _is_positive_finite_number(observed_cost) or not math.isclose(
        observed_cost,
        expected_cost,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ModalD3EvidenceError("elapsed worker cost estimate does not reconcile")
    try:
        measure_log = files["logs/measure.log"].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ModalD3EvidenceError("measure log is not UTF-8") from error
    command_lines = [line for line in measure_log.splitlines() if line.startswith("$ ")]
    if len(command_lines) != 24 or any(
        not line.startswith("$ taskset -c ") for line in command_lines
    ):
        raise ModalD3EvidenceError("measure log does not contain exactly 24 taskset commands")
    toolchains = {_digest(record.get("toolchain")) for record in records}
    if len(toolchains) != 1:
        raise ModalD3EvidenceError("toolchain or package identity changed across coordinates")
    coordinate_native = {record["variant"]: record["native_module"]["sha256"] for record in records}
    if any(coordinate_native[variant] != native_modules[variant]["sha256"] for variant in VARIANTS):
        raise ModalD3EvidenceError("coordinate/native artifact binding mismatch")
    return {"provenance": provenance, "summary": summary, "files": files}


def write_archive_atomically(
    value: bytes,
    destination: Path,
    *,
    expected_source: tuple[str, int, int],
    expected_diff_sha256: Mapping[str, str],
    expected_assets: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_archive(value)
    provenance = validated["provenance"]
    remote_source = provenance.get("uploaded_source", {})
    observed_source = (
        remote_source.get("sha256"),
        remote_source.get("file_count"),
        remote_source.get("bytes"),
    )
    if observed_source != expected_source:
        raise ModalD3EvidenceError(
            f"remote source identity differs from local upload: {observed_source} != {expected_source}"
        )
    if provenance.get("expected_diff_sha256") != dict(expected_diff_sha256):
        raise ModalD3EvidenceError("remote patch identities differ from local patch files")
    if provenance.get("assets") != dict(expected_assets):
        raise ModalD3EvidenceError("remote asset identity differs from local hash-pinned assets")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return validated
