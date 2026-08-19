"""Build and validate the Phase C still-image resize-v2 evidence overlay.

The Phase C v1 report remains immutable historical evidence.  This overlay is
valid only when the production-code delta from that report's tested revision is
confined to the named resize implementation files, and a fresh installed wheel
passes the frozen resize-v2 holdout.  It deliberately cannot turn an arbitrary
new wheel plus an old passing report into current evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import subprocess
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

from .phase_c_conformance import (
    PROFILES,
    _assert_wheel_runtime_binding,
    _load_actual_array,
    _run_candidate,
    _wheel_runtime_identity,
    repository_root,
    unpatchify_image,
)
from .resize_quality_v2 import CONTRACT_ID, compare_candidate, verify_corpus

SCHEMA_ID = "qwen-mm-phase-c-conformance-overlay-v2"
SCHEMA_VERSION = 2
REPORT_PATH = Path("reference/phase-c/v2/report.json")
SUMMARY_PATH = Path("reference/phase-c/v2/summary.md")
SCHEMA_PATH = Path("reference/phase-c/v2/schema-v2.json")
BASE_REPORT_PATH = Path("reference/phase-c/v1/report.json")
RESIZE_DIRECTORY = Path("reference/resize/v2")
PRODUCTION_BLOB_PATH = Path("reference/phase-c/v2/production-resize.rgb8.bin")
PRODUCTION_RESULT_PATH = Path("reference/phase-c/v2/production-resize-result.json")
PLATFORM_CAPTURE_MODE = "architecture-local-installed-wheel-v1"
PLATFORM_BLOB_NAME = "installed-wheel-resize.rgb8.bin"
QUALITY_RECOMPUTE_REL_TOLERANCE = 1e-12
QUALITY_RECOMPUTE_ABS_TOLERANCE = 1e-12

# These are the only package-tree files permitted to differ from the exact
# Phase C v1 candidate revision. The examples and conformance test module are
# named explicitly because they changed with the backend but do not enter the
# normal library build.
ALLOWED_PRODUCTION_DELTA = (
    "Cargo.lock",
    "Cargo.toml",
    "crates/qwen-mm-core/Cargo.toml",
    "crates/qwen-mm-core/examples/resize_quality_v2_candidates.rs",
    "crates/qwen-mm-core/examples/still_resize_bakeoff.rs",
    "crates/qwen-mm-core/src/media_conformance_tests.rs",
    "crates/qwen-mm-core/src/observability.rs",
    "crates/qwen-mm-core/src/resize.rs",
)
REQUIRED_BACKEND = {
    "crate": "pic-scale",
    "version": "0.7.11",
    "filter": "Bicubic",
    "workload_strategy": "PreferQuality",
    "threading_policy": "Single",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _portable_quality_result(
    value: Mapping[str, Any], candidate_path: Path, *, display_path: Path | None = None
) -> dict[str, Any]:
    result = json.loads(json.dumps(value))
    root = repository_root()
    result["candidate"]["path"] = (
        display_path.as_posix()
        if display_path is not None
        else candidate_path.relative_to(root).as_posix()
        if candidate_path.is_relative_to(root)
        else str(candidate_path)
    )
    return result


def _assert_quality_result_equivalent(
    recorded: Any, recomputed: Any, *, label: str, path: str = "<root>"
) -> None:
    """Require the same gate result while allowing float64 reduction roundoff."""

    if isinstance(recorded, Mapping) and isinstance(recomputed, Mapping):
        if set(recorded) != set(recomputed):
            raise ValueError(f"{label} fields differ at {path}")
        for key in sorted(recorded):
            _assert_quality_result_equivalent(
                recorded[key], recomputed[key], label=label, path=f"{path}.{key}"
            )
        return
    if isinstance(recorded, list) and isinstance(recomputed, list):
        if len(recorded) != len(recomputed):
            raise ValueError(f"{label} list length differs at {path}")
        for index, (recorded_item, recomputed_item) in enumerate(
            zip(recorded, recomputed, strict=True)
        ):
            _assert_quality_result_equivalent(
                recorded_item,
                recomputed_item,
                label=label,
                path=f"{path}[{index}]",
            )
        return
    if isinstance(recorded, float) and isinstance(recomputed, float):
        if not math.isclose(
            recorded,
            recomputed,
            rel_tol=QUALITY_RECOMPUTE_REL_TOLERANCE,
            abs_tol=QUALITY_RECOMPUTE_ABS_TOLERANCE,
        ):
            raise ValueError(
                f"{label} numeric value differs at {path}: {recorded!r} vs {recomputed!r}"
            )
        return
    if type(recorded) is not type(recomputed) or recorded != recomputed:
        raise ValueError(f"{label} value differs at {path}: {recorded!r} vs {recomputed!r}")


def _record(root: Path, path: Path) -> dict[str, Any]:
    absolute = root / path
    return {
        "path": path.as_posix(),
        "byte_length": absolute.stat().st_size,
        "sha256": _sha256_file(absolute),
    }


def _merge_public_processor_outputs(
    production_blob: bytes, outputs: Sequence[tuple[Mapping[str, Any], bytes]]
) -> bytes:
    """Overlay same-host Processor output while retaining direct-core-only slices."""

    merged = bytearray(production_blob)
    for case, packed in outputs:
        record = case["pillow_image_rgb8"]
        expected_length = record["byte_length"]
        if len(packed) != expected_length:
            raise RuntimeError(f"{case['id']}: installed-wheel RGB length differs")
        start = record["offset"]
        end = start + expected_length
        if start < 0 or end > len(merged):
            raise RuntimeError(f"{case['id']}: installed-wheel RGB slice is invalid")
        merged[start:end] = packed
    return bytes(merged)


def _read_record(root: Path, value: Any, *, label: str) -> bytes:
    if not isinstance(value, Mapping) or set(value) != {"path", "byte_length", "sha256"}:
        raise ValueError(f"{label} record is incomplete")
    path_text = value.get("path")
    length = value.get("byte_length")
    digest = value.get("sha256")
    if (
        not isinstance(path_text, str)
        or not path_text
        or Path(path_text).is_absolute()
        or ".." in Path(path_text).parts
        or isinstance(length, bool)
        or not isinstance(length, int)
        or length < 0
        or not isinstance(digest, str)
        or not HEX64.fullmatch(digest)
    ):
        raise ValueError(f"{label} record is invalid")
    data = (root / path_text).read_bytes()
    if len(data) != length or _sha256_bytes(data) != digest:
        raise ValueError(f"{label} authentication failed")
    return data


def _git(*arguments: str, root: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root() if root is None else root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _production_delta(base_revision: str, current_revision: str) -> list[str]:
    prefixes = ("Cargo.toml", "Cargo.lock", "crates/qwen-mm-core", "crates/qwen-mm-python")
    output = _git("diff", "--name-only", f"{base_revision}..{current_revision}", "--", *prefixes)
    return sorted(line for line in output.splitlines() if line)


def _revision_has_records(revision: str, records: Sequence[Mapping[str, Any]]) -> bool:
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        return False
    for record in records:
        path = record.get("path")
        digest = record.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            return False
        completed = subprocess.run(
            ["git", "show", f"{revision}:{path}"],
            cwd=repository_root(),
            check=False,
            capture_output=True,
        )
        if completed.returncode or _sha256_bytes(completed.stdout) != digest:
            return False
    return True


def _wheel_sbom_identity(wheel_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(wheel_path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".cyclonedx.json")]
        if len(names) != 1:
            raise ValueError("wheel must contain exactly one CycloneDX SBOM")
        sbom_data = archive.read(names[0])
    sbom = json.loads(sbom_data)
    components = sbom.get("components")
    if not isinstance(components, list):
        raise ValueError("wheel SBOM component inventory is missing")
    matches = [
        component
        for component in components
        if component.get("name") == REQUIRED_BACKEND["crate"]
        and component.get("version") == REQUIRED_BACKEND["version"]
    ]
    if len(matches) != 1:
        raise ValueError("wheel SBOM does not contain the selected pic-scale version")
    return {
        "path": names[0],
        "byte_length": len(sbom_data),
        "sha256": _sha256_bytes(sbom_data),
        "selected_component": {
            "name": matches[0]["name"],
            "version": matches[0]["version"],
        },
    }


def capture_installed_wheel_resize(
    *,
    candidate_python: Path,
    wheel_path: Path,
    assets_root: Path,
    output_directory: Path,
    production_blob_source: Path,
    candidate_blob_path: Path,
    evidence_root: Path,
    platform_blob_path: Path,
    write_candidate_blob: bool = True,
) -> dict[str, Any]:
    """Bind direct production output to an isolated installed-wheel sample.

    The low-level holdout intentionally includes arbitrary one-axis dimensions
    that are not legal final Qwen patch geometry.  Such cases cannot be routed
    through ``Processor`` without the public planner rounding them.  The full
    blob therefore starts from the selected production core backend. Every
    holdout case whose destination is legal public geometry is independently
    rerun through the installed wheel and replaces its slice in a same-host
    evidence blob. This preserves the two direct-core-only cases without
    incorrectly requiring SIMD implementations on different architectures to
    be byte-identical inside the frozen quality envelope.
    """

    root = repository_root()
    manifest, _, _ = verify_corpus(root / RESIZE_DIRECTORY)
    source_blob = root / RESIZE_DIRECTORY / "sources.rgb8.bin"
    public_cases = [
        case
        for case in manifest["cases"]
        if case["destination"]["height"] % 32 == 0 and case["destination"]["width"] % 32 == 0
    ]
    requests = []
    for case in public_cases:
        source = case["source"]
        destination = case["destination"]
        requests.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "input_index": 0,
                                "options": {
                                    "resized_height": destination["height"],
                                    "resized_width": destination["width"],
                                },
                            },
                            {"type": "text", "text": f"resize-v2:{case['id']}"},
                        ],
                    }
                ],
                "options": {"add_generation_prompt": False},
                "images": [
                    {
                        "kind": "raw",
                        "path": str(source_blob),
                        "offset": source["offset"],
                        "length": source["byte_length"],
                        "sha256": source["sha256"],
                        "height": source["height"],
                        "width": source["width"],
                        "row_stride": source["stride_bytes"],
                        "layout": "positive",
                    }
                ],
            }
        )
    profile = PROFILES[0]
    models = _json(root / "reference/models.json")
    model = models[profile]
    assets_directory = (
        assets_root
        / ("models--" + model["model_id"].replace("/", "--"))
        / "snapshots"
        / model["revision"]
    )
    actual, actual_root = _run_candidate(
        candidate_python,
        {
            "profile": profile,
            "assets_directory": str(assets_directory),
            "requests": requests,
        },
        output_directory,
    )
    if actual.get("status") != "success":
        raise RuntimeError(f"installed-wheel resize capture failed: {actual}")
    if actual.get("isolation", {}).get("forbidden_imports"):
        raise RuntimeError("installed-wheel resize capture imported an oracle module")
    pixels = _load_actual_array(actual, actual_root, "pixel_values")
    grids = _load_actual_array(actual, actual_root, "image_grid_thw")
    image_metadata = actual.get("metadata", {}).get("images")
    if not isinstance(image_metadata, list) or len(image_metadata) != len(public_cases):
        raise RuntimeError("installed-wheel resize capture occurrence inventory differs")
    if grids.shape != (len(public_cases), 3):
        raise RuntimeError("installed-wheel resize capture grid inventory differs")
    production_blob = production_blob_source.read_bytes()
    expected_total = manifest["artifacts"]["pillow-image-rgb8.bin"]["byte_length"]
    if len(production_blob) != expected_total:
        raise RuntimeError("direct production resize blob length differs from the frozen holdout")
    public_outputs: list[tuple[Mapping[str, Any], bytes]] = []
    for index, (case, metadata) in enumerate(zip(public_cases, image_metadata, strict=True)):
        destination = case["destination"]
        expected_grid = [1, destination["height"] // 16, destination["width"] // 16]
        if grids[index].tolist() != expected_grid:
            raise RuntimeError(f"{case['id']}: installed-wheel grid differs")
        if (
            metadata.get("geometry", {}).get("height") != destination["height"]
            or metadata.get("geometry", {}).get("width") != destination["width"]
        ):
            raise RuntimeError(f"{case['id']}: installed-wheel geometry differs")
        start, end = metadata["pixel_rows"]
        prepared = unpatchify_image(pixels[start:end], destination["height"], destination["width"])
        expected_length = case["pillow_image_rgb8"]["byte_length"]
        packed = np.ascontiguousarray(prepared).tobytes(order="C")
        if len(packed) != expected_length:
            raise RuntimeError(f"{case['id']}: installed-wheel RGB length differs")
        public_outputs.append((case, packed))
    platform_blob = _merge_public_processor_outputs(production_blob, public_outputs)
    evidence_root = evidence_root.resolve()
    platform_blob_path = platform_blob_path.resolve()
    if not platform_blob_path.is_relative_to(evidence_root):
        raise RuntimeError("installed-wheel RGB evidence escapes its Phase C directory")
    platform_blob_path.parent.mkdir(parents=True, exist_ok=True)
    platform_blob_path.write_bytes(platform_blob)
    if write_candidate_blob:
        if platform_blob != production_blob:
            raise RuntimeError("same-host installed wheel differs from production core blob")
        candidate_blob_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_blob_path.write_bytes(production_blob)
    elif not candidate_blob_path.is_file() or candidate_blob_path.read_bytes() != production_blob:
        raise RuntimeError("committed production RGB8 evidence differs from capture input")
    quality = _portable_quality_result(
        compare_candidate(platform_blob_path, root / RESIZE_DIRECTORY),
        platform_blob_path,
        display_path=platform_blob_path.relative_to(evidence_root),
    )
    if quality.get("passed") is not True:
        raise RuntimeError("installed-wheel output failed the frozen resize-v2 contract")
    wheel_runtime = _wheel_runtime_identity(wheel_path)
    _assert_wheel_runtime_binding(wheel_path, actual["runtime_identity"])
    if wheel_runtime != actual["runtime_identity"]:
        raise RuntimeError("installed-wheel capture runtime is not bound to the supplied wheel")
    pixel_values_path = actual_root / actual["arrays"]["pixel_values"]["path"]
    image_grid_path = actual_root / actual["arrays"]["image_grid_thw"]["path"]
    evidence_paths = {
        "candidate_case": output_directory / "case.json",
        "candidate_result": actual_root / "result.json",
        "pixel_values": pixel_values_path,
        "image_grid_thw": image_grid_path,
        "rgb8": platform_blob_path,
    }
    if any(not path.resolve().is_relative_to(evidence_root) for path in evidence_paths.values()):
        raise RuntimeError("installed-wheel evidence escapes its Phase C directory")
    evidence = {
        name: _record(evidence_root, path.resolve().relative_to(evidence_root))
        for name, path in evidence_paths.items()
    }
    public_ids = [case["id"] for case in public_cases]
    direct_core_ids = [case["id"] for case in manifest["cases"] if case["id"] not in public_ids]
    return {
        "profile": profile,
        "case_count": len(manifest["cases"]),
        "public_processor_case_count": len(public_cases),
        "public_processor_case_ids": public_ids,
        "direct_core_case_ids": direct_core_ids,
        "public_processor_matches_installed_wheel_evidence": True,
        "installed_wheel_evidence": {"mode": PLATFORM_CAPTURE_MODE, **evidence},
        "runtime_identity": actual["runtime_identity"],
        "candidate_environment": actual["candidate_environment"],
        "isolation": actual["isolation"],
        "wheel": {
            "filename": wheel_path.name,
            "byte_length": wheel_path.stat().st_size,
            "sha256": _sha256_file(wheel_path),
            "runtime_identity": wheel_runtime,
            "sbom": _wheel_sbom_identity(wheel_path),
        },
        "quality_result": quality,
    }


def _load_evidence_array(
    evidence_root: Path,
    record: Any,
    descriptor: Mapping[str, Any],
    *,
    label: str,
) -> np.ndarray[Any, Any]:
    data = _read_record(evidence_root, record, label=label)
    try:
        array = np.load(io.BytesIO(data), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} is not a valid NumPy array") from error
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{label} is not an ndarray")
    raw = np.ascontiguousarray(array).tobytes(order="C")
    if descriptor.get("path") != Path(record["path"]).name and not str(record["path"]).endswith(
        "/" + str(descriptor.get("path"))
    ):
        raise ValueError(f"{label} path differs from the candidate descriptor")
    if (
        descriptor.get("shape") != list(array.shape)
        or descriptor.get("dtype") != array.dtype.name
        or descriptor.get("nbytes") != array.nbytes
        or descriptor.get("sha256") != _sha256_bytes(raw)
        or descriptor.get("c_contiguous") is not True
        or not array.flags.c_contiguous
    ):
        raise ValueError(f"{label} differs from the candidate descriptor")
    return array


def _validate_architecture_local_capture(
    capture: Mapping[str, Any],
    *,
    evidence_root: Path,
    manifest: Mapping[str, Any],
    committed_production_blob: bytes,
) -> dict[str, Any]:
    evidence = capture.get("installed_wheel_evidence")
    expected_evidence_fields = {
        "mode",
        "candidate_case",
        "candidate_result",
        "pixel_values",
        "image_grid_thw",
        "rgb8",
    }
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != expected_evidence_fields
        or evidence.get("mode") != PLATFORM_CAPTURE_MODE
        or capture.get("public_processor_matches_installed_wheel_evidence") is not True
    ):
        raise ValueError("architecture-local installed-wheel evidence is incomplete")
    expected_public = [
        case
        for case in manifest["cases"]
        if case["destination"]["height"] % 32 == 0 and case["destination"]["width"] % 32 == 0
    ]
    expected_public_ids = [case["id"] for case in expected_public]
    expected_direct_ids = [
        case["id"] for case in manifest["cases"] if case["id"] not in expected_public_ids
    ]
    if (
        capture.get("public_processor_case_ids") != expected_public_ids
        or capture.get("direct_core_case_ids") != expected_direct_ids
    ):
        raise ValueError("architecture-local installed-wheel case inventory differs")

    try:
        candidate_case = json.loads(
            _read_record(evidence_root, evidence["candidate_case"], label="candidate case")
        )
        candidate_result = json.loads(
            _read_record(evidence_root, evidence["candidate_result"], label="candidate result")
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("architecture-local candidate evidence is not valid JSON") from error
    requests = candidate_case.get("requests")
    try:
        observed_case_ids = [
            request["messages"][0]["content"][1]["text"].removeprefix("resize-v2:")
            for request in requests
        ]
    except (KeyError, TypeError) as error:
        raise ValueError("architecture-local candidate requests are invalid") from error
    if (
        candidate_case.get("profile") != PROFILES[0]
        or observed_case_ids != expected_public_ids
        or candidate_result.get("status") != "success"
        or candidate_result.get("runtime_identity") != capture.get("runtime_identity")
        or candidate_result.get("isolation") != capture.get("isolation")
    ):
        raise ValueError("architecture-local candidate execution binding differs")
    arrays = candidate_result.get("arrays")
    if not isinstance(arrays, Mapping):
        raise ValueError("architecture-local candidate arrays are missing")
    try:
        pixels = _load_evidence_array(
            evidence_root,
            evidence["pixel_values"],
            arrays["pixel_values"],
            label="candidate pixel_values",
        )
        grids = _load_evidence_array(
            evidence_root,
            evidence["image_grid_thw"],
            arrays["image_grid_thw"],
            label="candidate image_grid_thw",
        )
    except KeyError as error:
        raise ValueError("architecture-local candidate array descriptor is missing") from error
    image_metadata = candidate_result.get("metadata", {}).get("images")
    if not isinstance(image_metadata, list) or len(image_metadata) != len(expected_public):
        raise ValueError("architecture-local candidate occurrence inventory differs")
    if grids.shape != (len(expected_public), 3):
        raise ValueError("architecture-local candidate grid inventory differs")

    platform_blob = _read_record(evidence_root, evidence["rgb8"], label="installed-wheel RGB8")
    expected_total = manifest["artifacts"]["pillow-image-rgb8.bin"]["byte_length"]
    if len(platform_blob) != expected_total or len(committed_production_blob) != expected_total:
        raise ValueError("architecture-local RGB8 blob length differs from the frozen holdout")
    public_ids = set(expected_public_ids)
    metadata_by_id = dict(zip(expected_public_ids, image_metadata, strict=True))
    grid_by_id = dict(zip(expected_public_ids, grids, strict=True))
    for case in manifest["cases"]:
        record = case["pillow_image_rgb8"]
        start = record["offset"]
        end = start + record["byte_length"]
        if case["id"] not in public_ids:
            if platform_blob[start:end] != committed_production_blob[start:end]:
                raise ValueError(f"{case['id']}: direct-core evidence differs from selection")
            continue
        metadata = metadata_by_id[case["id"]]
        destination = case["destination"]
        expected_grid = [1, destination["height"] // 16, destination["width"] // 16]
        if grid_by_id[case["id"]].tolist() != expected_grid:
            raise ValueError(f"{case['id']}: architecture-local grid differs")
        if (
            metadata.get("geometry", {}).get("height") != destination["height"]
            or metadata.get("geometry", {}).get("width") != destination["width"]
        ):
            raise ValueError(f"{case['id']}: architecture-local geometry differs")
        row_start, row_end = metadata["pixel_rows"]
        prepared = unpatchify_image(
            pixels[row_start:row_end], destination["height"], destination["width"]
        )
        packed = np.ascontiguousarray(prepared).tobytes(order="C")
        if packed != platform_blob[start:end]:
            raise ValueError(f"{case['id']}: archived RGB8 differs from installed wheel")

    rgb8_path = evidence_root / evidence["rgb8"]["path"]
    local_quality = _portable_quality_result(
        compare_candidate(rgb8_path, repository_root() / RESIZE_DIRECTORY),
        rgb8_path,
        display_path=Path(evidence["rgb8"]["path"]),
    )
    if local_quality.get("passed") is not True:
        raise ValueError("architecture-local installed-wheel output failed resize-v2")
    _assert_quality_result_equivalent(
        capture.get("quality_result"),
        local_quality,
        label="architecture-local installed-wheel quality result",
    )
    return local_quality


def build_overlay(
    *,
    wheel_path: Path,
    capture: Mapping[str, Any],
    candidate_blob_path: Path,
    current_revision: str | None = None,
    capture_evidence_root: Path | None = None,
) -> dict[str, Any]:
    root = repository_root()
    base = _json(root / BASE_REPORT_PATH)
    revision = current_revision or _git("rev-parse", "HEAD")
    base_revision = base["provenance"]["git"]["revision"]
    delta = _production_delta(base_revision, revision)
    report = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "status": "pass",
        "passed": True,
        "created_at": datetime.now(UTC).isoformat(),
        "base_phase_c_v1": {
            "artifact": _record(root, BASE_REPORT_PATH),
            "schema_id": base["schema_id"],
            "contract_id": base["contract_id"],
            "passed": base["passed"],
            "candidate_case_count": len(base["results"]),
            "skipped_case_count": len(base["scope"]["skipped_case_ids"]),
            "tested_revision": base_revision,
            "authenticated_input_count": len(base["inputs"]),
            "source_fingerprint": base["source_fingerprint"],
        },
        "current_candidate": {
            "revision": revision,
            "production_delta_from_v1": delta,
            "allowed_production_delta": list(ALLOWED_PRODUCTION_DELTA),
            "selected_backend": dict(REQUIRED_BACKEND),
            "backend_source": _record(root, Path("crates/qwen-mm-core/src/resize.rs")),
            "dependency_lock": _record(root, Path("Cargo.lock")),
            "capture": dict(capture),
        },
        "resize_v2": {
            "contract": _record(root, Path("docs/image-resize-contract-v2.md")),
            "holdout_manifest": _record(root, RESIZE_DIRECTORY / "manifest.json"),
            "holdout_sources": _record(root, RESIZE_DIRECTORY / "sources.rgb8.bin"),
            "pillow_oracle": _record(root, RESIZE_DIRECTORY / "pillow-image-rgb8.bin"),
            "production_rgb8": _record(root, candidate_blob_path.relative_to(root)),
            "production_quality_result": _record(root, PRODUCTION_RESULT_PATH),
        },
        "provenance": {
            "command": "python -m qwen_mm_reference.phase_c_overlay_v2 capture",
            "profiles_inherited_from_v1": list(PROFILES),
            "claim": (
                "The authenticated v1 report supplies exact structural and non-resize "
                "coverage; its production-code delta is restricted to the selected resize "
                "backend, whose fresh installed-wheel output passes resize-v2."
            ),
        },
    }
    validate_overlay(report, wheel_path=wheel_path, evidence_root=capture_evidence_root)
    return report


def validate_overlay(
    report: Mapping[str, Any],
    *,
    wheel_path: Path | None = None,
    evidence_root: Path | None = None,
) -> None:
    root = repository_root()
    errors = sorted(
        Draft202012Validator(_json(root / SCHEMA_PATH)).iter_errors(report),
        key=lambda item: list(item.path),
    )
    if errors:
        first = errors[0]
        location = "/".join(str(part) for part in first.path) or "<root>"
        raise ValueError(f"Phase C v2 schema validation failed at {location}: {first.message}")
    if (
        report.get("schema_id") != SCHEMA_ID
        or report.get("schema_version") != SCHEMA_VERSION
        or report.get("contract_id") != CONTRACT_ID
        or report.get("status") != "pass"
        or report.get("passed") is not True
    ):
        raise ValueError("unsupported or non-passing Phase C v2 overlay")
    base_binding = report.get("base_phase_c_v1")
    if not isinstance(base_binding, Mapping):
        raise ValueError("Phase C v1 binding is missing")
    base_data = _read_record(root, base_binding.get("artifact"), label="Phase C v1 report")
    base = json.loads(base_data)
    if (
        base.get("schema_id") != "qwen-mm-phase-c-conformance-report-v1"
        or base.get("contract_id") != "qwen-mm-compat-v1"
        or base.get("passed") is not True
        or base.get("status") != "pass"
        or base.get("scope", {}).get("executed_case_ids")
        != base.get("scope", {}).get("declared_case_ids")
        or base.get("scope", {}).get("skipped_case_ids") != []
        or any(
            item.get("passed") is not True
            or item.get("candidate_executed") is not True
            or item.get("issues") != []
            for item in base.get("results", [])
        )
    ):
        raise ValueError("authenticated Phase C v1 report is not complete and passing")
    expected_base = {
        "artifact": base_binding["artifact"],
        "schema_id": base["schema_id"],
        "contract_id": base["contract_id"],
        "passed": True,
        "candidate_case_count": len(base["results"]),
        "skipped_case_count": 0,
        "tested_revision": base["provenance"]["git"]["revision"],
        "authenticated_input_count": len(base["inputs"]),
        "source_fingerprint": base["source_fingerprint"],
    }
    if dict(base_binding) != expected_base:
        raise ValueError("Phase C v1 summary binding is stale or incomplete")
    if not _revision_has_records(expected_base["tested_revision"], base["inputs"]):
        raise ValueError("Phase C v1 input records do not belong to its tested revision")

    current = report.get("current_candidate")
    if not isinstance(current, Mapping):
        raise ValueError("current candidate binding is missing")
    revision = current.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise ValueError("current candidate revision is invalid")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", expected_base["tested_revision"], revision],
        cwd=root,
        check=False,
        capture_output=True,
    ).returncode:
        raise ValueError("current candidate does not descend from Phase C v1 evidence")
    delta = _production_delta(expected_base["tested_revision"], revision)
    if (
        current.get("production_delta_from_v1") != delta
        or current.get("allowed_production_delta") != list(ALLOWED_PRODUCTION_DELTA)
        or not set(delta).issubset(ALLOWED_PRODUCTION_DELTA)
        or "crates/qwen-mm-core/src/resize.rs" not in delta
    ):
        raise ValueError("production delta exceeds the resize-v2 overlay scope")
    if current.get("selected_backend") != REQUIRED_BACKEND:
        raise ValueError("selected production backend identity differs")
    resize_source = _read_record(root, current.get("backend_source"), label="resize backend")
    lock = _read_record(root, current.get("dependency_lock"), label="dependency lock")
    if b"PreferQuality" not in resize_source or b"Bicubic" not in resize_source:
        raise ValueError("selected resize source does not contain the frozen configuration")
    if b'name = "pic-scale"' not in lock or b'version = "0.7.11"' not in lock:
        raise ValueError("selected pic-scale dependency is not exact-pinned in Cargo.lock")
    for record in (current["backend_source"], current["dependency_lock"]):
        completed = subprocess.run(
            ["git", "show", f"{revision}:{record['path']}"],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if completed.returncode or _sha256_bytes(completed.stdout) != record["sha256"]:
            raise ValueError("current candidate records do not belong to its named revision")

    resize = report.get("resize_v2")
    if not isinstance(resize, Mapping):
        raise ValueError("resize-v2 evidence binding is missing")
    for key, label in (
        ("contract", "resize-v2 contract"),
        ("holdout_manifest", "resize-v2 manifest"),
        ("holdout_sources", "resize-v2 sources"),
        ("pillow_oracle", "resize-v2 Pillow oracle"),
        ("production_rgb8", "production RGB8"),
        ("production_quality_result", "production quality result"),
    ):
        _read_record(root, resize.get(key), label=label)
    manifest, _, _ = verify_corpus(root / RESIZE_DIRECTORY)
    if resize["holdout_manifest"]["sha256"] != _sha256_file(
        root / RESIZE_DIRECTORY / "manifest.json"
    ):
        raise ValueError("resize-v2 holdout manifest binding differs")
    production_path = root / resize["production_rgb8"]["path"]
    recomputed_quality = _portable_quality_result(
        compare_candidate(production_path, root / RESIZE_DIRECTORY), production_path
    )
    if recomputed_quality.get("passed") is not True or len(recomputed_quality["cases"]) != len(
        manifest["cases"]
    ):
        raise ValueError("production installed-wheel output does not pass resize-v2")

    capture = current.get("capture")
    if not isinstance(capture, Mapping):
        raise ValueError("installed-wheel capture is missing")
    serialized_quality = json.loads(
        _read_record(
            root,
            resize["production_quality_result"],
            label="production quality result",
        )
    )
    _assert_quality_result_equivalent(
        serialized_quality,
        recomputed_quality,
        label="serialized production quality result",
    )
    expected_public_ids = [
        case["id"]
        for case in manifest["cases"]
        if case["destination"]["height"] % 32 == 0 and case["destination"]["width"] % 32 == 0
    ]
    if (
        capture.get("case_count") != len(manifest["cases"])
        or capture.get("profile") != PROFILES[0]
        or capture.get("public_processor_case_count") != len(expected_public_ids)
        or capture.get("public_processor_case_ids") != expected_public_ids
    ):
        raise ValueError("installed-wheel capture scope is incomplete")
    if "installed_wheel_evidence" in capture:
        if evidence_root is None:
            raise ValueError("architecture-local Phase C evidence root is required")
        _validate_architecture_local_capture(
            capture,
            evidence_root=evidence_root.resolve(),
            manifest=manifest,
            committed_production_blob=production_path.read_bytes(),
        )
    elif capture.get("public_processor_matches_production_blob") is not True:
        raise ValueError("legacy installed-wheel quality result is missing or stale")
    else:
        _assert_quality_result_equivalent(
            capture.get("quality_result"),
            recomputed_quality,
            label="legacy installed-wheel quality result",
        )
    isolation = capture.get("isolation")
    if (
        not isinstance(isolation, Mapping)
        or isolation.get("installed_origin") is not True
        or isolation.get("forbidden_imports") != []
        or isolation.get("oracle_accesses") != []
    ):
        raise ValueError("installed-wheel capture isolation is invalid")
    wheel = capture.get("wheel")
    runtime = capture.get("runtime_identity")
    if (
        not isinstance(wheel, Mapping)
        or wheel.get("runtime_identity") != runtime
        or wheel.get("sbom", {}).get("selected_component")
        != {"name": "pic-scale", "version": "0.7.11"}
    ):
        raise ValueError("installed-wheel runtime/SBOM binding is invalid")
    if wheel_path is not None:
        if (
            wheel.get("sha256") != _sha256_file(wheel_path)
            or wheel.get("byte_length") != wheel_path.stat().st_size
        ):
            raise ValueError("supplied wheel differs from overlay wheel")
        if (
            _wheel_runtime_identity(wheel_path) != runtime
            or _wheel_sbom_identity(wheel_path) != wheel["sbom"]
        ):
            raise ValueError("supplied wheel runtime or SBOM differs from overlay")


def _write_summary(path: Path, report: Mapping[str, Any]) -> None:
    base = report["base_phase_c_v1"]
    capture = report["current_candidate"]["capture"]
    lines = [
        "# Phase C text/image conformance overlay v2",
        "",
        "Status: **pass**",
        "",
        f"Inherited exact Phase C v1 cases: {base['candidate_case_count']}; skipped: 0.",
        f"Frozen resize-v2 holdout cases: {capture['case_count']}.",
        (
            "Installed-wheel byte matches at legal Qwen geometry: "
            f"{capture['public_processor_case_count']}; direct-core-only low-level cases: "
            f"{capture['case_count'] - capture['public_processor_case_count']}."
        ),
        "",
        "The v1 report remains unchanged. This overlay authenticates it, restricts the",
        "production delta to the selected resize backend, and binds the frozen holdout",
        "result to the supplied installed wheel, native module, and CycloneDX pic-scale entry.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--candidate-python", type=Path, required=True)
    capture_parser.add_argument("--wheel", type=Path, required=True)
    capture_parser.add_argument(
        "--assets-root", type=Path, default=Path("reference/.cache/huggingface")
    )
    capture_parser.add_argument("--output", type=Path)
    capture_parser.add_argument("--production-blob", type=Path, required=True)
    capture_parser.add_argument(
        "--reuse-committed-production-evidence",
        action="store_true",
        help=(
            "authenticate but do not rewrite the committed production RGB/result files; "
            "used by read-only controlled performance captures"
        ),
    )
    capture_parser.add_argument("--report", type=Path, default=REPORT_PATH)
    capture_parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--report", type=Path, default=REPORT_PATH)
    validate_parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    root = repository_root()
    if args.command == "validate":
        report_path = root / args.report
        validate_overlay(
            _json(report_path),
            wheel_path=args.wheel,
            evidence_root=report_path.parent.resolve(),
        )
        print(f"valid Phase C v2 overlay: {report_path}")
        return
    wheel = args.wheel.resolve()
    report_path = root / args.report
    summary_path = root / args.summary
    output = args.output or Path(tempfile.mkdtemp(prefix="phase-c-v2-capture-"))
    evidence_root = report_path.parent.resolve()
    platform_blob_path = evidence_root / "outputs" / PLATFORM_BLOB_NAME
    capture = capture_installed_wheel_resize(
        candidate_python=Path(os.path.abspath(args.candidate_python)),
        wheel_path=wheel,
        assets_root=(root / args.assets_root).resolve(),
        output_directory=output,
        production_blob_source=args.production_blob.resolve(),
        candidate_blob_path=root / PRODUCTION_BLOB_PATH,
        evidence_root=evidence_root,
        platform_blob_path=platform_blob_path,
        write_candidate_blob=not args.reuse_committed_production_evidence,
    )
    if args.reuse_committed_production_evidence:
        committed_quality = _portable_quality_result(
            compare_candidate(root / PRODUCTION_BLOB_PATH, root / RESIZE_DIRECTORY),
            root / PRODUCTION_BLOB_PATH,
        )
        _assert_quality_result_equivalent(
            _json(root / PRODUCTION_RESULT_PATH),
            committed_quality,
            label="committed production quality result",
        )
    else:
        committed_quality = _portable_quality_result(
            compare_candidate(root / PRODUCTION_BLOB_PATH, root / RESIZE_DIRECTORY),
            root / PRODUCTION_BLOB_PATH,
        )
        _write_json(root / PRODUCTION_RESULT_PATH, committed_quality)
    report = build_overlay(
        wheel_path=wheel,
        capture=capture,
        candidate_blob_path=root / PRODUCTION_BLOB_PATH,
        capture_evidence_root=evidence_root,
    )
    _write_json(report_path, report)
    _write_summary(summary_path, report)
    print(json.dumps({"status": "pass", "cases": capture["case_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
