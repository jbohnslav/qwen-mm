"""Run one D4 build label through conformance, timing, and conformance again."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from d4_capture_support import (  # noqa: E402
    BUILD_LABELS,
    COMPACT_CASES_BY_THREAD_BUDGET,
    COMPACT_MINIMUM_SAMPLES,
    COMPACT_MINIMUM_SECONDS,
    COMPACT_PROCESS_REPETITIONS,
    COMPACT_THREAD_BUDGETS,
    COMPACT_WARMUPS,
    D4_RANDOM_SEED,
    PRODUCTION_THREAD_BUDGET,
    THREAD_BUDGETS,
    D4CaptureError,
    capture_failure_report,
    initialize_private_environment_integrity,
    installed_build_identity,
    normalized_capture_environment,
    result_noise_assessment,
    verify_private_environment_integrity,
)
from profile_capture_support import (  # noqa: E402
    benchmark_validation_command,
    phase_c_overlay_command,
    phase_c_overlay_validation_command,
)

REPOSITORY_ROOT = SCRIPT_DIRECTORY.parent
WORKLOAD = REPOSITORY_ROOT / "benchmarks/workloads-v2.json"
PROFILES = ("qwen3-vl-8b", "qwen3.5-9b")
CACHED_CASE = "repeat24_cached"
SMOKE_CASES = ("text_long", "image1", "image24")
THREAD_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
)
ProgressCallback = Callable[[str, Mapping[str, object]], None]


def _report_progress(progress: ProgressCallback | None, stage: str, **details: object) -> None:
    if progress is not None:
        progress(stage, details)


def _run_logged(
    command: Sequence[str],
    *,
    log: Path,
    environment: Mapping[str, str],
    append: bool,
    execute: bool,
) -> None:
    rendered = "$ " + shlex.join(command) + "\n"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a" if append else "w", encoding="utf-8") as output:
        output.write(rendered)
    if not execute:
        return
    completed = subprocess.run(
        list(command),
        cwd=REPOSITORY_ROOT,
        env=dict(environment),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    with log.open("a", encoding="utf-8") as output:
        output.write(completed.stdout)
        if not completed.stdout.endswith("\n"):
            output.write("\n")
    if completed.returncode != 0:
        raise RuntimeError(
            f"D4 command failed ({completed.returncode}): {shlex.join(command)}\n"
            f"{completed.stdout[-4000:]}"
        )


def _paired_benchmark_command(
    *,
    python: Path,
    build_label: str,
    thread_budget: int,
    cases: Sequence[str],
    mode: str,
    process_repetitions: int,
    warmups: int,
    minimum_samples: int,
    minimum_seconds: float,
    phase_c_report: Path,
    assets_root: Path,
    output: Path,
    report: Path,
    affinity_cpus: Sequence[int] | None,
) -> list[str]:
    if (
        build_label not in BUILD_LABELS
        or thread_budget <= 0
        or not cases
        or len(cases) != len(set(cases))
        or process_repetitions <= 0
        or warmups < 0
        or minimum_samples <= 0
        or minimum_seconds < 0
    ):
        raise D4CaptureError("invalid D4 build/thread coordinate")
    command = [
        str(python),
        "-m",
        "qwen_mm_reference.benchmark_v2",
        "run",
        "--workload",
        str(WORKLOAD),
        "--mode",
        mode,
        "--reference-adapter",
        "official",
        "--candidate-adapter",
        "qwen_mm.benchmark:create_adapter",
        "--profiles",
        ",".join(PROFILES),
        "--cases",
        ",".join(cases),
        "--process-repetitions",
        str(process_repetitions),
        "--seed",
        str(D4_RANDOM_SEED),
        "--warmups",
        str(warmups),
        "--minimum-samples",
        str(minimum_samples),
        "--minimum-seconds",
        str(minimum_seconds),
        "--thread-regimes",
        f"t{thread_budget}",
        "--build-labels",
        build_label,
        "--production-thread-budget",
        str(PRODUCTION_THREAD_BUDGET),
        "--phase-c-report",
        str(phase_c_report),
        "--phase-c-assets-root",
        str(assets_root),
        "--output",
        str(output),
        "--report",
        str(report),
    ]
    if affinity_cpus is not None:
        command.extend(("--affinity-cpus", ",".join(str(cpu) for cpu in affinity_cpus)))
    return command


def benchmark_command(
    *,
    python: Path,
    build_label: str,
    thread_budget: int,
    phase_c_report: Path,
    assets_root: Path,
    output: Path,
    report: Path,
    affinity_cpus: Sequence[int] | None,
) -> list[str]:
    """Build the archived exhaustive-matrix command for one build/thread label."""

    if thread_budget not in THREAD_BUDGETS:
        raise D4CaptureError("invalid exhaustive D4 thread coordinate")
    selected_cases = ["image24"] if thread_budget in {2, 4} else _release_case_ids()
    return _paired_benchmark_command(
        python=python,
        build_label=build_label,
        thread_budget=thread_budget,
        cases=selected_cases,
        mode="dedicated",
        process_repetitions=5,
        warmups=3,
        minimum_samples=30,
        minimum_seconds=5,
        phase_c_report=phase_c_report,
        assets_root=assets_root,
        output=output,
        report=report,
        affinity_cpus=affinity_cpus,
    )


def smoke_command(
    *,
    python: Path,
    assets_root: Path,
    phase_c_report: Path,
    output: Path,
    report: Path,
) -> list[str]:
    """Return the real-adapter, non-gating local D4 smoke command."""

    return _paired_benchmark_command(
        python=python,
        build_label="shipping",
        thread_budget=1,
        cases=SMOKE_CASES,
        mode="smoke",
        process_repetitions=1,
        warmups=1,
        minimum_samples=3,
        minimum_seconds=0,
        phase_c_report=phase_c_report,
        assets_root=assets_root,
        output=output,
        report=report,
        affinity_cpus=None,
    )


def _release_case_ids() -> list[str]:
    workload = json.loads(WORKLOAD.read_text(encoding="utf-8"))
    case_ids = [
        str(case["case_id"])
        for case in workload["cases"]
        if "release" in case.get("tags", []) and case.get("case_id") != CACHED_CASE
    ]
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise D4CaptureError("release workload case inventory is empty or duplicated")
    return case_ids


def _taskset(command: Sequence[str], mask: Sequence[int] | None) -> list[str]:
    if mask is None:
        return list(command)
    if not mask or len(mask) != len(set(mask)):
        raise D4CaptureError("Linux taskset mask is empty or duplicated")
    return ["taskset", "--cpu-list", ",".join(str(cpu) for cpu in mask), *command]


def capture_plan(
    *,
    python: Path,
    wheel: Path,
    build_label: str,
    assets_root: Path,
    output_root: Path,
    affinity_masks: Mapping[int, Sequence[int] | None],
) -> dict[str, Any]:
    """Return every command before paid or hours-long execution starts."""

    if build_label not in BUILD_LABELS or set(affinity_masks) != set(THREAD_BUDGETS):
        raise D4CaptureError("capture plan does not cover the frozen D4 matrix")
    phase_c_root = output_root / "phase-c" / build_label
    pre_report = phase_c_root / "pre/report.json"
    plan: dict[str, Any] = {
        "build_label": build_label,
        "python": str(python),
        "wheel": str(wheel),
        "pre_conformance": phase_c_overlay_command(
            python=python,
            wheel=wheel,
            assets_root=assets_root,
            output=phase_c_root / "pre/outputs",
            report=pre_report,
            summary=phase_c_root / "pre/summary.md",
        ),
        "timed_matrix": [],
        "post_conformance": phase_c_overlay_command(
            python=python,
            wheel=wheel,
            assets_root=assets_root,
            output=phase_c_root / "post/outputs",
            report=phase_c_root / "post/report.json",
            summary=phase_c_root / "post/summary.md",
        ),
    }
    for budget in THREAD_BUDGETS:
        capture_root = output_root / "captures" / build_label / f"t{budget}"
        command = benchmark_command(
            python=python,
            build_label=build_label,
            thread_budget=budget,
            phase_c_report=pre_report,
            assets_root=assets_root,
            output=capture_root / "result.json",
            report=capture_root / "report.md",
            affinity_cpus=affinity_masks[budget],
        )
        plan["timed_matrix"].append(
            {
                "thread_budget": budget,
                "affinity": None
                if affinity_masks[budget] is None
                else list(affinity_masks[budget] or ()),
                "command": _taskset(command, affinity_masks[budget]),
                "output": str(capture_root / "result.json"),
                "report": str(capture_root / "report.md"),
            }
        )
    return plan


def compact_capture_plan(
    *,
    python: Path,
    wheel: Path,
    build_label: str,
    assets_root: Path,
    output_root: Path,
    affinity_masks: Mapping[int, Sequence[int] | None],
) -> dict[str, Any]:
    """Return the selected 84-subprocess D4 benchmark plan."""

    if build_label != "shipping" or set(affinity_masks) != set(COMPACT_THREAD_BUDGETS):
        raise D4CaptureError("compact capture requires shipping with exact t1/t8 masks")
    phase_c_root = output_root / "phase-c" / build_label
    pre_report = phase_c_root / "pre/report.json"
    plan: dict[str, Any] = {
        "suite": "compact",
        "build_label": build_label,
        "python": str(python),
        "wheel": str(wheel),
        "pre_conformance": phase_c_overlay_command(
            python=python,
            wheel=wheel,
            assets_root=assets_root,
            output=phase_c_root / "pre/outputs",
            report=pre_report,
            summary=phase_c_root / "pre/summary.md",
        ),
        "timed_matrix": [],
        "post_conformance": phase_c_overlay_command(
            python=python,
            wheel=wheel,
            assets_root=assets_root,
            output=phase_c_root / "post/outputs",
            report=phase_c_root / "post/report.json",
            summary=phase_c_root / "post/summary.md",
        ),
    }
    for budget in COMPACT_THREAD_BUDGETS:
        capture_root = output_root / "captures" / build_label / f"t{budget}"
        command = _paired_benchmark_command(
            python=python,
            build_label=build_label,
            thread_budget=budget,
            cases=COMPACT_CASES_BY_THREAD_BUDGET[budget],
            mode="dedicated",
            process_repetitions=COMPACT_PROCESS_REPETITIONS,
            warmups=COMPACT_WARMUPS,
            minimum_samples=COMPACT_MINIMUM_SAMPLES,
            minimum_seconds=COMPACT_MINIMUM_SECONDS,
            phase_c_report=pre_report,
            assets_root=assets_root,
            output=capture_root / "result.json",
            report=capture_root / "report.md",
            affinity_cpus=affinity_masks[budget],
        )
        plan["timed_matrix"].append(
            {
                "thread_budget": budget,
                "affinity": None
                if affinity_masks[budget] is None
                else list(affinity_masks[budget] or ()),
                "cases": list(COMPACT_CASES_BY_THREAD_BUDGET[budget]),
                "command": _taskset(command, affinity_masks[budget]),
                "output": str(capture_root / "result.json"),
                "report": str(capture_root / "report.md"),
            }
        )
    return plan


def _cached_unsupported_attestations(
    result: Mapping[str, Any], *, build_label: str, thread_budget: int
) -> list[dict[str, Any]]:
    """Prove both adapters reject cache certification without timing that row."""

    pairs = result.get("pairs")
    if not isinstance(pairs, list) or any(
        isinstance(pair, Mapping) and pair.get("case_id") == CACHED_CASE for pair in pairs
    ):
        raise D4CaptureError("repeat24_cached must never appear in timed benchmark pairs")
    if thread_budget not in {1, PRODUCTION_THREAD_BUDGET}:
        return []
    records: list[dict[str, Any]] = []
    for profile in PROFILES:
        witnesses = [
            pair
            for pair in pairs
            if isinstance(pair, Mapping)
            and pair.get("profile_alias") == profile
            and pair.get("case_id") == "repeat24_uncached"
        ]
        if len(witnesses) != 5:
            raise D4CaptureError("cached unsupported attestation lacks five uncached witnesses")
        identity_payload: list[dict[str, Any]] = []
        for pair in witnesses:
            implementations = pair.get("implementations")
            if not isinstance(implementations, Mapping):
                raise D4CaptureError("cached unsupported witness lacks implementations")
            witness_identity: dict[str, Any] = {
                "pair_id": pair.get("pair_id"),
                "repetition": pair.get("repetition"),
            }
            for implementation in ("reference", "candidate"):
                try:
                    worker = implementations[implementation]
                    supported = worker["resource_census"]["adapter_metrics"]["cache_supported"]
                    nonce = worker["worker_nonce"]
                except (KeyError, TypeError) as error:
                    raise D4CaptureError("adapter cache-support evidence is missing") from error
                if supported is not False:
                    raise D4CaptureError(f"{implementation} unexpectedly claims cache support")
                witness_identity[f"{implementation}_worker_nonce"] = nonce
            identity_payload.append(witness_identity)
        witness_sha256 = hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        records.append(
            {
                "profile_alias": profile,
                "thread_budget": thread_budget,
                "reference_cache_supported": False,
                "candidate_cache_supported": False,
                "attested_from_case": "repeat24_uncached",
                "witness_pair_count": 5,
                "witness_sha256": witness_sha256,
            }
        )
    return records


def run_compact_capture(
    *,
    python: Path,
    wheel: Path,
    assets_root: Path,
    output_root: Path,
    affinity_masks: Mapping[int, Sequence[int] | None],
    execute: bool,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run semantic conformance immediately around the compact timed matrix."""

    build_label = "shipping"
    plan = compact_capture_plan(
        python=python,
        wheel=wheel,
        build_label=build_label,
        assets_root=assets_root,
        output_root=output_root,
        affinity_masks=affinity_masks,
    )
    build_root = output_root / "builds" / build_label
    build_root.mkdir(parents=True, exist_ok=True)
    (build_root / "build.json").write_text(
        json.dumps(
            {
                "suite": "compact",
                "build_label": build_label,
                "identity": installed_build_identity(python=python, wheel=wheel),
                "plan": plan,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    matrix_log = output_root / "logs" / build_label / "matrix.log"
    environment = normalized_capture_environment(os.environ)
    noise_by_budget: dict[int, dict[str, Any]] = {}
    integrity_path = build_root / "environment-integrity.json"
    if execute:
        initialize_private_environment_integrity(
            python.parent.parent,
            build_label=build_label,
            evidence_path=integrity_path,
        )
        verify_private_environment_integrity(
            python.parent.parent,
            evidence_path=integrity_path,
            checkpoint="before-pre-phase-c-v2",
        )

    _report_progress(progress, "pre_conformance_started", build_label=build_label)
    _run_logged(
        plan["pre_conformance"],
        log=matrix_log,
        environment=environment,
        append=False,
        execute=execute,
    )
    if execute:
        pre_report = output_root / "phase-c" / build_label / "pre/report.json"
        _run_logged(
            phase_c_overlay_validation_command(
                python=python,
                wheel=wheel,
                report=pre_report,
            ),
            log=matrix_log,
            environment=environment,
            append=True,
            execute=True,
        )
        verify_private_environment_integrity(
            python.parent.parent,
            evidence_path=integrity_path,
            checkpoint="after-pre-phase-c-v2",
        )
    _report_progress(progress, "pre_conformance_completed", build_label=build_label)

    for coordinate in plan["timed_matrix"]:
        budget = int(coordinate["thread_budget"])
        _report_progress(
            progress,
            "timed_budget_started",
            build_label=build_label,
            thread_budget=budget,
        )
        if execute:
            verify_private_environment_integrity(
                python.parent.parent,
                evidence_path=integrity_path,
                checkpoint=f"before-t{budget}",
            )
        _run_logged(
            coordinate["command"],
            log=matrix_log,
            environment=environment,
            append=True,
            execute=execute,
        )
        if execute:
            result_path = Path(coordinate["output"])
            _run_logged(
                benchmark_validation_command(python=python, result=result_path),
                log=matrix_log,
                environment=environment,
                append=True,
                execute=True,
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            noise = result_noise_assessment(
                result, minimum_observations=COMPACT_PROCESS_REPETITIONS
            )
            noise_by_budget[budget] = noise
            result_path.with_name("noise.json").write_text(
                json.dumps(noise, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            verify_private_environment_integrity(
                python.parent.parent,
                evidence_path=integrity_path,
                checkpoint=f"after-t{budget}",
            )
        _report_progress(
            progress,
            "timed_budget_completed",
            build_label=build_label,
            thread_budget=budget,
        )

    _report_progress(progress, "post_conformance_started", build_label=build_label)
    if execute:
        verify_private_environment_integrity(
            python.parent.parent,
            evidence_path=integrity_path,
            checkpoint="before-post-phase-c-v2",
        )
    _run_logged(
        plan["post_conformance"],
        log=matrix_log,
        environment=environment,
        append=True,
        execute=execute,
    )
    if execute:
        post_report = output_root / "phase-c" / build_label / "post/report.json"
        _run_logged(
            phase_c_overlay_validation_command(
                python=python,
                wheel=wheel,
                report=post_report,
            ),
            log=matrix_log,
            environment=environment,
            append=True,
            execute=True,
        )
        verify_private_environment_integrity(
            python.parent.parent,
            evidence_path=integrity_path,
            checkpoint="after-post-phase-c-v2",
        )
        failure_report = capture_failure_report(
            build_label=build_label,
            noise_by_budget=noise_by_budget,
            budgets=COMPACT_THREAD_BUDGETS,
        )
        (build_root / "failures.json").write_text(
            json.dumps(failure_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    _report_progress(progress, "post_conformance_completed", build_label=build_label)
    completed = {
        "schema_id": "qwen-mm-d4-compact-build-capture-v1",
        "schema_version": 1,
        "suite": "compact",
        "build_label": build_label,
        "executed": execute,
        "created_at": datetime.now(UTC).isoformat(),
        "phase_c_pre_and_post": True,
        "sample_pruning": "forbidden",
        "plan": plan,
    }
    (build_root / "capture.json").write_text(
        json.dumps(completed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return completed


def run_capture(
    *,
    python: Path,
    wheel: Path,
    build_label: str,
    assets_root: Path,
    output_root: Path,
    affinity_masks: Mapping[int, Sequence[int] | None],
    execute: bool,
) -> dict[str, Any]:
    """Run full installed-wheel Phase C immediately around the timed matrix."""

    plan = capture_plan(
        python=python,
        wheel=wheel,
        build_label=build_label,
        assets_root=assets_root,
        output_root=output_root,
        affinity_masks=affinity_masks,
    )
    build_root = output_root / "builds" / build_label
    build_root.mkdir(parents=True, exist_ok=True)
    (build_root / "build.json").write_text(
        json.dumps(
            {
                "build_label": build_label,
                "identity": installed_build_identity(python=python, wheel=wheel),
                "plan": plan,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    matrix_log = output_root / "logs" / build_label / "matrix.log"
    environment = normalized_capture_environment(os.environ)
    cached_unsupported: list[dict[str, Any]] = []
    noise_by_budget: dict[int, dict[str, Any]] = {}
    integrity_path = build_root / "environment-integrity.json"
    if execute:
        initialize_private_environment_integrity(
            python.parent.parent,
            build_label=build_label,
            evidence_path=integrity_path,
        )
        verify_private_environment_integrity(
            python.parent.parent,
            evidence_path=integrity_path,
            checkpoint="before-pre-phase-c-v2",
        )

    pre_command = list(plan["pre_conformance"])
    _run_logged(
        pre_command,
        log=matrix_log,
        environment=environment,
        append=False,
        execute=execute,
    )
    if execute:
        pre_report = output_root / "phase-c" / build_label / "pre/report.json"
        _run_logged(
            phase_c_overlay_validation_command(
                python=python,
                wheel=wheel,
                report=pre_report,
            ),
            log=matrix_log,
            environment=environment,
            append=True,
            execute=True,
        )
        verify_private_environment_integrity(
            python.parent.parent,
            evidence_path=integrity_path,
            checkpoint="after-pre-phase-c-v2",
        )

    for coordinate in plan["timed_matrix"]:
        if execute:
            verify_private_environment_integrity(
                python.parent.parent,
                evidence_path=integrity_path,
                checkpoint=f"before-t{coordinate['thread_budget']}",
            )
        _run_logged(
            coordinate["command"],
            log=matrix_log,
            environment=environment,
            append=True,
            execute=execute,
        )
        if execute:
            result_path = Path(coordinate["output"])
            _run_logged(
                benchmark_validation_command(python=python, result=result_path),
                log=matrix_log,
                environment=environment,
                append=True,
                execute=True,
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            noise = result_noise_assessment(result)
            noise_by_budget[int(coordinate["thread_budget"])] = noise
            result_path.with_name("noise.json").write_text(
                json.dumps(noise, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            cached_unsupported.extend(
                _cached_unsupported_attestations(
                    result,
                    build_label=build_label,
                    thread_budget=int(coordinate["thread_budget"]),
                )
            )
            verify_private_environment_integrity(
                python.parent.parent,
                evidence_path=integrity_path,
                checkpoint=f"after-t{coordinate['thread_budget']}",
            )

    if execute:
        verify_private_environment_integrity(
            python.parent.parent,
            evidence_path=integrity_path,
            checkpoint="before-post-phase-c-v2",
        )
    _run_logged(
        plan["post_conformance"],
        log=matrix_log,
        environment=environment,
        append=True,
        execute=execute,
    )
    if execute:
        post_report = output_root / "phase-c" / build_label / "post/report.json"
        _run_logged(
            phase_c_overlay_validation_command(
                python=python,
                wheel=wheel,
                report=post_report,
            ),
            log=matrix_log,
            environment=environment,
            append=True,
            execute=True,
        )
        verify_private_environment_integrity(
            python.parent.parent,
            evidence_path=integrity_path,
            checkpoint="after-post-phase-c-v2",
        )
        cached_path = output_root / "captures" / build_label / "repeat24_cached-unsupported.json"
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        cached_path.write_text(
            json.dumps(
                {
                    "schema_id": "qwen-mm-d4-cached-unsupported-v1",
                    "schema_version": 1,
                    "build_label": build_label,
                    "case_id": CACHED_CASE,
                    "cache_mode": "enabled",
                    "support_status": "unsupported",
                    "support_reason": "adapter_cache_supported_false",
                    "timing_kind": "unsupported",
                    "timed_pair_count": 0,
                    "timed_sample_count": 0,
                    "coordinates": cached_unsupported,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        failure_report = capture_failure_report(
            build_label=build_label, noise_by_budget=noise_by_budget
        )
        (build_root / "failures.json").write_text(
            json.dumps(failure_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    completed = {
        "schema_id": "qwen-mm-d4-build-capture-v1",
        "schema_version": 1,
        "build_label": build_label,
        "executed": execute,
        "created_at": datetime.now(UTC).isoformat(),
        "phase_c_pre_and_post": True,
        "sample_pruning": "forbidden",
        "plan": plan,
    }
    (build_root / "capture.json").write_text(
        json.dumps(completed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return completed


def _load_affinity(path: Path | None) -> dict[int, Sequence[int] | None]:
    if path is None:
        return {budget: None for budget in THREAD_BUDGETS}
    value = json.loads(path.read_text(encoding="utf-8"))
    try:
        return {budget: value[f"t{budget}"] for budget in THREAD_BUDGETS}
    except (KeyError, TypeError) as error:
        raise D4CaptureError("affinity JSON must contain t1/t2/t4/t8") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--build-label", choices=BUILD_LABELS, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--affinity-json", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = run_capture(
        python=args.python.resolve(),
        wheel=args.wheel.resolve(),
        build_label=args.build_label,
        assets_root=args.assets_root.resolve(),
        output_root=args.output_root.resolve(),
        affinity_masks=_load_affinity(args.affinity_json),
        execute=args.execute,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
