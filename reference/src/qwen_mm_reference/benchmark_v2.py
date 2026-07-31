from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark_protocol import (
    RESULT_SCHEMA_ID,
    RESULT_SCHEMA_VERSION,
    BenchmarkProtocolError,
    architecture_family,
    environment_metadata,
    load_workload,
    percentile,
    physical_cpu_count,
    run_worker,
    select_cases,
    workload_provenance,
)
from .fixtures import repository_root

DEFAULT_WORKLOAD = Path("benchmarks/workloads-v2.json")
MODE_DEFAULTS: dict[str, dict[str, Any]] = {
    "smoke": {
        "process_repetitions": 1,
        "warmups": 1,
        "minimum_samples": 3,
        "minimum_seconds": 0.0,
        "tag": "smoke",
        "thread_regimes": ["one"],
        "build_labels": ["shipping"],
    },
    "dedicated": {
        "process_repetitions": 5,
        "warmups": 3,
        "minimum_samples": 30,
        "minimum_seconds": 5.0,
        "tag": "release",
        "thread_regimes": ["one", "production"],
        "build_labels": ["shipping", "native"],
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else repository_root() / path


def bootstrap_confidence_interval(
    values: Sequence[float], *, seed: int, resamples: int = 2_000
) -> dict[str, float | int]:
    if not values:
        raise BenchmarkProtocolError("cannot bootstrap an empty speedup set")
    materialized = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(materialized), size=(resamples, len(materialized)))
    medians = np.median(materialized[indices], axis=1)
    return {
        "method": "paired process-repetition bootstrap of median speedup",
        "resamples": resamples,
        "lower": float(np.quantile(medians, 0.025)),
        "upper": float(np.quantile(medians, 0.975)),
    }


def _thread_budget(regime: str) -> int:
    if regime == "one":
        return 1
    if regime == "production":
        return physical_cpu_count()
    try:
        value = int(regime)
    except ValueError as error:
        raise BenchmarkProtocolError(f"invalid thread regime: {regime}") from error
    if value <= 0:
        raise BenchmarkProtocolError("thread budget must be positive")
    return value


def randomized_orders(repetitions: int, *, seed: int) -> list[list[str]]:
    if repetitions <= 0:
        raise BenchmarkProtocolError("process repetitions must be positive")
    rng = random.Random(seed)
    orders: list[list[str]] = []
    for repetition in range(repetitions):
        order = ["reference", "candidate"]
        if rng.randrange(2):
            order.reverse()
        if repetition > 0 and order == orders[-1] and repetitions > 2:
            order.reverse()
        orders.append(order)
    return orders


def _run_subprocess_worker(
    *,
    config: Mapping[str, Any],
    workload_path: Path,
    case_id: str,
    temporary_directory: Path,
) -> dict[str, Any]:
    token = (
        f"{config['implementation']}-{case_id}-{config['profile_alias']}-"
        f"{config['build_label']}-{config['thread_budget']}-{config['repetition']}"
    )
    config_path = temporary_directory / f"{token}-config.json"
    output_path = temporary_directory / f"{token}-output.json"
    _write_json(config_path, config)
    command = [
        sys.executable,
        "-m",
        "qwen_mm_reference.benchmark_v2",
        "_worker",
        "--config",
        str(config_path),
        "--workload",
        str(workload_path),
        "--case",
        case_id,
        "--output",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        cwd=repository_root(),
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if completed.returncode != 0:
        raise BenchmarkProtocolError(
            f"benchmark worker failed ({' '.join(command)}):\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    result = _load_json(output_path)
    result["worker_stdout"] = completed.stdout.strip()
    result["worker_stderr"] = completed.stderr.strip()
    return result


def _pair_key(pair: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(pair["profile_alias"]),
        str(pair["case_id"]),
        str(pair["build_label"]),
        int(pair["thread_budget"]),
    )


def summarize_pairs(pairs: Sequence[Mapping[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[_pair_key(pair)].append(pair)
    summaries: list[dict[str, Any]] = []
    for index, (key, group) in enumerate(sorted(grouped.items())):
        profile_alias, case_id, build_label, thread_budget = key
        reference_samples = [
            float(sample["wall_ms"])
            for pair in group
            for sample in pair["implementations"]["reference"]["samples"]
        ]
        candidate_samples = [
            float(sample["wall_ms"])
            for pair in group
            for sample in pair["implementations"]["candidate"]["samples"]
        ]
        speedups = [float(pair["paired_speedup_p50"]) for pair in group]
        summaries.append(
            {
                "profile_alias": profile_alias,
                "case_id": case_id,
                "release_name": group[0].get("release_name"),
                "build_label": build_label,
                "thread_budget": thread_budget,
                "process_repetitions": len(group),
                "reference": {
                    "sample_count": len(reference_samples),
                    "p50_ms": percentile(reference_samples, 0.50),
                    "p90_ms": percentile(reference_samples, 0.90),
                    "p99_qualified": len(reference_samples) >= 100,
                    "p99_ms": (
                        percentile(reference_samples, 0.99)
                        if len(reference_samples) >= 100
                        else None
                    ),
                },
                "candidate": {
                    "sample_count": len(candidate_samples),
                    "p50_ms": percentile(candidate_samples, 0.50),
                    "p90_ms": percentile(candidate_samples, 0.90),
                    "p99_qualified": len(candidate_samples) >= 100,
                    "p99_ms": (
                        percentile(candidate_samples, 0.99)
                        if len(candidate_samples) >= 100
                        else None
                    ),
                },
                "paired_process_speedups": speedups,
                "speedup_p50": statistics.median(speedups),
                "speedup_bootstrap_95_ci": bootstrap_confidence_interval(
                    speedups, seed=seed + index
                ),
            }
        )
    return summaries


def validate_result(result: Mapping[str, Any]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise BenchmarkProtocolError("unsupported benchmark result schema version")
    if result.get("schema_id") != RESULT_SCHEMA_ID:
        raise BenchmarkProtocolError("benchmark result schema ID mismatch")
    if result.get("mode") not in MODE_DEFAULTS:
        raise BenchmarkProtocolError("benchmark result has an invalid mode")
    family = result.get("architecture_family")
    if family not in {"arm64", "x86_64", "other"}:
        raise BenchmarkProtocolError("benchmark result has an invalid architecture family")
    pairs = result.get("pairs")
    summaries = result.get("summaries")
    if not isinstance(pairs, list) or not isinstance(summaries, list):
        raise BenchmarkProtocolError("benchmark result pairs and summaries must be arrays")
    for pair in pairs:
        implementations = pair.get("implementations")
        if not isinstance(implementations, Mapping):
            raise BenchmarkProtocolError("benchmark pair lacks implementations")
        if set(implementations) != {"reference", "candidate"}:
            raise BenchmarkProtocolError("benchmark pair must contain reference and candidate")
        fingerprints = {
            implementation["input_fingerprint"] for implementation in implementations.values()
        }
        if len(fingerprints) != 1:
            raise BenchmarkProtocolError("paired implementations received different inputs")
        for implementation in implementations.values():
            if implementation.get("environment", {}).get("architecture_family") != family:
                raise BenchmarkProtocolError("mixed architectures are not allowed in one result")
            conformance = implementation.get("conformance", {})
            if conformance.get("pre_measurement") != "pass":
                raise BenchmarkProtocolError("pre-measurement conformance did not pass")
            if conformance.get("post_measurement") != "pass":
                raise BenchmarkProtocolError("post-measurement conformance did not pass")
            if conformance.get("all_measured_iterations_stable") is not True:
                raise BenchmarkProtocolError("measured outputs were not stable")


def run_benchmark(
    *,
    workload_path: Path,
    mode: str,
    reference_adapter: str,
    candidate_adapter: str,
    profiles: Sequence[str],
    case_ids: Sequence[str] = (),
    seed: int = 20260731,
    process_repetitions: int | None = None,
    warmups: int | None = None,
    minimum_samples: int | None = None,
    minimum_seconds: float | None = None,
    thread_regimes: Sequence[str] = (),
    build_labels: Sequence[str] = (),
) -> dict[str, Any]:
    if mode not in MODE_DEFAULTS:
        raise BenchmarkProtocolError(f"unknown benchmark mode: {mode}")
    defaults = MODE_DEFAULTS[mode]
    workload_path = _resolve_repo_path(workload_path)
    workload = load_workload(workload_path)
    model_registry = _load_json(repository_root() / "reference" / "models.json")
    unknown_profiles = sorted(set(profiles) - model_registry.keys())
    if unknown_profiles:
        raise BenchmarkProtocolError(f"unknown benchmark profiles: {', '.join(unknown_profiles)}")
    selected = select_cases(
        workload,
        case_ids=case_ids,
        tag=None if case_ids else str(defaults["tag"]),
    )
    repetitions = process_repetitions or int(defaults["process_repetitions"])
    effective_warmups = int(defaults["warmups"] if warmups is None else warmups)
    effective_samples = int(
        defaults["minimum_samples"] if minimum_samples is None else minimum_samples
    )
    effective_seconds = float(
        defaults["minimum_seconds"] if minimum_seconds is None else minimum_seconds
    )
    effective_regimes = list(thread_regimes or defaults["thread_regimes"])
    effective_builds = list(build_labels or defaults["build_labels"])
    if not profiles or effective_warmups < 0 or effective_samples <= 0 or effective_seconds < 0:
        raise BenchmarkProtocolError("invalid benchmark protocol values")

    pairs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="qwen-mm-benchmark-v2-") as temporary:
        temporary_directory = Path(temporary)
        schedule_index = 0
        for profile_alias in profiles:
            for case in selected:
                for build_label in effective_builds:
                    for regime in effective_regimes:
                        thread_budget = _thread_budget(regime)
                        orders = randomized_orders(repetitions, seed=seed + schedule_index)
                        schedule_index += 1
                        for repetition, order in enumerate(orders):
                            implementations: dict[str, Any] = {}
                            for implementation in order:
                                adapter_spec = (
                                    reference_adapter
                                    if implementation == "reference"
                                    else candidate_adapter
                                )
                                config = {
                                    "implementation": implementation,
                                    "adapter_spec": adapter_spec,
                                    "oracle_adapter_spec": reference_adapter,
                                    "profile_alias": profile_alias,
                                    "build_label": build_label,
                                    "thread_budget": thread_budget,
                                    "thread_regime": regime,
                                    "repetition": repetition,
                                    "warmups": effective_warmups,
                                    "minimum_samples": effective_samples,
                                    "minimum_seconds": effective_seconds,
                                }
                                implementations[implementation] = _run_subprocess_worker(
                                    config=config,
                                    workload_path=workload_path,
                                    case_id=case["case_id"],
                                    temporary_directory=temporary_directory,
                                )
                            fingerprints = {
                                result["input_fingerprint"] for result in implementations.values()
                            }
                            if len(fingerprints) != 1:
                                raise BenchmarkProtocolError(
                                    f"{case['case_id']}: paired subprocess input mismatch"
                                )
                            reference_p50 = implementations["reference"]["summary"]["wall_ms"][
                                "p50"
                            ]
                            candidate_p50 = implementations["candidate"]["summary"]["wall_ms"][
                                "p50"
                            ]
                            pairs.append(
                                {
                                    "pair_id": (
                                        f"{profile_alias}/{case['case_id']}/{build_label}/"
                                        f"{regime}/{repetition}"
                                    ),
                                    "profile_alias": profile_alias,
                                    "case_id": case["case_id"],
                                    "release_name": case.get("release_name"),
                                    "boundary": case["boundary"],
                                    "cache_mode": case["cache_mode"],
                                    "build_label": build_label,
                                    "thread_regime": regime,
                                    "thread_budget": thread_budget,
                                    "repetition": repetition,
                                    "order": order,
                                    "implementations": implementations,
                                    "paired_speedup_p50": reference_p50 / candidate_p50,
                                }
                            )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "schema_id": RESULT_SCHEMA_ID,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "architecture_family": architecture_family(),
        "environment": environment_metadata(),
        "workload": workload_provenance(workload_path),
        "protocol": {
            "reference_adapter": reference_adapter,
            "candidate_adapter": candidate_adapter,
            "profiles": list(profiles),
            "profile_models": {profile: model_registry[profile] for profile in profiles},
            "cases": [case["case_id"] for case in selected],
            "process_repetitions": repetitions,
            "random_seed": seed,
            "order": "randomized AB/BA per process repetition",
            "warmups": effective_warmups,
            "minimum_samples": effective_samples,
            "minimum_seconds": effective_seconds,
            "thread_regimes": effective_regimes,
            "build_labels": effective_builds,
            "p99_minimum_samples": 100,
            "threshold_enforcement": "none; D4 owns release performance gates",
            "self_test_only": "synthetic" in {reference_adapter, candidate_adapter},
        },
        "pairs": pairs,
        "summaries": summarize_pairs(pairs, seed=seed),
    }
    validate_result(result)
    return result


def render_report(result: Mapping[str, Any]) -> str:
    validate_result(result)
    lines = [
        "# qwen-mm paired benchmark v2",
        "",
        f"- Created: `{result['created_at']}`",
        f"- Mode: `{result['mode']}`",
        f"- Architecture family: `{result['architecture_family']}`",
        f"- Process repetitions: `{result['protocol']['process_repetitions']}`",
        f"- Threshold enforcement: {result['protocol']['threshold_enforcement']}",
        "",
        "All implementations passed full pre/post conformance and measured-output stability.",
        "",
        "| Profile | Case | Build | Threads | Reference p50 | Candidate p50 | Speedup | 95% CI |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in result["summaries"]:
        interval = summary["speedup_bootstrap_95_ci"]
        lines.append(
            "| {profile} | {case} | {build} | {threads} | {reference:.3f} ms | "
            "{candidate:.3f} ms | {speedup:.3f}x | {lower:.3f}–{upper:.3f}x |".format(
                profile=summary["profile_alias"],
                case=summary["case_id"],
                build=summary["build_label"],
                threads=summary["thread_budget"],
                reference=summary["reference"]["p50_ms"],
                candidate=summary["candidate"]["p50_ms"],
                speedup=summary["speedup_p50"],
                lower=interval["lower"],
                upper=interval["upper"],
            )
        )
    lines.extend(
        (
            "",
            "p99 is reported only when an implementation has at least 100 raw samples. "
            "ARM and x86 measurements are intentionally stored in separate result files.",
            "",
        )
    )
    return "\n".join(lines)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _worker_command(args: argparse.Namespace) -> None:
    config = _load_json(args.config)
    workload = load_workload(args.workload)
    cases = select_cases(workload, case_ids=[args.case])
    result = run_worker(config, cases[0])
    _write_json(args.output, result)
    print(f"wrote worker result {args.output}")


def _run_command(args: argparse.Namespace) -> None:
    result = run_benchmark(
        workload_path=args.workload,
        mode=args.mode,
        reference_adapter=args.reference_adapter,
        candidate_adapter=args.candidate_adapter,
        profiles=_parse_csv(args.profiles),
        case_ids=_parse_csv(args.cases),
        seed=args.seed,
        process_repetitions=args.process_repetitions,
        warmups=args.warmups,
        minimum_samples=args.minimum_samples,
        minimum_seconds=args.minimum_seconds,
        thread_regimes=_parse_csv(args.thread_regimes),
        build_labels=_parse_csv(args.build_labels),
    )
    _write_json(args.output, result)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(result), encoding="utf-8")
    print(f"wrote paired benchmark result {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired qwen-mm benchmark protocol v2.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run paired fresh-process measurements")
    run_parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    run_parser.add_argument("--mode", choices=sorted(MODE_DEFAULTS), default="smoke")
    run_parser.add_argument("--reference-adapter", default="official")
    run_parser.add_argument("--candidate-adapter", required=True)
    run_parser.add_argument("--profiles", default="qwen3-vl-8b,qwen3.5-9b")
    run_parser.add_argument("--cases", default="")
    run_parser.add_argument("--seed", type=int, default=20260731)
    run_parser.add_argument("--process-repetitions", type=int)
    run_parser.add_argument("--warmups", type=int)
    run_parser.add_argument("--minimum-samples", type=int)
    run_parser.add_argument("--minimum-seconds", type=float)
    run_parser.add_argument("--thread-regimes", default="")
    run_parser.add_argument("--build-labels", default="")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--report", type=Path)
    run_parser.set_defaults(func=_run_command)

    worker_parser = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--config", type=Path, required=True)
    worker_parser.add_argument("--workload", type=Path, required=True)
    worker_parser.add_argument("--case", required=True)
    worker_parser.add_argument("--output", type=Path, required=True)
    worker_parser.set_defaults(func=_worker_command)

    validate_parser = subparsers.add_parser("validate", help="validate a result file")
    validate_parser.add_argument("result", type=Path)
    validate_parser.set_defaults(func=lambda args: validate_result(_load_json(args.result)))

    report_parser = subparsers.add_parser("report", help="render a Markdown result report")
    report_parser.add_argument("result", type=Path)
    report_parser.add_argument("--output", type=Path, required=True)

    def report_command(args: argparse.Namespace) -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_report(_load_json(args.result)), encoding="utf-8")

    report_parser.set_defaults(func=report_command)
    args = parser.parse_args()
    try:
        args.func(args)
    except BenchmarkProtocolError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
