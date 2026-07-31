from __future__ import annotations

import argparse
import copy
import json
import random
import shlex
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from qwen_mm_reference.fixtures import repository_root

CATALOG_PATH = Path("reference/conformance/v1/corpus.json")
RULES_PATH = Path("reference/conformance/v1/rules.json")
PROFILES = {"qwen3-vl-8b", "qwen3.5-9b"}
TIERS = {"golden", "rule", "live"}
U64_MAX = (1 << 64) - 1


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repository_root() / value


def evaluate_rule(rule: Mapping[str, Any]) -> Any:
    kind = rule["kind"]
    inputs = rule["input"]
    if kind in {"round_by_factor", "smart_resize", "smart_nframes"}:
        from qwen_vl_utils import vision_process

        function = getattr(vision_process, kind)
        if kind == "smart_nframes":
            return function(inputs["options"], inputs["total_frames"], inputs["video_fps"])
        result = function(**inputs)
        return list(result) if isinstance(result, tuple) else result
    if kind == "explicit_dimensions":
        height = inputs.get("resized_height")
        width = inputs.get("resized_width")
        if (height is None) != (width is None):
            raise ValueError("resized_height and resized_width must be provided together")
        if height is None or int(height) <= 0 or int(width) <= 0:
            raise ValueError("explicit dimensions must be positive")
        return [int(height), int(width)]
    if kind == "sample_indices":
        indices = np.rint(np.linspace(0, inputs["total_frames"] - 1, inputs["nframes"])).astype(
            np.int64
        )
        timestamps = [round(int(index) / float(inputs["video_fps"]), 10) for index in indices]
        return {"indices": indices.tolist(), "timestamps": timestamps}
    if kind == "placeholder_count":
        patches = int(np.prod(inputs["grid_thw"], dtype=np.int64))
        return patches // int(inputs["merge_size"]) ** 2
    if kind == "checked_product_u64":
        product = 1
        for value in inputs["values"]:
            value = int(value)
            if value < 0 or (value and product > U64_MAX // value):
                raise OverflowError("u64 product overflow")
            product *= value
        return product
    raise ValueError(f"unknown rule kind: {kind}")


def validate_rules(document: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if document.get("schema_version") != 1:
        errors.append("rules: unsupported schema_version")
    seen: set[str] = set()
    for rule in document.get("rules", []):
        rule_id = rule.get("id")
        if not isinstance(rule_id, str) or not rule_id:
            errors.append("rules: every rule requires a non-empty id")
            continue
        if rule_id in seen:
            errors.append(f"rules: duplicate id {rule_id}")
        seen.add(rule_id)
        try:
            observed = evaluate_rule(rule)
        except Exception as error:  # noqa: BLE001 - exception type is the fixture output
            expected_error = rule.get("expected_error")
            if expected_error is None:
                errors.append(f"{rule_id}: unexpected {type(error).__name__}: {error}")
            elif type(error).__name__ != expected_error.get("exception"):
                errors.append(
                    f"{rule_id}: expected {expected_error.get('exception')}, "
                    f"got {type(error).__name__}"
                )
        else:
            if "expected_error" in rule:
                errors.append(f"{rule_id}: expected an error, got {observed!r}")
            elif observed != rule.get("expected"):
                errors.append(f"{rule_id}: expected {rule.get('expected')!r}, got {observed!r}")
    return errors


def validate_catalog(
    catalog: Mapping[str, Any], *, rules: Mapping[str, Any] | None = None
) -> list[str]:
    errors: list[str] = []
    if catalog.get("schema_version") != 1:
        errors.append("catalog: unsupported schema_version")
    if set(catalog.get("profiles", [])) != PROFILES:
        errors.append("catalog: profiles must contain both pinned aliases")
    entries = catalog.get("entries", [])
    seen: set[str] = set()
    observed_tags: set[str] = set()
    observed_profiles: set[str] = set()
    for entry in entries:
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            errors.append("catalog: every entry requires a non-empty id")
            continue
        if entry_id in seen:
            errors.append(f"catalog: duplicate id {entry_id}")
        seen.add(entry_id)
        if entry.get("tier") not in TIERS:
            errors.append(f"{entry_id}: unknown tier {entry.get('tier')}")
        profiles = set(entry.get("profiles", []))
        if not profiles or not profiles <= PROFILES:
            errors.append(f"{entry_id}: invalid profiles {sorted(profiles)}")
        observed_profiles.update(profiles)
        observed_tags.update(entry.get("tags", []))
        for field_name in ("case_path", "manifest_path", "rules_path"):
            if field_name in entry and not _repo_path(entry[field_name]).exists():
                errors.append(f"{entry_id}: missing {field_name} {entry[field_name]}")
        if entry.get("tier") == "live" and "recipe" not in entry:
            errors.append(f"{entry_id}: live entries require a recipe")
    missing_tags = set(catalog.get("required_tags", [])) - observed_tags
    if missing_tags:
        errors.append(f"catalog: missing required tags {sorted(missing_tags)}")
    if observed_profiles != PROFILES:
        errors.append(
            f"catalog: entries do not cover profiles {sorted(PROFILES - observed_profiles)}"
        )
    if rules is not None:
        errors.extend(validate_rules(rules))
    return errors


def load_and_validate() -> tuple[dict[str, Any], dict[str, Any]]:
    catalog = _load_json(_repo_path(CATALOG_PATH))
    rules = _load_json(_repo_path(RULES_PATH))
    errors = validate_catalog(catalog, rules=rules)
    if errors:
        raise ValueError("invalid conformance corpus:\n- " + "\n- ".join(errors))
    return catalog, rules


def generate_seeded_cases(seed: int, count: int) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    texts = (
        "naïve café 👩🏽‍💻",
        "e\u0301 and é must remain distinct UTF-8 inputs",
        "literal <|vision_start|><|image_pad|><|vision_end|> tokens",
        "line one\r\nline two",
        "short",
    )
    images = tuple(f"fixtures/baseline/image24/image-{index:02d}.jpg" for index in range(8))
    cases: list[dict[str, Any]] = []
    for case_index in range(count):
        request_count = 1 + rng.randrange(3)
        requests: list[dict[str, Any]] = []
        for request_index in range(request_count):
            content: list[dict[str, Any]] = [
                {"type": "text", "text": texts[rng.randrange(len(texts))]}
            ]
            image_count = rng.randrange(3)
            selected = [images[rng.randrange(len(images))] for _ in range(image_count)]
            for image_index, image in enumerate(selected):
                visual = {
                    "type": "image",
                    "image": {"path": image},
                    "resized_height": 64 + 32 * rng.randrange(2),
                    "resized_width": 64 + 32 * rng.randrange(2),
                }
                insertion = 0 if image_index % 2 else len(content)
                content.insert(insertion, visual)
            requests.append(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": f"seed={seed}; case={case_index}; request={request_index}",
                        },
                        {"role": "user", "content": content},
                    ],
                    "options": {
                        "add_generation_prompt": bool(rng.getrandbits(1)),
                        "add_vision_id": bool(selected and rng.getrandbits(1)),
                    },
                }
            )
        cases.append(
            {
                "schema_version": 1,
                "case_id": f"seed-{seed:08x}-{case_index:04d}",
                "generation": {"algorithm": "python-random-v1", "seed": seed, "index": case_index},
                "requests": requests,
            }
        )
    return cases


def _minimize_list(values: list[Any], keep_failure: Callable[[list[Any]], bool]) -> list[Any]:
    current = copy.deepcopy(values)
    granularity = 2
    while current:
        chunk = max(1, (len(current) + granularity - 1) // granularity)
        reduced = False
        for start in range(0, len(current), chunk):
            candidate = current[:start] + current[start + chunk :]
            if keep_failure(candidate):
                current = candidate
                granularity = max(2, granularity - 1)
                reduced = True
                break
        if reduced:
            continue
        if granularity >= len(current):
            break
        granularity = min(len(current), granularity * 2)
    return current


def minimize_case(
    case: Mapping[str, Any], predicate: Callable[[Mapping[str, Any]], bool]
) -> dict[str, Any]:
    current = copy.deepcopy(dict(case))
    if not predicate(current):
        raise ValueError("the supplied case does not reproduce the failure")

    original_requests = current.get("requests", [])
    current["requests"] = _minimize_list(
        original_requests,
        lambda requests: predicate({**current, "requests": requests}),
    )
    for request_index, request in enumerate(list(current.get("requests", []))):
        messages = request.get("messages", [])

        def keep_messages(candidate: list[Any], request_index: int = request_index) -> bool:
            trial = copy.deepcopy(current)
            trial["requests"][request_index]["messages"] = candidate
            return predicate(trial)

        current["requests"][request_index]["messages"] = _minimize_list(messages, keep_messages)
        for message_index, message in enumerate(
            list(current["requests"][request_index].get("messages", []))
        ):
            content = message.get("content")
            if not isinstance(content, list):
                continue

            def keep_content(
                candidate: list[Any],
                request_index: int = request_index,
                message_index: int = message_index,
            ) -> bool:
                trial = copy.deepcopy(current)
                trial["requests"][request_index]["messages"][message_index]["content"] = candidate
                return predicate(trial)

            current["requests"][request_index]["messages"][message_index]["content"] = (
                _minimize_list(content, keep_content)
            )
    current["minimized_from"] = case.get("case_id")
    return current


def _command_predicate(command: str) -> Callable[[Mapping[str, Any]], bool]:
    tokens = shlex.split(command)

    def predicate(case: Mapping[str, Any]) -> bool:
        with tempfile.TemporaryDirectory(prefix="qwen-mm-minimize-") as directory:
            case_path = Path(directory) / "case.json"
            _write_json(case_path, case)
            argv = [token.replace("{case}", str(case_path)) for token in tokens]
            result = subprocess.run(argv, check=False, capture_output=True)
            return result.returncode != 0

    return predicate


def promote_case(case: Mapping[str, Any], case_id: str, output_directory: Path) -> Path:
    if not case_id or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in case_id
    ):
        raise ValueError("case id must contain only lowercase letters, digits, '-' and '_'")
    promoted = copy.deepcopy(dict(case))
    promoted["case_id"] = case_id
    promoted["promoted_regression"] = True
    destination = output_directory / f"{case_id}.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing regression: {destination}")
    _write_json(destination, promoted)
    return destination


def run_live_differential(
    case_path: Path,
    profile: str,
    candidate_command: str,
    output_directory: Path,
) -> dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    from qwen_mm_reference.conformance import compare_manifests
    from qwen_mm_reference.golden import export_case

    case = _load_json(case_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    expected_path = output_directory / "expected" / "manifest.json"
    actual_path = output_directory / "actual" / "manifest.json"
    expected = export_case(
        case,
        profile_alias=profile,
        output_directory=expected_path.parent,
        inline_max_bytes=1_048_576,
        write_arrays=True,
    )
    _write_json(expected_path, expected)
    actual_path.parent.mkdir(parents=True, exist_ok=True)
    replacements = {
        "{case}": str(case_path.resolve()),
        "{profile}": profile,
        "{expected}": str(expected_path.resolve()),
        "{actual}": str(actual_path.resolve()),
    }
    argv = []
    for token in shlex.split(candidate_command):
        for marker, value in replacements.items():
            token = token.replace(marker, value)
        argv.append(token)
    result = subprocess.run(
        argv,
        cwd=repository_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"candidate command exited {result.returncode}: "
            f"{(result.stderr or result.stdout)[-2000:]}"
        )
    if not actual_path.exists():
        raise FileNotFoundError(f"candidate did not write {actual_path}")
    actual = _load_json(actual_path)
    report = compare_manifests(
        expected,
        actual,
        expected_root=expected_path.parent,
        actual_root=actual_path.parent,
    )
    report_path = output_directory / "report.json"
    _write_json(report_path, report.to_dict())
    return report.to_dict()


def run_seeded_matrix(
    *,
    seed: int,
    count: int,
    candidate_command: str,
    output_directory: Path,
    profiles: Sequence[str] = tuple(sorted(PROFILES)),
) -> dict[str, Any]:
    unknown = set(profiles) - PROFILES
    if unknown:
        raise ValueError(f"unknown profiles: {sorted(unknown)}")
    cases = generate_seeded_cases(seed, count)
    cases_directory = output_directory / "cases"
    results: list[dict[str, Any]] = []
    for case in cases:
        case_path = cases_directory / f"{case['case_id']}.json"
        _write_json(case_path, case)
        for profile in profiles:
            result_directory = output_directory / profile / case["case_id"]
            report = run_live_differential(case_path, profile, candidate_command, result_directory)
            results.append(
                {
                    "case_id": case["case_id"],
                    "profile": profile,
                    "passed": report["passed"],
                    "issue_count": report["issue_count"],
                    "report": str((result_directory / "report.json").relative_to(output_directory)),
                }
            )
    summary = {
        "schema_version": 1,
        "seed": seed,
        "count": count,
        "profiles": list(profiles),
        "passed": all(result["passed"] for result in results),
        "results": results,
    }
    _write_json(output_directory / "summary.json", summary)
    return summary


def _tags(entries: Iterable[Mapping[str, Any]]) -> set[str]:
    return {tag for entry in entries for tag in entry.get("tags", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the versioned qwen-mm conformance corpus.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="validate coverage and execute compact rules")
    generate_parser = subparsers.add_parser("generate", help="generate reproducible live cases")
    generate_parser.add_argument("--seed", type=int, required=True)
    generate_parser.add_argument("--count", type=int, required=True)
    generate_parser.add_argument("--output-directory", type=Path, required=True)
    minimize_parser = subparsers.add_parser("minimize", help="delta-debug a failing case")
    minimize_parser.add_argument("--case", type=Path, required=True)
    minimize_parser.add_argument("--predicate-command", required=True)
    minimize_parser.add_argument("--output", type=Path, required=True)
    promote_parser = subparsers.add_parser("promote", help="promote a minimized regression")
    promote_parser.add_argument("--case", type=Path, required=True)
    promote_parser.add_argument("--id", required=True)
    promote_parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("reference/cases/v1/regressions"),
    )
    differential_parser = subparsers.add_parser(
        "differential", help="run a case against the pinned oracle and a candidate command"
    )
    differential_parser.add_argument("--case", type=Path, required=True)
    differential_parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    differential_parser.add_argument(
        "--candidate-command",
        required=True,
        help="command template using {case}, {profile}, {expected}, and {actual}",
    )
    differential_parser.add_argument("--output-directory", type=Path, required=True)
    matrix_parser = subparsers.add_parser(
        "matrix", help="run reproducible seeded live cases across pinned profiles"
    )
    matrix_parser.add_argument("--seed", type=int, required=True)
    matrix_parser.add_argument("--count", type=int, required=True)
    matrix_parser.add_argument("--candidate-command", required=True)
    matrix_parser.add_argument("--output-directory", type=Path, required=True)
    matrix_parser.add_argument(
        "--profiles",
        default=",".join(sorted(PROFILES)),
        help="comma-separated pinned profile aliases",
    )
    args = parser.parse_args()

    if args.command == "validate":
        catalog, rules = load_and_validate()
        entries = catalog["entries"]
        print(
            f"valid corpus: {len(entries)} entries, {len(rules['rules'])} executable rules, "
            f"{len(_tags(entries))} coverage tags"
        )
        return
    if args.command == "generate":
        cases = generate_seeded_cases(args.seed, args.count)
        for case in cases:
            destination = args.output_directory / f"{case['case_id']}.json"
            _write_json(destination, case)
            print(f"generated: {destination}")
        return
    if args.command == "minimize":
        case = _load_json(args.case)
        minimized = minimize_case(case, _command_predicate(args.predicate_command))
        _write_json(args.output, minimized)
        print(f"minimized: {args.case} -> {args.output}")
        return
    if args.command == "promote":
        case = _load_json(args.case)
        destination = promote_case(case, args.id, args.output_directory)
        print(f"promoted: {destination}")
        return
    if args.command == "differential":
        report = run_live_differential(
            args.case,
            args.profile,
            args.candidate_command,
            args.output_directory,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        raise SystemExit(0 if report["passed"] else 1)
    profiles = tuple(value.strip() for value in args.profiles.split(",") if value.strip())
    summary = run_seeded_matrix(
        seed=args.seed,
        count=args.count,
        candidate_command=args.candidate_command,
        output_directory=args.output_directory,
        profiles=profiles,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
