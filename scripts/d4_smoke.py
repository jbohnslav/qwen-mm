"""Run the real-adapter D4 gut-check suite on the current local environment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from d4_worker import smoke_command  # noqa: E402
from profile_capture_support import benchmark_validation_command  # noqa: E402


def absolute_without_resolving(path: Path) -> Path:
    """Make a path absolute while preserving virtualenv interpreter symlinks."""

    return path if path.is_absolute() else Path.cwd() / path


def run_smoke(
    *,
    python: Path,
    assets_root: Path,
    phase_c_report: Path,
    output: Path,
    report: Path,
) -> None:
    """Execute and validate one bounded local smoke result."""

    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    build_environment = dict(os.environ)
    build_environment["VIRTUAL_ENV"] = str(python.parent.parent)
    subprocess.run(
        [
            str(REPOSITORY_ROOT / "scripts/with-cargo.sh"),
            str(python.parent / "maturin"),
            "develop",
            "--release",
            "--locked",
            "--skip-install",
        ],
        cwd=REPOSITORY_ROOT,
        env=build_environment,
        check=True,
    )
    command = smoke_command(
        python=python,
        assets_root=assets_root,
        phase_c_report=phase_c_report,
        output=output,
        report=report,
    )
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
    subprocess.run(
        benchmark_validation_command(
            python=python,
            result=output,
            phase_c_source_report=phase_c_report,
        ),
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    print(f"D4 smoke passed; report: {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        type=Path,
        default=REPOSITORY_ROOT / ".venv/bin/python",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=REPOSITORY_ROOT / "reference/.cache/huggingface",
    )
    parser.add_argument(
        "--phase-c-report",
        type=Path,
        default=REPOSITORY_ROOT / "reference/phase-c/v2/report.json",
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/qwen-mm-d4-smoke.json"))
    parser.add_argument("--report", type=Path, default=Path("/tmp/qwen-mm-d4-smoke.md"))
    args = parser.parse_args()
    run_smoke(
        python=absolute_without_resolving(args.python),
        assets_root=args.assets_root.resolve(),
        phase_c_report=args.phase_c_report.resolve(),
        output=args.output.resolve(),
        report=args.report.resolve(),
    )


if __name__ == "__main__":
    main()
