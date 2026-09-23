"""Compare two prebuilt wheels with the existing paired benchmark harness.

Run with the repository's reference Python environment. Wheels are unpacked into
separate temporary import roots; neither installation nor dependencies change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-wheel", type=Path, required=True)
    parser.add_argument("--candidate-wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {"platform": platform.platform(), "python": sys.version, "runs": []}
    for label, wheel in [("before", args.baseline_wheel), ("after", args.candidate_wheel)]:
        with tempfile.TemporaryDirectory(prefix=f"qwen-tokenizers-{label}-") as imported:
            with zipfile.ZipFile(wheel) as archive:
                archive.extractall(imported)
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join([imported, str(ROOT / "reference/src")])
            command = [
                sys.executable,
                "-m",
                "qwen_mm_reference.benchmark_v2",
                "run",
                "--mode",
                "smoke",
                "--candidate-adapter",
                "qwen_mm.benchmark:create_adapter",
                "--profiles",
                "qwen3-vl-8b,qwen3.5-9b",
                "--cases",
                "text_short,text_long,image1,jpeg24_requests",
                "--thread-regimes",
                "t1",
                "--build-labels",
                "shipping",
                "--production-thread-budget",
                "8",
                "--seed",
                "20260731",
                "--process-repetitions",
                "2",
                "--warmups",
                "2",
                "--minimum-samples",
                "10",
                "--minimum-seconds",
                "0.1",
                "--output",
                str(args.output.resolve() / f"{label}.json"),
                "--report",
                str(args.output.resolve() / f"{label}.md"),
            ]
            provenance["runs"].append(
                {
                    "label": label,
                    "wheel": wheel.name,
                    "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                    "command": command,
                    "import_root": imported,
                }
            )
            if not args.validate_only:
                (args.output / "provenance.json").write_text(
                    json.dumps(provenance, indent=2) + "\n"
                )
                print(f"Starting {label}: {wheel}", flush=True)
                subprocess.run(command, env=env, cwd=ROOT, check=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "qwen_mm_reference.benchmark_v2",
                    "validate",
                    str(args.output.resolve() / f"{label}.json"),
                ],
                env=env,
                cwd=ROOT,
                check=True,
            )

            print(f"Validated {label} against its measured wheel", flush=True)


if __name__ == "__main__":
    main()
