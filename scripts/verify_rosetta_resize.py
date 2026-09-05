"""Capture current-wheel resize fidelity without rewriting historical Phase C evidence."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from qwen_mm_reference.phase_c_overlay_v2 import capture_installed_wheel_resize

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-python", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Evidence ZIP destination")
    args = parser.parse_args()
    assert args.output.suffix == ".zip"
    production = ROOT / "reference/phase-c/v2/production-resize.rgb8.bin"
    with tempfile.TemporaryDirectory(prefix="qwen-mm-rosetta-resize-") as temporary:
        evidence = Path(temporary)
        capture = capture_installed_wheel_resize(
            candidate_python=args.candidate_python.absolute(),
            wheel_path=args.wheel.resolve(),
            assets_root=ROOT / "reference/.cache/huggingface",
            output_directory=evidence / "candidate",
            production_blob_source=production,
            candidate_blob_path=production,
            evidence_root=evidence,
            platform_blob_path=evidence / "outputs/resize.rgb8.bin",
            write_candidate_blob=False,
        )
        capture["scope"] = (
            "Fresh public installed-wheel cases; the two non-public geometries retain "
            "historical direct-core bytes. This is not a renewed Phase C source overlay "
            "or a performance certificate. Frozen fidelity thresholds are unchanged."
        )
        report = json.dumps(capture, indent=2) + "\n"
        (evidence / "report.json").write_text(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".json").write_text(report)
        shutil.make_archive(str(args.output.with_suffix("")), "zip", evidence)
        print(f"Current-wheel resize fidelity passed: {args.output}")


if __name__ == "__main__":
    main()
