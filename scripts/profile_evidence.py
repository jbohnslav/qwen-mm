"""Ingest, validate, and merge Phase D1 host evidence archives."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import profile_capture_support as support  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="transactionally publish both host archives")
    ingest.add_argument("--arm-archive", type=Path, required=True)
    ingest.add_argument("--x86-archive", type=Path, required=True)
    ingest.add_argument("--python", type=Path, default=ROOT / ".venv/bin/python")
    ingest.add_argument("--assets-root", type=Path, default=ROOT / "reference/.cache/huggingface")
    host = commands.add_parser("ingest-host", help="diagnostic single-host publication")
    host.add_argument("--architecture", choices=("arm64", "x86_64"), required=True)
    host.add_argument("--archive", type=Path, required=True)
    host.add_argument("--python", type=Path, default=ROOT / ".venv/bin/python")
    host.add_argument("--assets-root", type=Path, default=ROOT / "reference/.cache/huggingface")
    merge = commands.add_parser("merge", help="validate both hosts and write the final report")
    merge.add_argument("--python", type=Path, default=ROOT / ".venv/bin/python")
    merge.add_argument("--assets-root", type=Path, default=ROOT / "reference/.cache/huggingface")
    validate = commands.add_parser("validate", help="rerun all canonical installed validators")
    validate.add_argument("--python", type=Path, default=ROOT / ".venv/bin/python")
    validate.add_argument("--assets-root", type=Path, default=ROOT / "reference/.cache/huggingface")
    return parser


def _read_archive(path: Path) -> bytes:
    if not path.is_file():
        raise support.ProfileCaptureArtifactError(f"capture archive is missing: {path}")
    if path.stat().st_size > support.MAX_ARCHIVE_COMPRESSED_BYTES:
        raise support.ProfileCaptureArtifactError("capture archive exceeds the compressed size cap")
    return path.read_bytes()


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "ingest":
            validated = support.ingest_profile_artifacts(
                {
                    "arm64": _read_archive(args.arm_archive),
                    "x86_64": _read_archive(args.x86_archive),
                },
                repository_root=ROOT,
                python=args.python.resolve(),
                assets_root=args.assets_root.resolve(),
            )
            revisions = {value["provenance"]["source"]["revision"] for value in validated.values()}
            if len(revisions) != 1:
                raise support.ProfileCaptureArtifactError(
                    "published host archives do not attest the same implementation commit"
                )
            print(
                "published and canonically validated transactional ARM64+x86-64 evidence "
                f"from {revisions.pop()}"
            )
        elif args.command == "ingest-host":
            validated = support.ingest_profile_artifact(
                _read_archive(args.archive),
                repository_root=ROOT,
                python=args.python.resolve(),
                assets_root=args.assets_root.resolve(),
                architecture=args.architecture,
            )
            provenance = validated["provenance"]
            print(
                f"published and canonically validated {args.architecture} evidence from "
                f"{provenance['source']['revision']}"
            )
        elif args.command == "merge":
            support.merge_profile_evidence(
                repository_root=ROOT,
                python=args.python.resolve(),
                assets_root=args.assets_root.resolve(),
            )
            print(f"wrote and validated {support.FINAL_PROFILE_BUNDLE}")
            print(f"wrote final report {support.FINAL_PROFILE_REPORT}")
        else:
            for architecture in ("arm64", "x86_64"):
                support.canonical_validate_installed_host(
                    repository_root=ROOT,
                    python=args.python.resolve(),
                    assets_root=args.assets_root.resolve(),
                    architecture=architecture,
                )
            support.validate_final_profile_evidence(
                repository_root=ROOT, python=args.python.resolve()
            )
            print("canonical Phase C, benchmark, and profile validation passed for both hosts")
    except support.ProfileCaptureArtifactError as error:
        raise SystemExit(f"profile evidence failed: {error}") from error


if __name__ == "__main__":
    main()
