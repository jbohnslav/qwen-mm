"""Run one D3 coordinate in a fresh process and attest its actual affinity."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant")
    parser.add_argument("profile")
    parser.add_argument("case")
    parser.add_argument("output", type=Path)
    parser.add_argument("--pinned-cpu", type=int, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--coordinate-index", type=int, required=True)
    parser.add_argument("--order-index", type=int, required=True)
    parser.add_argument("--variant-order", nargs=4, required=True)
    parser.add_argument("--matrix-run-nonce", required=True)
    parser.add_argument("--coordinate-run-nonce", required=True)
    args = parser.parse_args()

    expected_affinity = {args.pinned_cpu}
    wrapper_affinity = os.sched_getaffinity(0)
    if wrapper_affinity != expected_affinity:
        raise RuntimeError(
            f"coordinate wrapper affinity mismatch: {wrapper_affinity} != {expected_affinity}"
        )
    command = [
        sys.executable,
        "benchmarks/d3-evidence-v1/measure.py",
        args.variant,
        args.profile,
        args.case,
        str(args.output),
    ]
    child = subprocess.Popen(command)
    child_affinity = os.sched_getaffinity(child.pid)
    if child_affinity != expected_affinity:
        child.terminate()
        child.wait()
        raise RuntimeError(
            f"measurement child affinity mismatch: {child_affinity} != {expected_affinity}"
        )
    returncode = child.wait()
    if returncode != 0:
        raise RuntimeError(f"measurement child failed with exit code {returncode}")
    record = json.loads(args.output.read_text(encoding="utf-8"))
    record["modal_protocol"] = {
        "sequence": args.sequence,
        "coordinate_index": args.coordinate_index,
        "order_index": args.order_index,
        "variant_order": args.variant_order,
        "fresh_subprocess": True,
        "pinned_cpu": args.pinned_cpu,
        "wrapper_pid": os.getpid(),
        "wrapper_sched_getaffinity": sorted(wrapper_affinity),
        "measurement_pid": child.pid,
        "measurement_sched_getaffinity": sorted(child_affinity),
        "matrix_run_nonce": args.matrix_run_nonce,
        "coordinate_run_nonce": args.coordinate_run_nonce,
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
