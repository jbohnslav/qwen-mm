#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
test_directory=$(mktemp -d "${TMPDIR:-/tmp}/qwen-mm-phase-c.XXXXXX")
trap 'rm -rf -- "$test_directory"' EXIT HUP INT TERM

python_bin=${PYTHON:-$("$repository_root/scripts/with-cargo.sh" uv python find)}
assets_root=${PHASE_C_ASSETS_ROOT:-$repository_root/reference/.cache/huggingface}
report_path=${PHASE_C_REPORT_OUTPUT:-$repository_root/reference/phase-c/v1/report.json}
summary_path=${PHASE_C_SUMMARY_OUTPUT:-$repository_root/reference/phase-c/v1/summary.md}
artifact_output=${PHASE_C_ARTIFACT_OUTPUT:-$test_directory/evidence}

"$repository_root/scripts/with-cargo.sh" uv run \
    --project "$repository_root" \
    --locked \
    maturin build \
    --locked \
    --interpreter "$python_bin" \
    --out "$test_directory/dist"

"$repository_root/scripts/with-cargo.sh" uv venv \
    --python "$python_bin" \
    "$test_directory/candidate"
wheel_path=$(find "$test_directory/dist" -maxdepth 1 -name '*.whl' -print -quit)
test -n "$wheel_path"
"$repository_root/scripts/with-cargo.sh" uv pip install \
    --python "$test_directory/candidate/bin/python" \
    "$wheel_path"

PYTHONNOUSERSITE=1 PYTHONPATH= \
"$repository_root/scripts/with-cargo.sh" uv run \
    --project "$repository_root" \
    --locked \
    --package qwen-mm-reference \
    python -m qwen_mm_reference.phase_c_conformance run \
    --candidate-python "$test_directory/candidate/bin/python" \
    --wheel "$wheel_path" \
    --assets-root "$assets_root" \
    --output "$artifact_output" \
    --report "$report_path" \
    --summary "$summary_path"

"$repository_root/scripts/with-cargo.sh" uv run \
    --project "$repository_root" \
    --locked \
    --package qwen-mm-reference \
    python -m qwen_mm_reference.phase_c_conformance validate \
    --assets-root "$assets_root" \
    --report "$report_path"
