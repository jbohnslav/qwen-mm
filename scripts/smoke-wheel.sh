#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
smoke_directory=$(mktemp -d "${TMPDIR:-/tmp}/qwen-mm-wheel-smoke.XXXXXX")
trap 'rm -rf -- "$smoke_directory"' EXIT HUP INT TERM

python_bin=${PYTHON:-$(uv python find)}

uv run --project "$repository_root" --locked maturin build \
    --locked \
    --interpreter "$python_bin" \
    --out "$smoke_directory/dist"

uv venv --python "$python_bin" "$smoke_directory/venv"
wheel_path=$(find "$smoke_directory/dist" -maxdepth 1 -name '*.whl' -print -quit)
test -n "$wheel_path"
uv pip install --python "$smoke_directory/venv/bin/python" "$wheel_path"
"$smoke_directory/venv/bin/python" \
    "$repository_root/crates/qwen-mm-python/tests/smoke.py"
