#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
test_directory=$(mktemp -d "${TMPDIR:-/tmp}/qwen-mm-python-binding.XXXXXX")
trap 'rm -rf -- "$test_directory"' EXIT HUP INT TERM

python_bin=${PYTHON:-$("$repository_root/scripts/with-cargo.sh" uv python find)}

"$repository_root/scripts/with-cargo.sh" uv run \
    --project "$repository_root" \
    --locked \
    maturin build \
    --locked \
    --features test-hooks \
    --interpreter "$python_bin" \
    --out "$test_directory/dist"

"$repository_root/scripts/with-cargo.sh" uv venv \
    --python "$python_bin" \
    "$test_directory/venv"
wheel_path=$(find "$test_directory/dist" -maxdepth 1 -name '*.whl' -print -quit)
test -n "$wheel_path"
"$repository_root/scripts/with-cargo.sh" uv pip install \
    --python "$test_directory/venv/bin/python" \
    "$wheel_path"
"$test_directory/venv/bin/python" \
    "$repository_root/crates/qwen-mm-python/tests/pretrained.py"
"$test_directory/venv/bin/python" \
    "$repository_root/crates/qwen-mm-python/tests/docs_examples.py"
"$test_directory/venv/bin/python" \
    "$repository_root/crates/qwen-mm-python/tests/media_sources.py"
"$test_directory/venv/bin/python" \
    "$repository_root/crates/qwen-mm-python/tests/usability.py"
"$test_directory/venv/bin/python" \
    "$repository_root/crates/qwen-mm-python/tests/binding.py"
