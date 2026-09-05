#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
test_directory=$(mktemp -d "${TMPDIR:-/tmp}/qwen-mm-transformers-consumer.XXXXXX")
trap 'rm -rf -- "$test_directory"' EXIT HUP INT TERM
python_bin=${PYTHON:-$("$repository_root/scripts/with-cargo.sh" uv python find)}

"$repository_root/scripts/with-cargo.sh" uv run \
    --project "$repository_root" --locked maturin build \
    --release --locked --interpreter "$python_bin" --out "$test_directory/dist"
"$repository_root/scripts/with-cargo.sh" uv export \
    --project "$repository_root" --locked --package qwen-mm-reference \
    --no-emit-workspace --no-dev --format requirements-txt \
    --output-file "$test_directory/requirements.txt" > /dev/null
"$repository_root/scripts/with-cargo.sh" uv venv --python "$python_bin" "$test_directory/venv"
"$repository_root/scripts/with-cargo.sh" uv pip install \
    --python "$test_directory/venv/bin/python" --require-hashes \
    -r "$test_directory/requirements.txt"
wheel_path=$(find "$test_directory/dist" -maxdepth 1 -name '*.whl' -print -quit)
test -n "$wheel_path"
"$repository_root/scripts/with-cargo.sh" uv pip install \
    --python "$test_directory/venv/bin/python" --no-deps "$wheel_path"
"$test_directory/venv/bin/python" "$repository_root/scripts/verify_transformers_consumer.py" "$@"
