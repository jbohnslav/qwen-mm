#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
test_directory=$(mktemp -d "${TMPDIR:-/tmp}/qwen-mm-rosetta-wheel.XXXXXX")
trap 'rm -rf -- "$test_directory"' EXIT HUP INT TERM
python_bin=${PYTHON:-$("$repository_root/scripts/with-cargo.sh" uv python find)}
cd "$repository_root"

"$repository_root/scripts/with-cargo.sh" uv run --locked maturin build \
    --release --locked --interpreter "$python_bin" --out "$test_directory/dist"
"$repository_root/scripts/with-cargo.sh" uv export --locked --package qwen-mm-reference \
    --no-emit-workspace --no-dev --format requirements-txt \
    --output-file "$test_directory/requirements.txt" > /dev/null
"$repository_root/scripts/with-cargo.sh" uv venv --python "$python_bin" "$test_directory/venv"
"$repository_root/scripts/with-cargo.sh" uv pip install \
    --python "$test_directory/venv/bin/python" --require-hashes -r "$test_directory/requirements.txt"
wheel_path=$(find "$test_directory/dist" -maxdepth 1 -name '*.whl' -print -quit)
test -n "$wheel_path"
if [ -n "${ROSETTA_WHEEL_DIR:-}" ]; then
    mkdir -p "$ROSETTA_WHEEL_DIR"
    cp "$wheel_path" "$ROSETTA_WHEEL_DIR/"
fi
"$repository_root/scripts/with-cargo.sh" uv pip install \
    --python "$test_directory/venv/bin/python" --no-deps "$wheel_path"
"$test_directory/venv/bin/python" - <<'PY'
from qwen_mm import Processor
for profile in Processor.supported_profiles():
    Processor.from_pretrained(profile["model_id"], cache_dir="reference/.cache/huggingface")
PY
"$test_directory/venv/bin/python" crates/qwen-mm-python/tests/rosetta_regressions.py
"$test_directory/venv/bin/python" crates/qwen-mm-python/tests/rosetta_ergonomics.py
"$test_directory/venv/bin/python" scripts/rosetta_suite.py "$@"
if [ -n "${ROSETTA_RESIZE_OUTPUT:-}" ]; then
    "$repository_root/scripts/with-cargo.sh" uv run --locked --package qwen-mm-reference \
        python scripts/verify_rosetta_resize.py \
        --candidate-python "$test_directory/venv/bin/python" --wheel "$wheel_path" \
        --output "$ROSETTA_RESIZE_OUTPUT"
fi
