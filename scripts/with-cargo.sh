#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cargo_bin_dir=$("$repository_root/scripts/cargo.sh" --print-bin-dir)

PATH="$cargo_bin_dir:$PATH"
export PATH

exec "$@"
