#!/bin/sh
set -eu

cargo_path=
if command -v cargo >/dev/null 2>&1; then
    cargo_path=$(command -v cargo)
fi

rustup_cargo_home=${CARGO_HOME:-}
if [ -z "$rustup_cargo_home" ] && [ -n "${HOME:-}" ]; then
    rustup_cargo_home="$HOME/.cargo"
fi

if [ -z "$cargo_path" ] && [ -n "$rustup_cargo_home" ] && [ -x "$rustup_cargo_home/bin/cargo" ]; then
    cargo_path="$rustup_cargo_home/bin/cargo"
fi

if [ -z "$cargo_path" ]; then
    printf '%s\n' \
        "error: Cargo was not found on PATH or in the standard rustup location." \
        "Install Rust with rustup (https://rustup.rs), then retry." >&2
    exit 127
fi

if [ "${1:-}" = "--print-bin-dir" ]; then
    dirname -- "$cargo_path"
    exit 0
fi

exec "$cargo_path" "$@"
