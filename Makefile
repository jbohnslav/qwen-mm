.PHONY: check core-check rust-check reference-smoke wheel-smoke

PYO3_PYTHON ?= $(shell uv python find 3.11)

check: rust-check reference-smoke wheel-smoke

rust-check:
	cargo fmt --all -- --check
	PYO3_PYTHON="$(PYO3_PYTHON)" cargo clippy --workspace --all-targets --locked -- -D warnings
	PYO3_PYTHON="$(PYO3_PYTHON)" cargo test --workspace --all-targets --locked
	PYO3_PYTHON="$(PYO3_PYTHON)" cargo test --workspace --doc --locked
	$(MAKE) core-check

core-check:
	cargo build --package qwen-mm-core --locked --offline

reference-smoke:
	cd reference && uv sync --locked
	cd reference && uv run --locked python -m qwen_mm_reference.fixtures verify

wheel-smoke:
	./scripts/smoke-wheel.sh
