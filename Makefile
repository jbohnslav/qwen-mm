.PHONY: check conformance-full conformance-smoke core-check format hooks lint pre-commit-check rust-check sync reference-smoke wheel-smoke

CARGO_CMD ?= ./scripts/cargo.sh
UV_CMD ?= ./scripts/with-cargo.sh uv
PYO3_PYTHON ?= $(shell $(UV_CMD) python find)
CONFORMANCE_SEED ?= 1364677966
CONFORMANCE_CASES ?= 16
CONFORMANCE_OUTPUT ?= reference/results/conformance-local

check: lint rust-check reference-smoke wheel-smoke

sync:
	$(UV_CMD) sync --locked --all-packages

hooks:
	$(UV_CMD) run --locked pre-commit install

lint:
	$(UV_CMD) run --locked ruff check .
	$(UV_CMD) run --locked ruff format --check .

format:
	$(UV_CMD) run --locked ruff check --fix .
	$(UV_CMD) run --locked ruff format .
	$(CARGO_CMD) fmt --all

pre-commit-check:
	$(UV_CMD) run --locked pre-commit run --all-files

rust-check:
	$(CARGO_CMD) fmt --all -- --check
	PYO3_PYTHON="$(PYO3_PYTHON)" $(CARGO_CMD) clippy --workspace --all-targets --locked -- -D warnings
	PYO3_PYTHON="$(PYO3_PYTHON)" $(CARGO_CMD) test --workspace --all-targets --locked
	PYO3_PYTHON="$(PYO3_PYTHON)" $(CARGO_CMD) test --workspace --doc --locked
	$(MAKE) core-check

core-check:
	$(CARGO_CMD) build --package qwen-mm-core --locked --offline

reference-smoke:
	$(UV_CMD) sync --locked --inexact --package qwen-mm-reference
	$(UV_CMD) run --locked --no-sync --package qwen-mm-reference python -m qwen_mm_reference.fixtures verify
	$(UV_CMD) run --locked --no-sync --package qwen-mm-reference python -m unittest discover -s reference/tests
	$(UV_CMD) run --locked --no-sync --package qwen-mm-reference python -m qwen_mm_reference.golden validate reference/goldens/v1
	$(MAKE) conformance-smoke

conformance-smoke:
	$(UV_CMD) run --locked --no-sync --package qwen-mm-reference python -m qwen_mm_reference.corpus validate
	$(UV_CMD) run --locked --no-sync --package qwen-mm-reference python -m qwen_mm_reference.conformance check reference/goldens/v1

conformance-full: conformance-smoke
	$(if $(CANDIDATE_COMMAND),,$(error CANDIDATE_COMMAND is required; see docs/conformance-v1.md))
	$(UV_CMD) run --locked --no-sync --package qwen-mm-reference python -m qwen_mm_reference.corpus matrix --seed $(CONFORMANCE_SEED) --count $(CONFORMANCE_CASES) --candidate-command '$(CANDIDATE_COMMAND)' --output-directory $(CONFORMANCE_OUTPUT)

wheel-smoke:
	./scripts/smoke-wheel.sh
