.PHONY: benchmark-v2-self-test benchmark-v2-smoke chat-conformance check conformance-full \
	conformance-smoke core-check format hooks lint pre-commit-check reference-smoke \
	media-conformance media-conformance-regenerate media-report-macos resize-conformance \
	resize-conformance-regenerate resize-report-macos rust-check sync wheel-smoke \
	phase-b-conformance phase-b-conformance-regenerate phase-c-binding-check python-binding-test

CARGO_CMD ?= ./scripts/cargo.sh
UV_CMD ?= ./scripts/with-cargo.sh uv
PYO3_PYTHON ?= $(shell $(UV_CMD) python find)
CONFORMANCE_SEED ?= 1364677966
CONFORMANCE_CASES ?= 16
CONFORMANCE_OUTPUT ?= reference/results/conformance-local
BENCHMARK_CANDIDATE ?=
BENCHMARK_OUTPUT ?= /tmp/qwen-mm-benchmark-v2.json
BENCHMARK_REPORT ?= /tmp/qwen-mm-benchmark-v2.md
RESIZE_REPORT_OUTPUT ?= reference/resize/v1/results/macos-arm64-native.json
MEDIA_REPORT_OUTPUT ?= reference/media/v1/results/macos-arm64-native.json
PHASE_B_ASSETS_ROOT ?= reference/.cache/huggingface
PHASE_B_REPORT_OUTPUT ?= /tmp/qwen-mm-phase-b-report.json

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

resize-conformance:
	$(CARGO_CMD) test --locked --package qwen-mm-core --all-targets

resize-conformance-regenerate:
	$(UV_CMD) run --locked --no-sync --package qwen-mm-reference python -m qwen_mm_reference.resize_conformance
	$(MAKE) resize-conformance

resize-report-macos: resize-conformance
	$(CARGO_CMD) run --locked --package qwen-mm-core --example resize_stage_report -- \
		--output "$(RESIZE_REPORT_OUTPUT)" --host-id macos-arm64-native --execution native

media-conformance:
	$(CARGO_CMD) test --locked --package qwen-mm-core --all-targets

media-conformance-regenerate:
	$(UV_CMD) run --locked --no-sync --package qwen-mm-reference python -m qwen_mm_reference.media_conformance
	$(MAKE) media-conformance

media-report-macos: media-conformance
	$(CARGO_CMD) run --locked --package qwen-mm-core --example media_stage_report -- \
		--output "$(MEDIA_REPORT_OUTPUT)" --host-id macos-arm64-native --execution native

phase-b-conformance:
	$(CARGO_CMD) run --locked --offline --package qwen-mm-core --example phase_b_conformance -- \
		--assets-root "$(PHASE_B_ASSETS_ROOT)" --output "$(PHASE_B_REPORT_OUTPUT)"

phase-b-conformance-regenerate:
	$(UV_CMD) run --locked --no-sync --package qwen-mm-reference \
		python -m qwen_mm_reference.phase_b_conformance
	$(MAKE) phase-b-conformance

reference-smoke:
	$(UV_CMD) sync --locked --inexact --package qwen-mm-reference
	$(UV_CMD) run --locked --no-sync --package qwen-mm-reference python -m qwen_mm_reference.fixtures verify
	$(UV_CMD) run --locked --no-sync --package qwen-mm-reference python -m unittest discover -s reference/tests
	$(UV_CMD) run --locked --no-sync --package qwen-mm-reference python -m qwen_mm_reference.golden validate reference/goldens/v1
	$(MAKE) conformance-smoke

conformance-smoke:
	$(UV_CMD) run --locked --no-sync --package qwen-mm-reference python -m qwen_mm_reference.corpus validate
	$(UV_CMD) run --locked --no-sync --package qwen-mm-reference python -m qwen_mm_reference.conformance check reference/goldens/v1

chat-conformance:
	$(UV_CMD) run --locked --no-sync --package qwen-mm-reference python -m qwen_mm_reference.chat_conformance

conformance-full: conformance-smoke
	$(if $(CANDIDATE_COMMAND),,$(error CANDIDATE_COMMAND is required; see docs/conformance-v1.md))
	$(UV_CMD) run --locked --no-sync --package qwen-mm-reference python -m qwen_mm_reference.corpus matrix --seed $(CONFORMANCE_SEED) --count $(CONFORMANCE_CASES) --candidate-command '$(CANDIDATE_COMMAND)' --output-directory $(CONFORMANCE_OUTPUT)

benchmark-v2-self-test:
	$(UV_CMD) run --locked --no-sync --package qwen-mm-reference python -m qwen_mm_reference.benchmark_v2 run \
		--mode smoke --reference-adapter synthetic --candidate-adapter synthetic \
		--profiles qwen3-vl-8b --cases text_short,image1 \
		--output "$(BENCHMARK_OUTPUT)" --report "$(BENCHMARK_REPORT)"

benchmark-v2-smoke:
	@test -n "$(BENCHMARK_CANDIDATE)" || \
		{ echo "set BENCHMARK_CANDIDATE=module:factory" >&2; exit 2; }
	$(UV_CMD) run --locked --no-sync --package qwen-mm-reference python -m qwen_mm_reference.benchmark_v2 run \
		--mode smoke --candidate-adapter "$(BENCHMARK_CANDIDATE)" \
		--output "$(BENCHMARK_OUTPUT)" --report "$(BENCHMARK_REPORT)"

wheel-smoke:
	./scripts/smoke-wheel.sh

python-binding-test:
	./scripts/test-python-binding.sh

phase-c-binding-check: check phase-b-conformance python-binding-test
