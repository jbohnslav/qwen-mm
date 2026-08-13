.PHONY: benchmark-v2-self-test benchmark-v2-smoke chat-conformance check conformance-full \
	conformance-smoke core-check format hooks lint pre-commit-check reference-smoke \
	media-conformance media-conformance-regenerate media-report-macos resize-conformance \
	resize-conformance-regenerate resize-report-macos rust-check sync wheel-smoke \
	phase-b-conformance phase-b-conformance-regenerate phase-c-binding-check \
	phase-c-conformance phase-c-conformance-validate phase-c-v2-validate phase-c-release-check python-binding-test \
	d4-arm-capture d4-certify d4-linux-capture d4-modal-capture d4-test d4-validate \
	modal-benchmark-test profile-d1-arm-archive profile-d1-ingest profile-d1-ingest-arm \
	profile-d1-ingest-x86 profile-d1-merge profile-d1-modal-archive profile-d1-test \
	profile-d1-validate

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
PHASE_C_ASSETS_ROOT ?= reference/.cache/huggingface
PROFILE_D1_ARM_ARCHIVE ?= /tmp/qwen-mm-arm-d1-profile.zip
PROFILE_D1_X86_ARCHIVE ?= /tmp/qwen-mm-modal-d1-profile.zip
D4_ARM_ARCHIVE ?= /tmp/qwen-mm-d4-arm64.zip
D4_X86_ARCHIVE ?= /tmp/qwen-mm-d4-x86_64.zip
D4_ARTIFACT ?= benchmarks/performance-certification-v1/result.json
D4_REPORT ?= benchmarks/performance-certification-v1/report.md
D4_HOST_LABEL ?= local-m4
D4_LINUX_HOST_LABEL ?= local-linux-x86
D4_LINUX_ALLOCATION_ID ?=

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

modal-benchmark-test:
	$(UV_CMD) run --locked python -m unittest discover -s scripts/tests -p 'test_modal_benchmark.py'

profile-d1-test:
	$(UV_CMD) run --locked python -m unittest discover -s scripts/tests -p 'test_profile_capture_support.py'

profile-d1-arm-archive:
	$(UV_CMD) run --locked python scripts/local_profile.py "$(PROFILE_D1_ARM_ARCHIVE)"

profile-d1-modal-archive:
	modal run scripts/modal_profile.py --output "$(PROFILE_D1_X86_ARCHIVE)"

profile-d1-ingest-arm:
	$(UV_CMD) run --locked python scripts/profile_evidence.py ingest-host \
		--architecture arm64 --archive "$(PROFILE_D1_ARM_ARCHIVE)"

profile-d1-ingest-x86:
	$(UV_CMD) run --locked python scripts/profile_evidence.py ingest-host \
		--architecture x86_64 --archive "$(PROFILE_D1_X86_ARCHIVE)"

profile-d1-ingest:
	$(UV_CMD) run --locked python scripts/profile_evidence.py ingest \
		--arm-archive "$(PROFILE_D1_ARM_ARCHIVE)" \
		--x86-archive "$(PROFILE_D1_X86_ARCHIVE)"

profile-d1-merge:
	$(UV_CMD) run --locked python scripts/profile_evidence.py merge

profile-d1-validate:
	$(UV_CMD) run --locked python scripts/profile_evidence.py validate

d4-test:
	$(UV_CMD) run --locked --no-sync --package qwen-mm-reference python -m unittest \
		reference.tests.test_benchmark_v2 \
		reference.tests.test_performance_certification_v1 \
		scripts.tests.test_d4_capture \
		scripts.tests.test_d4_linux \
		scripts.tests.test_d4_evidence

d4-arm-capture:
	$(UV_CMD) run --locked --no-sync --package qwen-mm-reference python scripts/d4_local.py \
		--execute --host-label "$(D4_HOST_LABEL)" --output "$(D4_ARM_ARCHIVE)"

d4-modal-capture:
	modal run scripts/modal_d4.py --output "$(D4_X86_ARCHIVE)"

d4-linux-capture:
	$(UV_CMD) run --locked --no-sync --package qwen-mm-reference python scripts/d4_linux.py \
		--execute --dedicated-capture --stable \
		--host-label "$(D4_LINUX_HOST_LABEL)" \
		--allocation-id "$(D4_LINUX_ALLOCATION_ID)" --output "$(D4_X86_ARCHIVE)"

d4-certify:
	$(UV_CMD) run --locked --no-sync --package qwen-mm-reference python scripts/d4_evidence.py \
		--arm64 "$(D4_ARM_ARCHIVE)" --x86-64 "$(D4_X86_ARCHIVE)" \
		--artifact "$(D4_ARTIFACT)" --report "$(D4_REPORT)"

d4-validate:
	$(UV_CMD) run --locked --no-sync --package qwen-mm-reference python scripts/d4_evidence.py \
		--arm64 "$(D4_ARM_ARCHIVE)" --x86-64 "$(D4_X86_ARCHIVE)" \
		--artifact "$(D4_ARTIFACT)" --report "$(D4_REPORT)" --validate-only

wheel-smoke:
	./scripts/smoke-wheel.sh

python-binding-test:
	./scripts/test-python-binding.sh

phase-c-binding-check: check phase-b-conformance python-binding-test

phase-c-conformance:
	PHASE_C_ASSETS_ROOT="$(abspath $(PHASE_C_ASSETS_ROOT))" ./scripts/test-phase-c-conformance.sh

phase-c-conformance-validate:
	$(UV_CMD) run --locked --package qwen-mm-reference \
		python -m qwen_mm_reference.phase_c_conformance validate \
		--assets-root "$(PHASE_C_ASSETS_ROOT)" \
		--report reference/phase-c/v1/report.json

phase-c-v2-validate:
	$(UV_CMD) run --locked --no-sync --package qwen-mm-reference \
		python -m qwen_mm_reference.phase_c_overlay_v2 validate

phase-c-release-check: phase-c-binding-check phase-c-conformance
