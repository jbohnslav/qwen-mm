from __future__ import annotations

import copy
import hashlib
import subprocess
import tempfile
import unittest
import zipfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
from qwen_mm_reference.phase_c_conformance import (
    CANONICAL_COMPARATOR,
    PHASE_E_EXCLUSIONS,
    PROFILES,
    SCHEMA_ID,
    _assert_wheel_runtime_binding,
    _byte_order,
    _candidate_message,
    _compare_exact_array,
    _create_candidate_processor,
    _descriptor,
    _git_revision_is_ancestor,
    _git_revision_matches_inputs,
    _has_forbidden_claim_string,
    _lexical_absolute,
    _validate_report_schema,
    _validate_schema_contract,
    _wheel_runtime_identity,
    expected_candidate_case_ids,
    expected_installed_boundary_rule_map,
    numeric_diagnostics,
    patchify_rgb,
    repository_root,
    source_inputs,
    unpatchify_image,
    validate_report,
)


def _schema_report() -> dict:
    runtime = {
        "package": "qwen_mm",
        "version": "0.1.0",
        "package_artifact_sha256": "1" * 64,
        "native_module": "qwen_mm._native",
        "native_artifact_sha256": "2" * 64,
    }
    asset = {
        "model_id": "model",
        "revision": "revision",
        "fingerprint": "3" * 64,
        "directory": "models--Qwen--test/snapshots/revision",
        "files": [{"name": "tokenizer.json", "sha256": "4" * 64}],
    }
    fixture_authentication = [
        {
            "case_id": f"rule:{rule_id}",
            "tier": "reference-fixture-authentication",
            "profile": "both",
            "passed": True,
            "issues": [],
            "candidate_executed": False,
            "classification": "helper-only oracle fixture",
        }
        for rule_id in expected_installed_boundary_rule_map()
    ]
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": 1,
        "contract_id": "qwen-mm-compat-v1",
        "status": "pass",
        "passed": True,
        "created_at": "2026-08-02T00:00:00+00:00",
        "scope": {
            "kind": "text_image",
            "profiles": list(PROFILES),
            "declared_case_ids": ["phase-b:qwen3-vl-8b:minimal"],
            "executed_case_ids": ["phase-b:qwen3-vl-8b:minimal"],
            "skipped_case_ids": [],
        },
        "candidate": {
            "runtime_identity": runtime,
            "environment": {
                "python": "3.11.13",
                "implementation": "CPython",
                "numpy": "2.4.6",
                "abi_tag": "cpython-311",
                "platform": "test-platform",
                "package_origin": "lib/qwen_mm/__init__.py",
                "native_origin": "lib/qwen_mm/_native.so",
                "installed_origin": True,
            },
            "wheel": {
                "filename": "qwen_mm-0.1.0-cp311-cp311-test.whl",
                "sha256": "5" * 64,
                "tag": "cp311-cp311-test",
                "build_command": "maturin build --locked",
                "bound_runtime_identity": runtime,
            },
            "gate_invocation_identity": {
                "entrypoint": "qwen_mm:Processor",
                "python_isolated_mode": True,
                "forbidden_modules": [
                    "PIL",
                    "torch",
                    "torchvision",
                    "transformers",
                    "qwen_vl_utils",
                ],
            },
        },
        "inputs": [{"path": "input", "sha256": "6" * 64}],
        "source_fingerprint": "7" * 64,
        "profile_assets": {profile: copy.deepcopy(asset) for profile in PROFILES},
        "comparator": copy.deepcopy(CANONICAL_COMPARATOR),
        "installed_boundary_rule_map": expected_installed_boundary_rule_map(),
        "phase_e_exclusions": list(PHASE_E_EXCLUSIONS),
        "fixture_authentication": fixture_authentication,
        "results": [
            {
                "case_id": "phase-b:qwen3-vl-8b:minimal",
                "tier": "committed-phase-b",
                "profile": "qwen3-vl-8b",
                "passed": True,
                "issues": [],
                "candidate_executed": True,
            }
        ],
        "provenance": {
            "command": "make phase-c-conformance",
            "git": {
                "revision": "8" * 40,
                "gate_inputs_clean": True,
                "gate_input_status": [],
            },
            "platform": {
                "python": "3.11.13",
                "implementation": "CPython",
                "system": "Darwin",
                "release": "test",
                "machine": "arm64",
                "byte_order": "little",
            },
        },
    }


class PhaseCConformanceTests(unittest.TestCase):
    @staticmethod
    @contextmanager
    def _validator_patches(report: dict):
        patchers = (
            patch(
                "qwen_mm_reference.phase_c_conformance.expected_candidate_case_ids",
                return_value=report["scope"]["declared_case_ids"],
            ),
            patch(
                "qwen_mm_reference.phase_c_conformance._input_records",
                return_value=report["inputs"],
            ),
            patch(
                "qwen_mm_reference.phase_c_conformance._source_fingerprint",
                return_value=report["source_fingerprint"],
            ),
            patch(
                "qwen_mm_reference.phase_c_conformance._asset_records",
                return_value=report["profile_assets"],
            ),
            patch(
                "qwen_mm_reference.phase_c_conformance._git_provenance",
                return_value=report["provenance"]["git"],
            ),
            patch(
                "qwen_mm_reference.phase_c_conformance._git_revision_is_ancestor",
                return_value=True,
            ),
            patch(
                "qwen_mm_reference.phase_c_conformance._git_revision_matches_inputs",
                return_value=True,
            ),
        )
        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            yield

    def test_patch_layout_round_trips_prepared_rgb_exactly(self) -> None:
        values = np.arange(64 * 96 * 3, dtype=np.uint32).reshape(64, 96, 3).astype(np.uint8)
        pixels = patchify_rgb(values)
        self.assertEqual(pixels.shape, (24, 1536))
        np.testing.assert_array_equal(unpatchify_image(pixels, 64, 96), values)

    def test_pixel_diagnostic_localizes_first_value_beyond_bound(self) -> None:
        expected = np.zeros((4, 1536), dtype=np.float32)
        actual = expected.copy()
        actual[0, 0] = 5.0e-7
        actual[2, 1024 + 3 * 16 + 7] = 0.25
        metadata = [
            {
                "request_index": 3,
                "pixel_rows": [0, 4],
                "geometry": {"image_grid_thw": [1, 2, 2]},
            }
        ]
        diagnostics = numeric_diagnostics(expected, actual, 1.0e-6, pixel_metadata=metadata)
        self.assertEqual(diagnostics["first_difference"]["index"], [2, 1079])
        coordinate = diagnostics["first_difference"]["semantic_coordinate"]
        self.assertEqual(coordinate["request"], 3)
        self.assertEqual(coordinate["patch"], 2)
        self.assertEqual(coordinate["channel"], 2)
        self.assertEqual(coordinate["temporal"], 0)
        self.assertEqual((coordinate["patch_y"], coordinate["patch_x"]), (3, 7))

    def test_phase_b_message_normalization_strips_fixture_source(self) -> None:
        message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "input_index": 0,
                    "source": "raw-a",
                    "options": {"resized_height": 64, "resized_width": 96},
                }
            ],
        }
        observed = _candidate_message(message)
        self.assertNotIn("source", observed["content"][0])
        self.assertEqual(observed["content"][0]["input_index"], 0)
        self.assertEqual(observed["content"][0]["options"]["resized_width"], 96)

    def test_descriptor_and_exact_comparator_include_byte_order_and_shape_fault(self) -> None:
        expected = np.zeros((2, 3), dtype=np.int64)
        actual = np.zeros((3, 2), dtype=np.int64)
        self.assertEqual(_descriptor(expected)["byte_order"], _byte_order(expected.dtype))
        issues: list[dict] = []
        _compare_exact_array(issues, "input_ids", expected, actual)
        self.assertTrue(any(issue["path"] == "input_ids.shape" for issue in issues))

    def test_source_inventory_ignores_pyc_and_cache_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "reference/src/qwen_mm_reference"
            package.mkdir(parents=True)
            (package / "kept.py").write_text("x = 1\n")
            cache = package / "__pycache__"
            cache.mkdir()
            (cache / "kept.cpython-311.pyc").write_bytes(b"host-specific")
            with patch("qwen_mm_reference.phase_c_conformance.repository_root", return_value=root):
                observed = source_inputs()
            self.assertIn(Path("reference/src/qwen_mm_reference/kept.py"), observed)
            self.assertFalse(any(path.suffix == ".pyc" for path in observed))

    def test_canonical_candidate_inventory_has_every_matrix_row(self) -> None:
        ids = expected_candidate_case_ids()
        self.assertEqual(len(ids), 290)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(sum(value.startswith("media:") for value in ids), 108)
        self.assertEqual(sum(value.startswith("resource:") for value in ids), 84)

    def test_candidate_processor_passes_limits_as_keyword(self) -> None:
        calls = []

        class Module:
            @staticmethod
            def Processor(*args, **kwargs):
                calls.append((args, kwargs))
                return object()

        _create_candidate_processor(
            Module,
            {"profile": "p", "assets_directory": "a", "limits": {"requests_per_batch": 1}},
        )
        self.assertEqual(calls[0][0], ("p", "a"))
        self.assertEqual(calls[0][1], {"limits": {"requests_per_batch": 1}})

    def test_lexical_absolute_preserves_virtualenv_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "base-python"
            target.write_text("binary")
            venv_python = root / "venv/bin/python"
            venv_python.parent.mkdir(parents=True)
            venv_python.symlink_to(target)
            observed = _lexical_absolute(venv_python)
            self.assertEqual(observed, venv_python)
            self.assertNotEqual(observed, observed.resolve())

    def test_json_schema_is_closed_and_matches_runtime_contract(self) -> None:
        _validate_schema_contract()

    def test_real_report_shape_passes_schema_and_nested_extras_fail(self) -> None:
        report = _schema_report()
        _validate_report_schema(report)
        mutations = (
            ("profile_assets", lambda value: value["profile_assets"].update({"extra": {}})),
            (
                "boundary map",
                lambda value: value["installed_boundary_rule_map"]["round-half-0-even"].update(
                    {"extra": True}
                ),
            ),
            ("git provenance", lambda value: value["provenance"]["git"].update({"extra": True})),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                changed = copy.deepcopy(report)
                mutate(changed)
                with self.assertRaisesRegex(ValueError, "JSON Schema violation"):
                    _validate_report_schema(changed)

    def test_validator_pins_every_comparator_field(self) -> None:
        report = _schema_report()
        mutations = {
            "policy_id": "widened-policy",
            "prepared_lossless_absolute_byte_error_max": 1,
            "prepared_lossy_absolute_byte_error_max": 2,
            "final_strict_atol": 2.0e-6,
            "final_lossy_atol": 0.01,
            "diagnostic_fields": [*CANONICAL_COMPARATOR["diagnostic_fields"], "mean_error"],
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                changed = copy.deepcopy(report)
                changed["comparator"][field] = value
                with self.assertRaisesRegex(ValueError, "JSON Schema violation"):
                    _validate_report_schema(changed)

    def test_full_validator_rejects_authenticated_input_asset_and_git_tampering(self) -> None:
        report = _schema_report()
        with self._validator_patches(report):
            validate_report(report, assets_root=Path("unused"))
        mutations = (
            (
                "inputs",
                lambda value: value["inputs"][0].update({"sha256": "a" * 64}),
                "source inputs are stale",
            ),
            (
                "assets",
                lambda value: value["profile_assets"][PROFILES[0]].update(
                    {"fingerprint": "b" * 64}
                ),
                "profile assets are stale",
            ),
            (
                "git",
                lambda value: value["provenance"]["git"].update(
                    {"gate_inputs_clean": False, "gate_input_status": [" M Makefile"]}
                ),
                "JSON Schema violation",
            ),
        )
        for label, mutate, message in mutations:
            with self.subTest(label=label):
                changed = copy.deepcopy(report)
                mutate(changed)
                with self._validator_patches(report):
                    with self.assertRaisesRegex(ValueError, message):
                        validate_report(changed, assets_root=Path("unused"))

    def test_full_validator_rejects_claims_in_non_structural_string_fields(self) -> None:
        report = _schema_report()
        mutations = (
            lambda value: value.update({"created_at": "throughput 2x faster"}),
            lambda value: value["profile_assets"][PROFILES[0]].update(
                {"directory": "benchmark/results"}
            ),
            lambda value: value["candidate"]["environment"].update({"platform": "faster platform"}),
            lambda value: value["provenance"]["platform"].update({"release": "speedup release"}),
        )
        for mutate in mutations:
            changed = copy.deepcopy(report)
            mutate(changed)
            with self._validator_patches(report):
                with self.assertRaisesRegex(ValueError, "forbidden claim"):
                    validate_report(changed, assets_root=Path("unused"))

    def test_full_validator_requires_timestamp_and_exact_asset_directory(self) -> None:
        report = _schema_report()
        invalid_timestamp = copy.deepcopy(report)
        invalid_timestamp["created_at"] = "not-a-date"
        with self._validator_patches(report):
            with self.assertRaisesRegex(ValueError, "creation timestamp is invalid"):
                validate_report(invalid_timestamp, assets_root=Path("unused"))
        wrong_directory = copy.deepcopy(report)
        wrong_directory["profile_assets"][PROFILES[0]]["directory"] = "other/assets"
        with self._validator_patches(report):
            with self.assertRaisesRegex(ValueError, "profile assets are stale"):
                validate_report(wrong_directory, assets_root=Path("unused"))

    def test_passing_schema_requires_every_case_to_execute_and_pass(self) -> None:
        report = _schema_report()
        mutations = (
            lambda value: value["results"][0].update({"candidate_executed": False}),
            lambda value: value["results"][0].update({"passed": False}),
            lambda value: value["results"][0]["issues"].append({"kind": "mismatch"}),
            lambda value: value["scope"]["skipped_case_ids"].append("phase-b:qwen3-vl-8b:minimal"),
        )
        for mutate in mutations:
            changed = copy.deepcopy(report)
            mutate(changed)
            with self.assertRaisesRegex(ValueError, "JSON Schema violation"):
                _validate_report_schema(changed)

    def test_claim_scanner_rejects_performance_language_in_string_values(self) -> None:
        self.assertTrue(_has_forbidden_claim_string({"tier": "throughput 2x faster"}))
        self.assertFalse(_has_forbidden_claim_string({"tier": "live-media"}))

    def test_git_provenance_accepts_an_ancestor_but_rejects_unknown_revision(self) -> None:
        root = repository_root()
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.assertTrue(_git_revision_is_ancestor(root, revision))
        self.assertFalse(_git_revision_is_ancestor(root, "0" * 40))

    def test_recorded_revision_must_contain_exact_authenticated_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(
                ["git", "config", "user.email", "phase-c@example.invalid"],
                cwd=root,
                check=True,
            )
            subprocess.run(["git", "config", "user.name", "Phase C Test"], cwd=root, check=True)
            source = root / "gate.txt"
            source.write_text("old\n", encoding="utf-8")
            subprocess.run(["git", "add", "gate.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "old"], cwd=root, check=True)
            old_revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            source.write_text("new\n", encoding="utf-8")
            subprocess.run(["git", "add", "gate.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "new"], cwd=root, check=True)
            new_revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            records = [
                {
                    "path": "gate.txt",
                    "sha256": hashlib.sha256(b"new\n").hexdigest(),
                }
            ]
            self.assertTrue(_git_revision_matches_inputs(root, new_revision, records))
            self.assertFalse(_git_revision_matches_inputs(root, old_revision, records))

    def test_wheel_runtime_binding_rejects_a_swapped_wheel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def make_wheel(name: str, package: bytes, native: bytes) -> Path:
                path = root / name
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("qwen_mm/__init__.py", package)
                    archive.writestr("qwen_mm/_native.test.so", native)
                    archive.writestr(
                        "qwen_mm-0.1.0.dist-info/METADATA",
                        "Metadata-Version: 2.4\nName: qwen-mm\nVersion: 0.1.0\n",
                    )
                return path

            wheel_a = make_wheel("a.whl", b"package-a", b"native-a")
            wheel_b = make_wheel("b.whl", b"package-b", b"native-b")
            runtime_a = _wheel_runtime_identity(wheel_a)
            self.assertEqual(_assert_wheel_runtime_binding(wheel_a, runtime_a), runtime_a)
            with self.assertRaisesRegex(ValueError, "does not match supplied wheel"):
                _assert_wheel_runtime_binding(wheel_b, runtime_a)

    def test_validator_rejects_duplicate_result_rows(self) -> None:
        report = _schema_report()
        report["results"].append(copy.deepcopy(report["results"][0]))
        with self._validator_patches(report):
            with self.assertRaisesRegex(ValueError, "result rows do not match"):
                validate_report(report, assets_root=Path("unused"))

    def test_validator_rejects_dropped_candidate_case(self) -> None:
        ids = ["case-a", "case-b"]
        inputs = [{"path": "x", "sha256": "0" * 64}]
        assets = {
            profile: {
                "model_id": profile,
                "revision": "r",
                "fingerprint": "f",
                "files": [],
            }
            for profile in PROFILES
        }
        report = {
            "schema_id": SCHEMA_ID,
            "schema_version": 1,
            "contract_id": "qwen-mm-compat-v1",
            "status": "pass",
            "passed": True,
            "created_at": "2026-08-02T00:00:00+00:00",
            "scope": {
                "kind": "text_image",
                "profiles": list(PROFILES),
                "declared_case_ids": ids,
                "executed_case_ids": ids,
                "skipped_case_ids": [],
            },
            "candidate": {
                "runtime_identity": {
                    "package": "qwen_mm",
                    "version": "0.1.0",
                    "package_artifact_sha256": "1" * 64,
                    "native_module": "qwen_mm._native",
                    "native_artifact_sha256": "2" * 64,
                }
            },
            "inputs": inputs,
            "source_fingerprint": "fingerprint",
            "profile_assets": assets,
            "phase_e_exclusions": list(PHASE_E_EXCLUSIONS),
            "fixture_authentication": [
                {"candidate_executed": False, "passed": True, "issues": []} for _ in range(16)
            ],
            "results": [
                {"case_id": case_id, "candidate_executed": True, "passed": True, "issues": []}
                for case_id in ids
            ],
        }
        dropped = copy.deepcopy(report)
        dropped["scope"]["executed_case_ids"] = ["case-a"]
        with (
            patch(
                "qwen_mm_reference.phase_c_conformance.expected_candidate_case_ids",
                return_value=ids,
            ),
            patch("qwen_mm_reference.phase_c_conformance._input_records", return_value=inputs),
            patch(
                "qwen_mm_reference.phase_c_conformance._source_fingerprint",
                return_value="fingerprint",
            ),
            patch("qwen_mm_reference.phase_c_conformance._asset_records", return_value=assets),
            patch("qwen_mm_reference.phase_c_conformance._validate_report_schema"),
        ):
            with self.assertRaisesRegex(ValueError, "inventory is incomplete"):
                validate_report(dropped, assets_root=Path("unused"))


if __name__ == "__main__":
    unittest.main()
