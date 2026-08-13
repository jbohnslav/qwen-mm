from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qwen_mm_reference import phase_c_overlay_v2 as overlay
from qwen_mm_reference.benchmark_v2 import evaluate_image_release_gate
from qwen_mm_reference.phase_c_overlay_v2 import REPORT_PATH, repository_root, validate_overlay


class PhaseCOverlayV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.report = json.loads((repository_root() / REPORT_PATH).read_text(encoding="utf-8"))

    def test_committed_overlay_is_current_and_passing(self) -> None:
        validate_overlay(self.report)

    def test_current_overlay_satisfies_the_benchmark_correctness_gate(self) -> None:
        runtime = self.report["current_candidate"]["capture"]["runtime_identity"]
        eligibility = evaluate_image_release_gate(
            mode="smoke",
            selected_cases=[{"boundary": "encoded_to_numpy"}],
            candidate_identity={"resolved": True, "runtime_identity": runtime},
            phase_c_report_path=repository_root() / REPORT_PATH,
        )

        self.assertEqual(eligibility["phase_c"]["status"], "pass")
        self.assertEqual(
            eligibility["phase_c"]["evidence"]["schema_id"],
            "qwen-mm-phase-c-conformance-overlay-v2",
        )

    def test_cannot_substitute_old_report(self) -> None:
        hostile = copy.deepcopy(self.report)
        hostile["base_phase_c_v1"]["artifact"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "authentication failed"):
            validate_overlay(hostile)

    def test_cannot_expand_production_delta(self) -> None:
        hostile = copy.deepcopy(self.report)
        hostile["current_candidate"]["production_delta_from_v1"].append(
            "crates/qwen-mm-python/src/lib.rs"
        )
        with self.assertRaisesRegex(ValueError, "delta exceeds"):
            validate_overlay(hostile)

    def test_cannot_replace_selected_backend_identity(self) -> None:
        hostile = copy.deepcopy(self.report)
        hostile["current_candidate"]["selected_backend"]["filter"] = "Bilinear"
        with self.assertRaisesRegex(ValueError, "backend identity"):
            validate_overlay(hostile)

    def test_cannot_downgrade_quality_result(self) -> None:
        hostile = copy.deepcopy(self.report)
        hostile["current_candidate"]["capture"]["quality_result"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "quality result"):
            validate_overlay(hostile)

    def test_cannot_hide_oracle_import(self) -> None:
        hostile = copy.deepcopy(self.report)
        hostile["current_candidate"]["capture"]["isolation"]["forbidden_imports"] = ["PIL"]
        with self.assertRaisesRegex(ValueError, "isolation"):
            validate_overlay(hostile)

    def test_controlled_capture_reuses_production_evidence_without_rewriting_it(self) -> None:
        root = repository_root()
        blob_path = root / overlay.PRODUCTION_BLOB_PATH
        result_path = root / overlay.PRODUCTION_RESULT_PATH
        before_blob = blob_path.read_bytes()
        before_result = result_path.read_bytes()
        capture = {"case_count": 45, "quality_result": json.loads(before_result)}
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            with (
                mock.patch.object(
                    overlay, "capture_installed_wheel_resize", return_value=capture
                ) as capture_call,
                mock.patch.object(overlay, "build_overlay", return_value={"status": "pass"}),
                mock.patch.object(overlay, "_write_summary"),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "phase_c_overlay_v2",
                        "capture",
                        "--candidate-python",
                        "/venv/bin/python",
                        "--wheel",
                        str(temporary_root / "candidate.whl"),
                        "--production-blob",
                        str(blob_path),
                        "--reuse-committed-production-evidence",
                        "--report",
                        str(temporary_root / "report.json"),
                        "--summary",
                        str(temporary_root / "summary.md"),
                    ],
                ),
            ):
                overlay.main()
        self.assertFalse(capture_call.call_args.kwargs["write_candidate_blob"])
        self.assertEqual(blob_path.read_bytes(), before_blob)
        self.assertEqual(result_path.read_bytes(), before_result)


if __name__ == "__main__":
    unittest.main()
