from __future__ import annotations

import copy
import json
import unittest

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


if __name__ == "__main__":
    unittest.main()
