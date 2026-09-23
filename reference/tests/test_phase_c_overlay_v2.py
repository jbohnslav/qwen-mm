from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qwen_mm_reference import phase_c_overlay_v2 as overlay
from qwen_mm_reference import resize_quality_v2
from qwen_mm_reference.benchmark_v2 import evaluate_image_release_gate
from qwen_mm_reference.phase_c_overlay_v2 import REPORT_PATH, repository_root, validate_overlay


class PhaseCOverlayV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.report_path = repository_root() / REPORT_PATH
        self.report = json.loads(self.report_path.read_text(encoding="utf-8"))
        # This report certifies its recorded resize-only revision, not today's
        # tokenizer/dependency tree. Materialize its source records while keeping
        # the real Git history and hash-authenticated archived evidence available.
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.fixture_root = Path(temporary.name)
        root = repository_root()
        for name in (".git", "reference", "docs", "fixtures"):
            (self.fixture_root / name).symlink_to(root / name)
        current = self.report["current_candidate"]
        for key in ("backend_source", "dependency_lock"):
            record = current[key]
            path = self.fixture_root / record["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(
                subprocess.check_output(
                    ["git", "show", f"{current['revision']}:{record['path']}"],
                    cwd=root,
                )
            )
        for patcher in (
            mock.patch.object(overlay, "repository_root", return_value=self.fixture_root),
            mock.patch.object(resize_quality_v2, "ROOT", self.fixture_root),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def validate(self, report: dict[str, object]) -> None:
        validate_overlay(report, evidence_root=self.report_path.parent.resolve())

    def test_archived_overlay_is_authentic_and_passing(self) -> None:
        self.validate(self.report)

    def test_archived_overlay_satisfies_the_gate_for_its_recorded_runtime(self) -> None:
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

    def test_archived_overlay_rejects_different_dependency_lock(self) -> None:
        path = self.fixture_root / self.report["current_candidate"]["dependency_lock"]["path"]
        path.write_bytes(path.read_bytes() + b"\n# changed dependency tree\n")
        with self.assertRaisesRegex(ValueError, "dependency lock authentication failed"):
            self.validate(self.report)

    def test_cannot_substitute_old_report(self) -> None:
        hostile = copy.deepcopy(self.report)
        hostile["base_phase_c_v1"]["artifact"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "authentication failed"):
            self.validate(hostile)

    def test_cannot_expand_production_delta(self) -> None:
        hostile = copy.deepcopy(self.report)
        hostile["current_candidate"]["production_delta_from_v1"].append(
            "crates/qwen-mm-python/src/lib.rs"
        )
        with self.assertRaisesRegex(ValueError, "delta exceeds"):
            self.validate(hostile)

    def test_cannot_replace_selected_backend_identity(self) -> None:
        hostile = copy.deepcopy(self.report)
        hostile["current_candidate"]["selected_backend"]["filter"] = "Bilinear"
        with self.assertRaisesRegex(ValueError, "backend identity"):
            self.validate(hostile)

    def test_cannot_downgrade_quality_result(self) -> None:
        hostile = copy.deepcopy(self.report)
        hostile["current_candidate"]["capture"]["quality_result"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "quality result"):
            self.validate(hostile)

    def test_cannot_hide_oracle_import(self) -> None:
        hostile = copy.deepcopy(self.report)
        hostile["current_candidate"]["capture"]["isolation"]["forbidden_imports"] = ["PIL"]
        with self.assertRaisesRegex(ValueError, "execution binding"):
            self.validate(hostile)

    def test_architecture_local_public_outputs_replace_only_their_frozen_slices(self) -> None:
        cases = [
            {"id": "direct", "pillow_image_rgb8": {"offset": 0, "byte_length": 3}},
            {"id": "public", "pillow_image_rgb8": {"offset": 3, "byte_length": 3}},
        ]
        merged = overlay._merge_public_processor_outputs(b"abcdef", [(cases[1], b"XYZ")])
        self.assertEqual(merged, b"abcXYZ")
        with self.assertRaisesRegex(RuntimeError, "RGB length differs"):
            overlay._merge_public_processor_outputs(b"abcdef", [(cases[1], b"too-long")])

    def test_quality_recomputation_allows_only_float64_roundoff(self) -> None:
        recorded = {"passed": True, "metric": 0.98, "gates": {"ssim": True}}
        overlay._assert_quality_result_equivalent(
            recorded,
            {"passed": True, "metric": 0.98 + 5e-13, "gates": {"ssim": True}},
            label="quality",
        )
        with self.assertRaisesRegex(ValueError, "numeric value differs"):
            overlay._assert_quality_result_equivalent(
                recorded,
                {"passed": True, "metric": 0.98 + 1e-8, "gates": {"ssim": True}},
                label="quality",
            )
        with self.assertRaisesRegex(ValueError, "value differs"):
            overlay._assert_quality_result_equivalent(
                recorded,
                {"passed": False, "metric": 0.98, "gates": {"ssim": True}},
                label="quality",
            )

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
