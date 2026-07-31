from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import numpy as np
from qwen_mm_reference.fixtures import repository_root
from qwen_mm_reference.golden import describe_array, validate_manifest


def committed_manifests() -> list[Path]:
    return sorted((repository_root() / "reference" / "goldens" / "v1").rglob("manifest.json"))


def load_manifest(profile: str, case: str) -> dict:
    path = repository_root() / "reference" / "goldens" / "v1" / profile / case / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


class ArrayDescriptorTests(unittest.TestCase):
    def test_inline_descriptor_records_layout_and_complete_values(self) -> None:
        source = np.arange(12, dtype=np.int64).reshape(3, 4)[:, ::2]

        descriptor = describe_array(source, name="stride-test", inline_max_bytes=1_000)

        self.assertEqual(descriptor["shape"], [3, 2])
        self.assertEqual(descriptor["dtype"], "int64")
        self.assertEqual(descriptor["strides"], [32, 16])
        self.assertEqual(descriptor["byte_order"], "little")
        self.assertFalse(descriptor["c_contiguous"])
        self.assertEqual(descriptor["storage"], "inline_json")
        self.assertEqual(descriptor["data"], [[0, 2], [4, 6], [8, 10]])
        self.assertEqual(len(descriptor["sha256"]), 64)

    def test_large_descriptor_defaults_to_authenticated_signature(self) -> None:
        descriptor = describe_array(
            np.zeros((32,), dtype=np.float32), name="large-test", inline_max_bytes=0
        )

        self.assertEqual(descriptor["storage"], "signature_only")
        self.assertNotIn("data", descriptor)
        self.assertEqual(descriptor["nbytes"], 128)
        self.assertEqual(len(descriptor["sha256"]), 64)


class ManifestValidationTests(unittest.TestCase):
    def test_all_committed_manifests_validate(self) -> None:
        manifests = committed_manifests()
        self.assertGreaterEqual(len(manifests), 4)
        for path in manifests:
            with self.subTest(path=path):
                validate_manifest(json.loads(path.read_text(encoding="utf-8")))

    def test_missing_conditional_output_is_rejected(self) -> None:
        manifest = load_manifest("qwen3-vl-8b", "multimodal-smoke")
        broken = copy.deepcopy(manifest)
        broken["output"]["keys"].remove("mm_token_type_ids")
        del broken["output"]["arrays"]["mm_token_type_ids"]

        with self.assertRaisesRegex(ValueError, "official output keys mismatch"):
            validate_manifest(broken, verify_integrity=False)

    def test_missing_provenance_is_rejected(self) -> None:
        manifest = load_manifest("qwen3-vl-8b", "text-smoke")
        broken = copy.deepcopy(manifest)
        del broken["provenance"]["source_files"]

        with self.assertRaisesRegex(ValueError, "missing required manifest field"):
            validate_manifest(broken, verify_integrity=False)

    def test_integrity_change_is_rejected(self) -> None:
        manifest = load_manifest("qwen3-vl-8b", "text-smoke")
        broken = copy.deepcopy(manifest)
        broken["case_id"] = "tampered"

        with self.assertRaisesRegex(ValueError, "integrity mismatch"):
            validate_manifest(broken)


if __name__ == "__main__":
    unittest.main()
