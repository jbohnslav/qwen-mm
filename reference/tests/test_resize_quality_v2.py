from __future__ import annotations

import math

import numpy as np
from qwen_mm_reference.resize_quality_v2 import (
    DEFAULT_OUTPUT,
    _nearest_rank,
    canonical_ssim,
    channel_metrics,
    compare_candidate,
    verify_corpus,
)


def _direct_ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    coordinates = np.arange(-5, 6, dtype=np.float64)
    kernel = np.exp(
        -(coordinates[:, np.newaxis] ** 2 + coordinates[np.newaxis, :] ** 2) / (2.0 * 1.5**2)
    )
    kernel /= kernel.sum(dtype=np.float64)
    reference_padded = np.pad(reference.astype(np.float64), 5, mode="reflect")
    candidate_padded = np.pad(candidate.astype(np.float64), 5, mode="reflect")
    values = []
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    for row in range(reference.shape[0]):
        for column in range(reference.shape[1]):
            reference_window = reference_padded[row : row + 11, column : column + 11]
            candidate_window = candidate_padded[row : row + 11, column : column + 11]
            mu_reference = float(np.sum(kernel * reference_window, dtype=np.float64))
            mu_candidate = float(np.sum(kernel * candidate_window, dtype=np.float64))
            variance_reference = float(
                np.sum(kernel * (reference_window - mu_reference) ** 2, dtype=np.float64)
            )
            variance_candidate = float(
                np.sum(kernel * (candidate_window - mu_candidate) ** 2, dtype=np.float64)
            )
            covariance = float(
                np.sum(
                    kernel * (reference_window - mu_reference) * (candidate_window - mu_candidate),
                    dtype=np.float64,
                )
            )
            values.append(
                ((2.0 * mu_reference * mu_candidate + c1) * (2.0 * covariance + c2))
                / (
                    (mu_reference**2 + mu_candidate**2 + c1)
                    * (variance_reference + variance_candidate + c2)
                )
            )
    return float(np.mean(values, dtype=np.float64))


def test_canonical_ssim_matches_direct_contract_definition() -> None:
    reference = np.arange(12 * 13, dtype=np.uint8).reshape(12, 13)
    candidate = reference.copy()
    candidate[2:6, 4:9] = np.clip(candidate[2:6, 4:9].astype(np.int16) + 3, 0, 255)
    assert math.isclose(
        canonical_ssim(reference, candidate),
        _direct_ssim(reference, candidate),
        rel_tol=0.0,
        abs_tol=2e-14,
    )
    assert canonical_ssim(reference, reference) == 1.0


def test_nearest_rank_and_per_channel_gate_diagnostics() -> None:
    values = np.arange(1, 101, dtype=np.float64)
    assert _nearest_rank(values, 0.50) == 50.0
    assert _nearest_rank(values, 0.90) == 90.0
    assert _nearest_rank(values, 0.99) == 99.0

    reference = np.zeros((10, 10), dtype=np.uint8)
    candidate = reference.copy()
    candidate[0, 0] = 33
    metrics = channel_metrics(reference, candidate)
    assert metrics["maximum_absolute_error"] == 33.0
    assert metrics["absolute_error_p99"] == 0.0
    assert not metrics["gates"]["maximum_absolute_error"]
    assert not metrics["passed"]


def test_committed_holdout_authenticates_and_pillow_self_comparison_passes() -> None:
    manifest, _, _ = verify_corpus(DEFAULT_OUTPUT)
    assert manifest["holdout_seed"] == 20_260_813
    assert len(manifest["cases"]) >= 45
    assert {
        "natural_image",
        "procedural_fixture_crop",
        "checkerboard",
        "one_pixel_stripes",
        "single_channel",
        "extreme_anisotropic",
        "factor_boundary",
        "min_boundary",
        "max_boundary",
    }.issubset(manifest["coverage_tags"])
    assert manifest["natural_image_provenance"]["credit"] == ("NASA Earth Observatory, Blue Marble")
    natural_cases = [case for case in manifest["cases"] if "natural_image" in case["tags"]]
    assert len(natural_cases) == 4
    assert all(case["pattern"] == "nasa_blue_marble_natural_crop" for case in natural_cases)
    assert all(
        "natural_image" not in case["tags"]
        for case in manifest["cases"]
        if case["pattern"] == "decoded_procedural_fixture_crop"
    )

    report = compare_candidate(DEFAULT_OUTPUT / "pillow-image-rgb8.bin", DEFAULT_OUTPUT)
    assert report["passed"]
    assert all(case["passed"] for case in report["cases"])
    assert all(item["passed"] for item in report["cross_case_exact_invariants"])
    assert all(
        channel["psnr"] == "Infinity" for case in report["cases"] for channel in case["channels"]
    )
