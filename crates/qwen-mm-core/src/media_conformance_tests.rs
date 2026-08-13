use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{
    ErrorCategory, ImageFormat, ImageInput, ImageOptions, ProfileAlias, ProfileRegistry,
    ResourceLimits, prepare_image_rgb8,
};

const MANIFEST: &str = include_str!("../../../reference/media/v1/manifest.json");
const ENCODED: &[u8] = include_bytes!("../../../reference/media/v1/encoded.bin");
const PREPARED_RGB8: &[u8] = include_bytes!("../../../reference/media/v1/prepared-rgb8.bin");
const GENERATOR: &[u8] =
    include_bytes!("../../../reference/src/qwen_mm_reference/media_conformance.py");
const COMPATIBILITY: &[u8] = include_bytes!("../../../reference/compatibility/v1.json");

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    contract_id: String,
    stage: String,
    artifacts: BTreeMap<String, ArtifactDigest>,
    provenance: Provenance,
    integrity: Integrity,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Provenance {
    generator: ProvenanceFile,
    compatibility_manifest: ProvenanceFile,
}

#[derive(Deserialize)]
struct ProvenanceFile {
    path: String,
    sha256: String,
}

#[derive(Deserialize)]
struct Integrity {
    algorithm: String,
    canonical_json_without_integrity_sha256: String,
}

#[derive(Deserialize)]
struct ArtifactDigest {
    byte_length: usize,
    sha256: String,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    format: String,
    tags: Vec<String>,
    options: Options,
    encoded: Artifact,
    prepared_rgb8: Option<PreparedArtifact>,
    expected_error: Option<ExpectedError>,
}

#[derive(Clone, Copy, Default, Deserialize)]
struct Options {
    min_pixels: Option<u64>,
    max_pixels: Option<u64>,
}

#[derive(Deserialize)]
struct Artifact {
    offset: usize,
    byte_length: usize,
    sha256: String,
}

#[derive(Deserialize)]
struct PreparedArtifact {
    height: u64,
    width: u64,
    absolute_byte_error_max: u8,
    offset: usize,
    byte_length: usize,
    sha256: String,
}

#[derive(Deserialize)]
struct ExpectedError {
    category: String,
}

#[test]
#[allow(clippy::too_many_lines)] // One loop authenticates and exercises the entire media corpus.
fn prepared_rgb_stage_matches_pinned_pillow_oracle() {
    let fixture: Fixture = serde_json::from_str(MANIFEST).expect("media fixture manifest");
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.contract_id, "qwen-mm-compat-v1");
    assert_eq!(fixture.stage, "prepared_hwc_rgb8");
    assert_eq!(fixture.integrity.algorithm, "sha256");
    let mut canonical: Value = serde_json::from_str(MANIFEST).expect("media manifest JSON");
    canonical
        .as_object_mut()
        .expect("media manifest object")
        .remove("integrity")
        .expect("media manifest integrity");
    let canonical = serde_json::to_vec(&canonical).expect("canonical media manifest");
    assert_eq!(
        sha256(&canonical),
        fixture.integrity.canonical_json_without_integrity_sha256,
        "media manifest canonical integrity"
    );
    assert_eq!(
        fixture.provenance.generator.path,
        "reference/src/qwen_mm_reference/media_conformance.py"
    );
    assert_eq!(
        sha256(GENERATOR),
        fixture.provenance.generator.sha256,
        "media generator provenance"
    );
    assert_eq!(
        fixture.provenance.compatibility_manifest.path,
        "reference/compatibility/v1.json"
    );
    assert_eq!(
        sha256(COMPATIBILITY),
        fixture.provenance.compatibility_manifest.sha256,
        "compatibility manifest provenance"
    );
    verify_blob(ENCODED, &fixture.artifacts["encoded.bin"], "encoded.bin");
    verify_blob(
        PREPARED_RGB8,
        &fixture.artifacts["prepared-rgb8.bin"],
        "prepared-rgb8.bin",
    );
    let registry = ProfileRegistry::bundled().expect("bundled profiles");
    for alias in [ProfileAlias::Qwen3Vl8b, ProfileAlias::Qwen35_9b] {
        let visual = &registry.get(alias).visual;
        for case in &fixture.cases {
            let encoded = artifact_slice(ENCODED, &case.encoded, &case.id);
            let format = match case.format.as_str() {
                "jpeg" => ImageFormat::Jpeg,
                "png" => ImageFormat::Png,
                "webp" => ImageFormat::WebP,
                other => panic!("unknown format {other} in {}", case.id),
            };
            let result = prepare_image_rgb8(
                ImageInput::Encoded {
                    data: encoded,
                    format,
                },
                visual,
                ImageOptions {
                    min_pixels: case.options.min_pixels,
                    max_pixels: case.options.max_pixels,
                    resized_height: None,
                    resized_width: None,
                },
                ResourceLimits::default(),
            );

            if let Some(expected_error) = &case.expected_error {
                let Err(actual) = result else {
                    panic!("{} unexpectedly prepared successfully", case.id)
                };
                assert_eq!(
                    actual.category(),
                    category(&expected_error.category),
                    "{} returned {actual}",
                    case.id
                );
                continue;
            }

            let expected_artifact = case
                .prepared_rgb8
                .as_ref()
                .unwrap_or_else(|| panic!("{} has no expected output", case.id));
            if case.tags.iter().any(|tag| tag == "lossless") {
                assert_eq!(
                    expected_artifact.absolute_byte_error_max, 0,
                    "{} lossless fixture must remain an exact gate",
                    case.id
                );
            }
            if case.id == "png-luma-off-grid-resize" {
                assert_eq!(
                    (expected_artifact.height, expected_artifact.width),
                    (64, 96),
                    "65x97 grayscale regression must retain its off-grid destination"
                );
                assert_eq!(
                    expected_artifact.absolute_byte_error_max, 0,
                    "65x97 grayscale regression must reject the former dynamic-i16 gap"
                );
            }
            let expected = artifact_slice(PREPARED_RGB8, expected_artifact, &case.id);
            let actual = result.unwrap_or_else(|error| panic!("{} returned {error}", case.id));
            assert_eq!(
                actual.geometry.height, expected_artifact.height,
                "{} height",
                case.id
            );
            assert_eq!(
                actual.geometry.width, expected_artifact.width,
                "{} width",
                case.id
            );
            assert_eq!(actual.rgb.len(), expected.len(), "{} byte length", case.id);
            let diagnostics = compare(&actual.rgb, expected, expected_artifact.width);
            eprintln!(
                "profile={} media-case={} max_abs={} rmse={:.10} p50={} p90={} p99={} nonzero={} over_bound={} first_difference={:?}",
                alias.as_str(),
                case.id,
                diagnostics.maximum,
                diagnostics.rmse,
                diagnostics.p50,
                diagnostics.p90,
                diagnostics.p99,
                diagnostics.nonzero,
                diagnostics.over_bound(expected_artifact.absolute_byte_error_max),
                diagnostics.first
            );
            if case.tags.iter().any(|tag| tag == "resize") {
                assert_v2_resize_quality(
                    &actual.rgb,
                    expected,
                    expected_artifact.height,
                    expected_artifact.width,
                    &case.id,
                );
            } else {
                assert!(
                    diagnostics.maximum <= expected_artifact.absolute_byte_error_max,
                    "{} exceeded byte bound {}: max={}, rmse={}, p99={}, count_over={}, first={:?}",
                    case.id,
                    expected_artifact.absolute_byte_error_max,
                    diagnostics.maximum,
                    diagnostics.rmse,
                    diagnostics.p99,
                    diagnostics.over_bound(expected_artifact.absolute_byte_error_max),
                    diagnostics.first
                );
            }
        }
    }
}

trait ArtifactLike {
    fn offset(&self) -> usize;
    fn byte_length(&self) -> usize;
    fn sha256(&self) -> &str;
}

impl ArtifactLike for Artifact {
    fn offset(&self) -> usize {
        self.offset
    }

    fn byte_length(&self) -> usize {
        self.byte_length
    }

    fn sha256(&self) -> &str {
        &self.sha256
    }
}

impl ArtifactLike for PreparedArtifact {
    fn offset(&self) -> usize {
        self.offset
    }

    fn byte_length(&self) -> usize {
        self.byte_length
    }

    fn sha256(&self) -> &str {
        &self.sha256
    }
}

fn artifact_slice<'a>(blob: &'a [u8], artifact: &impl ArtifactLike, case_id: &str) -> &'a [u8] {
    let end = artifact
        .offset()
        .checked_add(artifact.byte_length())
        .unwrap_or_else(|| panic!("{case_id} artifact extent overflow"));
    let data = blob
        .get(artifact.offset()..end)
        .unwrap_or_else(|| panic!("{case_id} artifact is out of bounds"));
    assert_eq!(
        sha256(data),
        artifact.sha256(),
        "{case_id} artifact SHA-256"
    );
    data
}

#[derive(Debug)]
struct Diagnostics {
    maximum: u8,
    differences: Vec<u8>,
    rmse: f64,
    p50: u8,
    p90: u8,
    p99: u8,
    nonzero: usize,
    first: Option<(u64, u64, u64, u8, u8)>,
}

impl Diagnostics {
    fn over_bound(&self, bound: u8) -> usize {
        self.differences
            .iter()
            .filter(|&&difference| difference > bound)
            .count()
    }
}

#[allow(clippy::cast_precision_loss)] // The configured prepared-image cap keeps this sum exact in f64.
fn compare(actual: &[u8], expected: &[u8], width: u64) -> Diagnostics {
    let mut maximum = 0_u8;
    let mut squared_error = 0_u64;
    let mut nonzero = 0_usize;
    let mut differences = Vec::with_capacity(actual.len());
    let mut first = None;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let difference = actual.abs_diff(expected);
        maximum = maximum.max(difference);
        squared_error += u64::from(difference).pow(2);
        nonzero += usize::from(difference != 0);
        differences.push(difference);
        if difference != 0 && first.is_none() {
            let pixel = u64::try_from(index / 3).expect("index fits u64");
            first = Some((
                pixel / width,
                pixel % width,
                u64::try_from(index % 3).expect("channel fits u64"),
                expected,
                actual,
            ));
        }
    }
    let rmse = (squared_error as f64 / actual.len() as f64).sqrt();
    let mut sorted = differences.clone();
    sorted.sort_unstable();
    Diagnostics {
        maximum,
        differences,
        rmse,
        p50: percentile(&sorted, 50),
        p90: percentile(&sorted, 90),
        p99: percentile(&sorted, 99),
        nonzero,
        first,
    }
}

fn percentile(sorted: &[u8], percentile: usize) -> u8 {
    sorted[((sorted.len() - 1) * percentile + 50) / 100]
}

#[allow(clippy::cast_precision_loss)] // The prepared-image resource cap keeps exact integer sums in f64.
fn assert_v2_resize_quality(
    actual: &[u8],
    expected: &[u8],
    height: u64,
    width: u64,
    case_id: &str,
) {
    let height = usize::try_from(height).expect("height fits usize");
    let width = usize::try_from(width).expect("width fits usize");
    let pixels = height.checked_mul(width).expect("pixel count");
    assert_eq!(actual.len(), pixels * 3, "{case_id} RGB extent");
    for channel in 0..3 {
        let mut absolute_errors = Vec::with_capacity(pixels);
        let mut squared_error = 0.0_f64;
        let mut signed_error = 0.0_f64;
        for pixel in 0..pixels {
            let index = pixel * 3 + channel;
            let error = f64::from(actual[index]) - f64::from(expected[index]);
            absolute_errors.push(actual[index].abs_diff(expected[index]));
            squared_error = error.mul_add(error, squared_error);
            signed_error += error;
        }
        absolute_errors.sort_unstable();
        let maximum = absolute_errors.last().copied().unwrap_or(0);
        let rmse = (squared_error / pixels as f64).sqrt();
        let p99_rank = (99 * pixels).div_ceil(100).max(1);
        let p99 = absolute_errors[p99_rank - 1];
        let bias = (signed_error / pixels as f64).abs();
        let ssim = channel_ssim(actual, expected, height, width, channel);
        assert!(
            maximum <= 32 && rmse <= 5.0 && p99 <= 16 && bias <= 2.0 && ssim >= 0.98,
            "{case_id} channel {channel} failed resize-v2: max={maximum}, rmse={rmse}, p99={p99}, abs_bias={bias}, ssim={ssim}"
        );
    }
}

#[allow(clippy::cast_precision_loss)] // Pixel coordinates are bounded by the prepared-image cap.
fn channel_ssim(
    actual: &[u8],
    expected: &[u8],
    height: usize,
    width: usize,
    channel: usize,
) -> f64 {
    const RADIUS: isize = 5;
    const C1: f64 = 6.5025;
    const C2: f64 = 58.5225;
    let mut weights = Vec::with_capacity(121);
    let mut weight_sum = 0.0_f64;
    for y in -RADIUS..=RADIUS {
        for x in -RADIUS..=RADIUS {
            let squared_radius = (x * x + y * y) as f64;
            let weight = (-squared_radius / (2.0 * 1.5_f64.powi(2))).exp();
            weights.push(weight);
            weight_sum += weight;
        }
    }
    for weight in &mut weights {
        *weight /= weight_sum;
    }

    let mut total = 0.0_f64;
    for output_y in 0..height {
        for output_x in 0..width {
            let mut expected_mean = 0.0_f64;
            let mut actual_mean = 0.0_f64;
            for (weight_index, (offset_y, offset_x)) in (-RADIUS..=RADIUS)
                .flat_map(|y| (-RADIUS..=RADIUS).map(move |x| (y, x)))
                .enumerate()
            {
                let source_y = reflect_101(
                    isize::try_from(output_y).expect("output y") + offset_y,
                    height,
                );
                let source_x = reflect_101(
                    isize::try_from(output_x).expect("output x") + offset_x,
                    width,
                );
                let index = (source_y * width + source_x) * 3 + channel;
                expected_mean += weights[weight_index] * f64::from(expected[index]);
                actual_mean += weights[weight_index] * f64::from(actual[index]);
            }
            let mut expected_variance = 0.0_f64;
            let mut actual_variance = 0.0_f64;
            let mut covariance = 0.0_f64;
            for (weight_index, (offset_y, offset_x)) in (-RADIUS..=RADIUS)
                .flat_map(|y| (-RADIUS..=RADIUS).map(move |x| (y, x)))
                .enumerate()
            {
                let source_y = reflect_101(
                    isize::try_from(output_y).expect("output y") + offset_y,
                    height,
                );
                let source_x = reflect_101(
                    isize::try_from(output_x).expect("output x") + offset_x,
                    width,
                );
                let index = (source_y * width + source_x) * 3 + channel;
                let expected_delta = f64::from(expected[index]) - expected_mean;
                let actual_delta = f64::from(actual[index]) - actual_mean;
                expected_variance += weights[weight_index] * expected_delta * expected_delta;
                actual_variance += weights[weight_index] * actual_delta * actual_delta;
                covariance += weights[weight_index] * expected_delta * actual_delta;
            }
            let numerator = (2.0 * expected_mean * actual_mean + C1) * (2.0 * covariance + C2);
            let denominator = (expected_mean.powi(2) + actual_mean.powi(2) + C1)
                * (expected_variance + actual_variance + C2);
            total += numerator / denominator;
        }
    }
    total / (height * width) as f64
}

fn reflect_101(index: isize, length: usize) -> usize {
    debug_assert!(length > 5);
    if index < 0 {
        usize::try_from(-index).expect("reflected index")
    } else if usize::try_from(index).expect("nonnegative index") >= length {
        usize::try_from(2 * isize::try_from(length).expect("length") - index - 2)
            .expect("reflected index")
    } else {
        usize::try_from(index).expect("in-bounds index")
    }
}

fn verify_blob(data: &[u8], expected: &ArtifactDigest, label: &str) {
    assert_eq!(data.len(), expected.byte_length, "{label} byte length");
    assert_eq!(sha256(data), expected.sha256, "{label} SHA-256");
}

fn sha256(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

fn category(value: &str) -> ErrorCategory {
    match value {
        "unsupported_media" => ErrorCategory::UnsupportedMedia,
        "media_decode" => ErrorCategory::MediaDecode,
        "media_geometry" => ErrorCategory::MediaGeometry,
        "resource_limit" => ErrorCategory::ResourceLimit,
        "arithmetic_overflow" => ErrorCategory::ArithmeticOverflow,
        other => panic!("unknown expected category: {other}"),
    }
}
