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
