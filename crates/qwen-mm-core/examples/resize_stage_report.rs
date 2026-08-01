//! Execute the frozen B5 resize corpus and emit host-specific diagnostics.

use std::{collections::BTreeMap, env, fs, path::PathBuf, process::Command};

use fast_image_resize::{
    FilterType, ResizeAlg, ResizeOptions, Resizer,
    images::{TypedImage, TypedImageRef},
    pixels::{F32x3, U8x3},
};
use qwen_mm_core::{
    ImageOptions, ProfileAlias, ProfileRegistry, ResourceLimits, plan_image_geometry,
    resize_image_rgb8, resize_video_rgb8_to_f32,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const DEFAULT_MANIFEST: &str = "reference/resize/v1/manifest.json";
const IMPLEMENTATION_PATH: &str = "crates/qwen-mm-core/src/resize.rs";
const REPORT_SOURCE_PATH: &str = "crates/qwen-mm-core/examples/resize_stage_report.rs";
const SELECTED_CANDIDATE: &str = "qwen-mm-exact-pillow-image-and-torchvision-video-v1";
const IMAGE_TOLERANCE: f64 = 0.0;
const VIDEO_TOLERANCE: f64 = 1.0e-4;

#[derive(Deserialize)]
struct Fixture {
    contract_id: String,
    stage_id: String,
    artifacts: BTreeMap<String, ArtifactDigest>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct ArtifactDigest {
    byte_length: usize,
    sha256: String,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    tags: Vec<String>,
    source: Source,
    geometry_options: GeometryOptions,
    destination: Dimensions,
    pillow_image_rgb8: ArtifactSlice,
    torchvision_video_rgb_f32: ArtifactSlice,
}

#[derive(Deserialize)]
struct Source {
    height: u64,
    width: u64,
    stride_bytes: u64,
    offset: usize,
    byte_length: usize,
    sha256: String,
}

#[derive(Deserialize)]
struct GeometryOptions {
    min_pixels: Option<u64>,
    max_pixels: Option<u64>,
}

#[derive(Deserialize, Serialize)]
struct Dimensions {
    height: u64,
    width: u64,
}

#[derive(Deserialize)]
struct ArtifactSlice {
    offset: usize,
    byte_length: usize,
    sha256: String,
}

#[derive(Serialize)]
struct Report {
    schema_version: u8,
    contract_id: String,
    stage_id: String,
    fixture_manifest: FileDigest,
    selected_implementation: FileDigest,
    evidence_source: FileDigest,
    generated_by: String,
    host: Host,
    candidates: BTreeMap<String, CandidateSummary>,
    cases: Vec<CaseReport>,
    status: &'static str,
}

#[derive(Serialize)]
struct FileDigest {
    path: String,
    byte_length: usize,
    sha256: String,
}

#[derive(Serialize)]
struct Host {
    id: String,
    execution: String,
    os: &'static str,
    architecture: &'static str,
    uname: String,
    rustc: String,
}

#[derive(Default, Serialize)]
struct CandidateSummary {
    exact_version: String,
    selected: bool,
    image_cases_failed: usize,
    video_cases_failed: usize,
    passed: bool,
}

#[derive(Serialize)]
struct CaseReport {
    id: String,
    tags: Vec<String>,
    source: Dimensions,
    destination: Dimensions,
    candidates: BTreeMap<String, CandidateCase>,
}

#[derive(Serialize)]
struct CandidateCase {
    image: BoundaryResult,
    video: BoundaryResult,
}

#[derive(Serialize)]
struct BoundaryResult {
    tolerance: f64,
    passed: bool,
    diagnostics: Diagnostics,
}

#[derive(Serialize)]
struct Diagnostics {
    maximum_absolute_error: f64,
    rmse: f64,
    absolute_error_p50: f64,
    absolute_error_p90: f64,
    absolute_error_p99: f64,
    count_nonzero_differences: usize,
    count_over_tolerance: usize,
    maximum_ulp_distance: u64,
    first_difference: Option<FirstDifference>,
}

#[derive(Serialize)]
struct FirstDifference {
    element_index: usize,
    y: usize,
    x: usize,
    channel: usize,
    actual: f64,
    expected: f64,
}

#[allow(clippy::too_many_lines)] // The report runner intentionally keeps one linear evidence flow.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args().skip(1);
    let mut manifest_path = PathBuf::from(DEFAULT_MANIFEST);
    let mut output_path = None;
    let mut host_id = None;
    let mut execution = None;
    while let Some(argument) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value after {argument}"))?;
        match argument.as_str() {
            "--manifest" => manifest_path = PathBuf::from(value),
            "--output" => output_path = Some(PathBuf::from(value)),
            "--host-id" => host_id = Some(value),
            "--execution" => execution = Some(value),
            _ => return Err(format!("unknown argument: {argument}").into()),
        }
    }
    let output_path = output_path.ok_or("--output is required")?;
    let host_id = host_id.ok_or("--host-id is required")?;
    let execution = execution.ok_or("--execution is required")?;

    let manifest_bytes = fs::read(&manifest_path)?;
    let fixture: Fixture = serde_json::from_slice(&manifest_bytes)?;
    let fixture_directory = manifest_path.parent().ok_or("manifest has no parent")?;
    let sources = read_verified_artifact(fixture_directory, "sources.rgb8.bin", &fixture)?;
    let pillow = read_verified_artifact(fixture_directory, "pillow-image-rgb8.bin", &fixture)?;
    let torchvision =
        read_verified_artifact(fixture_directory, "torchvision-video-f32le.bin", &fixture)?;

    let registry = ProfileRegistry::bundled()?;
    let visual = &registry.get(ProfileAlias::Qwen3Vl8b).visual;
    let mut case_reports = Vec::with_capacity(fixture.cases.len());

    for case in fixture.cases {
        let source = checked_slice(
            &sources,
            case.source.offset,
            case.source.byte_length,
            &case.id,
        )?;
        verify_digest(source, &case.source.sha256, &case.id)?;
        let expected_image = checked_slice(
            &pillow,
            case.pillow_image_rgb8.offset,
            case.pillow_image_rgb8.byte_length,
            &case.id,
        )?;
        verify_digest(expected_image, &case.pillow_image_rgb8.sha256, &case.id)?;
        let expected_video_bytes = checked_slice(
            &torchvision,
            case.torchvision_video_rgb_f32.offset,
            case.torchvision_video_rgb_f32.byte_length,
            &case.id,
        )?;
        verify_digest(
            expected_video_bytes,
            &case.torchvision_video_rgb_f32.sha256,
            &case.id,
        )?;
        let expected_video = expected_video_bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
            .collect::<Vec<_>>();

        let plan = plan_image_geometry(
            visual,
            case.source.height,
            case.source.width,
            ImageOptions {
                min_pixels: case.geometry_options.min_pixels,
                max_pixels: case.geometry_options.max_pixels,
                ..ImageOptions::default()
            },
            ResourceLimits::default(),
        )?;
        if [plan.height, plan.width] != [case.destination.height, case.destination.width] {
            return Err(format!("{} destination does not match B3", case.id).into());
        }
        let destination_width = usize::try_from(case.destination.width)?;
        let packed_candidate_source = pack_rgb8(source, &case.source)?;

        let selected_image = resize_image_rgb8(
            source,
            case.source.height,
            case.source.width,
            case.source.stride_bytes,
            &plan,
        )?;
        let selected_video = resize_video_rgb8_to_f32(
            source,
            case.source.height,
            case.source.width,
            case.source.stride_bytes,
            &plan,
        )?;
        let fir_image =
            fast_image_resize_u8(&packed_candidate_source, &case.source, &case.destination)?;
        let fir_video =
            fast_image_resize_f32(&packed_candidate_source, &case.source, &case.destination)?;

        let mut candidates = BTreeMap::new();
        candidates.insert(
            SELECTED_CANDIDATE.to_owned(),
            CandidateCase {
                image: compare_u8(&selected_image, expected_image, destination_width)?,
                video: compare_f32(&selected_video, &expected_video, destination_width)?,
            },
        );
        candidates.insert(
            "fast_image_resize-6.1.0-catmull-rom".to_owned(),
            CandidateCase {
                image: compare_u8(&fir_image, expected_image, destination_width)?,
                video: compare_f32(&fir_video, &expected_video, destination_width)?,
            },
        );
        case_reports.push(CaseReport {
            id: case.id,
            tags: case.tags,
            source: Dimensions {
                height: case.source.height,
                width: case.source.width,
            },
            destination: case.destination,
            candidates,
        });
    }

    let mut candidates = BTreeMap::from([
        (
            SELECTED_CANDIDATE.to_owned(),
            CandidateSummary {
                exact_version: format!(
                    "qwen-mm-core {} source-faithful Pillow image / source-faithful TorchVision video",
                    env!("CARGO_PKG_VERSION")
                ),
                selected: true,
                ..CandidateSummary::default()
            },
        ),
        (
            "fast_image_resize-6.1.0-catmull-rom".to_owned(),
            CandidateSummary {
                exact_version: "fast_image_resize=6.1.0".to_owned(),
                selected: false,
                ..CandidateSummary::default()
            },
        ),
    ]);
    for case in &case_reports {
        for (name, result) in &case.candidates {
            let summary = candidates.get_mut(name).expect("known candidate");
            summary.image_cases_failed += usize::from(!result.image.passed);
            summary.video_cases_failed += usize::from(!result.video.passed);
        }
    }
    for summary in candidates.values_mut() {
        summary.passed = summary.image_cases_failed == 0 && summary.video_cases_failed == 0;
    }
    let selected_passed = candidates
        .values()
        .find(|candidate| candidate.selected)
        .is_some_and(|candidate| candidate.passed);

    let report = Report {
        schema_version: 1,
        contract_id: fixture.contract_id,
        stage_id: fixture.stage_id,
        fixture_manifest: FileDigest {
            path: manifest_path.display().to_string(),
            byte_length: manifest_bytes.len(),
            sha256: sha256(&manifest_bytes),
        },
        selected_implementation: file_digest(IMPLEMENTATION_PATH)?,
        evidence_source: file_digest(REPORT_SOURCE_PATH)?,
        generated_by: ("./scripts/cargo.sh run --locked -p qwen-mm-core --example ".to_owned()
            + "resize_stage_report -- --output <path> --host-id <id> --execution <kind>"),
        host: Host {
            id: host_id,
            execution,
            os: env::consts::OS,
            architecture: env::consts::ARCH,
            uname: command_output("uname", &["-a"]),
            rustc: command_output("rustc", &["-Vv"]),
        },
        candidates,
        cases: case_reports,
        status: if selected_passed { "pass" } else { "fail" },
    };
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output_path, serde_json::to_vec_pretty(&report)?)?;
    println!("wrote {} ({})", output_path.display(), report.status);
    if !selected_passed {
        return Err("selected resize implementation failed the frozen corpus".into());
    }
    Ok(())
}

fn fast_image_resize_u8(
    source: &[u8],
    dimensions: &Source,
    destination: &Dimensions,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let pixels = source
        .chunks_exact(3)
        .map(|pixel| U8x3::new([pixel[0], pixel[1], pixel[2]]))
        .collect::<Vec<_>>();
    let source_image = TypedImageRef::new(
        u32::try_from(dimensions.width)?,
        u32::try_from(dimensions.height)?,
        &pixels,
    )?;
    let mut output = TypedImage::<U8x3>::new(
        u32::try_from(destination.width)?,
        u32::try_from(destination.height)?,
    );
    run_fir(&source_image, &mut output)?;
    Ok(output.pixels().iter().flat_map(|pixel| pixel.0).collect())
}

fn fast_image_resize_f32(
    source: &[u8],
    dimensions: &Source,
    destination: &Dimensions,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let pixels = source
        .chunks_exact(3)
        .map(|pixel| {
            F32x3::new([
                f32::from(pixel[0]),
                f32::from(pixel[1]),
                f32::from(pixel[2]),
            ])
        })
        .collect::<Vec<_>>();
    let source_image = TypedImageRef::new(
        u32::try_from(dimensions.width)?,
        u32::try_from(dimensions.height)?,
        &pixels,
    )?;
    let mut output = TypedImage::<F32x3>::new(
        u32::try_from(destination.width)?,
        u32::try_from(destination.height)?,
    );
    run_fir(&source_image, &mut output)?;
    Ok(output
        .pixels()
        .iter()
        .flat_map(|pixel| {
            pixel
                .0
                .map(|value| value.clamp(0.0, 255.0).round_ties_even())
        })
        .collect())
}

fn run_fir<P: fast_image_resize::PixelTrait + Default + Copy + std::fmt::Debug>(
    source: &TypedImageRef<'_, P>,
    destination: &mut TypedImage<'_, P>,
) -> Result<(), fast_image_resize::ResizeError> {
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom));
    Resizer::new().resize_typed(source, destination, &options)
}

fn compare_u8(
    actual: &[u8],
    expected: &[u8],
    width: usize,
) -> Result<BoundaryResult, Box<dyn std::error::Error>> {
    validate_equal_nonempty_lengths(actual.len(), expected.len())?;
    let differences = actual
        .iter()
        .zip(expected)
        .map(|(&actual, &expected)| f64::from(actual.abs_diff(expected)))
        .collect::<Vec<_>>();
    let first = actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, pair)| pair.0 != pair.1)
        .map(|(index, (&actual, &expected))| first_difference(index, width, actual, expected));
    let maximum_ulp_distance = actual
        .iter()
        .zip(expected)
        .map(|(&actual, &expected)| u64::from(actual.abs_diff(expected)))
        .max()
        .unwrap_or(0);
    Ok(boundary_result(
        differences,
        IMAGE_TOLERANCE,
        maximum_ulp_distance,
        first,
    ))
}

fn compare_f32(
    actual: &[f32],
    expected: &[f32],
    width: usize,
) -> Result<BoundaryResult, Box<dyn std::error::Error>> {
    validate_equal_nonempty_lengths(actual.len(), expected.len())?;
    if let Some((index, value)) = actual
        .iter()
        .chain(expected)
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "non-finite resize comparison value at combined index {index}: {value}"
        )
        .into());
    }
    let differences = actual
        .iter()
        .zip(expected)
        .map(|(&actual, &expected)| f64::from((actual - expected).abs()))
        .collect::<Vec<_>>();
    let first = actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, pair)| pair.0.to_bits() != pair.1.to_bits())
        .map(|(index, (&actual, &expected))| first_difference(index, width, actual, expected));
    let maximum_ulp_distance = actual
        .iter()
        .zip(expected)
        .map(|(&actual, &expected)| ulp_distance(actual, expected))
        .max()
        .unwrap_or(0);
    Ok(boundary_result(
        differences,
        VIDEO_TOLERANCE,
        maximum_ulp_distance,
        first,
    ))
}

fn validate_equal_nonempty_lengths(
    actual: usize,
    expected: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if actual == 0 || actual != expected {
        return Err(format!(
            "resize comparison lengths must be equal and non-zero: actual={actual}, expected={expected}"
        )
        .into());
    }
    Ok(())
}

fn boundary_result(
    mut differences: Vec<f64>,
    tolerance: f64,
    maximum_ulp_distance: u64,
    first_difference: Option<FirstDifference>,
) -> BoundaryResult {
    let count_over_tolerance = differences
        .iter()
        .filter(|&&difference| difference > tolerance)
        .count();
    let count_nonzero_differences = differences
        .iter()
        .filter(|&&difference| difference != 0.0)
        .count();
    let maximum_absolute_error = differences.iter().copied().fold(0.0_f64, f64::max);
    let count = u32::try_from(differences.len()).expect("resize fixture element count fits u32");
    let rmse =
        (differences.iter().map(|value| value * value).sum::<f64>() / f64::from(count)).sqrt();
    differences.sort_by(f64::total_cmp);
    BoundaryResult {
        tolerance,
        passed: count_over_tolerance == 0,
        diagnostics: Diagnostics {
            maximum_absolute_error,
            rmse,
            absolute_error_p50: percentile(&differences, 50),
            absolute_error_p90: percentile(&differences, 90),
            absolute_error_p99: percentile(&differences, 99),
            count_nonzero_differences,
            count_over_tolerance,
            maximum_ulp_distance,
            first_difference,
        },
    }
}

fn percentile(sorted: &[f64], percent: usize) -> f64 {
    let index = (percent * sorted.len().saturating_sub(1) + 50) / 100;
    sorted[index]
}

fn first_difference<A: Into<f64>, E: Into<f64>>(
    index: usize,
    width: usize,
    actual: A,
    expected: E,
) -> FirstDifference {
    let pixel = index / 3;
    FirstDifference {
        element_index: index,
        y: pixel / width,
        x: pixel % width,
        channel: index % 3,
        actual: actual.into(),
        expected: expected.into(),
    }
}

fn ulp_distance(left: f32, right: f32) -> u64 {
    u64::from(left.to_bits().abs_diff(right.to_bits()))
}

fn read_verified_artifact(
    directory: &std::path::Path,
    name: &str,
    fixture: &Fixture,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let expected = fixture
        .artifacts
        .get(name)
        .ok_or_else(|| format!("manifest does not describe {name}"))?;
    let data = fs::read(directory.join(name))?;
    if data.len() != expected.byte_length || sha256(&data) != expected.sha256 {
        return Err(format!("artifact digest mismatch: {name}").into());
    }
    Ok(data)
}

fn file_digest(path: &str) -> Result<FileDigest, Box<dyn std::error::Error>> {
    let data = fs::read(path)?;
    Ok(FileDigest {
        path: path.to_owned(),
        byte_length: data.len(),
        sha256: sha256(&data),
    })
}

fn pack_rgb8(source: &[u8], dimensions: &Source) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let height = usize::try_from(dimensions.height)?;
    let width = usize::try_from(dimensions.width)?;
    let stride = usize::try_from(dimensions.stride_bytes)?;
    let row_bytes = width.checked_mul(3).ok_or("packed RGB row overflow")?;
    if stride < row_bytes {
        return Err("source stride is narrower than packed RGB".into());
    }
    let required = height
        .saturating_sub(1)
        .checked_mul(stride)
        .and_then(|offset| offset.checked_add(row_bytes))
        .ok_or("strided RGB source capacity overflow")?;
    if source.len() < required {
        return Err("strided RGB source is too short".into());
    }
    let capacity = height
        .checked_mul(row_bytes)
        .ok_or("packed RGB source capacity overflow")?;
    let mut packed = Vec::with_capacity(capacity);
    for row in 0..height {
        let start = row.checked_mul(stride).ok_or("RGB row offset overflow")?;
        packed.extend_from_slice(&source[start..start + row_bytes]);
    }
    Ok(packed)
}

fn checked_slice<'a>(
    data: &'a [u8],
    offset: usize,
    length: usize,
    case_id: &str,
) -> Result<&'a [u8], Box<dyn std::error::Error>> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| format!("{case_id}: artifact slice overflow"))?;
    data.get(offset..end)
        .ok_or_else(|| format!("{case_id}: artifact slice out of bounds").into())
}

fn verify_digest(
    data: &[u8],
    expected: &str,
    case_id: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if sha256(data) != expected {
        return Err(format!("{case_id}: per-case artifact digest mismatch").into());
    }
    Ok(())
}

fn sha256(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

fn command_output(program: &str, arguments: &[&str]) -> String {
    Command::new(program).args(arguments).output().map_or_else(
        |error| format!("unavailable: {error}"),
        |output| String::from_utf8_lossy(&output.stdout).trim().to_owned(),
    )
}

#[cfg(test)]
mod tests {
    use super::{compare_f32, compare_u8};

    #[test]
    fn comparisons_reject_prefixes_and_non_finite_values() {
        assert!(compare_u8(&[1], &[1, 2], 1).is_err());
        assert!(compare_f32(&[f32::NAN], &[0.0], 1).is_err());
        assert!(compare_f32(&[0.0], &[f32::INFINITY], 1).is_err());
    }
}
