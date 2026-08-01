//! Execute the frozen B6 media corpus and emit host-specific diagnostics.

use std::{collections::BTreeMap, env, fs, path::PathBuf, process::Command};

use qwen_mm_core::{
    ErrorCategory, ImageFormat, ImageInput, ImageOptions, ProfileAlias, ProfileRegistry,
    ResourceLimits, Rgb8, VisualProfile, prepare_image_rgb8,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

const DEFAULT_MANIFEST: &str = "reference/media/v1/manifest.json";
const MEDIA_IMPLEMENTATION: &str = "crates/qwen-mm-core/src/media.rs";
const RESIZE_IMPLEMENTATION: &str = "crates/qwen-mm-core/src/resize.rs";
const REPORT_SOURCE: &str = "crates/qwen-mm-core/examples/media_stage_report.rs";

#[derive(Deserialize)]
struct Fixture {
    contract_id: String,
    stage: String,
    oracle: Value,
    provenance: Value,
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
    format: String,
    tags: Vec<String>,
    options: Options,
    encoded: ArtifactSlice,
    prepared_rgb8: Option<PreparedSlice>,
    expected_error: Option<ExpectedError>,
}

#[derive(Clone, Copy, Default, Deserialize)]
struct Options {
    min_pixels: Option<u64>,
    max_pixels: Option<u64>,
}

#[derive(Deserialize)]
struct ArtifactSlice {
    offset: usize,
    byte_length: usize,
    sha256: String,
}

#[derive(Deserialize)]
struct PreparedSlice {
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

#[derive(Serialize)]
struct Report {
    schema_version: u8,
    contract_id: String,
    stage: String,
    fixture_manifest: FileDigest,
    input_artifacts: BTreeMap<String, FileDigest>,
    build_inputs: BTreeMap<String, FileDigest>,
    implementations: BTreeMap<String, FileDigest>,
    evidence_source: FileDigest,
    generated_by: String,
    oracle: Value,
    provenance: Value,
    rust_codec_stack: BTreeMap<&'static str, &'static str>,
    host: Host,
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

#[derive(Serialize)]
struct CaseReport {
    profile: &'static str,
    id: String,
    tags: Vec<String>,
    expected: String,
    actual: String,
    passed: bool,
    source: Option<Dimensions>,
    destination: Option<Dimensions>,
    tolerance: Option<u8>,
    diagnostics: Option<Diagnostics>,
}

#[derive(Serialize)]
struct Dimensions {
    height: u64,
    width: u64,
}

#[derive(Serialize)]
struct Diagnostics {
    maximum_absolute_error: u8,
    rmse: f64,
    absolute_error_p50: u8,
    absolute_error_p90: u8,
    absolute_error_p99: u8,
    count_nonzero_differences: usize,
    count_over_tolerance: usize,
    maximum_ulp_distance: u8,
    first_difference: Option<FirstDifference>,
}

#[derive(Serialize)]
struct FirstDifference {
    element_index: usize,
    y: usize,
    x: usize,
    channel: usize,
    actual: u8,
    expected: u8,
}

#[allow(clippy::too_many_lines)] // One linear flow keeps the evidence runner auditable.
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
    let encoded = read_verified_artifact(fixture_directory, "encoded.bin", &fixture)?;
    let expected_rgb = read_verified_artifact(fixture_directory, "prepared-rgb8.bin", &fixture)?;
    let registry = ProfileRegistry::bundled()?;
    let mut reports = Vec::with_capacity(fixture.cases.len() * 2);

    for alias in [ProfileAlias::Qwen3Vl8b, ProfileAlias::Qwen35_9b] {
        let visual = &registry.get(alias).visual;
        let profile = alias.as_str();
        for case in &fixture.cases {
            let input = verified_slice(&encoded, &case.encoded, &case.id)?;
            let result = prepare_image_rgb8(
                ImageInput::Encoded {
                    data: input,
                    format: image_format(&case.format)?,
                },
                visual,
                ImageOptions {
                    min_pixels: case.options.min_pixels,
                    max_pixels: case.options.max_pixels,
                    ..ImageOptions::default()
                },
                ResourceLimits::default(),
            );
            if let Some(expected_error) = &case.expected_error {
                let (actual, passed) = match result {
                    Ok(_) => ("success".to_owned(), false),
                    Err(error) => {
                        let actual = category_name(error.category()).to_owned();
                        let passed = actual == expected_error.category;
                        (actual, passed)
                    }
                };
                reports.push(CaseReport {
                    profile,
                    id: case.id.clone(),
                    tags: case.tags.clone(),
                    expected: expected_error.category.clone(),
                    actual,
                    passed,
                    source: None,
                    destination: None,
                    tolerance: None,
                    diagnostics: None,
                });
                continue;
            }

            let expected = case
                .prepared_rgb8
                .as_ref()
                .ok_or_else(|| format!("{} has no expected outcome", case.id))?;
            if case.tags.iter().any(|tag| tag == "lossless")
                && expected.absolute_byte_error_max != 0
            {
                return Err(format!("{} weakens the exact lossless gate", case.id).into());
            }
            let oracle = verified_slice(&expected_rgb, expected, &case.id)?;
            let (actual, source, destination, diagnostics, passed) = match result {
                Ok(image) => {
                    let diagnostics = compare_rgb8(
                        &image.rgb,
                        oracle,
                        usize::try_from(expected.width)?,
                        expected.absolute_byte_error_max,
                    )?;
                    let dimensions_match = image.geometry.height == expected.height
                        && image.geometry.width == expected.width;
                    let passed = dimensions_match && diagnostics.count_over_tolerance == 0;
                    (
                        "prepared_rgb8".to_owned(),
                        Some(Dimensions {
                            height: image.source_height,
                            width: image.source_width,
                        }),
                        Some(Dimensions {
                            height: image.geometry.height,
                            width: image.geometry.width,
                        }),
                        Some(diagnostics),
                        passed,
                    )
                }
                Err(error) => (
                    category_name(error.category()).to_owned(),
                    None,
                    None,
                    None,
                    false,
                ),
            };
            reports.push(CaseReport {
                profile,
                id: case.id.clone(),
                tags: case.tags.clone(),
                expected: "prepared_rgb8".to_owned(),
                actual,
                passed,
                source,
                destination,
                tolerance: Some(expected.absolute_byte_error_max),
                diagnostics,
            });
        }

        let raw_no_op = rgb_pattern(64, 96, 1);
        reports.push(execute_raw_case(
            profile,
            "raw-packed-aligned-no-op",
            &["raw", "packed", "no_op", "lossless"],
            &raw_no_op,
            64,
            96,
            96 * 3,
            visual,
            &raw_no_op,
            64,
            96,
        )?);
        let (raw_no_op_padded, no_op_stride) = padded_exact_extent(&raw_no_op, 64, 96, 7)?;
        reports.push(execute_raw_case(
            profile,
            "raw-padded-exact-last-row-no-op",
            &[
                "raw",
                "padded_stride",
                "exact_last_row",
                "no_op",
                "lossless",
            ],
            &raw_no_op_padded,
            64,
            96,
            no_op_stride,
            visual,
            &raw_no_op,
            64,
            96,
        )?);

        let off_grid_case = fixture
            .cases
            .iter()
            .find(|case| case.id == "png-luma-off-grid-resize")
            .ok_or("missing raw off-grid oracle case")?;
        let off_grid_record = off_grid_case
            .prepared_rgb8
            .as_ref()
            .ok_or("raw off-grid oracle has no prepared output")?;
        let off_grid_oracle =
            verified_slice(&expected_rgb, off_grid_record, "raw-luma-off-grid-oracle")?;
        let raw_off_grid = luma_rgb_pattern(65, 97);
        reports.push(execute_raw_case(
            profile,
            "raw-packed-luma-off-grid-resize",
            &["raw", "packed", "off_grid", "resize", "lossless"],
            &raw_off_grid,
            65,
            97,
            97 * 3,
            visual,
            off_grid_oracle,
            off_grid_record.height,
            off_grid_record.width,
        )?);
        let (raw_off_grid_padded, off_grid_stride) = padded_exact_extent(&raw_off_grid, 65, 97, 5)?;
        reports.push(execute_raw_case(
            profile,
            "raw-padded-luma-off-grid-resize",
            &[
                "raw",
                "padded_stride",
                "exact_last_row",
                "off_grid",
                "resize",
                "lossless",
            ],
            &raw_off_grid_padded,
            65,
            97,
            off_grid_stride,
            visual,
            off_grid_oracle,
            off_grid_record.height,
            off_grid_record.width,
        )?);
    }

    let passed = reports.iter().all(|case| case.passed);
    let mut input_artifacts = BTreeMap::new();
    for name in ["encoded.bin", "prepared-rgb8.bin"] {
        input_artifacts.insert(name.to_owned(), file_digest(fixture_directory.join(name))?);
    }
    let report = Report {
        schema_version: 1,
        contract_id: fixture.contract_id,
        stage: fixture.stage,
        fixture_manifest: digest(&manifest_path, &manifest_bytes),
        input_artifacts,
        build_inputs: BTreeMap::from([
            ("workspace_manifest".to_owned(), file_digest("Cargo.toml")?),
            (
                "core_manifest".to_owned(),
                file_digest("crates/qwen-mm-core/Cargo.toml")?,
            ),
            ("cargo_lock".to_owned(), file_digest("Cargo.lock")?),
        ]),
        implementations: BTreeMap::from([
            (
                "decode_color".to_owned(),
                file_digest(MEDIA_IMPLEMENTATION)?,
            ),
            ("resize".to_owned(), file_digest(RESIZE_IMPLEMENTATION)?),
        ]),
        evidence_source: file_digest(REPORT_SOURCE)?,
        generated_by: ("./scripts/cargo.sh run --locked -p qwen-mm-core --example ".to_owned()
            + "media_stage_report -- --output <path> --host-id <id> --execution <kind>"),
        oracle: fixture.oracle,
        provenance: fixture.provenance,
        rust_codec_stack: BTreeMap::from([
            ("image", "0.25.10 (default-features=false; png,webp)"),
            ("image-webp", "0.2.4"),
            (
                "libjpeg-turbo-rs",
                "0.8.0 (default-features=false; simd,std)",
            ),
            ("png", "0.18.1"),
        ]),
        host: Host {
            id: host_id,
            execution,
            os: env::consts::OS,
            architecture: env::consts::ARCH,
            uname: command_output("uname", &["-a"]),
            rustc: command_output("rustc", &["-Vv"]),
        },
        cases: reports,
        status: if passed { "pass" } else { "fail" },
    };
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output_path, serde_json::to_vec_pretty(&report)?)?;
    println!("wrote {} ({})", output_path.display(), report.status);
    if !passed {
        return Err("media implementation failed the frozen corpus".into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)] // The explicit shape and stride are the raw-view contract.
fn execute_raw_case(
    profile: &'static str,
    id: &str,
    tags: &[&str],
    input: &[u8],
    source_height: usize,
    source_width: usize,
    stride: usize,
    visual: &VisualProfile,
    expected: &[u8],
    expected_height: u64,
    expected_width: u64,
) -> Result<CaseReport, Box<dyn std::error::Error>> {
    let result = prepare_image_rgb8(
        ImageInput::Rgb8(Rgb8 {
            data: input,
            height: source_height,
            width: source_width,
            row_stride: stride,
        }),
        visual,
        ImageOptions::default(),
        ResourceLimits::default(),
    );
    let (actual, destination, diagnostics, passed) = match result {
        Ok(image) => {
            let diagnostics =
                compare_rgb8(&image.rgb, expected, usize::try_from(expected_width)?, 0)?;
            let dimensions_match =
                image.geometry.height == expected_height && image.geometry.width == expected_width;
            (
                "prepared_rgb8".to_owned(),
                Some(Dimensions {
                    height: image.geometry.height,
                    width: image.geometry.width,
                }),
                Some(diagnostics),
                dimensions_match,
            )
        }
        Err(error) => (
            category_name(error.category()).to_owned(),
            None,
            None,
            false,
        ),
    };
    let passed = passed
        && diagnostics
            .as_ref()
            .is_some_and(|diagnostics| diagnostics.count_over_tolerance == 0);
    Ok(CaseReport {
        profile,
        id: id.to_owned(),
        tags: tags.iter().map(|tag| (*tag).to_owned()).collect(),
        expected: "prepared_rgb8".to_owned(),
        actual,
        passed,
        source: Some(Dimensions {
            height: u64::try_from(source_height)?,
            width: u64::try_from(source_width)?,
        }),
        destination,
        tolerance: Some(0),
        diagnostics,
    })
}

fn rgb_pattern(height: usize, width: usize, seed: usize) -> Vec<u8> {
    let mut output = Vec::with_capacity(height * width * 3);
    for y in 0..height {
        for x in 0..width {
            output.extend([
                u8::try_from((x * 37 + y * 11 + seed) % 256).expect("bounded pattern"),
                u8::try_from((x * 7 + y * 53 + seed * 3) % 256).expect("bounded pattern"),
                u8::try_from((x * 19 + y * 23 + seed * 5) % 256).expect("bounded pattern"),
            ]);
        }
    }
    output
}

fn luma_rgb_pattern(height: usize, width: usize) -> Vec<u8> {
    let mut output = Vec::with_capacity(height * width * 3);
    for y in 0..height {
        for x in 0..width {
            let value = u8::try_from((x * 29 + y * 47 + 13) % 256).expect("bounded pattern");
            output.extend([value; 3]);
        }
    }
    output
}

fn padded_exact_extent(
    packed: &[u8],
    height: usize,
    width: usize,
    padding: usize,
) -> Result<(Vec<u8>, usize), Box<dyn std::error::Error>> {
    let row_bytes = width.checked_mul(3).ok_or("raw row overflow")?;
    let stride = row_bytes
        .checked_add(padding)
        .ok_or("raw stride overflow")?;
    let extent = height
        .saturating_sub(1)
        .checked_mul(stride)
        .and_then(|offset| offset.checked_add(row_bytes))
        .ok_or("raw padded extent overflow")?;
    let mut padded = vec![0xA5; extent];
    for row in 0..height {
        let source = &packed[row * row_bytes..(row + 1) * row_bytes];
        let destination = row * stride;
        padded[destination..destination + row_bytes].copy_from_slice(source);
    }
    Ok((padded, stride))
}

trait SliceRecord {
    fn offset(&self) -> usize;
    fn byte_length(&self) -> usize;
    fn sha256(&self) -> &str;
}

impl SliceRecord for ArtifactSlice {
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

impl SliceRecord for PreparedSlice {
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

fn verified_slice<'a>(
    data: &'a [u8],
    record: &impl SliceRecord,
    case_id: &str,
) -> Result<&'a [u8], Box<dyn std::error::Error>> {
    let end = record
        .offset()
        .checked_add(record.byte_length())
        .ok_or("artifact slice extent overflow")?;
    let slice = data
        .get(record.offset()..end)
        .ok_or("artifact slice is out of bounds")?;
    if sha256(slice) != record.sha256() {
        return Err(format!("{case_id} artifact digest mismatch").into());
    }
    Ok(slice)
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

#[allow(clippy::cast_precision_loss)] // The configured prepared-image cap keeps this sum exact in f64.
fn compare_rgb8(
    actual: &[u8],
    expected: &[u8],
    width: usize,
    tolerance: u8,
) -> Result<Diagnostics, Box<dyn std::error::Error>> {
    if actual.is_empty() || actual.len() != expected.len() {
        return Err(format!(
            "RGB comparison lengths must be equal and non-zero: actual={}, expected={}",
            actual.len(),
            expected.len()
        )
        .into());
    }
    let mut differences = Vec::with_capacity(actual.len());
    let mut first_difference = None;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let difference = actual.abs_diff(expected);
        differences.push(difference);
        if difference != 0 && first_difference.is_none() {
            let pixel = index / 3;
            first_difference = Some(FirstDifference {
                element_index: index,
                y: pixel / width,
                x: pixel % width,
                channel: index % 3,
                actual,
                expected,
            });
        }
    }
    let maximum = differences.iter().copied().max().unwrap_or(0);
    let squared_error = differences
        .iter()
        .map(|&difference| u64::from(difference).pow(2))
        .sum::<u64>();
    let count = u32::try_from(differences.len())?;
    let rmse = (squared_error as f64 / f64::from(count)).sqrt();
    let count_nonzero_differences = differences
        .iter()
        .filter(|&&difference| difference != 0)
        .count();
    let count_over_tolerance = differences
        .iter()
        .filter(|&&difference| difference > tolerance)
        .count();
    differences.sort_unstable();
    Ok(Diagnostics {
        maximum_absolute_error: maximum,
        rmse,
        absolute_error_p50: percentile(&differences, 50),
        absolute_error_p90: percentile(&differences, 90),
        absolute_error_p99: percentile(&differences, 99),
        count_nonzero_differences,
        count_over_tolerance,
        maximum_ulp_distance: maximum,
        first_difference,
    })
}

fn percentile(sorted: &[u8], percent: usize) -> u8 {
    let index = (percent * sorted.len().saturating_sub(1) + 50) / 100;
    sorted[index]
}

fn image_format(format: &str) -> Result<ImageFormat, Box<dyn std::error::Error>> {
    match format {
        "jpeg" => Ok(ImageFormat::Jpeg),
        "png" => Ok(ImageFormat::Png),
        "webp" => Ok(ImageFormat::WebP),
        _ => Err(format!("unknown fixture format: {format}").into()),
    }
}

fn category_name(category: ErrorCategory) -> &'static str {
    match category {
        ErrorCategory::UnsupportedMedia => "unsupported_media",
        ErrorCategory::MediaDecode => "media_decode",
        ErrorCategory::MediaGeometry => "media_geometry",
        ErrorCategory::ResourceLimit => "resource_limit",
        ErrorCategory::ArithmeticOverflow => "arithmetic_overflow",
        _ => "unexpected_error_category",
    }
}

fn file_digest(path: impl AsRef<std::path::Path>) -> Result<FileDigest, std::io::Error> {
    let path = path.as_ref();
    let data = fs::read(path)?;
    Ok(digest(path, &data))
}

fn digest(path: &std::path::Path, data: &[u8]) -> FileDigest {
    FileDigest {
        path: path.display().to_string(),
        byte_length: data.len(),
        sha256: sha256(data),
    }
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
