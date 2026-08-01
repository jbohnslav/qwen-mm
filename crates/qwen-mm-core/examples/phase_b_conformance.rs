use std::{collections::BTreeMap, env, fs, path::PathBuf, process::ExitCode};

use qwen_mm_core::{
    ContentItem, ImageFormat, ImageInput, ImageOptions, ImageRef, Message, MessageContent,
    ProfileAlias, ProfileRegistry, QwenImageProcessor, Request, RequestOptions, ResourceLimits,
    Rgb8, Role,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

const MANIFEST_TEXT: &str = include_str!("../../../reference/phase-b/v1/manifest.json");
const SOURCES: &[u8] = include_bytes!("../../../reference/phase-b/v1/sources.bin");
const EXPECTED: &[u8] = include_bytes!("../../../reference/phase-b/v1/expected.bin");
const GENERATOR: &str =
    include_str!("../../../reference/src/qwen_mm_reference/phase_b_conformance.py");
const COMPATIBILITY: &str = include_str!("../../../reference/compatibility/v1.json");

#[derive(Debug, Deserialize)]
struct Manifest {
    schema_version: u32,
    contract_id: String,
    profiles: BTreeMap<String, ProfileRecord>,
    generator: GeneratorRecord,
    blobs: BlobRecords,
    sources: BTreeMap<String, SourceRecord>,
    cases: Vec<Case>,
    integrity: Integrity,
}

#[derive(Debug, Deserialize)]
struct ProfileRecord {
    fingerprint: String,
}

#[derive(Debug, Deserialize)]
struct GeneratorRecord {
    module: String,
    command: String,
    source_sha256: String,
    compatibility_sha256: String,
}

#[derive(Debug, Deserialize)]
struct BlobRecords {
    sources: BlobRecord,
    expected: BlobRecord,
}

#[derive(Debug, Deserialize)]
struct BlobRecord {
    path: String,
    length: usize,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct Integrity {
    algorithm: String,
    canonical_json_sha256: String,
}

#[derive(Debug, Deserialize)]
struct SourceRecord {
    kind: String,
    format: Option<String>,
    height: Option<usize>,
    width: Option<usize>,
    row_stride: Option<usize>,
    offset: usize,
    length: usize,
    sha256: String,
    lossless: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct Case {
    id: String,
    inputs: Vec<String>,
    messages: Vec<MessageSpec>,
    #[serde(default)]
    options: OptionsSpec,
    tags: Vec<String>,
    expectations: BTreeMap<String, Expectation>,
}

#[derive(Debug, Deserialize)]
struct MessageSpec {
    role: String,
    content: ContentSpec,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ContentSpec {
    Text(String),
    Items(Vec<ItemSpec>),
}

#[derive(Debug, Deserialize)]
struct ItemSpec {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
    source: Option<String>,
    input_index: Option<usize>,
    #[serde(default)]
    options: ImageOptionsSpec,
}

#[derive(Debug)]
struct ExpectedOccurrence {
    message_index: usize,
    content_item_index: usize,
    input_index: usize,
    source_id: String,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
struct ImageOptionsSpec {
    min_pixels: Option<u64>,
    max_pixels: Option<u64>,
    resized_height: Option<u64>,
    resized_width: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
struct OptionsSpec {
    #[serde(default)]
    add_generation_prompt: bool,
    #[serde(default)]
    add_vision_id: bool,
}

#[derive(Debug, Deserialize)]
struct Expectation {
    rendered_prompt: TextRecord,
    expanded_prompt: TextRecord,
    replacements: Vec<ReplacementRecord>,
    official_keys: Vec<String>,
    prepared_images: Vec<ArrayRecord>,
    arrays: BTreeMap<String, ArrayRecord>,
}

#[derive(Debug, Deserialize)]
struct TextRecord {
    text: String,
    bytes: usize,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct ReplacementRecord {
    #[serde(rename = "type")]
    kind: String,
    original_codepoint_span: [i64; 2],
    expanded_codepoint_span: [i64; 2],
    expanded_token_span: [i64; 2],
}

#[derive(Debug, Deserialize)]
struct ArrayRecord {
    shape: Vec<usize>,
    dtype: String,
    strides: Vec<usize>,
    offset: usize,
    length: usize,
    sha256: String,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    contract_id: &'static str,
    fixture_sha256: String,
    fixture_expected_sha256: String,
    execution: &'static str,
    passed: bool,
    cases: Vec<CaseReport>,
}

#[derive(Debug, Serialize)]
struct CaseReport {
    case_id: String,
    profile: String,
    passed: bool,
    official_keys: Vec<String>,
    occurrence_order: Vec<usize>,
    comparisons: Vec<Comparison>,
    errors: Vec<String>,
}

#[derive(Debug, Serialize)]
struct Comparison {
    array: String,
    bound: f64,
    maximum_absolute_error: f64,
    rmse: f64,
    absolute_error_p50: f64,
    absolute_error_p90: f64,
    absolute_error_p99: f64,
    count_over_bound: usize,
    maximum_ulp_error: u32,
    first_difference: Option<DifferenceCoordinate>,
}

#[derive(Debug, Serialize)]
struct DifferenceCoordinate {
    flat_index: usize,
    row: usize,
    column: usize,
    occurrence: Option<usize>,
    request_index: Option<usize>,
    message_index: Option<usize>,
    content_index: Option<usize>,
    input_index: Option<usize>,
    grid_row: Option<usize>,
    patch: Option<usize>,
    grid_t: Option<usize>,
    grid_y: Option<usize>,
    grid_x: Option<usize>,
    channel: Option<usize>,
    temporal: Option<usize>,
    patch_y: Option<usize>,
    patch_x: Option<usize>,
    prepared_y: Option<usize>,
    prepared_x: Option<usize>,
}

struct Arguments {
    assets_root: PathBuf,
    output: Option<PathBuf>,
    profile: Option<String>,
}

fn main() -> ExitCode {
    let arguments = match arguments() {
        Ok(arguments) => arguments,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };
    match run(&arguments) {
        Ok(report) => {
            let json = serde_json::to_string_pretty(&report).expect("serializable report") + "\n";
            if let Some(path) = &arguments.output {
                if let Some(parent) = path.parent()
                    && let Err(error) = fs::create_dir_all(parent)
                {
                    eprintln!("cannot create report directory: {error}");
                    return ExitCode::FAILURE;
                }
                if let Err(error) = fs::write(path, &json) {
                    eprintln!("cannot write report: {error}");
                    return ExitCode::FAILURE;
                }
            } else {
                print!("{json}");
            }
            if report.passed {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(error) => {
            eprintln!("phase B conformance failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn arguments() -> Result<Arguments, String> {
    let mut assets_root = PathBuf::from("reference/.cache/huggingface");
    let mut output = None;
    let mut profile = None;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--assets-root" => {
                assets_root = PathBuf::from(args.next().ok_or("--assets-root needs a path")?);
            }
            "--output" => output = Some(PathBuf::from(args.next().ok_or("--output needs a path")?)),
            "--profile" => profile = Some(args.next().ok_or("--profile needs an alias")?),
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(Arguments {
        assets_root,
        output,
        profile,
    })
}

fn authenticate_fixture() -> Result<Manifest, String> {
    let mut value: Value =
        serde_json::from_str(MANIFEST_TEXT).map_err(|error| error.to_string())?;
    let integrity = value
        .as_object_mut()
        .and_then(|object| object.remove("integrity"))
        .ok_or("manifest integrity is absent")?;
    let expected_integrity = integrity
        .get("canonical_json_sha256")
        .and_then(Value::as_str)
        .ok_or("manifest integrity digest is absent")?;
    let actual_integrity = digest(&serde_json::to_vec(&value).map_err(|error| error.to_string())?);
    if actual_integrity != expected_integrity {
        return Err(format!(
            "manifest integrity mismatch: expected {expected_integrity}, got {actual_integrity}"
        ));
    }
    let manifest: Manifest =
        serde_json::from_str(MANIFEST_TEXT).map_err(|error| error.to_string())?;
    if manifest.schema_version != 1
        || manifest.contract_id != "qwen-mm-compat-v1"
        || manifest.integrity.algorithm != "sha256"
        || manifest.integrity.canonical_json_sha256 != expected_integrity
    {
        return Err("manifest envelope is not Phase B schema v1".to_owned());
    }
    for (record, bytes, expected_path) in [
        (&manifest.blobs.sources, SOURCES, "sources.bin"),
        (&manifest.blobs.expected, EXPECTED, "expected.bin"),
    ] {
        if record.path != expected_path
            || record.length != bytes.len()
            || record.sha256 != digest(bytes)
        {
            return Err(format!("authenticated blob mismatch: {}", record.path));
        }
    }
    if manifest.generator.module != "qwen_mm_reference.phase_b_conformance"
        || manifest.generator.command != "make phase-b-conformance-regenerate"
        || manifest.generator.source_sha256 != digest(GENERATOR.as_bytes())
        || manifest.generator.compatibility_sha256 != digest(COMPATIBILITY.as_bytes())
    {
        return Err("generator or compatibility provenance mismatch".to_owned());
    }
    for (source_id, source) in &manifest.sources {
        let bytes = slice(SOURCES, source.offset, source.length)?;
        if digest(bytes) != source.sha256 {
            return Err(format!("source slice digest mismatch: {source_id}"));
        }
    }
    for case in &manifest.cases {
        for expectation in case.expectations.values() {
            for (index, array) in expectation.prepared_images.iter().enumerate() {
                authenticate_expected_array(array, &format!("{}/prepared_image_{index}", case.id))?;
            }
            for (name, array) in &expectation.arrays {
                authenticate_expected_array(array, &format!("{}/{name}", case.id))?;
            }
        }
    }
    for required in [
        "text-only",
        "one-image",
        "multi-image",
        "interleaved",
        "repeated-reference",
        "index-order",
        "raw-rgb",
        "png",
        "webp",
        "jpeg",
        "lossless",
        "lossy",
    ] {
        if !manifest
            .cases
            .iter()
            .any(|case| case.tags.iter().any(|tag| tag == required))
        {
            return Err(format!("fixture coverage tag is absent: {required}"));
        }
    }
    Ok(manifest)
}

fn authenticate_expected_array(array: &ArrayRecord, label: &str) -> Result<(), String> {
    let bytes = slice(EXPECTED, array.offset, array.length)?;
    if digest(bytes) != array.sha256 {
        return Err(format!("expected slice digest mismatch: {label}"));
    }
    Ok(())
}

fn run(arguments: &Arguments) -> Result<Report, String> {
    let manifest = authenticate_fixture()?;
    let registry = ProfileRegistry::bundled().map_err(|error| error.to_string())?;
    let aliases = match arguments.profile.as_deref() {
        None => vec![ProfileAlias::Qwen3Vl8b, ProfileAlias::Qwen35_9b],
        Some("qwen3-vl-8b") => vec![ProfileAlias::Qwen3Vl8b],
        Some("qwen3.5-9b") => vec![ProfileAlias::Qwen35_9b],
        Some(other) => return Err(format!("unknown profile alias: {other}")),
    };
    let mut case_reports = Vec::new();
    for alias in aliases {
        let profile = registry.get(alias);
        let manifest_profile = manifest
            .profiles
            .get(alias.as_str())
            .ok_or_else(|| format!("fixture profile is absent: {}", alias.as_str()))?;
        if manifest_profile.fingerprint != profile.fingerprint {
            return Err(format!("fixture profile drift: {}", alias.as_str()));
        }
        let processor = QwenImageProcessor::from_local_assets(
            profile,
            asset_directory(&arguments.assets_root, alias),
            ResourceLimits::default(),
        )
        .map_err(|error| error.to_string())?;
        for case in &manifest.cases {
            let expectation = case
                .expectations
                .get(alias.as_str())
                .ok_or_else(|| format!("missing expectation: {}/{}", case.id, alias.as_str()))?;
            case_reports.push(run_case(&processor, &manifest, case, expectation, alias)?);
        }
    }
    let passed = case_reports.iter().all(|report| report.passed);
    Ok(Report {
        schema_version: 1,
        contract_id: "qwen-mm-compat-v1",
        fixture_sha256: digest(MANIFEST_TEXT.as_bytes()),
        fixture_expected_sha256: digest(EXPECTED),
        execution: "native-rust-no-python-runtime",
        passed,
        cases: case_reports,
    })
}

#[allow(clippy::too_many_lines)]
fn run_case(
    processor: &QwenImageProcessor,
    manifest: &Manifest,
    case: &Case,
    expectation: &Expectation,
    alias: ProfileAlias,
) -> Result<CaseReport, String> {
    let expected_occurrences = expected_occurrences(case)?;
    let inputs = case
        .inputs
        .iter()
        .map(|source_id| image_input(manifest, source_id))
        .collect::<Result<Vec<_>, _>>()?;
    let content_rows = case
        .messages
        .iter()
        .map(|message| match &message.content {
            ContentSpec::Text(_) => Ok(Vec::new()),
            ContentSpec::Items(items) => items
                .iter()
                .map(|item| match item.kind.as_str() {
                    "text" => Ok(ContentItem::Text(
                        item.text.as_deref().ok_or("text item has no text")?,
                    )),
                    "image" => Ok(ContentItem::Image(ImageRef {
                        input_index: item.input_index.ok_or("image item has no input index")?,
                        options: ImageOptions {
                            min_pixels: item.options.min_pixels,
                            max_pixels: item.options.max_pixels,
                            resized_height: item.options.resized_height,
                            resized_width: item.options.resized_width,
                        },
                    })),
                    other => Err(format!("unsupported fixture item: {other}")),
                })
                .collect(),
        })
        .collect::<Result<Vec<Vec<_>>, String>>()?;
    let messages = case
        .messages
        .iter()
        .zip(&content_rows)
        .map(|(message, items)| {
            Ok(Message {
                role: match message.role.as_str() {
                    "system" => Role::System,
                    "user" => Role::User,
                    "assistant" => Role::Assistant,
                    "tool" => Role::Tool,
                    other => return Err(format!("unsupported fixture role: {other}")),
                },
                content: match &message.content {
                    ContentSpec::Text(text) => MessageContent::Text(text),
                    ContentSpec::Items(_) => MessageContent::Items(items),
                },
                tool_calls: &[],
                reasoning_content: None,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let trace = processor
        .prepare_with_media_trace(Request {
            messages: &messages,
            images: &inputs,
            videos: &[],
            options: RequestOptions {
                add_generation_prompt: case.options.add_generation_prompt,
                add_vision_id: case.options.add_vision_id,
                ..RequestOptions::default()
            },
        })
        .map_err(|error| format!("{}/{}: {error}", alias.as_str(), case.id))?;
    let output = &trace.output;

    let mut errors = Vec::new();
    compare_text(
        "rendered",
        &output.text.rendered_prompt,
        &expectation.rendered_prompt,
        &mut errors,
    );
    compare_text(
        "expanded",
        &output.text.expanded_prompt,
        &expectation.expanded_prompt,
        &mut errors,
    );
    if output.text.replacements.len() != expectation.replacements.len() {
        errors.push(format!(
            "replacement count: expected {}, got {}",
            expectation.replacements.len(),
            output.text.replacements.len()
        ));
    }
    for (index, (actual, expected)) in output
        .text
        .replacements
        .iter()
        .zip(&expectation.replacements)
        .enumerate()
    {
        let actual_kind = match actual.modality {
            qwen_mm_core::VisualModality::Image => "image",
            qwen_mm_core::VisualModality::Video => "video",
        };
        if actual_kind != expected.kind
            || [
                actual.rendered_code_points.start,
                actual.rendered_code_points.end,
            ] != expected.original_codepoint_span
            || [
                actual.expanded_code_points.start,
                actual.expanded_code_points.end,
            ] != expected.expanded_codepoint_span
            || [actual.expanded_tokens.start, actual.expanded_tokens.end]
                != expected.expanded_token_span
        {
            errors.push(format!("replacement {index} spans or modality differ"));
        }
    }
    let actual_keys = output
        .batch
        .official_keys()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if actual_keys != expectation.official_keys {
        errors.push(format!(
            "official keys differ: expected {:?}, got {actual_keys:?}",
            expectation.official_keys
        ));
    }

    let arrays = output.batch.arrays();
    let mut comparisons = Vec::new();
    compare_occurrence_metadata(
        processor,
        manifest,
        expectation,
        &expected_occurrences,
        &trace,
        &mut errors,
    )?;
    if trace.prepared_images.len() != expectation.prepared_images.len() {
        errors.push(format!(
            "prepared image count: expected {}, got {}",
            expectation.prepared_images.len(),
            trace.prepared_images.len()
        ));
    }
    for (index, (actual, expected)) in trace
        .prepared_images
        .iter()
        .zip(&expectation.prepared_images)
        .enumerate()
    {
        let source = expected_occurrences
            .get(index)
            .and_then(|occurrence| manifest.sources.get(&occurrence.source_id))
            .ok_or_else(|| format!("prepared occurrence {index} source is absent"))?;
        let bound = if source.kind == "raw_rgb8"
            || source.format.as_deref() == Some("png")
            || source.lossless == Some(true)
        {
            0.0
        } else {
            1.0
        };
        compare_u8(
            &format!("prepared_image_{index}"),
            &actual.prepared_rgb.rgb,
            [
                usize::try_from(actual.prepared_rgb.geometry.height)
                    .map_err(|_| "prepared height does not fit usize")?,
                usize::try_from(actual.prepared_rgb.geometry.width)
                    .map_err(|_| "prepared width does not fit usize")?,
                3,
            ],
            expected,
            bound,
            index,
            &actual.location,
            &mut comparisons,
            &mut errors,
        )?;
    }
    compare_i64(
        "input_ids",
        &arrays.input_ids,
        expectation,
        &mut comparisons,
        &mut errors,
    )?;
    compare_i64(
        "attention_mask",
        &arrays.attention_mask,
        expectation,
        &mut comparisons,
        &mut errors,
    )?;
    compare_i64(
        "mm_token_type_ids",
        &arrays.mm_token_type_ids,
        expectation,
        &mut comparisons,
        &mut errors,
    )?;
    if let Some(actual) = &arrays.image_grid_thw {
        compare_i64(
            "image_grid_thw",
            actual,
            expectation,
            &mut comparisons,
            &mut errors,
        )?;
    }
    if let Some(actual) = &arrays.pixel_values {
        compare_f32(
            "pixel_values",
            actual,
            expectation,
            0.0,
            &output.images,
            &processor.profile().visual,
            &mut comparisons,
            &mut errors,
        )?;
    }
    let occurrence_order = output
        .images
        .iter()
        .map(|occurrence| occurrence.location.input_index)
        .collect::<Vec<_>>();
    let expected_order = expected_occurrences
        .iter()
        .map(|occurrence| occurrence.input_index)
        .collect::<Vec<_>>();
    if occurrence_order != expected_order {
        errors.push(format!(
            "occurrence order differs: expected {expected_order:?}, got {occurrence_order:?}"
        ));
    }
    Ok(CaseReport {
        case_id: case.id.clone(),
        profile: alias.as_str().to_owned(),
        passed: errors.is_empty(),
        official_keys: actual_keys,
        occurrence_order,
        comparisons,
        errors,
    })
}

fn expected_occurrences(case: &Case) -> Result<Vec<ExpectedOccurrence>, String> {
    let mut occurrences = Vec::new();
    for (message_index, message) in case.messages.iter().enumerate() {
        let ContentSpec::Items(items) = &message.content else {
            continue;
        };
        for (content_item_index, item) in items.iter().enumerate() {
            if item.kind != "image" {
                continue;
            }
            let input_index = item
                .input_index
                .ok_or_else(|| format!("{}/image item has no input index", case.id))?;
            let source_id = item
                .source
                .as_ref()
                .ok_or_else(|| format!("{}/image item has no source", case.id))?;
            let input_source = case
                .inputs
                .get(input_index)
                .ok_or_else(|| format!("{}/image input index is out of bounds", case.id))?;
            if input_source != source_id {
                return Err(format!(
                    "{}/image source {source_id} disagrees with inputs[{input_index}]={input_source}",
                    case.id
                ));
            }
            occurrences.push(ExpectedOccurrence {
                message_index,
                content_item_index,
                input_index,
                source_id: source_id.clone(),
            });
        }
    }
    Ok(occurrences)
}

#[allow(clippy::too_many_lines)]
fn compare_occurrence_metadata(
    processor: &QwenImageProcessor,
    manifest: &Manifest,
    expectation: &Expectation,
    expected: &[ExpectedOccurrence],
    trace: &qwen_mm_core::PreparedImageTrace,
    errors: &mut Vec<String>,
) -> Result<(), String> {
    let output = &trace.output;
    if output.images.len() != expected.len() {
        errors.push(format!(
            "occurrence count: expected {}, got {}",
            expected.len(),
            output.images.len()
        ));
    }
    if output.batch.sidecar().images.len() != expected.len() {
        errors.push(format!(
            "image sidecar count: expected {}, got {}",
            expected.len(),
            output.batch.sidecar().images.len()
        ));
    }
    if trace.prepared_images.len() != expected.len() {
        errors.push(format!(
            "trace occurrence count: expected {}, got {}",
            expected.len(),
            trace.prepared_images.len()
        ));
    }
    if expected.is_empty() {
        return Ok(());
    }

    let grid = expected_i64_values(expectation, "image_grid_thw")?;
    if grid.len() != expected.len().saturating_mul(3) {
        return Err("authenticated image grid does not match occurrence count".to_owned());
    }
    if expectation.prepared_images.len() != expected.len() {
        return Err("authenticated prepared-image count does not match occurrences".to_owned());
    }

    let merge_area = processor
        .profile()
        .visual
        .merge_size
        .checked_mul(processor.profile().visual.merge_size)
        .ok_or("profile merge area overflowed")?;
    let mut pixel_start = 0_u64;
    for (index, expected_occurrence) in expected.iter().enumerate() {
        let Some(actual) = output.images.get(index) else {
            continue;
        };
        let expected_location = qwen_mm_core::OccurrenceLocation {
            request_index: 0,
            message_index: expected_occurrence.message_index,
            content_item_index: expected_occurrence.content_item_index,
            input_index: expected_occurrence.input_index,
        };
        if actual.location != expected_location {
            errors.push(format!(
                "occurrence {index} location differs: expected {expected_location:?}, got {:?}",
                actual.location
            ));
        }
        if actual.grid_row != index {
            errors.push(format!(
                "occurrence {index} grid row differs: got {}",
                actual.grid_row
            ));
        }

        let source = manifest
            .sources
            .get(&expected_occurrence.source_id)
            .ok_or_else(|| {
                format!(
                    "unknown occurrence source: {}",
                    expected_occurrence.source_id
                )
            })?;
        let source_height = u64::try_from(source.height.ok_or("source height is absent")?)
            .map_err(|_| "source height does not fit u64")?;
        let source_width = u64::try_from(source.width.ok_or("source width is absent")?)
            .map_err(|_| "source width does not fit u64")?;
        if [actual.source_height, actual.source_width] != [source_height, source_width] {
            errors.push(format!(
                "occurrence {index} source dimensions differ: expected {source_height}x{source_width}, got {}x{}",
                actual.source_height, actual.source_width
            ));
        }

        let grid_row = &grid[index * 3..index * 3 + 3];
        let expected_grid = grid_row
            .iter()
            .map(|&value| {
                u64::try_from(value).map_err(|_| "authenticated grid dimension is not positive")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let patch_rows = expected_grid.iter().try_fold(1_u64, |product, &value| {
            product
                .checked_mul(value)
                .ok_or("authenticated grid product overflowed")
        })?;
        if patch_rows == 0 || !patch_rows.is_multiple_of(merge_area) {
            return Err("authenticated grid violates placeholder arithmetic".to_owned());
        }
        let pixel_end = pixel_start
            .checked_add(patch_rows)
            .ok_or("authenticated pixel row range overflowed")?;
        let expected_pixel_rows = qwen_mm_core::CoordinateRange {
            start: i64::try_from(pixel_start).map_err(|_| "pixel start does not fit i64")?,
            end: i64::try_from(pixel_end).map_err(|_| "pixel end does not fit i64")?,
        };
        if actual.pixel_rows != expected_pixel_rows {
            errors.push(format!(
                "occurrence {index} pixel rows differ: expected {expected_pixel_rows:?}, got {:?}",
                actual.pixel_rows
            ));
        }
        pixel_start = pixel_end;

        let prepared = &expectation.prepared_images[index];
        let [prepared_height, prepared_width, channels]: [usize; 3] = prepared
            .shape
            .as_slice()
            .try_into()
            .map_err(|_| "authenticated prepared image is not rank-three")?;
        if channels != 3 {
            return Err("authenticated prepared image is not RGB".to_owned());
        }
        let expected_height =
            u64::try_from(prepared_height).map_err(|_| "prepared height does not fit u64")?;
        let expected_width =
            u64::try_from(prepared_width).map_err(|_| "prepared width does not fit u64")?;
        let expected_rgb_stride = expected_width
            .checked_mul(3)
            .ok_or("prepared RGB row stride overflowed")?;
        let expected_pixel_stride = processor
            .profile()
            .visual
            .patch_width
            .checked_mul(4)
            .ok_or("prepared pixel row stride overflowed")?;
        let expected_pixel_capacity = patch_rows
            .checked_mul(expected_pixel_stride)
            .ok_or("prepared pixel capacity overflowed")?;
        if actual.geometry.height != expected_height
            || actual.geometry.width != expected_width
            || actual.geometry.image_grid_thw != expected_grid.as_slice()
            || actual.geometry.patch_rows != patch_rows
            || actual.geometry.placeholder_count != patch_rows / merge_area
            || actual.geometry.rgb_row_stride_bytes != expected_rgb_stride
            || actual.geometry.rgb_capacity_bytes
                != u64::try_from(prepared.length)
                    .map_err(|_| "prepared byte length does not fit u64")?
            || actual.geometry.pixel_values_row_stride_bytes != expected_pixel_stride
            || actual.geometry.pixel_values_capacity_bytes != expected_pixel_capacity
            || actual.geometry.image_grid_row_stride_bytes != 24
            || actual.geometry.image_grid_capacity_bytes != 24
        {
            errors.push(format!("occurrence {index} geometry metadata differs"));
        }

        if let Some(traced) = trace.prepared_images.get(index)
            && (traced.location != expected_location
                || traced.prepared_rgb.source_height != source_height
                || traced.prepared_rgb.source_width != source_width
                || traced.prepared_rgb.geometry != actual.geometry)
        {
            errors.push(format!("occurrence {index} trace metadata differs"));
        }
        if let (Some(sidecar), Some(replacement)) = (
            output.batch.sidecar().images.get(index),
            output.text.replacements.get(index),
        ) && (sidecar.request_index != 0
            || sidecar.grid_row != i64::try_from(index).map_err(|_| "grid row does not fit i64")?
            || sidecar.replacement.code_points != replacement.rendered_code_points
            || sidecar.replacement.tokens != replacement.expanded_tokens)
        {
            errors.push(format!("occurrence {index} sidecar metadata differs"));
        }
    }
    Ok(())
}

fn image_input<'a>(manifest: &Manifest, source_id: &str) -> Result<ImageInput<'a>, String> {
    let source = manifest
        .sources
        .get(source_id)
        .ok_or_else(|| format!("unknown source: {source_id}"))?;
    let data = slice(SOURCES, source.offset, source.length)?;
    match source.kind.as_str() {
        "raw_rgb8" => Ok(ImageInput::Rgb8(Rgb8 {
            data,
            height: source.height.ok_or("raw height is absent")?,
            width: source.width.ok_or("raw width is absent")?,
            row_stride: source.row_stride.ok_or("raw stride is absent")?,
        })),
        "encoded" => Ok(ImageInput::Encoded {
            data,
            format: match source.format.as_deref() {
                Some("jpeg") => ImageFormat::Jpeg,
                Some("png") => ImageFormat::Png,
                Some("webp") => ImageFormat::WebP,
                other => return Err(format!("unsupported source format: {other:?}")),
            },
        }),
        other => Err(format!("unsupported source kind: {other}")),
    }
}

fn compare_text(label: &str, actual: &str, expected: &TextRecord, errors: &mut Vec<String>) {
    if actual != expected.text
        || actual.len() != expected.bytes
        || digest(actual.as_bytes()) != expected.sha256
    {
        errors.push(format!("{label} prompt bytes differ"));
    }
}

fn expected_i64_values(expectation: &Expectation, name: &str) -> Result<Vec<i64>, String> {
    let record = expectation
        .arrays
        .get(name)
        .ok_or_else(|| format!("expected array is absent: {name}"))?;
    Ok(slice(EXPECTED, record.offset, record.length)?
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("eight-byte chunk")))
        .collect())
}

#[allow(clippy::cast_precision_loss)]
fn compare_i64(
    name: &str,
    actual: &qwen_mm_core::Matrix<i64>,
    expectation: &Expectation,
    comparisons: &mut Vec<Comparison>,
    errors: &mut Vec<String>,
) -> Result<(), String> {
    let record = expectation
        .arrays
        .get(name)
        .ok_or_else(|| format!("expected array is absent: {name}"))?;
    validate_descriptor(
        name,
        actual.shape(),
        actual.byte_strides().map_err(|e| e.to_string())?,
        "int64",
        record,
        errors,
    );
    let expected = expected_i64_values(expectation, name)?;
    let actual_values = actual.as_slice();
    let first = actual_values
        .iter()
        .zip(&expected)
        .position(|(left, right)| left != right);
    let count = actual_values
        .iter()
        .zip(&expected)
        .filter(|(left, right)| left != right)
        .count()
        + actual_values.len().abs_diff(expected.len());
    let mut differences = actual_values
        .iter()
        .zip(&expected)
        .map(|(&left, &right)| (i128::from(left) - i128::from(right)).unsigned_abs() as f64)
        .collect::<Vec<_>>();
    let maximum = differences.iter().copied().fold(0.0_f64, f64::max);
    let rmse = if differences.is_empty() {
        0.0
    } else {
        (differences.iter().map(|value| value * value).sum::<f64>()
            / sample_count_as_f64(differences.len()))
        .sqrt()
    };
    differences.sort_by(f64::total_cmp);
    if count != 0 {
        errors.push(format!("{name}: {count} exact values differ"));
    }
    comparisons.push(Comparison {
        array: name.to_owned(),
        bound: 0.0,
        maximum_absolute_error: maximum,
        rmse,
        absolute_error_p50: percentile(&differences, 50, 100),
        absolute_error_p90: percentile(&differences, 90, 100),
        absolute_error_p99: percentile(&differences, 99, 100),
        count_over_bound: count,
        maximum_ulp_error: 0,
        first_difference: first.map(|index| matrix_coordinate(index, actual.shape())),
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn compare_f32(
    name: &str,
    actual: &qwen_mm_core::Matrix<f32>,
    expectation: &Expectation,
    bound: f64,
    occurrences: &[qwen_mm_core::ProcessedImageOccurrence],
    visual: &qwen_mm_core::VisualProfile,
    comparisons: &mut Vec<Comparison>,
    errors: &mut Vec<String>,
) -> Result<(), String> {
    let record = expectation
        .arrays
        .get(name)
        .ok_or_else(|| format!("expected array is absent: {name}"))?;
    validate_descriptor(
        name,
        actual.shape(),
        actual.byte_strides().map_err(|e| e.to_string())?,
        "float32",
        record,
        errors,
    );
    let expected = slice(EXPECTED, record.offset, record.length)?
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    let length = actual.as_slice().len().min(expected.len());
    let mut differences = actual.as_slice()[..length]
        .iter()
        .zip(&expected[..length])
        .map(|(&left, &right)| absolute_difference_f32(left, right))
        .collect::<Vec<_>>();
    let maximum = differences.iter().copied().fold(0.0_f64, f64::max);
    let rmse = if differences.is_empty() {
        0.0
    } else {
        (differences.iter().map(|value| value * value).sum::<f64>()
            / sample_count_as_f64(differences.len()))
        .sqrt()
    };
    let count = differences.iter().filter(|&&value| value > bound).count()
        + actual.as_slice().len().abs_diff(expected.len());
    let first = differences.iter().position(|&value| value > bound);
    let maximum_ulp_error = actual.as_slice()[..length]
        .iter()
        .zip(&expected[..length])
        .map(|(&left, &right)| ulp_distance(left, right))
        .max()
        .unwrap_or(0);
    differences.sort_by(f64::total_cmp);
    if count != 0 {
        errors.push(format!("{name}: {count} values exceed bound {bound}"));
    }
    comparisons.push(Comparison {
        array: name.to_owned(),
        bound,
        maximum_absolute_error: maximum,
        rmse,
        absolute_error_p50: percentile(&differences, 50, 100),
        absolute_error_p90: percentile(&differences, 90, 100),
        absolute_error_p99: percentile(&differences, 99, 100),
        count_over_bound: count,
        maximum_ulp_error,
        first_difference: first
            .map(|index| pixel_coordinate(index, actual.shape(), occurrences, visual)),
    });
    Ok(())
}

fn absolute_difference_f32(left: f32, right: f32) -> f64 {
    if left.is_finite() && right.is_finite() {
        (f64::from(left) - f64::from(right)).abs()
    } else if left.to_bits() == right.to_bits() {
        0.0
    } else {
        // Keep the report JSON-serializable while making every non-finite
        // mismatch exceed any frozen compatibility bound.
        f64::from(f32::MAX)
    }
}

#[allow(clippy::too_many_arguments)]
fn compare_u8(
    name: &str,
    actual: &[u8],
    shape: [usize; 3],
    record: &ArrayRecord,
    bound: f64,
    occurrence: usize,
    location: &qwen_mm_core::OccurrenceLocation,
    comparisons: &mut Vec<Comparison>,
    errors: &mut Vec<String>,
) -> Result<(), String> {
    let strides = [shape[1] * shape[2], shape[2], 1];
    if record.shape != shape || record.strides != strides || record.dtype != "uint8" {
        errors.push(format!("{name}: shape/dtype/strides differ"));
    }
    let expected = slice(EXPECTED, record.offset, record.length)?;
    let length = actual.len().min(expected.len());
    let mut differences = actual[..length]
        .iter()
        .zip(&expected[..length])
        .map(|(&left, &right)| f64::from(left.abs_diff(right)))
        .collect::<Vec<_>>();
    let maximum = differences.iter().copied().fold(0.0_f64, f64::max);
    let rmse = if differences.is_empty() {
        0.0
    } else {
        (differences.iter().map(|value| value * value).sum::<f64>()
            / sample_count_as_f64(differences.len()))
        .sqrt()
    };
    let count = differences.iter().filter(|&&value| value > bound).count()
        + actual.len().abs_diff(expected.len());
    let first = differences.iter().position(|&value| value > bound);
    differences.sort_by(f64::total_cmp);
    if count != 0 {
        errors.push(format!("{name}: {count} bytes exceed bound {bound}"));
    }
    comparisons.push(Comparison {
        array: name.to_owned(),
        bound,
        maximum_absolute_error: maximum,
        rmse,
        absolute_error_p50: percentile(&differences, 50, 100),
        absolute_error_p90: percentile(&differences, 90, 100),
        absolute_error_p99: percentile(&differences, 99, 100),
        count_over_bound: count,
        maximum_ulp_error: actual[..length]
            .iter()
            .zip(&expected[..length])
            .map(|(&left, &right)| u32::from(left.abs_diff(right)))
            .max()
            .unwrap_or(0),
        first_difference: first
            .map(|index| prepared_coordinate(index, shape, occurrence, location)),
    });
    Ok(())
}

fn validate_descriptor(
    name: &str,
    shape: [usize; 2],
    strides: [u64; 2],
    dtype: &str,
    expected: &ArrayRecord,
    errors: &mut Vec<String>,
) {
    let actual_strides = strides.map(|value| usize::try_from(value).ok());
    let expected_strides = expected
        .strides
        .as_slice()
        .try_into()
        .ok()
        .map(|values: [usize; 2]| values.map(Some));
    if expected.shape != shape
        || Some(actual_strides) != expected_strides
        || expected.dtype != dtype
    {
        errors.push(format!("{name}: shape/dtype/strides differ"));
    }
}

fn sample_count_as_f64(length: usize) -> f64 {
    f64::from(u32::try_from(length).expect("authenticated fixture arrays fit in u32"))
}

fn percentile(sorted: &[f64], numerator: usize, denominator: usize) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = (sorted.len() - 1)
        .saturating_mul(numerator)
        .saturating_add(denominator / 2)
        / denominator;
    sorted[index]
}

fn ulp_distance(left: f32, right: f32) -> u32 {
    fn ordered(value: f32) -> u32 {
        let bits = value.to_bits();
        if bits & 0x8000_0000 == 0 {
            bits | 0x8000_0000
        } else {
            !bits
        }
    }
    ordered(left).abs_diff(ordered(right))
}

fn matrix_coordinate(index: usize, shape: [usize; 2]) -> DifferenceCoordinate {
    DifferenceCoordinate {
        flat_index: index,
        row: index / shape[1],
        column: index % shape[1],
        occurrence: None,
        request_index: None,
        message_index: None,
        content_index: None,
        input_index: None,
        grid_row: None,
        patch: None,
        grid_t: None,
        grid_y: None,
        grid_x: None,
        channel: None,
        temporal: None,
        patch_y: None,
        patch_x: None,
        prepared_y: None,
        prepared_x: None,
    }
}

fn pixel_coordinate(
    index: usize,
    shape: [usize; 2],
    occurrences: &[qwen_mm_core::ProcessedImageOccurrence],
    visual: &qwen_mm_core::VisualProfile,
) -> DifferenceCoordinate {
    let row = index / shape[1];
    let column = index % shape[1];
    let located = occurrences.iter().enumerate().find(|(_, occurrence)| {
        usize::try_from(occurrence.pixel_rows.start).is_ok_and(|start| row >= start)
            && usize::try_from(occurrence.pixel_rows.end).is_ok_and(|end| row < end)
    });
    let (occurrence_index, location, grid_row, patch, grid_t, grid_y, grid_x) = located.map_or(
        (None, None, None, None, None, None, None),
        |(index, item)| {
            let start = usize::try_from(item.pixel_rows.start).expect("validated non-negative row");
            let patch = row - start;
            let grid_height =
                usize::try_from(item.geometry.image_grid_thw[1]).expect("validated grid height");
            let grid_width =
                usize::try_from(item.geometry.image_grid_thw[2]).expect("validated grid width");
            let spatial_patches = grid_height
                .checked_mul(grid_width)
                .expect("validated spatial grid product");
            let temporal = patch / spatial_patches;
            let spatial_patch = patch % spatial_patches;
            let merge = usize::try_from(visual.merge_size).expect("validated merge size");
            let merge_area = merge.checked_mul(merge).expect("validated merge area");
            let block = spatial_patch / merge_area;
            let within_block = spatial_patch % merge_area;
            let outer_width = grid_width / merge;
            let grid_y = block / outer_width * merge + within_block / merge;
            let grid_x = block % outer_width * merge + within_block % merge;
            (
                Some(index),
                Some(item.location),
                Some(item.grid_row),
                Some(patch),
                Some(temporal),
                Some(grid_y),
                Some(grid_x),
            )
        },
    );
    let patch_size = usize::try_from(visual.patch_size).expect("validated patch size");
    let temporal_size =
        usize::try_from(visual.temporal_patch_size).expect("validated temporal patch size");
    let patch_area = patch_size
        .checked_mul(patch_size)
        .expect("validated patch area");
    let channel_width = temporal_size
        .checked_mul(patch_area)
        .expect("validated channel width");
    let channel = column / channel_width;
    let within_channel = column % channel_width;
    let temporal = within_channel / patch_area;
    let within_patch = within_channel % patch_area;
    DifferenceCoordinate {
        flat_index: index,
        row,
        column,
        occurrence: occurrence_index,
        request_index: location.map(|item| item.request_index),
        message_index: location.map(|item| item.message_index),
        content_index: location.map(|item| item.content_item_index),
        input_index: location.map(|item| item.input_index),
        grid_row,
        patch,
        grid_t,
        grid_y,
        grid_x,
        channel: Some(channel),
        temporal: Some(temporal),
        patch_y: Some(within_patch / patch_size),
        patch_x: Some(within_patch % patch_size),
        prepared_y: None,
        prepared_x: None,
    }
}

fn prepared_coordinate(
    index: usize,
    shape: [usize; 3],
    occurrence: usize,
    location: &qwen_mm_core::OccurrenceLocation,
) -> DifferenceCoordinate {
    let row = index / (shape[1] * shape[2]);
    let column = index % (shape[1] * shape[2]);
    DifferenceCoordinate {
        flat_index: index,
        row,
        column,
        occurrence: Some(occurrence),
        request_index: Some(location.request_index),
        message_index: Some(location.message_index),
        content_index: Some(location.content_item_index),
        input_index: Some(location.input_index),
        grid_row: Some(occurrence),
        patch: None,
        grid_t: None,
        grid_y: None,
        grid_x: None,
        channel: Some(index % shape[2]),
        temporal: None,
        patch_y: None,
        patch_x: None,
        prepared_y: Some(row),
        prepared_x: Some(index / shape[2] % shape[1]),
    }
}

fn slice(bytes: &[u8], offset: usize, length: usize) -> Result<&[u8], String> {
    let end = offset
        .checked_add(length)
        .ok_or("fixture slice offset overflowed")?;
    bytes
        .get(offset..end)
        .ok_or_else(|| format!("fixture slice is out of bounds: {offset}..{end}"))
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn asset_directory(root: &std::path::Path, alias: ProfileAlias) -> PathBuf {
    match alias {
        ProfileAlias::Qwen3Vl8b => root.join(
            "models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        ),
        ProfileAlias::Qwen35_9b => {
            root.join("models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Arguments, absolute_difference_f32, authenticate_fixture, expected_occurrences,
        pixel_coordinate, run,
    };
    use qwen_mm_core::{
        CoordinateRange, ImageGeometryPlan, OccurrenceLocation, ProcessedImageOccurrence,
        ProfileAlias, ProfileRegistry,
    };
    use std::path::PathBuf;

    #[test]
    fn committed_fixture_authenticates_without_python_or_model_assets() {
        let fixture = authenticate_fixture().expect("authenticated fixture");
        assert_eq!(fixture.cases.len(), 5);
        assert_eq!(fixture.profiles.len(), 2);
        let repeated = fixture
            .cases
            .iter()
            .find(|case| case.id == "repeat-index-order")
            .expect("repeat case");
        let occurrences = expected_occurrences(repeated).expect("manifest traversal");
        assert_eq!(
            occurrences
                .iter()
                .map(|occurrence| occurrence.input_index)
                .collect::<Vec<_>>(),
            [1, 0, 1]
        );
        assert_eq!(occurrences[2].content_item_index, 4);
    }

    #[test]
    fn non_finite_float_mismatches_never_pass_a_numeric_bound() {
        assert_eq!(absolute_difference_f32(1.0, 1.0), 0.0);
        assert!(absolute_difference_f32(f32::NAN, 0.0) > 1.0);
        assert!(absolute_difference_f32(f32::INFINITY, 1.0) > 1.0);
        assert!(absolute_difference_f32(f32::NEG_INFINITY, f32::INFINITY) > 1.0);
    }

    #[test]
    fn final_pixel_diagnostic_decodes_every_patch_coordinate() {
        let occurrence = ProcessedImageOccurrence {
            location: OccurrenceLocation {
                request_index: 0,
                message_index: 1,
                content_item_index: 2,
                input_index: 3,
            },
            grid_row: 4,
            pixel_rows: CoordinateRange { start: 7, end: 23 },
            source_height: 64,
            source_width: 64,
            geometry: ImageGeometryPlan {
                height: 64,
                width: 64,
                image_grid_thw: [1, 4, 4],
                patch_rows: 16,
                placeholder_count: 4,
                rgb_row_stride_bytes: 192,
                rgb_capacity_bytes: 12_288,
                pixel_values_row_stride_bytes: 6_144,
                pixel_values_capacity_bytes: 98_304,
                image_grid_row_stride_bytes: 24,
                image_grid_capacity_bytes: 24,
            },
        };
        let column = 2 * 2 * 16 * 16 + 1 * 16 * 16 + 5 * 16 + 9;
        let flat = 11 * 1536 + column;
        let registry = ProfileRegistry::bundled().expect("profiles");
        let coordinate = pixel_coordinate(
            flat,
            [32, 1536],
            &[occurrence],
            &registry.get(ProfileAlias::Qwen3Vl8b).visual,
        );
        assert_eq!(coordinate.occurrence, Some(0));
        assert_eq!(coordinate.request_index, Some(0));
        assert_eq!(coordinate.message_index, Some(1));
        assert_eq!(coordinate.content_index, Some(2));
        assert_eq!(coordinate.input_index, Some(3));
        assert_eq!(coordinate.grid_row, Some(4));
        assert_eq!(coordinate.patch, Some(4));
        assert_eq!(coordinate.grid_t, Some(0));
        assert_eq!(coordinate.grid_y, Some(0));
        assert_eq!(coordinate.grid_x, Some(2));
        assert_eq!(coordinate.channel, Some(2));
        assert_eq!(coordinate.temporal, Some(1));
        assert_eq!(coordinate.patch_y, Some(5));
        assert_eq!(coordinate.patch_x, Some(9));
    }

    #[test]
    #[ignore = "requires both hash-pinned local model snapshots under reference/.cache"]
    fn complete_phase_b_corpus_matches_both_profiles() {
        let report = run(&Arguments {
            assets_root: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../reference/.cache/huggingface"),
            output: None,
            profile: None,
        })
        .expect("Phase B report");
        assert!(report.passed, "{report:#?}");
        assert_eq!(report.cases.len(), 10);
    }
}
