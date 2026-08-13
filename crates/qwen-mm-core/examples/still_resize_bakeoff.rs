//! Compare the selected production still-image resizer and alternatives against
//! the authenticated 17-case corpus.
//!
//! This is an informational post-selection diagnostic. It does not change the
//! production implementation, frozen tolerance, or video path.

use std::{
    collections::BTreeMap,
    env, fs,
    hint::black_box,
    path::PathBuf,
    process::Command,
    time::{Duration, Instant},
};

use fast_image_resize::{
    FilterType as FirFilter, ResizeAlg, ResizeOptions, Resizer,
    images::{TypedImage, TypedImageRef},
    pixels::U8x3,
};
use pic_scale::{
    ImageSize, ImageStore, ImageStoreMut, ResamplingFunction, Scaler, ThreadingPolicy,
    WorkloadStrategy,
};
use qwen_mm_core::{
    ImageOptions, ProfileAlias, ProfileRegistry, ResourceLimits, plan_image_geometry,
    resize_image_rgb8,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const DEFAULT_MANIFEST: &str = "reference/resize/v1/manifest.json";
const WARMUP_ITERATIONS: usize = 5;
const TIMED_ITERATIONS: usize = 30;
const PRODUCTION_SELECTED: &str = "qwen-mm-selected-production";

#[derive(Deserialize)]
struct Fixture {
    contract_id: String,
    stage_id: String,
    artifacts: BTreeMap<String, ArtifactDigest>,
    comparison_policy: ComparisonPolicy,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct ArtifactDigest {
    byte_length: usize,
    sha256: String,
}

#[derive(Deserialize)]
struct ComparisonPolicy {
    pillow_image_rgb8: ImageComparisonPolicy,
}

#[derive(Deserialize)]
struct ImageComparisonPolicy {
    absolute_byte_error_max: u8,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    tags: Vec<String>,
    source: Source,
    geometry_options: GeometryOptions,
    destination: Dimensions,
    pillow_image_rgb8: ArtifactSlice,
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

#[derive(Clone, Deserialize, Serialize)]
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
    scope: &'static str,
    status: &'static str,
    contract_id: String,
    stage_id: String,
    manifest_path: String,
    manifest_sha256: String,
    image_contract_tolerance: u8,
    actual_resize_case_count: usize,
    no_op_case_count: usize,
    warmup_iterations: usize,
    timed_iterations: usize,
    timing_policy: &'static str,
    diagnostics_scope: &'static str,
    excluded_candidates: BTreeMap<&'static str, &'static str>,
    host: Host,
    candidates: BTreeMap<String, CandidateSummary>,
    cases: Vec<CaseReport>,
}

#[derive(Serialize)]
struct Host {
    os: &'static str,
    architecture: &'static str,
    uname: String,
    rustc: String,
}

#[derive(Default, Serialize)]
struct CandidateSummary {
    exact_version: &'static str,
    license: &'static str,
    filter: &'static str,
    setup_and_reuse: &'static str,
    exact_case_failures: usize,
    contract_case_failures: usize,
    integration_cases_faster_than_production_selected: usize,
    integration_geometric_mean_speedup_vs_production_selected: f64,
    actual_resize_integration_cases_faster_than_production_selected: usize,
    actual_resize_integration_geometric_mean_speedup_vs_production_selected: f64,
}

#[derive(Serialize)]
struct CaseReport {
    id: String,
    tags: Vec<String>,
    resize_required: bool,
    source: Dimensions,
    destination: Dimensions,
    candidates: BTreeMap<String, CandidateCase>,
}

#[derive(Serialize)]
struct CandidateCase {
    exact_passed: bool,
    contract_passed: bool,
    diagnostics: Diagnostics,
    resize_only_timing: Timing,
    end_to_end_integration_timing: Timing,
}

#[derive(Serialize)]
struct Diagnostics {
    maximum_absolute_error: u8,
    mean_absolute_error: f64,
    rmse: f64,
    absolute_error_p50: u8,
    absolute_error_p90: u8,
    absolute_error_p99: u8,
    signed_bias_actual_minus_expected: f64,
    count_nonzero_differences: usize,
    count_over_contract_tolerance: usize,
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

#[derive(Serialize)]
struct Timing {
    samples: usize,
    minimum_microseconds: f64,
    median_microseconds: f64,
    p90_microseconds: f64,
}

struct Measurement {
    output: Vec<u8>,
    resize_only_timing: Timing,
    end_to_end_integration_timing: Timing,
}

#[derive(Clone, Copy)]
enum CandidateKind {
    ProductionSelected,
    Fir(FirFilter),
    Pic(ResamplingFunction, WorkloadStrategy),
}

struct CandidateDefinition {
    name: &'static str,
    kind: CandidateKind,
    version: &'static str,
    license: &'static str,
    filter: &'static str,
    setup_and_reuse: &'static str,
}

fn candidate_definitions() -> Vec<CandidateDefinition> {
    vec![
        CandidateDefinition {
            name: PRODUCTION_SELECTED,
            kind: CandidateKind::ProductionSelected,
            version: "qwen-mm-core=0.1.0; pic-scale=0.7.11",
            license: "project-local; BSD-3-Clause OR Apache-2.0 dependency",
            filter: "selected pic-scale Bicubic / PreferQuality / single-thread",
            setup_and_reuse: "production API creates a plan and allocates scratch/output per call",
        },
        CandidateDefinition {
            name: "fir-6.1.0-catmull-rom",
            kind: CandidateKind::Fir(FirFilter::CatmullRom),
            version: "fast_image_resize=6.1.0",
            license: "MIT OR Apache-2.0",
            filter: "Convolution(CatmullRom)",
            setup_and_reuse: "source pixels, Resizer, options, and destination reused",
        },
        CandidateDefinition {
            name: "fir-6.1.0-bilinear",
            kind: CandidateKind::Fir(FirFilter::Bilinear),
            version: "fast_image_resize=6.1.0",
            license: "MIT OR Apache-2.0",
            filter: "Convolution(Bilinear)",
            setup_and_reuse: "source pixels, Resizer, options, and destination reused",
        },
        CandidateDefinition {
            name: "pic-scale-0.7.11-catmull-rom",
            kind: CandidateKind::Pic(
                ResamplingFunction::CatmullRom,
                WorkloadStrategy::PreferSpeed,
            ),
            version: "pic-scale=0.7.11",
            license: "BSD-3-Clause OR Apache-2.0",
            filter: "CatmullRom / PreferSpeed / single-thread",
            setup_and_reuse: "plan, scratch, source view, and destination reused",
        },
        CandidateDefinition {
            name: "pic-scale-0.7.11-bilinear",
            kind: CandidateKind::Pic(ResamplingFunction::Bilinear, WorkloadStrategy::PreferSpeed),
            version: "pic-scale=0.7.11",
            license: "BSD-3-Clause OR Apache-2.0",
            filter: "Bilinear / PreferSpeed / single-thread",
            setup_and_reuse: "plan, scratch, source view, and destination reused",
        },
        CandidateDefinition {
            name: "pic-scale-0.7.11-bicubic",
            kind: CandidateKind::Pic(ResamplingFunction::Bicubic, WorkloadStrategy::PreferSpeed),
            version: "pic-scale=0.7.11",
            license: "BSD-3-Clause OR Apache-2.0",
            filter: "Bicubic / PreferSpeed / single-thread",
            setup_and_reuse: "plan, scratch, source view, and destination reused",
        },
        CandidateDefinition {
            name: "pic-scale-0.7.11-area",
            kind: CandidateKind::Pic(ResamplingFunction::Area, WorkloadStrategy::PreferSpeed),
            version: "pic-scale=0.7.11",
            license: "BSD-3-Clause OR Apache-2.0",
            filter: "Area / PreferSpeed / single-thread",
            setup_and_reuse: "plan, scratch, source view, and destination reused",
        },
        CandidateDefinition {
            name: "pic-scale-0.7.11-catmull-rom-prefer-quality",
            kind: CandidateKind::Pic(
                ResamplingFunction::CatmullRom,
                WorkloadStrategy::PreferQuality,
            ),
            version: "pic-scale=0.7.11",
            license: "BSD-3-Clause OR Apache-2.0",
            filter: "CatmullRom / PreferQuality / single-thread",
            setup_and_reuse: "plan, scratch, source view, and destination reused",
        },
        CandidateDefinition {
            name: "pic-scale-0.7.11-bicubic-prefer-quality",
            kind: CandidateKind::Pic(ResamplingFunction::Bicubic, WorkloadStrategy::PreferQuality),
            version: "pic-scale=0.7.11",
            license: "BSD-3-Clause OR Apache-2.0",
            filter: "Bicubic / PreferQuality / single-thread",
            setup_and_reuse: "plan, scratch, source view, and destination reused",
        },
    ]
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args().skip(1);
    let mut manifest_path = PathBuf::from(DEFAULT_MANIFEST);
    let mut output_path = None;
    while let Some(argument) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value after {argument}"))?;
        match argument.as_str() {
            "--manifest" => manifest_path = PathBuf::from(value),
            "--output" => output_path = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument: {argument}").into()),
        }
    }
    let output_path = output_path.ok_or("--output is required")?;
    let manifest_bytes = fs::read(&manifest_path)?;
    let fixture: Fixture = serde_json::from_slice(&manifest_bytes)?;
    let fixture_directory = manifest_path.parent().ok_or("manifest has no parent")?;
    let sources = read_verified_artifact(fixture_directory, "sources.rgb8.bin", &fixture)?;
    let pillow = read_verified_artifact(fixture_directory, "pillow-image-rgb8.bin", &fixture)?;
    let tolerance = fixture
        .comparison_policy
        .pillow_image_rgb8
        .absolute_byte_error_max;
    let definitions = candidate_definitions();
    let registry = ProfileRegistry::bundled()?;
    let visual = &registry.get(ProfileAlias::Qwen3Vl8b).visual;
    let mut cases = Vec::with_capacity(fixture.cases.len());

    for case in fixture.cases {
        let source = checked_slice(
            &sources,
            case.source.offset,
            case.source.byte_length,
            &case.id,
        )?;
        verify_digest(source, &case.source.sha256, &case.id)?;
        let expected = checked_slice(
            &pillow,
            case.pillow_image_rgb8.offset,
            case.pillow_image_rgb8.byte_length,
            &case.id,
        )?;
        verify_digest(expected, &case.pillow_image_rgb8.sha256, &case.id)?;
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
            return Err(format!("{} destination does not match geometry plan", case.id).into());
        }
        let packed = pack_rgb8(source, &case.source)?;
        let mut candidates = BTreeMap::new();
        for definition in &definitions {
            let measurement = measure_candidate(
                definition.kind,
                source,
                &packed,
                &case.source,
                &case.destination,
                &plan,
            )?;
            candidates.insert(
                definition.name.to_owned(),
                CandidateCase::new(
                    &measurement.output,
                    expected,
                    usize::try_from(case.destination.width)?,
                    tolerance,
                    measurement.resize_only_timing,
                    measurement.end_to_end_integration_timing,
                )?,
            );
        }
        cases.push(CaseReport {
            id: case.id,
            tags: case.tags,
            resize_required: case.source.height != case.destination.height
                || case.source.width != case.destination.width,
            source: Dimensions {
                height: case.source.height,
                width: case.source.width,
            },
            destination: case.destination,
            candidates,
        });
    }

    let mut candidates = definitions
        .iter()
        .map(|definition| {
            (
                definition.name.to_owned(),
                CandidateSummary {
                    exact_version: definition.version,
                    license: definition.license,
                    filter: definition.filter,
                    setup_and_reuse: definition.setup_and_reuse,
                    ..CandidateSummary::default()
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    for case in &cases {
        for (name, result) in &case.candidates {
            let summary = candidates.get_mut(name).expect("known candidate");
            summary.exact_case_failures += usize::from(!result.exact_passed);
            summary.contract_case_failures += usize::from(!result.contract_passed);
            let production_integration_us = case.candidates[PRODUCTION_SELECTED]
                .end_to_end_integration_timing
                .median_microseconds;
            summary.integration_cases_faster_than_production_selected += usize::from(
                result.end_to_end_integration_timing.median_microseconds
                    < production_integration_us,
            );
            if case.resize_required
                && result.end_to_end_integration_timing.median_microseconds
                    < production_integration_us
            {
                summary.actual_resize_integration_cases_faster_than_production_selected += 1;
            }
        }
    }
    for (name, summary) in &mut candidates {
        summary.integration_geometric_mean_speedup_vs_production_selected =
            geometric_mean(cases.iter().map(|case| {
                case.candidates[PRODUCTION_SELECTED]
                    .end_to_end_integration_timing
                    .median_microseconds
                    / case.candidates[name]
                        .end_to_end_integration_timing
                        .median_microseconds
            }));
        summary.actual_resize_integration_geometric_mean_speedup_vs_production_selected =
            geometric_mean(
                cases
                    .iter()
                    .filter(|case| case.resize_required)
                    .map(|case| {
                        case.candidates[PRODUCTION_SELECTED]
                            .end_to_end_integration_timing
                            .median_microseconds
                            / case.candidates[name]
                                .end_to_end_integration_timing
                                .median_microseconds
                    }),
            );
    }

    let actual_resize_case_count = cases.iter().filter(|case| case.resize_required).count();
    let report = Report {
        schema_version: 1,
        scope: "still-image-only informational candidate bake-off; video unchanged",
        status: "informational-post-selection",
        contract_id: fixture.contract_id,
        stage_id: fixture.stage_id,
        manifest_path: manifest_path.display().to_string(),
        manifest_sha256: sha256(&manifest_bytes),
        image_contract_tolerance: tolerance,
        actual_resize_case_count,
        no_op_case_count: cases.len() - actual_resize_case_count,
        warmup_iterations: WARMUP_ITERATIONS,
        timed_iterations: TIMED_ITERATIONS,
        timing_policy: "for direct library candidates, resize_only_timing is a warm steady-state kernel call with setup excluded and state/output reused; for qwen-mm-selected-production it is a public API call and therefore includes its per-call plan, scratch, and output setup; end_to_end_integration_timing starts from the authenticated possibly-padded RGB8 view and includes validation/packing, weight or plan setup, allocation, resize, and packed Vec output materialization for every candidate",
        diagnostics_scope: "legacy aggregate RGB-channel byte diagnostics; per-channel quality analysis remains future work",
        excluded_candidates: BTreeMap::from([(
            "zenresize=0.3.1",
            "historical diagnostic trial excluded from the checked-in harness: AGPL-3.0-only OR commercial license; its Catmull-Rom sRGB path was slower than the former scalar port on all 17 local ARM64 cases",
        )]),
        host: Host {
            os: env::consts::OS,
            architecture: env::consts::ARCH,
            uname: command_output("uname", &["-a"]),
            rustc: command_output("rustc", &["-Vv"]),
        },
        candidates,
        cases,
    };
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output_path, serde_json::to_vec_pretty(&report)?)?;
    println!("wrote {} ({})", output_path.display(), report.status);
    Ok(())
}

fn measure_candidate(
    kind: CandidateKind,
    source: &[u8],
    packed: &[u8],
    dimensions: &Source,
    destination: &Dimensions,
    plan: &qwen_mm_core::ImageGeometryPlan,
) -> Result<Measurement, Box<dyn std::error::Error>> {
    match kind {
        CandidateKind::ProductionSelected => measure_production_selected(source, dimensions, plan),
        CandidateKind::Fir(filter) => measure_fir(source, packed, dimensions, destination, filter),
        CandidateKind::Pic(filter, workload) => {
            measure_pic(source, dimensions, destination, filter, workload)
        }
    }
}

fn measure_production_selected(
    source: &[u8],
    dimensions: &Source,
    plan: &qwen_mm_core::ImageGeometryPlan,
) -> Result<Measurement, Box<dyn std::error::Error>> {
    for _ in 0..WARMUP_ITERATIONS {
        black_box(resize_image_rgb8(
            source,
            dimensions.height,
            dimensions.width,
            dimensions.stride_bytes,
            plan,
        )?);
    }
    let mut durations = Vec::with_capacity(TIMED_ITERATIONS);
    let mut output = Vec::new();
    for _ in 0..TIMED_ITERATIONS {
        let start = Instant::now();
        output = resize_image_rgb8(
            black_box(source),
            dimensions.height,
            dimensions.width,
            dimensions.stride_bytes,
            plan,
        )?;
        durations.push(start.elapsed());
        black_box(&output);
    }
    Ok(Measurement {
        output,
        resize_only_timing: Timing::new(durations),
        end_to_end_integration_timing: benchmark_output(|| {
            resize_image_rgb8(
                source,
                dimensions.height,
                dimensions.width,
                dimensions.stride_bytes,
                plan,
            )
            .map_err(Into::into)
        })?,
    })
}

fn measure_fir(
    strided_source: &[u8],
    packed_source: &[u8],
    dimensions: &Source,
    destination: &Dimensions,
    filter: FirFilter,
) -> Result<Measurement, Box<dyn std::error::Error>> {
    let source_image = TypedImageRef::<U8x3>::from_buffer(
        u32::try_from(dimensions.width)?,
        u32::try_from(dimensions.height)?,
        packed_source,
    )?;
    let mut output = vec![0_u8; usize::try_from(destination.width * destination.height * 3)?];
    let mut output_image = TypedImage::<U8x3>::from_buffer(
        u32::try_from(destination.width)?,
        u32::try_from(destination.height)?,
        &mut output,
    )?;
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(filter));
    let mut resizer = Resizer::new();
    for _ in 0..WARMUP_ITERATIONS {
        resizer.resize_typed(&source_image, &mut output_image, &options)?;
    }
    let mut durations = Vec::with_capacity(TIMED_ITERATIONS);
    for _ in 0..TIMED_ITERATIONS {
        let start = Instant::now();
        resizer.resize_typed(black_box(&source_image), &mut output_image, &options)?;
        durations.push(start.elapsed());
        black_box(output_image.pixels());
    }
    drop(output_image);
    Ok(Measurement {
        output,
        resize_only_timing: Timing::new(durations),
        end_to_end_integration_timing: benchmark_output(|| {
            fir_once(strided_source, dimensions, destination, filter)
        })?,
    })
}

fn measure_pic(
    strided_source: &[u8],
    dimensions: &Source,
    destination: &Dimensions,
    filter: ResamplingFunction,
    workload: WorkloadStrategy,
) -> Result<Measurement, Box<dyn std::error::Error>> {
    let source_width = usize::try_from(dimensions.width)?;
    let source_height = usize::try_from(dimensions.height)?;
    let destination_width = usize::try_from(destination.width)?;
    let destination_height = usize::try_from(destination.height)?;
    let mut source_store =
        ImageStore::<u8, 3>::borrow(strided_source, source_width, source_height)?;
    source_store.stride = usize::try_from(dimensions.stride_bytes)?;
    let scaler = Scaler::new(filter)
        .set_threading_policy(ThreadingPolicy::Single)
        .set_workload_strategy(workload);
    let plan = scaler.plan_rgb_resampling(
        ImageSize::new(source_width, source_height),
        ImageSize::new(destination_width, destination_height),
    )?;
    let mut scratch = plan.alloc_scratch();
    let mut output = vec![0_u8; destination_width * destination_height * 3];
    {
        let mut destination_store =
            ImageStoreMut::<u8, 3>::borrow(&mut output, destination_width, destination_height)?;
        for _ in 0..WARMUP_ITERATIONS {
            plan.resample_with_scratch(&source_store, &mut destination_store, &mut scratch)?;
        }
        let mut durations = Vec::with_capacity(TIMED_ITERATIONS);
        for _ in 0..TIMED_ITERATIONS {
            let start = Instant::now();
            plan.resample_with_scratch(
                black_box(&source_store),
                &mut destination_store,
                &mut scratch,
            )?;
            durations.push(start.elapsed());
            black_box(destination_store.buffer.borrow());
        }
        Ok(Measurement {
            output,
            resize_only_timing: Timing::new(durations),
            end_to_end_integration_timing: benchmark_output(|| {
                pic_once(strided_source, dimensions, destination, filter, workload)
            })?,
        })
    }
}

fn fir_once(
    source: &[u8],
    dimensions: &Source,
    destination: &Dimensions,
    filter: FirFilter,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let packed = pack_rgb8(source, dimensions)?;
    let source_image = TypedImageRef::<U8x3>::from_buffer(
        u32::try_from(dimensions.width)?,
        u32::try_from(dimensions.height)?,
        &packed,
    )?;
    let mut output = vec![0_u8; usize::try_from(destination.width * destination.height * 3)?];
    let mut output_image = TypedImage::<U8x3>::from_buffer(
        u32::try_from(destination.width)?,
        u32::try_from(destination.height)?,
        &mut output,
    )?;
    Resizer::new().resize_typed(
        &source_image,
        &mut output_image,
        &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(filter)),
    )?;
    drop(output_image);
    Ok(output)
}

fn pic_once(
    source: &[u8],
    dimensions: &Source,
    destination: &Dimensions,
    filter: ResamplingFunction,
    workload: WorkloadStrategy,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let source_width = usize::try_from(dimensions.width)?;
    let source_height = usize::try_from(dimensions.height)?;
    let destination_width = usize::try_from(destination.width)?;
    let destination_height = usize::try_from(destination.height)?;
    let mut source_store = ImageStore::<u8, 3>::borrow(source, source_width, source_height)?;
    source_store.stride = usize::try_from(dimensions.stride_bytes)?;
    let scaler = Scaler::new(filter)
        .set_threading_policy(ThreadingPolicy::Single)
        .set_workload_strategy(workload);
    let plan = scaler.plan_rgb_resampling(
        ImageSize::new(source_width, source_height),
        ImageSize::new(destination_width, destination_height),
    )?;
    let mut scratch = plan.alloc_scratch();
    let mut output = vec![0_u8; destination_width * destination_height * 3];
    {
        let mut destination_store =
            ImageStoreMut::<u8, 3>::borrow(&mut output, destination_width, destination_height)?;
        plan.resample_with_scratch(&source_store, &mut destination_store, &mut scratch)?;
    }
    Ok(output)
}

fn benchmark_output(
    mut operation: impl FnMut() -> Result<Vec<u8>, Box<dyn std::error::Error>>,
) -> Result<Timing, Box<dyn std::error::Error>> {
    for _ in 0..WARMUP_ITERATIONS {
        black_box(operation()?);
    }
    let mut durations = Vec::with_capacity(TIMED_ITERATIONS);
    for _ in 0..TIMED_ITERATIONS {
        let start = Instant::now();
        let output = operation()?;
        durations.push(start.elapsed());
        black_box(output);
    }
    Ok(Timing::new(durations))
}

impl CandidateCase {
    fn new(
        actual: &[u8],
        expected: &[u8],
        width: usize,
        tolerance: u8,
        resize_only_timing: Timing,
        end_to_end_integration_timing: Timing,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if actual.is_empty() || actual.len() != expected.len() {
            return Err(format!(
                "resize comparison lengths must be equal and non-zero: actual={}, expected={}",
                actual.len(),
                expected.len()
            )
            .into());
        }
        let mut absolute = Vec::with_capacity(actual.len());
        let mut signed_sum = 0_i64;
        let mut squared_sum = 0_f64;
        let mut first = None;
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let difference = actual.abs_diff(expected);
            absolute.push(difference);
            let signed = i64::from(actual) - i64::from(expected);
            signed_sum += signed;
            squared_sum += f64::from(difference).powi(2);
            if first.is_none() && difference != 0 {
                let pixel = index / 3;
                first = Some(FirstDifference {
                    element_index: index,
                    y: pixel / width,
                    x: pixel % width,
                    channel: index % 3,
                    actual,
                    expected,
                });
            }
        }
        let count = u32::try_from(absolute.len())?;
        let count_nonzero_differences = absolute.iter().filter(|&&value| value != 0).count();
        let count_over_contract_tolerance =
            absolute.iter().filter(|&&value| value > tolerance).count();
        let maximum_absolute_error = absolute.iter().copied().max().unwrap_or(0);
        let mean_absolute_error =
            absolute.iter().map(|&value| f64::from(value)).sum::<f64>() / f64::from(count);
        let rmse = (squared_sum / f64::from(count)).sqrt();
        absolute.sort_unstable();
        Ok(Self {
            exact_passed: count_nonzero_differences == 0,
            contract_passed: count_over_contract_tolerance == 0,
            diagnostics: Diagnostics {
                maximum_absolute_error,
                mean_absolute_error,
                rmse,
                absolute_error_p50: percentile(&absolute, 50),
                absolute_error_p90: percentile(&absolute, 90),
                absolute_error_p99: percentile(&absolute, 99),
                signed_bias_actual_minus_expected: f64::from(i32::try_from(signed_sum)?)
                    / f64::from(count),
                count_nonzero_differences,
                count_over_contract_tolerance,
                first_difference: first,
            },
            resize_only_timing,
            end_to_end_integration_timing,
        })
    }
}

impl Timing {
    fn new(durations: Vec<Duration>) -> Self {
        let mut microseconds = durations
            .into_iter()
            .map(|duration| duration.as_secs_f64() * 1_000_000.0)
            .collect::<Vec<_>>();
        microseconds.sort_by(f64::total_cmp);
        Self {
            samples: microseconds.len(),
            minimum_microseconds: microseconds[0],
            median_microseconds: percentile(&microseconds, 50),
            p90_microseconds: percentile(&microseconds, 90),
        }
    }
}

fn percentile<T: Copy>(sorted: &[T], percent: usize) -> T {
    let index = (percent * sorted.len().saturating_sub(1) + 50) / 100;
    sorted[index]
}

fn geometric_mean(values: impl Iterator<Item = f64>) -> f64 {
    let values = values.collect::<Vec<_>>();
    let count = u32::try_from(values.len()).expect("resize corpus case count fits u32");
    (values.iter().map(|value| value.ln()).sum::<f64>() / f64::from(count)).exp()
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
    let mut packed = Vec::with_capacity(height * row_bytes);
    for row in 0..height {
        let start = row * stride;
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
    use super::{CandidateCase, Timing};

    #[test]
    fn exact_and_contract_results_are_distinct() {
        let result = CandidateCase::new(
            &[11, 22, 33],
            &[10, 20, 33],
            1,
            1,
            Timing::new(vec![std::time::Duration::from_micros(1)]),
            Timing::new(vec![std::time::Duration::from_micros(2)]),
        )
        .unwrap();
        assert!(!result.exact_passed);
        assert!(!result.contract_passed);
        assert_eq!(result.diagnostics.count_nonzero_differences, 2);
        assert_eq!(result.diagnostics.count_over_contract_tolerance, 1);
    }
}
