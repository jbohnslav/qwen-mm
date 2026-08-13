//! Produce packed Rust candidate blobs for the authenticated resize-quality v2 holdout.
//!
//! This tool deliberately does not read the Pillow pixel blob. It authenticates
//! the frozen manifest and source blob, then writes candidate pixels at the
//! Pillow-declared offsets for evaluation by the Python comparator.

use std::{collections::BTreeMap, env, fs, path::PathBuf};

use fast_image_resize::{
    FilterType as FirFilter, ResizeAlg, ResizeOptions, Resizer,
    images::{TypedImage, TypedImageRef},
    pixels::U8x3,
};
use pic_scale::{
    ImageSize, ImageStore, ImageStoreMut, ResamplingFunction, Scaler, ThreadingPolicy,
    WorkloadStrategy,
};
use qwen_mm_core::{ImageGeometryPlan, resize_image_rgb8};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const DEFAULT_MANIFEST: &str = "reference/resize/v2/manifest.json";
const DEFAULT_OUTPUT_DIRECTORY: &str = "/tmp/qwen-mm-resize-v2";
const MANIFEST_SHA256: &str = "c19c0de491480a3a9f08ceb988b3ae2b755c24732d54e58521f95509e026f4f7";
const CONTRACT_ID: &str = "qwen-mm-still-image-resize-v2";
const STAGE_ID: &str = "qwen-mm-still-image-resize-holdout-v2";
const HOLDOUT_SEED: u64 = 20_260_813;

#[derive(Deserialize)]
struct Manifest {
    schema_version: u8,
    contract_id: String,
    stage_id: String,
    holdout_seed: u64,
    candidate_blind: bool,
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
    source: Source,
    destination: Dimensions,
    pillow_image_rgb8: OutputSlice,
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
struct Dimensions {
    height: u64,
    width: u64,
}

#[derive(Deserialize)]
struct OutputSlice {
    offset: usize,
    byte_length: usize,
    shape: [usize; 3],
}

#[derive(Clone, Copy)]
enum Backend {
    Scalar,
    FirCatmullRom,
    Pic(ResamplingFunction, WorkloadStrategy),
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os().skip(1);
    let manifest_path = arguments
        .next()
        .map_or_else(|| PathBuf::from(DEFAULT_MANIFEST), PathBuf::from);
    let output_directory = arguments
        .next()
        .map_or_else(|| PathBuf::from(DEFAULT_OUTPUT_DIRECTORY), PathBuf::from);
    if arguments.next().is_some() {
        return Err("usage: resize_quality_v2_candidates [manifest] [output-directory]".into());
    }

    let manifest_bytes = fs::read(&manifest_path)?;
    verify_digest(&manifest_bytes, MANIFEST_SHA256, "manifest")?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
    authenticate_manifest(&manifest)?;

    let corpus_directory = manifest_path
        .parent()
        .ok_or("manifest path has no parent directory")?;
    let sources = read_verified_artifact(corpus_directory, "sources.rgb8.bin", &manifest)?;
    let output_length = authenticate_case_layout(&manifest, &sources)?;
    fs::create_dir_all(&output_directory)?;

    for (name, backend) in [
        ("scalar", Backend::Scalar),
        ("fir-catmull-rom", Backend::FirCatmullRom),
        (
            "pic-scale-bicubic",
            Backend::Pic(ResamplingFunction::Bicubic, WorkloadStrategy::PreferSpeed),
        ),
        (
            "pic-scale-catmull-rom",
            Backend::Pic(
                ResamplingFunction::CatmullRom,
                WorkloadStrategy::PreferSpeed,
            ),
        ),
        (
            "pic-scale-bicubic-prefer-quality",
            Backend::Pic(ResamplingFunction::Bicubic, WorkloadStrategy::PreferQuality),
        ),
        (
            "pic-scale-catmull-rom-prefer-quality",
            Backend::Pic(
                ResamplingFunction::CatmullRom,
                WorkloadStrategy::PreferQuality,
            ),
        ),
    ] {
        let mut blob = vec![0_u8; output_length];
        for case in &manifest.cases {
            let source = checked_slice(
                &sources,
                case.source.offset,
                case.source.byte_length,
                &case.id,
            )?;
            let output = resize_case(backend, source, case)?;
            if output.len() != case.pillow_image_rgb8.byte_length {
                return Err(format!(
                    "{}: backend {name} produced {} bytes, expected {}",
                    case.id,
                    output.len(),
                    case.pillow_image_rgb8.byte_length
                )
                .into());
            }
            let destination = checked_slice_mut(
                &mut blob,
                case.pillow_image_rgb8.offset,
                case.pillow_image_rgb8.byte_length,
                &case.id,
            )?;
            destination.copy_from_slice(&output);
        }
        let path = output_directory.join(format!("{name}.rgb8.bin"));
        fs::write(&path, blob)?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

fn authenticate_manifest(manifest: &Manifest) -> Result<(), Box<dyn std::error::Error>> {
    if manifest.schema_version != 2
        || manifest.contract_id != CONTRACT_ID
        || manifest.stage_id != STAGE_ID
        || manifest.holdout_seed != HOLDOUT_SEED
        || !manifest.candidate_blind
    {
        return Err("manifest identity does not match the frozen candidate-blind v2 corpus".into());
    }
    if manifest.cases.is_empty() {
        return Err("manifest contains no cases".into());
    }
    Ok(())
}

fn authenticate_case_layout(
    manifest: &Manifest,
    sources: &[u8],
) -> Result<usize, Box<dyn std::error::Error>> {
    let expected_output_length = manifest
        .artifacts
        .get("pillow-image-rgb8.bin")
        .ok_or("manifest does not describe pillow-image-rgb8.bin")?
        .byte_length;
    let mut source_end = 0;
    let mut output_end = 0;
    for case in &manifest.cases {
        if case.source.offset != source_end || case.pillow_image_rgb8.offset != output_end {
            return Err(format!("{}: case artifact offsets are not contiguous", case.id).into());
        }
        let source = checked_slice(
            sources,
            case.source.offset,
            case.source.byte_length,
            &case.id,
        )?;
        verify_digest(source, &case.source.sha256, &case.id)?;
        let source_row_bytes = usize::try_from(case.source.width)?
            .checked_mul(3)
            .ok_or("source row byte length overflow")?;
        let required_source_bytes = usize::try_from(case.source.height)?
            .saturating_sub(1)
            .checked_mul(usize::try_from(case.source.stride_bytes)?)
            .and_then(|value| value.checked_add(source_row_bytes))
            .ok_or("source byte length overflow")?;
        let full_strided_bytes = usize::try_from(case.source.height)?
            .checked_mul(usize::try_from(case.source.stride_bytes)?)
            .ok_or("full strided source byte length overflow")?;
        if case.source.byte_length < required_source_bytes
            || case.source.byte_length > full_strided_bytes
        {
            return Err(format!("{}: inconsistent strided source length", case.id).into());
        }
        let output_bytes = usize::try_from(case.destination.height)?
            .checked_mul(usize::try_from(case.destination.width)?)
            .and_then(|value| value.checked_mul(3))
            .ok_or("destination byte length overflow")?;
        if case.pillow_image_rgb8.byte_length != output_bytes
            || case.pillow_image_rgb8.shape
                != [
                    usize::try_from(case.destination.height)?,
                    usize::try_from(case.destination.width)?,
                    3,
                ]
        {
            return Err(format!("{}: inconsistent destination layout", case.id).into());
        }
        source_end = case
            .source
            .offset
            .checked_add(case.source.byte_length)
            .ok_or("source artifact offset overflow")?;
        output_end = case
            .pillow_image_rgb8
            .offset
            .checked_add(case.pillow_image_rgb8.byte_length)
            .ok_or("candidate artifact offset overflow")?;
    }
    if source_end != sources.len() || output_end != expected_output_length {
        return Err("manifest case slices do not exactly cover their artifacts".into());
    }
    Ok(expected_output_length)
}

fn resize_case(
    backend: Backend,
    source: &[u8],
    case: &Case,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    if [case.source.height, case.source.width] == [case.destination.height, case.destination.width]
    {
        return pack_rgb8(source, &case.source);
    }
    match backend {
        Backend::Scalar => {
            let plan = resize_plan(&case.destination)?;
            Ok(resize_image_rgb8(
                source,
                case.source.height,
                case.source.width,
                case.source.stride_bytes,
                &plan,
            )?)
        }
        Backend::FirCatmullRom => fir_resize(source, &case.source, &case.destination),
        Backend::Pic(filter, workload) => {
            pic_resize(source, &case.source, &case.destination, filter, workload)
        }
    }
}

fn resize_plan(destination: &Dimensions) -> Result<ImageGeometryPlan, Box<dyn std::error::Error>> {
    let rgb_row_stride_bytes = destination
        .width
        .checked_mul(3)
        .ok_or("destination RGB stride overflow")?;
    let rgb_capacity_bytes = destination
        .height
        .checked_mul(rgb_row_stride_bytes)
        .ok_or("destination RGB capacity overflow")?;
    // The scalar resize entry point consumes only destination dimensions and
    // the checked RGB layout. The other plan fields belong to later patching.
    Ok(ImageGeometryPlan {
        height: destination.height,
        width: destination.width,
        image_grid_thw: [0; 3],
        patch_rows: 0,
        placeholder_count: 0,
        rgb_row_stride_bytes,
        rgb_capacity_bytes,
        pixel_values_row_stride_bytes: 0,
        pixel_values_capacity_bytes: 0,
        image_grid_row_stride_bytes: 0,
        image_grid_capacity_bytes: 0,
    })
}

fn fir_resize(
    source: &[u8],
    dimensions: &Source,
    destination: &Dimensions,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // fast_image_resize's safe typed RGB API requires packed input.
    let packed = pack_rgb8(source, dimensions)?;
    let source_image = TypedImageRef::<U8x3>::from_buffer(
        u32::try_from(dimensions.width)?,
        u32::try_from(dimensions.height)?,
        &packed,
    )?;
    let mut output = vec![
        0_u8;
        usize::try_from(destination.width)?
            .checked_mul(usize::try_from(destination.height)?)
            .and_then(|value| value.checked_mul(3))
            .ok_or("destination capacity overflow")?
    ];
    let mut output_image = TypedImage::<U8x3>::from_buffer(
        u32::try_from(destination.width)?,
        u32::try_from(destination.height)?,
        &mut output,
    )?;
    Resizer::new().resize_typed(
        &source_image,
        &mut output_image,
        &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FirFilter::CatmullRom)),
    )?;
    drop(output_image);
    Ok(output)
}

fn pic_resize(
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

fn pack_rgb8(source: &[u8], dimensions: &Source) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let height = usize::try_from(dimensions.height)?;
    let stride = usize::try_from(dimensions.stride_bytes)?;
    let row_bytes = usize::try_from(dimensions.width)?
        .checked_mul(3)
        .ok_or("packed RGB row overflow")?;
    let mut packed = Vec::with_capacity(
        height
            .checked_mul(row_bytes)
            .ok_or("packed RGB capacity overflow")?,
    );
    for row in 0..height {
        let start = row
            .checked_mul(stride)
            .ok_or("source row offset overflow")?;
        packed.extend_from_slice(
            source
                .get(start..start + row_bytes)
                .ok_or("strided RGB source is too short")?,
        );
    }
    Ok(packed)
}

fn read_verified_artifact(
    directory: &std::path::Path,
    name: &str,
    manifest: &Manifest,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let expected = manifest
        .artifacts
        .get(name)
        .ok_or_else(|| format!("manifest does not describe {name}"))?;
    let data = fs::read(directory.join(name))?;
    if data.len() != expected.byte_length {
        return Err(format!("artifact length mismatch: {name}").into());
    }
    verify_digest(&data, &expected.sha256, name)?;
    Ok(data)
}

fn verify_digest(
    data: &[u8],
    expected: &str,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let actual = sha256(data);
    if actual != expected {
        return Err(
            format!("SHA-256 mismatch for {label}: expected {expected}, got {actual}").into(),
        );
    }
    Ok(())
}

fn sha256(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

fn checked_slice<'a>(
    data: &'a [u8],
    offset: usize,
    length: usize,
    label: &str,
) -> Result<&'a [u8], Box<dyn std::error::Error>> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| format!("{label}: artifact slice overflow"))?;
    data.get(offset..end)
        .ok_or_else(|| format!("{label}: artifact slice out of bounds").into())
}

fn checked_slice_mut<'a>(
    data: &'a mut [u8],
    offset: usize,
    length: usize,
    label: &str,
) -> Result<&'a mut [u8], Box<dyn std::error::Error>> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| format!("{label}: artifact slice overflow"))?;
    data.get_mut(offset..end)
        .ok_or_else(|| format!("{label}: artifact slice out of bounds").into())
}
