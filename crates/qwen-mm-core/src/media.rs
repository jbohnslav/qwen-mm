//! Bounded still-image decode, color conversion, raw RGB packing, and resize.

use std::{
    io::{BufReader, Cursor},
    mem,
};

use image::{DynamicImage, ImageError, ImageFormat as DecoderFormat, ImageReader, Limits};
use libjpeg_turbo_rs::{DecodeLimits, Decoder as JpegDecoder, JpegError, PixelFormat};

use crate::{
    error::{ErrorCategory, QwenError, Result},
    geometry::{ImageGeometryPlan, plan_image_geometry},
    limits::{ResourceLimits, checked_add, checked_mul},
    observability::{BufferClass, ObservationRecorder, ObservationScope},
    profile::VisualProfile,
    request::{ImageFormat, ImageInput, ImageOptions, Rgb8},
    resize::{resize_image_rgb8, resize_image_rgb8_observed},
};

const RGB_CHANNELS: u64 = 3;
const RGBA_CHANNELS: u64 = 4;

/// A packed HWC RGB8 still image after the frozen color and resize stages.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedRgbImage {
    /// Source height read from the encoded header or raw view.
    pub source_height: u64,
    /// Source width read from the encoded header or raw view.
    pub source_width: u64,
    /// Exact B3 geometry for the prepared raster.
    pub geometry: ImageGeometryPlan,
    /// Packed HWC RGB8 bytes with row stride `geometry.width * 3`.
    pub rgb: Vec<u8>,
}

/// Prepares one encoded or caller-owned RGB8 still image without Python.
///
/// The validation order is frozen: encoded-byte or raw checked-resource
/// preflight, bounded header probe, decoded edge/pixel and prepared/output
/// preflight, full decode, color conversion, then B5 resize. EXIF orientation
/// metadata is deliberately never queried or applied.
///
/// # Errors
///
/// Returns `unsupported_media` for an excluded encoded mode or animated WebP,
/// `media_decode` for malformed/truncated recognized media, `media_geometry`
/// for invalid raw views or resize geometry, `resource_limit` for configured
/// byte/pixel/edge/output excess, and `arithmetic_overflow` for checked size,
/// stride, or capacity overflow.
pub fn prepare_image_rgb8(
    input: ImageInput<'_>,
    visual: &VisualProfile,
    options: ImageOptions,
    limits: ResourceLimits,
) -> Result<PreparedRgbImage> {
    execute_image_plan(plan_image_rgb8(input, visual, options, limits)?)
}

/// A bounded, allocation-free image plan that has not performed full decode.
#[derive(Clone, Debug)]
pub(crate) struct ImagePreparationPlan<'a> {
    source: PlannedSource<'a>,
    geometry: Result<ImageGeometryPlan>,
    limits: ResourceLimits,
}

impl ImagePreparationPlan<'_> {
    pub(crate) fn geometry(&self) -> Result<ImageGeometryPlan> {
        self.geometry.clone()
    }
}

#[derive(Clone, Debug)]
enum PlannedSource<'a> {
    Encoded {
        data: &'a [u8],
        format: ImageFormat,
        header: Header,
    },
    Raw {
        raw: Rgb8<'a>,
        height: u64,
        width: u64,
        stride: u64,
        packed_stride: u64,
        required: u64,
    },
}

/// Performs byte/header/raw resource preflight and output planning without a
/// full codec decode or RGB allocation. Geometry errors are retained so the
/// composed processor can run every decode before selecting a later-stage
/// geometry failure.
pub(crate) fn plan_image_rgb8<'a>(
    input: ImageInput<'a>,
    visual: &VisualProfile,
    options: ImageOptions,
    limits: ResourceLimits,
) -> Result<ImagePreparationPlan<'a>> {
    match input {
        ImageInput::Encoded { data, format } => {
            check_encoded_bytes(data, limits)?;
            let header = probe_header(data, format)?;
            check_source_resources(header.height, header.width, limits)?;
            let geometry =
                plan_and_check_output(visual, header.height, header.width, options, limits);
            if let Err(error) = &geometry
                && matches!(
                    error.category(),
                    ErrorCategory::ResourceLimit | ErrorCategory::ArithmeticOverflow
                )
            {
                return Err(error.clone());
            }
            if header.animated {
                return Err(unsupported(
                    "animated media is outside the still-image contract",
                ));
            }
            if matches!(header.mode, SourceMode::LumaAlpha | SourceMode::Other) {
                return Err(
                    unsupported("encoded image mode is outside compatibility contract v1")
                        .with_context("mode", mode_name(header.mode)),
                );
            }
            Ok(ImagePreparationPlan {
                source: PlannedSource::Encoded {
                    data,
                    format,
                    header,
                },
                geometry,
                limits,
            })
        }
        ImageInput::Rgb8(raw) => {
            let height = u64::try_from(raw.height).map_err(|_| overflow("raw RGB height"))?;
            let width = u64::try_from(raw.width).map_err(|_| overflow("raw RGB width"))?;
            let stride =
                u64::try_from(raw.row_stride).map_err(|_| overflow("raw RGB row stride"))?;
            let pixels = checked_mul("decoded source pixels", height, width)?;
            let packed_stride = checked_mul("packed raw RGB stride", width, RGB_CHANNELS)?;
            let final_row =
                checked_mul("raw RGB final row offset", height.saturating_sub(1), stride)?;
            let required = checked_add("raw RGB readable byte span", final_row, packed_stride)?;
            check_source_resource_values(height, width, pixels, limits)?;
            let geometry = plan_and_check_output(visual, height, width, options, limits);
            if let Err(error) = &geometry
                && matches!(
                    error.category(),
                    ErrorCategory::ResourceLimit | ErrorCategory::ArithmeticOverflow
                )
            {
                return Err(error.clone());
            }
            Ok(ImagePreparationPlan {
                source: PlannedSource::Raw {
                    raw,
                    height,
                    width,
                    stride,
                    packed_stride,
                    required,
                },
                geometry,
                limits,
            })
        }
    }
}

/// Executes one image plan, performing full decode before deferred geometry.
pub(crate) fn execute_image_plan(plan: ImagePreparationPlan<'_>) -> Result<PreparedRgbImage> {
    execute_image_plan_internal(plan, None, ObservationScope::default())
}

/// Executes one image plan with bounded decode/color, resize, allocation, and
/// copy observations. The unobserved path calls the same implementation with
/// no recorder.
pub(crate) fn execute_image_plan_observed(
    plan: ImagePreparationPlan<'_>,
    recorder: &mut ObservationRecorder,
    scope: ObservationScope,
) -> Result<PreparedRgbImage> {
    execute_image_plan_internal(plan, Some(recorder), scope)
}

#[allow(clippy::too_many_lines)]
fn execute_image_plan_internal(
    plan: ImagePreparationPlan<'_>,
    mut recorder: Option<&mut ObservationRecorder>,
    scope: ObservationScope,
) -> Result<PreparedRgbImage> {
    match plan.source {
        PlannedSource::Encoded {
            data,
            format,
            header,
        } => {
            let decode_span = recorder.as_deref_mut().map(|recorder| {
                recorder.begin("native.media.decode_color", scope, data.len() as u64)
            });
            let decoded = match decode_to_rgb(data, format, header, plan.limits) {
                Ok(decoded) => {
                    if let Some(recorder) = recorder.as_deref_mut() {
                        if let Some(span) = decode_span {
                            recorder.finish_success(
                                span,
                                decoded.len() as u64,
                                &[header.height, header.width, RGB_CHANNELS],
                            );
                        }
                        recorder.record_allocation(
                            "decoded_rgb",
                            BufferClass::Transient,
                            scope,
                            vector_capacity_bytes(&decoded),
                        );
                    }
                    decoded
                }
                Err(error) => {
                    if let (Some(recorder), Some(span)) = (recorder.as_deref_mut(), decode_span) {
                        recorder.finish_error(span, &error);
                    }
                    return Err(error);
                }
            };
            let geometry = match plan.geometry {
                Ok(geometry) => geometry,
                Err(error) => {
                    if let Some(recorder) = recorder.as_deref_mut() {
                        recorder.release_transient(
                            "decoded_rgb",
                            scope,
                            vector_capacity_bytes(&decoded),
                        );
                    }
                    return Err(error);
                }
            };
            let decoded_stride =
                checked_mul("decoded packed RGB stride", header.width, RGB_CHANNELS)?;
            let resize_span = recorder
                .as_deref_mut()
                .map(|recorder| recorder.begin("native.media.resize", scope, decoded.len() as u64));
            let rgb = if header.height == geometry.height && header.width == geometry.width {
                // The decoder already returned exact packed RGB ownership.
                // Preserve that allocation as prepared RGB instead of copying
                // it through the borrowed public resize surface.
                if let Some(recorder) = recorder.as_deref_mut() {
                    if let Some(span) = resize_span {
                        recorder.finish_success(
                            span,
                            decoded.len() as u64,
                            &[geometry.height, geometry.width, RGB_CHANNELS],
                        );
                    }
                    recorder.rename_live_transient("decoded_rgb", "prepared_rgb", scope);
                }
                decoded
            } else {
                let resized = if let Some(recorder) = recorder.as_deref_mut() {
                    resize_image_rgb8_observed(
                        &decoded,
                        header.height,
                        header.width,
                        decoded_stride,
                        &geometry,
                        recorder,
                        scope,
                    )
                } else {
                    resize_image_rgb8(
                        &decoded,
                        header.height,
                        header.width,
                        decoded_stride,
                        &geometry,
                    )
                };
                match resized {
                    Ok(rgb) => {
                        if let Some(recorder) = recorder.as_deref_mut() {
                            if let Some(span) = resize_span {
                                recorder.finish_success(
                                    span,
                                    rgb.len() as u64,
                                    &[geometry.height, geometry.width, RGB_CHANNELS],
                                );
                            }
                            recorder.release_transient(
                                "decoded_rgb",
                                scope,
                                vector_capacity_bytes(&decoded),
                            );
                        }
                        rgb
                    }
                    Err(error) => {
                        if let Some(recorder) = recorder.as_deref_mut() {
                            if let Some(span) = resize_span {
                                recorder.finish_error(span, &error);
                            }
                            recorder.release_transient(
                                "decoded_rgb",
                                scope,
                                vector_capacity_bytes(&decoded),
                            );
                        }
                        return Err(error);
                    }
                }
            };
            if u64::try_from(rgb.len()).map_err(|_| overflow("prepared RGB length"))?
                != geometry.rgb_capacity_bytes
            {
                return Err(invariant(
                    "prepared RGB capacity disagrees with its geometry",
                ));
            }
            Ok(PreparedRgbImage {
                source_height: header.height,
                source_width: header.width,
                geometry,
                rgb,
            })
        }
        PlannedSource::Raw {
            raw,
            height,
            width,
            stride,
            packed_stride,
            required,
        } => {
            let actual =
                u64::try_from(raw.data.len()).map_err(|_| overflow("raw RGB buffer length"))?;
            let decode_span = recorder
                .as_deref_mut()
                .map(|recorder| recorder.begin("native.media.decode_color", scope, required));
            if height == 0 || width == 0 {
                let error = geometry_error("raw RGB dimensions must be non-zero");
                if let (Some(recorder), Some(span)) = (recorder.as_deref_mut(), decode_span) {
                    recorder.finish_error(span, &error);
                }
                return Err(error);
            }
            if stride < packed_stride {
                let error = geometry_error("raw RGB stride is smaller than packed width")
                    .with_context("stride", stride)
                    .with_context("packed_stride", packed_stride);
                if let (Some(recorder), Some(span)) = (recorder.as_deref_mut(), decode_span) {
                    recorder.finish_error(span, &error);
                }
                return Err(error);
            }
            if actual < required {
                let error =
                    geometry_error("raw RGB buffer is shorter than its dimensions and stride")
                        .with_context("actual_bytes", actual)
                        .with_context("required_bytes", required);
                if let (Some(recorder), Some(span)) = (recorder.as_deref_mut(), decode_span) {
                    recorder.finish_error(span, &error);
                }
                return Err(error);
            }
            if let (Some(recorder), Some(span)) = (recorder.as_deref_mut(), decode_span) {
                recorder.finish_success(span, 0, &[height, width, RGB_CHANNELS]);
            }
            let geometry = plan.geometry?;
            let resize_span = recorder
                .as_deref_mut()
                .map(|recorder| recorder.begin("native.media.resize", scope, required));
            let resized = if let Some(recorder) = recorder.as_deref_mut() {
                resize_image_rgb8_observed(
                    raw.data, height, width, stride, &geometry, recorder, scope,
                )
            } else {
                resize_image_rgb8(raw.data, height, width, stride, &geometry)
            };
            let rgb = match resized {
                Ok(rgb) => {
                    if let Some(recorder) = recorder.as_deref_mut()
                        && let Some(span) = resize_span
                    {
                        recorder.finish_success(
                            span,
                            rgb.len() as u64,
                            &[geometry.height, geometry.width, RGB_CHANNELS],
                        );
                    }
                    rgb
                }
                Err(error) => {
                    if let (Some(recorder), Some(span)) = (recorder, resize_span) {
                        recorder.finish_error(span, &error);
                    }
                    return Err(error);
                }
            };
            Ok(PreparedRgbImage {
                source_height: height,
                source_width: width,
                geometry,
                rgb,
            })
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SourceMode {
    Luma,
    Rgb,
    Rgba,
    Cmyk,
    LumaAlpha,
    Other,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Header {
    width: u64,
    height: u64,
    mode: SourceMode,
    animated: bool,
}

fn check_encoded_bytes(data: &[u8], limits: ResourceLimits) -> Result<()> {
    let actual = u64::try_from(data.len()).map_err(|_| overflow("encoded image byte length"))?;
    check_limit(
        "encoded_bytes_per_item",
        actual,
        limits.encoded_bytes_per_item(),
    )?;
    check_limit(
        "encoded_bytes_per_batch",
        actual,
        limits.encoded_bytes_per_batch(),
    )
}

#[allow(clippy::ptr_arg)] // Vec capacity, rather than slice length, is the allocation metric.
fn vector_capacity_bytes<T>(values: &Vec<T>) -> u64 {
    let capacity = u64::try_from(values.capacity()).unwrap_or(u64::MAX);
    let element_bytes = u64::try_from(mem::size_of::<T>()).unwrap_or(u64::MAX);
    capacity.saturating_mul(element_bytes)
}

fn check_source_resources(height: u64, width: u64, limits: ResourceLimits) -> Result<()> {
    let pixels = checked_mul("decoded source pixels", height, width)?;
    let _decoder_capacity =
        checked_mul("worst-case decoded color capacity", pixels, RGBA_CHANNELS)?;
    check_source_resource_values(height, width, pixels, limits)
}

fn check_source_resource_values(
    height: u64,
    width: u64,
    pixels: u64,
    limits: ResourceLimits,
) -> Result<()> {
    check_limit(
        "decoded_source_pixels",
        pixels,
        limits.decoded_pixels_per_image_or_frame(),
    )?;
    check_limit("decoded_edge_length", height, limits.decoded_edge_length())?;
    check_limit("decoded_edge_length", width, limits.decoded_edge_length())
}

fn plan_and_check_output(
    visual: &VisualProfile,
    height: u64,
    width: u64,
    options: ImageOptions,
    limits: ResourceLimits,
) -> Result<ImageGeometryPlan> {
    let geometry = plan_image_geometry(visual, height, width, options, limits)?;
    limits.check_materialized_output_bytes(geometry.rgb_capacity_bytes)?;
    usize::try_from(geometry.rgb_capacity_bytes)
        .map_err(|_| overflow("prepared RGB capacity does not fit usize"))?;
    Ok(geometry)
}

fn probe_header(data: &[u8], format: ImageFormat) -> Result<Header> {
    if let Some(recognized) = recognized_format(data) {
        match recognized {
            RecognizedFormat::Supported(actual) if actual != format => {
                return Err(
                    decode_error("encoded signature disagrees with the declared format")
                        .with_context("declared", format_name(format))
                        .with_context("actual", format_name(actual)),
                );
            }
            RecognizedFormat::Unsupported(name) => {
                return Err(
                    unsupported("encoded codec is outside compatibility contract v1")
                        .with_context("codec", name),
                );
            }
            RecognizedFormat::Supported(_) => {}
        }
    }
    match format {
        ImageFormat::Jpeg => probe_jpeg(data),
        ImageFormat::Png => probe_png(data),
        ImageFormat::WebP => probe_webp(data),
    }
}

fn probe_jpeg(data: &[u8]) -> Result<Header> {
    if !data.starts_with(&[0xff, 0xd8]) {
        return Err(decode_error("JPEG signature is missing"));
    }
    let mut offset = 2_usize;
    while offset < data.len() {
        while data.get(offset) == Some(&0xff) {
            offset += 1;
        }
        let marker = *data
            .get(offset)
            .ok_or_else(|| decode_error("JPEG marker is truncated"))?;
        offset += 1;
        if marker == 0xd9 || marker == 0xda {
            break;
        }
        if marker == 0x00 || marker == 0x01 || (0xd0..=0xd7).contains(&marker) {
            continue;
        }
        let length = read_be_u16(data, offset, "JPEG segment length")? as usize;
        if length < 2 {
            return Err(decode_error("JPEG segment length is invalid"));
        }
        let segment_end = offset
            .checked_add(length)
            .ok_or_else(|| overflow("JPEG segment offset"))?;
        if segment_end > data.len() {
            return Err(decode_error("JPEG segment is truncated"));
        }
        if is_start_of_frame(marker) {
            if length < 8 {
                return Err(decode_error("JPEG start-of-frame segment is truncated"));
            }
            let precision = data[offset + 2];
            let height = u64::from(read_be_u16(data, offset + 3, "JPEG height")?);
            let width = u64::from(read_be_u16(data, offset + 5, "JPEG width")?);
            let components = data[offset + 7];
            let expected_length = usize::from(components)
                .checked_mul(3)
                .and_then(|value| value.checked_add(8))
                .ok_or_else(|| overflow("JPEG start-of-frame length"))?;
            if components == 0
                || length != expected_length
                || width == 0
                || height == 0
                || !jpeg_precision_is_legal(marker, precision)
                || !jpeg_component_records_are_legal(
                    &data[offset + 8..segment_end],
                    is_lossless_start_of_frame(marker),
                )
            {
                return Err(decode_error("JPEG start-of-frame fields are malformed"));
            }
            let mode = match (precision, components) {
                (8, 1) => SourceMode::Luma,
                (8, 3) => SourceMode::Rgb,
                (8, 4) => SourceMode::Cmyk,
                _ => SourceMode::Other,
            };
            return Ok(Header {
                width,
                height,
                mode,
                animated: false,
            });
        }
        offset = segment_end;
    }
    Err(decode_error("JPEG start-of-frame marker was not found"))
}

fn is_start_of_frame(marker: u8) -> bool {
    matches!(
        marker,
        0xc0 | 0xc1 | 0xc2 | 0xc3 | 0xc5 | 0xc6 | 0xc7 | 0xc9 | 0xca | 0xcb | 0xcd | 0xce | 0xcf
    )
}

fn is_lossless_start_of_frame(marker: u8) -> bool {
    matches!(marker, 0xc3 | 0xc7 | 0xcb | 0xcf)
}

fn jpeg_precision_is_legal(marker: u8, precision: u8) -> bool {
    if marker == 0xc0 {
        precision == 8
    } else if is_lossless_start_of_frame(marker) {
        (2..=16).contains(&precision)
    } else {
        matches!(precision, 8 | 12)
    }
}

fn jpeg_component_records_are_legal(records: &[u8], lossless: bool) -> bool {
    let mut identifiers = [false; 256];
    for record in records.chunks_exact(3) {
        let identifier = usize::from(record[0]);
        let horizontal_sampling = record[1] >> 4;
        let vertical_sampling = record[1] & 0x0f;
        let table = record[2];
        if identifiers[identifier]
            || !(1..=4).contains(&horizontal_sampling)
            || !(1..=4).contains(&vertical_sampling)
            || if lossless { table != 0 } else { table > 3 }
        {
            return false;
        }
        identifiers[identifier] = true;
    }
    true
}

fn probe_png(data: &[u8]) -> Result<Header> {
    const SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
    if !data.starts_with(SIGNATURE) {
        return Err(decode_error("PNG signature is missing"));
    }
    if data.len() < 33 || &data[12..16] != b"IHDR" || read_be_u32(data, 8, "PNG IHDR length")? != 13
    {
        return Err(decode_error("PNG IHDR is missing or truncated"));
    }
    let width = u64::from(read_be_u32(data, 16, "PNG width")?);
    let height = u64::from(read_be_u32(data, 20, "PNG height")?);
    let bit_depth = data[24];
    let color_type = data[25];
    let compression_method = data[26];
    let filter_method = data[27];
    let interlace_method = data[28];
    if width == 0
        || height == 0
        || !png_ihdr_is_legal(bit_depth, color_type)
        || compression_method != 0
        || filter_method != 0
        || interlace_method > 1
    {
        return Err(decode_error("PNG IHDR contains an invalid encoding tuple")
            .with_context("bit_depth", u64::from(bit_depth))
            .with_context("color_type", u64::from(color_type))
            .with_context("compression_method", u64::from(compression_method))
            .with_context("filter_method", u64::from(filter_method))
            .with_context("interlace_method", u64::from(interlace_method)));
    }
    let ihdr_crc = read_be_u32(data, 29, "PNG IHDR CRC")?;
    if ihdr_crc != png_crc32(&data[12..29]) {
        return Err(decode_error("PNG IHDR CRC is invalid"));
    }
    let mode = match (bit_depth, color_type) {
        (8, 0) => SourceMode::Luma,
        (8, 2) => SourceMode::Rgb,
        (8, 4) => SourceMode::LumaAlpha,
        (8, 6) => SourceMode::Rgba,
        _ => SourceMode::Other,
    };
    Ok(Header {
        width,
        height,
        mode,
        animated: png_has_animation_control(data)?,
    })
}

fn png_ihdr_is_legal(bit_depth: u8, color_type: u8) -> bool {
    match color_type {
        0 => matches!(bit_depth, 1 | 2 | 4 | 8 | 16),
        2 | 4 | 6 => matches!(bit_depth, 8 | 16),
        3 => matches!(bit_depth, 1 | 2 | 4 | 8),
        _ => false,
    }
}

fn png_crc32(data: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = 0_u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

fn png_has_animation_control(data: &[u8]) -> Result<bool> {
    let mut offset = 8_usize;
    while offset < data.len() {
        let length = usize::try_from(read_be_u32(data, offset, "PNG chunk length")?)
            .map_err(|_| overflow("PNG chunk length"))?;
        let kind_start = offset
            .checked_add(4)
            .ok_or_else(|| overflow("PNG chunk type offset"))?;
        let payload_start = kind_start
            .checked_add(4)
            .ok_or_else(|| overflow("PNG chunk payload offset"))?;
        let next = payload_start
            .checked_add(length)
            .and_then(|value| value.checked_add(4))
            .ok_or_else(|| overflow("PNG chunk extent"))?;
        if next > data.len() {
            return Err(decode_error("PNG chunk is truncated"));
        }
        let kind = &data[kind_start..payload_start];
        let payload_end = next - 4;
        let expected_crc = read_be_u32(data, payload_end, "PNG chunk CRC")?;
        if expected_crc != png_crc32(&data[kind_start..payload_end]) {
            return Err(decode_error("PNG chunk CRC is invalid"));
        }
        if kind == b"acTL" {
            if length != 8 || read_be_u32(data, payload_start, "APNG frame count")? == 0 {
                return Err(decode_error("APNG animation control is malformed"));
            }
            return Ok(true);
        }
        if kind == b"IDAT" || kind == b"IEND" {
            return Ok(false);
        }
        offset = next;
    }
    Ok(false)
}

fn probe_webp(data: &[u8]) -> Result<Header> {
    if data.len() < 20 || &data[..4] != b"RIFF" || &data[8..12] != b"WEBP" {
        return Err(decode_error("WebP RIFF signature is missing or truncated"));
    }
    let declared_size = usize::try_from(read_le_u32(data, 4, "WebP RIFF length")?)
        .map_err(|_| overflow("WebP RIFF length"))?
        .checked_add(8)
        .ok_or_else(|| overflow("WebP RIFF extent"))?;
    if declared_size > data.len() {
        return Err(decode_error("WebP RIFF payload is truncated"));
    }
    let chunk = &data[12..16];
    let chunk_length = usize::try_from(read_le_u32(data, 16, "WebP chunk length")?)
        .map_err(|_| overflow("WebP chunk length"))?;
    let end = 20_usize
        .checked_add(chunk_length)
        .ok_or_else(|| overflow("WebP chunk extent"))?;
    let padded_end = end
        .checked_add(chunk_length & 1)
        .ok_or_else(|| overflow("WebP padded chunk extent"))?;
    if padded_end > declared_size || padded_end > data.len() {
        return Err(decode_error("WebP image header chunk is truncated"));
    }
    let payload = &data[20..end];
    match chunk {
        b"VP8 " => {
            if payload.len() < 10 || payload[3..6] != [0x9d, 0x01, 0x2a] {
                return Err(decode_error("lossy WebP frame header is malformed"));
            }
            let width = u64::from(u16::from_le_bytes([payload[6], payload[7]]) & 0x3fff);
            let height = u64::from(u16::from_le_bytes([payload[8], payload[9]]) & 0x3fff);
            if width == 0 || height == 0 {
                return Err(decode_error("lossy WebP frame dimensions are zero"));
            }
            Ok(Header {
                width,
                height,
                mode: SourceMode::Rgb,
                animated: false,
            })
        }
        b"VP8L" => {
            if payload.len() < 5 || payload[0] != 0x2f {
                return Err(decode_error("lossless WebP frame header is malformed"));
            }
            let bits = u32::from_le_bytes([payload[1], payload[2], payload[3], payload[4]]);
            if bits >> 29 != 0 {
                return Err(decode_error("lossless WebP version bits are invalid"));
            }
            let width = u64::from((bits & 0x3fff) + 1);
            let height = u64::from(((bits >> 14) & 0x3fff) + 1);
            let alpha = ((bits >> 28) & 1) != 0;
            Ok(Header {
                width,
                height,
                mode: if alpha {
                    SourceMode::Rgba
                } else {
                    SourceMode::Rgb
                },
                animated: false,
            })
        }
        b"VP8X" => {
            if payload.len() != 10 {
                return Err(decode_error("extended WebP frame header length is invalid"));
            }
            let flags = payload[0];
            if flags & 0xc1 != 0 || payload[1..4] != [0, 0, 0] {
                return Err(decode_error("extended WebP reserved fields are non-zero"));
            }
            let width = u64::from(read_le_u24(payload, 4)? + 1);
            let height = u64::from(read_le_u24(payload, 7)? + 1);
            Ok(Header {
                width,
                height,
                mode: if flags & 0x10 != 0 {
                    SourceMode::Rgba
                } else {
                    SourceMode::Rgb
                },
                animated: flags & 0x02 != 0,
            })
        }
        _ => Err(decode_error("WebP image header chunk is unsupported")),
    }
}

fn decode_to_rgb(
    data: &[u8],
    format: ImageFormat,
    header: Header,
    limits: ResourceLimits,
) -> Result<Vec<u8>> {
    if format == ImageFormat::Jpeg {
        return decode_jpeg_to_rgb(data, header, limits);
    }
    let pixels = checked_mul("decoded source pixels", header.height, header.width)?;
    let maximum_decoder_bytes = checked_mul("decoder output capacity", pixels, RGBA_CHANNELS)?;
    let max_edge = u32::try_from(limits.decoded_edge_length())
        .map_err(|_| overflow("decoded edge limit does not fit decoder"))?;
    let mut reader =
        ImageReader::with_format(BufReader::new(Cursor::new(data)), decoder_format(format));
    let mut decoder_limits = Limits::default();
    decoder_limits.max_image_width = Some(max_edge);
    decoder_limits.max_image_height = Some(max_edge);
    // Best-effort defense in depth. The strict checked preflight above is the
    // contract boundary; image's allocation limit is not relied on.
    decoder_limits.max_alloc = Some(maximum_decoder_bytes);
    reader.limits(decoder_limits);
    let image = reader.decode().map_err(map_decoder_error)?;
    if u64::from(image.width()) != header.width || u64::from(image.height()) != header.height {
        return Err(decode_error(
            "decoded dimensions disagree with the probed header",
        ));
    }
    dynamic_to_rgb(image, header.mode)
}

fn decode_jpeg_to_rgb(data: &[u8], header: Header, limits: ResourceLimits) -> Result<Vec<u8>> {
    if !jpeg_has_end_of_image(data) {
        return Err(decode_error("JPEG end-of-image marker is missing"));
    }
    // Bound codec scratch space from the already-enforced decoded-pixel cap.
    // Do not reuse the prepared-output cap: it intentionally permits a small
    // final image after geometry downsamples a larger, still-admissible source.
    let maximum_decoder_memory = checked_mul(
        "JPEG decoder memory limit",
        limits.decoded_pixels_per_image_or_frame(),
        32,
    )?;
    let decode_limits = DecodeLimits {
        max_width: usize::try_from(limits.decoded_edge_length())
            .map_err(|_| overflow("JPEG edge limit does not fit usize"))?,
        max_height: usize::try_from(limits.decoded_edge_length())
            .map_err(|_| overflow("JPEG edge limit does not fit usize"))?,
        max_pixels: limits.decoded_pixels_per_image_or_frame(),
        max_scans: 8_192,
        max_memory: Some(maximum_decoder_memory),
    };
    let mut decoder = JpegDecoder::new_with_limits(data, decode_limits).map_err(map_jpeg_error)?;
    decoder.set_lenient(false);
    decoder.set_stop_on_warning(true);
    decoder.set_output_format(match header.mode {
        SourceMode::Luma => PixelFormat::Grayscale,
        SourceMode::Rgb | SourceMode::Cmyk => PixelFormat::Rgb,
        _ => {
            return Err(unsupported(
                "JPEG mode is outside compatibility contract v1",
            ));
        }
    });
    let image = decoder.decode_image().map_err(map_jpeg_error)?;
    if u64::try_from(image.width).map_err(|_| overflow("decoded JPEG width"))? != header.width
        || u64::try_from(image.height).map_err(|_| overflow("decoded JPEG height"))?
            != header.height
    {
        return Err(decode_error(
            "decoded JPEG dimensions disagree with the probed header",
        ));
    }
    match (header.mode, image.pixel_format) {
        (SourceMode::Luma, PixelFormat::Grayscale) => {
            let capacity = image
                .data
                .len()
                .checked_mul(3)
                .ok_or_else(|| overflow("JPEG luma RGB capacity"))?;
            let mut rgb = Vec::new();
            rgb.try_reserve_exact(capacity).map_err(|error| {
                resource("unable to reserve JPEG luma RGB conversion")
                    .with_context("bytes", capacity)
                    .with_context("detail", error.to_string())
            })?;
            for value in image.data {
                rgb.extend_from_slice(&[value, value, value]);
            }
            Ok(rgb)
        }
        (SourceMode::Rgb | SourceMode::Cmyk, PixelFormat::Rgb) => Ok(image.data),
        _ => Err(decode_error(
            "decoded JPEG layout disagrees with the accepted encoded mode",
        )),
    }
}

fn jpeg_has_end_of_image(data: &[u8]) -> bool {
    if !data.starts_with(&[0xff, 0xd8]) {
        return false;
    }
    let mut offset = 2_usize;
    let mut pending = None;
    loop {
        let marker = if let Some(marker) = pending.take() {
            marker
        } else {
            if data.get(offset) != Some(&0xff) {
                return false;
            }
            while data.get(offset) == Some(&0xff) {
                offset += 1;
            }
            let Some(&marker) = data.get(offset) else {
                return false;
            };
            offset += 1;
            marker
        };

        if marker == 0xd9 {
            return true;
        }
        if marker == 0x00 || marker == 0x01 || (0xd0..=0xd7).contains(&marker) {
            continue;
        }
        let Some(length_bytes) = data.get(offset..offset.saturating_add(2)) else {
            return false;
        };
        let length = usize::from(u16::from_be_bytes([length_bytes[0], length_bytes[1]]));
        if length < 2 {
            return false;
        }
        let Some(segment_end) = offset.checked_add(length) else {
            return false;
        };
        if segment_end > data.len() {
            return false;
        }
        offset = segment_end;
        if marker != 0xda {
            continue;
        }

        // Scan entropy until an unstuffed, non-restart structural marker.
        loop {
            while data.get(offset).is_some_and(|&byte| byte != 0xff) {
                offset += 1;
            }
            if offset >= data.len() {
                return false;
            }
            while data.get(offset) == Some(&0xff) {
                offset += 1;
            }
            let Some(&next_marker) = data.get(offset) else {
                return false;
            };
            offset += 1;
            if next_marker == 0x00 || (0xd0..=0xd7).contains(&next_marker) {
                continue;
            }
            pending = Some(next_marker);
            break;
        }
    }
}

fn dynamic_to_rgb(image: DynamicImage, mode: SourceMode) -> Result<Vec<u8>> {
    match (mode, image) {
        (SourceMode::Luma, DynamicImage::ImageLuma8(buffer)) => {
            let luma = buffer.into_raw();
            let capacity = luma
                .len()
                .checked_mul(3)
                .ok_or_else(|| overflow("luma RGB conversion capacity"))?;
            let mut rgb = Vec::new();
            rgb.try_reserve_exact(capacity).map_err(|error| {
                resource("unable to reserve luma RGB conversion")
                    .with_context("bytes", capacity)
                    .with_context("detail", error.to_string())
            })?;
            for value in luma {
                rgb.extend_from_slice(&[value, value, value]);
            }
            Ok(rgb)
        }
        (SourceMode::Luma, DynamicImage::ImageLumaA8(buffer)) => {
            let luma_alpha = buffer.into_raw();
            let pixels = luma_alpha.len() / 2;
            let capacity = pixels
                .checked_mul(3)
                .ok_or_else(|| overflow("transparent luma RGB conversion capacity"))?;
            let mut rgb = Vec::new();
            rgb.try_reserve_exact(capacity).map_err(|error| {
                resource("unable to reserve transparent luma RGB conversion")
                    .with_context("bytes", capacity)
                    .with_context("detail", error.to_string())
            })?;
            for pixel in luma_alpha.chunks_exact(2) {
                rgb.extend_from_slice(&[pixel[0], pixel[0], pixel[0]]);
            }
            Ok(rgb)
        }
        (SourceMode::Rgb | SourceMode::Cmyk, DynamicImage::ImageRgb8(buffer)) => {
            Ok(buffer.into_raw())
        }
        (SourceMode::Rgb, DynamicImage::ImageRgba8(buffer)) => {
            let rgba = buffer.into_raw();
            let pixels = rgba.len() / 4;
            let capacity = pixels
                .checked_mul(3)
                .ok_or_else(|| overflow("transparent RGB conversion capacity"))?;
            let mut rgb = Vec::new();
            rgb.try_reserve_exact(capacity).map_err(|error| {
                resource("unable to reserve transparent RGB conversion")
                    .with_context("bytes", capacity)
                    .with_context("detail", error.to_string())
            })?;
            for pixel in rgba.chunks_exact(4) {
                rgb.extend_from_slice(&pixel[..3]);
            }
            Ok(rgb)
        }
        (SourceMode::Rgba, DynamicImage::ImageRgba8(buffer)) => {
            let rgba = buffer.into_raw();
            let pixels = rgba.len() / 4;
            let capacity = pixels
                .checked_mul(3)
                .ok_or_else(|| overflow("RGBA composite capacity"))?;
            let mut rgb = Vec::new();
            rgb.try_reserve_exact(capacity).map_err(|error| {
                resource("unable to reserve RGBA composite")
                    .with_context("bytes", capacity)
                    .with_context("detail", error.to_string())
            })?;
            for pixel in rgba.chunks_exact(4) {
                let alpha = u32::from(pixel[3]);
                for &value in &pixel[..3] {
                    let blended = u32::from(value) * alpha + 255 * (255 - alpha);
                    rgb.push(u8::try_from((blended + 127) / 255).expect("blend is RGB8"));
                }
            }
            Ok(rgb)
        }
        _ => Err(decode_error(
            "decoded color layout disagrees with the accepted encoded mode",
        )),
    }
}

fn decoder_format(format: ImageFormat) -> DecoderFormat {
    match format {
        ImageFormat::Jpeg => DecoderFormat::Jpeg,
        ImageFormat::Png => DecoderFormat::Png,
        ImageFormat::WebP => DecoderFormat::WebP,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RecognizedFormat {
    Supported(ImageFormat),
    Unsupported(&'static str),
}

fn recognized_format(data: &[u8]) -> Option<RecognizedFormat> {
    if data.starts_with(&[0xff, 0xd8]) {
        Some(RecognizedFormat::Supported(ImageFormat::Jpeg))
    } else if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some(RecognizedFormat::Supported(ImageFormat::Png))
    } else if data.len() >= 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WEBP" {
        Some(RecognizedFormat::Supported(ImageFormat::WebP))
    } else if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
        Some(RecognizedFormat::Unsupported("gif"))
    } else if data.starts_with(b"BM") {
        Some(RecognizedFormat::Unsupported("bmp"))
    } else if data.starts_with(b"II*\0") || data.starts_with(b"MM\0*") {
        Some(RecognizedFormat::Unsupported("tiff"))
    } else if data.len() >= 12 && &data[4..12] == b"ftypavif" {
        Some(RecognizedFormat::Unsupported("avif"))
    } else {
        None
    }
}

fn format_name(format: ImageFormat) -> &'static str {
    match format {
        ImageFormat::Jpeg => "jpeg",
        ImageFormat::Png => "png",
        ImageFormat::WebP => "webp",
    }
}

fn map_decoder_error(error: ImageError) -> QwenError {
    match error {
        ImageError::Unsupported(error) => unsupported("encoded image feature is unsupported")
            .with_context("detail", error.to_string()),
        ImageError::Limits(error) => resource("decoder allocation limit was exceeded")
            .with_context("detail", error.to_string()),
        other => decode_error("recognized encoded image could not be decoded")
            .with_context("detail", other.to_string()),
    }
}

fn map_jpeg_error(error: JpegError) -> QwenError {
    match error {
        JpegError::LimitExceeded { .. } => resource("JPEG decoder resource limit was exceeded")
            .with_context("detail", error.to_string()),
        JpegError::Unsupported(_) => {
            unsupported("JPEG feature is unsupported").with_context("detail", error.to_string())
        }
        other => decode_error("recognized JPEG could not be decoded")
            .with_context("detail", other.to_string()),
    }
}

fn read_be_u16(data: &[u8], offset: usize, label: &'static str) -> Result<u16> {
    let bytes = data
        .get(offset..offset + 2)
        .ok_or_else(|| decode_error(format!("{label} is truncated")))?;
    Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
}

fn read_be_u32(data: &[u8], offset: usize, label: &'static str) -> Result<u32> {
    let bytes = data
        .get(offset..offset + 4)
        .ok_or_else(|| decode_error(format!("{label} is truncated")))?;
    Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_le_u32(data: &[u8], offset: usize, label: &'static str) -> Result<u32> {
    let bytes = data
        .get(offset..offset + 4)
        .ok_or_else(|| decode_error(format!("{label} is truncated")))?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_le_u24(data: &[u8], offset: usize) -> Result<u32> {
    let bytes = data
        .get(offset..offset + 3)
        .ok_or_else(|| decode_error("WebP canvas dimensions are truncated"))?;
    Ok(u32::from(bytes[0]) | (u32::from(bytes[1]) << 8) | (u32::from(bytes[2]) << 16))
}

fn mode_name(mode: SourceMode) -> &'static str {
    match mode {
        SourceMode::Luma => "L",
        SourceMode::Rgb => "RGB",
        SourceMode::Rgba => "RGBA",
        SourceMode::Cmyk => "CMYK",
        SourceMode::LumaAlpha => "LA",
        SourceMode::Other => "other",
    }
}

fn check_limit(name: &'static str, actual: u64, limit: u64) -> Result<()> {
    if actual <= limit {
        return Ok(());
    }
    Err(resource("resource limit exceeded")
        .with_context("resource", name)
        .with_context("actual", actual)
        .with_context("limit", limit))
}

fn unsupported(message: impl Into<String>) -> QwenError {
    QwenError::new(ErrorCategory::UnsupportedMedia, message)
}

fn decode_error(message: impl Into<String>) -> QwenError {
    QwenError::new(ErrorCategory::MediaDecode, message)
}

fn geometry_error(message: impl Into<String>) -> QwenError {
    QwenError::new(ErrorCategory::MediaGeometry, message)
}

fn resource(message: impl Into<String>) -> QwenError {
    QwenError::new(ErrorCategory::ResourceLimit, message)
}

fn overflow(message: impl Into<String>) -> QwenError {
    QwenError::new(ErrorCategory::ArithmeticOverflow, message)
}

fn invariant(message: impl Into<String>) -> QwenError {
    QwenError::new(ErrorCategory::InternalInvariant, message)
}

#[cfg(test)]
mod tests {
    use image::{ExtendedColorType, ImageEncoder, codecs::png::PngEncoder};

    use super::{
        execute_image_plan_observed, plan_image_rgb8, png_crc32, prepare_image_rgb8, probe_webp,
    };
    use crate::{
        error::ErrorCategory,
        limits::{LimitOverrides, ResourceLimits},
        observability::{ObservationRecorder, ObservationScope, StageOutcome},
        profile::{ProfileAlias, ProfileRegistry, VisualProfile},
        request::{ImageFormat, ImageInput, ImageOptions, Rgb8},
    };

    fn visual() -> VisualProfile {
        ProfileRegistry::bundled()
            .expect("bundled profiles")
            .get(ProfileAlias::Qwen3Vl8b)
            .visual
            .clone()
    }

    #[test]
    fn raw_packed_and_padded_no_op_views_are_byte_exact() {
        let mut packed = vec![0_u8; 64 * 64 * 3];
        for (index, value) in packed.iter_mut().enumerate() {
            *value = u8::try_from((index * 37 + 11) % 256).expect("bounded");
        }
        let mut padded = vec![0xa5_u8; 64 * 200];
        for row in 0..64 {
            padded[row * 200..row * 200 + 192].copy_from_slice(&packed[row * 192..(row + 1) * 192]);
        }
        let exact_extent = &padded[..63 * 200 + 192];
        for (data, stride) in [(&packed[..], 192), (&padded[..], 200), (exact_extent, 200)] {
            let result = prepare_image_rgb8(
                ImageInput::Rgb8(Rgb8 {
                    data,
                    height: 64,
                    width: 64,
                    row_stride: stride,
                }),
                &visual(),
                ImageOptions::default(),
                ResourceLimits::default(),
            )
            .expect("valid raw view");
            assert_eq!(result.rgb, packed);
        }
    }

    #[test]
    fn observed_encoded_no_op_reuses_decoded_ownership_without_a_copy() {
        const EDGE: u32 = 64;
        let rgb = (0_u8..=u8::MAX)
            .cycle()
            .take(EDGE as usize * EDGE as usize * 3)
            .collect::<Vec<_>>();
        let mut encoded = Vec::new();
        PngEncoder::new(&mut encoded)
            .write_image(&rgb, EDGE, EDGE, ExtendedColorType::Rgb8)
            .expect("encode PNG");
        let plan = plan_image_rgb8(
            ImageInput::Encoded {
                data: &encoded,
                format: ImageFormat::Png,
            },
            &visual(),
            ImageOptions::default(),
            ResourceLimits::default(),
        )
        .expect("encoded plan");
        let scope = ObservationScope::media_at(0, 0, 0, 0, 0);
        let mut recorder = ObservationRecorder::new(16);
        let prepared = execute_image_plan_observed(plan, &mut recorder, scope)
            .expect("observed encoded no-op");
        assert_eq!(prepared.rgb, rgb);

        let report = recorder.report();
        assert_eq!(report.allocations.allocation_count, 1);
        assert_eq!(report.allocations.copy_count, 0);
        assert_eq!(report.allocations.copied_bytes, 0);
        assert!(report.copies.is_empty());
        assert_eq!(report.buffers.len(), 1);
        assert_eq!(report.buffers[0].name, "prepared_rgb");
        assert_eq!(report.buffers[0].scope, scope);
        assert!(report.buffers[0].released_at_ns.is_none());
        assert!(
            report
                .buffers
                .iter()
                .all(|buffer| buffer.name != "resize.noop.source_copy")
        );
    }

    fn png_header(width: u32, height: u32, bit_depth: u8, color_type: u8) -> Vec<u8> {
        let mut data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR".to_vec();
        data.extend_from_slice(&width.to_be_bytes());
        data.extend_from_slice(&height.to_be_bytes());
        data.extend_from_slice(&[bit_depth, color_type, 0, 0, 0]);
        data.extend_from_slice(&png_crc32(&data[12..29]).to_be_bytes());
        data
    }

    fn rewrite_png_ihdr_crc(data: &mut [u8]) {
        let crc = png_crc32(&data[12..29]);
        data[29..33].copy_from_slice(&crc.to_be_bytes());
    }

    fn append_png_chunk(data: &mut Vec<u8>, kind: [u8; 4], payload: &[u8]) {
        data.extend_from_slice(
            &u32::try_from(payload.len())
                .expect("test PNG chunk length")
                .to_be_bytes(),
        );
        data.extend_from_slice(&kind);
        data.extend_from_slice(payload);
        let start = data.len() - kind.len() - payload.len();
        data.extend_from_slice(&png_crc32(&data[start..]).to_be_bytes());
    }

    #[test]
    fn every_encoded_resource_dimension_is_preflighted() {
        let header = png_header(64, 64, 8, 2);
        let cases = [
            (
                LimitOverrides {
                    decoded_pixels_per_image_or_frame: Some(4_095),
                    ..LimitOverrides::default()
                },
                ImageOptions::default(),
                "decoded pixels",
            ),
            (
                LimitOverrides {
                    decoded_edge_length: Some(63),
                    ..LimitOverrides::default()
                },
                ImageOptions::default(),
                "decoded edge",
            ),
            (
                LimitOverrides::default(),
                ImageOptions {
                    resized_height: Some(4_096),
                    resized_width: Some(4_128),
                    ..ImageOptions::default()
                },
                "prepared pixels",
            ),
            (
                LimitOverrides {
                    materialized_output_bytes_per_batch: Some(12_287),
                    ..LimitOverrides::default()
                },
                ImageOptions::default(),
                "prepared output bytes",
            ),
        ];
        for (overrides, options, label) in cases {
            let limits = ResourceLimits::default()
                .lowered(overrides)
                .expect("lowered limits");
            let error = prepare_image_rgb8(
                ImageInput::Encoded {
                    data: &header,
                    format: ImageFormat::Png,
                },
                &visual(),
                options,
                limits,
            )
            .expect_err(label);
            assert_eq!(error.category(), ErrorCategory::ResourceLimit, "{label}");
        }
    }

    #[test]
    fn encoded_precedence_is_overflow_then_resource_then_decode_then_geometry() {
        let overflow_header = png_header(u32::MAX, u32::MAX, 8, 6);
        assert_eq!(
            prepare_image_rgb8(
                ImageInput::Encoded {
                    data: &overflow_header,
                    format: ImageFormat::Png,
                },
                &visual(),
                ImageOptions::default(),
                ResourceLimits::default(),
            )
            .expect_err("decoder capacity overflow")
            .category(),
            ErrorCategory::ArithmeticOverflow
        );

        let la_header = png_header(64, 64, 8, 4);
        let output_limited = ResourceLimits::default()
            .lowered(LimitOverrides {
                materialized_output_bytes_per_batch: Some(1),
                ..LimitOverrides::default()
            })
            .expect("lowered output limit");
        assert_eq!(
            prepare_image_rgb8(
                ImageInput::Encoded {
                    data: &la_header,
                    format: ImageFormat::Png,
                },
                &visual(),
                ImageOptions::default(),
                output_limited,
            )
            .expect_err("resource preflight beats unsupported mode")
            .category(),
            ErrorCategory::ResourceLimit
        );
        assert_eq!(
            prepare_image_rgb8(
                ImageInput::Encoded {
                    data: &la_header,
                    format: ImageFormat::Png,
                },
                &visual(),
                ImageOptions::default(),
                ResourceLimits::default(),
            )
            .expect_err("LA is excluded")
            .category(),
            ErrorCategory::UnsupportedMedia
        );

        let corrupt_bad_aspect = png_header(12_864, 64, 8, 2);
        assert_eq!(
            prepare_image_rgb8(
                ImageInput::Encoded {
                    data: &corrupt_bad_aspect,
                    format: ImageFormat::Png,
                },
                &visual(),
                ImageOptions::default(),
                ResourceLimits::default(),
            )
            .expect_err("decode failure beats deferred aspect geometry")
            .category(),
            ErrorCategory::MediaDecode
        );
    }

    #[test]
    fn observed_deferred_geometry_error_releases_successfully_decoded_rgb() {
        const WIDTH: u32 = 12_864;
        const HEIGHT: u32 = 64;
        let rgb = vec![29_u8; WIDTH as usize * HEIGHT as usize * 3];
        let mut encoded = Vec::new();
        PngEncoder::new(&mut encoded)
            .write_image(&rgb, WIDTH, HEIGHT, ExtendedColorType::Rgb8)
            .expect("encode bad-aspect PNG");
        let plan = plan_image_rgb8(
            ImageInput::Encoded {
                data: &encoded,
                format: ImageFormat::Png,
            },
            &visual(),
            ImageOptions::default(),
            ResourceLimits::default(),
        )
        .expect("decode plan defers geometry");
        assert_eq!(
            plan.geometry().expect_err("bad aspect").category(),
            ErrorCategory::MediaGeometry
        );
        let scope = ObservationScope::media_at(0, 1, 2, 3, 4);
        let mut recorder = ObservationRecorder::new(16);
        let error = execute_image_plan_observed(plan, &mut recorder, scope)
            .expect_err("geometry after decode");
        assert_eq!(error.category(), ErrorCategory::MediaGeometry);
        let report = recorder.report();
        assert_eq!(report.allocations.transient_live_bytes, 0);
        assert!(report.buffers.iter().any(|buffer| {
            buffer.name == "decoded_rgb" && buffer.scope == scope && buffer.released_at_ns.is_some()
        }));
        assert!(report.spans.iter().any(|span| {
            span.name == "native.media.decode_color"
                && span.scope == scope
                && span.outcome == StageOutcome::Success
        }));
    }

    #[test]
    fn raw_resource_errors_precede_stride_geometry() {
        let limits = ResourceLimits::default()
            .lowered(LimitOverrides {
                decoded_pixels_per_image_or_frame: Some(1),
                ..LimitOverrides::default()
            })
            .expect("lowered limits");
        let error = prepare_image_rgb8(
            ImageInput::Rgb8(Rgb8 {
                data: &[],
                height: 2,
                width: 2,
                row_stride: 0,
            }),
            &visual(),
            ImageOptions::default(),
            limits,
        )
        .expect_err("pixel cap precedes invalid stride");
        assert_eq!(error.category(), ErrorCategory::ResourceLimit);

        let error = prepare_image_rgb8(
            ImageInput::Rgb8(Rgb8 {
                data: &[],
                height: 64,
                width: 64,
                row_stride: 0,
            }),
            &visual(),
            ImageOptions {
                resized_height: Some(4_096),
                resized_width: Some(4_128),
                ..ImageOptions::default()
            },
            ResourceLimits::default(),
        )
        .expect_err("prepared cap precedes invalid stride and short extent");
        assert_eq!(error.category(), ErrorCategory::ResourceLimit);
    }

    #[test]
    fn raw_view_rejects_zero_narrow_short_and_overflowing_layouts() {
        let cases = [
            Rgb8 {
                data: &[],
                height: 0,
                width: 1,
                row_stride: 3,
            },
            Rgb8 {
                data: &[],
                height: 1,
                width: 1,
                row_stride: 2,
            },
            Rgb8 {
                data: &[0, 0],
                height: 1,
                width: 1,
                row_stride: 3,
            },
        ];
        for raw in cases {
            assert_eq!(
                prepare_image_rgb8(
                    ImageInput::Rgb8(raw),
                    &visual(),
                    ImageOptions::default(),
                    ResourceLimits::default(),
                )
                .expect_err("invalid raw view")
                .category(),
                ErrorCategory::MediaGeometry
            );
        }
        assert_eq!(
            prepare_image_rgb8(
                ImageInput::Rgb8(Rgb8 {
                    data: &[],
                    height: 2,
                    width: usize::MAX,
                    row_stride: usize::MAX,
                }),
                &visual(),
                ImageOptions::default(),
                ResourceLimits::default(),
            )
            .expect_err("overflowing raw view")
            .category(),
            ErrorCategory::ArithmeticOverflow
        );
    }

    #[test]
    fn encoded_byte_limit_precedes_header_decode() {
        let limits = ResourceLimits::default()
            .lowered(LimitOverrides {
                encoded_bytes_per_item: Some(1),
                ..LimitOverrides::default()
            })
            .expect("lowered limits");
        let error = prepare_image_rgb8(
            ImageInput::Encoded {
                data: b"not a JPEG",
                format: ImageFormat::Jpeg,
            },
            &visual(),
            ImageOptions::default(),
            limits,
        )
        .expect_err("byte cap precedes signature");
        assert_eq!(error.category(), ErrorCategory::ResourceLimit);
    }

    #[test]
    fn oversized_headers_fail_before_full_decode() {
        let jpeg = [
            0xff, 0xd8, 0xff, 0xc0, 0x00, 0x11, 0x08, 0x80, 0x01, 0x00, 0x01, 0x03, 0x01, 0x11,
            0x00, 0x02, 0x11, 0x00, 0x03, 0x11, 0x00,
        ];
        assert_eq!(
            prepare_image_rgb8(
                ImageInput::Encoded {
                    data: &jpeg,
                    format: ImageFormat::Jpeg,
                },
                &visual(),
                ImageOptions::default(),
                ResourceLimits::default(),
            )
            .expect_err("oversized JPEG header")
            .category(),
            ErrorCategory::ResourceLimit
        );

        let webp =
            b"RIFF\x16\x00\x00\x00WEBPVP8X\x0a\x00\x00\x00\x00\x00\x00\x00\x00\x80\x00\x00\x00\x00";
        assert_eq!(
            prepare_image_rgb8(
                ImageInput::Encoded {
                    data: webp,
                    format: ImageFormat::WebP,
                },
                &visual(),
                ImageOptions::default(),
                ResourceLimits::default(),
            )
            .expect_err("oversized WebP header")
            .category(),
            ErrorCategory::ResourceLimit
        );
    }

    #[test]
    fn malformed_and_truncated_recognized_inputs_are_media_decode() {
        for (format, data) in [
            (ImageFormat::Jpeg, &b"\xff\xd8\xff\xda"[..]),
            (ImageFormat::Png, &b"\x89PNG\r\n\x1a\n"[..]),
            (ImageFormat::WebP, &b"RIFF\x04\x00\x00\x00WEBP"[..]),
        ] {
            assert_eq!(
                prepare_image_rgb8(
                    ImageInput::Encoded { data, format },
                    &visual(),
                    ImageOptions::default(),
                    ResourceLimits::default(),
                )
                .expect_err("damaged image")
                .category(),
                ErrorCategory::MediaDecode
            );
        }
    }

    #[test]
    fn malformed_png_ihdr_is_decode_but_valid_excluded_modes_are_unsupported() {
        for data in [png_header(64, 64, 3, 2), png_header(64, 64, 8, 1)] {
            assert_eq!(
                prepare_image_rgb8(
                    ImageInput::Encoded {
                        data: &data,
                        format: ImageFormat::Png,
                    },
                    &visual(),
                    ImageOptions::default(),
                    ResourceLimits::default(),
                )
                .expect_err("invalid PNG IHDR tuple")
                .category(),
                ErrorCategory::MediaDecode
            );
        }
        for data in [
            png_header(64, 64, 8, 3),
            png_header(64, 64, 8, 4),
            png_header(64, 64, 16, 2),
        ] {
            assert_eq!(
                prepare_image_rgb8(
                    ImageInput::Encoded {
                        data: &data,
                        format: ImageFormat::Png,
                    },
                    &visual(),
                    ImageOptions::default(),
                    ResourceLimits::default(),
                )
                .expect_err("valid but excluded PNG mode")
                .category(),
                ErrorCategory::UnsupportedMedia
            );
        }

        let mut invalid_method = png_header(64, 64, 8, 4);
        invalid_method[26] = 1;
        rewrite_png_ihdr_crc(&mut invalid_method);
        let mut invalid_crc = png_header(64, 64, 8, 4);
        invalid_crc[32] ^= 1;
        for data in [invalid_method, invalid_crc] {
            assert_eq!(
                prepare_image_rgb8(
                    ImageInput::Encoded {
                        data: &data,
                        format: ImageFormat::Png,
                    },
                    &visual(),
                    ImageOptions::default(),
                    ResourceLimits::default(),
                )
                .expect_err("malformed excluded-mode PNG header")
                .category(),
                ErrorCategory::MediaDecode
            );
        }
    }

    #[test]
    fn malformed_jpeg_and_webp_headers_do_not_masquerade_as_unsupported_media() {
        let jpeg = [
            0xff, 0xd8, 0xff, 0xc0, 0x00, 0x11, 0x0c, 0x00, 0x40, 0x00, 0x40, 0x03, 0x01, 0x11,
            0x00, 0x02, 0x11, 0x00, 0x03, 0x11, 0x00,
        ];
        let webp_reserved_animation =
            b"RIFF\x16\x00\x00\x00WEBPVP8X\x0a\x00\x00\x00\x83\x00\x00\x00\x3f\x00\x00\x3f\x00\x00";
        let webp_lossless_bad_version =
            b"RIFF\x11\x00\x00\x00WEBPVP8L\x05\x00\x00\x00\x2f\x00\x00\x00\xe0";
        let webp_bad_extended_length = b"RIFF\x17\x00\x00\x00WEBPVP8X\x0b\x00\x00\x00\x02\x00\x00\x00\x3f\x00\x00\x3f\x00\x00\x00";
        let jpeg_bad_component_count = [
            0xff, 0xd8, 0xff, 0xc0, 0x00, 0x08, 0x08, 0x00, 0x40, 0x00, 0x40, 0x00,
        ];
        let jpeg_bad_excluded_components = [
            0xff, 0xd8, 0xff, 0xc0, 0x00, 0x0e, 0x08, 0x00, 0x40, 0x00, 0x40, 0x02, 0x01, 0x01,
            0x00, 0x02, 0x11, 0x00,
        ];
        let webp_chunk_outside_declared_riff =
            b"RIFF\x04\x00\x00\x00WEBPVP8X\x0a\x00\x00\x00\x02\x00\x00\x00\x3f\x00\x00\x3f\x00\x00";
        for (format, data) in [
            (ImageFormat::Jpeg, &jpeg[..]),
            (ImageFormat::Jpeg, &jpeg_bad_component_count[..]),
            (ImageFormat::Jpeg, &jpeg_bad_excluded_components[..]),
            (ImageFormat::WebP, &webp_reserved_animation[..]),
            (ImageFormat::WebP, &webp_lossless_bad_version[..]),
            (ImageFormat::WebP, &webp_bad_extended_length[..]),
            (ImageFormat::WebP, &webp_chunk_outside_declared_riff[..]),
        ] {
            assert_eq!(
                prepare_image_rgb8(
                    ImageInput::Encoded { data, format },
                    &visual(),
                    ImageOptions::default(),
                    ResourceLimits::default(),
                )
                .expect_err("malformed encoded header")
                .category(),
                ErrorCategory::MediaDecode
            );
        }

        let jpeg_valid_excluded_components = [
            0xff, 0xd8, 0xff, 0xc0, 0x00, 0x0e, 0x08, 0x00, 0x40, 0x00, 0x40, 0x02, 0x01, 0x11,
            0x00, 0x02, 0x11, 0x00,
        ];
        assert_eq!(
            prepare_image_rgb8(
                ImageInput::Encoded {
                    data: &jpeg_valid_excluded_components,
                    format: ImageFormat::Jpeg,
                },
                &visual(),
                ImageOptions::default(),
                ResourceLimits::default(),
            )
            .expect_err("valid excluded two-component JPEG")
            .category(),
            ErrorCategory::UnsupportedMedia
        );
    }

    #[test]
    fn malformed_apng_control_is_decode_not_unsupported_animation() {
        let mut wrong_length = png_header(64, 64, 8, 2);
        append_png_chunk(&mut wrong_length, *b"acTL", &[0, 0, 0, 1]);
        let mut zero_frames = png_header(64, 64, 8, 2);
        append_png_chunk(&mut zero_frames, *b"acTL", &[0, 0, 0, 0, 0, 0, 0, 0]);
        for data in [wrong_length, zero_frames] {
            assert_eq!(
                prepare_image_rgb8(
                    ImageInput::Encoded {
                        data: &data,
                        format: ImageFormat::Png,
                    },
                    &visual(),
                    ImageOptions::default(),
                    ResourceLimits::default(),
                )
                .expect_err("malformed APNG control")
                .category(),
                ErrorCategory::MediaDecode
            );
        }
    }

    #[test]
    fn embedded_eoi_bytes_and_truncated_animation_chunks_do_not_mask_corruption() {
        let mut jpeg = vec![0xff, 0xd8, 0xff, 0xe0, 0x00, 0x04, 0xff, 0xd9];
        jpeg.extend_from_slice(&[
            0xff, 0xc0, 0x00, 0x11, 0x08, 0x00, 0x40, 0x00, 0x40, 0x03, 0x01, 0x11, 0x00, 0x02,
            0x11, 0x00, 0x03, 0x11, 0x00,
        ]);
        jpeg.extend_from_slice(&[
            0xff, 0xda, 0x00, 0x0c, 0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3f, 0x00,
            0x12, 0x34,
        ]);
        assert_eq!(
            prepare_image_rgb8(
                ImageInput::Encoded {
                    data: &jpeg,
                    format: ImageFormat::Jpeg,
                },
                &visual(),
                ImageOptions::default(),
                ResourceLimits::default(),
            )
            .expect_err("APP payload EOI is not a terminal marker")
            .category(),
            ErrorCategory::MediaDecode
        );

        let mut apng = png_header(64, 64, 8, 6);
        apng.extend_from_slice(&[0, 0, 0, 8]);
        apng.extend_from_slice(b"acTL");
        assert_eq!(
            prepare_image_rgb8(
                ImageInput::Encoded {
                    data: &apng,
                    format: ImageFormat::Png,
                },
                &visual(),
                ImageOptions::default(),
                ResourceLimits::default(),
            )
            .expect_err("truncated acTL is corruption, not valid animation")
            .category(),
            ErrorCategory::MediaDecode
        );
    }

    #[test]
    fn webp_animation_flag_is_detected_without_decoding() {
        let webp =
            b"RIFF\x16\x00\x00\x00WEBPVP8X\x0a\x00\x00\x00\x02\x00\x00\x00\x3f\x00\x00\x3f\x00\x00";
        assert!(probe_webp(webp).expect("header").animated);
    }
}
