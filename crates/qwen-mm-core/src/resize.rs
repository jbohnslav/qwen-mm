//! Reference-compatible spatial resize entry points.

use std::mem;

use crate::{
    error::{ErrorCategory, QwenError, Result},
    geometry::ImageGeometryPlan,
    limits::{checked_add, checked_mul},
    observability::{BufferClass, ObservationRecorder, ObservationScope},
};

const RGB_CHANNELS: u64 = 3;

/// Resizes one RGB8 image with a Pillow-compatible bicubic kernel.
///
/// The source may have row padding. The returned image is packed HWC RGB8 and
/// its dimensions are taken from `plan`.
///
/// # Errors
///
/// Returns `media_geometry` for an inconsistent source buffer or stride,
/// `arithmetic_overflow` for unrepresentable capacities, and
/// `internal_invariant` if the validated plan cannot be represented by the
/// selected kernel.
pub fn resize_image_rgb8(
    source: &[u8],
    source_height: u64,
    source_width: u64,
    source_stride_bytes: u64,
    plan: &ImageGeometryPlan,
) -> Result<Vec<u8>> {
    resize_image_rgb8_internal(
        source,
        source_height,
        source_width,
        source_stride_bytes,
        plan,
        None,
        ObservationScope::default(),
    )
}

/// Observed still-image resize used by the complete batch recorder.
pub(crate) fn resize_image_rgb8_observed(
    source: &[u8],
    source_height: u64,
    source_width: u64,
    source_stride_bytes: u64,
    plan: &ImageGeometryPlan,
    recorder: &mut ObservationRecorder,
    scope: ObservationScope,
) -> Result<Vec<u8>> {
    resize_image_rgb8_internal(
        source,
        source_height,
        source_width,
        source_stride_bytes,
        plan,
        Some(recorder),
        scope,
    )
}

#[allow(clippy::too_many_arguments)]
fn resize_image_rgb8_internal(
    source: &[u8],
    source_height: u64,
    source_width: u64,
    source_stride_bytes: u64,
    plan: &ImageGeometryPlan,
    recorder: Option<&mut ObservationRecorder>,
    scope: ObservationScope,
) -> Result<Vec<u8>> {
    let (destination_height, destination_width) = validate_resize_plan(plan)?;
    let source = validate_source(source, source_height, source_width, source_stride_bytes)?;

    resize_fixed_point_u8(
        source,
        destination_height,
        destination_width,
        recorder,
        scope,
    )
}

/// Resizes one RGB8 video frame to packed HWC `float32` in the `0..255`
/// domain using the pinned `TorchVision` bicubic-antialias kernel surface.
///
/// # Errors
///
/// Returns the same stable categories as [`resize_image_rgb8`].
pub fn resize_video_rgb8_to_f32(
    source: &[u8],
    source_height: u64,
    source_width: u64,
    source_stride_bytes: u64,
    plan: &ImageGeometryPlan,
) -> Result<Vec<f32>> {
    let (destination_height, destination_width) = validate_resize_plan(plan)?;
    let source = packed_source(
        source,
        source_height,
        source_width,
        source_stride_bytes,
        None,
        ObservationScope::default(),
    )?;
    resize_torchvision_f32(
        &source,
        usize_capacity(source_height, "source height")?,
        usize_capacity(source_width, "source width")?,
        destination_height,
        destination_width,
    )
}

#[derive(Debug)]
struct FloatWeights {
    bounds: Vec<(usize, usize)>,
    coefficients: Vec<f32>,
    kernel_size: usize,
}

/// Ports the exact `TorchVision` tensor wrapper semantics: cast RGB8 to
/// `float32`, perform `PyTorch`'s separable Keys bicubic-antialias convolution,
/// clamp, round ties to even, cast back to uint8, then expose `float32`.
fn resize_torchvision_f32(
    source: &[u8],
    source_height: usize,
    source_width: usize,
    destination_height: usize,
    destination_width: usize,
) -> Result<Vec<f32>> {
    let source = source.iter().copied().map(f32::from).collect::<Vec<_>>();
    let horizontal = if source_width == destination_width {
        source
    } else {
        let weights = float_cubic_weights(source_width, destination_width)?;
        convolve_horizontal_f32(
            &source,
            source_height,
            source_width,
            destination_width,
            &weights,
        )?
    };
    let resized = if source_height == destination_height {
        horizontal
    } else {
        let weights = float_cubic_weights(source_height, destination_height)?;
        convolve_vertical_f32(
            &horizontal,
            source_height,
            destination_height,
            destination_width,
            &weights,
        )?
    };
    Ok(resized
        .into_iter()
        .map(quantize_torchvision_u8_to_f32)
        .collect())
}

fn quantize_torchvision_u8_to_f32(value: f32) -> f32 {
    value.clamp(0.0, 255.0).round_ties_even()
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)] // All casts deliberately mirror pinned PyTorch C++ conversions.
fn float_cubic_weights(input_size: usize, output_size: usize) -> Result<FloatWeights> {
    if input_size == 0 || output_size == 0 {
        return Err(geometry("resize dimensions must be non-zero"));
    }
    let scale = input_size as f32 / output_size as f32;
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    let kernel_size = usize::try_from(support.ceil() as u64)
        .map_err(|_| overflow("video resize kernel size"))?
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| overflow("video resize kernel size"))?;
    let coefficients_len = output_size
        .checked_mul(kernel_size)
        .ok_or_else(|| overflow("video resize weights capacity"))?;
    let inverse_scale = if scale >= 1.0 { scale.recip() } else { 1.0 };
    let mut coefficients = vec![0.0_f32; coefficients_len];
    let mut bounds = Vec::with_capacity(output_size);

    for output_index in 0..output_size {
        let center = scale * (output_index as f32 + 0.5);
        let minimum_unclamped = (center - support + 0.5) as i64;
        let maximum_unclamped = (center + support + 0.5) as i64;
        let minimum = minimum_unclamped.max(0) as usize;
        let maximum = maximum_unclamped
            .min(i64::try_from(input_size).map_err(|_| overflow("video input dimension"))?)
            .max(i64::try_from(minimum).map_err(|_| overflow("video weight bound"))?)
            as usize;
        let count = (maximum - minimum).min(kernel_size);
        if count == 0 {
            return Err(invariant("video resize produced an empty filter window"));
        }
        bounds.push((minimum, count));

        let row = &mut coefficients[output_index * kernel_size..(output_index + 1) * kernel_size];
        let mut total = 0.0_f32;
        for (index, weight) in row.iter_mut().take(count).enumerate() {
            let distance = (index as f32 + minimum as f32 - center + 0.5) * inverse_scale;
            *weight = keys_cubic_f32(distance);
            total += *weight;
        }
        if total != 0.0 {
            for weight in row.iter_mut().take(count) {
                *weight /= total;
            }
        }
    }
    Ok(FloatWeights {
        bounds,
        coefficients,
        kernel_size,
    })
}

fn keys_cubic_f32(mut value: f32) -> f32 {
    const A: f32 = -0.5;
    value = value.abs();
    if value < 1.0 {
        let first = (A + 2.0).mul_add(value, -(A + 3.0));
        (first * value).mul_add(value, 1.0)
    } else if value < 2.0 {
        let first = A.mul_add(value, -5.0 * A);
        let second = first.mul_add(value, 8.0 * A);
        second.mul_add(value, -4.0 * A)
    } else {
        0.0
    }
}

fn convolve_horizontal_f32(
    source: &[f32],
    height: usize,
    source_width: usize,
    destination_width: usize,
    weights: &FloatWeights,
) -> Result<Vec<f32>> {
    let destination_len = height
        .checked_mul(destination_width)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| overflow("horizontal video resize capacity"))?;
    let mut destination = vec![0.0_f32; destination_len];
    for row in 0..height {
        for output_x in 0..destination_width {
            let (minimum, count) = weights.bounds[output_x];
            let coefficients = &weights.coefficients
                [output_x * weights.kernel_size..output_x * weights.kernel_size + count];
            for channel in 0..3 {
                let first_source = (row * source_width + minimum) * 3 + channel;
                let mut value = source[first_source] * coefficients[0];
                for (index, &coefficient) in coefficients.iter().enumerate().skip(1) {
                    value = source[first_source + index * 3].mul_add(coefficient, value);
                }
                destination[(row * destination_width + output_x) * 3 + channel] = value;
            }
        }
    }
    Ok(destination)
}

fn convolve_vertical_f32(
    source: &[f32],
    source_height: usize,
    destination_height: usize,
    width: usize,
    weights: &FloatWeights,
) -> Result<Vec<f32>> {
    let row_elements = width
        .checked_mul(3)
        .ok_or_else(|| overflow("vertical video resize row"))?;
    let destination_len = destination_height
        .checked_mul(row_elements)
        .ok_or_else(|| overflow("vertical video resize capacity"))?;
    let mut destination = vec![0.0_f32; destination_len];
    for output_y in 0..destination_height {
        let (minimum, count) = weights.bounds[output_y];
        debug_assert!(minimum + count <= source_height);
        let coefficients = &weights.coefficients
            [output_y * weights.kernel_size..output_y * weights.kernel_size + count];
        for element in 0..row_elements {
            let first_source = minimum * row_elements + element;
            let mut value = source[first_source] * coefficients[0];
            for (index, &coefficient) in coefficients.iter().enumerate().skip(1) {
                value = source[first_source + index * row_elements].mul_add(coefficient, value);
            }
            destination[output_y * row_elements + element] = value;
        }
    }
    Ok(destination)
}

#[derive(Debug)]
struct FixedWeights {
    bounds: Vec<(usize, usize)>,
    coefficients: Vec<i32>,
    kernel_size: usize,
}

/// A validated positive-stride RGB8 view. Still-image resize reads this view
/// directly so row padding never requires a whole-image packing allocation.
#[derive(Clone, Copy, Debug)]
struct Rgb8Source<'a> {
    data: &'a [u8],
    height: usize,
    width: usize,
    stride: usize,
    row_bytes: usize,
    packed_capacity: usize,
}

#[derive(Clone, Copy)]
struct ResizeAllocationNames {
    floating: &'static str,
    bounds: &'static str,
    coefficients: &'static str,
    destination: &'static str,
}

const HORIZONTAL_RESIZE_NAMES: ResizeAllocationNames = ResizeAllocationNames {
    floating: "resize.horizontal.weights_f64",
    bounds: "resize.horizontal.bounds",
    coefficients: "resize.horizontal.coefficients_i32",
    destination: "resize.horizontal.destination",
};

const VERTICAL_RESIZE_NAMES: ResizeAllocationNames = ResizeAllocationNames {
    floating: "resize.vertical.weights_f64",
    bounds: "resize.vertical.bounds",
    coefficients: "resize.vertical.coefficients_i32",
    destination: "resize.vertical.destination",
};

/// A source-faithful port of Pillow's separable 8-bit bicubic resampler.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn resize_fixed_point_u8(
    source: Rgb8Source<'_>,
    destination_height: usize,
    destination_width: usize,
    mut recorder: Option<&mut ObservationRecorder>,
    scope: ObservationScope,
) -> Result<Vec<u8>> {
    if source.height == destination_height && source.width == destination_width {
        let destination = copy_source_rows(source);
        if let Some(recorder) = recorder {
            recorder.record_allocation(
                "resize.noop.source_copy",
                BufferClass::Transient,
                scope,
                vec_capacity_bytes(&destination),
            );
            recorder.record_copy(
                "resize.noop.source_copy",
                scope,
                usize_bytes(destination.len()),
            );
            recorder.rename_live_transient("resize.noop.source_copy", "prepared_rgb", scope);
        }
        return Ok(destination);
    }

    // When width is already final, run the vertical kernel against the
    // validated stride directly. The old path first packed/copied every row.
    if source.width == destination_width {
        let weights = fixed_cubic_weights(
            source.height,
            destination_height,
            VERTICAL_RESIZE_NAMES,
            recorder.as_deref_mut(),
            scope,
        )?;
        let destination = convolve_vertical_source(
            source,
            destination_height,
            &weights,
            recorder.as_deref_mut(),
            scope,
        )?;
        release_fixed_weights(
            &weights,
            VERTICAL_RESIZE_NAMES,
            recorder.as_deref_mut(),
            scope,
        );
        if let Some(recorder) = recorder {
            recorder.rename_live_transient(
                VERTICAL_RESIZE_NAMES.destination,
                "prepared_rgb",
                scope,
            );
        }
        return Ok(destination);
    }

    let weights = fixed_cubic_weights(
        source.width,
        destination_width,
        HORIZONTAL_RESIZE_NAMES,
        recorder.as_deref_mut(),
        scope,
    )?;
    let horizontal = convolve_horizontal(
        source,
        destination_width,
        &weights,
        recorder.as_deref_mut(),
        scope,
    )?;
    release_fixed_weights(
        &weights,
        HORIZONTAL_RESIZE_NAMES,
        recorder.as_deref_mut(),
        scope,
    );

    if source.height == destination_height {
        if let Some(recorder) = recorder.as_deref_mut() {
            recorder.rename_live_transient(
                HORIZONTAL_RESIZE_NAMES.destination,
                "prepared_rgb",
                scope,
            );
        }
        return Ok(horizontal);
    }
    let weights = fixed_cubic_weights(
        source.height,
        destination_height,
        VERTICAL_RESIZE_NAMES,
        recorder.as_deref_mut(),
        scope,
    )?;
    let destination = convolve_vertical(
        &horizontal,
        source.height,
        destination_height,
        destination_width,
        &weights,
        recorder.as_deref_mut(),
        scope,
    )?;
    release_fixed_weights(
        &weights,
        VERTICAL_RESIZE_NAMES,
        recorder.as_deref_mut(),
        scope,
    );
    if let Some(recorder) = recorder {
        recorder.release_transient(
            HORIZONTAL_RESIZE_NAMES.destination,
            scope,
            vec_capacity_bytes(&horizontal),
        );
        recorder.rename_live_transient("resize.vertical.destination", "prepared_rgb", scope);
    }
    Ok(destination)
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)] // All casts deliberately mirror pinned PyTorch C++ conversions.
fn fixed_cubic_weights(
    input_size: usize,
    output_size: usize,
    names: ResizeAllocationNames,
    mut recorder: Option<&mut ObservationRecorder>,
    scope: ObservationScope,
) -> Result<FixedWeights> {
    const PRECISION: u32 = 22;
    if input_size == 0 || output_size == 0 {
        return Err(geometry("resize dimensions must be non-zero"));
    }
    let scale = input_size as f64 / output_size as f64;
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    let kernel_size = usize::try_from(support.ceil() as u64)
        .map_err(|_| overflow("video resize kernel size"))?
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| overflow("video resize kernel size"))?;
    let inverse_scale = if scale >= 1.0 { scale.recip() } else { 1.0 };
    let floating_len = output_size
        .checked_mul(kernel_size)
        .ok_or_else(|| overflow("image resize weights capacity"))?;
    let mut floating = vec![0.0_f64; floating_len];
    let mut bounds = Vec::with_capacity(output_size);
    if let Some(recorder) = recorder.as_deref_mut() {
        recorder.record_allocation(
            names.floating,
            BufferClass::Transient,
            scope,
            vec_capacity_bytes(&floating),
        );
        recorder.record_allocation(
            names.bounds,
            BufferClass::Transient,
            scope,
            vec_capacity_bytes(&bounds),
        );
    }

    for output_index in 0..output_size {
        let center = scale * (output_index as f64 + 0.5);
        // C++ conversion truncates toward zero. Preserve that behavior at the
        // left edge instead of substituting mathematical floor.
        let minimum_unclamped = (center - support + 0.5) as i64;
        let maximum_unclamped = (center + support + 0.5) as i64;
        let minimum = minimum_unclamped.max(0) as usize;
        let maximum = maximum_unclamped
            .min(i64::try_from(input_size).map_err(|_| overflow("video input dimension"))?)
            .max(i64::try_from(minimum).map_err(|_| overflow("video weight bound"))?)
            as usize;
        let count = (maximum - minimum).min(kernel_size);
        if count == 0 {
            return Err(invariant("image resize produced an empty filter window"));
        }
        bounds.push((minimum, count));

        let row = &mut floating[output_index * kernel_size..(output_index + 1) * kernel_size];
        let mut total = 0.0_f64;
        for (index, weight) in row.iter_mut().take(count).enumerate() {
            let distance = (index as f64 + minimum as f64 - center + 0.5) * inverse_scale;
            *weight = keys_cubic(distance);
            total += *weight;
        }
        if total != 0.0 {
            for weight in row.iter_mut().take(count) {
                *weight /= total;
            }
        }
    }

    let scale = f64::from(1_u32 << PRECISION);
    let floating_bytes = vec_capacity_bytes(&floating);
    let coefficients = floating
        .into_iter()
        .map(|weight| {
            let scaled = weight * scale;
            if scaled < 0.0 {
                (scaled - 0.5) as i32
            } else {
                (scaled + 0.5) as i32
            }
        })
        .collect::<Vec<_>>();
    if let Some(recorder) = recorder {
        recorder.record_allocation(
            names.coefficients,
            BufferClass::Transient,
            scope,
            vec_capacity_bytes(&coefficients),
        );
        recorder.release_transient(names.floating, scope, floating_bytes);
    }

    Ok(FixedWeights {
        bounds,
        coefficients,
        kernel_size,
    })
}

fn release_fixed_weights(
    weights: &FixedWeights,
    names: ResizeAllocationNames,
    recorder: Option<&mut ObservationRecorder>,
    scope: ObservationScope,
) {
    if let Some(recorder) = recorder {
        recorder.release_transient(names.bounds, scope, vec_capacity_bytes(&weights.bounds));
        recorder.release_transient(
            names.coefficients,
            scope,
            vec_capacity_bytes(&weights.coefficients),
        );
    }
}

fn keys_cubic(mut value: f64) -> f64 {
    const A: f64 = -0.5;
    value = value.abs();
    if value < 1.0 {
        ((A + 2.0) * value - (A + 3.0)) * value * value + 1.0
    } else if value < 2.0 {
        (((value - 5.0) * value + 8.0) * value - 4.0) * A
    } else {
        0.0
    }
}

fn convolve_horizontal(
    source: Rgb8Source<'_>,
    destination_width: usize,
    weights: &FixedWeights,
    recorder: Option<&mut ObservationRecorder>,
    scope: ObservationScope,
) -> Result<Vec<u8>> {
    const PRECISION: u32 = 22;
    let destination_len = source
        .height
        .checked_mul(destination_width)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| overflow("horizontal video resize capacity"))?;
    let mut destination = vec![0_u8; destination_len];
    if let Some(recorder) = recorder {
        recorder.record_allocation(
            HORIZONTAL_RESIZE_NAMES.destination,
            BufferClass::Transient,
            scope,
            vec_capacity_bytes(&destination),
        );
    }
    for row in 0..source.height {
        for output_x in 0..destination_width {
            let (minimum, count) = weights.bounds[output_x];
            let coefficients = &weights.coefficients
                [output_x * weights.kernel_size..output_x * weights.kernel_size + count];
            for channel in 0..3 {
                let mut sum = 1_i64 << (PRECISION - 1);
                for (index, &coefficient) in coefficients.iter().enumerate() {
                    let source_index = row * source.stride + (minimum + index) * 3 + channel;
                    sum += i64::from(source.data[source_index]) * i64::from(coefficient);
                }
                let value =
                    u8::try_from((sum >> PRECISION).clamp(0, 255)).expect("clamped RGB8 value");
                destination[(row * destination_width + output_x) * 3 + channel] = value;
            }
        }
    }
    Ok(destination)
}

fn convolve_vertical_source(
    source: Rgb8Source<'_>,
    destination_height: usize,
    weights: &FixedWeights,
    recorder: Option<&mut ObservationRecorder>,
    scope: ObservationScope,
) -> Result<Vec<u8>> {
    const PRECISION: u32 = 22;
    debug_assert_eq!(weights.bounds.len(), destination_height);
    let destination_len = destination_height
        .checked_mul(source.row_bytes)
        .ok_or_else(|| overflow("vertical video resize capacity"))?;
    let mut destination = vec![0_u8; destination_len];
    if let Some(recorder) = recorder {
        recorder.record_allocation(
            VERTICAL_RESIZE_NAMES.destination,
            BufferClass::Transient,
            scope,
            vec_capacity_bytes(&destination),
        );
    }
    for output_y in 0..destination_height {
        let (minimum, count) = weights.bounds[output_y];
        debug_assert!(minimum + count <= source.height);
        let coefficients = &weights.coefficients
            [output_y * weights.kernel_size..output_y * weights.kernel_size + count];
        for byte in 0..source.row_bytes {
            let mut sum = 1_i64 << (PRECISION - 1);
            for (index, &coefficient) in coefficients.iter().enumerate() {
                sum += i64::from(source.data[(minimum + index) * source.stride + byte])
                    * i64::from(coefficient);
            }
            destination[output_y * source.row_bytes + byte] =
                u8::try_from((sum >> PRECISION).clamp(0, 255)).expect("clamped RGB8 value");
        }
    }
    Ok(destination)
}

fn convolve_vertical(
    source: &[u8],
    source_height: usize,
    destination_height: usize,
    width: usize,
    weights: &FixedWeights,
    recorder: Option<&mut ObservationRecorder>,
    scope: ObservationScope,
) -> Result<Vec<u8>> {
    const PRECISION: u32 = 22;
    debug_assert_eq!(weights.bounds.len(), destination_height);
    let row_bytes = width
        .checked_mul(3)
        .ok_or_else(|| overflow("vertical video resize row"))?;
    let destination_len = destination_height
        .checked_mul(row_bytes)
        .ok_or_else(|| overflow("vertical video resize capacity"))?;
    let mut destination = vec![0_u8; destination_len];
    if let Some(recorder) = recorder {
        recorder.record_allocation(
            VERTICAL_RESIZE_NAMES.destination,
            BufferClass::Transient,
            scope,
            vec_capacity_bytes(&destination),
        );
    }
    for output_y in 0..destination_height {
        let (minimum, count) = weights.bounds[output_y];
        debug_assert!(minimum + count <= source_height);
        let coefficients = &weights.coefficients
            [output_y * weights.kernel_size..output_y * weights.kernel_size + count];
        for byte in 0..row_bytes {
            let mut sum = 1_i64 << (PRECISION - 1);
            for (index, &coefficient) in coefficients.iter().enumerate() {
                sum += i64::from(source[(minimum + index) * row_bytes + byte])
                    * i64::from(coefficient);
            }
            destination[output_y * row_bytes + byte] =
                u8::try_from((sum >> PRECISION).clamp(0, 255)).expect("clamped RGB8 value");
        }
    }
    Ok(destination)
}

fn packed_source(
    source: &[u8],
    source_height: u64,
    source_width: u64,
    source_stride_bytes: u64,
    recorder: Option<&mut ObservationRecorder>,
    scope: ObservationScope,
) -> Result<Vec<u8>> {
    if source_height == 0 || source_width == 0 {
        return Err(geometry("source dimensions must be non-zero"));
    }
    let packed_stride = checked_mul("packed source RGB stride", source_width, RGB_CHANNELS)?;
    if source_stride_bytes < packed_stride {
        return Err(geometry("source RGB stride is smaller than packed width")
            .with_context("stride", source_stride_bytes)
            .with_context("packed_stride", packed_stride));
    }
    let preceding_rows = source_height - 1;
    let required = checked_add(
        "source RGB buffer length",
        checked_mul(
            "source RGB preceding rows",
            preceding_rows,
            source_stride_bytes,
        )?,
        packed_stride,
    )?;
    let required_usize = usize_capacity(required, "source RGB buffer length")?;
    if source.len() < required_usize {
        return Err(
            geometry("source RGB buffer is shorter than its dimensions and stride")
                .with_context("actual_bytes", source.len())
                .with_context("required_bytes", required),
        );
    }

    let packed_capacity = checked_mul("packed source RGB capacity", source_height, packed_stride)?;
    let mut packed = Vec::with_capacity(usize_capacity(
        packed_capacity,
        "packed source RGB capacity",
    )?);
    let stride = usize_capacity(source_stride_bytes, "source RGB stride")?;
    let row_bytes = usize_capacity(packed_stride, "packed source RGB stride")?;
    for row in 0..usize_capacity(source_height, "source RGB height")? {
        let start = row
            .checked_mul(stride)
            .ok_or_else(|| overflow("source RGB row offset"))?;
        packed.extend_from_slice(&source[start..start + row_bytes]);
    }
    if let Some(recorder) = recorder {
        recorder.record_allocation(
            "resize.packed_source",
            BufferClass::Transient,
            scope,
            vec_capacity_bytes(&packed),
        );
        recorder.record_copy("resize.packed_source", scope, packed_capacity);
    }
    Ok(packed)
}

fn validate_source(
    source: &[u8],
    source_height: u64,
    source_width: u64,
    source_stride_bytes: u64,
) -> Result<Rgb8Source<'_>> {
    if source_height == 0 || source_width == 0 {
        return Err(geometry("source dimensions must be non-zero"));
    }
    let packed_stride = checked_mul("packed source RGB stride", source_width, RGB_CHANNELS)?;
    if source_stride_bytes < packed_stride {
        return Err(geometry("source RGB stride is smaller than packed width")
            .with_context("stride", source_stride_bytes)
            .with_context("packed_stride", packed_stride));
    }
    let required = checked_add(
        "source RGB buffer length",
        checked_mul(
            "source RGB preceding rows",
            source_height - 1,
            source_stride_bytes,
        )?,
        packed_stride,
    )?;
    if source.len() < usize_capacity(required, "source RGB buffer length")? {
        return Err(
            geometry("source RGB buffer is shorter than its dimensions and stride")
                .with_context("actual_bytes", source.len())
                .with_context("required_bytes", required),
        );
    }
    let packed_capacity = checked_mul("packed source RGB capacity", source_height, packed_stride)?;
    // Preserve the selected kernel's original source-dimension representability
    // gate after layout validation and before any allocation or indexing.
    let height = dimension_u32("source_height", source_height)? as usize;
    let width = dimension_u32("source_width", source_width)? as usize;
    Ok(Rgb8Source {
        data: source,
        height,
        width,
        stride: usize_capacity(source_stride_bytes, "source RGB stride")?,
        row_bytes: usize_capacity(packed_stride, "packed source RGB stride")?,
        packed_capacity: usize_capacity(packed_capacity, "packed source RGB capacity")?,
    })
}

fn copy_source_rows(source: Rgb8Source<'_>) -> Vec<u8> {
    let mut packed = Vec::with_capacity(source.packed_capacity);
    for row in 0..source.height {
        let start = row * source.stride;
        packed.extend_from_slice(&source.data[start..start + source.row_bytes]);
    }
    packed
}

#[allow(clippy::ptr_arg)] // Vec capacity, rather than slice length, is the allocation metric.
fn vec_capacity_bytes<T>(values: &Vec<T>) -> u64 {
    let capacity = u64::try_from(values.capacity()).unwrap_or(u64::MAX);
    let element_bytes = u64::try_from(mem::size_of::<T>()).unwrap_or(u64::MAX);
    capacity.saturating_mul(element_bytes)
}

fn usize_bytes(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn validate_resize_plan(plan: &ImageGeometryPlan) -> Result<(usize, usize)> {
    if plan.height == 0 || plan.width == 0 {
        return Err(invariant("resize plan dimensions must be non-zero")
            .with_context("height", plan.height)
            .with_context("width", plan.width));
    }
    let expected_stride = checked_mul("resize plan RGB stride", plan.width, RGB_CHANNELS)?;
    let expected_capacity = checked_mul("resize plan RGB capacity", plan.height, expected_stride)?;
    if plan.rgb_row_stride_bytes != expected_stride || plan.rgb_capacity_bytes != expected_capacity
    {
        return Err(invariant("resize plan RGB layout is inconsistent")
            .with_context("expected_stride", expected_stride)
            .with_context("actual_stride", plan.rgb_row_stride_bytes)
            .with_context("expected_capacity", expected_capacity)
            .with_context("actual_capacity", plan.rgb_capacity_bytes));
    }
    let height = dimension_u32("destination_height", plan.height)? as usize;
    let width = dimension_u32("destination_width", plan.width)? as usize;
    Ok((height, width))
}

fn dimension_u32(name: &'static str, value: u64) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        invariant("resize dimension does not fit selected kernel")
            .with_context("dimension", name)
            .with_context("value", value)
    })
}

fn usize_capacity(value: u64, name: &'static str) -> Result<usize> {
    usize::try_from(value).map_err(|_| overflow(name))
}

fn geometry(message: impl Into<String>) -> QwenError {
    QwenError::new(ErrorCategory::MediaGeometry, message)
}

fn overflow(message: impl Into<String>) -> QwenError {
    QwenError::new(ErrorCategory::ArithmeticOverflow, message)
}

fn invariant(message: impl Into<String>) -> QwenError {
    QwenError::new(ErrorCategory::InternalInvariant, message)
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::{
        quantize_torchvision_u8_to_f32, resize_image_rgb8, resize_image_rgb8_observed,
        resize_video_rgb8_to_f32,
    };
    use crate::{
        error::ErrorCategory,
        geometry::plan_image_geometry,
        limits::ResourceLimits,
        observability::{BufferClass, ObservationRecorder, ObservationScope},
        profile::{ProfileAlias, ProfileRegistry},
        request::ImageOptions,
    };

    fn plan(height: u64, width: u64) -> crate::geometry::ImageGeometryPlan {
        plan_with_options(height, width, ImageOptions::default())
    }

    fn plan_with_options(
        height: u64,
        width: u64,
        options: ImageOptions,
    ) -> crate::geometry::ImageGeometryPlan {
        let registry = ProfileRegistry::bundled().expect("bundled profiles");
        plan_image_geometry(
            &registry.get(ProfileAlias::Qwen3Vl8b).visual,
            height,
            width,
            options,
            ResourceLimits::default(),
        )
        .expect("plan")
    }

    const MANIFEST: &str = include_str!("../../../reference/resize/v1/manifest.json");
    const SOURCES: &[u8] = include_bytes!("../../../reference/resize/v1/sources.rgb8.bin");
    const PILLOW: &[u8] = include_bytes!("../../../reference/resize/v1/pillow-image-rgb8.bin");
    const TORCHVISION: &[u8] =
        include_bytes!("../../../reference/resize/v1/torchvision-video-f32le.bin");

    #[test]
    fn observed_two_axis_resize_reconciles_owned_scratch_and_copy_bytes() {
        let geometry = plan_with_options(
            64,
            64,
            ImageOptions {
                resized_height: Some(96),
                resized_width: Some(96),
                ..ImageOptions::default()
            },
        );
        assert_ne!([geometry.height, geometry.width], [64, 64]);
        let source = vec![17_u8; 64 * 64 * 3];
        let scope = ObservationScope::media_at(2, 3, 4, 5, 6);
        let mut recorder = ObservationRecorder::new(64);
        let prepared =
            resize_image_rgb8_observed(&source, 64, 64, 64 * 3, &geometry, &mut recorder, scope)
                .expect("observed resize");
        let report = recorder.report();
        assert_eq!(report.allocations.copy_count, 0);
        assert_eq!(report.allocations.copied_bytes, 0);
        assert!(report.copies.is_empty());
        assert_eq!(report.allocations.allocation_count, 8);
        assert_eq!(report.buffers.len(), 8);
        assert_eq!(
            report.allocations.allocated_bytes,
            report
                .buffers
                .iter()
                .map(|buffer| buffer.bytes)
                .sum::<u64>()
        );
        let names = report
            .buffers
            .iter()
            .map(|buffer| buffer.name.as_str())
            .collect::<Vec<_>>();
        for required in [
            "resize.horizontal.weights_f64",
            "resize.horizontal.bounds",
            "resize.horizontal.coefficients_i32",
            "resize.horizontal.destination",
            "resize.vertical.weights_f64",
            "resize.vertical.bounds",
            "resize.vertical.coefficients_i32",
            "prepared_rgb",
        ] {
            assert!(names.contains(&required), "missing {required}: {names:?}");
        }
        assert!(!names.contains(&"resize.packed_source"));
        let live = report
            .buffers
            .iter()
            .filter(|buffer| buffer.released_at_ns.is_none())
            .collect::<Vec<_>>();
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].name, "prepared_rgb");
        assert_eq!(live[0].class, BufferClass::Transient);
        assert_eq!(
            report.allocations.transient_live_bytes,
            u64::try_from(prepared.capacity()).expect("capacity")
        );
        assert!(
            report.allocations.peak_transient_live_bytes > report.allocations.transient_live_bytes
        );
        recorder.release_transient(
            "prepared_rgb",
            scope,
            u64::try_from(prepared.capacity()).expect("capacity"),
        );
        drop(prepared);
        assert_eq!(recorder.report().allocations.transient_live_bytes, 0);
    }

    #[test]
    fn observed_height_only_resize_reads_strided_source_without_a_clone() {
        let geometry = plan_with_options(
            64,
            64,
            ImageOptions {
                resized_height: Some(96),
                resized_width: Some(64),
                ..ImageOptions::default()
            },
        );
        let source = (0..64 * 64 * 3)
            .map(|index| u8::try_from(index % 251).expect("value"))
            .collect::<Vec<_>>();
        let expected = resize_image_rgb8(&source, 64, 64, 64 * 3, &geometry).expect("plain");
        let scope = ObservationScope::media(0, 0, 0);
        let mut recorder = ObservationRecorder::new(32);
        let actual =
            resize_image_rgb8_observed(&source, 64, 64, 64 * 3, &geometry, &mut recorder, scope)
                .expect("observed");
        assert_eq!(actual, expected);
        let report = recorder.report();
        assert_eq!(report.allocations.allocation_count, 4);
        assert_eq!(report.allocations.copy_count, 0);
        assert_eq!(report.allocations.copied_bytes, 0);
        assert!(report.copies.is_empty());
        assert!(report.buffers.iter().all(|buffer| {
            buffer.name != "resize.packed_source" && buffer.name != "resize.horizontal.source_copy"
        }));
        assert_eq!(
            report
                .buffers
                .iter()
                .filter(|buffer| buffer.name == "prepared_rgb")
                .count(),
            1
        );
    }

    #[test]
    fn height_only_resize_matches_packed_oracle_for_padded_rows() {
        const HEIGHT: usize = 64;
        const WIDTH: usize = 64;
        const ROW_BYTES: usize = WIDTH * 3;
        const PADDED_STRIDE: usize = ROW_BYTES + 7;
        let packed = (0_u8..=u8::MAX)
            .cycle()
            .take(HEIGHT * ROW_BYTES)
            .collect::<Vec<_>>();
        let mut padded = vec![0xa5_u8; HEIGHT * PADDED_STRIDE];
        for row in 0..HEIGHT {
            padded[row * PADDED_STRIDE..row * PADDED_STRIDE + ROW_BYTES]
                .copy_from_slice(&packed[row * ROW_BYTES..(row + 1) * ROW_BYTES]);
        }
        let source_height = u64::try_from(HEIGHT).expect("height");
        let source_width = u64::try_from(WIDTH).expect("width");
        let geometry = plan_with_options(
            source_height,
            source_width,
            ImageOptions {
                resized_height: Some(96),
                resized_width: Some(source_width),
                ..ImageOptions::default()
            },
        );
        let expected = resize_image_rgb8(
            &packed,
            source_height,
            source_width,
            u64::try_from(ROW_BYTES).expect("row bytes"),
            &geometry,
        )
        .expect("packed vertical resize");
        let scope = ObservationScope::media_at(1, 2, 3, 4, 5);
        let mut recorder = ObservationRecorder::new(16);
        let actual = resize_image_rgb8_observed(
            &padded,
            source_height,
            source_width,
            u64::try_from(PADDED_STRIDE).expect("padded stride"),
            &geometry,
            &mut recorder,
            scope,
        )
        .expect("padded vertical resize");

        assert_eq!(actual, expected);
        let report = recorder.report();
        assert_eq!(report.allocations.copy_count, 0);
        assert_eq!(report.allocations.copied_bytes, 0);
        assert!(report.copies.is_empty());
        assert!(report.buffers.iter().all(|buffer| {
            buffer.name != "resize.packed_source" && buffer.name != "resize.horizontal.source_copy"
        }));
    }

    #[derive(Deserialize)]
    struct Fixture {
        cases: Vec<Case>,
    }

    #[derive(Deserialize)]
    struct Case {
        id: String,
        tags: Vec<String>,
        source: Source,
        geometry_options: GeometryOptions,
        destination: Dimensions,
        pillow_image_rgb8: Artifact,
        torchvision_video_rgb_f32: Artifact,
    }

    #[derive(Deserialize)]
    struct Source {
        height: u64,
        width: u64,
        stride_bytes: u64,
        offset: usize,
        byte_length: usize,
    }

    #[derive(Deserialize)]
    struct GeometryOptions {
        min_pixels: Option<u64>,
        max_pixels: Option<u64>,
    }

    #[derive(Deserialize)]
    struct Dimensions {
        height: u64,
        width: u64,
    }

    #[derive(Deserialize)]
    struct Artifact {
        offset: usize,
        byte_length: usize,
    }

    #[test]
    fn no_op_is_an_exact_copy_in_both_domains() {
        let source = (0_u8..=255).cycle().take(64 * 64 * 3).collect::<Vec<_>>();
        let geometry = plan(64, 64);
        assert_eq!(
            resize_image_rgb8(&source, 64, 64, 192, &geometry).expect("image"),
            source
        );
        assert_eq!(
            resize_video_rgb8_to_f32(&source, 64, 64, 192, &geometry).expect("video"),
            source.iter().copied().map(f32::from).collect::<Vec<_>>()
        );
    }

    #[test]
    fn observed_no_op_performs_only_the_required_owned_output_copy() {
        let source = (0_u8..=255).cycle().take(64 * 64 * 3).collect::<Vec<_>>();
        let geometry = plan(64, 64);
        let plain = resize_image_rgb8(&source, 64, 64, 192, &geometry).expect("plain image");
        let scope = ObservationScope::media_at(1, 2, 3, 4, 5);
        let mut recorder = ObservationRecorder::new(8);
        let observed =
            resize_image_rgb8_observed(&source, 64, 64, 192, &geometry, &mut recorder, scope)
                .expect("observed image");
        assert_eq!(observed, plain);
        assert_eq!(observed, source);
        assert_ne!(plain.as_ptr(), source.as_ptr());
        assert_ne!(observed.as_ptr(), source.as_ptr());
        assert_eq!(plain.capacity(), observed.capacity());

        let report = recorder.report();
        assert_eq!(report.allocations.allocation_count, 1);
        assert_eq!(report.allocations.copy_count, 1);
        assert_eq!(
            report.allocations.copied_bytes,
            u64::try_from(source.len()).expect("copy bytes")
        );
        assert_eq!(
            report
                .copies
                .iter()
                .map(|copy| copy.name.as_str())
                .collect::<Vec<_>>(),
            vec!["resize.noop.source_copy"]
        );
        assert!(report.copies.iter().all(|copy| copy.scope == scope));
        assert!(
            report
                .buffers
                .iter()
                .all(|buffer| buffer.name != "resize.packed_source")
        );
        assert!(
            report
                .buffers
                .iter()
                .any(|buffer| { buffer.name == "prepared_rgb" && buffer.released_at_ns.is_none() })
        );
        assert_eq!(
            report.allocations.transient_live_bytes,
            u64::try_from(observed.capacity()).expect("capacity")
        );
    }

    #[test]
    fn padded_rows_are_accepted_and_short_or_narrow_inputs_fail_stably() {
        let mut source = vec![0_u8; 64 * 200];
        for (row, bytes) in source.chunks_exact_mut(200).enumerate() {
            for (column, value) in bytes[..192].iter_mut().enumerate() {
                *value = u8::try_from((row + column) % 256).expect("bounded");
            }
        }
        let geometry = plan(64, 64);
        let padded = resize_image_rgb8(&source, 64, 64, 200, &geometry).expect("padded");
        let packed = source
            .chunks_exact(200)
            .flat_map(|row| row[..192].iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(padded, packed);
        assert_eq!(
            resize_image_rgb8(&source, 64, 64, 191, &geometry)
                .expect_err("narrow stride")
                .category(),
            ErrorCategory::MediaGeometry
        );
        assert_eq!(
            resize_image_rgb8(&source[..100], 64, 64, 200, &geometry)
                .expect_err("short buffer")
                .category(),
            ErrorCategory::MediaGeometry
        );
    }

    #[test]
    fn forged_resize_plans_fail_before_kernel_indexing() {
        let source = vec![0_u8; 64 * 64 * 3];
        let mut inconsistent = plan(64, 64);
        inconsistent.rgb_capacity_bytes -= 1;
        assert_eq!(
            resize_image_rgb8(&source, 64, 64, 192, &inconsistent)
                .expect_err("inconsistent capacity")
                .category(),
            ErrorCategory::InternalInvariant
        );

        let mut zero = plan(64, 64);
        zero.width = 0;
        zero.rgb_row_stride_bytes = 0;
        zero.rgb_capacity_bytes = 0;
        assert_eq!(
            resize_video_rgb8_to_f32(&source, 64, 64, 192, &zero)
                .expect_err("zero plan")
                .category(),
            ErrorCategory::InternalInvariant
        );
    }

    #[test]
    fn video_quantization_clamps_and_rounds_ties_to_even() {
        let values = [-1.0, 0.5, 1.5, 2.5, 254.5, 255.5, 300.0];
        let actual = values.map(quantize_torchvision_u8_to_f32).map(f32::to_bits);
        let expected = [0.0, 0.0, 2.0, 2.0, 254.0, 255.0, 255.0].map(f32::to_bits);
        assert_eq!(actual, expected);
    }

    #[test]
    #[allow(clippy::too_many_lines)] // The corpus gate keeps coverage and both boundaries together.
    fn frozen_pillow_and_torchvision_resize_stage_corpus_passes() {
        let fixture: Fixture = serde_json::from_str(MANIFEST).expect("resize manifest");
        let required_tags = [
            "no_op",
            "upsample",
            "downsample",
            "aligned",
            "off_grid",
            "factor_boundary",
            "min_boundary",
            "max_boundary",
            "portrait",
            "landscape",
            "impulse",
            "ramp",
            "checkerboard",
            "edges",
            "high_frequency",
            "independent_rgb",
            "horizontal_only",
            "vertical_only",
            "padded_stride",
        ];
        for tag in required_tags {
            assert!(
                fixture
                    .cases
                    .iter()
                    .any(|case| case.tags.iter().any(|candidate| candidate == tag)),
                "missing required resize corpus tag: {tag}"
            );
        }

        for case in fixture.cases {
            let source = &SOURCES[case.source.offset..case.source.offset + case.source.byte_length];
            let options = ImageOptions {
                min_pixels: case.geometry_options.min_pixels,
                max_pixels: case.geometry_options.max_pixels,
                ..ImageOptions::default()
            };
            let geometry = plan_with_options(case.source.height, case.source.width, options);
            assert_eq!(
                [geometry.height, geometry.width],
                [case.destination.height, case.destination.width],
                "{} geometry",
                case.id
            );

            let actual_image = resize_image_rgb8(
                source,
                case.source.height,
                case.source.width,
                case.source.stride_bytes,
                &geometry,
            )
            .unwrap_or_else(|error| panic!("{} image resize: {error}", case.id));
            let expected_image = &PILLOW[case.pillow_image_rgb8.offset
                ..case.pillow_image_rgb8.offset + case.pillow_image_rgb8.byte_length];
            assert_eq!(actual_image, expected_image, "{} Pillow RGB8", case.id);

            let actual_video = resize_video_rgb8_to_f32(
                source,
                case.source.height,
                case.source.width,
                case.source.stride_bytes,
                &geometry,
            )
            .unwrap_or_else(|error| panic!("{} video resize: {error}", case.id));
            let expected_video = TORCHVISION[case.torchvision_video_rgb_f32.offset
                ..case.torchvision_video_rgb_f32.offset
                    + case.torchvision_video_rgb_f32.byte_length]
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four bytes")))
                .collect::<Vec<_>>();
            assert_eq!(
                actual_video.len(),
                expected_video.len(),
                "{} video length",
                case.id
            );
            let maximum_video_error = actual_video
                .iter()
                .zip(&expected_video)
                .map(|(&actual, &expected)| (actual - expected).abs())
                .fold(0.0_f32, f32::max);
            let first_video_difference = actual_video
                .iter()
                .zip(&expected_video)
                .enumerate()
                .find(|(_, pair)| pair.0.to_bits() != pair.1.to_bits())
                .map(|(index, (&actual, &expected))| (index, actual, expected));
            assert!(
                maximum_video_error <= 1.0e-4,
                "{} TorchVision max absolute error {maximum_video_error}, first difference {first_video_difference:?}",
                case.id,
            );
        }
    }
}
