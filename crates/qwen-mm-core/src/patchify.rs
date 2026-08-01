//! Normalization and Qwen patch-major layout for prepared packed RGB images.

use crate::{
    error::{ErrorCategory, QwenError, Result},
    geometry::ImageGeometryPlan,
    limits::{ResourceLimits, checked_add, checked_mul},
    output::Matrix,
    profile::VisualProfile,
};

const RGB_CHANNELS: u64 = 3;
const F32_BYTES: u64 = 4;
const I64_BYTES: u64 = 8;
const GRID_COLUMNS: u64 = 3;
const RESCALE_DENOMINATOR: f32 = 255.0;

/// A completely validated allocation plan for one prepared RGB image.
///
/// Construct this first when a caller needs to prove capacities without
/// materializing the potentially large patch matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImagePatchifyPlan {
    geometry: ImageGeometryPlan,
    input_bytes: u64,
    output_elements: u64,
    materialized_bytes: u64,
}

impl ImagePatchifyPlan {
    /// Returns the validated B3 geometry embedded in this plan.
    #[must_use]
    pub const fn geometry(self) -> ImageGeometryPlan {
        self.geometry
    }

    /// Returns the exact required packed-HWC RGB byte length.
    #[must_use]
    pub const fn input_bytes(self) -> u64 {
        self.input_bytes
    }

    /// Returns the exact number of `float32` output elements.
    #[must_use]
    pub const fn output_elements(self) -> u64 {
        self.output_elements
    }

    /// Returns the combined `pixel_values` and `image_grid_thw` byte count.
    #[must_use]
    pub const fn materialized_bytes(self) -> u64 {
        self.materialized_bytes
    }
}

/// Parity-facing arrays for one prepared still image.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedImage {
    /// C-contiguous `float32 [patch_rows, 1536]` patch-major values.
    pub pixel_values: Matrix<f32>,
    /// Exact C-contiguous `int64 [1, 3]` temporal/height/width patch grid.
    pub image_grid_thw: Matrix<i64>,
}

/// Validates all input, plan, limit, offset, and capacity arithmetic needed to
/// normalize and patchify one already-sized packed-HWC RGB8 image.
///
/// This function performs no allocation. `input_length` must be the exact byte
/// length: arbitrary row strides and trailing storage are intentionally outside
/// the B4 prepared-RGB boundary.
///
/// # Errors
///
/// Returns `media_geometry` for a packed-input length mismatch,
/// `resource_limit` for a configured pixel/output excess,
/// `arithmetic_overflow` for an unrepresentable product or host capacity, and
/// `internal_invariant` when the supplied B3 plan or visual constants are stale.
pub fn plan_image_patchify(
    visual: &VisualProfile,
    geometry: ImageGeometryPlan,
    input_length: u64,
    limits: ResourceLimits,
) -> Result<ImagePatchifyPlan> {
    validate_visual_profile(visual)?;

    let prepared_pixels = checked_mul(
        "patchify prepared image pixels",
        geometry.height,
        geometry.width,
    )?;
    check_limit(
        "prepared_image_pixels_per_occurrence",
        prepared_pixels,
        limits.prepared_image_pixels_per_occurrence(),
    )?;

    let expected = expected_geometry(visual, geometry.height, geometry.width)?;
    if geometry != expected {
        return Err(invariant("image geometry plan is stale or inconsistent")
            .with_context("expected_patch_rows", expected.patch_rows)
            .with_context("actual_patch_rows", geometry.patch_rows)
            .with_context(
                "expected_pixel_values_bytes",
                expected.pixel_values_capacity_bytes,
            )
            .with_context(
                "actual_pixel_values_bytes",
                geometry.pixel_values_capacity_bytes,
            ));
    }

    if input_length != geometry.rgb_capacity_bytes {
        return Err(QwenError::new(
            ErrorCategory::MediaGeometry,
            "prepared packed RGB length does not match its planned geometry",
        )
        .with_context("expected_bytes", geometry.rgb_capacity_bytes)
        .with_context("actual_bytes", input_length));
    }

    let (output_elements, materialized_bytes) =
        validate_offsets_and_capacities(visual, geometry, input_length)?;
    limits.check_materialized_output_bytes(materialized_bytes)?;

    // Conversion is part of preflight even though execution repeats it for
    // convenient local indexing. No platform can begin writing first.
    validate_host_dimensions(visual, geometry, input_length, output_elements)?;

    Ok(ImagePatchifyPlan {
        geometry,
        input_bytes: input_length,
        output_elements,
        materialized_bytes,
    })
}

fn validate_offsets_and_capacities(
    visual: &VisualProfile,
    geometry: ImageGeometryPlan,
    input_length: u64,
) -> Result<(u64, u64)> {
    // Prove the last readable byte and last writable element before either
    // source indexing or destination allocation begins.
    let last_row = checked_mul(
        "patchify final RGB row offset",
        geometry.height - 1,
        geometry.rgb_row_stride_bytes,
    )?;
    let last_pixel = checked_mul(
        "patchify final RGB pixel offset",
        geometry.width - 1,
        RGB_CHANNELS,
    )?;
    let last_channel = checked_add(
        "patchify final RGB channel offset",
        checked_add("patchify final RGB byte offset", last_row, last_pixel)?,
        RGB_CHANNELS - 1,
    )?;
    let readable_span = checked_add("patchify readable RGB byte span", last_channel, 1)?;
    if readable_span != input_length {
        return Err(invariant("packed RGB offsets disagree with input capacity")
            .with_context("offset_span", readable_span)
            .with_context("input_bytes", input_length));
    }

    let output_elements = checked_mul(
        "patchify output element capacity",
        geometry.patch_rows,
        visual.patch_width,
    )?;
    let pixel_values_bytes =
        checked_mul("patchify output byte capacity", output_elements, F32_BYTES)?;
    if pixel_values_bytes != geometry.pixel_values_capacity_bytes {
        return Err(invariant(
            "patch output capacity disagrees with geometry plan",
        ));
    }
    let last_output = checked_add(
        "patchify final output element offset",
        output_elements
            .checked_sub(1)
            .ok_or_else(|| invariant("patchify plan unexpectedly contains no output elements"))?,
        1,
    )?;
    if last_output != output_elements {
        return Err(invariant("patch output offset disagrees with capacity"));
    }

    let grid_bytes = checked_mul("patchify grid byte capacity", GRID_COLUMNS, I64_BYTES)?;
    let materialized_bytes = checked_add(
        "patchify materialized output bytes",
        pixel_values_bytes,
        grid_bytes,
    )?;
    Ok((output_elements, materialized_bytes))
}

fn validate_host_dimensions(
    visual: &VisualProfile,
    geometry: ImageGeometryPlan,
    input_length: u64,
    output_elements: u64,
) -> Result<()> {
    usize::try_from(input_length).map_err(|_| overflow("RGB input length does not fit usize"))?;
    usize::try_from(output_elements)
        .map_err(|_| overflow("patch output element count does not fit usize"))?;
    for (name, value) in [
        ("height", geometry.height),
        ("width", geometry.width),
        ("patch_size", visual.patch_size),
        ("temporal_patch_size", visual.temporal_patch_size),
        ("merge_size", visual.merge_size),
    ] {
        usize::try_from(value).map_err(|_| {
            overflow("patchify dimension does not fit usize").with_context("dimension", name)
        })?;
    }
    for value in geometry.image_grid_thw {
        i64::try_from(value).map_err(|_| overflow("image grid value does not fit int64 output"))?;
    }
    Ok(())
}

/// Normalizes and patchifies one already-sized packed-HWC RGB8 still image.
///
/// The frozen Qwen order is merged-block row-major for patch rows, then
/// `[channel, temporal, patch_y, patch_x]` within each row. The single still
/// image is duplicated across the temporal factor before flattening.
///
/// # Errors
///
/// Returns a stable categorized error from [`plan_image_patchify`] before any
/// output is visible, or `resource_limit` if the host cannot reserve the
/// already-validated output allocation.
pub fn patchify_image_rgb8(
    visual: &VisualProfile,
    geometry: ImageGeometryPlan,
    rgb: &[u8],
    limits: ResourceLimits,
) -> Result<PreparedImage> {
    let input_length = u64::try_from(rgb.len())
        .map_err(|_| overflow("RGB input length does not fit capacity arithmetic"))?;
    let plan = plan_image_patchify(visual, geometry, input_length, limits)?;

    let output_elements = usize::try_from(plan.output_elements)
        .map_err(|_| overflow("patch output element count does not fit usize"))?;
    let mut values = Vec::new();
    values.try_reserve_exact(output_elements).map_err(|error| {
        QwenError::new(
            ErrorCategory::ResourceLimit,
            "unable to reserve patch output allocation",
        )
        .with_context("elements", plan.output_elements)
        .with_context("detail", error.to_string())
    })?;
    values.resize(output_elements, 0.0_f32);

    let height = usize::try_from(geometry.height)
        .map_err(|_| overflow("patchify height does not fit usize"))?;
    let width = usize::try_from(geometry.width)
        .map_err(|_| overflow("patchify width does not fit usize"))?;
    let patch = usize::try_from(visual.patch_size)
        .map_err(|_| overflow("patch size does not fit usize"))?;
    let temporal = usize::try_from(visual.temporal_patch_size)
        .map_err(|_| overflow("temporal patch size does not fit usize"))?;
    let merge = usize::try_from(visual.merge_size)
        .map_err(|_| overflow("merge size does not fit usize"))?;
    let grid_height = height / patch;
    let grid_width = width / patch;
    let outer_height = grid_height / merge;
    let outer_width = grid_width / merge;

    let (means, stds) = fused_normalization_constants(visual);
    let mut destination = 0_usize;
    for outer_y in 0..outer_height {
        for outer_x in 0..outer_width {
            for merge_y in 0..merge {
                for merge_x in 0..merge {
                    let source_y = (outer_y * merge + merge_y) * patch;
                    let source_x = (outer_x * merge + merge_x) * patch;
                    for channel in 0..3_usize {
                        for _temporal in 0..temporal {
                            for patch_y in 0..patch {
                                let row = (source_y + patch_y) * width;
                                for patch_x in 0..patch {
                                    let source = (row + source_x + patch_x) * 3 + channel;
                                    values[destination] =
                                        (f32::from(rgb[source]) - means[channel]) / stds[channel];
                                    destination += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    debug_assert_eq!(destination, output_elements);

    let rows = usize::try_from(geometry.patch_rows)
        .map_err(|_| overflow("patch row count does not fit usize"))?;
    let columns = usize::try_from(visual.patch_width)
        .map_err(|_| overflow("patch width does not fit usize"))?;
    let pixel_values = Matrix::new(rows, columns, values)?;
    let image_grid_thw = Matrix::new(
        1,
        3,
        geometry
            .image_grid_thw
            .map(|value| {
                i64::try_from(value)
                    .map_err(|_| overflow("image grid value does not fit int64 output"))
            })
            .into_iter()
            .collect::<Result<Vec<_>>>()?,
    )?;

    Ok(PreparedImage {
        pixel_values,
        image_grid_thw,
    })
}

#[allow(clippy::cast_possible_truncation)]
fn fused_normalization_constants(visual: &VisualProfile) -> ([f32; 3], [f32; 3]) {
    // This intentionally mirrors the pinned Torch backend: construct f32
    // mean/std tensors first, then fuse the 1/255 rescale into both.
    (
        visual
            .image_mean
            .map(|value| value as f32 * RESCALE_DENOMINATOR),
        visual
            .image_std
            .map(|value| value as f32 * RESCALE_DENOMINATOR),
    )
}

fn validate_visual_profile(visual: &VisualProfile) -> Result<()> {
    let spatial_area = checked_mul(
        "patchify spatial patch area",
        visual.patch_size,
        visual.patch_size,
    )?;
    let expected_width = checked_mul(
        "patchify flattened patch width",
        checked_mul(
            "patchify channel-temporal width",
            RGB_CHANNELS,
            visual.temporal_patch_size,
        )?,
        spatial_area,
    )?;
    if visual.patch_size == 0
        || visual.temporal_patch_size == 0
        || visual.merge_size == 0
        || visual.patch_width != expected_width
    {
        return Err(invariant("visual profile patch constants are inconsistent")
            .with_context("expected_patch_width", expected_width)
            .with_context("actual_patch_width", visual.patch_width));
    }
    for channel in 0..3_usize {
        if !visual.image_mean[channel].is_finite()
            || !visual.image_std[channel].is_finite()
            || visual.image_std[channel] <= 0.0
        {
            return Err(
                invariant("visual profile normalization constants are invalid")
                    .with_context("channel", channel),
            );
        }
    }
    Ok(())
}

fn expected_geometry(visual: &VisualProfile, height: u64, width: u64) -> Result<ImageGeometryPlan> {
    if height == 0
        || width == 0
        || !height.is_multiple_of(visual.patch_size)
        || !width.is_multiple_of(visual.patch_size)
    {
        return Err(invariant(
            "patchify geometry dimensions are not positive patch multiples",
        ));
    }
    let grid_height = height / visual.patch_size;
    let grid_width = width / visual.patch_size;
    if !grid_height.is_multiple_of(visual.merge_size)
        || !grid_width.is_multiple_of(visual.merge_size)
    {
        return Err(invariant(
            "patchify grid dimensions are not divisible by merge size",
        ));
    }
    let patch_rows = checked_mul("patchify expected patch rows", grid_height, grid_width)?;
    let merge_area = checked_mul(
        "patchify expected merge area",
        visual.merge_size,
        visual.merge_size,
    )?;
    let placeholder_count = patch_rows / merge_area;
    let rgb_row_stride_bytes = checked_mul("patchify expected RGB stride", width, RGB_CHANNELS)?;
    let rgb_capacity_bytes = checked_mul(
        "patchify expected RGB capacity",
        height,
        rgb_row_stride_bytes,
    )?;
    let pixel_values_row_stride_bytes = checked_mul(
        "patchify expected pixel row stride",
        visual.patch_width,
        F32_BYTES,
    )?;
    let pixel_values_capacity_bytes = checked_mul(
        "patchify expected pixel capacity",
        patch_rows,
        pixel_values_row_stride_bytes,
    )?;
    let image_grid_row_stride_bytes =
        checked_mul("patchify expected grid row stride", GRID_COLUMNS, I64_BYTES)?;

    Ok(ImageGeometryPlan {
        height,
        width,
        image_grid_thw: [1, grid_height, grid_width],
        patch_rows,
        placeholder_count,
        rgb_row_stride_bytes,
        rgb_capacity_bytes,
        pixel_values_row_stride_bytes,
        pixel_values_capacity_bytes,
        image_grid_row_stride_bytes,
        image_grid_capacity_bytes: image_grid_row_stride_bytes,
    })
}

fn check_limit(name: &'static str, actual: u64, limit: u64) -> Result<()> {
    if actual <= limit {
        return Ok(());
    }
    Err(
        QwenError::new(ErrorCategory::ResourceLimit, "resource limit exceeded")
            .with_context("resource", name)
            .with_context("actual", actual)
            .with_context("limit", limit),
    )
}

fn invariant(message: impl Into<String>) -> QwenError {
    QwenError::new(ErrorCategory::InternalInvariant, message)
}

fn overflow(message: impl Into<String>) -> QwenError {
    QwenError::new(ErrorCategory::ArithmeticOverflow, message)
}

#[cfg(test)]
mod tests {
    use super::{patchify_image_rgb8, plan_image_patchify};
    use crate::{
        error::ErrorCategory,
        geometry::{ImageGeometryPlan, plan_image_geometry},
        limits::{LimitOverrides, ResourceLimits},
        profile::{ProfileAlias, ProfileRegistry, VisualProfile},
        request::ImageOptions,
    };

    fn visual(alias: ProfileAlias) -> VisualProfile {
        ProfileRegistry::bundled()
            .expect("bundled profiles")
            .get(alias)
            .visual
            .clone()
    }

    fn geometry(
        visual: &VisualProfile,
        height: u64,
        width: u64,
        options: ImageOptions,
    ) -> ImageGeometryPlan {
        plan_image_geometry(visual, height, width, options, ResourceLimits::default())
            .expect("valid geometry")
    }

    #[allow(clippy::cast_possible_truncation)]
    fn normalized(value: u8, mean: f64, std: f64) -> f32 {
        (f32::from(value) - mean as f32 * 255.0) / (std as f32 * 255.0)
    }

    #[test]
    fn constant_channels_and_temporal_duplication_are_exactly_located() {
        let visual = visual(ProfileAlias::Qwen3Vl8b);
        let geometry = geometry(&visual, 64, 64, ImageOptions::default());
        let mut rgb = Vec::with_capacity(64 * 64 * 3);
        for _ in 0..64 * 64 {
            rgb.extend([0_u8, 127, 255]);
        }
        let prepared = patchify_image_rgb8(&visual, geometry, &rgb, ResourceLimits::default())
            .expect("patchify");
        let row = &prepared.pixel_values.as_slice()[..1536];
        for channel in 0..3_usize {
            let expected = normalized(
                [0_u8, 127, 255][channel],
                visual.image_mean[channel],
                visual.image_std[channel],
            );
            let channel_start = channel * 2 * 16 * 16;
            assert!(
                row[channel_start..channel_start + 512]
                    .iter()
                    .all(|&value| value.to_bits() == expected.to_bits())
            );
            assert_eq!(
                &row[channel_start..channel_start + 256],
                &row[channel_start + 256..channel_start + 512],
                "still temporal planes must be byte-for-byte duplicates"
            );
        }
        assert_eq!(prepared.pixel_values.shape(), [16, 1536]);
        assert_eq!(
            prepared.pixel_values.byte_strides().expect("strides"),
            [6144, 4]
        );
        assert_eq!(prepared.image_grid_thw.as_slice(), &[1, 4, 4]);
        assert_eq!(
            prepared.image_grid_thw.byte_strides().expect("strides"),
            [24, 8]
        );
        let grid_product = prepared
            .image_grid_thw
            .as_slice()
            .iter()
            .copied()
            .product::<i64>();
        assert_eq!(
            grid_product,
            i64::try_from(prepared.pixel_values.rows()).expect("patch rows")
        );
        assert_eq!(
            geometry.placeholder_count * visual.merge_size * visual.merge_size,
            geometry.patch_rows
        );
    }

    #[test]
    fn asymmetric_data_asserts_merged_patch_rows_and_inner_flattening_independently() {
        let visual = visual(ProfileAlias::Qwen3Vl8b);
        let geometry = geometry(&visual, 64, 96, ImageOptions::default());
        let mut rgb = vec![0_u8; 64 * 96 * 3];
        for y in 0..64_usize {
            for x in 0..96_usize {
                for channel in 0..3_usize {
                    rgb[(y * 96 + x) * 3 + channel] =
                        u8::try_from((y * 37 + x * 11 + channel * 83) % 256).expect("coded byte");
                }
            }
        }
        let prepared = patchify_image_rgb8(&visual, geometry, &rgb, ResourceLimits::default())
            .expect("patchify");
        let expected_patch_coordinates = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (0, 4),
            (0, 5),
            (1, 4),
            (1, 5),
            (2, 0),
            (2, 1),
            (3, 0),
            (3, 1),
            (2, 2),
            (2, 3),
            (3, 2),
            (3, 3),
            (2, 4),
            (2, 5),
            (3, 4),
            (3, 5),
        ];
        assert_eq!(prepared.pixel_values.shape(), [24, 1536]);
        for (row, &(patch_y, patch_x)) in expected_patch_coordinates.iter().enumerate() {
            for (channel, temporal, inner_y, inner_x) in
                [(0, 0, 0, 0), (0, 1, 7, 13), (1, 0, 15, 2), (2, 1, 3, 15)]
            {
                let source_y = patch_y * 16 + inner_y;
                let source_x = patch_x * 16 + inner_x;
                let source = (source_y * 96 + source_x) * 3 + channel;
                let column = (((channel * 2 + temporal) * 16 + inner_y) * 16) + inner_x;
                let actual = prepared.pixel_values.as_slice()[row * 1536 + column];
                let expected = normalized(
                    rgb[source],
                    visual.image_mean[channel],
                    visual.image_std[channel],
                );
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "row={row}, column={column}"
                );
            }
        }
    }

    #[test]
    fn odd_source_and_aligned_geometries_materialize_the_b3_destination() {
        let visual = visual(ProfileAlias::Qwen35_9b);
        for (source, destination) in [
            ((65, 95), (64, 96)),
            ((95, 65), (96, 64)),
            ((64, 96), (64, 96)),
        ] {
            let plan = geometry(&visual, source.0, source.1, ImageOptions::default());
            assert_eq!((plan.height, plan.width), destination);
            let rgb = vec![17_u8; usize::try_from(plan.rgb_capacity_bytes).expect("capacity")];
            let prepared = patchify_image_rgb8(&visual, plan, &rgb, ResourceLimits::default())
                .expect("patchify");
            assert_eq!(
                prepared.pixel_values.rows(),
                usize::try_from(plan.patch_rows).unwrap()
            );
            assert_eq!(
                prepared.image_grid_thw.as_slice(),
                &[
                    1,
                    i64::try_from(destination.0 / 16).expect("grid height"),
                    i64::try_from(destination.1 / 16).expect("grid width"),
                ]
            );
        }
    }

    #[test]
    fn minimum_and_maximum_plans_validate_without_large_allocations() {
        let visual = visual(ProfileAlias::Qwen3Vl8b);
        let minimum = geometry(&visual, 32, 32, ImageOptions::default());
        assert_eq!((minimum.height, minimum.width), (64, 64));
        plan_image_patchify(
            &visual,
            minimum,
            minimum.rgb_capacity_bytes,
            ResourceLimits::default(),
        )
        .expect("minimum plan");

        let maximum = geometry(&visual, 4096, 4096, ImageOptions::default());
        assert_eq!(maximum.height * maximum.width, 16_777_216);
        let patchify = plan_image_patchify(
            &visual,
            maximum,
            maximum.rgb_capacity_bytes,
            ResourceLimits::default(),
        )
        .expect("maximum plan");
        assert_eq!(patchify.output_elements(), 65_536 * 1_536);
        assert_eq!(patchify.materialized_bytes(), 402_653_184 + 24);
    }

    #[test]
    fn failures_are_stable_and_happen_in_preflight() {
        let visual = visual(ProfileAlias::Qwen3Vl8b);
        let geometry = geometry(&visual, 64, 64, ImageOptions::default());
        assert_eq!(
            plan_image_patchify(
                &visual,
                geometry,
                geometry.rgb_capacity_bytes - 1,
                ResourceLimits::default(),
            )
            .expect_err("short RGB")
            .category(),
            ErrorCategory::MediaGeometry
        );

        let mut stale = geometry;
        stale.patch_rows += 1;
        assert_eq!(
            plan_image_patchify(
                &visual,
                stale,
                geometry.rgb_capacity_bytes,
                ResourceLimits::default(),
            )
            .expect_err("stale plan")
            .category(),
            ErrorCategory::InternalInvariant
        );
        let low_pixels = ResourceLimits::default()
            .lowered(LimitOverrides {
                prepared_image_pixels_per_occurrence: Some(4_095),
                ..LimitOverrides::default()
            })
            .expect("lower limits");
        assert_eq!(
            plan_image_patchify(&visual, geometry, geometry.rgb_capacity_bytes, low_pixels)
                .expect_err("pixel resource excess")
                .category(),
            ErrorCategory::ResourceLimit
        );

        let low_output = ResourceLimits::default()
            .lowered(LimitOverrides {
                materialized_output_bytes_per_batch: Some(98_327),
                ..LimitOverrides::default()
            })
            .expect("lower limits");
        assert_eq!(
            plan_image_patchify(&visual, geometry, geometry.rgb_capacity_bytes, low_output)
                .expect_err("output resource excess")
                .category(),
            ErrorCategory::ResourceLimit
        );

        let mut overflow_visual = visual.clone();
        overflow_visual.patch_size = u64::MAX;
        assert_eq!(
            plan_image_patchify(
                &overflow_visual,
                geometry,
                geometry.rgb_capacity_bytes,
                ResourceLimits::default(),
            )
            .expect_err("capacity arithmetic overflow")
            .category(),
            ErrorCategory::ArithmeticOverflow
        );
    }

    #[test]
    fn every_derived_geometry_field_is_revalidated_before_execution() {
        let visual = visual(ProfileAlias::Qwen3Vl8b);
        let geometry = geometry(&visual, 64, 64, ImageOptions::default());
        for stale in [
            ImageGeometryPlan {
                rgb_row_stride_bytes: geometry.rgb_row_stride_bytes + 1,
                ..geometry
            },
            ImageGeometryPlan {
                rgb_capacity_bytes: geometry.rgb_capacity_bytes + 1,
                ..geometry
            },
            ImageGeometryPlan {
                pixel_values_row_stride_bytes: geometry.pixel_values_row_stride_bytes + 4,
                ..geometry
            },
            ImageGeometryPlan {
                pixel_values_capacity_bytes: geometry.pixel_values_capacity_bytes + 4,
                ..geometry
            },
            ImageGeometryPlan {
                image_grid_row_stride_bytes: geometry.image_grid_row_stride_bytes + 8,
                ..geometry
            },
            ImageGeometryPlan {
                image_grid_capacity_bytes: geometry.image_grid_capacity_bytes + 8,
                ..geometry
            },
            ImageGeometryPlan {
                image_grid_thw: [1, 4, 5],
                ..geometry
            },
            ImageGeometryPlan {
                placeholder_count: geometry.placeholder_count + 1,
                ..geometry
            },
        ] {
            assert_eq!(
                plan_image_patchify(
                    &visual,
                    stale,
                    stale.rgb_capacity_bytes,
                    ResourceLimits::default(),
                )
                .expect_err("stale capacity or grid field")
                .category(),
                ErrorCategory::InternalInvariant
            );
        }
    }
}
