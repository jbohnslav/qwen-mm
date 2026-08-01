//! Allocation-free image resize selection and output geometry planning.

use crate::{
    error::{ErrorCategory, QwenError, Result},
    limits::{ResourceLimits, checked_add, checked_capacity_bytes, checked_mul},
    profile::VisualProfile,
    request::ImageOptions,
};

const MAX_ASPECT_RATIO: u64 = 200;
const DEFAULT_MIN_TOKENS: u64 = 4;
const DEFAULT_MAX_TOKENS: u64 = 16_384;
const RGB_CHANNELS: u64 = 3;
const F32_BYTES: u64 = 4;
const I64_BYTES: u64 = 8;
const GRID_COLUMNS: u64 = 3;

/// Every checked dimension, count, stride, and capacity needed by the image
/// preparation stages after decoding.
///
/// The plan is a fixed-size value. Constructing it performs no allocation and
/// work does not scale with the source or destination pixel count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageGeometryPlan {
    /// Prepared image height in pixels.
    pub height: u64,
    /// Prepared image width in pixels.
    pub width: u64,
    /// Exact `[temporal, height, width]` image patch grid.
    pub image_grid_thw: [u64; 3],
    /// Rows in the flattened `pixel_values` matrix.
    pub patch_rows: u64,
    /// Visual placeholder tokens after spatial merging.
    pub placeholder_count: u64,
    /// Packed RGB destination row stride in bytes.
    pub rgb_row_stride_bytes: u64,
    /// Packed RGB destination capacity in bytes.
    pub rgb_capacity_bytes: u64,
    /// `pixel_values` row stride in bytes.
    pub pixel_values_row_stride_bytes: u64,
    /// `pixel_values` capacity in bytes.
    pub pixel_values_capacity_bytes: u64,
    /// `image_grid_thw` row stride in bytes.
    pub image_grid_row_stride_bytes: u64,
    /// One `image_grid_thw` row's capacity in bytes.
    pub image_grid_capacity_bytes: u64,
}

/// Rounds a non-negative integer to the nearest factor multiple using
/// Python's ties-to-even rule.
///
/// This uses quotient and remainder arithmetic, so the result is independent
/// of the host floating-point rounding mode and remains exact above `2^53`.
///
/// # Errors
///
/// Returns `media_geometry` for a zero factor and `arithmetic_overflow` if the
/// rounded multiple is not representable as `u64`.
pub fn round_by_factor(number: u64, factor: u64) -> Result<u64> {
    if factor == 0 {
        return Err(geometry("resize factor must be non-zero"));
    }

    let quotient = number / factor;
    let remainder = number % factor;
    let distance_up = factor - remainder;
    let round_up = remainder > distance_up || (remainder == distance_up && quotient % 2 == 1);
    let rounded_quotient = if round_up {
        checked_add("ties-to-even factor quotient", quotient, 1)?
    } else {
        quotient
    };
    checked_mul("ties-to-even factor multiple", rounded_quotient, factor)
}

/// Applies the frozen Qwen VL Utils `smart_resize` operation.
///
/// Omitted budgets are `4 * factor^2` and `16_384 * factor^2`, exactly as in
/// Qwen VL Utils 0.0.14. The v1 image factor is 32.
///
/// # Errors
///
/// Returns `media_geometry` for invalid dimensions, factor, budgets, or aspect
/// ratio, and `arithmetic_overflow` when required integer arithmetic exceeds
/// `u64`.
#[allow(clippy::cast_precision_loss)] // Python's frozen sqrt path converts these integers to f64.
pub fn smart_resize(
    height: u64,
    width: u64,
    factor: u64,
    min_pixels: Option<u64>,
    max_pixels: Option<u64>,
) -> Result<[u64; 2]> {
    if factor == 0 {
        return Err(geometry("resize factor must be non-zero"));
    }
    let max_pixels = match max_pixels {
        Some(value) => value,
        None => checked_mul(
            "default maximum prepared pixels",
            DEFAULT_MAX_TOKENS,
            checked_mul("resize factor squared", factor, factor)?,
        )?,
    };
    let min_pixels = match min_pixels {
        Some(value) => value,
        None => checked_mul(
            "default minimum prepared pixels",
            DEFAULT_MIN_TOKENS,
            checked_mul("resize factor squared", factor, factor)?,
        )?,
    };
    if max_pixels < min_pixels {
        return Err(
            geometry("max_pixels must be greater than or equal to min_pixels")
                .with_context("min_pixels", min_pixels)
                .with_context("max_pixels", max_pixels),
        );
    }
    if max_pixels == 0 {
        return Err(geometry("max_pixels must be non-zero"));
    }
    validate_dimensions(height, width)?;

    // Check this product once before any conversion to f64. Python integers do
    // not overflow, but the public parity capacity model is deliberately u64.
    let source_pixels = checked_mul("source image pixels", height, width)?;
    let rounded_height = factor.max(round_by_factor(height, factor)?);
    let rounded_width = factor.max(round_by_factor(width, factor)?);
    let rounded_pixels = checked_mul("factor-rounded image pixels", rounded_height, rounded_width)?;

    let [resized_height, resized_width] = if rounded_pixels > max_pixels {
        let beta = ((source_pixels as f64) / (max_pixels as f64)).sqrt();
        [
            scaled_by_factor((height as f64) / beta, factor, ScaleDirection::Floor)?,
            scaled_by_factor((width as f64) / beta, factor, ScaleDirection::Floor)?,
        ]
    } else if rounded_pixels < min_pixels {
        let beta = ((min_pixels as f64) / (source_pixels as f64)).sqrt();
        [
            scaled_by_factor((height as f64) * beta, factor, ScaleDirection::Ceil)?,
            scaled_by_factor((width as f64) * beta, factor, ScaleDirection::Ceil)?,
        ]
    } else {
        [rounded_height, rounded_width]
    };

    if resized_height == 0 || resized_width == 0 {
        return Err(geometry("pixel budget produces a zero prepared dimension"));
    }
    // Make the returned destination product representable before it reaches a
    // decoder, resize kernel, or allocation.
    checked_mul("prepared image pixels", resized_height, resized_width)?;
    Ok([resized_height, resized_width])
}

/// Validates paired explicit image dimensions.
///
/// # Errors
///
/// Returns `media_geometry` for an unpaired, zero, or excessive-aspect pair,
/// and `arithmetic_overflow` if aspect validation cannot be represented.
pub fn explicit_dimensions(options: ImageOptions) -> Result<Option<[u64; 2]>> {
    match (options.resized_height, options.resized_width) {
        (None, None) => Ok(None),
        (Some(height), Some(width)) => {
            validate_dimensions(height, width)?;
            Ok(Some([height, width]))
        }
        _ => Err(geometry(
            "resized_height and resized_width must be supplied together",
        )),
    }
}

/// Plans exact image dimensions and downstream capacities without allocating.
///
/// Explicit dimensions follow the composed Python path: their pair becomes the
/// input to `smart_resize` with default budgets. Per-item min/max values are
/// validated but apply only when explicit dimensions are absent.
///
/// # Errors
///
/// Returns the first stable `resource_limit`, `media_geometry`, or
/// `arithmetic_overflow` error established by contract-v1 validation order.
pub fn plan_image_geometry(
    visual: &VisualProfile,
    source_height: u64,
    source_width: u64,
    options: ImageOptions,
    limits: ResourceLimits,
) -> Result<ImageGeometryPlan> {
    preflight_image_resources(source_height, source_width, options, limits)?;
    validate_dimensions(source_height, source_width)?;
    if options
        .min_pixels
        .zip(options.max_pixels)
        .is_some_and(|(min, max)| max < min)
    {
        return Err(geometry(
            "max_pixels must be greater than or equal to min_pixels",
        ));
    }
    let explicit = explicit_dimensions(options)?;
    let factor = checked_mul(
        "effective image resize factor",
        visual.patch_size,
        visual.merge_size,
    )?;
    let [height, width] = if let Some([height, width]) = explicit {
        smart_resize(height, width, factor, None, None)?
    } else {
        smart_resize(
            source_height,
            source_width,
            factor,
            options
                .min_pixels
                .or(Some(visual.composed_image_min_pixels)),
            options
                .max_pixels
                .or(Some(visual.composed_image_max_pixels)),
        )?
    };

    let prepared_pixels = checked_mul("prepared image pixels", height, width)?;
    check_limit(
        "prepared_image_pixels_per_occurrence",
        prepared_pixels,
        limits.prepared_image_pixels_per_occurrence(),
    )?;
    build_image_plan(visual, height, width)
}

fn preflight_image_resources(
    source_height: u64,
    source_width: u64,
    options: ImageOptions,
    limits: ResourceLimits,
) -> Result<()> {
    // Resource preflight precedes geometry in contract v1. Arithmetic is
    // checked before a numeric limit so overflow cannot masquerade as excess.
    let source_pixels = checked_mul("decoded source pixels", source_height, source_width)?;
    check_limit(
        "decoded_source_pixels",
        source_pixels,
        limits.decoded_pixels_per_image_or_frame(),
    )?;
    check_limit(
        "decoded_edge_length",
        source_height,
        limits.decoded_edge_length(),
    )?;
    check_limit(
        "decoded_edge_length",
        source_width,
        limits.decoded_edge_length(),
    )?;
    for (option, value) in [
        ("min_pixels", options.min_pixels),
        ("max_pixels", options.max_pixels),
    ] {
        if let Some(value) = value {
            check_limit(
                "prepared_image_pixels_per_occurrence",
                value,
                limits.prepared_image_pixels_per_occurrence(),
            )
            .map_err(|error| error.with_context("option", option))?;
        }
    }
    if let (Some(height), Some(width)) = (options.resized_height, options.resized_width) {
        let pixels = checked_mul("explicit resized image pixels", height, width)?;
        check_limit(
            "prepared_image_pixels_per_occurrence",
            pixels,
            limits.prepared_image_pixels_per_occurrence(),
        )?;
    }
    Ok(())
}

fn build_image_plan(visual: &VisualProfile, height: u64, width: u64) -> Result<ImageGeometryPlan> {
    if !height.is_multiple_of(visual.patch_size) || !width.is_multiple_of(visual.patch_size) {
        return Err(invariant(
            "smart_resize result is not divisible by the spatial patch size",
        ));
    }
    let grid_height = height / visual.patch_size;
    let grid_width = width / visual.patch_size;
    let patch_rows = checked_mul("image patch rows", grid_height, grid_width)?;
    let merge_area = checked_mul("spatial merge area", visual.merge_size, visual.merge_size)?;
    if merge_area == 0 || patch_rows % merge_area != 0 {
        return Err(invariant(
            "image patch grid is not divisible by the spatial merge area",
        ));
    }
    let placeholder_count = patch_rows / merge_area;

    let rgb_row_stride_bytes = checked_mul("packed RGB row stride", width, RGB_CHANNELS)?;
    let rgb_capacity_bytes = checked_mul(
        "packed RGB destination capacity",
        height,
        rgb_row_stride_bytes,
    )?;
    let pixel_values_row_stride_bytes = checked_mul(
        "pixel_values row byte stride",
        visual.patch_width,
        F32_BYTES,
    )?;
    let pixel_values_capacity_bytes = checked_mul(
        "pixel_values byte capacity",
        patch_rows,
        pixel_values_row_stride_bytes,
    )?;
    let image_grid_row_stride_bytes =
        checked_mul("image_grid_thw row byte stride", GRID_COLUMNS, I64_BYTES)?;
    let image_grid_capacity_bytes = checked_capacity_bytes(1, GRID_COLUMNS, I64_BYTES)?;

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
        image_grid_capacity_bytes,
    })
}

#[derive(Clone, Copy)]
enum ScaleDirection {
    Floor,
    Ceil,
}

#[allow(clippy::cast_precision_loss)] // The frozen Python scaling path operates in f64.
fn scaled_by_factor(value: f64, factor: u64, direction: ScaleDirection) -> Result<u64> {
    if !value.is_finite() || value < 0.0 {
        return Err(QwenError::new(
            ErrorCategory::ArithmeticOverflow,
            "scaled dimension is not a finite non-negative value",
        ));
    }
    let quotient = value / (factor as f64);
    let rounded = match direction {
        ScaleDirection::Floor => quotient.floor(),
        ScaleDirection::Ceil => quotient.ceil(),
    };
    // `u64::MAX as f64` rounds to 2^64, so equality is already outside the
    // set of exactly convertible u64 values.
    if !rounded.is_finite() || rounded < 0.0 || rounded >= (u64::MAX as f64) {
        return Err(QwenError::new(
            ErrorCategory::ArithmeticOverflow,
            "scaled factor quotient does not fit u64",
        ));
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    let rounded = rounded as u64;
    checked_mul("scaled factor multiple", rounded, factor)
}

fn validate_dimensions(height: u64, width: u64) -> Result<()> {
    if height == 0 || width == 0 {
        return Err(geometry("media dimensions must be non-zero")
            .with_context("height", height)
            .with_context("width", width));
    }
    let (larger, smaller) = if height >= width {
        (height, width)
    } else {
        (width, height)
    };
    let maximum = checked_mul("aspect ratio boundary", smaller, MAX_ASPECT_RATIO)?;
    if larger > maximum {
        return Err(geometry("media aspect ratio exceeds 200")
            .with_context("height", height)
            .with_context("width", width));
    }
    Ok(())
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

fn geometry(message: impl Into<String>) -> QwenError {
    QwenError::new(ErrorCategory::MediaGeometry, message)
}

fn invariant(message: impl Into<String>) -> QwenError {
    QwenError::new(ErrorCategory::InternalInvariant, message)
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::{
        ImageGeometryPlan, explicit_dimensions, plan_image_geometry, round_by_factor, smart_resize,
    };
    use crate::{
        error::ErrorCategory,
        limits::{LimitOverrides, ResourceLimits, checked_mul},
        profile::{ProfileAlias, ProfileRegistry},
        request::ImageOptions,
    };

    const RULES: &str = include_str!("../../../reference/conformance/v1/rules.json");

    fn visual() -> crate::profile::VisualProfile {
        ProfileRegistry::bundled()
            .expect("bundled profiles")
            .get(ProfileAlias::Qwen3Vl8b)
            .visual
            .clone()
    }

    fn plan(height: u64, width: u64, options: ImageOptions) -> ImageGeometryPlan {
        plan_image_geometry(&visual(), height, width, options, ResourceLimits::default())
            .expect("valid image geometry")
    }

    fn input_u64(input: &Value, name: &str) -> u64 {
        input[name].as_u64().expect("fixture u64")
    }

    #[test]
    fn every_a3_image_geometry_rule_matches_the_committed_oracle() {
        let document: Value = serde_json::from_str(RULES).expect("valid A3 rules JSON");
        for rule in document["rules"].as_array().expect("rules array") {
            let kind = rule["kind"].as_str().expect("rule kind");
            if !matches!(
                kind,
                "round_by_factor" | "smart_resize" | "explicit_dimensions"
            ) {
                continue;
            }
            let id = rule["id"].as_str().expect("rule id");
            let input = &rule["input"];
            let result = match kind {
                "round_by_factor" => {
                    round_by_factor(input_u64(input, "number"), input_u64(input, "factor"))
                        .map(Value::from)
                }
                "smart_resize" => smart_resize(
                    input_u64(input, "height"),
                    input_u64(input, "width"),
                    input_u64(input, "factor"),
                    input.get("min_pixels").and_then(Value::as_u64),
                    input.get("max_pixels").and_then(Value::as_u64),
                )
                .map(|[height, width]| serde_json::json!([height, width])),
                "explicit_dimensions" => {
                    let options = ImageOptions {
                        resized_height: input.get("resized_height").and_then(Value::as_u64),
                        resized_width: input.get("resized_width").and_then(Value::as_u64),
                        ..ImageOptions::default()
                    };
                    explicit_dimensions(options)
                        .map(|value| value.map_or(Value::Null, |pair| serde_json::json!(pair)))
                }
                _ => unreachable!(),
            };

            if let Some(expected) = rule.get("expected") {
                assert_eq!(
                    result.unwrap_or_else(|error| panic!("{id}: {error}")),
                    *expected,
                    "{id}"
                );
            } else {
                assert_eq!(
                    result.expect_err(id).category(),
                    ErrorCategory::MediaGeometry,
                    "{id}"
                );
            }
        }
    }

    #[test]
    fn ties_to_even_cover_below_at_and_above_each_factor_half() {
        for (half, below, tied, above) in [
            (16, 0, 0, 32),
            (48, 32, 64, 64),
            (80, 64, 64, 96),
            (112, 96, 128, 128),
        ] {
            assert_eq!(round_by_factor(half - 1, 32).expect("below"), below);
            assert_eq!(round_by_factor(half, 32).expect("tie"), tied);
            assert_eq!(round_by_factor(half + 1, 32).expect("above"), above);
        }
        assert_eq!(
            round_by_factor(u64::MAX - 31, 32).expect("large aligned"),
            u64::MAX - 31
        );
        assert_eq!(
            round_by_factor(u64::MAX, 32)
                .expect_err("rounding beyond u64")
                .category(),
            ErrorCategory::ArithmeticOverflow
        );
    }

    #[test]
    fn smart_resize_transition_neighborhoods_and_no_op_alignment() {
        let oracle_cases = [
            (31, 31, None, None, [64, 64]),
            (32, 32, None, None, [64, 64]),
            (33, 33, None, None, [64, 64]),
            (63, 63, None, None, [64, 64]),
            (64, 64, None, None, [64, 64]),
            (65, 65, None, None, [64, 64]),
            (4_079, 4_079, None, None, [4_064, 4_064]),
            (4_080, 4_080, None, None, [4_096, 4_096]),
            (4_081, 4_081, None, None, [4_096, 4_096]),
            (64, 95, Some(6_143), Some(6_143), [64, 64]),
            (64, 96, Some(6_144), Some(6_144), [64, 96]),
            (64, 97, Some(6_145), Some(6_145), [64, 128]),
            (32, 128, Some(4_095), Some(16_777_216), [32, 128]),
            (32, 128, Some(4_096), Some(16_777_216), [32, 128]),
            (32, 128, Some(4_097), Some(16_777_216), [64, 160]),
            (128, 128, Some(4_096), Some(16_383), [96, 96]),
            (128, 128, Some(4_096), Some(16_384), [128, 128]),
            (128, 128, Some(4_096), Some(16_385), [128, 128]),
        ];
        for (height, width, min_pixels, max_pixels, expected) in oracle_cases {
            assert_eq!(
                smart_resize(height, width, 32, min_pixels, max_pixels).expect("resize"),
                expected,
                "{height}x{width}, min={min_pixels:?}, max={max_pixels:?}"
            );
        }
        assert_eq!(
            smart_resize(64, 96, 32, None, None).expect("no-op"),
            [64, 96]
        );
    }

    #[test]
    fn resize_is_symmetric_for_portrait_and_landscape_inputs() {
        for (height, width, min, max) in [
            (31, 127, None, None),
            (2_000, 1_000, Some(4_096), Some(65_536)),
            (100, 20_000, None, None),
        ] {
            let forward = smart_resize(height, width, 32, min, max).expect("forward");
            let reverse = smart_resize(width, height, 32, min, max).expect("reverse");
            assert_eq!(forward, [reverse[1], reverse[0]]);
        }
    }

    #[test]
    fn plans_have_exact_grid_counts_strides_and_capacities() {
        let plan = plan(64, 96, ImageOptions::default());
        assert_eq!(plan.height, 64);
        assert_eq!(plan.width, 96);
        assert_eq!(plan.image_grid_thw, [1, 4, 6]);
        assert_eq!(plan.patch_rows, 24);
        assert_eq!(plan.placeholder_count, 6);
        assert_eq!(plan.rgb_row_stride_bytes, 288);
        assert_eq!(plan.rgb_capacity_bytes, 18_432);
        assert_eq!(plan.pixel_values_row_stride_bytes, 6_144);
        assert_eq!(plan.pixel_values_capacity_bytes, 147_456);
        assert_eq!(plan.image_grid_row_stride_bytes, 24);
        assert_eq!(plan.image_grid_capacity_bytes, 24);
    }

    #[test]
    fn explicit_dimensions_are_factor_aligned_with_default_budgets() {
        let plan = plan(
            320,
            240,
            ImageOptions {
                min_pixels: Some(1),
                max_pixels: Some(16_777_216),
                resized_height: Some(65),
                resized_width: Some(95),
            },
        );
        assert_eq!([plan.height, plan.width], [64, 96]);
    }

    #[test]
    fn validation_categories_and_precedence_are_stable() {
        assert_eq!(
            smart_resize(0, 1, 32, None, None)
                .expect_err("zero dimension")
                .category(),
            ErrorCategory::MediaGeometry
        );
        assert_eq!(
            smart_resize(1, 201, 32, None, None)
                .expect_err("aspect")
                .category(),
            ErrorCategory::MediaGeometry
        );
        assert_eq!(
            smart_resize(64, 64, 32, Some(2), Some(1))
                .expect_err("budget order")
                .category(),
            ErrorCategory::MediaGeometry
        );
        assert_eq!(
            explicit_dimensions(ImageOptions {
                resized_height: Some(64),
                ..ImageOptions::default()
            })
            .expect_err("unpaired")
            .category(),
            ErrorCategory::MediaGeometry
        );

        let low_limit = ResourceLimits::default()
            .lowered(LimitOverrides {
                decoded_edge_length: Some(63),
                ..LimitOverrides::default()
            })
            .expect("lowered limits");
        let error = plan_image_geometry(
            &visual(),
            64,
            64,
            ImageOptions {
                resized_height: Some(1),
                ..ImageOptions::default()
            },
            low_limit,
        )
        .expect_err("resources precede invalid option pairing");
        assert_eq!(error.category(), ErrorCategory::ResourceLimit);

        for (width, accepted) in [(199, true), (200, true), (201, false)] {
            let result = smart_resize(1, width, 32, None, None);
            assert_eq!(result.is_ok(), accepted, "aspect ratio 1:{width}");
        }

        let low_prepared_limit = ResourceLimits::default()
            .lowered(LimitOverrides {
                prepared_image_pixels_per_occurrence: Some(4_095),
                ..LimitOverrides::default()
            })
            .expect("lowered prepared limit");
        assert_eq!(
            plan_image_geometry(
                &visual(),
                64,
                64,
                ImageOptions::default(),
                low_prepared_limit,
            )
            .expect_err("planned prepared pixels exceed runtime policy")
            .category(),
            ErrorCategory::ResourceLimit
        );
    }

    #[test]
    fn overflow_is_distinct_from_geometry_and_resource_excess() {
        assert_eq!(
            smart_resize(u64::MAX, u64::MAX, 32, None, None)
                .expect_err("large dimension arithmetic")
                .category(),
            ErrorCategory::ArithmeticOverflow
        );
        assert_eq!(
            checked_mul("A3 largest", 4_294_967_295, 4_294_967_297).expect("largest u64 product"),
            u64::MAX
        );
        assert_eq!(
            checked_mul("A3 overflow", 4_294_967_296, 4_294_967_296)
                .expect_err("one beyond u64")
                .category(),
            ErrorCategory::ArithmeticOverflow
        );

        let mut oversized_patch = visual();
        oversized_patch.patch_width = u64::MAX;
        assert_eq!(
            plan_image_geometry(
                &oversized_patch,
                64,
                64,
                ImageOptions::default(),
                ResourceLimits::default(),
            )
            .expect_err("pixel row stride overflows")
            .category(),
            ErrorCategory::ArithmeticOverflow
        );
    }
}
