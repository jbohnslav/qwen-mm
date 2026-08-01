//! Always-on B4 stage comparison against the committed A2 oracle.

use serde_json::Value;

use crate::{
    ProfileAlias, ProfileRegistry, ResourceLimits, patchify_image_rgb8, plan_image_geometry,
    request::ImageOptions,
};

const A2_MULTIMODAL_SMOKE: &str =
    include_str!("../../../reference/goldens/v1/qwen3-vl-8b/multimodal-smoke/manifest.json");

#[derive(Debug)]
struct NumericDiagnostics {
    max_absolute: f64,
    rmse: f64,
    p50_absolute: f64,
    p90_absolute: f64,
    p99_absolute: f64,
    count_over_bound: usize,
    max_ulp: u32,
    first_difference: Option<String>,
}

impl NumericDiagnostics {
    fn render(&self) -> String {
        format!(
            "max_absolute={:.9e}, rmse={:.9e}, p50_absolute={:.9e}, \
             p90_absolute={:.9e}, p99_absolute={:.9e}, count_over_bound={}, \
             max_ulp={}, first_difference={}",
            self.max_absolute,
            self.rmse,
            self.p50_absolute,
            self.p90_absolute,
            self.p99_absolute,
            self.count_over_bound,
            self.max_ulp,
            self.first_difference.as_deref().unwrap_or("none")
        )
    }
}

#[test]
fn committed_a2_prepared_rgb_matches_every_patch_for_both_profiles() {
    let manifest: Value = serde_json::from_str(A2_MULTIMODAL_SMOKE).expect("valid A2 manifest");
    let rgb_descriptor = &manifest["stages"]["prepared_media"]["images"][0]["array"];
    assert_eq!(rgb_descriptor["shape"], serde_json::json!([64, 64, 3]));
    assert_eq!(rgb_descriptor["dtype"], "uint8");
    assert_eq!(rgb_descriptor["strides"], serde_json::json!([192, 3, 1]));
    let rgb = flatten_u8(&rgb_descriptor["data"]);

    let expected_descriptor = &manifest["output"]["arrays"]["pixel_values"];
    assert_eq!(expected_descriptor["shape"], serde_json::json!([16, 1536]));
    assert_eq!(expected_descriptor["dtype"], "float32");
    assert_eq!(expected_descriptor["strides"], serde_json::json!([6144, 4]));
    let expected = flatten_f32(&expected_descriptor["data"]);
    let expected_grid = flatten_i64(&manifest["output"]["arrays"]["image_grid_thw"]["data"]);
    let tolerance = manifest["comparison_policy"]["numeric"]["normalize_and_patchify"]["atol"]
        .as_f64()
        .expect("numeric tolerance");

    let registry = ProfileRegistry::bundled().expect("bundled profiles");
    for alias in [ProfileAlias::Qwen3Vl8b, ProfileAlias::Qwen35_9b] {
        let visual = &registry.get(alias).visual;
        let geometry = plan_image_geometry(
            visual,
            64,
            64,
            ImageOptions::default(),
            ResourceLimits::default(),
        )
        .expect("oracle geometry");
        let actual = patchify_image_rgb8(visual, geometry, &rgb, ResourceLimits::default())
            .expect("oracle patchify");

        assert_eq!(actual.pixel_values.shape(), [16, 1536], "{alias:?}");
        assert_eq!(
            actual.pixel_values.byte_strides().expect("pixel strides"),
            [6144, 4],
            "{alias:?}"
        );
        assert_eq!(actual.image_grid_thw.as_slice(), expected_grid, "{alias:?}");
        assert_eq!(
            actual.image_grid_thw.byte_strides().expect("grid strides"),
            [24, 8],
            "{alias:?}"
        );

        let diagnostics = diagnose(
            &expected,
            actual.pixel_values.as_slice(),
            tolerance,
            2,
            16,
            2,
            4,
        );
        assert_eq!(
            diagnostics.count_over_bound,
            0,
            "A2 normalize/patchify mismatch for {}: {}",
            alias.as_str(),
            diagnostics.render()
        );
    }
}

#[test]
fn failure_diagnostics_include_every_a3_numeric_field_and_decoded_coordinate() {
    let expected = vec![0.0_f32; 1536];
    let mut actual = expected.clone();
    actual[1024 + 3 * 16 + 7] = 0.25;
    let diagnostics = diagnose(&expected, &actual, 1.0e-6, 2, 16, 2, 2);
    let rendered = diagnostics.render();
    for field in [
        "max_absolute=",
        "rmse=",
        "p50_absolute=",
        "p90_absolute=",
        "p99_absolute=",
        "count_over_bound=",
        "max_ulp=",
        "request=0",
        "media_occurrence=0",
        "grid_row=0",
        "patch=0",
        "temporal=0",
        "channel=2",
        "patch_y=3",
        "patch_x=7",
    ] {
        assert!(rendered.contains(field), "missing {field}: {rendered}");
    }
}

#[allow(clippy::cast_precision_loss)]
fn diagnose(
    expected: &[f32],
    actual: &[f32],
    atol: f64,
    temporal: usize,
    patch: usize,
    merge: usize,
    grid_width: usize,
) -> NumericDiagnostics {
    assert_eq!(
        expected.len(),
        actual.len(),
        "stage arrays must have equal shapes"
    );
    let mut absolute = Vec::with_capacity(expected.len());
    let mut squared_sum = 0.0_f64;
    let mut count_over_bound = 0_usize;
    let mut max_ulp = 0_u32;
    let mut first_index = None;
    for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        let difference = (f64::from(actual) - f64::from(expected)).abs();
        absolute.push(difference);
        squared_sum += difference * difference;
        max_ulp = max_ulp.max(ulp_distance(expected, actual));
        if difference > atol {
            count_over_bound += 1;
            first_index.get_or_insert(index);
        }
    }
    absolute.sort_by(f64::total_cmp);
    let max_absolute = absolute.last().copied().unwrap_or(0.0);
    let rmse = if absolute.is_empty() {
        0.0
    } else {
        (squared_sum / absolute.len() as f64).sqrt()
    };

    NumericDiagnostics {
        max_absolute,
        rmse,
        p50_absolute: percentile(&absolute, 0.50),
        p90_absolute: percentile(&absolute, 0.90),
        p99_absolute: percentile(&absolute, 0.99),
        count_over_bound,
        max_ulp,
        first_difference: first_index
            .map(|index| decode_coordinate(index, temporal, patch, merge, grid_width)),
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = quantile * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let fraction = rank - lower as f64;
    sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
}

fn ulp_distance(left: f32, right: f32) -> u32 {
    ordered_float_bits(left).abs_diff(ordered_float_bits(right))
}

fn ordered_float_bits(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 == 0 {
        bits | 0x8000_0000
    } else {
        !bits
    }
}

fn decode_coordinate(
    flat_index: usize,
    temporal: usize,
    patch: usize,
    merge: usize,
    grid_width: usize,
) -> String {
    let patch_width = 3 * temporal * patch * patch;
    let patch_row = flat_index / patch_width;
    let mut column = flat_index % patch_width;
    let channel = column / (temporal * patch * patch);
    column %= temporal * patch * patch;
    let temporal_offset = column / (patch * patch);
    column %= patch * patch;
    let patch_y = column / patch;
    let patch_x = column % patch;

    let rows_per_block = merge * merge;
    let block = patch_row / rows_per_block;
    let within_block = patch_row % rows_per_block;
    let outer_width = grid_width / merge;
    let grid_y = (block / outer_width) * merge + within_block / merge;
    let grid_x = (block % outer_width) * merge + within_block % merge;
    format!(
        "request=0, media_occurrence=0, grid_row=0, patch={patch_row}, \
         grid_y={grid_y}, grid_x={grid_x}, temporal={temporal_offset}, \
         channel={channel}, patch_y={patch_y}, patch_x={patch_x}, column={}",
        flat_index % patch_width
    )
}

fn flatten_u8(value: &Value) -> Vec<u8> {
    let mut output = Vec::new();
    flatten_numbers(value, &mut |number| {
        output.push(u8::try_from(number.as_u64().expect("u8 JSON value")).expect("u8 range"));
    });
    output
}

fn flatten_i64(value: &Value) -> Vec<i64> {
    let mut output = Vec::new();
    flatten_numbers(value, &mut |number| {
        output.push(number.as_i64().expect("i64 JSON value"));
    });
    output
}

#[allow(clippy::cast_possible_truncation)]
fn flatten_f32(value: &Value) -> Vec<f32> {
    let mut output = Vec::new();
    flatten_numbers(value, &mut |number| {
        output.push(number.as_f64().expect("f32 JSON value") as f32);
    });
    output
}

fn flatten_numbers(value: &Value, push: &mut impl FnMut(&serde_json::Number)) {
    match value {
        Value::Array(values) => {
            for value in values {
                flatten_numbers(value, push);
            }
        }
        Value::Number(number) => push(number),
        other => panic!("expected nested numeric JSON array, got {other}"),
    }
}
