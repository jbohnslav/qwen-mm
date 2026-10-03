//! Checked composed video resizing, temporal metadata, and fused patch packing.

use rayon::prelude::*;
use sha2::{Digest, Sha256};

use crate::{
    error::{ErrorCategory, QwenError, Result},
    geometry::{ImageGeometryPlan, build_image_plan, smart_resize},
    limits::{ResourceLimits, checked_add, checked_mul},
    profile::VisualProfile,
    request::{ImageOptions, Rgb8, VideoInput, VideoOptions},
    resize::{VideoFrameResizer, resize_image_rgb8},
};

/// Exact prepared video grid and checked model-output capacities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VideoGeometryPlan {
    /// Prepared frame height.
    pub height: u64,
    /// Prepared frame width.
    pub width: u64,
    /// Number of frames including repeated last-frame temporal padding.
    pub padded_frames: u64,
    /// `[temporal, height, width]` patch grid.
    pub video_grid_thw: [u64; 3],
    /// Rows in `pixel_values_videos`.
    pub patch_rows: u64,
    /// Number of video placeholder tokens.
    pub placeholder_count: u64,
    /// Exact prepared float RGB storage for original decoded frames.
    pub rgb_capacity_bytes: u64,
    /// Exact `pixel_values_videos` capacity.
    pub pixel_values_capacity_bytes: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct VideoMetadata {
    pub(crate) timestamps: Vec<f64>,
    pub(crate) fps: f64,
    pub(crate) sample_fps: f64,
    pub(crate) frames_indices: Vec<i64>,
    pub(crate) total_num_frames: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct VideoPreparationPlan<'a> {
    input: VideoInput<'a>,
    pub(crate) geometry: VideoGeometryPlan,
    pub(crate) metadata: VideoMetadata,
    first_resize: Option<ImageGeometryPlan>,
    final_resize: ImageGeometryPlan,
    pub(crate) cache_key: [u8; 32],
}

#[derive(Debug)]
pub(crate) struct PreparedRgbVideo {
    pub(crate) frames: Vec<Vec<f32>>,
    pub(crate) geometry: VideoGeometryPlan,
    pub(crate) metadata: VideoMetadata,
    pub(crate) source_height: u64,
    pub(crate) source_width: u64,
    pub(crate) cache_key: [u8; 32],
}

/// Plans a decoded clip without sampling or allocating pixel buffers.
#[allow(clippy::too_many_lines)]
pub(crate) fn plan_video_rgb8<'a>(
    input: VideoInput<'a>,
    visual: &VisualProfile,
    options: VideoOptions,
    limits: ResourceLimits,
) -> Result<VideoPreparationPlan<'a>> {
    let first = input
        .frames
        .first()
        .ok_or_else(|| geometry("video must contain at least one frame"))?;
    let count = input.frames.len() as u64;
    check_limit("raw_frames_per_video", count, limits.raw_frames_per_video())?;
    for (frame_index, frame) in input.frames.iter().enumerate() {
        validate_frame(*frame, limits)
            .map_err(|error| error.with_context("frame_index", frame_index))?;
        if (!input.image_frames || input.preprocessed)
            && (frame.height, frame.width) != (first.height, first.width)
        {
            return Err(geometry("video frames must have common dimensions")
                .with_context("frame_index", frame_index));
        }
    }
    let temporal = visual.temporal_patch_size;
    if temporal == 0 {
        return Err(invariant("video temporal patch size must be positive"));
    }
    let padded_frames = checked_mul(
        "padded video frame count",
        count.div_ceil(temporal),
        temporal,
    )?;
    let metadata = plan_metadata(input, options, padded_frames, temporal)?;
    let factor = checked_mul("video resize factor", visual.patch_size, visual.merge_size)?;
    let image_options = ImageOptions {
        min_pixels: options.min_pixels,
        max_pixels: options.max_pixels,
        resized_height: options.resized_height,
        resized_width: options.resized_width,
    };
    let first_resize = if input.image_frames && !input.preprocessed {
        let first_resize = image_frame_geometry(*first, visual, options, factor)?;
        for (frame_index, frame) in input.frames.iter().enumerate().skip(1) {
            let frame_plan = image_frame_geometry(*frame, visual, options, factor)?;
            if (frame_plan.height, frame_plan.width) != (first_resize.height, first_resize.width) {
                return Err(
                    geometry("image-list frames must resize to common dimensions")
                        .with_context("frame_index", frame_index),
                );
            }
        }
        Some(first_resize)
    } else {
        None
    };
    let (source_height, source_width) = first_resize
        .map_or((first.height as u64, first.width as u64), |plan| {
            (plan.height, plan.width)
        });
    let [height, width] = if input.preprocessed {
        if !source_height.is_multiple_of(factor) || !source_width.is_multiple_of(factor) {
            return Err(geometry(
                "preprocessed video dimensions must be multiples of the spatial resize factor",
            ));
        }
        [source_height, source_width]
    } else {
        let factor_area = checked_mul("video resize factor area", factor, factor)?;
        let minimum = options.min_pixels.unwrap_or(checked_mul(
            "video default minimum pixels",
            128,
            factor_area,
        )?);
        let default_maximum = checked_mul("video default maximum pixels", 768, factor_area)?;
        let budget_frames = if input.image_frames {
            padded_frames
        } else {
            count
        };
        let maximum = video_maximum(
            options,
            minimum,
            default_maximum,
            factor_area,
            budget_frames,
        )?;
        let explicit = crate::geometry::explicit_dimensions(image_options)?;
        if let Some([height, width]) = explicit {
            smart_resize(height, width, factor, None, None)?
        } else {
            smart_resize(
                source_height,
                source_width,
                factor,
                Some(minimum),
                Some(maximum),
            )?
        }
    };
    let final_resize = frame_geometry(visual, height, width, limits)?;
    let grid_t = padded_frames / temporal;
    let patch_rows = checked_mul("video patch rows", final_resize.patch_rows, grid_t)?;
    let output_elements = checked_mul("video patch elements", patch_rows, visual.patch_width)?;
    let pixel_values_capacity_bytes = checked_mul("video patch bytes", output_elements, 4)?;
    let rgb_capacity_bytes = checked_mul(
        "video prepared float RGB bytes",
        checked_mul("video frame RGB bytes", final_resize.rgb_capacity_bytes, 4)?,
        count,
    )?;
    limits.check_materialized_output_bytes(checked_add(
        "video model output bytes",
        pixel_values_capacity_bytes,
        24,
    )?)?;
    usize::try_from(output_elements)
        .map_err(|_| overflow("video patch output does not fit usize"))?;
    usize::try_from(rgb_capacity_bytes)
        .map_err(|_| overflow("video prepared RGB does not fit usize"))?;
    let video_grid_thw = [
        grid_t,
        height / visual.patch_size,
        width / visual.patch_size,
    ];
    for value in video_grid_thw {
        i64::try_from(value).map_err(|_| overflow("video grid does not fit int64"))?;
    }
    let geometry = VideoGeometryPlan {
        height,
        width,
        padded_frames,
        video_grid_thw,
        patch_rows,
        placeholder_count: checked_mul(
            "video placeholder tokens",
            final_resize.placeholder_count,
            grid_t,
        )?,
        rgb_capacity_bytes,
        pixel_values_capacity_bytes,
    };
    Ok(VideoPreparationPlan {
        input,
        geometry,
        metadata,
        first_resize,
        final_resize,
        cache_key: [0; 32],
    })
}

fn image_frame_geometry(
    frame: Rgb8<'_>,
    visual: &VisualProfile,
    options: VideoOptions,
    factor: u64,
) -> Result<ImageGeometryPlan> {
    let image_factor = checked_mul("frame-list image resize factor", factor, 2)?;
    let image_factor_area =
        checked_mul("frame-list image factor area", image_factor, image_factor)?;
    let explicit = crate::geometry::explicit_dimensions(ImageOptions {
        resized_height: options.resized_height,
        resized_width: options.resized_width,
        ..ImageOptions::default()
    })?;
    let [height, width] = if let Some([height, width]) = explicit {
        smart_resize(height, width, image_factor, None, None)?
    } else {
        smart_resize(
            frame.height as u64,
            frame.width as u64,
            image_factor,
            options.min_pixels.or(Some(checked_mul(
                "frame-list min pixels",
                4,
                image_factor_area,
            )?)),
            options.max_pixels.or(Some(checked_mul(
                "frame-list max pixels",
                16_384,
                image_factor_area,
            )?)),
        )?
    };
    build_image_plan(visual, height, width)
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn video_maximum(
    options: VideoOptions,
    minimum: u64,
    default_maximum: u64,
    factor_area: u64,
    frames: u64,
) -> Result<u64> {
    if minimum == 0 || options.max_pixels == Some(0) || options.total_pixels == Some(0) {
        return Err(geometry("video pixel budgets must be positive"));
    }
    let total = options
        .total_pixels
        .map_or(128_000.0 * factor_area as f64 * 0.9, |value| value as f64);
    let derived = (default_maximum as f64)
        .min(total / frames as f64 * 2.0)
        .max((minimum as f64 * 1.05).trunc());
    let maximum = options
        .max_pixels
        .map_or(derived, |value| (value as f64).min(derived));
    if maximum < minimum as f64 {
        return Err(geometry(
            "max_pixels must be greater than or equal to min_pixels",
        ));
    }
    if maximum >= u64::MAX as f64 {
        return Err(overflow("video pixel budget does not fit u64"));
    }
    Ok(maximum as u64)
}

fn frame_geometry(
    visual: &VisualProfile,
    height: u64,
    width: u64,
    _limits: ResourceLimits,
) -> Result<ImageGeometryPlan> {
    build_image_plan(visual, height, width)
}

#[allow(clippy::cast_precision_loss)]
fn plan_metadata(
    input: VideoInput<'_>,
    options: VideoOptions,
    padded_frames: u64,
    temporal: u64,
) -> Result<VideoMetadata> {
    let sample_fps = options.sample_fps.unwrap_or(2.0);
    let fps = options.raw_fps.or(input.fps).unwrap_or(sample_fps);
    for (name, value) in [("sample_fps", sample_fps), ("fps", fps)] {
        if !value.is_finite() || value <= 0.0 {
            return Err(geometry("video frame rate must be finite and positive")
                .with_context("option", name));
        }
    }
    let count = input.frames.len();
    let mut indices = if let Some(indices) = input.frames_indices {
        if indices.len() != count {
            return Err(geometry(
                "video frame indices must match decoded frame count",
            ));
        }
        indices.to_vec()
    } else {
        (0..count)
            .map(|index| i64::try_from(index).expect("bounded video frame count"))
            .collect()
    };
    if indices.iter().any(|index| *index < 0) || indices.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(geometry(
            "video frame indices must be nonnegative and ordered",
        ));
    }
    let supplied_times = input.timestamps;
    if let Some(times) = supplied_times {
        if times.len() != count {
            return Err(geometry("video timestamps must match decoded frame count"));
        }
        if times.iter().any(|time| !time.is_finite() || *time < 0.0)
            || times.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err(geometry(
                "video timestamps must be finite, nonnegative, and ordered",
            ));
        }
    }
    let mut times = supplied_times.map_or_else(
        || {
            indices
                .iter()
                .map(|index| *index as f64 / fps)
                .collect::<Vec<_>>()
        },
        <[f64]>::to_vec,
    );
    let synthetic_padding =
        input.image_frames && input.frames_indices.is_none() && supplied_times.is_none();
    let padded_count = usize::try_from(padded_frames)
        .map_err(|_| overflow("padded video count does not fit usize"))?;
    let temporal_count = usize::try_from(temporal)
        .map_err(|_| overflow("temporal patch size does not fit usize"))?;
    while indices.len() < padded_count {
        let last = *indices.last().expect("nonempty video");
        let next = if synthetic_padding {
            last.checked_add(1)
                .ok_or_else(|| overflow("padded frame index overflow"))?
        } else {
            last
        };
        indices.push(next);
        times.push(if synthetic_padding {
            next as f64 / fps
        } else {
            *times.last().expect("nonempty timestamps")
        });
    }
    let metadata_frames = if input.image_frames {
        padded_frames
    } else {
        count as u64
    };
    let total_num_frames = input
        .total_num_frames
        .unwrap_or(metadata_frames as f64 / sample_fps * fps);
    if !total_num_frames.is_finite() || total_num_frames <= 0.0 {
        return Err(geometry("total_num_frames must be finite and positive"));
    }
    let timestamps = times
        .chunks(temporal_count)
        .map(|chunk| chunk[0].midpoint(chunk[chunk.len() - 1]))
        .collect();
    Ok(VideoMetadata {
        timestamps,
        fps,
        sample_fps,
        frames_indices: indices,
        total_num_frames,
    })
}

fn validate_frame(frame: Rgb8<'_>, limits: ResourceLimits) -> Result<()> {
    let height = frame.height as u64;
    let width = frame.width as u64;
    let pixels = checked_mul("decoded video frame pixels", height, width)?;
    let row_bytes = checked_mul("video RGB row bytes", width, 3)?;
    let span = checked_add(
        "video RGB readable span",
        checked_mul(
            "video RGB last row offset",
            height.saturating_sub(1),
            frame.row_stride as u64,
        )?,
        row_bytes,
    )?;
    check_limit(
        "decoded_source_pixels",
        pixels,
        limits.decoded_pixels_per_image_or_frame(),
    )?;
    check_limit(
        "decoded_edge_length",
        height.max(width),
        limits.decoded_edge_length(),
    )?;
    if height == 0
        || width == 0
        || (frame.row_stride as u64) < row_bytes
        || (frame.data.len() as u64) < span
    {
        return Err(geometry(
            "video RGB frame does not cover its declared dimensions and row stride",
        ));
    }
    let ratio_bound = checked_mul("video aspect ratio boundary", height.min(width), 200)?;
    if height.max(width) > ratio_bound {
        return Err(geometry("video frame aspect ratio exceeds 200"));
    }
    Ok(())
}

pub(crate) fn execute_video_plan(
    plan: VideoPreparationPlan<'_>,
    parallel: bool,
) -> Result<PreparedRgbVideo> {
    let first_frame = plan.input.frames[0];
    let (height, width) = plan.first_resize.map_or(
        (first_frame.height as u64, first_frame.width as u64),
        |first| (first.height, first.width),
    );
    let resizer = VideoFrameResizer::new(height, width, &plan.final_resize)?;
    let resize = |frame: &Rgb8<'_>| -> Result<Vec<f32>> {
        if let Some(first) = plan.first_resize {
            let rgb = resize_image_rgb8(
                frame.data,
                frame.height as u64,
                frame.width as u64,
                frame.row_stride as u64,
                &first,
            )?;
            resizer.resize(&rgb, first.rgb_row_stride_bytes)
        } else {
            resizer.resize(frame.data, frame.row_stride as u64)
        }
    };
    let results = if parallel {
        plan.input.frames.par_iter().map(resize).collect::<Vec<_>>()
    } else {
        plan.input.frames.iter().map(resize).collect::<Vec<_>>()
    };
    let mut frames = Vec::with_capacity(results.len());
    for result in results {
        frames.push(result?);
    }
    let first = plan.input.frames[0];
    Ok(PreparedRgbVideo {
        frames,
        geometry: plan.geometry,
        metadata: plan.metadata,
        source_height: first.height as u64,
        source_width: first.width as u64,
        cache_key: plan.cache_key,
    })
}

#[allow(clippy::cast_possible_truncation)]
pub(crate) fn patchify_video_into(
    video: &PreparedRgbVideo,
    visual: &VisualProfile,
    pixels: &mut [f32],
    grid: &mut [i64],
    parallel: bool,
) {
    let geometry = video.geometry;
    let patch = visual.patch_size as usize;
    let temporal = visual.temporal_patch_size as usize;
    let merge = visual.merge_size as usize;
    let width = geometry.width as usize;
    let grid_h = geometry.video_grid_thw[1] as usize;
    let grid_w = geometry.video_grid_thw[2] as usize;
    let rows_per_group = grid_h * grid_w;
    let elements_per_group = rows_per_group * visual.patch_width as usize;
    let means = visual.image_mean.map(|value| value as f32 * 255.0);
    let stds = visual.image_std.map(|value| value as f32 * 255.0);
    let write = |(group, values): (usize, &mut [f32])| {
        let mut destination = 0;
        for outer_y in 0..grid_h / merge {
            for outer_x in 0..grid_w / merge {
                for merge_y in 0..merge {
                    for merge_x in 0..merge {
                        let source_y = (outer_y * merge + merge_y) * patch;
                        let source_x = (outer_x * merge + merge_x) * patch;
                        for channel in 0..3 {
                            for time in 0..temporal {
                                let frame = &video.frames
                                    [(group * temporal + time).min(video.frames.len() - 1)];
                                for patch_y in 0..patch {
                                    for patch_x in 0..patch {
                                        let source =
                                            ((source_y + patch_y) * width + source_x + patch_x) * 3
                                                + channel;
                                        values[destination] =
                                            (frame[source] - means[channel]) / stds[channel];
                                        destination += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        debug_assert_eq!(destination, elements_per_group);
    };
    if parallel {
        pixels
            .par_chunks_mut(elements_per_group)
            .enumerate()
            .for_each(write);
    } else {
        pixels
            .chunks_mut(elements_per_group)
            .enumerate()
            .for_each(write);
    }
    for (target, value) in grid.iter_mut().zip(geometry.video_grid_thw) {
        *target = i64::try_from(value).expect("validated video grid");
    }
}

pub(crate) fn video_cache_key(
    input: VideoInput<'_>,
    options: VideoOptions,
    profile_fingerprint: &str,
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"qwen-mm-compat-v1\0video\0");
    digest.update(profile_fingerprint.as_bytes());
    digest.update([u8::from(input.image_frames), u8::from(input.preprocessed)]);
    digest.update((input.frames.len() as u64).to_le_bytes());
    for frame in input.frames {
        digest.update((frame.height as u64).to_le_bytes());
        digest.update((frame.width as u64).to_le_bytes());
        for row in 0..frame.height {
            digest.update(
                &frame.data[row * frame.row_stride..row * frame.row_stride + frame.width * 3],
            );
        }
    }
    for value in [
        options.min_pixels,
        options.max_pixels,
        options.total_pixels,
        options.resized_height,
        options.resized_width,
    ] {
        digest.update([u8::from(value.is_some())]);
        if let Some(value) = value {
            digest.update(value.to_le_bytes());
        }
    }
    for value in [
        options.sample_fps,
        options.raw_fps,
        input.fps,
        input.total_num_frames,
    ] {
        digest.update([u8::from(value.is_some())]);
        if let Some(value) = value {
            digest.update(value.to_bits().to_le_bytes());
        }
    }
    for values in [
        input.timestamps.map(|values| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        }),
        input.frames_indices.map(|values| {
            values
                .iter()
                .map(|value| u64::from_le_bytes(value.to_le_bytes()))
                .collect()
        }),
    ] {
        digest.update([u8::from(values.is_some())]);
        if let Some(values) = values {
            digest.update((values.len() as u64).to_le_bytes());
            for value in values {
                digest.update(value.to_le_bytes());
            }
        }
    }
    digest.finalize().into()
}

fn check_limit(name: &'static str, actual: u64, limit: u64) -> Result<()> {
    if actual > limit {
        return Err(QwenError::new(
            ErrorCategory::ResourceLimit,
            "video resource limit exceeded",
        )
        .with_context("limit", name)
        .with_context("actual", actual)
        .with_context("maximum", limit));
    }
    Ok(())
}
fn geometry(message: &'static str) -> QwenError {
    QwenError::new(ErrorCategory::MediaGeometry, message)
}
fn overflow(message: &'static str) -> QwenError {
    QwenError::new(ErrorCategory::ArithmeticOverflow, message)
}
fn invariant(message: &'static str) -> QwenError {
    QwenError::new(ErrorCategory::InternalInvariant, message)
}

#[cfg(test)]
mod tests {
    use super::{execute_video_plan, patchify_video_into, plan_video_rgb8, video_cache_key};
    use crate::{
        ErrorCategory, LimitOverrides, ProfileAlias, ProfileRegistry, ResourceLimits, Rgb8,
        VideoInput, VideoOptions,
    };

    fn visual() -> crate::VisualProfile {
        ProfileRegistry::bundled()
            .expect("profiles")
            .get(ProfileAlias::Qwen3Vl8b)
            .visual
            .clone()
    }

    fn fixed_size(height: u64, width: u64) -> VideoOptions {
        VideoOptions {
            resized_height: Some(height),
            resized_width: Some(width),
            ..VideoOptions::default()
        }
    }

    #[test]
    fn asymmetric_frames_locate_every_spatial_channel_and_temporal_value() {
        let data = (0..3)
            .map(|time| {
                (0..64 * 96 * 3)
                    .map(|index| {
                        let channel = index % 3;
                        let x = index / 3 % 96;
                        let y = index / 3 / 96;
                        u8::try_from((time * 71 + channel * 43 + y * 3 + x * 5) % 256)
                            .expect("byte")
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let frames = data
            .iter()
            .map(|data| Rgb8 {
                data,
                height: 64,
                width: 96,
                row_stride: 96 * 3,
            })
            .collect::<Vec<_>>();
        let times = [0.0, 1.0, 4.0];
        let indices = [0, 2, 8];
        let input = VideoInput {
            frames: &frames,
            timestamps: Some(&times),
            frames_indices: Some(&indices),
            fps: Some(2.0),
            total_num_frames: Some(9.0),
            ..VideoInput::default()
        };
        let visual = visual();
        let plan = plan_video_rgb8(
            input,
            &visual,
            fixed_size(64, 96),
            ResourceLimits::default(),
        )
        .expect("plan");
        assert_eq!(plan.geometry.video_grid_thw, [2, 4, 6]);
        assert_eq!(plan.geometry.placeholder_count, 12);
        assert_eq!(plan.metadata.timestamps, [0.5, 4.0]);
        assert_eq!(plan.metadata.frames_indices, [0, 2, 8, 8]);
        let video = execute_video_plan(plan, false).expect("resize");
        let mut pixels = vec![0.0; 48 * 1536];
        let mut grid = [0; 3];
        patchify_video_into(&video, &visual, &mut pixels, &mut grid, false);
        assert_eq!(grid, [2, 4, 6]);
        for (row, values) in pixels.chunks_exact(1536).enumerate() {
            let group = row / 24;
            let spatial = row % 24;
            let block = spatial / 4;
            let inner = spatial % 4;
            let patch_y = block / 3 * 2 + inner / 2;
            let patch_x = block % 3 * 2 + inner % 2;
            for (element, actual) in values.iter().enumerate() {
                let channel = element / 512;
                let time = element / 256 % 2;
                let y = patch_y * 16 + element / 16 % 16;
                let x = patch_x * 16 + element % 16;
                let frame = (group * 2 + time).min(2);
                let expected = (f32::from(data[frame][(y * 96 + x) * 3 + channel]) - 127.5) / 127.5;
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "row={row} element={element}"
                );
            }
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn decoded_padding_repeats_metadata_while_image_lists_extend_synthetic_indices() {
        let bytes = vec![85; 64 * 64 * 3];
        let frames = [Rgb8 {
            data: &bytes,
            height: 64,
            width: 64,
            row_stride: 192,
        }; 3];
        let visual = visual();
        let input = VideoInput {
            frames: &frames,
            ..VideoInput::default()
        };
        let decoded = plan_video_rgb8(
            input,
            &visual,
            fixed_size(32, 32),
            ResourceLimits::default(),
        )
        .expect("decoded");
        assert_eq!(decoded.metadata.timestamps, [0.25, 1.0]);
        assert_eq!(decoded.metadata.frames_indices, [0, 1, 2, 2]);
        assert_eq!(decoded.metadata.total_num_frames, 3.0);
        let list = plan_video_rgb8(
            VideoInput {
                image_frames: true,
                ..input
            },
            &visual,
            fixed_size(32, 32),
            ResourceLimits::default(),
        )
        .expect("list");
        assert_eq!(list.first_resize.expect("Pillow stage").height, 128);
        assert_eq!(list.geometry.height, 64);
        assert_eq!(list.metadata.timestamps, [0.25, 1.25]);
        assert_eq!(list.metadata.frames_indices, [0, 1, 2, 3]);
        assert_eq!(list.metadata.total_num_frames, 4.0);
        let supplied_times = [0.0, 0.7, 2.0];
        let explicit = plan_video_rgb8(
            VideoInput {
                image_frames: true,
                timestamps: Some(&supplied_times),
                ..input
            },
            &visual,
            fixed_size(32, 32),
            ResourceLimits::default(),
        )
        .expect("explicit times");
        assert_eq!(explicit.metadata.timestamps, [0.35, 2.0]);
    }

    #[test]
    fn empty_malformed_metadata_and_frame_layouts_fail_before_pixel_allocation() {
        let bytes = vec![0; 32 * 32 * 3];
        let frame = Rgb8 {
            data: &bytes,
            height: 32,
            width: 32,
            row_stride: 96,
        };
        let frames = [frame, frame];
        let visual = visual();
        let input = VideoInput {
            frames: &frames,
            ..VideoInput::default()
        };
        for invalid in [
            VideoInput::default(),
            VideoInput {
                timestamps: Some(&[0.0]),
                ..input
            },
            VideoInput {
                timestamps: Some(&[0.0, f64::NAN]),
                ..input
            },
            VideoInput {
                frames_indices: Some(&[1, 0]),
                ..input
            },
            VideoInput {
                fps: Some(0.0),
                ..input
            },
            VideoInput {
                total_num_frames: Some(f64::INFINITY),
                ..input
            },
        ] {
            assert_eq!(
                plan_video_rgb8(
                    invalid,
                    &visual,
                    fixed_size(32, 32),
                    ResourceLimits::default()
                )
                .expect_err("invalid metadata")
                .category(),
                ErrorCategory::MediaGeometry
            );
        }
        let malformed = [Rgb8 {
            data: &bytes[..3],
            ..frame
        }];
        assert_eq!(
            plan_video_rgb8(
                VideoInput {
                    frames: &malformed,
                    ..input
                },
                &visual,
                fixed_size(32, 32),
                ResourceLimits::default()
            )
            .expect_err("short frame")
            .category(),
            ErrorCategory::MediaGeometry
        );
        let narrow = [Rgb8 {
            row_stride: 1,
            ..frame
        }];
        assert_eq!(
            plan_video_rgb8(
                VideoInput {
                    frames: &narrow,
                    ..input
                },
                &visual,
                fixed_size(32, 32),
                ResourceLimits::default()
            )
            .expect_err("narrow row")
            .category(),
            ErrorCategory::MediaGeometry
        );
    }

    #[test]
    fn frame_and_output_resource_limits_precede_resize_and_ignore_still_image_ceiling() {
        let bytes = vec![0; 32 * 32 * 3];
        let frames = [Rgb8 {
            data: &bytes,
            height: 32,
            width: 32,
            row_stride: 96,
        }; 3];
        let input = VideoInput {
            frames: &frames,
            ..VideoInput::default()
        };
        let visual = visual();
        let limits = ResourceLimits::default()
            .lowered(LimitOverrides {
                raw_frames_per_video: Some(2),
                ..LimitOverrides::default()
            })
            .expect("limits");
        assert_eq!(
            plan_video_rgb8(input, &visual, fixed_size(32, 32), limits)
                .expect_err("frame limit")
                .category(),
            ErrorCategory::ResourceLimit
        );
        let limits = ResourceLimits::default()
            .lowered(LimitOverrides {
                materialized_output_bytes_per_batch: Some(24),
                ..LimitOverrides::default()
            })
            .expect("limits");
        assert_eq!(
            plan_video_rgb8(input, &visual, fixed_size(32, 32), limits)
                .expect_err("output limit")
                .category(),
            ErrorCategory::ResourceLimit
        );
        let limits = ResourceLimits::default()
            .lowered(LimitOverrides {
                prepared_image_pixels_per_occurrence: Some(1),
                ..LimitOverrides::default()
            })
            .expect("limits");
        assert!(plan_video_rgb8(input, &visual, fixed_size(32, 32), limits).is_ok());
    }

    #[test]
    fn cache_identity_ignores_row_padding_and_covers_timing_options_and_list_mode() {
        let bytes = vec![93; 32 * 32 * 3];
        let mut padded = vec![18; 32 * 100];
        for (source, destination) in bytes.chunks_exact(96).zip(padded.chunks_exact_mut(100)) {
            destination[..96].copy_from_slice(source);
        }
        let frames = [Rgb8 {
            data: &bytes,
            height: 32,
            width: 32,
            row_stride: 96,
        }];
        let padded_frames = [Rgb8 {
            data: &padded,
            height: 32,
            width: 32,
            row_stride: 100,
        }];
        let input = VideoInput {
            frames: &frames,
            ..VideoInput::default()
        };
        let options = fixed_size(32, 32);
        let expected = video_cache_key(input, options, "profile");
        assert_eq!(
            expected,
            video_cache_key(
                VideoInput {
                    frames: &padded_frames,
                    ..input
                },
                options,
                "profile"
            )
        );
        assert_ne!(
            expected,
            video_cache_key(
                VideoInput {
                    timestamps: Some(&[1.0]),
                    ..input
                },
                options,
                "profile"
            )
        );
        assert_ne!(
            expected,
            video_cache_key(
                VideoInput {
                    image_frames: true,
                    ..input
                },
                options,
                "profile"
            )
        );
        assert_ne!(expected, video_cache_key(input, options, "other profile"));
    }

    #[test]
    fn differently_sized_image_frames_work_when_the_first_stage_resolves_a_common_size() {
        let bytes = vec![73; 64 * 64 * 3];
        let frames = [
            Rgb8 {
                data: &bytes,
                height: 64,
                width: 64,
                row_stride: 192,
            },
            Rgb8 {
                data: &bytes[..48 * 32 * 3],
                height: 48,
                width: 32,
                row_stride: 96,
            },
        ];
        let input = VideoInput {
            frames: &frames,
            image_frames: true,
            ..VideoInput::default()
        };
        let visual = visual();
        let plan = plan_video_rgb8(
            input,
            &visual,
            fixed_size(64, 64),
            ResourceLimits::default(),
        )
        .expect("image resize aligns dimensions");
        let output = execute_video_plan(plan, false).expect("resize");
        assert_eq!(output.frames[0].len(), output.frames[1].len());
        assert_eq!(
            plan_video_rgb8(
                VideoInput {
                    image_frames: false,
                    ..input
                },
                &visual,
                fixed_size(64, 64),
                ResourceLimits::default()
            )
            .expect_err("decoded mismatch")
            .category(),
            ErrorCategory::MediaGeometry
        );
    }

    #[test]
    fn preprocessed_video_keeps_minimum_factor_dimensions_and_custom_pixel_budgets() {
        let bytes = vec![67; 32 * 32 * 3];
        let frames = [Rgb8 {
            data: &bytes,
            height: 32,
            width: 32,
            row_stride: 96,
        }; 1];
        let visual = visual();
        let input = VideoInput {
            frames: &frames,
            preprocessed: true,
            ..VideoInput::default()
        };
        let plan = plan_video_rgb8(
            input,
            &visual,
            VideoOptions::default(),
            ResourceLimits::default(),
        )
        .expect("prepared");
        assert_eq!(plan.geometry.video_grid_thw, [1, 2, 2]);
        assert_eq!(plan.geometry.pixel_values_capacity_bytes, 4 * 1536 * 4);
        let video = execute_video_plan(plan, false).expect("no-op");
        assert!(
            video.frames[0]
                .iter()
                .all(|value| value.to_bits() == 67.0_f32.to_bits())
        );
        let malformed = [Rgb8 {
            width: 31,
            ..frames[0]
        }];
        assert_eq!(
            plan_video_rgb8(
                VideoInput {
                    frames: &malformed,
                    ..input
                },
                &visual,
                VideoOptions::default(),
                ResourceLimits::default()
            )
            .expect_err("alignment")
            .category(),
            ErrorCategory::MediaGeometry
        );
    }

    #[test]
    fn processor_owned_parallelism_preserves_every_float_bit() {
        let data = vec![102; 65 * 81 * 3];
        let frames = [Rgb8 {
            data: &data,
            height: 65,
            width: 81,
            row_stride: 243,
        }; 5];
        let visual = visual();
        let plan = plan_video_rgb8(
            VideoInput {
                frames: &frames,
                ..VideoInput::default()
            },
            &visual,
            fixed_size(64, 96),
            ResourceLimits::default(),
        )
        .expect("plan");
        let serial = execute_video_plan(plan.clone(), false).expect("serial");
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .expect("pool");
        let parallel = pool
            .install(|| execute_video_plan(plan, true))
            .expect("parallel");
        assert_eq!(serial.frames, parallel.frames);
        let mut serial_pixels = vec![0.0; 72 * 1536];
        let mut parallel_pixels = serial_pixels.clone();
        patchify_video_into(&serial, &visual, &mut serial_pixels, &mut [0; 3], false);
        pool.install(|| {
            patchify_video_into(&parallel, &visual, &mut parallel_pixels, &mut [0; 3], true);
        });
        assert_eq!(serial_pixels, parallel_pixels);
    }
}
