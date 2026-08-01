//! C-contiguous parity outputs and ordered integration sidecars.

use std::mem;

use crate::{
    error::{ErrorCategory, QwenError, Result},
    limits::{checked_capacity_bytes, checked_mul},
};

/// One owned, C-contiguous two-dimensional array.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<T> {
    rows: usize,
    columns: usize,
    data: Vec<T>,
}

impl<T> Matrix<T> {
    /// Builds a matrix after checking its exact element capacity.
    ///
    /// # Errors
    ///
    /// Returns `arithmetic_overflow` if the shape product overflows, or
    /// `internal_invariant` if `data` has the wrong length.
    pub fn new(rows: usize, columns: usize, data: Vec<T>) -> Result<Self> {
        let expected = rows.checked_mul(columns).ok_or_else(|| {
            QwenError::new(
                ErrorCategory::ArithmeticOverflow,
                "matrix element capacity overflowed",
            )
            .with_context("rows", rows)
            .with_context("columns", columns)
        })?;
        if data.len() != expected {
            return Err(QwenError::new(
                ErrorCategory::InternalInvariant,
                "matrix data length does not match its shape",
            )
            .with_context("rows", rows)
            .with_context("columns", columns)
            .with_context("expected_elements", expected)
            .with_context("actual_elements", data.len()));
        }
        Ok(Self {
            rows,
            columns,
            data,
        })
    }

    /// Returns `[rows, columns]`.
    #[must_use]
    pub const fn shape(&self) -> [usize; 2] {
        [self.rows, self.columns]
    }

    /// Returns the row count.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the column count.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Returns the C-contiguous data in row-major order.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Consumes the matrix and returns its row-major allocation.
    #[must_use]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Returns byte strides `[row_stride, element_stride]`.
    ///
    /// # Errors
    ///
    /// Returns `arithmetic_overflow` if shape or stride arithmetic cannot be
    /// represented by the parity-facing `u64` capacity model.
    pub fn byte_strides(&self) -> Result<[u64; 2]> {
        let element_bytes = u64::try_from(mem::size_of::<T>()).map_err(|_| {
            QwenError::new(
                ErrorCategory::ArithmeticOverflow,
                "element size does not fit parity stride arithmetic",
            )
        })?;
        let columns = u64::try_from(self.columns).map_err(|_| {
            QwenError::new(
                ErrorCategory::ArithmeticOverflow,
                "column count does not fit parity stride arithmetic",
            )
        })?;
        Ok([
            checked_mul("matrix row byte stride", columns, element_bytes)?,
            element_bytes,
        ])
    }

    /// Returns the exact owned byte capacity represented by the shape.
    ///
    /// # Errors
    ///
    /// Returns `arithmetic_overflow` if shape or byte-capacity arithmetic
    /// cannot be represented by the parity-facing `u64` capacity model.
    pub fn byte_capacity(&self) -> Result<u64> {
        checked_capacity_bytes(
            u64::try_from(self.rows).map_err(|_| {
                QwenError::new(
                    ErrorCategory::ArithmeticOverflow,
                    "row count does not fit parity capacity arithmetic",
                )
            })?,
            u64::try_from(self.columns).map_err(|_| {
                QwenError::new(
                    ErrorCategory::ArithmeticOverflow,
                    "column count does not fit parity capacity arithmetic",
                )
            })?,
            u64::try_from(mem::size_of::<T>()).map_err(|_| {
                QwenError::new(
                    ErrorCategory::ArithmeticOverflow,
                    "element size does not fit parity capacity arithmetic",
                )
            })?,
        )
    }
}

/// Half-open coordinates in one parity-facing `i64` coordinate space.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoordinateRange {
    /// Inclusive start coordinate.
    pub start: i64,
    /// Exclusive end coordinate.
    pub end: i64,
}

impl CoordinateRange {
    fn is_valid(self) -> bool {
        self.start >= 0 && self.end > self.start
    }
}

/// Exact placeholder replacement range in both prompt coordinate systems.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplacementRange {
    /// Unicode code-point coordinates in the rendered prompt.
    pub code_points: CoordinateRange,
    /// Token coordinates after placeholder expansion.
    pub tokens: CoordinateRange,
}

/// Per-image ordered adapter metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageSidecar {
    /// Request row containing this occurrence.
    pub request_index: i64,
    /// Corresponding row in `image_grid_thw`.
    pub grid_row: i64,
    /// Exact prompt replacement coordinates.
    pub replacement: ReplacementRange,
}

/// Per-video ordered adapter metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoSidecar {
    /// Request row containing this occurrence.
    pub request_index: i64,
    /// Corresponding row in `video_grid_thw`.
    pub grid_row: i64,
    /// Source or adapter FPS.
    pub fps: f64,
    /// Exact sampled source-frame indices.
    pub frames_indices: Vec<i64>,
    /// Total number of source frames, matching upstream `float64` metadata.
    pub total_num_frames: f64,
    /// Effective sampling FPS.
    pub sample_fps: f64,
    /// Exact prompt replacement coordinates.
    pub replacement: ReplacementRange,
}

/// Ordered non-model-input metadata retained for downstream adapters.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IntegrationSidecar {
    /// Image occurrences in traversal/grid-row order.
    pub images: Vec<ImageSidecar>,
    /// Video occurrences in traversal/grid-row order.
    pub videos: Vec<VideoSidecar>,
}

/// Exact official model-input arrays before cross-field validation.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedArrays {
    /// `int64 [batch, sequence]`.
    pub input_ids: Matrix<i64>,
    /// `int64 [batch, sequence]`.
    pub attention_mask: Matrix<i64>,
    /// `int64 [batch, sequence]`.
    pub mm_token_type_ids: Matrix<i64>,
    /// `float32 [image_patches, 1536]`, only when images are present.
    pub pixel_values: Option<Matrix<f32>>,
    /// `int64 [image_occurrences, 3]`, only when images are present.
    pub image_grid_thw: Option<Matrix<i64>>,
    /// `float32 [video_patches, 1536]`, only when videos are present.
    pub pixel_values_videos: Option<Matrix<f32>>,
    /// `int64 [video_occurrences, 3]`, only when videos are present.
    pub video_grid_thw: Option<Matrix<i64>>,
}

/// A complete owned parity result with the official conditional key set.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedBatch {
    arrays: PreparedArrays,
    sidecar: IntegrationSidecar,
}

impl PreparedBatch {
    /// Validates cross-field shapes, conditional presence, C strides, and
    /// sidecar ordering before constructing a prepared batch.
    ///
    /// # Errors
    ///
    /// Returns `internal_invariant` for inconsistent shapes, presence, or
    /// sidecars, and `arithmetic_overflow` for unrepresentable coordinates.
    pub fn new(arrays: PreparedArrays, sidecar: IntegrationSidecar) -> Result<Self> {
        let text_shape = arrays.input_ids.shape();
        if text_shape[0] == 0 || text_shape[1] == 0 {
            return Err(invariant(
                "text arrays must have non-zero batch and sequence dimensions",
            ));
        }
        if arrays.attention_mask.shape() != text_shape
            || arrays.mm_token_type_ids.shape() != text_shape
        {
            return Err(invariant("all text arrays must have the same shape"));
        }
        validate_modality(
            "image",
            arrays.pixel_values.as_ref(),
            arrays.image_grid_thw.as_ref(),
            sidecar.images.len(),
        )?;
        validate_modality(
            "video",
            arrays.pixel_values_videos.as_ref(),
            arrays.video_grid_thw.as_ref(),
            sidecar.videos.len(),
        )?;
        validate_image_sidecar(&sidecar.images, text_shape[0])?;
        validate_video_sidecar(&sidecar.videos, text_shape[0])?;
        Ok(Self { arrays, sidecar })
    }

    /// Returns all official arrays.
    #[must_use]
    pub const fn arrays(&self) -> &PreparedArrays {
        &self.arrays
    }

    /// Returns the ordered adapter-only metadata.
    #[must_use]
    pub const fn sidecar(&self) -> &IntegrationSidecar {
        &self.sidecar
    }

    /// Returns the official keys present, in frozen processor order.
    #[must_use]
    pub fn official_keys(&self) -> Vec<&'static str> {
        let mut keys = vec!["input_ids", "attention_mask", "mm_token_type_ids"];
        if self.arrays.pixel_values.is_some() {
            keys.extend(["pixel_values", "image_grid_thw"]);
        }
        if self.arrays.pixel_values_videos.is_some() {
            keys.extend(["pixel_values_videos", "video_grid_thw"]);
        }
        keys
    }
}

fn validate_modality(
    name: &'static str,
    pixels: Option<&Matrix<f32>>,
    grid: Option<&Matrix<i64>>,
    sidecar_len: usize,
) -> Result<()> {
    match (pixels, grid) {
        (None, None) if sidecar_len == 0 => Ok(()),
        (Some(pixels), Some(grid)) => {
            if pixels.rows() == 0 || grid.rows() == 0 {
                return Err(
                    invariant("present modalities cannot use zero-length placeholders")
                        .with_context("modality", name),
                );
            }
            if pixels.columns() != 1536 {
                return Err(invariant("pixel array has the wrong patch width")
                    .with_context("modality", name)
                    .with_context("actual", pixels.columns())
                    .with_context("expected", 1536_usize));
            }
            if grid.columns() != 3 {
                return Err(invariant("grid array must have three columns")
                    .with_context("modality", name)
                    .with_context("actual", grid.columns()));
            }
            if grid.rows() != sidecar_len {
                return Err(invariant("sidecar and grid occurrence counts differ")
                    .with_context("modality", name)
                    .with_context("grid_rows", grid.rows())
                    .with_context("sidecar_entries", sidecar_len));
            }
            let mut expected_pixel_rows = 0_u64;
            for (row_index, row) in grid.as_slice().chunks_exact(3).enumerate() {
                if row.iter().any(|&value| value <= 0) {
                    return Err(invariant("grid dimensions must be positive")
                        .with_context("modality", name)
                        .with_context("grid_row", row_index));
                }
                let temporal = u64::try_from(row[0])
                    .map_err(|_| invariant("grid temporal dimension is invalid"))?;
                let height = u64::try_from(row[1])
                    .map_err(|_| invariant("grid height dimension is invalid"))?;
                let width = u64::try_from(row[2])
                    .map_err(|_| invariant("grid width dimension is invalid"))?;
                let rows = checked_mul("grid temporal-height product", temporal, height)?;
                let rows = checked_mul("grid patch-row product", rows, width)?;
                expected_pixel_rows =
                    crate::limits::checked_add("grid patch-row sum", expected_pixel_rows, rows)?;
            }
            let actual_pixel_rows = u64::try_from(pixels.rows()).map_err(|_| {
                QwenError::new(
                    ErrorCategory::ArithmeticOverflow,
                    "pixel row count does not fit grid arithmetic",
                )
            })?;
            if actual_pixel_rows != expected_pixel_rows {
                return Err(invariant("pixel rows do not equal summed grid products")
                    .with_context("modality", name)
                    .with_context("expected_rows", expected_pixel_rows)
                    .with_context("actual_rows", actual_pixel_rows));
            }
            Ok(())
        }
        _ => Err(
            invariant("conditional modality arrays and sidecar must appear together")
                .with_context("modality", name),
        ),
    }
}

fn validate_image_sidecar(sidecars: &[ImageSidecar], batch_rows: usize) -> Result<()> {
    for (index, sidecar) in sidecars.iter().enumerate() {
        if index != 0 && sidecar.request_index < sidecars[index - 1].request_index {
            return Err(
                invariant("image sidecar request order must follow batch traversal")
                    .with_context("sidecar_index", index)
                    .with_context("request_index", sidecar.request_index),
            );
        }
        validate_common_sidecar(
            sidecar.request_index,
            sidecar.grid_row,
            sidecar.replacement,
            index,
            batch_rows,
            "image",
        )?;
    }
    Ok(())
}

fn validate_video_sidecar(sidecars: &[VideoSidecar], batch_rows: usize) -> Result<()> {
    for (index, sidecar) in sidecars.iter().enumerate() {
        if index != 0 && sidecar.request_index < sidecars[index - 1].request_index {
            return Err(
                invariant("video sidecar request order must follow batch traversal")
                    .with_context("sidecar_index", index)
                    .with_context("request_index", sidecar.request_index),
            );
        }
        validate_common_sidecar(
            sidecar.request_index,
            sidecar.grid_row,
            sidecar.replacement,
            index,
            batch_rows,
            "video",
        )?;
        if !sidecar.fps.is_finite()
            || sidecar.fps <= 0.0
            || !sidecar.sample_fps.is_finite()
            || sidecar.sample_fps <= 0.0
            || !sidecar.total_num_frames.is_finite()
            || sidecar.total_num_frames <= 0.0
            || sidecar.frames_indices.is_empty()
            || sidecar.frames_indices.iter().any(|&value| value < 0)
        {
            return Err(invariant("video sidecar contains invalid numeric metadata")
                .with_context("sidecar_index", index));
        }
    }
    Ok(())
}

fn validate_common_sidecar(
    request_index: i64,
    grid_row: i64,
    replacement: ReplacementRange,
    index: usize,
    batch_rows: usize,
    modality: &'static str,
) -> Result<()> {
    let expected_grid_row = i64::try_from(index).map_err(|_| {
        QwenError::new(
            ErrorCategory::ArithmeticOverflow,
            "sidecar index does not fit parity metadata",
        )
    })?;
    let batch_rows = i64::try_from(batch_rows).map_err(|_| {
        QwenError::new(
            ErrorCategory::ArithmeticOverflow,
            "batch rows do not fit parity metadata",
        )
    })?;
    if request_index < 0 || request_index >= batch_rows {
        return Err(invariant("sidecar request index is out of bounds")
            .with_context("modality", modality)
            .with_context("sidecar_index", index)
            .with_context("request_index", request_index));
    }
    if grid_row != expected_grid_row {
        return Err(invariant("sidecar order must match grid row order")
            .with_context("modality", modality)
            .with_context("sidecar_index", index)
            .with_context("grid_row", grid_row));
    }
    if !replacement.code_points.is_valid() || !replacement.tokens.is_valid() {
        return Err(
            invariant("replacement ranges must be ordered and non-negative")
                .with_context("modality", modality)
                .with_context("sidecar_index", index),
        );
    }
    Ok(())
}

fn invariant(message: impl Into<String>) -> QwenError {
    QwenError::new(ErrorCategory::InternalInvariant, message)
}

#[cfg(test)]
mod tests {
    use super::{
        CoordinateRange, ImageSidecar, IntegrationSidecar, Matrix, PreparedArrays, PreparedBatch,
        ReplacementRange, VideoSidecar,
    };
    use crate::error::ErrorCategory;

    fn range() -> ReplacementRange {
        ReplacementRange {
            code_points: CoordinateRange { start: 2, end: 3 },
            tokens: CoordinateRange { start: 4, end: 8 },
        }
    }

    fn text_arrays() -> PreparedArrays {
        PreparedArrays {
            input_ids: Matrix::new(1, 2, vec![1, 2]).expect("matrix"),
            attention_mask: Matrix::new(1, 2, vec![1, 1]).expect("matrix"),
            mm_token_type_ids: Matrix::new(1, 2, vec![0, 0]).expect("matrix"),
            pixel_values: None,
            image_grid_thw: None,
            pixel_values_videos: None,
            video_grid_thw: None,
        }
    }

    #[test]
    fn text_only_has_exact_key_set_and_i64_strides() {
        let batch = PreparedBatch::new(text_arrays(), IntegrationSidecar::default())
            .expect("text-only output");
        assert_eq!(
            batch.official_keys(),
            ["input_ids", "attention_mask", "mm_token_type_ids"]
        );
        assert_eq!(
            batch.arrays().input_ids.byte_strides().expect("strides"),
            [16, 8]
        );
    }

    #[test]
    fn image_outputs_are_f32_i64_contiguous_and_ordered() {
        let mut arrays = text_arrays();
        arrays.pixel_values = Some(Matrix::new(4, 1536, vec![0.0_f32; 4 * 1536]).expect("pixels"));
        arrays.image_grid_thw = Some(Matrix::new(1, 3, vec![1_i64, 2, 2]).expect("grid"));
        let sidecar = IntegrationSidecar {
            images: vec![ImageSidecar {
                request_index: 0,
                grid_row: 0,
                replacement: range(),
            }],
            videos: vec![],
        };
        let batch = PreparedBatch::new(arrays, sidecar).expect("image output");
        assert_eq!(
            batch
                .arrays()
                .pixel_values
                .as_ref()
                .expect("pixels")
                .byte_strides()
                .expect("strides"),
            [6144, 4]
        );
        assert_eq!(
            batch
                .arrays()
                .image_grid_thw
                .as_ref()
                .expect("grid")
                .byte_strides()
                .expect("strides"),
            [24, 8]
        );
        assert_eq!(
            batch.official_keys(),
            [
                "input_ids",
                "attention_mask",
                "mm_token_type_ids",
                "pixel_values",
                "image_grid_thw"
            ]
        );
    }

    #[test]
    fn video_outputs_retain_every_adapter_sidecar_field() {
        let mut arrays = text_arrays();
        arrays.pixel_values_videos =
            Some(Matrix::new(2, 1536, vec![0.0_f32; 2 * 1536]).expect("video pixels"));
        arrays.video_grid_thw = Some(Matrix::new(1, 3, vec![2_i64, 1, 1]).expect("video grid"));
        let sidecar = IntegrationSidecar {
            images: vec![],
            videos: vec![VideoSidecar {
                request_index: 0,
                grid_row: 0,
                fps: 30.0,
                frames_indices: vec![0, 1],
                total_num_frames: 2.0,
                sample_fps: 30.0,
                replacement: range(),
            }],
        };
        let batch = PreparedBatch::new(arrays, sidecar).expect("video output");
        assert_eq!(
            batch.official_keys(),
            [
                "input_ids",
                "attention_mask",
                "mm_token_type_ids",
                "pixel_values_videos",
                "video_grid_thw"
            ]
        );
        assert_eq!(batch.sidecar().videos[0].frames_indices, [0, 1]);
        assert_eq!(
            batch.sidecar().videos[0].total_num_frames.to_bits(),
            2.0_f64.to_bits()
        );
    }

    #[test]
    fn absent_modalities_never_accept_placeholder_arrays() {
        let mut arrays = text_arrays();
        arrays.pixel_values = Some(Matrix::new(0, 1536, vec![]).expect("empty shape"));
        arrays.image_grid_thw = Some(Matrix::new(0, 3, vec![]).expect("empty shape"));
        let error = PreparedBatch::new(arrays, IntegrationSidecar::default())
            .expect_err("zero placeholders are forbidden");
        assert_eq!(error.category(), ErrorCategory::InternalInvariant);
    }

    #[test]
    fn sidecar_request_indices_cannot_regress() {
        let mut arrays = PreparedArrays {
            input_ids: Matrix::new(2, 1, vec![1, 2]).expect("matrix"),
            attention_mask: Matrix::new(2, 1, vec![1, 1]).expect("matrix"),
            mm_token_type_ids: Matrix::new(2, 1, vec![0, 0]).expect("matrix"),
            pixel_values: None,
            image_grid_thw: None,
            pixel_values_videos: None,
            video_grid_thw: None,
        };
        arrays.pixel_values = Some(Matrix::new(2, 1536, vec![0.0_f32; 2 * 1536]).expect("pixels"));
        arrays.image_grid_thw = Some(Matrix::new(2, 3, vec![1_i64; 6]).expect("grid"));
        let sidecar = IntegrationSidecar {
            images: vec![
                ImageSidecar {
                    request_index: 1,
                    grid_row: 0,
                    replacement: range(),
                },
                ImageSidecar {
                    request_index: 0,
                    grid_row: 1,
                    replacement: range(),
                },
            ],
            videos: vec![],
        };
        let error = PreparedBatch::new(arrays, sidecar).expect_err("traversal order regression");
        assert_eq!(error.category(), ErrorCategory::InternalInvariant);
    }

    #[test]
    fn matrix_shape_is_checked_before_storage_is_used() {
        let error = Matrix::new(2, 3, vec![0_i64; 5]).expect_err("wrong capacity");
        assert_eq!(error.category(), ErrorCategory::InternalInvariant);
    }

    #[test]
    fn grid_products_must_match_pixel_rows_and_use_checked_arithmetic() {
        let mut arrays = text_arrays();
        arrays.pixel_values = Some(Matrix::new(1, 1536, vec![0.0_f32; 1536]).expect("pixels"));
        arrays.image_grid_thw = Some(Matrix::new(1, 3, vec![1_i64, 2, 2]).expect("grid"));
        let sidecar = IntegrationSidecar {
            images: vec![ImageSidecar {
                request_index: 0,
                grid_row: 0,
                replacement: range(),
            }],
            videos: vec![],
        };
        assert_eq!(
            PreparedBatch::new(arrays, sidecar)
                .expect_err("one row cannot represent a four-row grid")
                .category(),
            ErrorCategory::InternalInvariant
        );

        let mut arrays = text_arrays();
        arrays.pixel_values = Some(Matrix::new(1, 1536, vec![0.0_f32; 1536]).expect("pixels"));
        arrays.image_grid_thw =
            Some(Matrix::new(1, 3, vec![i64::MAX, i64::MAX, i64::MAX]).expect("grid"));
        let sidecar = IntegrationSidecar {
            images: vec![ImageSidecar {
                request_index: 0,
                grid_row: 0,
                replacement: range(),
            }],
            videos: vec![],
        };
        assert_eq!(
            PreparedBatch::new(arrays, sidecar)
                .expect_err("grid product must be checked")
                .category(),
            ErrorCategory::ArithmeticOverflow
        );
    }
}
