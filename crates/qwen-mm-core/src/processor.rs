//! Serial composition of the frozen text and still-image stages.

use std::path::Path;

use crate::{
    ImageGeometryPlan,
    error::{ErrorCategory, QwenError, Result},
    limits::{
        ProfiledRequest, ResourceLimits, checked_add,
        preflight_batch_after_structure_through_resources,
    },
    media::{ImagePreparationPlan, PreparedRgbImage, execute_image_plan, plan_image_rgb8},
    output::{
        CoordinateRange, ImageSidecar, IntegrationSidecar, Matrix, PreparedArrays, PreparedBatch,
        ReplacementRange,
    },
    patchify::{patchify_image_rgb8, plan_image_patchify},
    profile::{Profile, ProfileRegistry},
    request::{
        ContentItem, MessageContent, OccurrenceLocation, Request, validate_request_structure,
    },
    text::{PreparedTextRequest, TextProcessor, VisualExpansion},
};

/// One completely processed still-image occurrence in traversal order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessedImageOccurrence {
    /// Original request/message/content-item coordinates and referenced input.
    pub location: OccurrenceLocation,
    /// Corresponding row in the official `image_grid_thw` array.
    pub grid_row: usize,
    /// Half-open row range in the official `pixel_values` array.
    pub pixel_rows: CoordinateRange,
    /// Source height observed before resize.
    pub source_height: u64,
    /// Source width observed before resize.
    pub source_width: u64,
    /// Exact prepared geometry and grid for this occurrence.
    pub geometry: ImageGeometryPlan,
}

struct PendingImageOccurrence {
    metadata: ProcessedImageOccurrence,
    prepared_rgb: PreparedRgbImage,
}

struct PlannedImageOccurrence<'a> {
    occurrence_index: usize,
    location: OccurrenceLocation,
    plan: ImagePreparationPlan<'a>,
}

/// One prepared RGB stage result retained only by the explicit conformance
/// trace API. Production [`QwenImageProcessor::prepare`] never retains it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TracedPreparedImage {
    /// Exact occurrence coordinates in request traversal order.
    pub location: OccurrenceLocation,
    /// The exact single-pass RGB stage output consumed by patchification.
    pub prepared_rgb: PreparedRgbImage,
}

/// Complete Phase B result plus conformance-only prepared RGB evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedImageTrace {
    /// Compact production result.
    pub output: PreparedImageRequest,
    /// Prepared RGB buffers from the same decode/resize pass.
    pub prepared_images: Vec<TracedPreparedImage>,
}

/// Complete result of the serial single-request Phase B image path.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedImageRequest {
    /// Rendered/expanded prompts and exact replacement coordinates.
    pub text: PreparedTextRequest,
    /// Complete official arrays and integration sidecar.
    pub batch: PreparedBatch,
    /// Every image occurrence in message/content traversal order.
    pub images: Vec<ProcessedImageOccurrence>,
}

/// Profile-bound native Rust processor for one text/image request.
pub struct QwenImageProcessor {
    text: TextProcessor,
    limits: ResourceLimits,
}

impl QwenImageProcessor {
    /// Loads the hash-pinned tokenizer/template assets from one local snapshot.
    ///
    /// This constructor performs no network access and has no Python fallback.
    ///
    /// # Errors
    ///
    /// Returns `profile_mismatch` when the profile or any local text asset has
    /// drifted from compatibility contract v1.
    pub fn from_local_assets(
        profile: &Profile,
        assets_directory: impl AsRef<Path>,
        limits: ResourceLimits,
    ) -> Result<Self> {
        Ok(Self {
            text: TextProcessor::from_local_assets(profile, assets_directory)?,
            limits,
        })
    }

    /// Returns the immutable profile bound to this processor.
    #[must_use]
    pub const fn profile(&self) -> &Profile {
        self.text.profile()
    }

    /// Runs the complete serial text and still-image path for one request.
    ///
    /// Media occurrences are deliberately not deduplicated. The function
    /// returns either one complete owned result or an error; intermediate RGB,
    /// token, and patch allocations are never externally observable.
    ///
    /// # Errors
    ///
    /// Returns the first stable compatibility-v1 category in request,
    /// profile, option, resource, decode, then geometry/stage order. Raw-frame
    /// and encoded video remain outside the Phase B processor.
    pub fn prepare(&self, request: Request<'_>) -> Result<PreparedImageRequest> {
        self.prepare_internal(request, false)
            .map(|(output, _)| output)
    }

    /// Runs the same serial operation while retaining each prepared RGB stage
    /// output for conformance export. This deliberately increases retained
    /// memory by the sum of prepared RGB capacities; it never decodes, resizes,
    /// normalizes, or patchifies an occurrence twice.
    ///
    /// # Errors
    ///
    /// Returns the same stable errors and precedence as [`Self::prepare`].
    pub fn prepare_with_media_trace(&self, request: Request<'_>) -> Result<PreparedImageTrace> {
        let (output, prepared_images) = self.prepare_internal(request, true)?;
        Ok(PreparedImageTrace {
            output,
            prepared_images,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn prepare_internal(
        &self,
        request: Request<'_>,
        retain_media_trace: bool,
    ) -> Result<(PreparedImageRequest, Vec<TracedPreparedImage>)> {
        validate_request_structure(&request, 0)?;
        let rendered_prompt = self.text.render_validated_request(&request, 0)?;
        let registry = ProfileRegistry::bundled()?;
        let profiled = [ProfiledRequest {
            profile_alias: self.profile().alias.as_str(),
            request,
        }];
        preflight_batch_after_structure_through_resources(&registry, &profiled, self.limits)?;
        reject_videos(&request)?;

        let mut planned_images = Vec::new();
        let mut media_errors = Vec::new();
        let mut additional_output_bytes = 0_u64;
        let mut next_pixel_row = 0_u64;
        let mut occurrence_index = 0_usize;
        for (message_index, message) in request.messages.iter().enumerate() {
            let MessageContent::Items(items) = message.content else {
                continue;
            };
            for (content_item_index, item) in items.iter().enumerate() {
                let ContentItem::Image(reference) = item else {
                    continue;
                };
                let input = request.images[reference.input_index];
                match plan_image_rgb8(
                    input,
                    &self.profile().visual,
                    reference.options,
                    self.limits,
                ) {
                    Ok(plan) => planned_images.push(PlannedImageOccurrence {
                        occurrence_index,
                        location: OccurrenceLocation {
                            request_index: 0,
                            message_index,
                            content_item_index,
                            input_index: reference.input_index,
                        },
                        plan,
                    }),
                    Err(error) => {
                        media_errors.push((occurrence_index, error));
                    }
                }
                occurrence_index += 1;
            }
        }

        let mut visuals = Vec::new();
        let mut all_geometry_valid = media_errors.is_empty();
        for occurrence in &planned_images {
            let Ok(geometry) = occurrence.plan.geometry() else {
                all_geometry_valid = false;
                continue;
            };
            let patch_plan = match plan_image_patchify(
                &self.profile().visual,
                geometry,
                geometry.rgb_capacity_bytes,
                self.limits,
            ) {
                Ok(plan) => plan,
                Err(error) => {
                    media_errors.push((occurrence.occurrence_index, error));
                    all_geometry_valid = false;
                    continue;
                }
            };
            additional_output_bytes = match checked_add(
                "composed image output bytes",
                additional_output_bytes,
                patch_plan.materialized_bytes(),
            ) {
                Ok(bytes) => bytes,
                Err(error) => {
                    media_errors.push((occurrence.occurrence_index, error));
                    all_geometry_valid = false;
                    continue;
                }
            };
            next_pixel_row = match checked_add(
                "composed image patch rows",
                next_pixel_row,
                geometry.patch_rows,
            ) {
                Ok(rows) => rows,
                Err(error) => {
                    media_errors.push((occurrence.occurrence_index, error));
                    all_geometry_valid = false;
                    continue;
                }
            };
            visuals.push(VisualExpansion::Image {
                input_index: occurrence.location.input_index,
                grid_thw: geometry.image_grid_thw,
            });
        }
        if let Some(error) = preferred_predecode_error(&media_errors) {
            return Err(error);
        }

        let text_plan = if all_geometry_valid {
            Some(self.text.plan_single_from_rendered(
                request,
                &visuals,
                rendered_prompt,
                self.limits,
                additional_output_bytes,
            )?)
        } else {
            None
        };

        let mut pending_images = Vec::new();
        for occurrence in planned_images {
            match execute_image_plan(occurrence.plan) {
                Ok(prepared) => {
                    let pixel_start = pending_images
                        .last()
                        .map_or(0_i64, |item: &PendingImageOccurrence| {
                            item.metadata.pixel_rows.end
                        });
                    let pixel_end = checked_add(
                        "executed image patch rows",
                        u64::try_from(pixel_start)
                            .map_err(|_| invariant("negative image patch row start"))?,
                        prepared.geometry.patch_rows,
                    )?;
                    pending_images.push(PendingImageOccurrence {
                        metadata: ProcessedImageOccurrence {
                            location: occurrence.location,
                            grid_row: pending_images.len(),
                            pixel_rows: CoordinateRange {
                                start: pixel_start,
                                end: to_i64(pixel_end, "image patch row end")?,
                            },
                            source_height: prepared.source_height,
                            source_width: prepared.source_width,
                            geometry: prepared.geometry,
                        },
                        prepared_rgb: prepared,
                    });
                }
                Err(error) => media_errors.push((occurrence.occurrence_index, error)),
            }
        }
        if !media_errors.is_empty() {
            return Err(select_preferred_error(media_errors));
        }
        let text_plan = text_plan.ok_or_else(|| {
            invariant("successful image execution had no complete text allocation plan")
        })?;
        let mut text = self.text.execute_single_text_plan(text_plan, self.limits)?;
        let prepared_text = text.requests.pop().ok_or_else(|| {
            invariant("single-request text processor returned no request metadata")
        })?;
        let expected_placeholders =
            pending_images.iter().try_fold(0_u64, |total, occurrence| {
                checked_add(
                    "image placeholder count",
                    total,
                    occurrence.metadata.geometry.placeholder_count,
                )
            })?;
        let actual_placeholders = u64::try_from(
            text.mm_token_type_ids
                .as_slice()
                .iter()
                .filter(|&&value| value == 1)
                .count(),
        )
        .map_err(|_| overflow("image placeholder count does not fit u64"))?;
        if actual_placeholders != expected_placeholders {
            return Err(invariant(
                "image placeholder tokens do not equal summed merged grid products",
            )
            .with_context("expected", expected_placeholders)
            .with_context("actual", actual_placeholders));
        }

        let total_pixel_elements = next_pixel_row
            .checked_mul(self.profile().visual.patch_width)
            .ok_or_else(|| overflow("composed pixel element count overflowed"))?;
        let mut pixel_values = Vec::new();
        pixel_values
            .try_reserve_exact(to_usize(total_pixel_elements, "pixel element count")?)
            .map_err(|error| allocation("pixel_values", total_pixel_elements, &error))?;
        let grid_elements = pending_images
            .len()
            .checked_mul(3)
            .ok_or_else(|| overflow("composed grid element count overflowed"))?;
        let mut image_grid_thw = Vec::new();
        image_grid_thw
            .try_reserve_exact(grid_elements)
            .map_err(|error| allocation("image_grid_thw", grid_elements as u64, &error))?;

        for occurrence in &pending_images {
            let image = patchify_image_rgb8(
                &self.profile().visual,
                occurrence.metadata.geometry,
                &occurrence.prepared_rgb.rgb,
                self.limits,
            )?;
            pixel_values.extend(image.pixel_values.into_vec());
            image_grid_thw.extend(image.image_grid_thw.into_vec());
        }

        let mut sidecar = IntegrationSidecar::default();
        for (grid_row, replacement) in prepared_text.replacements.iter().enumerate() {
            sidecar.images.push(ImageSidecar {
                request_index: 0,
                grid_row: to_i64(
                    u64::try_from(grid_row).map_err(|_| overflow("image grid row"))?,
                    "image grid row",
                )?,
                replacement: ReplacementRange {
                    code_points: replacement.rendered_code_points,
                    tokens: replacement.expanded_tokens,
                },
            });
        }
        if sidecar.images.len() != pending_images.len() {
            return Err(invariant(
                "text replacements and processed image occurrences disagree",
            ));
        }

        let (pixels, grid) = if pending_images.is_empty() {
            (None, None)
        } else {
            (
                Some(Matrix::new(
                    to_usize(next_pixel_row, "pixel row count")?,
                    to_usize(self.profile().visual.patch_width, "patch width")?,
                    pixel_values,
                )?),
                Some(Matrix::new(pending_images.len(), 3, image_grid_thw)?),
            )
        };
        let batch = PreparedBatch::new(
            PreparedArrays {
                input_ids: text.input_ids,
                attention_mask: text.attention_mask,
                mm_token_type_ids: text.mm_token_type_ids,
                pixel_values: pixels,
                image_grid_thw: grid,
                pixel_values_videos: None,
                video_grid_thw: None,
            },
            sidecar,
        )?;
        let mut images = Vec::with_capacity(pending_images.len());
        let mut prepared_images = Vec::new();
        if retain_media_trace {
            prepared_images.reserve(pending_images.len());
        }
        for occurrence in pending_images {
            images.push(occurrence.metadata);
            if retain_media_trace {
                prepared_images.push(TracedPreparedImage {
                    location: occurrence.metadata.location,
                    prepared_rgb: occurrence.prepared_rgb,
                });
            }
        }
        Ok((
            PreparedImageRequest {
                text: prepared_text,
                batch,
                images,
            },
            prepared_images,
        ))
    }
}

fn reject_videos(request: &Request<'_>) -> Result<()> {
    if request.videos.is_empty()
        && !request.messages.iter().any(|message| {
            matches!(message.content, MessageContent::Items(items) if items.iter().any(|item| matches!(item, ContentItem::Video(_))))
        })
    {
        return Ok(());
    }
    Err(QwenError::new(
        ErrorCategory::UnsupportedMedia,
        "video processing is outside the Phase B single-request image path",
    ))
}

fn select_preferred_error(errors: Vec<(usize, QwenError)>) -> QwenError {
    errors
        .into_iter()
        .min_by_key(|(occurrence, error)| (category_precedence(error.category()), *occurrence))
        .expect("caller proved at least one media error")
        .1
}

fn preferred_predecode_error(errors: &[(usize, QwenError)]) -> Option<QwenError> {
    errors
        .iter()
        .filter(|(_, error)| {
            matches!(
                error.category(),
                ErrorCategory::ArithmeticOverflow | ErrorCategory::ResourceLimit
            )
        })
        .min_by_key(|(occurrence, error)| (category_precedence(error.category()), *occurrence))
        .map(|(_, error)| error.clone())
}

const fn category_precedence(category: ErrorCategory) -> u8 {
    match category {
        ErrorCategory::ArithmeticOverflow => 0,
        ErrorCategory::ResourceLimit => 1,
        ErrorCategory::UnsupportedMedia | ErrorCategory::MediaDecode => 2,
        ErrorCategory::MediaGeometry => 3,
        ErrorCategory::DestinationTooSmall => 4,
        ErrorCategory::InternalInvariant => 5,
        ErrorCategory::InvalidRequest
        | ErrorCategory::UnsupportedOption
        | ErrorCategory::ProfileMismatch => 6,
    }
}

fn to_i64(value: u64, label: &'static str) -> Result<i64> {
    i64::try_from(value).map_err(|_| overflow(label))
}

fn to_usize(value: u64, label: &'static str) -> Result<usize> {
    usize::try_from(value).map_err(|_| overflow(label))
}

fn overflow(message: impl Into<String>) -> QwenError {
    QwenError::new(ErrorCategory::ArithmeticOverflow, message)
}

fn invariant(message: impl Into<String>) -> QwenError {
    QwenError::new(ErrorCategory::InternalInvariant, message)
}

fn allocation(
    output: &'static str,
    elements: u64,
    error: &std::collections::TryReserveError,
) -> QwenError {
    QwenError::new(
        ErrorCategory::ResourceLimit,
        "unable to reserve composed output allocation",
    )
    .with_context("output", output)
    .with_context("elements", elements)
    .with_context("detail", error.to_string())
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::QwenImageProcessor;
    use crate::{
        ContentItem, CoordinateRange, ErrorCategory, ImageFormat, ImageInput, ImageOptions,
        ImageRef, LimitOverrides, Message, MessageContent, ProfileAlias, ProfileRegistry, Request,
        RequestOptions, ResourceLimits, Rgb8, Role,
    };

    fn asset_directory(alias: ProfileAlias) -> PathBuf {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference/.cache/huggingface");
        match alias {
            ProfileAlias::Qwen3Vl8b => root.join(
                "models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
            ),
            ProfileAlias::Qwen35_9b => root.join(
                "models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            ),
        }
    }

    fn processor(alias: ProfileAlias, limits: ResourceLimits) -> QwenImageProcessor {
        let registry = ProfileRegistry::bundled().expect("profiles");
        QwenImageProcessor::from_local_assets(registry.get(alias), asset_directory(alias), limits)
            .expect("pinned local assets")
    }

    fn png_header(width: u32, height: u32) -> Vec<u8> {
        fn crc32(bytes: &[u8]) -> u32 {
            let mut crc = 0xffff_ffff_u32;
            for &byte in bytes {
                crc ^= u32::from(byte);
                for _ in 0..8 {
                    crc = (crc >> 1) ^ (0xedb8_8320_u32 & (0_u32.wrapping_sub(crc & 1)));
                }
            }
            !crc
        }
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend_from_slice(&13_u32.to_be_bytes());
        let mut chunk = b"IHDR".to_vec();
        chunk.extend_from_slice(&width.to_be_bytes());
        chunk.extend_from_slice(&height.to_be_bytes());
        chunk.extend_from_slice(&[8, 2, 0, 0, 0]);
        png.extend_from_slice(&chunk);
        png.extend_from_slice(&crc32(&chunk).to_be_bytes());
        png
    }

    fn message(role: Role, content: MessageContent<'_>) -> Message<'_> {
        Message {
            role,
            content,
            tool_calls: &[],
            reasoning_content: None,
        }
    }

    #[test]
    #[ignore = "requires hash-pinned local model snapshots under reference/.cache"]
    fn text_only_omits_all_image_outputs_for_both_profiles() {
        let messages = [message(Role::User, MessageContent::Text("hello"))];
        let request = Request {
            messages: &messages,
            images: &[],
            videos: &[],
            options: RequestOptions::default(),
        };
        for alias in [ProfileAlias::Qwen3Vl8b, ProfileAlias::Qwen35_9b] {
            let output = processor(alias, ResourceLimits::default())
                .prepare(request)
                .expect("text only");
            assert_eq!(
                output.batch.official_keys(),
                ["input_ids", "attention_mask", "mm_token_type_ids"]
            );
            assert!(output.images.is_empty());
            assert!(output.batch.sidecar().images.is_empty());
        }
    }

    #[test]
    #[ignore = "requires hash-pinned local model snapshots under reference/.cache"]
    #[allow(clippy::too_many_lines)]
    fn traversal_order_repeated_reference_and_per_occurrence_options_are_preserved() {
        let first = vec![11_u8; 64 * 64 * 3];
        let second = vec![22_u8; 64 * 64 * 3];
        let images = [
            ImageInput::Rgb8(Rgb8 {
                data: &first,
                height: 64,
                width: 64,
                row_stride: 64 * 3,
            }),
            ImageInput::Rgb8(Rgb8 {
                data: &second,
                height: 64,
                width: 64,
                row_stride: 64 * 3,
            }),
        ];
        let items = [
            ContentItem::Image(ImageRef {
                input_index: 1,
                options: ImageOptions {
                    resized_height: Some(64),
                    resized_width: Some(64),
                    ..ImageOptions::default()
                },
            }),
            ContentItem::Text(" then "),
            ContentItem::Image(ImageRef {
                input_index: 0,
                options: ImageOptions {
                    resized_height: Some(64),
                    resized_width: Some(96),
                    ..ImageOptions::default()
                },
            }),
            ContentItem::Text(" and again "),
            ContentItem::Image(ImageRef {
                input_index: 1,
                options: ImageOptions {
                    resized_height: Some(96),
                    resized_width: Some(64),
                    ..ImageOptions::default()
                },
            }),
        ];
        let messages = [message(Role::User, MessageContent::Items(&items))];
        let request = Request {
            messages: &messages,
            images: &images,
            videos: &[],
            options: RequestOptions::default(),
        };
        let output = processor(ProfileAlias::Qwen3Vl8b, ResourceLimits::default())
            .prepare(request)
            .expect("interleaved repeated image request");
        assert_eq!(
            output
                .images
                .iter()
                .map(|item| item.location.input_index)
                .collect::<Vec<_>>(),
            [1, 0, 1]
        );
        assert_eq!(
            output
                .images
                .iter()
                .map(|item| [item.geometry.height, item.geometry.width])
                .collect::<Vec<_>>(),
            [[64, 64], [64, 96], [96, 64]]
        );
        assert_eq!(
            output
                .images
                .iter()
                .map(|item| item.grid_row)
                .collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert_eq!(output.text.replacements.len(), 3);
        assert_eq!(
            output
                .batch
                .arrays()
                .mm_token_type_ids
                .as_slice()
                .iter()
                .filter(|&&value| value == 1)
                .count(),
            output
                .images
                .iter()
                .map(|item| usize::try_from(item.geometry.placeholder_count).expect("count"))
                .sum::<usize>()
        );
        assert_eq!(output.batch.sidecar().images.len(), 3);
        assert_eq!(
            output
                .batch
                .arrays()
                .image_grid_thw
                .as_ref()
                .expect("grid")
                .as_slice(),
            [1, 4, 4, 1, 4, 6, 1, 6, 4]
        );
    }

    #[test]
    #[ignore = "requires hash-pinned local model snapshots under reference/.cache"]
    fn late_media_failure_is_atomic_and_does_not_poison_processor_reuse() {
        let processor = processor(ProfileAlias::Qwen3Vl8b, ResourceLimits::default());
        let valid_rgb = vec![23_u8; 64 * 64 * 3];
        let corrupt_png = png_header(64, 64);
        let failed_images = [
            ImageInput::Rgb8(Rgb8 {
                data: &valid_rgb,
                height: 64,
                width: 64,
                row_stride: 64 * 3,
            }),
            ImageInput::Encoded {
                data: &corrupt_png,
                format: ImageFormat::Png,
            },
        ];
        let failed_items = [
            ContentItem::Image(ImageRef::default()),
            ContentItem::Text(" then "),
            ContentItem::Image(ImageRef {
                input_index: 1,
                ..ImageRef::default()
            }),
        ];
        let failed_messages = [message(Role::User, MessageContent::Items(&failed_items))];
        let error = processor
            .prepare(Request {
                messages: &failed_messages,
                images: &failed_images,
                videos: &[],
                options: RequestOptions::default(),
            })
            .expect_err("second occurrence must fail after the first is prepared");
        assert_eq!(error.category(), ErrorCategory::MediaDecode);

        let valid_images = [failed_images[0]];
        let valid_items = [ContentItem::Image(ImageRef::default())];
        let valid_messages = [message(Role::User, MessageContent::Items(&valid_items))];
        let output = processor
            .prepare(Request {
                messages: &valid_messages,
                images: &valid_images,
                videos: &[],
                options: RequestOptions::default(),
            })
            .expect("same processor remains reusable after the atomic failure");
        assert_eq!(output.images.len(), 1);
        assert_eq!(output.text.replacements.len(), 1);
        assert_eq!(output.batch.sidecar().images.len(), 1);
        assert_eq!(
            output.images[0].pixel_rows,
            CoordinateRange { start: 0, end: 16 }
        );
    }

    #[test]
    #[ignore = "requires hash-pinned local model snapshots under reference/.cache"]
    fn request_and_render_failures_precede_corrupt_media_then_decode_precedes_geometry() {
        let corrupt_png = [137, 80, 78, 71, 13, 10, 26, 10];
        let images = [ImageInput::Encoded {
            data: &corrupt_png,
            format: ImageFormat::Png,
        }];
        let bad_geometry_items = [ContentItem::Image(ImageRef {
            input_index: 0,
            options: ImageOptions {
                resized_height: Some(64),
                ..ImageOptions::default()
            },
        })];
        let bad_geometry_messages = [message(
            Role::User,
            MessageContent::Items(&bad_geometry_items),
        )];
        let request = Request {
            messages: &bad_geometry_messages,
            images: &images,
            videos: &[],
            options: RequestOptions::default(),
        };
        assert_eq!(
            processor(ProfileAlias::Qwen3Vl8b, ResourceLimits::default())
                .prepare(request)
                .expect_err("corrupt decode wins")
                .category(),
            ErrorCategory::MediaDecode
        );

        let literal_items = [
            ContentItem::Text("literal <|image_pad|>"),
            ContentItem::Image(ImageRef::default()),
        ];
        let literal_messages = [message(Role::User, MessageContent::Items(&literal_items))];
        let literal = Request {
            messages: &literal_messages,
            ..request
        };
        assert_eq!(
            processor(
                ProfileAlias::Qwen3Vl8b,
                ResourceLimits::default()
                    .lowered(LimitOverrides {
                        encoded_bytes_per_item: Some(0),
                        encoded_bytes_per_batch: Some(0),
                        materialized_output_bytes_per_batch: Some(1),
                        ..LimitOverrides::default()
                    })
                    .expect("lowered limits"),
            )
            .prepare(literal)
            .expect_err("rendered mismatch wins over resource preflight")
            .category(),
            ErrorCategory::InvalidRequest
        );

        let qwen35_messages = [message(
            Role::Tool,
            MessageContent::Items(&bad_geometry_items),
        )];
        let qwen35 = Request {
            messages: &qwen35_messages,
            ..request
        };
        assert_eq!(
            processor(ProfileAlias::Qwen35_9b, ResourceLimits::default())
                .prepare(qwen35)
                .expect_err("missing Qwen3.5 user query wins")
                .category(),
            ErrorCategory::InvalidRequest
        );
    }

    #[test]
    #[ignore = "requires hash-pinned local model snapshots under reference/.cache"]
    fn aggregate_output_limit_is_checked_before_any_patch_matrix_is_built() {
        let rgb = vec![0_u8; 64 * 64 * 3];
        let images = [ImageInput::Rgb8(Rgb8 {
            data: &rgb,
            height: 64,
            width: 64,
            row_stride: 64 * 3,
        })];
        let items = [
            ContentItem::Image(ImageRef::default()),
            ContentItem::Image(ImageRef::default()),
        ];
        let messages = [message(Role::User, MessageContent::Items(&items))];
        let request = Request {
            messages: &messages,
            images: &images,
            videos: &[],
            options: RequestOptions::default(),
        };
        let limits = ResourceLimits::default()
            .lowered(LimitOverrides {
                materialized_output_bytes_per_batch: Some(120_000),
                ..LimitOverrides::default()
            })
            .expect("lowered limits");
        assert_eq!(
            processor(ProfileAlias::Qwen3Vl8b, limits)
                .prepare(request)
                .expect_err("combined arrays exceed cap")
                .category(),
            ErrorCategory::ResourceLimit
        );
    }

    #[test]
    #[ignore = "requires hash-pinned local model snapshots under reference/.cache"]
    #[allow(clippy::too_many_lines)]
    fn media_failures_are_selected_by_global_stage_not_first_occurrence() {
        let corrupt_png = [137, 80, 78, 71, 13, 10, 26, 10];
        let raw = vec![7_u8; 64 * 64 * 3];
        let geometry_then_decode_inputs = [
            ImageInput::Rgb8(Rgb8 {
                data: &raw,
                height: 64,
                width: 64,
                row_stride: 64 * 3,
            }),
            ImageInput::Encoded {
                data: &corrupt_png,
                format: ImageFormat::Png,
            },
        ];
        let geometry_then_decode_items = [
            ContentItem::Image(ImageRef {
                input_index: 0,
                options: ImageOptions {
                    resized_height: Some(64),
                    ..ImageOptions::default()
                },
            }),
            ContentItem::Image(ImageRef {
                input_index: 1,
                options: ImageOptions::default(),
            }),
        ];
        let messages = [message(
            Role::User,
            MessageContent::Items(&geometry_then_decode_items),
        )];
        let request = Request {
            messages: &messages,
            images: &geometry_then_decode_inputs,
            videos: &[],
            options: RequestOptions::default(),
        };
        assert_eq!(
            processor(ProfileAlias::Qwen3Vl8b, ResourceLimits::default())
                .prepare(request)
                .expect_err("later decode must beat earlier geometry")
                .category(),
            ErrorCategory::MediaDecode
        );

        let decode_then_resource_inputs = [
            ImageInput::Encoded {
                data: &corrupt_png,
                format: ImageFormat::Png,
            },
            ImageInput::Rgb8(Rgb8 {
                data: &raw,
                height: 64,
                width: 64,
                row_stride: 64 * 3,
            }),
        ];
        let decode_then_resource_items = [
            ContentItem::Image(ImageRef {
                input_index: 0,
                options: ImageOptions::default(),
            }),
            ContentItem::Image(ImageRef {
                input_index: 1,
                options: ImageOptions {
                    resized_height: Some(96),
                    resized_width: Some(96),
                    ..ImageOptions::default()
                },
            }),
        ];
        let messages = [message(
            Role::User,
            MessageContent::Items(&decode_then_resource_items),
        )];
        let request = Request {
            messages: &messages,
            images: &decode_then_resource_inputs,
            videos: &[],
            options: RequestOptions::default(),
        };
        let limits = ResourceLimits::default()
            .lowered(LimitOverrides {
                materialized_output_bytes_per_batch: Some(120_000),
                ..LimitOverrides::default()
            })
            .expect("lowered limits");
        assert_eq!(
            processor(ProfileAlias::Qwen3Vl8b, limits)
                .prepare(request)
                .expect_err("later resource must beat earlier decode")
                .category(),
            ErrorCategory::ResourceLimit
        );

        let corrupt_96 = png_header(96, 96);
        let images = [ImageInput::Encoded {
            data: &corrupt_96,
            format: ImageFormat::Png,
        }];
        let items = [ContentItem::Image(ImageRef::default())];
        let messages = [message(Role::User, MessageContent::Items(&items))];
        let request = Request {
            messages: &messages,
            images: &images,
            videos: &[],
            options: RequestOptions::default(),
        };
        let limits = ResourceLimits::default()
            .lowered(LimitOverrides {
                materialized_output_bytes_per_batch: Some(120_000),
                ..LimitOverrides::default()
            })
            .expect("lowered limits");
        assert_eq!(
            processor(ProfileAlias::Qwen3Vl8b, limits)
                .prepare(request)
                .expect_err("final output capacity must win before full decode")
                .category(),
            ErrorCategory::ResourceLimit
        );
    }
}
