//! Serial composition of the frozen text and still-image stages.

use std::{mem, path::Path};

use sha2::{Digest, Sha256};

use crate::{
    ImageGeometryPlan,
    error::{ErrorCategory, QwenError, Result},
    limits::{
        ProfiledRequest, ResourceLimits, checked_add, checked_capacity_bytes,
        preflight_batch_after_structure_through_resources,
    },
    media::{
        ImagePreparationPlan, PreparedRgbImage, execute_image_plan, execute_image_plan_observed,
        plan_image_rgb8,
    },
    observability::{
        BufferClass, DEFAULT_OBSERVATION_EVENT_CAPACITY, ObservationRecorder, ObservationScope,
        Observed,
    },
    output::{
        CoordinateRange, ImageSidecar, IntegrationSidecar, Matrix, MatrixView, PreparedArrayViews,
        PreparedArrays, PreparedBatch, ReplacementRange,
    },
    patchify::{
        ImagePatchifyPlan, execute_image_patchify_plan_into, patchify_image_rgb8,
        plan_image_patchify,
    },
    profile::{Profile, ProfileRegistry},
    request::{
        ContentItem, ImageInput, MessageContent, OccurrenceLocation, Request,
        validate_request_structure,
    },
    text::{BatchTextPlan, PreparedTextRequest, TextProcessor, VisualExpansion},
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

/// One processed batch image plus its stable prepared-media identity.
///
/// This batch-specific wrapper preserves the Phase B
/// [`ProcessedImageOccurrence`] API while carrying the A6 cache contract needed
/// by Python and vLLM adapters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessedBatchImageOccurrence {
    /// Existing Phase B occurrence metadata and exact output ranges.
    pub occurrence: ProcessedImageOccurrence,
    /// Stable SHA-256 identity bound to source, profile, and image options.
    pub cache_key: [u8; 32],
}

impl ProcessedBatchImageOccurrence {
    /// Returns the stable cache identity as lowercase hexadecimal SHA-256.
    #[must_use]
    pub fn cache_key_hex(self) -> String {
        hex_digest(self.cache_key)
    }
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

#[derive(Debug)]
struct BatchPlannedImageOccurrence {
    occurrence_index: usize,
    location: OccurrenceLocation,
    prepared_rgb: PreparedRgbImage,
    patch_plan: ImagePatchifyPlan,
    cache_key: [u8; 32],
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

/// Exact element and byte capacity for one C-contiguous public array.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArrayCapacity {
    /// Row-major two-dimensional shape.
    pub shape: [usize; 2],
    /// Exact number of elements required by the shape.
    pub elements: usize,
    /// Exact storage in bytes at the official dtype.
    pub bytes: u64,
}

/// Exact capacities for every official array in one batch plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchCapacities {
    /// Required `int64` token IDs.
    pub input_ids: ArrayCapacity,
    /// Required `int64` attention mask.
    pub attention_mask: ArrayCapacity,
    /// Required `int64` modality IDs.
    pub mm_token_type_ids: ArrayCapacity,
    /// Required `float32` still-image patches, absent for text-only batches.
    pub pixel_values: Option<ArrayCapacity>,
    /// Required `int64` still-image grids, absent for text-only batches.
    pub image_grid_thw: Option<ArrayCapacity>,
    /// Video pixels are outside the Phase C image batch API.
    pub pixel_values_videos: Option<ArrayCapacity>,
    /// Video grids are outside the Phase C image batch API.
    pub video_grid_thw: Option<ArrayCapacity>,
}

/// A half-open range in one flattened planned output.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BatchOutputRange {
    /// Inclusive element or row offset.
    pub start: usize,
    /// Exclusive element or row offset.
    pub end: usize,
}

/// Per-request offsets, token count, and right-padding requirement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchRequestLayout {
    /// Request row in the planned batch.
    pub request_index: usize,
    /// Flattened range in each official text matrix.
    pub text_elements: BatchOutputRange,
    /// Number of non-padding tokens in this row.
    pub token_count: usize,
    /// Number of right-padding elements in this row.
    pub right_padding: usize,
    /// Rows in `image_grid_thw` belonging to this request.
    pub image_grid_rows: BatchOutputRange,
    /// Rows in `pixel_values` belonging to this request.
    pub pixel_rows: BatchOutputRange,
}

/// Exact planned output coordinates for one still-image occurrence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchImageLayout {
    /// Original request/message/content coordinates.
    pub location: OccurrenceLocation,
    /// Row in `image_grid_thw`.
    pub grid_row: usize,
    /// Half-open row range in `pixel_values`.
    pub pixel_rows: BatchOutputRange,
    /// Planned prepared geometry and grid.
    pub geometry: ImageGeometryPlan,
    /// Stable SHA-256 identity bound to source, profile, and image options.
    pub cache_key: [u8; 32],
}

/// Reusable owned serial plan retaining prepared RGB and checked metadata.
#[derive(Debug)]
pub struct BatchPlan {
    text: BatchTextPlan,
    images: Vec<BatchPlannedImageOccurrence>,
    processed_images: Vec<ProcessedBatchImageOccurrence>,
    sidecar: IntegrationSidecar,
    capacities: BatchCapacities,
    request_layouts: Vec<BatchRequestLayout>,
    image_layouts: Vec<BatchImageLayout>,
    contract_id: String,
    profile_fingerprint: String,
    limits: ResourceLimits,
}

impl BatchPlan {
    /// Returns exact capacities for every conditional official output.
    #[must_use]
    pub const fn capacities(&self) -> &BatchCapacities {
        &self.capacities
    }

    /// Returns request rows and their exact text/image offsets.
    #[must_use]
    pub fn request_layouts(&self) -> &[BatchRequestLayout] {
        &self.request_layouts
    }

    /// Returns image occurrences in request/message/content traversal order.
    #[must_use]
    pub fn image_layouts(&self) -> &[BatchImageLayout] {
        &self.image_layouts
    }

    /// Returns processed occurrence metadata in deterministic batch order.
    #[must_use]
    pub fn processed_images(&self) -> &[ProcessedBatchImageOccurrence] {
        &self.processed_images
    }

    /// Returns the validated adapter sidecar retained by the plan.
    #[must_use]
    pub const fn sidecar(&self) -> &IntegrationSidecar {
        &self.sidecar
    }

    /// Returns the immutable compatibility contract identifier.
    #[must_use]
    pub fn contract_id(&self) -> &str {
        &self.contract_id
    }

    /// Returns the full immutable profile fingerprint.
    #[must_use]
    pub fn profile_fingerprint(&self) -> &str {
        &self.profile_fingerprint
    }

    /// Records the actual release of prepared-RGB plan storage and consumes
    /// the plan. Bindings should call this after their final metadata borrow.
    pub fn drop_observed(self, recorder: &mut ObservationRecorder) {
        for image in &self.images {
            recorder.release_transient(
                "prepared_rgb",
                media_observation_scope(image.location, image.occurrence_index),
                u8_vector_capacity_bytes(&image.prepared_rgb.rgb),
            );
        }
        drop(self);
    }
}

impl BatchImageLayout {
    /// Returns the stable cache identity as lowercase hexadecimal SHA-256.
    #[must_use]
    pub fn cache_key_hex(self) -> String {
        hex_digest(self.cache_key)
    }
}

/// Caller-owned storage for every official output of an image batch plan.
pub struct BatchDestinations<'a> {
    /// `int64 [batch, sequence]` storage.
    pub input_ids: &'a mut [i64],
    /// `int64 [batch, sequence]` storage.
    pub attention_mask: &'a mut [i64],
    /// `int64 [batch, sequence]` storage.
    pub mm_token_type_ids: &'a mut [i64],
    /// `float32 [image_patches, 1536]`, required exactly when planned.
    pub pixel_values: Option<&'a mut [f32]>,
    /// `int64 [image_occurrences, 3]`, required exactly when planned.
    pub image_grid_thw: Option<&'a mut [i64]>,
    /// Reserved video pixels; must match the plan's conditional presence.
    pub pixel_values_videos: Option<&'a mut [f32]>,
    /// Reserved video grid; must match the plan's conditional presence.
    pub video_grid_thw: Option<&'a mut [i64]>,
}

/// Complete borrowed batch result backed by caller-owned official arrays.
#[derive(Debug, PartialEq)]
pub struct PreparedBatchView<'plan, 'destination> {
    /// Immutable compatibility contract identifier.
    pub contract_id: &'plan str,
    /// Full immutable profile fingerprint.
    pub profile_fingerprint: &'plan str,
    /// Per-request rendered/expanded prompts and replacement coordinates.
    pub text: &'plan [PreparedTextRequest],
    arrays: PreparedArrayViews<'destination>,
    sidecar: &'plan IntegrationSidecar,
    /// Every image occurrence in deterministic batch traversal order.
    pub images: &'plan [ProcessedBatchImageOccurrence],
}

impl PreparedBatchView<'_, '_> {
    /// Returns all borrowed official arrays.
    #[must_use]
    pub const fn arrays(&self) -> &PreparedArrayViews<'_> {
        &self.arrays
    }

    /// Returns ordered adapter-only metadata.
    #[must_use]
    pub const fn sidecar(&self) -> &IntegrationSidecar {
        self.sidecar
    }

    /// Returns official keys in frozen processor order.
    #[must_use]
    pub fn official_keys(&self) -> Vec<&'static str> {
        let mut keys = vec!["input_ids", "attention_mask", "mm_token_type_ids"];
        if self.arrays.pixel_values.is_some() {
            keys.extend(["pixel_values", "image_grid_thw"]);
        }
        keys
    }
}

/// Complete owned result of the allocating serial batch wrapper.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedImageBatch {
    /// Immutable compatibility contract identifier.
    pub contract_id: String,
    /// Full immutable profile fingerprint.
    pub profile_fingerprint: String,
    /// Per-request rendered/expanded prompts and replacement coordinates.
    pub text: Vec<PreparedTextRequest>,
    /// Complete official arrays and integration sidecar.
    pub batch: PreparedBatch,
    /// Every image occurrence in deterministic batch traversal order.
    pub images: Vec<ProcessedBatchImageOccurrence>,
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

    /// Plans a heterogeneous text/still-image batch and completes all media
    /// decode, geometry, resize, token, and capacity validation without
    /// touching caller-owned output memory.
    ///
    /// The returned plan owns prepared RGB snapshots and is reusable. Its
    /// identity is bound to this processor's contract, profile, and limits.
    ///
    /// # Errors
    ///
    /// Returns the first stable compatibility-v1 failure in whole-batch stage
    /// order. Aggregate resource/capacity failures precede full media decode.
    #[allow(clippy::too_many_lines)]
    pub fn plan_batch(&self, requests: &[Request<'_>]) -> Result<BatchPlan> {
        self.plan_batch_internal(requests, None)
    }

    /// Observed counterpart to [`Self::plan_batch`]. Instrumentation is
    /// bounded by the caller's recorder and does not change validation order.
    ///
    /// # Errors
    ///
    /// Returns the same stable errors as [`Self::plan_batch`].
    pub fn plan_batch_observed(
        &self,
        requests: &[Request<'_>],
        recorder: &mut ObservationRecorder,
    ) -> Result<BatchPlan> {
        let span = recorder.begin("native.batch.plan", ObservationScope::default(), 0);
        let result = self.plan_batch_internal(requests, Some(recorder));
        match &result {
            Ok(plan) => recorder.finish_success(
                span,
                total_capacity_bytes(&plan.capacities),
                &[requests.len() as u64, plan.images.len() as u64],
            ),
            Err(error) => {
                recorder.finish_error(span, error);
            }
        }
        result
    }

    #[allow(clippy::too_many_lines)]
    fn plan_batch_internal(
        &self,
        requests: &[Request<'_>],
        mut recorder: Option<&mut ObservationRecorder>,
    ) -> Result<BatchPlan> {
        if requests.is_empty() {
            return Err(QwenError::new(
                ErrorCategory::InvalidRequest,
                "batch must contain at least one request",
            ));
        }

        let mut rendered_prompts = Vec::with_capacity(requests.len());
        for (request_index, request) in requests.iter().enumerate() {
            let validate_span = recorder.as_deref_mut().map(|recorder| {
                recorder.begin(
                    "native.request.validate",
                    ObservationScope::request(request_index),
                    0,
                )
            });
            if let Err(error) = validate_request_structure(request, request_index) {
                if let (Some(recorder), Some(span)) = (recorder.as_deref_mut(), validate_span) {
                    recorder.finish_error(span, &error);
                }
                return Err(error);
            }
            if let (Some(recorder), Some(span)) = (recorder.as_deref_mut(), validate_span) {
                recorder.finish_success(span, 0, &[]);
            }
            let render_span = recorder.as_deref_mut().map(|recorder| {
                recorder.begin(
                    "native.chat.render",
                    ObservationScope::request(request_index),
                    0,
                )
            });
            match self.text.render_validated_request(request, request_index) {
                Ok(prompt) => {
                    if let (Some(recorder), Some(span)) = (recorder.as_deref_mut(), render_span) {
                        recorder.finish_success(span, prompt.len() as u64, &[prompt.len() as u64]);
                    }
                    rendered_prompts.push(prompt);
                }
                Err(error) => {
                    if let (Some(recorder), Some(span)) = (recorder.as_deref_mut(), render_span) {
                        recorder.finish_error(span, &error);
                    }
                    return Err(error);
                }
            }
        }
        let registry = ProfileRegistry::bundled()?;
        let profiled = requests
            .iter()
            .copied()
            .map(|request| ProfiledRequest {
                profile_alias: self.profile().alias.as_str(),
                request,
            })
            .collect::<Vec<_>>();
        preflight_batch_after_structure_through_resources(&registry, &profiled, self.limits)?;
        let mut pending_images = Vec::new();
        let mut media_errors = Vec::new();
        let mut visuals = vec![Vec::new(); requests.len()];
        let mut image_layouts = Vec::new();
        let mut request_layouts = Vec::with_capacity(requests.len());
        let mut additional_output_bytes = 0_u64;
        let mut next_pixel_row = 0_u64;
        let mut occurrence_index = 0_usize;

        for (request_index, request) in requests.iter().enumerate() {
            let request_grid_start = image_layouts.len();
            let request_pixel_start = next_pixel_row;
            for (message_index, message) in request.messages.iter().enumerate() {
                let MessageContent::Items(items) = message.content else {
                    continue;
                };
                for (content_item_index, item) in items.iter().enumerate() {
                    if matches!(item, ContentItem::Image(_) | ContentItem::Video(_))
                        && let Some(recorder) = recorder.as_deref_mut()
                    {
                        let calls = recorder.calls_mut();
                        calls.native_visual_calls = calls.native_visual_calls.saturating_add(1);
                    }
                    let reference = match item {
                        ContentItem::Image(reference) => reference,
                        ContentItem::Video(_) => {
                            media_errors.push((
                                occurrence_index,
                                unsupported_batch_video()
                                    .with_context("request_index", request_index)
                                    .with_context("message_index", message_index)
                                    .with_context("content_item_index", content_item_index),
                            ));
                            occurrence_index += 1;
                            continue;
                        }
                        ContentItem::Text(_) => continue,
                    };
                    let input = request.images[reference.input_index];
                    let location = OccurrenceLocation {
                        request_index,
                        message_index,
                        content_item_index,
                        input_index: reference.input_index,
                    };
                    let scope = media_observation_scope(location, occurrence_index);
                    let plan_span = recorder.as_deref_mut().map(|recorder| {
                        recorder.begin("native.media.plan", scope, image_input_bytes(input))
                    });
                    let planned = (|| {
                        let plan = plan_image_rgb8(
                            input,
                            &self.profile().visual,
                            reference.options,
                            self.limits,
                        )?;
                        let geometry = plan.geometry()?;
                        let patch_plan = plan_image_patchify(
                            &self.profile().visual,
                            geometry,
                            geometry.rgb_capacity_bytes,
                            self.limits,
                        )?;
                        let next_output_bytes = checked_add(
                            "composed image output bytes",
                            additional_output_bytes,
                            patch_plan.materialized_bytes(),
                        )?;
                        let pixel_start = next_pixel_row;
                        let pixel_end = checked_add(
                            "composed image patch rows",
                            next_pixel_row,
                            geometry.patch_rows,
                        )?;
                        let pixel_rows = BatchOutputRange {
                            start: to_usize(pixel_start, "pixel row start")?,
                            end: to_usize(pixel_end, "pixel row end")?,
                        };
                        Result::Ok((
                            plan,
                            geometry,
                            patch_plan,
                            next_output_bytes,
                            pixel_end,
                            pixel_rows,
                        ))
                    })();
                    match planned {
                        Ok((
                            plan,
                            geometry,
                            patch_plan,
                            next_output_bytes,
                            pixel_end,
                            pixel_rows,
                        )) => {
                            if let (Some(recorder), Some(span)) =
                                (recorder.as_deref_mut(), plan_span)
                            {
                                recorder.finish_success(
                                    span,
                                    geometry.rgb_capacity_bytes,
                                    &[geometry.height, geometry.width, 3],
                                );
                            }
                            additional_output_bytes = next_output_bytes;
                            next_pixel_row = pixel_end;
                            let grid_row = image_layouts.len();
                            image_layouts.push(BatchImageLayout {
                                location,
                                grid_row,
                                pixel_rows,
                                geometry,
                                cache_key: [0; 32],
                            });
                            visuals[request_index].push(VisualExpansion::Image {
                                input_index: reference.input_index,
                                grid_thw: geometry.image_grid_thw,
                            });
                            pending_images.push((
                                PlannedImageOccurrence {
                                    occurrence_index,
                                    location,
                                    plan,
                                },
                                patch_plan,
                                input,
                                reference.options,
                            ));
                        }
                        Err(error) => {
                            if error.category() == ErrorCategory::MediaDecode
                                && let Some(recorder) = recorder.as_deref_mut()
                            {
                                let decode_span = recorder.begin(
                                    "native.media.decode_color",
                                    scope,
                                    image_input_bytes(input),
                                );
                                recorder.finish_error(decode_span, &error);
                            }
                            if let (Some(recorder), Some(span)) =
                                (recorder.as_deref_mut(), plan_span)
                            {
                                recorder.finish_error(span, &error);
                            }
                            media_errors.push((occurrence_index, error));
                        }
                    }
                    occurrence_index += 1;
                }
            }
            request_layouts.push(BatchRequestLayout {
                request_index,
                text_elements: BatchOutputRange::default(),
                token_count: 0,
                right_padding: 0,
                image_grid_rows: BatchOutputRange {
                    start: request_grid_start,
                    end: image_layouts.len(),
                },
                pixel_rows: BatchOutputRange {
                    start: to_usize(request_pixel_start, "request pixel row start")?,
                    end: to_usize(next_pixel_row, "request pixel row end")?,
                },
            });
        }
        if let Some(error) = preferred_predecode_error(&media_errors) {
            return Err(error);
        }
        if !media_errors.is_empty() {
            for (occurrence, _, _, _) in &pending_images {
                let scope =
                    media_observation_scope(occurrence.location, occurrence.occurrence_index);
                let result = if let Some(recorder) = recorder.as_deref_mut() {
                    execute_image_plan_observed(occurrence.plan.clone(), recorder, scope)
                } else {
                    execute_image_plan(occurrence.plan.clone())
                };
                match result {
                    Ok(prepared) => {
                        if let Some(recorder) = recorder.as_deref_mut() {
                            recorder.release_transient(
                                "prepared_rgb",
                                scope,
                                u8_vector_capacity_bytes(&prepared.rgb),
                            );
                        }
                        drop(prepared);
                    }
                    Err(error) => media_errors.push((occurrence.occurrence_index, error)),
                }
            }
            return Err(select_preferred_error(media_errors));
        }

        let text = if let Some(recorder) = recorder.as_deref_mut() {
            self.text.plan_batch_from_rendered_observed(
                requests,
                &visuals,
                rendered_prompts,
                self.limits,
                additional_output_bytes,
                recorder,
            )?
        } else {
            self.text.plan_batch_from_rendered(
                requests,
                &visuals,
                rendered_prompts,
                self.limits,
                additional_output_bytes,
            )?
        };
        let expected_placeholders = image_layouts.iter().try_fold(0_u64, |total, image| {
            checked_add(
                "image placeholder count",
                total,
                image.geometry.placeholder_count,
            )
        })?;
        let actual_placeholders = u64::try_from(self.text.image_token_count(&text))
            .map_err(|_| overflow("image placeholder count does not fit u64"))?;
        if actual_placeholders != expected_placeholders {
            return Err(invariant(
                "image placeholder tokens do not equal summed merged grid products",
            )
            .with_context("expected", expected_placeholders)
            .with_context("actual", actual_placeholders));
        }

        let text_elements = text.rows().checked_mul(text.columns()).ok_or_else(|| {
            overflow("text matrix element capacity overflowed")
                .with_context("rows", text.rows())
                .with_context("columns", text.columns())
        })?;
        for layout in &mut request_layouts {
            let start = layout
                .request_index
                .checked_mul(text.columns())
                .ok_or_else(|| {
                    overflow("request text row offset overflowed")
                        .with_context("request_index", layout.request_index)
                })?;
            layout.text_elements = BatchOutputRange {
                start,
                end: start + text.columns(),
            };
            layout.token_count = text.token_count(layout.request_index);
            layout.right_padding = text.columns() - layout.token_count;
        }

        let text_capacity = array_capacity(text.rows(), text.columns(), mem::size_of::<i64>())?;
        debug_assert_eq!(text_capacity.elements, text_elements);
        let (pixel_values, image_grid_thw) = if image_layouts.is_empty() {
            (None, None)
        } else {
            (
                Some(array_capacity(
                    to_usize(next_pixel_row, "pixel row count")?,
                    to_usize(self.profile().visual.patch_width, "patch width")?,
                    mem::size_of::<f32>(),
                )?),
                Some(array_capacity(
                    image_layouts.len(),
                    3,
                    mem::size_of::<i64>(),
                )?),
            )
        };
        let capacities = BatchCapacities {
            input_ids: text_capacity,
            attention_mask: text_capacity,
            mm_token_type_ids: text_capacity,
            pixel_values,
            image_grid_thw,
            pixel_values_videos: None,
            video_grid_thw: None,
        };

        let mut images = Vec::with_capacity(pending_images.len());
        let mut decode_errors = Vec::new();
        for (occurrence, patch_plan, input, options) in pending_images {
            let scope = media_observation_scope(occurrence.location, occurrence.occurrence_index);
            let result = if let Some(recorder) = recorder.as_deref_mut() {
                execute_image_plan_observed(occurrence.plan, recorder, scope)
            } else {
                execute_image_plan(occurrence.plan)
            };
            match result {
                Ok(prepared_rgb) => {
                    let cache_key = image_cache_key(self.profile(), input, options);
                    images.push(BatchPlannedImageOccurrence {
                        occurrence_index: occurrence.occurrence_index,
                        location: occurrence.location,
                        prepared_rgb,
                        patch_plan,
                        cache_key,
                    });
                }
                Err(error) => decode_errors.push((occurrence.occurrence_index, error)),
            }
        }
        if !decode_errors.is_empty() {
            if let Some(recorder) = recorder {
                release_planned_images(&images, recorder);
            }
            return Err(select_preferred_error(decode_errors));
        }
        for (layout, image) in image_layouts.iter_mut().zip(&images) {
            layout.cache_key = image.cache_key;
        }

        let mut sidecar = IntegrationSidecar::default();
        let mut replacement_index = 0_usize;
        for (request_index, text_request) in text.requests().iter().enumerate() {
            for replacement in &text_request.replacements {
                let image = image_layouts.get(replacement_index).ok_or_else(|| {
                    invariant("text replacements and planned image occurrences disagree")
                })?;
                if image.location.request_index != request_index {
                    return Err(invariant(
                        "text replacement request and image plan request disagree",
                    ));
                }
                sidecar.images.push(ImageSidecar {
                    request_index: usize_to_i64(request_index, "sidecar request index")?,
                    grid_row: usize_to_i64(image.grid_row, "image grid row")?,
                    replacement: ReplacementRange {
                        code_points: replacement.rendered_code_points,
                        tokens: replacement.expanded_tokens,
                    },
                });
                replacement_index += 1;
            }
        }
        if replacement_index != images.len() {
            return Err(invariant(
                "text replacements and planned image occurrences disagree",
            ));
        }

        let mut processed_images = Vec::with_capacity(images.len());
        for (image, layout) in images.iter().zip(&image_layouts) {
            if image.location != layout.location
                || image.prepared_rgb.geometry != layout.geometry
                || image.patch_plan.geometry() != layout.geometry
                || image.cache_key != layout.cache_key
            {
                return Err(invariant("owned image plan metadata is inconsistent"));
            }
            processed_images.push(ProcessedBatchImageOccurrence {
                occurrence: ProcessedImageOccurrence {
                    location: layout.location,
                    grid_row: layout.grid_row,
                    pixel_rows: CoordinateRange {
                        start: usize_to_i64(layout.pixel_rows.start, "pixel row start")?,
                        end: usize_to_i64(layout.pixel_rows.end, "pixel row end")?,
                    },
                    source_height: image.prepared_rgb.source_height,
                    source_width: image.prepared_rgb.source_width,
                    geometry: layout.geometry,
                },
                cache_key: layout.cache_key,
            });
        }

        Ok(BatchPlan {
            text,
            images,
            processed_images,
            sidecar,
            capacities,
            request_layouts,
            image_layouts,
            contract_id: registry.contract_id().to_owned(),
            profile_fingerprint: self.profile().fingerprint.clone(),
            limits: self.limits,
        })
    }

    /// Executes one owned plan into caller storage. Identity and every
    /// conditional destination are validated before the first write; all code
    /// after that boundary is infallible.
    ///
    /// # Errors
    ///
    /// Returns `profile_mismatch` for a stale/wrong-processor plan or
    /// `destination_too_small` for missing, incompatible, or short storage.
    ///
    /// # Panics
    ///
    /// Only if a private, previously validated plan invariant is corrupted;
    /// safe callers cannot construct or mutate such a plan.
    #[allow(clippy::too_many_lines)]
    pub fn execute_plan_into<'plan, 'destination>(
        &self,
        plan: &'plan BatchPlan,
        destinations: BatchDestinations<'destination>,
    ) -> Result<PreparedBatchView<'plan, 'destination>> {
        self.execute_plan_into_internal(plan, destinations, None)
    }

    /// Observed counterpart to [`Self::execute_plan_into`].
    ///
    /// # Errors
    ///
    /// Returns the same stable errors as [`Self::execute_plan_into`].
    pub fn execute_plan_into_observed<'plan, 'destination>(
        &self,
        plan: &'plan BatchPlan,
        destinations: BatchDestinations<'destination>,
        recorder: &mut ObservationRecorder,
    ) -> Result<PreparedBatchView<'plan, 'destination>> {
        let span = recorder.begin(
            "native.destination.execute",
            ObservationScope::default(),
            total_capacity_bytes(&plan.capacities),
        );
        let result = self.execute_plan_into_internal(plan, destinations, Some(recorder));
        match &result {
            Ok(_) => recorder.finish_success(
                span,
                total_capacity_bytes(&plan.capacities),
                &[plan.text.rows() as u64, plan.text.columns() as u64],
            ),
            Err(error) => {
                recorder.finish_error(span, error);
            }
        }
        result
    }

    fn execute_plan_into_internal<'plan, 'destination>(
        &self,
        plan: &'plan BatchPlan,
        destinations: BatchDestinations<'destination>,
        mut recorder: Option<&mut ObservationRecorder>,
    ) -> Result<PreparedBatchView<'plan, 'destination>> {
        validate_plan_identity(self, plan)?;
        validate_destinations(&plan.capacities, &destinations)?;
        let BatchDestinations {
            input_ids,
            attention_mask,
            mm_token_type_ids,
            mut pixel_values,
            mut image_grid_thw,
            pixel_values_videos: _,
            video_grid_thw: _,
        } = destinations;

        self.text
            .write_batch_plan(&plan.text, input_ids, attention_mask, mm_token_type_ids);
        if let (Some(pixels), Some(grids)) =
            (pixel_values.as_deref_mut(), image_grid_thw.as_deref_mut())
        {
            let patch_width = usize::try_from(self.profile().visual.patch_width)
                .expect("validated patch width must fit usize");
            for (image, layout) in plan.images.iter().zip(&plan.image_layouts) {
                let scope = media_observation_scope(image.location, image.occurrence_index);
                let span = recorder.as_deref_mut().map(|recorder| {
                    recorder.begin(
                        "native.media.normalize_patchify_layout",
                        scope,
                        image.prepared_rgb.rgb.len() as u64,
                    )
                });
                let pixel_start = layout.pixel_rows.start * patch_width;
                let pixel_end = layout.pixel_rows.end * patch_width;
                let grid_start = layout.grid_row * 3;
                execute_image_patchify_plan_into(
                    image.patch_plan,
                    &self.profile().visual,
                    &image.prepared_rgb.rgb,
                    &mut pixels[pixel_start..pixel_end],
                    &mut grids[grid_start..grid_start + 3],
                );
                if let Some(recorder) = recorder.as_deref_mut()
                    && let Some(span) = span
                {
                    recorder.finish_success(
                        span,
                        image.patch_plan.materialized_bytes(),
                        &[
                            image.patch_plan.geometry().patch_rows,
                            self.profile().visual.patch_width,
                        ],
                    );
                }
            }
        }

        let text_elements = plan.capacities.input_ids.elements;
        let arrays = PreparedArrayViews {
            input_ids: MatrixView::from_validated(
                plan.capacities.input_ids.shape[0],
                plan.capacities.input_ids.shape[1],
                &input_ids[..text_elements],
            ),
            attention_mask: MatrixView::from_validated(
                plan.capacities.attention_mask.shape[0],
                plan.capacities.attention_mask.shape[1],
                &attention_mask[..text_elements],
            ),
            mm_token_type_ids: MatrixView::from_validated(
                plan.capacities.mm_token_type_ids.shape[0],
                plan.capacities.mm_token_type_ids.shape[1],
                &mm_token_type_ids[..text_elements],
            ),
            pixel_values: plan.capacities.pixel_values.map(|capacity| {
                MatrixView::from_validated(
                    capacity.shape[0],
                    capacity.shape[1],
                    &pixel_values.expect("validated conditional destination")[..capacity.elements],
                )
            }),
            image_grid_thw: plan.capacities.image_grid_thw.map(|capacity| {
                MatrixView::from_validated(
                    capacity.shape[0],
                    capacity.shape[1],
                    &image_grid_thw.expect("validated conditional destination")
                        [..capacity.elements],
                )
            }),
            pixel_values_videos: None,
            video_grid_thw: None,
        };
        Ok(PreparedBatchView {
            contract_id: &plan.contract_id,
            profile_fingerprint: &plan.profile_fingerprint,
            text: plan.text.requests(),
            arrays,
            sidecar: &plan.sidecar,
            images: &plan.processed_images,
        })
    }

    /// Allocates exact official arrays and executes the same two-phase plan.
    ///
    /// # Errors
    ///
    /// Returns exactly the planning/execution failures documented by
    /// [`Self::plan_batch`] and [`Self::execute_plan_into`].
    pub fn prepare_batch(&self, requests: &[Request<'_>]) -> Result<PreparedImageBatch> {
        self.prepare_batch_internal(requests, None, None)
    }

    /// Allocates and executes one batch with the default bounded observation
    /// capacity. The envelope retains the report even when processing fails.
    #[must_use]
    pub fn prepare_batch_observed(&self, requests: &[Request<'_>]) -> Observed<PreparedImageBatch> {
        self.prepare_batch_observed_with_capacity(requests, DEFAULT_OBSERVATION_EVENT_CAPACITY)
    }

    /// Allocates and executes one batch with an explicit hard event bound.
    #[must_use]
    pub fn prepare_batch_observed_with_capacity(
        &self,
        requests: &[Request<'_>],
        event_capacity: usize,
    ) -> Observed<PreparedImageBatch> {
        let mut recorder = ObservationRecorder::new(event_capacity);
        recorder.calls_mut().native_batch_calls = 1;
        let span = recorder.begin("native.batch.prepare", ObservationScope::default(), 0);
        let result = self.prepare_batch_with_observer(requests, &mut recorder);
        match &result {
            Ok(output) => recorder.finish_success(
                span,
                prepared_batch_output_bytes(output),
                &[requests.len() as u64, output.images.len() as u64],
            ),
            Err(error) => {
                recorder.discard_retained_outputs();
                recorder.release_all_transients();
                recorder.finish_error(span, error);
            }
        }
        Observed {
            result,
            report: recorder.report(),
        }
    }

    /// Populates a caller-owned recorder. This is the low-level surface used
    /// by bindings that allocate destination vectors for zero-copy transfer.
    ///
    /// # Errors
    ///
    /// Returns the same stable errors as [`Self::prepare_batch`].
    pub fn prepare_batch_with_observer(
        &self,
        requests: &[Request<'_>],
        recorder: &mut ObservationRecorder,
    ) -> Result<PreparedImageBatch> {
        self.prepare_batch_internal(requests, Some(recorder), None)
    }

    #[cfg(test)]
    fn prepare_batch_with_observer_and_allocation_failure(
        &self,
        requests: &[Request<'_>],
        recorder: &mut ObservationRecorder,
        fail_before: &'static str,
    ) -> Result<PreparedImageBatch> {
        self.prepare_batch_internal(requests, Some(recorder), Some(fail_before))
    }

    #[allow(clippy::too_many_lines)]
    fn prepare_batch_internal(
        &self,
        requests: &[Request<'_>],
        mut recorder: Option<&mut ObservationRecorder>,
        allocation_failure_before: Option<&'static str>,
    ) -> Result<PreparedImageBatch> {
        let plan = if let Some(recorder) = recorder.as_deref_mut() {
            self.plan_batch_observed(requests, recorder)?
        } else {
            self.plan_batch(requests)?
        };
        let capacities = plan.capacities;
        let allocate_span = recorder.as_deref_mut().map(|recorder| {
            recorder.begin(
                "native.destination.allocate",
                ObservationScope::default(),
                0,
            )
        });
        let allocated = (|| {
            let input_ids = allocate_output_with_failure::<i64>(
                "input_ids",
                capacities.input_ids.elements,
                allocation_failure_before,
            )?;
            record_output_allocation(
                recorder.as_deref_mut(),
                "input_ids",
                capacities.input_ids.bytes,
            );
            let attention_mask = allocate_output_with_failure::<i64>(
                "attention_mask",
                capacities.attention_mask.elements,
                allocation_failure_before,
            )?;
            record_output_allocation(
                recorder.as_deref_mut(),
                "attention_mask",
                capacities.attention_mask.bytes,
            );
            let mm_token_type_ids = allocate_output_with_failure::<i64>(
                "mm_token_type_ids",
                capacities.mm_token_type_ids.elements,
                allocation_failure_before,
            )?;
            record_output_allocation(
                recorder.as_deref_mut(),
                "mm_token_type_ids",
                capacities.mm_token_type_ids.bytes,
            );
            let pixel_values = if let Some(capacity) = capacities.pixel_values {
                let values = allocate_output_with_failure::<f32>(
                    "pixel_values",
                    capacity.elements,
                    allocation_failure_before,
                )?;
                record_output_allocation(recorder.as_deref_mut(), "pixel_values", capacity.bytes);
                Some(values)
            } else {
                None
            };
            let image_grid_thw = if let Some(capacity) = capacities.image_grid_thw {
                let values = allocate_output_with_failure::<i64>(
                    "image_grid_thw",
                    capacity.elements,
                    allocation_failure_before,
                )?;
                record_output_allocation(recorder.as_deref_mut(), "image_grid_thw", capacity.bytes);
                Some(values)
            } else {
                None
            };
            Result::Ok((
                input_ids,
                attention_mask,
                mm_token_type_ids,
                pixel_values,
                image_grid_thw,
            ))
        })();
        let (
            mut input_ids,
            mut attention_mask,
            mut mm_token_type_ids,
            mut pixel_values,
            mut image_grid_thw,
        ) = match allocated {
            Ok(allocated) => {
                if let Some(recorder) = recorder.as_deref_mut()
                    && let Some(span) = allocate_span
                {
                    recorder.finish_success(
                        span,
                        total_capacity_bytes(&capacities),
                        &[total_capacity_bytes(&capacities)],
                    );
                }
                allocated
            }
            Err(error) => {
                if let Some(recorder) = recorder.as_deref_mut() {
                    if let Some(span) = allocate_span {
                        recorder.finish_error(span, &error);
                    }
                    recorder.discard_retained_outputs();
                    plan.drop_observed(recorder);
                } else {
                    drop(plan);
                }
                return Err(error);
            }
        };
        let result = (|| {
            let destinations = BatchDestinations {
                input_ids: &mut input_ids,
                attention_mask: &mut attention_mask,
                mm_token_type_ids: &mut mm_token_type_ids,
                pixel_values: pixel_values.as_deref_mut(),
                image_grid_thw: image_grid_thw.as_deref_mut(),
                pixel_values_videos: None,
                video_grid_thw: None,
            };
            let view = if let Some(recorder) = recorder.as_deref_mut() {
                self.execute_plan_into_observed(&plan, destinations, recorder)?
            } else {
                self.execute_plan_into(&plan, destinations)?
            };
            let PreparedBatchView {
                contract_id,
                profile_fingerprint,
                text,
                arrays: _,
                sidecar,
                images,
            } = view;
            let contract_id = contract_id.to_owned();
            let profile_fingerprint = profile_fingerprint.to_owned();
            let text = text.to_vec();
            let sidecar = sidecar.clone();
            let images = images.to_vec();
            let arrays = PreparedArrays {
                input_ids: Matrix::new(
                    capacities.input_ids.shape[0],
                    capacities.input_ids.shape[1],
                    input_ids,
                )?,
                attention_mask: Matrix::new(
                    capacities.attention_mask.shape[0],
                    capacities.attention_mask.shape[1],
                    attention_mask,
                )?,
                mm_token_type_ids: Matrix::new(
                    capacities.mm_token_type_ids.shape[0],
                    capacities.mm_token_type_ids.shape[1],
                    mm_token_type_ids,
                )?,
                pixel_values: pixel_values
                    .zip(capacities.pixel_values)
                    .map(|(values, capacity)| {
                        Matrix::new(capacity.shape[0], capacity.shape[1], values)
                    })
                    .transpose()?,
                image_grid_thw: image_grid_thw
                    .zip(capacities.image_grid_thw)
                    .map(|(values, capacity)| {
                        Matrix::new(capacity.shape[0], capacity.shape[1], values)
                    })
                    .transpose()?,
                pixel_values_videos: None,
                video_grid_thw: None,
            };
            Ok(PreparedImageBatch {
                contract_id,
                profile_fingerprint,
                text,
                batch: PreparedBatch::new(arrays, sidecar)?,
                images,
            })
        })();
        if let Some(recorder) = recorder {
            if result.is_err() {
                recorder.discard_retained_outputs();
            }
            plan.drop_observed(recorder);
        } else {
            drop(plan);
        }
        result
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

fn validate_plan_identity(processor: &QwenImageProcessor, plan: &BatchPlan) -> Result<()> {
    let registry = ProfileRegistry::bundled()?;
    if plan.contract_id != registry.contract_id()
        || plan.profile_fingerprint != processor.profile().fingerprint
        || plan.limits != processor.limits
    {
        return Err(QwenError::new(
            ErrorCategory::ProfileMismatch,
            "batch plan identity does not match the executing processor",
        )
        .with_context("plan_contract_id", plan.contract_id.clone())
        .with_context("processor_contract_id", registry.contract_id())
        .with_context("plan_profile_fingerprint", plan.profile_fingerprint.clone())
        .with_context(
            "processor_profile_fingerprint",
            processor.profile().fingerprint.clone(),
        ));
    }
    Ok(())
}

fn validate_destinations(
    capacities: &BatchCapacities,
    destinations: &BatchDestinations<'_>,
) -> Result<()> {
    validate_destination_len(
        "input_ids",
        destinations.input_ids.len(),
        capacities.input_ids.elements,
    )?;
    validate_destination_len(
        "attention_mask",
        destinations.attention_mask.len(),
        capacities.attention_mask.elements,
    )?;
    validate_destination_len(
        "mm_token_type_ids",
        destinations.mm_token_type_ids.len(),
        capacities.mm_token_type_ids.elements,
    )?;
    validate_optional_destination(
        "pixel_values",
        destinations.pixel_values.as_deref().map(<[f32]>::len),
        capacities.pixel_values.map(|capacity| capacity.elements),
    )?;
    validate_optional_destination(
        "image_grid_thw",
        destinations.image_grid_thw.as_deref().map(<[i64]>::len),
        capacities.image_grid_thw.map(|capacity| capacity.elements),
    )?;
    validate_optional_destination(
        "pixel_values_videos",
        destinations
            .pixel_values_videos
            .as_deref()
            .map(<[f32]>::len),
        capacities
            .pixel_values_videos
            .map(|capacity| capacity.elements),
    )?;
    validate_optional_destination(
        "video_grid_thw",
        destinations.video_grid_thw.as_deref().map(<[i64]>::len),
        capacities.video_grid_thw.map(|capacity| capacity.elements),
    )
}

fn validate_destination_len(output: &'static str, actual: usize, required: usize) -> Result<()> {
    if actual < required {
        return Err(
            destination_error("destination is smaller than its planned capacity")
                .with_context("output", output)
                .with_context("required_elements", required)
                .with_context("actual_elements", actual),
        );
    }
    Ok(())
}

fn validate_optional_destination(
    output: &'static str,
    actual: Option<usize>,
    required: Option<usize>,
) -> Result<()> {
    match (actual, required) {
        (Some(actual), Some(required)) => validate_destination_len(output, actual, required),
        (None, None) => Ok(()),
        (None, Some(required)) => Err(destination_error("required destination is missing")
            .with_context("output", output)
            .with_context("required_elements", required)),
        (Some(actual), None) => Err(destination_error(
            "destination is present for an unplanned conditional output",
        )
        .with_context("output", output)
        .with_context("actual_elements", actual)),
    }
}

fn array_capacity(rows: usize, columns: usize, element_bytes: usize) -> Result<ArrayCapacity> {
    let elements = rows.checked_mul(columns).ok_or_else(|| {
        overflow("array element capacity overflowed")
            .with_context("rows", rows)
            .with_context("columns", columns)
    })?;
    let bytes = checked_capacity_bytes(
        u64::try_from(rows).map_err(|_| overflow("array rows do not fit u64"))?,
        u64::try_from(columns).map_err(|_| overflow("array columns do not fit u64"))?,
        u64::try_from(element_bytes)
            .map_err(|_| overflow("array element size does not fit u64"))?,
    )?;
    Ok(ArrayCapacity {
        shape: [rows, columns],
        elements,
        bytes,
    })
}

fn total_capacity_bytes(capacities: &BatchCapacities) -> u64 {
    capacities
        .input_ids
        .bytes
        .saturating_add(capacities.attention_mask.bytes)
        .saturating_add(capacities.mm_token_type_ids.bytes)
        .saturating_add(capacities.pixel_values.map_or(0, |capacity| capacity.bytes))
        .saturating_add(
            capacities
                .image_grid_thw
                .map_or(0, |capacity| capacity.bytes),
        )
        .saturating_add(
            capacities
                .pixel_values_videos
                .map_or(0, |capacity| capacity.bytes),
        )
        .saturating_add(
            capacities
                .video_grid_thw
                .map_or(0, |capacity| capacity.bytes),
        )
}

fn image_input_bytes(input: ImageInput<'_>) -> u64 {
    let bytes = match input {
        ImageInput::Encoded { data, .. } => data.len(),
        ImageInput::Rgb8(raw) => raw.data.len(),
    };
    u64::try_from(bytes).unwrap_or(u64::MAX)
}

const fn media_observation_scope(
    location: OccurrenceLocation,
    occurrence_index: usize,
) -> ObservationScope {
    ObservationScope::media_at(
        location.request_index,
        location.message_index,
        location.content_item_index,
        occurrence_index,
        location.input_index,
    )
}

fn release_planned_images(
    images: &[BatchPlannedImageOccurrence],
    recorder: &mut ObservationRecorder,
) {
    for image in images {
        recorder.release_transient(
            "prepared_rgb",
            media_observation_scope(image.location, image.occurrence_index),
            u8_vector_capacity_bytes(&image.prepared_rgb.rgb),
        );
    }
}

fn record_output_allocation(
    recorder: Option<&mut ObservationRecorder>,
    name: &'static str,
    bytes: u64,
) {
    if let Some(recorder) = recorder {
        recorder.record_allocation(
            name,
            BufferClass::RetainedOutput,
            ObservationScope::default(),
            bytes,
        );
    }
}

fn prepared_batch_output_bytes(output: &PreparedImageBatch) -> u64 {
    let arrays = output.batch.arrays();
    matrix_storage_bytes(&arrays.input_ids)
        .saturating_add(matrix_storage_bytes(&arrays.attention_mask))
        .saturating_add(matrix_storage_bytes(&arrays.mm_token_type_ids))
        .saturating_add(arrays.pixel_values.as_ref().map_or(0, matrix_storage_bytes))
        .saturating_add(
            arrays
                .image_grid_thw
                .as_ref()
                .map_or(0, matrix_storage_bytes),
        )
        .saturating_add(
            arrays
                .pixel_values_videos
                .as_ref()
                .map_or(0, matrix_storage_bytes),
        )
        .saturating_add(
            arrays
                .video_grid_thw
                .as_ref()
                .map_or(0, matrix_storage_bytes),
        )
}

fn matrix_storage_bytes<T>(matrix: &Matrix<T>) -> u64 {
    let elements = u64::try_from(matrix.as_slice().len()).unwrap_or(u64::MAX);
    let element_bytes = u64::try_from(mem::size_of::<T>()).unwrap_or(u64::MAX);
    elements.saturating_mul(element_bytes)
}

#[allow(clippy::ptr_arg)] // Vec capacity is required for allocation accounting.
fn u8_vector_capacity_bytes(values: &Vec<u8>) -> u64 {
    u64::try_from(values.capacity()).unwrap_or(u64::MAX)
}

fn allocate_output<T: Default + Clone>(output: &'static str, elements: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(elements).map_err(|error| {
        QwenError::new(
            ErrorCategory::ResourceLimit,
            "unable to reserve planned output allocation",
        )
        .with_context("output", output)
        .with_context("elements", elements)
        .with_context("detail", error.to_string())
    })?;
    values.resize(elements, T::default());
    Ok(values)
}

fn allocate_output_with_failure<T: Default + Clone>(
    output: &'static str,
    elements: usize,
    fail_before: Option<&'static str>,
) -> Result<Vec<T>> {
    if fail_before == Some(output) {
        return Err(QwenError::new(
            ErrorCategory::ResourceLimit,
            "unable to reserve planned output allocation",
        )
        .with_context("output", output)
        .with_context("elements", elements)
        .with_context("detail", "injected allocation failure"));
    }
    allocate_output(output, elements)
}

fn image_cache_key(
    profile: &Profile,
    input: crate::request::ImageInput<'_>,
    options: crate::request::ImageOptions,
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"qwen-mm-compat-v1\0image\0");
    update_digest_bytes(&mut digest, profile.fingerprint.as_bytes());
    match input {
        crate::request::ImageInput::Encoded { data, format } => {
            digest.update(b"encoded\0");
            digest.update([match format {
                crate::request::ImageFormat::Jpeg => 0,
                crate::request::ImageFormat::Png => 1,
                crate::request::ImageFormat::WebP => 2,
            }]);
            update_digest_bytes(&mut digest, data);
        }
        crate::request::ImageInput::Rgb8(rgb) => {
            digest.update(b"rgb8\0");
            digest.update(canonical_usize(rgb.height, "RGB height").to_le_bytes());
            digest.update(canonical_usize(rgb.width, "RGB width").to_le_bytes());
            let row_bytes = rgb
                .width
                .checked_mul(3)
                .expect("validated RGB logical row width must fit usize");
            let logical_bytes = rgb
                .height
                .checked_mul(row_bytes)
                .expect("validated RGB logical byte count must fit usize");
            digest.update(canonical_usize(logical_bytes, "RGB logical byte length").to_le_bytes());
            for row in 0..rgb.height {
                let start = row
                    .checked_mul(rgb.row_stride)
                    .expect("validated RGB row offset must fit usize");
                digest.update(&rgb.data[start..start + row_bytes]);
            }
        }
    }
    for value in [
        options.min_pixels,
        options.max_pixels,
        options.resized_height,
        options.resized_width,
    ] {
        match value {
            Some(value) => {
                digest.update([1]);
                digest.update(value.to_le_bytes());
            }
            None => digest.update([0]),
        }
    }
    digest.finalize().into()
}

fn update_digest_bytes(digest: &mut Sha256, bytes: &[u8]) {
    digest.update(canonical_usize(bytes.len(), "digest byte length").to_le_bytes());
    digest.update(bytes);
}

fn canonical_usize(value: usize, label: &'static str) -> u64 {
    u64::try_from(value).unwrap_or_else(|_| panic!("{label} must fit the stable u64 identity"))
}

fn hex_digest(bytes: [u8; 32]) -> String {
    use std::fmt::Write as _;

    let mut output = String::with_capacity(64);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn usize_to_i64(value: usize, label: &'static str) -> Result<i64> {
    i64::try_from(value).map_err(|_| overflow(label))
}

fn destination_error(message: impl Into<String>) -> QwenError {
    QwenError::new(ErrorCategory::DestinationTooSmall, message)
}

fn unsupported_batch_video() -> QwenError {
    QwenError::new(
        ErrorCategory::UnsupportedMedia,
        "video processing is outside the Phase C image batch path",
    )
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

    use image::{
        ExtendedColorType, ImageEncoder,
        codecs::{png::PngEncoder, webp::WebPEncoder},
    };

    use super::{
        BatchDestinations, BatchPlan, QwenImageProcessor, hex_digest, image_cache_key,
        prepared_batch_output_bytes,
    };
    use crate::{
        BufferClass, ContentItem, CoordinateRange, ErrorCategory, ImageFormat, ImageInput,
        ImageOptions, ImageRef, LimitOverrides, Message, MessageContent, ObservationRecorder,
        ObservationReport, ProfileAlias, ProfileRegistry, Request, RequestOptions, ResourceLimits,
        Rgb8, Role, StageOutcome, VideoInput, VideoRef,
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

    fn assert_failed_envelope_containment(report: &ObservationReport, envelope_name: &str) {
        let envelopes = report
            .spans
            .iter()
            .filter(|span| span.name == envelope_name)
            .collect::<Vec<_>>();
        assert_eq!(envelopes.len(), 1);
        let envelope = envelopes[0];
        let envelope_end = envelope.started_ns + envelope.duration_ns;
        assert_eq!(envelope.outcome, StageOutcome::Error);
        assert!(report.spans.iter().all(|span| {
            envelope.started_ns <= span.started_ns
                && span.started_ns + span.duration_ns <= envelope_end
        }));
        assert!(report.buffers.iter().all(|buffer| {
            if buffer.class == BufferClass::RetainedOutput {
                return false;
            }
            buffer.allocated_at_ns >= envelope.started_ns
                && buffer.allocated_at_ns <= envelope_end
                && buffer.released_at_ns.is_some_and(|released| {
                    released >= buffer.allocated_at_ns && released <= envelope_end
                })
        }));
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

    fn encode_rgb(format: ImageFormat, value: u8) -> Vec<u8> {
        let rgb = vec![value; 64 * 64 * 3];
        let mut encoded = Vec::new();
        match format {
            ImageFormat::Png => PngEncoder::new(&mut encoded)
                .write_image(&rgb, 64, 64, ExtendedColorType::Rgb8)
                .expect("encode PNG"),
            ImageFormat::WebP => WebPEncoder::new_lossless(&mut encoded)
                .write_image(&rgb, 64, 64, ExtendedColorType::Rgb8)
                .expect("encode WebP"),
            ImageFormat::Jpeg => panic!("test helper does not encode JPEG"),
        }
        encoded
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
    fn observed_text_batch_is_identical_bounded_and_request_correlated() {
        let messages = [message(
            Role::User,
            MessageContent::Text("hello observability"),
        )];
        let request = Request {
            messages: &messages,
            images: &[],
            videos: &[],
            options: RequestOptions::default(),
        };
        let processor = processor(ProfileAlias::Qwen3Vl8b, ResourceLimits::default());
        let expected = processor.prepare_batch(&[request]).expect("plain batch");
        let observed = processor.prepare_batch_observed_with_capacity(&[request], 64);
        let output = observed.result.expect("observed batch");
        assert_eq!(output, expected);
        let report = observed.report;
        assert_eq!(report.outcome, StageOutcome::Success);
        assert_eq!(report.error_category, None);
        assert_eq!(report.dropped_events, 0);
        assert_eq!(report.calls.native_batch_calls, 1);
        assert_eq!(report.calls.native_visual_calls, 0);
        assert_eq!(report.calls.python_callbacks, 0);
        assert_eq!(report.calls.hugging_face_calls, 0);
        assert_eq!(report.calls.qwen_vl_utils_calls, 0);
        assert_eq!(report.calls.pillow_calls, 0);
        assert_eq!(report.calls.torchvision_calls, 0);
        assert_eq!(report.allocations.allocation_count, 3);
        assert_eq!(report.allocations.transient_live_bytes, 0);
        assert_eq!(report.allocations.peak_transient_live_bytes, 0);
        assert_eq!(
            report.allocations.retained_final_output_bytes,
            prepared_batch_output_bytes(&output)
        );
        assert!(
            report
                .buffers
                .iter()
                .all(|buffer| buffer.class == BufferClass::RetainedOutput)
        );
        let names = report
            .spans
            .iter()
            .map(|span| span.name.as_str())
            .collect::<Vec<_>>();
        for required in [
            "native.batch.prepare",
            "native.batch.plan",
            "native.request.validate",
            "native.chat.render",
            "native.chat.tokenize",
            "native.destination.allocate",
            "native.destination.execute",
        ] {
            assert!(names.contains(&required), "missing {required}: {names:?}");
        }
        assert!(
            report
                .spans
                .windows(2)
                .all(|pair| pair[0].sequence < pair[1].sequence)
        );
        for span in &report.spans {
            assert!(span.exclusive_duration_ns <= span.duration_ns);
            if let Some(parent) = span.parent_sequence {
                assert!(parent < span.sequence);
                assert!(
                    report
                        .spans
                        .iter()
                        .any(|candidate| candidate.sequence == parent)
                );
            }
        }
        for span in report.spans.iter().filter(|span| {
            matches!(
                span.name.as_str(),
                "native.request.validate" | "native.chat.render" | "native.chat.tokenize"
            )
        }) {
            assert_eq!(span.scope.request_index, Some(0));
        }
    }

    #[test]
    fn observed_qwen35_text_batch_matches_unobserved_output() {
        let messages = [message(
            Role::User,
            MessageContent::Text("qwen3.5 observability parity"),
        )];
        let request = Request {
            messages: &messages,
            images: &[],
            videos: &[],
            options: RequestOptions::default(),
        };
        let processor = processor(ProfileAlias::Qwen35_9b, ResourceLimits::default());
        let expected = processor.prepare_batch(&[request]).expect("plain batch");
        let observed = processor.prepare_batch_observed_with_capacity(&[request], 64);
        let output = observed.result.expect("observed batch");
        assert_eq!(output, expected);
        assert_eq!(observed.report.outcome, StageOutcome::Success);
        assert_eq!(observed.report.error_category, None);
        assert_eq!(observed.report.calls.native_batch_calls, 1);
        assert_eq!(observed.report.calls.native_visual_calls, 0);
        assert_eq!(observed.report.allocations.transient_live_bytes, 0);
        assert_eq!(
            observed.report.allocations.retained_final_output_bytes,
            prepared_batch_output_bytes(&output)
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn observed_mixed_repeated_media_reconciles_buffers_copies_and_scopes() {
        let raw = vec![23_u8; 64 * 64 * 3];
        let encoded = encode_rgb(ImageFormat::Png, 47);
        let request_zero_images = [ImageInput::Rgb8(Rgb8 {
            data: &raw,
            height: 64,
            width: 64,
            row_stride: 64 * 3,
        })];
        let request_zero_items = [
            ContentItem::Image(ImageRef::default()),
            ContentItem::Text("between"),
            ContentItem::Image(ImageRef::default()),
        ];
        let request_zero_messages = [message(
            Role::User,
            MessageContent::Items(&request_zero_items),
        )];
        let request_one_images = [ImageInput::Encoded {
            data: &encoded,
            format: ImageFormat::Png,
        }];
        let request_one_items = [ContentItem::Image(ImageRef::default())];
        let request_one_messages = [message(
            Role::User,
            MessageContent::Items(&request_one_items),
        )];
        let requests = [
            Request {
                messages: &request_zero_messages,
                images: &request_zero_images,
                videos: &[],
                options: RequestOptions::default(),
            },
            Request {
                messages: &request_one_messages,
                images: &request_one_images,
                videos: &[],
                options: RequestOptions::default(),
            },
        ];
        let processor = processor(ProfileAlias::Qwen3Vl8b, ResourceLimits::default());
        let expected = processor.prepare_batch(&requests).expect("plain batch");
        let observed = processor.prepare_batch_observed_with_capacity(&requests, 256);
        let output = observed.result.expect("observed batch");
        assert_eq!(output, expected);
        let report = observed.report;
        assert_eq!(report.calls.native_batch_calls, 1);
        assert_eq!(report.calls.native_visual_calls, 3);
        assert_eq!(report.allocations.copy_count, 6);
        assert_eq!(
            report.allocations.copied_bytes,
            u64::try_from((raw.len() * 2 + 64 * 64 * 3) * 2).expect("copy bytes")
        );
        assert_eq!(report.copies.len(), 6);
        assert_eq!(
            report.copies.iter().map(|copy| copy.bytes).sum::<u64>(),
            report.allocations.copied_bytes
        );
        assert_eq!(
            report
                .copies
                .iter()
                .filter(|copy| copy.name == "resize.packed_source")
                .count(),
            3
        );
        assert_eq!(
            report
                .copies
                .iter()
                .filter(|copy| copy.name == "resize.noop.source_copy")
                .count(),
            3
        );
        assert_eq!(
            report
                .copies
                .iter()
                .map(|copy| copy.scope.media_index)
                .collect::<Vec<_>>(),
            vec![Some(0), Some(0), Some(1), Some(1), Some(2), Some(2)]
        );
        assert_eq!(report.allocations.transient_live_bytes, 0);
        assert!(report.allocations.peak_transient_live_bytes > 0);
        assert_eq!(
            report.allocations.retained_final_output_bytes,
            prepared_batch_output_bytes(&output)
        );
        assert_eq!(
            report.allocations.allocated_bytes,
            report
                .buffers
                .iter()
                .map(|buffer| buffer.bytes)
                .sum::<u64>()
        );
        assert!(report.buffers.iter().all(|buffer| {
            buffer.class != BufferClass::Transient || buffer.released_at_ns.is_some()
        }));
        let fused = report
            .spans
            .iter()
            .filter(|span| span.name == "native.media.normalize_patchify_layout")
            .collect::<Vec<_>>();
        assert_eq!(fused.len(), 3);
        assert_eq!(
            fused
                .iter()
                .map(|span| (
                    span.scope.request_index,
                    span.scope.message_index,
                    span.scope.content_item_index,
                    span.scope.media_index,
                    span.scope.input_index,
                ))
                .collect::<Vec<_>>(),
            vec![
                (Some(0), Some(0), Some(0), Some(0), Some(0)),
                (Some(0), Some(0), Some(2), Some(1), Some(0)),
                (Some(1), Some(0), Some(0), Some(2), Some(0)),
            ]
        );
    }

    #[test]
    fn observed_invalid_request_failure_is_contained() {
        let processor = processor(ProfileAlias::Qwen3Vl8b, ResourceLimits::default());
        let invalid = Request {
            messages: &[],
            images: &[],
            videos: &[],
            options: RequestOptions::default(),
        };
        let observed = processor.prepare_batch_observed(&[invalid]);
        assert_eq!(
            observed.result.expect_err("invalid request").category(),
            ErrorCategory::InvalidRequest
        );
        assert_eq!(observed.report.outcome, StageOutcome::Error);
        assert_eq!(
            observed.report.error_category.as_deref(),
            Some("invalid_request")
        );
        assert_eq!(observed.report.allocations.transient_live_bytes, 0);
        assert_eq!(observed.report.allocations.retained_final_output_bytes, 0);
        assert!(observed.report.spans.iter().any(|span| {
            span.name == "native.request.validate" && span.outcome == StageOutcome::Error
        }));
        assert_failed_envelope_containment(&observed.report, "native.batch.prepare");
    }

    #[test]
    fn observed_media_failure_is_contained() {
        let processor = processor(ProfileAlias::Qwen3Vl8b, ResourceLimits::default());
        let corrupt = png_header(64, 64);
        let corrupt_images = [ImageInput::Encoded {
            data: &corrupt,
            format: ImageFormat::Png,
        }];
        let corrupt_items = [ContentItem::Image(ImageRef::default())];
        let corrupt_messages = [message(Role::User, MessageContent::Items(&corrupt_items))];
        let corrupt_request = Request {
            messages: &corrupt_messages,
            images: &corrupt_images,
            videos: &[],
            options: RequestOptions::default(),
        };
        let failed_media = processor.prepare_batch_observed(&[corrupt_request]);
        assert_eq!(
            failed_media.result.expect_err("truncated PNG").category(),
            ErrorCategory::MediaDecode
        );
        assert_eq!(failed_media.report.allocations.transient_live_bytes, 0);
        assert_eq!(
            failed_media.report.allocations.retained_final_output_bytes,
            0
        );
        assert_eq!(failed_media.report.calls.native_visual_calls, 1);
        assert!(failed_media.report.spans.iter().any(|span| {
            span.name == "native.media.decode_color"
                && span.outcome == StageOutcome::Error
                && span.scope.request_index == Some(0)
                && span.scope.message_index == Some(0)
                && span.scope.content_item_index == Some(0)
                && span.scope.media_index == Some(0)
        }));
        assert_failed_envelope_containment(&failed_media.report, "native.batch.prepare");
    }

    #[test]
    fn observed_plan_lifetimes_survive_destination_failure() {
        let processor = processor(ProfileAlias::Qwen3Vl8b, ResourceLimits::default());
        let raw = vec![9_u8; 64 * 64 * 3];
        let images = [ImageInput::Rgb8(Rgb8 {
            data: &raw,
            height: 64,
            width: 64,
            row_stride: 64 * 3,
        })];
        let items = [ContentItem::Image(ImageRef::default())];
        let messages = [message(Role::User, MessageContent::Items(&items))];
        let request = Request {
            messages: &messages,
            images: &images,
            videos: &[],
            options: RequestOptions::default(),
        };
        let mut recorder = ObservationRecorder::new(128);
        let plan = processor
            .plan_batch_observed(&[request], &mut recorder)
            .expect("plan");
        let capacities = *plan.capacities();
        assert!(recorder.report().allocations.transient_live_bytes > 0);
        let mut input_ids = vec![0; capacities.input_ids.elements - 1];
        let mut attention_mask = vec![0; capacities.attention_mask.elements];
        let mut mm_token_type_ids = vec![0; capacities.mm_token_type_ids.elements];
        let mut pixels = vec![0.0; capacities.pixel_values.expect("pixels").elements];
        let mut grids = vec![0; capacities.image_grid_thw.expect("grid").elements];
        let error = processor
            .execute_plan_into_observed(
                &plan,
                BatchDestinations {
                    input_ids: &mut input_ids,
                    attention_mask: &mut attention_mask,
                    mm_token_type_ids: &mut mm_token_type_ids,
                    pixel_values: Some(&mut pixels),
                    image_grid_thw: Some(&mut grids),
                    pixel_values_videos: None,
                    video_grid_thw: None,
                },
                &mut recorder,
            )
            .expect_err("short destination");
        assert_eq!(error.category(), ErrorCategory::DestinationTooSmall);
        assert!(recorder.report().allocations.transient_live_bytes > 0);
        plan.drop_observed(&mut recorder);
        assert_eq!(recorder.report().allocations.transient_live_bytes, 0);
    }

    #[test]
    fn observed_global_media_precedence_is_the_top_level_error() {
        let bad_aspect = png_header(12_864, 64);
        let corrupt = b"not a png";
        let images = [
            ImageInput::Encoded {
                data: &bad_aspect,
                format: ImageFormat::Png,
            },
            ImageInput::Encoded {
                data: corrupt,
                format: ImageFormat::Png,
            },
        ];
        let items = [
            ContentItem::Image(ImageRef::default()),
            ContentItem::Image(ImageRef {
                input_index: 1,
                options: ImageOptions::default(),
            }),
        ];
        let messages = [message(Role::User, MessageContent::Items(&items))];
        let request = Request {
            messages: &messages,
            images: &images,
            videos: &[],
            options: RequestOptions::default(),
        };
        let processor = processor(ProfileAlias::Qwen3Vl8b, ResourceLimits::default());
        let observed = processor.prepare_batch_observed(&[request]);
        assert_eq!(
            observed
                .result
                .expect_err("global media failure")
                .category(),
            ErrorCategory::MediaDecode
        );
        assert_eq!(
            observed.report.error_category.as_deref(),
            Some("media_decode")
        );
        assert_eq!(observed.report.calls.native_visual_calls, 2);
        let local_errors = observed
            .report
            .spans
            .iter()
            .filter(|span| span.name == "native.media.plan")
            .map(|span| span.error_category.as_deref())
            .collect::<Vec<_>>();
        assert_eq!(
            local_errors,
            vec![Some("media_geometry"), Some("media_decode")]
        );
        assert!(observed.report.spans.iter().any(|span| {
            span.name == "native.batch.plan"
                && span.error_category.as_deref() == Some("media_decode")
        }));
        assert!(observed.report.spans.iter().any(|span| {
            span.name == "native.media.decode_color"
                && span.scope.media_index == Some(1)
                && span.outcome == StageOutcome::Error
                && span.error_category.as_deref() == Some("media_decode")
        }));
    }

    #[test]
    fn observed_plan_error_releases_media_prepared_before_later_decode_failure() {
        let valid = encode_rgb(ImageFormat::Png, 17);
        let corrupt = png_header(64, 64);
        let images = [
            ImageInput::Encoded {
                data: &valid,
                format: ImageFormat::Png,
            },
            ImageInput::Encoded {
                data: &corrupt,
                format: ImageFormat::Png,
            },
        ];
        let items = [
            ContentItem::Image(ImageRef::default()),
            ContentItem::Image(ImageRef {
                input_index: 1,
                options: ImageOptions::default(),
            }),
        ];
        let messages = [message(Role::User, MessageContent::Items(&items))];
        let request = Request {
            messages: &messages,
            images: &images,
            videos: &[],
            options: RequestOptions::default(),
        };
        let processor = processor(ProfileAlias::Qwen3Vl8b, ResourceLimits::default());
        let mut recorder = ObservationRecorder::new(128);
        let error = processor
            .plan_batch_observed(&[request], &mut recorder)
            .expect_err("second image decode");
        assert_eq!(error.category(), ErrorCategory::MediaDecode);
        let report = recorder.report();
        assert_eq!(report.calls.native_visual_calls, 2);
        assert_eq!(report.allocations.transient_live_bytes, 0);
        assert!(report.buffers.iter().any(|buffer| {
            buffer.name == "prepared_rgb"
                && buffer.scope.media_index == Some(0)
                && buffer.released_at_ns.is_some()
        }));
    }

    #[test]
    fn observed_partial_destination_failure_discards_outputs_and_plan_storage() {
        let raw = vec![41_u8; 64 * 64 * 3];
        let images = [ImageInput::Rgb8(Rgb8 {
            data: &raw,
            height: 64,
            width: 64,
            row_stride: 64 * 3,
        })];
        let items = [ContentItem::Image(ImageRef::default())];
        let messages = [message(Role::User, MessageContent::Items(&items))];
        let request = Request {
            messages: &messages,
            images: &images,
            videos: &[],
            options: RequestOptions::default(),
        };
        let processor = processor(ProfileAlias::Qwen3Vl8b, ResourceLimits::default());
        let mut recorder = ObservationRecorder::new(128);
        let error = processor
            .prepare_batch_with_observer_and_allocation_failure(
                &[request],
                &mut recorder,
                "attention_mask",
            )
            .expect_err("injected second destination allocation");
        assert_eq!(error.category(), ErrorCategory::ResourceLimit);
        let report = recorder.report();
        assert_eq!(report.error_category.as_deref(), Some("resource_limit"));
        assert_eq!(report.allocations.transient_live_bytes, 0);
        assert_eq!(report.allocations.retained_final_output_bytes, 0);
        assert!(report.buffers.iter().any(|buffer| {
            buffer.name == "input_ids"
                && buffer.class == BufferClass::DiscardedOutput
                && buffer.released_at_ns.is_some()
        }));
        assert!(
            report
                .buffers
                .iter()
                .any(|buffer| { buffer.name == "prepared_rgb" && buffer.released_at_ns.is_some() })
        );
        assert!(report.spans.iter().any(|span| {
            span.name == "native.destination.allocate"
                && span.outcome == StageOutcome::Error
                && span.error_category.as_deref() == Some("resource_limit")
        }));
    }

    #[test]
    fn cache_identity_uses_a_platform_stable_encoding() {
        let registry = ProfileRegistry::bundled().expect("profiles");
        let data = [1_u8, 2, 3, 4, 5, 6];
        let key = image_cache_key(
            registry.get(ProfileAlias::Qwen3Vl8b),
            ImageInput::Rgb8(Rgb8 {
                data: &data,
                height: 1,
                width: 2,
                row_stride: 6,
            }),
            ImageOptions::default(),
        );
        assert_eq!(
            hex_digest(key),
            "1f61f85695031e6dbf03a8bf4263d2f08ff6adeb676a7e81c301c510657262b5"
        );
        let padded = [1_u8, 2, 3, 4, 5, 6, 99, 98, 97];
        assert_eq!(
            key,
            image_cache_key(
                registry.get(ProfileAlias::Qwen3Vl8b),
                ImageInput::Rgb8(Rgb8 {
                    data: &padded,
                    height: 1,
                    width: 2,
                    row_stride: 9,
                }),
                ImageOptions::default(),
            ),
            "row padding and trailing storage do not affect prepared pixels"
        );
        assert_ne!(
            key,
            image_cache_key(
                registry.get(ProfileAlias::Qwen35_9b),
                ImageInput::Rgb8(Rgb8 {
                    data: &data,
                    height: 1,
                    width: 2,
                    row_stride: 6,
                }),
                ImageOptions::default(),
            )
        );
    }

    #[test]
    #[ignore = "requires hash-pinned local model snapshots under reference/.cache"]
    fn malformed_raw_rgb_returns_media_geometry_before_cache_hashing() {
        let processor = processor(ProfileAlias::Qwen3Vl8b, ResourceLimits::default());
        let short = [1_u8];
        let narrow = vec![2_u8; 64 * 64 * 3];
        let cases = [
            Rgb8 {
                data: &short,
                height: 64,
                width: 64,
                row_stride: 64 * 3,
            },
            Rgb8 {
                data: &narrow,
                height: 64,
                width: 64,
                row_stride: 64 * 3 - 1,
            },
        ];
        let items = [ContentItem::Image(ImageRef::default())];
        let messages = [message(Role::User, MessageContent::Items(&items))];

        for rgb in cases {
            let images = [ImageInput::Rgb8(rgb)];
            let requests = [Request {
                messages: &messages,
                images: &images,
                videos: &[],
                options: RequestOptions::default(),
            }];
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                processor.plan_batch(&requests)
            }));
            let error = outcome
                .expect("malformed raw RGB must never panic")
                .expect_err("malformed raw RGB must fail planning");
            assert_eq!(error.category(), ErrorCategory::MediaGeometry);
        }
    }

    fn assert_short_destination_is_atomic(
        processor: &QwenImageProcessor,
        plan: &BatchPlan,
        short: &str,
    ) {
        let capacities = plan.capacities();
        let mut input_ids = vec![-11_i64; capacities.input_ids.elements];
        let mut attention_mask = vec![-12_i64; capacities.attention_mask.elements];
        let mut mm_token_type_ids = vec![-13_i64; capacities.mm_token_type_ids.elements];
        let mut pixel_values =
            vec![-14.0_f32; capacities.pixel_values.expect("image capacity").elements];
        let mut image_grid_thw =
            vec![-15_i64; capacities.image_grid_thw.expect("grid capacity").elements];
        match short {
            "input_ids" => {
                input_ids.pop();
            }
            "attention_mask" => {
                attention_mask.pop();
            }
            "mm_token_type_ids" => {
                mm_token_type_ids.pop();
            }
            "pixel_values" => {
                pixel_values.pop();
            }
            "image_grid_thw" => {
                image_grid_thw.pop();
            }
            _ => panic!("unknown destination"),
        }
        let error = processor
            .execute_plan_into(
                plan,
                BatchDestinations {
                    input_ids: &mut input_ids,
                    attention_mask: &mut attention_mask,
                    mm_token_type_ids: &mut mm_token_type_ids,
                    pixel_values: Some(&mut pixel_values),
                    image_grid_thw: Some(&mut image_grid_thw),
                    pixel_values_videos: None,
                    video_grid_thw: None,
                },
            )
            .expect_err("short destination must fail");
        assert_eq!(error.category(), ErrorCategory::DestinationTooSmall);
        assert!(input_ids.iter().all(|&value| value == -11));
        assert!(attention_mask.iter().all(|&value| value == -12));
        assert!(mm_token_type_ids.iter().all(|&value| value == -13));
        assert!(
            pixel_values
                .iter()
                .all(|value| value.to_bits() == (-14.0_f32).to_bits())
        );
        assert!(image_grid_thw.iter().all(|&value| value == -15));
    }

    #[test]
    #[ignore = "requires hash-pinned local model snapshots under reference/.cache"]
    #[allow(clippy::too_many_lines)]
    fn batch_plan_reports_exact_layouts_and_allocating_wrapper_matches_view() {
        let processor = processor(ProfileAlias::Qwen3Vl8b, ResourceLimits::default());
        let rgb = vec![31_u8; 64 * 64 * 3];
        let images = [ImageInput::Rgb8(Rgb8 {
            data: &rgb,
            height: 64,
            width: 64,
            row_stride: 64 * 3,
        })];
        let text_messages = [message(Role::User, MessageContent::Text("short"))];
        let image_items = [
            ContentItem::Image(ImageRef::default()),
            ContentItem::Text(" then "),
            ContentItem::Image(ImageRef::default()),
            ContentItem::Text(" and explicitly resized "),
            ContentItem::Image(ImageRef {
                input_index: 0,
                options: ImageOptions {
                    resized_height: Some(64),
                    resized_width: Some(64),
                    ..ImageOptions::default()
                },
            }),
        ];
        let image_messages = [message(Role::User, MessageContent::Items(&image_items))];
        let requests = [
            Request {
                messages: &text_messages,
                images: &[],
                videos: &[],
                options: RequestOptions::default(),
            },
            Request {
                messages: &image_messages,
                images: &images,
                videos: &[],
                options: RequestOptions::default(),
            },
        ];
        assert_eq!(
            processor
                .plan_batch(&[])
                .expect_err("empty batch must be rejected")
                .category(),
            ErrorCategory::InvalidRequest
        );
        let plan = processor.plan_batch(&requests).expect("mixed batch plan");
        assert_eq!(plan.contract_id(), "qwen-mm-compat-v1");
        assert_eq!(plan.profile_fingerprint(), processor.profile().fingerprint);
        assert_eq!(plan.capacities().input_ids.shape[0], 2);
        assert_eq!(
            plan.capacities().pixel_values.expect("pixels").shape,
            [48, 1536]
        );
        assert_eq!(
            plan.capacities().image_grid_thw.expect("grids").shape,
            [3, 3]
        );
        assert_eq!(plan.request_layouts()[0].image_grid_rows.start, 0);
        assert_eq!(plan.request_layouts()[0].image_grid_rows.end, 0);
        assert_eq!(plan.request_layouts()[1].pixel_rows.start, 0);
        assert_eq!(plan.request_layouts()[1].pixel_rows.end, 48);
        assert!(plan.request_layouts()[0].right_padding > 0);
        assert_eq!(
            plan.image_layouts()[0].cache_key,
            plan.image_layouts()[1].cache_key
        );
        assert_ne!(
            plan.image_layouts()[1].cache_key,
            plan.image_layouts()[2].cache_key
        );
        assert_eq!(plan.image_layouts()[0].cache_key_hex().len(), 64);
        let other_profile = self::processor(ProfileAlias::Qwen35_9b, ResourceLimits::default());
        let other_plan = other_profile
            .plan_batch(&requests)
            .expect("same heterogeneous batch under second profile");
        assert_ne!(
            plan.image_layouts()[0].cache_key,
            other_plan.image_layouts()[0].cache_key
        );
        assert_eq!(other_plan.capacities().input_ids.shape[0], 2);

        let capacities = *plan.capacities();
        let mut input_ids = vec![-101_i64; capacities.input_ids.elements + 2];
        let mut attention_mask = vec![-102_i64; capacities.attention_mask.elements + 2];
        let mut mm_token_type_ids = vec![-103_i64; capacities.mm_token_type_ids.elements + 2];
        let mut pixel_values =
            vec![-104.0_f32; capacities.pixel_values.expect("pixels").elements + 2];
        let mut image_grid_thw =
            vec![-105_i64; capacities.image_grid_thw.expect("grids").elements + 2];
        {
            let view = processor
                .execute_plan_into(
                    &plan,
                    BatchDestinations {
                        input_ids: &mut input_ids,
                        attention_mask: &mut attention_mask,
                        mm_token_type_ids: &mut mm_token_type_ids,
                        pixel_values: Some(&mut pixel_values),
                        image_grid_thw: Some(&mut image_grid_thw),
                        pixel_values_videos: None,
                        video_grid_thw: None,
                    },
                )
                .expect("caller-owned execution");
            assert_eq!(
                view.official_keys(),
                [
                    "input_ids",
                    "attention_mask",
                    "mm_token_type_ids",
                    "pixel_values",
                    "image_grid_thw"
                ]
            );
            assert_eq!(view.arrays().input_ids.shape(), capacities.input_ids.shape);
            assert_eq!(view.sidecar().images.len(), 3);
            assert_eq!(view.images[0].cache_key, view.images[1].cache_key);
        }
        let first_row = plan.request_layouts()[0];
        assert!(
            attention_mask[first_row.text_elements.start
                ..first_row.text_elements.start + first_row.token_count]
                .iter()
                .all(|&value| value == 1)
        );
        assert!(
            input_ids[first_row.text_elements.start + first_row.token_count
                ..first_row.text_elements.end]
                .iter()
                .all(|&value| value == processor.profile().tokenizer.pad_token_id)
        );
        assert!(
            attention_mask[first_row.text_elements.start + first_row.token_count
                ..first_row.text_elements.end]
                .iter()
                .all(|&value| value == 0)
        );
        assert!(
            mm_token_type_ids[first_row.text_elements.start + first_row.token_count
                ..first_row.text_elements.end]
                .iter()
                .all(|&value| value == 0)
        );
        assert_eq!(&input_ids[capacities.input_ids.elements..], &[-101_i64; 2]);
        assert_eq!(
            &attention_mask[capacities.attention_mask.elements..],
            &[-102_i64; 2]
        );
        assert_eq!(
            &mm_token_type_ids[capacities.mm_token_type_ids.elements..],
            &[-103_i64; 2]
        );
        assert!(
            pixel_values[capacities.pixel_values.expect("pixels").elements..]
                .iter()
                .all(|value| value.to_bits() == (-104.0_f32).to_bits())
        );
        assert_eq!(
            &image_grid_thw[capacities.image_grid_thw.expect("grids").elements..],
            &[-105_i64; 2]
        );

        let mut second_ids = vec![0_i64; capacities.input_ids.elements];
        let mut second_attention = vec![0_i64; capacities.attention_mask.elements];
        let mut second_modalities = vec![0_i64; capacities.mm_token_type_ids.elements];
        let mut second_pixels = vec![0.0_f32; capacities.pixel_values.expect("pixels").elements];
        let mut second_grids = vec![0_i64; capacities.image_grid_thw.expect("grids").elements];
        {
            processor
                .execute_plan_into(
                    &plan,
                    BatchDestinations {
                        input_ids: &mut second_ids,
                        attention_mask: &mut second_attention,
                        mm_token_type_ids: &mut second_modalities,
                        pixel_values: Some(&mut second_pixels),
                        image_grid_thw: Some(&mut second_grids),
                        pixel_values_videos: None,
                        video_grid_thw: None,
                    },
                )
                .expect("reusable plan execution");
        }
        assert_eq!(second_ids, input_ids[..capacities.input_ids.elements]);
        assert!(
            second_pixels
                .iter()
                .zip(&pixel_values)
                .all(|(left, right)| { left.to_bits() == right.to_bits() })
        );

        let owned = processor
            .prepare_batch(&requests)
            .expect("allocating batch");
        assert_eq!(
            owned.batch.arrays().input_ids.as_slice(),
            &input_ids[..capacities.input_ids.elements]
        );
        assert_eq!(
            owned.batch.arrays().attention_mask.as_slice(),
            &attention_mask[..capacities.attention_mask.elements]
        );
        assert_eq!(
            owned.batch.arrays().mm_token_type_ids.as_slice(),
            &mm_token_type_ids[..capacities.mm_token_type_ids.elements]
        );
        assert_eq!(
            owned
                .batch
                .arrays()
                .pixel_values
                .as_ref()
                .expect("pixels")
                .as_slice(),
            &pixel_values[..capacities.pixel_values.expect("pixels").elements]
        );
        assert_eq!(
            owned
                .batch
                .arrays()
                .image_grid_thw
                .as_ref()
                .expect("grids")
                .as_slice(),
            &image_grid_thw[..capacities.image_grid_thw.expect("grids").elements]
        );
    }

    #[test]
    #[ignore = "requires hash-pinned local model snapshots under reference/.cache"]
    fn every_short_destination_fails_before_any_sentinel_is_modified() {
        let processor = processor(ProfileAlias::Qwen3Vl8b, ResourceLimits::default());
        let rgb = vec![5_u8; 64 * 64 * 3];
        let images = [ImageInput::Rgb8(Rgb8 {
            data: &rgb,
            height: 64,
            width: 64,
            row_stride: 64 * 3,
        })];
        let items = [ContentItem::Image(ImageRef::default())];
        let messages = [message(Role::User, MessageContent::Items(&items))];
        let requests = [Request {
            messages: &messages,
            images: &images,
            videos: &[],
            options: RequestOptions::default(),
        }];
        let plan = processor.plan_batch(&requests).expect("image plan");
        for output in [
            "input_ids",
            "attention_mask",
            "mm_token_type_ids",
            "pixel_values",
            "image_grid_thw",
        ] {
            assert_short_destination_is_atomic(&processor, &plan, output);
        }

        let capacities = *plan.capacities();
        for missing in ["pixel_values", "image_grid_thw"] {
            let mut input_ids = vec![-21_i64; capacities.input_ids.elements];
            let mut attention_mask = vec![-22_i64; capacities.attention_mask.elements];
            let mut mm_token_type_ids = vec![-23_i64; capacities.mm_token_type_ids.elements];
            let mut pixel_values =
                vec![-24.0_f32; capacities.pixel_values.expect("pixels").elements];
            let mut image_grid_thw =
                vec![-25_i64; capacities.image_grid_thw.expect("grid").elements];
            let error = processor
                .execute_plan_into(
                    &plan,
                    BatchDestinations {
                        input_ids: &mut input_ids,
                        attention_mask: &mut attention_mask,
                        mm_token_type_ids: &mut mm_token_type_ids,
                        pixel_values: (missing != "pixel_values").then_some(&mut pixel_values),
                        image_grid_thw: (missing != "image_grid_thw")
                            .then_some(&mut image_grid_thw),
                        pixel_values_videos: None,
                        video_grid_thw: None,
                    },
                )
                .expect_err("missing conditional destination must fail");
            assert_eq!(error.category(), ErrorCategory::DestinationTooSmall);
            assert_eq!(error.context()["output"].to_string(), missing);
            assert!(input_ids.iter().all(|&value| value == -21));
            assert!(attention_mask.iter().all(|&value| value == -22));
            assert!(mm_token_type_ids.iter().all(|&value| value == -23));
            assert!(
                pixel_values
                    .iter()
                    .all(|value| value.to_bits() == (-24.0_f32).to_bits())
            );
            assert!(image_grid_thw.iter().all(|&value| value == -25));
        }
    }

    #[test]
    #[ignore = "requires hash-pinned local model snapshots under reference/.cache"]
    #[allow(clippy::too_many_lines)]
    fn stale_plan_and_unplanned_conditional_destination_fail_before_writes() {
        let messages = [message(Role::User, MessageContent::Text("hello"))];
        let requests = [Request {
            messages: &messages,
            images: &[],
            videos: &[],
            options: RequestOptions::default(),
        }];
        let planner = processor(ProfileAlias::Qwen3Vl8b, ResourceLimits::default());
        let plan = planner.plan_batch(&requests).expect("text plan");
        let executor = processor(ProfileAlias::Qwen35_9b, ResourceLimits::default());
        let capacities = *plan.capacities();
        let mut ids = vec![-1_i64; capacities.input_ids.elements];
        let mut attention = vec![-2_i64; capacities.attention_mask.elements];
        let mut modalities = vec![-3_i64; capacities.mm_token_type_ids.elements];
        let compatible = processor(ProfileAlias::Qwen3Vl8b, ResourceLimits::default());
        let mut compatible_ids = vec![0_i64; capacities.input_ids.elements];
        let mut compatible_attention = vec![0_i64; capacities.attention_mask.elements];
        let mut compatible_modalities = vec![0_i64; capacities.mm_token_type_ids.elements];
        {
            compatible
                .execute_plan_into(
                    &plan,
                    BatchDestinations {
                        input_ids: &mut compatible_ids,
                        attention_mask: &mut compatible_attention,
                        mm_token_type_ids: &mut compatible_modalities,
                        pixel_values: None,
                        image_grid_thw: None,
                        pixel_values_videos: None,
                        video_grid_thw: None,
                    },
                )
                .expect("a distinct compatible processor can reuse the plan");
        }

        let mut wrong_contract_plan = planner.plan_batch(&requests).expect("second text plan");
        wrong_contract_plan.contract_id.push_str("-stale");
        let error = planner
            .execute_plan_into(
                &wrong_contract_plan,
                BatchDestinations {
                    input_ids: &mut ids,
                    attention_mask: &mut attention,
                    mm_token_type_ids: &mut modalities,
                    pixel_values: None,
                    image_grid_thw: None,
                    pixel_values_videos: None,
                    video_grid_thw: None,
                },
            )
            .expect_err("wrong contract makes the plan stale");
        assert_eq!(error.category(), ErrorCategory::ProfileMismatch);
        assert!(ids.iter().all(|&value| value == -1));
        assert!(attention.iter().all(|&value| value == -2));
        assert!(modalities.iter().all(|&value| value == -3));

        let error = executor
            .execute_plan_into(
                &plan,
                BatchDestinations {
                    input_ids: &mut ids,
                    attention_mask: &mut attention,
                    mm_token_type_ids: &mut modalities,
                    pixel_values: None,
                    image_grid_thw: None,
                    pixel_values_videos: None,
                    video_grid_thw: None,
                },
            )
            .expect_err("different profile makes the plan stale");
        assert_eq!(error.category(), ErrorCategory::ProfileMismatch);
        assert!(ids.iter().all(|&value| value == -1));
        assert!(attention.iter().all(|&value| value == -2));
        assert!(modalities.iter().all(|&value| value == -3));

        let lowered_limits = ResourceLimits::default()
            .lowered(LimitOverrides {
                requests_per_batch: Some(1),
                ..LimitOverrides::default()
            })
            .expect("lowered limits");
        let differently_limited = processor(ProfileAlias::Qwen3Vl8b, lowered_limits);
        let error = differently_limited
            .execute_plan_into(
                &plan,
                BatchDestinations {
                    input_ids: &mut ids,
                    attention_mask: &mut attention,
                    mm_token_type_ids: &mut modalities,
                    pixel_values: None,
                    image_grid_thw: None,
                    pixel_values_videos: None,
                    video_grid_thw: None,
                },
            )
            .expect_err("different limit policy makes the plan stale");
        assert_eq!(error.category(), ErrorCategory::ProfileMismatch);
        assert!(ids.iter().all(|&value| value == -1));
        assert!(attention.iter().all(|&value| value == -2));
        assert!(modalities.iter().all(|&value| value == -3));

        let mut extra = [9.0_f32];
        let error = planner
            .execute_plan_into(
                &plan,
                BatchDestinations {
                    input_ids: &mut ids,
                    attention_mask: &mut attention,
                    mm_token_type_ids: &mut modalities,
                    pixel_values: Some(&mut extra),
                    image_grid_thw: None,
                    pixel_values_videos: None,
                    video_grid_thw: None,
                },
            )
            .expect_err("text-only plan rejects image destination");
        assert_eq!(error.category(), ErrorCategory::DestinationTooSmall);
        assert_eq!(extra[0].to_bits(), 9.0_f32.to_bits());
        assert!(ids.iter().all(|&value| value == -1));

        let mut extra_grid = [9_i64];
        let error = planner
            .execute_plan_into(
                &plan,
                BatchDestinations {
                    input_ids: &mut ids,
                    attention_mask: &mut attention,
                    mm_token_type_ids: &mut modalities,
                    pixel_values: None,
                    image_grid_thw: Some(&mut extra_grid),
                    pixel_values_videos: None,
                    video_grid_thw: None,
                },
            )
            .expect_err("text-only plan rejects image-grid destination");
        assert_eq!(error.context()["output"].to_string(), "image_grid_thw");
        assert_eq!(extra_grid, [9]);

        let mut extra_video = [9.0_f32];
        let error = planner
            .execute_plan_into(
                &plan,
                BatchDestinations {
                    input_ids: &mut ids,
                    attention_mask: &mut attention,
                    mm_token_type_ids: &mut modalities,
                    pixel_values: None,
                    image_grid_thw: None,
                    pixel_values_videos: Some(&mut extra_video),
                    video_grid_thw: None,
                },
            )
            .expect_err("text-only plan rejects video-pixel destination");
        assert_eq!(error.context()["output"].to_string(), "pixel_values_videos");
        assert_eq!(extra_video[0].to_bits(), 9.0_f32.to_bits());

        let mut extra_video_grid = [9_i64];
        let error = planner
            .execute_plan_into(
                &plan,
                BatchDestinations {
                    input_ids: &mut ids,
                    attention_mask: &mut attention,
                    mm_token_type_ids: &mut modalities,
                    pixel_values: None,
                    image_grid_thw: None,
                    pixel_values_videos: None,
                    video_grid_thw: Some(&mut extra_video_grid),
                },
            )
            .expect_err("text-only plan rejects video-grid destination");
        assert_eq!(error.context()["output"].to_string(), "video_grid_thw");
        assert_eq!(extra_video_grid, [9]);
        assert!(ids.iter().all(|&value| value == -1));
    }

    #[test]
    #[ignore = "requires hash-pinned local model snapshots under reference/.cache"]
    fn aggregate_capacity_precedes_full_decode_during_batch_planning() {
        let corrupt_png = png_header(96, 96);
        let images = [ImageInput::Encoded {
            data: &corrupt_png,
            format: ImageFormat::Png,
        }];
        let items = [ContentItem::Image(ImageRef::default())];
        let messages = [message(Role::User, MessageContent::Items(&items))];
        let requests = [Request {
            messages: &messages,
            images: &images,
            videos: &[],
            options: RequestOptions::default(),
        }];
        let low_limits = ResourceLimits::default()
            .lowered(LimitOverrides {
                materialized_output_bytes_per_batch: Some(120_000),
                ..LimitOverrides::default()
            })
            .expect("lowered limits");
        assert_eq!(
            processor(ProfileAlias::Qwen3Vl8b, low_limits)
                .plan_batch(&requests)
                .expect_err("aggregate capacity wins")
                .category(),
            ErrorCategory::ResourceLimit
        );
        assert_eq!(
            processor(ProfileAlias::Qwen3Vl8b, ResourceLimits::default())
                .plan_batch(&requests)
                .expect_err("full decode runs after capacity")
                .category(),
            ErrorCategory::MediaDecode
        );
    }

    #[test]
    #[ignore = "requires hash-pinned local model snapshots under reference/.cache"]
    fn batch_media_errors_follow_global_stage_and_occurrence_order() {
        let corrupt_png = png_header(64, 64);
        let image_inputs = [ImageInput::Encoded {
            data: &corrupt_png,
            format: ImageFormat::Png,
        }];
        let image_items = [ContentItem::Image(ImageRef::default())];
        let image_messages = [message(Role::User, MessageContent::Items(&image_items))];
        let image_request = Request {
            messages: &image_messages,
            images: &image_inputs,
            videos: &[],
            options: RequestOptions::default(),
        };

        let frame_data = vec![0_u8; 64 * 64 * 3];
        let frames = [Rgb8 {
            data: &frame_data,
            height: 64,
            width: 64,
            row_stride: 64 * 3,
        }];
        let video_inputs = [VideoInput { frames: &frames }];
        let video_items = [ContentItem::Video(VideoRef::default())];
        let video_messages = [message(Role::User, MessageContent::Items(&video_items))];
        let video_request = Request {
            messages: &video_messages,
            images: &[],
            videos: &video_inputs,
            options: RequestOptions::default(),
        };

        let processor = processor(ProfileAlias::Qwen3Vl8b, ResourceLimits::default());
        assert_eq!(
            processor
                .plan_batch(&[image_request, video_request])
                .expect_err("earlier image decode wins the same media stage")
                .category(),
            ErrorCategory::MediaDecode
        );
        assert_eq!(
            processor
                .plan_batch(&[video_request, image_request])
                .expect_err("earlier unsupported video wins the same media stage")
                .category(),
            ErrorCategory::UnsupportedMedia
        );

        let large_corrupt_png = png_header(96, 96);
        let large_inputs = [ImageInput::Encoded {
            data: &large_corrupt_png,
            format: ImageFormat::Png,
        }];
        let large_request = Request {
            images: &large_inputs,
            ..image_request
        };
        let low_limits = ResourceLimits::default()
            .lowered(LimitOverrides {
                materialized_output_bytes_per_batch: Some(120_000),
                ..LimitOverrides::default()
            })
            .expect("lowered limits");
        assert_eq!(
            self::processor(ProfileAlias::Qwen3Vl8b, low_limits)
                .plan_batch(&[video_request, large_request])
                .expect_err("later aggregate output limit wins before media errors")
                .category(),
            ErrorCategory::ResourceLimit
        );
    }

    #[test]
    #[ignore = "requires hash-pinned local model snapshots under reference/.cache"]
    #[allow(clippy::too_many_lines)]
    fn mixed_encoded_codecs_and_raw_rgb_share_deterministic_batch_order() {
        let processor = processor(ProfileAlias::Qwen3Vl8b, ResourceLimits::default());
        let png = encode_rgb(ImageFormat::Png, 17);
        let webp = encode_rgb(ImageFormat::WebP, 29);
        let jpeg = include_bytes!("../../../fixtures/baseline/image24/image-00.jpg");
        let raw = vec![43_u8; 64 * 80 * 3];
        let images = [
            ImageInput::Encoded {
                data: &png,
                format: ImageFormat::Png,
            },
            ImageInput::Encoded {
                data: &webp,
                format: ImageFormat::WebP,
            },
            ImageInput::Rgb8(Rgb8 {
                data: &raw,
                height: 64,
                width: 80,
                row_stride: 80 * 3,
            }),
            ImageInput::Encoded {
                data: jpeg,
                format: ImageFormat::Jpeg,
            },
        ];
        let items = [
            ContentItem::Image(ImageRef {
                input_index: 1,
                ..ImageRef::default()
            }),
            ContentItem::Image(ImageRef {
                input_index: 2,
                options: ImageOptions {
                    resized_height: Some(112),
                    resized_width: Some(112),
                    ..ImageOptions::default()
                },
            }),
            ContentItem::Image(ImageRef {
                input_index: 3,
                ..ImageRef::default()
            }),
            ContentItem::Image(ImageRef::default()),
        ];
        let messages = [message(Role::User, MessageContent::Items(&items))];
        let requests = [Request {
            messages: &messages,
            images: &images,
            videos: &[],
            options: RequestOptions::default(),
        }];
        let plan = processor.plan_batch(&requests).expect("mixed-codec plan");
        let output = processor.prepare_batch(&requests).expect("mixed codecs");
        assert_eq!(
            output
                .images
                .iter()
                .map(|image| image.occurrence.location.input_index)
                .collect::<Vec<_>>(),
            [1, 2, 3, 0]
        );
        assert_eq!(output.images[1].occurrence.geometry.height, 128);
        assert_eq!(output.images[1].occurrence.geometry.width, 128);
        assert_eq!(
            output
                .batch
                .arrays()
                .image_grid_thw
                .as_ref()
                .expect("grids")
                .shape(),
            [4, 3]
        );
        assert!(
            output
                .images
                .windows(2)
                .all(|pair| pair[0].cache_key != pair[1].cache_key)
        );

        let capacities = *plan.capacities();
        let mut ids = vec![0_i64; capacities.input_ids.elements];
        let mut attention = vec![0_i64; capacities.attention_mask.elements];
        let mut modalities = vec![0_i64; capacities.mm_token_type_ids.elements];
        let mut pixels = vec![0.0_f32; capacities.pixel_values.expect("pixels").elements];
        let mut grids = vec![0_i64; capacities.image_grid_thw.expect("grids").elements];
        let view = processor
            .execute_plan_into(
                &plan,
                BatchDestinations {
                    input_ids: &mut ids,
                    attention_mask: &mut attention,
                    mm_token_type_ids: &mut modalities,
                    pixel_values: Some(&mut pixels),
                    image_grid_thw: Some(&mut grids),
                    pixel_values_videos: None,
                    video_grid_thw: None,
                },
            )
            .expect("mixed-codec plan execution");
        assert_eq!(view.images, output.images.as_slice());
        assert_eq!(view.sidecar(), output.batch.sidecar());
        assert_eq!(
            view.arrays().input_ids.as_slice(),
            output.batch.arrays().input_ids.as_slice()
        );
        assert_eq!(
            view.arrays().attention_mask.as_slice(),
            output.batch.arrays().attention_mask.as_slice()
        );
        assert_eq!(
            view.arrays().mm_token_type_ids.as_slice(),
            output.batch.arrays().mm_token_type_ids.as_slice()
        );
        assert_eq!(
            view.arrays()
                .pixel_values
                .as_ref()
                .expect("view pixels")
                .as_slice(),
            output
                .batch
                .arrays()
                .pixel_values
                .as_ref()
                .expect("owned pixels")
                .as_slice()
        );
        assert_eq!(
            view.arrays()
                .image_grid_thw
                .as_ref()
                .expect("view grids")
                .as_slice(),
            output
                .batch
                .arrays()
                .image_grid_thw
                .as_ref()
                .expect("owned grids")
                .as_slice()
        );
    }

    #[test]
    #[ignore = "requires hash-pinned local model snapshots under reference/.cache"]
    #[allow(clippy::too_many_lines)]
    fn multi_request_images_match_legacy_rows_and_global_ranges() {
        let processor = processor(ProfileAlias::Qwen3Vl8b, ResourceLimits::default());
        let first_rgb = vec![7_u8; 64 * 64 * 3];
        let second_rgb = vec![11_u8; 64 * 64 * 3];
        let first_inputs = [ImageInput::Rgb8(Rgb8 {
            data: &first_rgb,
            height: 64,
            width: 64,
            row_stride: 64 * 3,
        })];
        let second_inputs = [ImageInput::Rgb8(Rgb8 {
            data: &second_rgb,
            height: 64,
            width: 64,
            row_stride: 64 * 3,
        })];
        let first_items = [
            ContentItem::Image(ImageRef::default()),
            ContentItem::Text(" first"),
        ];
        let second_items = [
            ContentItem::Text("second "),
            ContentItem::Image(ImageRef::default()),
        ];
        let first_messages = [message(Role::User, MessageContent::Items(&first_items))];
        let second_messages = [message(Role::User, MessageContent::Items(&second_items))];
        let first = Request {
            messages: &first_messages,
            images: &first_inputs,
            videos: &[],
            options: RequestOptions::default(),
        };
        let second = Request {
            messages: &second_messages,
            images: &second_inputs,
            videos: &[],
            options: RequestOptions::default(),
        };

        let first_legacy = processor.prepare(first).expect("first legacy request");
        let second_legacy = processor.prepare(second).expect("second legacy request");
        let batch = processor
            .prepare_batch(&[first, second])
            .expect("two image requests");
        let singleton = processor
            .prepare_batch(&[first])
            .expect("singleton allocating batch");
        assert_eq!(singleton.batch.arrays(), first_legacy.batch.arrays());
        assert_eq!(singleton.batch.sidecar(), first_legacy.batch.sidecar());
        assert_eq!(
            singleton.text.as_slice(),
            std::slice::from_ref(&first_legacy.text)
        );
        assert_eq!(singleton.images[0].occurrence, first_legacy.images[0]);
        assert_eq!(
            batch
                .images
                .iter()
                .map(|image| image.occurrence.location.request_index)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(batch.images[0].occurrence.grid_row, 0);
        assert_eq!(batch.images[1].occurrence.grid_row, 1);
        assert_eq!(
            batch.images[0].occurrence.pixel_rows,
            CoordinateRange { start: 0, end: 16 }
        );
        assert_eq!(
            batch.images[1].occurrence.pixel_rows,
            CoordinateRange { start: 16, end: 32 }
        );
        assert_eq!(
            batch
                .batch
                .sidecar()
                .images
                .iter()
                .map(|image| image.request_index)
                .collect::<Vec<_>>(),
            [0, 1]
        );

        let batch_pixels = batch
            .batch
            .arrays()
            .pixel_values
            .as_ref()
            .expect("batch pixels")
            .as_slice();
        let first_pixels = first_legacy
            .batch
            .arrays()
            .pixel_values
            .as_ref()
            .expect("first pixels")
            .as_slice();
        let second_pixels = second_legacy
            .batch
            .arrays()
            .pixel_values
            .as_ref()
            .expect("second pixels")
            .as_slice();
        assert_eq!(&batch_pixels[..first_pixels.len()], first_pixels);
        assert_eq!(&batch_pixels[first_pixels.len()..], second_pixels);

        let columns = batch.batch.arrays().input_ids.columns();
        for (row, legacy) in [first_legacy, second_legacy].iter().enumerate() {
            let token_count = legacy.batch.arrays().input_ids.columns();
            let start = row * columns;
            assert_eq!(
                &batch.batch.arrays().input_ids.as_slice()[start..start + token_count],
                legacy.batch.arrays().input_ids.as_slice()
            );
            assert!(
                batch.batch.arrays().attention_mask.as_slice()[start..start + token_count]
                    .iter()
                    .all(|&value| value == 1)
            );
            assert!(
                batch.batch.arrays().attention_mask.as_slice()
                    [start + token_count..start + columns]
                    .iter()
                    .all(|&value| value == 0)
            );
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
