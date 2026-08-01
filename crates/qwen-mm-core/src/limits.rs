//! Configurable resource limits, checked arithmetic, and staged preflight.

use crate::{
    error::{ErrorCategory, QwenError, Result},
    profile::ProfileRegistry,
    request::{
        ContentItem, ImageInput, ImageOptions, MessageContent, Request, Rgb8, VideoOptions,
        validate_request_options, validate_request_structure,
    },
};

/// Frozen default safety limits from compatibility contract v1.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceLimits {
    requests_per_batch: u64,
    messages_per_request: u64,
    content_items_per_request: u64,
    text_bytes_per_request: u64,
    media_per_request: u64,
    media_per_batch: u64,
    encoded_bytes_per_item: u64,
    encoded_bytes_per_batch: u64,
    decoded_pixels_per_image_or_frame: u64,
    decoded_edge_length: u64,
    prepared_image_pixels_per_occurrence: u64,
    raw_frames_per_video: u64,
    rendered_tokens_per_request: u64,
    rendered_tokens_per_batch: u64,
    materialized_output_bytes_per_batch: u64,
}

/// Optional caller-lowered values for every v1 resource dimension.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LimitOverrides {
    /// Requests per batch.
    pub requests_per_batch: Option<u64>,
    /// Messages per request.
    pub messages_per_request: Option<u64>,
    /// Content items per request.
    pub content_items_per_request: Option<u64>,
    /// UTF-8 text and tool JSON bytes per request.
    pub text_bytes_per_request: Option<u64>,
    /// Media occurrences per request.
    pub media_per_request: Option<u64>,
    /// Media occurrences per batch.
    pub media_per_batch: Option<u64>,
    /// Encoded bytes per item.
    pub encoded_bytes_per_item: Option<u64>,
    /// Encoded bytes per batch.
    pub encoded_bytes_per_batch: Option<u64>,
    /// Decoded source pixels per image or frame.
    pub decoded_pixels_per_image_or_frame: Option<u64>,
    /// Decoded source edge length.
    pub decoded_edge_length: Option<u64>,
    /// Prepared still-image pixels per occurrence.
    pub prepared_image_pixels_per_occurrence: Option<u64>,
    /// Raw frames per video.
    pub raw_frames_per_video: Option<u64>,
    /// Rendered tokens per request.
    pub rendered_tokens_per_request: Option<u64>,
    /// Rendered tokens per batch.
    pub rendered_tokens_per_batch: Option<u64>,
    /// Materialized output bytes per batch.
    pub materialized_output_bytes_per_batch: Option<u64>,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            requests_per_batch: 256,
            messages_per_request: 256,
            content_items_per_request: 1_024,
            text_bytes_per_request: 4 * 1024 * 1024,
            media_per_request: 64,
            media_per_batch: 256,
            encoded_bytes_per_item: 64 * 1024 * 1024,
            encoded_bytes_per_batch: 1024 * 1024 * 1024,
            decoded_pixels_per_image_or_frame: 67_108_864,
            decoded_edge_length: 32_768,
            prepared_image_pixels_per_occurrence: 16_777_216,
            raw_frames_per_video: 768,
            rendered_tokens_per_request: 262_144,
            rendered_tokens_per_batch: 1_048_576,
            materialized_output_bytes_per_batch: 4 * 1024 * 1024 * 1024,
        }
    }
}

macro_rules! limit_getters {
    ($(($name:ident, $field:ident)),+ $(,)?) => {
        $(
            #[doc = concat!("Returns the `", stringify!($field), "` limit.")]
            #[must_use]
            pub const fn $name(&self) -> u64 {
                self.$field
            }
        )+
    };
}

impl ResourceLimits {
    limit_getters!(
        (requests_per_batch, requests_per_batch),
        (messages_per_request, messages_per_request),
        (content_items_per_request, content_items_per_request),
        (text_bytes_per_request, text_bytes_per_request),
        (media_per_request, media_per_request),
        (media_per_batch, media_per_batch),
        (encoded_bytes_per_item, encoded_bytes_per_item),
        (encoded_bytes_per_batch, encoded_bytes_per_batch),
        (
            decoded_pixels_per_image_or_frame,
            decoded_pixels_per_image_or_frame
        ),
        (decoded_edge_length, decoded_edge_length),
        (
            prepared_image_pixels_per_occurrence,
            prepared_image_pixels_per_occurrence
        ),
        (raw_frames_per_video, raw_frames_per_video),
        (rendered_tokens_per_request, rendered_tokens_per_request),
        (rendered_tokens_per_batch, rendered_tokens_per_batch),
        (
            materialized_output_bytes_per_batch,
            materialized_output_bytes_per_batch
        ),
    );

    /// Creates an unnamed policy that only lowers the current limits.
    ///
    /// Raising a limit requires a distinct named runtime policy outside this
    /// compatibility constructor.
    ///
    /// # Errors
    ///
    /// Returns `unsupported_option` if an override raises a current limit.
    pub fn lowered(self, overrides: LimitOverrides) -> Result<Self> {
        Ok(Self {
            requests_per_batch: lower(
                "requests_per_batch",
                self.requests_per_batch,
                overrides.requests_per_batch,
            )?,
            messages_per_request: lower(
                "messages_per_request",
                self.messages_per_request,
                overrides.messages_per_request,
            )?,
            content_items_per_request: lower(
                "content_items_per_request",
                self.content_items_per_request,
                overrides.content_items_per_request,
            )?,
            text_bytes_per_request: lower(
                "text_bytes_per_request",
                self.text_bytes_per_request,
                overrides.text_bytes_per_request,
            )?,
            media_per_request: lower(
                "media_per_request",
                self.media_per_request,
                overrides.media_per_request,
            )?,
            media_per_batch: lower(
                "media_per_batch",
                self.media_per_batch,
                overrides.media_per_batch,
            )?,
            encoded_bytes_per_item: lower(
                "encoded_bytes_per_item",
                self.encoded_bytes_per_item,
                overrides.encoded_bytes_per_item,
            )?,
            encoded_bytes_per_batch: lower(
                "encoded_bytes_per_batch",
                self.encoded_bytes_per_batch,
                overrides.encoded_bytes_per_batch,
            )?,
            decoded_pixels_per_image_or_frame: lower(
                "decoded_pixels_per_image_or_frame",
                self.decoded_pixels_per_image_or_frame,
                overrides.decoded_pixels_per_image_or_frame,
            )?,
            decoded_edge_length: lower(
                "decoded_edge_length",
                self.decoded_edge_length,
                overrides.decoded_edge_length,
            )?,
            prepared_image_pixels_per_occurrence: lower(
                "prepared_image_pixels_per_occurrence",
                self.prepared_image_pixels_per_occurrence,
                overrides.prepared_image_pixels_per_occurrence,
            )?,
            raw_frames_per_video: lower(
                "raw_frames_per_video",
                self.raw_frames_per_video,
                overrides.raw_frames_per_video,
            )?,
            rendered_tokens_per_request: lower(
                "rendered_tokens_per_request",
                self.rendered_tokens_per_request,
                overrides.rendered_tokens_per_request,
            )?,
            rendered_tokens_per_batch: lower(
                "rendered_tokens_per_batch",
                self.rendered_tokens_per_batch,
                overrides.rendered_tokens_per_batch,
            )?,
            materialized_output_bytes_per_batch: lower(
                "materialized_output_bytes_per_batch",
                self.materialized_output_bytes_per_batch,
                overrides.materialized_output_bytes_per_batch,
            )?,
        })
    }

    /// Checks exact post-render token counts without allocating output arrays.
    ///
    /// # Errors
    ///
    /// Returns `resource_limit` when either token limit is exceeded, or
    /// `arithmetic_overflow` when the batch total cannot be represented.
    pub fn check_rendered_tokens(self, per_request: &[u64]) -> Result<()> {
        let mut batch = 0_u64;
        for (request_index, &actual) in per_request.iter().enumerate() {
            check_limit(
                "rendered_tokens_per_request",
                actual,
                self.rendered_tokens_per_request,
                Some(request_index),
            )?;
            batch = checked_add("rendered token batch count", batch, actual)?;
        }
        check_limit(
            "rendered_tokens_per_batch",
            batch,
            self.rendered_tokens_per_batch,
            None,
        )
    }

    /// Checks an exact planned materialized output capacity.
    ///
    /// # Errors
    ///
    /// Returns `resource_limit` when `bytes` exceeds the configured limit.
    pub fn check_materialized_output_bytes(self, bytes: u64) -> Result<()> {
        check_limit(
            "materialized_output_bytes_per_batch",
            bytes,
            self.materialized_output_bytes_per_batch,
            None,
        )
    }
}

/// A request paired with the exact local profile alias it requires.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProfiledRequest<'a> {
    /// Exact profile alias.
    pub profile_alias: &'a str,
    /// Structured core request.
    pub request: Request<'a>,
}

/// Counts established by allocation-free request preflight.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PreflightSummary {
    /// Requests in the batch.
    pub requests: u64,
    /// Messages across the batch.
    pub messages: u64,
    /// List content items across the batch.
    pub content_items: u64,
    /// UTF-8 text and tool JSON bytes across the batch.
    pub text_bytes: u64,
    /// Media occurrences across the batch.
    pub media_occurrences: u64,
    /// Encoded image bytes across the batch.
    pub encoded_bytes: u64,
    /// Caller-owned raw frames across the batch.
    pub raw_frames: u64,
}

/// Validates a whole batch in frozen stage order without writing output memory.
///
/// The order is request structure, profile, options, resource preflight, then
/// media geometry. Decode and destination validation are later stages.
///
/// # Errors
///
/// Returns the first categorized error in that frozen validation order.
pub fn preflight_batch(
    registry: &ProfileRegistry,
    batch: &[ProfiledRequest<'_>],
    limits: ResourceLimits,
) -> Result<PreflightSummary> {
    let summary = preflight_batch_through_resources(registry, batch, limits)?;
    for (request_index, item) in batch.iter().enumerate() {
        validate_geometry(&item.request, request_index)?;
    }
    Ok(summary)
}

/// Runs the common validation prefix through resource preflight, stopping
/// before media geometry. The composed image processor uses this boundary so
/// recognized corrupt media wins over later geometry/stage failures, as
/// required by [`crate::ValidationStage::ORDER`].
pub(crate) fn preflight_batch_through_resources(
    registry: &ProfileRegistry,
    batch: &[ProfiledRequest<'_>],
    limits: ResourceLimits,
) -> Result<PreflightSummary> {
    if batch.is_empty() {
        return Err(QwenError::new(
            ErrorCategory::InvalidRequest,
            "batch must contain at least one request",
        ));
    }
    for (request_index, item) in batch.iter().enumerate() {
        validate_request_structure(&item.request, request_index)?;
    }
    preflight_batch_after_structure_through_resources(registry, batch, limits)
}

/// Continues preflight after request/render structure has already been proven.
pub(crate) fn preflight_batch_after_structure_through_resources(
    registry: &ProfileRegistry,
    batch: &[ProfiledRequest<'_>],
    limits: ResourceLimits,
) -> Result<PreflightSummary> {
    for item in batch {
        registry.resolve(item.profile_alias)?;
    }
    for (request_index, item) in batch.iter().enumerate() {
        validate_request_options(
            &item.request,
            registry.resolve(item.profile_alias)?,
            request_index,
        )?;
    }
    validate_resources(batch, limits)
}

/// Checked `u64` addition with a categorized overflow.
///
/// # Errors
///
/// Returns `arithmetic_overflow` if the sum cannot be represented.
pub fn checked_add(operation: &'static str, left: u64, right: u64) -> Result<u64> {
    left.checked_add(right).ok_or_else(|| {
        QwenError::new(
            ErrorCategory::ArithmeticOverflow,
            "checked addition overflowed",
        )
        .with_context("operation", operation)
        .with_context("left", left)
        .with_context("right", right)
    })
}

/// Checked `u64` multiplication with a categorized overflow.
///
/// # Errors
///
/// Returns `arithmetic_overflow` if the product cannot be represented.
pub fn checked_mul(operation: &'static str, left: u64, right: u64) -> Result<u64> {
    left.checked_mul(right).ok_or_else(|| {
        QwenError::new(
            ErrorCategory::ArithmeticOverflow,
            "checked multiplication overflowed",
        )
        .with_context("operation", operation)
        .with_context("left", left)
        .with_context("right", right)
    })
}

/// Computes row-major byte capacity with checked count and stride arithmetic.
///
/// # Errors
///
/// Returns `arithmetic_overflow` if either product cannot be represented.
pub fn checked_capacity_bytes(rows: u64, columns: u64, element_bytes: u64) -> Result<u64> {
    let row_bytes = checked_mul("row byte stride", columns, element_bytes)?;
    checked_mul("matrix byte capacity", rows, row_bytes)
}

fn validate_resources(
    batch: &[ProfiledRequest<'_>],
    limits: ResourceLimits,
) -> Result<PreflightSummary> {
    let requests = usize_to_u64("request count", batch.len())?;
    check_limit(
        "requests_per_batch",
        requests,
        limits.requests_per_batch,
        None,
    )?;

    let mut summary = PreflightSummary {
        requests,
        ..PreflightSummary::default()
    };
    for (request_index, item) in batch.iter().enumerate() {
        validate_request_resources(&item.request, request_index, limits, &mut summary)?;
    }
    check_limit(
        "media_per_batch",
        summary.media_occurrences,
        limits.media_per_batch,
        None,
    )?;
    check_limit(
        "encoded_bytes_per_batch",
        summary.encoded_bytes,
        limits.encoded_bytes_per_batch,
        None,
    )?;
    Ok(summary)
}

fn validate_request_resources(
    request: &Request<'_>,
    request_index: usize,
    limits: ResourceLimits,
    summary: &mut PreflightSummary,
) -> Result<()> {
    let messages = usize_to_u64("message count", request.messages.len())?;
    check_limit(
        "messages_per_request",
        messages,
        limits.messages_per_request,
        Some(request_index),
    )?;
    summary.messages = checked_add("batch message count", summary.messages, messages)?;

    let (content_items, text_bytes, media_occurrences) = request_counts(request)?;
    for (name, actual, limit) in [
        (
            "content_items_per_request",
            content_items,
            limits.content_items_per_request,
        ),
        (
            "text_bytes_per_request",
            text_bytes,
            limits.text_bytes_per_request,
        ),
        (
            "media_per_request",
            media_occurrences,
            limits.media_per_request,
        ),
    ] {
        check_limit(name, actual, limit, Some(request_index))?;
    }
    summary.content_items = checked_add(
        "batch content item count",
        summary.content_items,
        content_items,
    )?;
    summary.text_bytes = checked_add("batch text byte count", summary.text_bytes, text_bytes)?;
    summary.media_occurrences = checked_add(
        "batch media occurrence count",
        summary.media_occurrences,
        media_occurrences,
    )?;

    for image in request.images {
        match image {
            ImageInput::Encoded { data, .. } => {
                let bytes = usize_to_u64("encoded image bytes", data.len())?;
                check_limit(
                    "encoded_bytes_per_item",
                    bytes,
                    limits.encoded_bytes_per_item,
                    Some(request_index),
                )?;
                summary.encoded_bytes =
                    checked_add("batch encoded bytes", summary.encoded_bytes, bytes)?;
            }
            ImageInput::Rgb8(rgb) => validate_rgb_resources(rgb, limits, request_index)?,
        }
    }
    for video in request.videos {
        let frames = usize_to_u64("raw frame count", video.frames.len())?;
        check_limit(
            "raw_frames_per_video",
            frames,
            limits.raw_frames_per_video,
            Some(request_index),
        )?;
        summary.raw_frames = checked_add("batch raw frame count", summary.raw_frames, frames)?;
        for frame in video.frames {
            validate_rgb_resources(frame, limits, request_index)?;
        }
    }
    validate_occurrence_sizes(request, limits, request_index)
}

fn validate_occurrence_sizes(
    request: &Request<'_>,
    limits: ResourceLimits,
    request_index: usize,
) -> Result<()> {
    for message in request.messages {
        if let MessageContent::Items(items) = message.content {
            for item in items {
                match item {
                    ContentItem::Image(reference) => {
                        validate_prepared_image_resources(
                            reference.options,
                            limits,
                            request_index,
                        )?;
                    }
                    ContentItem::Video(reference) => {
                        validate_video_explicit_size_arithmetic(
                            reference.options.resized_height,
                            reference.options.resized_width,
                        )?;
                    }
                    ContentItem::Text(_) => {}
                }
            }
        }
    }
    Ok(())
}

fn request_counts(request: &Request<'_>) -> Result<(u64, u64, u64)> {
    let mut content_items = 0_u64;
    let mut text_bytes = 0_u64;
    let mut media = 0_u64;
    for message in request.messages {
        match message.content {
            MessageContent::Text(text) => {
                text_bytes = checked_add(
                    "request text byte count",
                    text_bytes,
                    usize_to_u64("text bytes", text.len())?,
                )?;
            }
            MessageContent::Items(items) => {
                content_items = checked_add(
                    "request content item count",
                    content_items,
                    usize_to_u64("content item count", items.len())?,
                )?;
                for item in items {
                    match item {
                        ContentItem::Text(text) => {
                            text_bytes = checked_add(
                                "request text byte count",
                                text_bytes,
                                usize_to_u64("text bytes", text.len())?,
                            )?;
                        }
                        ContentItem::Image(_) | ContentItem::Video(_) => {
                            media = checked_add("request media occurrence count", media, 1)?;
                        }
                    }
                }
            }
        }
        if let Some(reasoning) = message.reasoning_content {
            text_bytes = checked_add(
                "request reasoning byte count",
                text_bytes,
                usize_to_u64("reasoning bytes", reasoning.len())?,
            )?;
        }
        for tool_call in message.tool_calls {
            let function = tool_call.function_call();
            if let Some(id) = tool_call.id() {
                text_bytes = checked_add(
                    "request tool-call byte count",
                    text_bytes,
                    usize_to_u64("tool-call ID bytes", id.len())?,
                )?;
            }
            text_bytes = checked_add(
                "request tool-call byte count",
                text_bytes,
                usize_to_u64("tool-call name bytes", function.name.len())?,
            )?;
            text_bytes = checked_add(
                "request tool-call byte count",
                text_bytes,
                usize_to_u64("tool-call JSON bytes", function.arguments_json.len())?,
            )?;
        }
    }
    for tool in request.options.tools {
        text_bytes = checked_add(
            "request tool definition byte count",
            text_bytes,
            usize_to_u64("tool definition JSON bytes", tool.json.len())?,
        )?;
    }
    Ok((content_items, text_bytes, media))
}

fn validate_rgb_resources(
    rgb: &Rgb8<'_>,
    limits: ResourceLimits,
    request_index: usize,
) -> Result<()> {
    let height = usize_to_u64("RGB height", rgb.height)?;
    let width = usize_to_u64("RGB width", rgb.width)?;
    let stride = usize_to_u64("RGB row stride", rgb.row_stride)?;
    // Products are checked before limits so overflow never masquerades as a
    // configurable resource failure.
    let pixels = checked_mul("decoded source pixels", height, width)?;
    let row_bytes = checked_mul("packed RGB row bytes", width, 3)?;
    let rows_before_last = height.saturating_sub(1);
    let row_offsets = checked_mul("RGB final row offset", rows_before_last, stride)?;
    let _required_bytes = checked_add("RGB readable byte span", row_offsets, row_bytes)?;

    check_limit(
        "decoded_source_pixels",
        pixels,
        limits.decoded_pixels_per_image_or_frame,
        Some(request_index),
    )?;
    check_limit(
        "decoded_edge_length",
        height,
        limits.decoded_edge_length,
        Some(request_index),
    )?;
    check_limit(
        "decoded_edge_length",
        width,
        limits.decoded_edge_length,
        Some(request_index),
    )
}

fn validate_prepared_image_resources(
    options: ImageOptions,
    limits: ResourceLimits,
    request_index: usize,
) -> Result<()> {
    for (option, pixels) in [
        ("min_pixels", options.min_pixels),
        ("max_pixels", options.max_pixels),
    ] {
        if let Some(pixels) = pixels {
            check_limit(
                "prepared_image_pixels_per_occurrence",
                pixels,
                limits.prepared_image_pixels_per_occurrence,
                Some(request_index),
            )
            .map_err(|error| error.with_context("option", option))?;
        }
    }
    if let (Some(height), Some(width)) = (options.resized_height, options.resized_width) {
        let pixels = checked_mul("explicit resized image pixels", height, width)?;
        check_limit(
            "prepared_image_pixels_per_occurrence",
            pixels,
            limits.prepared_image_pixels_per_occurrence,
            Some(request_index),
        )?;
    }
    Ok(())
}

fn validate_video_explicit_size_arithmetic(height: Option<u64>, width: Option<u64>) -> Result<()> {
    if let (Some(height), Some(width)) = (height, width) {
        let _pixels = checked_mul("explicit resized video pixels", height, width)?;
    }
    Ok(())
}

fn validate_geometry(request: &Request<'_>, request_index: usize) -> Result<()> {
    for image in request.images {
        if let ImageInput::Rgb8(rgb) = image {
            validate_rgb_geometry(rgb, request_index)?;
        }
    }
    for video in request.videos {
        if video.frames.is_empty() {
            return Err(geometry("raw-frame video must contain at least one frame")
                .with_context("request_index", request_index));
        }
        let expected = (video.frames[0].height, video.frames[0].width);
        for (frame_index, frame) in video.frames.iter().enumerate() {
            validate_rgb_geometry(frame, request_index)?;
            if (frame.height, frame.width) != expected {
                return Err(geometry("raw video frames must have common dimensions")
                    .with_context("request_index", request_index)
                    .with_context("frame_index", frame_index));
            }
        }
    }
    for message in request.messages {
        if let MessageContent::Items(items) = message.content {
            for item in items {
                match item {
                    ContentItem::Image(reference) => {
                        validate_image_options(reference.options, request_index)?;
                    }
                    ContentItem::Video(reference) => {
                        validate_video_options(reference.options, request_index)?;
                    }
                    ContentItem::Text(_) => {}
                }
            }
        }
    }
    Ok(())
}

fn validate_rgb_geometry(rgb: &Rgb8<'_>, request_index: usize) -> Result<()> {
    if rgb.height == 0 || rgb.width == 0 {
        return Err(geometry("RGB dimensions must be non-zero")
            .with_context("request_index", request_index));
    }
    let width = usize_to_u64("RGB width", rgb.width)?;
    let height = usize_to_u64("RGB height", rgb.height)?;
    validate_aspect_ratio(height, width, request_index)?;
    let row_bytes = checked_mul("packed RGB row bytes", width, 3)?;
    let stride = usize_to_u64("RGB row stride", rgb.row_stride)?;
    if stride < row_bytes {
        return Err(geometry("RGB row stride is smaller than a packed row")
            .with_context("request_index", request_index)
            .with_context("row_stride", stride)
            .with_context("minimum", row_bytes));
    }
    let offset = checked_mul("RGB final row offset", height - 1, stride)?;
    let required = checked_add("RGB readable byte span", offset, row_bytes)?;
    let available = usize_to_u64("RGB data bytes", rgb.data.len())?;
    if available < required {
        return Err(geometry("RGB source does not cover its declared geometry")
            .with_context("request_index", request_index)
            .with_context("required_bytes", required)
            .with_context("available_bytes", available));
    }
    Ok(())
}

fn validate_image_options(options: ImageOptions, request_index: usize) -> Result<()> {
    validate_pixel_options(
        options.min_pixels,
        options.max_pixels,
        options.resized_height,
        options.resized_width,
        request_index,
    )
}

fn validate_video_options(options: VideoOptions, request_index: usize) -> Result<()> {
    validate_pixel_options(
        options.min_pixels,
        options.max_pixels,
        options.resized_height,
        options.resized_width,
        request_index,
    )?;
    for (name, value) in [
        ("sample_fps", options.sample_fps),
        ("raw_fps", options.raw_fps),
    ] {
        if value.is_some_and(|value| !value.is_finite() || value <= 0.0) {
            return Err(geometry("video frame rate must be finite and positive")
                .with_context("request_index", request_index)
                .with_context("option", name));
        }
    }
    Ok(())
}

fn validate_pixel_options(
    min_pixels: Option<u64>,
    max_pixels: Option<u64>,
    resized_height: Option<u64>,
    resized_width: Option<u64>,
    request_index: usize,
) -> Result<()> {
    if min_pixels
        .zip(max_pixels)
        .is_some_and(|(min, max)| max < min)
    {
        return Err(
            geometry("max_pixels must be greater than or equal to min_pixels")
                .with_context("request_index", request_index),
        );
    }
    match (resized_height, resized_width) {
        (Some(height), Some(width)) => {
            if height == 0 || width == 0 {
                return Err(geometry("explicit resized dimensions must be non-zero")
                    .with_context("request_index", request_index));
            }
            validate_aspect_ratio(height, width, request_index)
        }
        (None, None) => Ok(()),
        _ => Err(
            geometry("resized_height and resized_width must be supplied together")
                .with_context("request_index", request_index),
        ),
    }
}

fn validate_aspect_ratio(height: u64, width: u64, request_index: usize) -> Result<()> {
    let (larger, smaller) = if height >= width {
        (height, width)
    } else {
        (width, height)
    };
    if smaller == 0 {
        return Err(geometry("media dimensions must be non-zero")
            .with_context("request_index", request_index));
    }
    let maximum = checked_mul("aspect ratio boundary", smaller, 200)?;
    if larger > maximum {
        return Err(geometry("media aspect ratio exceeds 200")
            .with_context("request_index", request_index)
            .with_context("height", height)
            .with_context("width", width));
    }
    Ok(())
}

fn lower(name: &'static str, current: u64, requested: Option<u64>) -> Result<u64> {
    match requested {
        Some(value) if value > current => Err(QwenError::new(
            ErrorCategory::UnsupportedOption,
            "unnamed resource policy cannot raise a compatibility default",
        )
        .with_context("limit", name)
        .with_context("default", current)
        .with_context("requested", value)),
        Some(value) => Ok(value),
        None => Ok(current),
    }
}

fn check_limit(
    name: &'static str,
    actual: u64,
    limit: u64,
    request_index: Option<usize>,
) -> Result<()> {
    if actual <= limit {
        return Ok(());
    }
    let mut error = QwenError::new(ErrorCategory::ResourceLimit, "resource limit exceeded")
        .with_context("resource", name)
        .with_context("actual", actual)
        .with_context("limit", limit);
    if let Some(index) = request_index {
        error = error.with_context("request_index", index);
    }
    Err(error)
}

fn usize_to_u64(operation: &'static str, value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| {
        QwenError::new(
            ErrorCategory::ArithmeticOverflow,
            "platform count does not fit parity arithmetic",
        )
        .with_context("operation", operation)
        .with_context("value", value)
    })
}

fn geometry(message: impl Into<String>) -> QwenError {
    QwenError::new(ErrorCategory::MediaGeometry, message)
}

#[cfg(test)]
mod tests {
    use super::{
        LimitOverrides, ProfiledRequest, ResourceLimits, checked_capacity_bytes, checked_mul,
        preflight_batch,
    };
    use crate::{
        error::ErrorCategory,
        profile::ProfileRegistry,
        request::{
            ContentItem, ExcludedOptions, ImageInput, ImageOptions, ImageRef, Message,
            MessageContent, Request, RequestOptions, Rgb8, Role, VideoInput, VideoOptions,
            VideoRef,
        },
    };

    fn text_request<'a>(messages: &'a [Message<'a>]) -> Request<'a> {
        Request {
            messages,
            images: &[],
            videos: &[],
            options: RequestOptions::default(),
        }
    }

    #[test]
    fn defaults_match_the_adr() {
        let limits = ResourceLimits::default();
        assert_eq!(limits.requests_per_batch(), 256);
        assert_eq!(limits.messages_per_request(), 256);
        assert_eq!(limits.content_items_per_request(), 1_024);
        assert_eq!(limits.text_bytes_per_request(), 4 * 1024 * 1024);
        assert_eq!(limits.media_per_request(), 64);
        assert_eq!(limits.media_per_batch(), 256);
        assert_eq!(limits.encoded_bytes_per_item(), 64 * 1024 * 1024);
        assert_eq!(limits.encoded_bytes_per_batch(), 1024 * 1024 * 1024);
        assert_eq!(limits.decoded_pixels_per_image_or_frame(), 67_108_864);
        assert_eq!(limits.decoded_edge_length(), 32_768);
        assert_eq!(limits.prepared_image_pixels_per_occurrence(), 16_777_216);
        assert_eq!(limits.raw_frames_per_video(), 768);
        assert_eq!(limits.rendered_tokens_per_request(), 262_144);
        assert_eq!(limits.rendered_tokens_per_batch(), 1_048_576);
        assert_eq!(
            limits.materialized_output_bytes_per_batch(),
            4 * 1024 * 1024 * 1024
        );
    }

    #[test]
    fn callers_can_lower_but_not_silently_raise_defaults() {
        let limits = ResourceLimits::default()
            .lowered(LimitOverrides {
                messages_per_request: Some(1),
                ..LimitOverrides::default()
            })
            .expect("lowering is valid");
        assert_eq!(limits.messages_per_request(), 1);

        let error = ResourceLimits::default()
            .lowered(LimitOverrides {
                messages_per_request: Some(257),
                ..LimitOverrides::default()
            })
            .expect_err("raising needs a named policy");
        assert_eq!(error.category(), ErrorCategory::UnsupportedOption);
    }

    #[test]
    fn boundaries_distinguish_limit_from_overflow() {
        let limits = ResourceLimits::default().lowered(LimitOverrides {
            text_bytes_per_request: Some(4),
            ..LimitOverrides::default()
        });
        let limits = limits.expect("lowered limits");
        let at_boundary = [Message {
            role: Role::User,
            content: MessageContent::Text("1234"),
            tool_calls: &[],
            reasoning_content: None,
        }];
        let over_boundary = [Message {
            content: MessageContent::Text("12345"),
            ..at_boundary[0]
        }];
        let registry = ProfileRegistry::bundled().expect("profiles");
        preflight_batch(
            &registry,
            &[ProfiledRequest {
                profile_alias: "qwen3-vl-8b",
                request: text_request(&at_boundary),
            }],
            limits,
        )
        .expect("limit is inclusive");
        let error = preflight_batch(
            &registry,
            &[ProfiledRequest {
                profile_alias: "qwen3-vl-8b",
                request: text_request(&over_boundary),
            }],
            limits,
        )
        .expect_err("one byte over");
        assert_eq!(error.category(), ErrorCategory::ResourceLimit);

        let error = checked_mul("test product", u64::MAX, 2).expect_err("overflow");
        assert_eq!(error.category(), ErrorCategory::ArithmeticOverflow);
        assert_eq!(
            checked_capacity_bytes(u64::MAX, 2, 8)
                .expect_err("capacity overflow")
                .category(),
            ErrorCategory::ArithmeticOverflow
        );

        let overflow_images = [ImageInput::Rgb8(Rgb8 {
            data: &[],
            height: 2,
            width: usize::MAX,
            row_stride: usize::MAX,
        })];
        let overflow_items = [ContentItem::Image(ImageRef::default())];
        let overflow_messages = [Message {
            role: Role::User,
            content: MessageContent::Items(&overflow_items),
            tool_calls: &[],
            reasoning_content: None,
        }];
        let overflow_request = Request {
            messages: &overflow_messages,
            images: &overflow_images,
            videos: &[],
            options: RequestOptions::default(),
        };
        assert_eq!(
            preflight_batch(
                &registry,
                &[ProfiledRequest {
                    profile_alias: "qwen3-vl-8b",
                    request: overflow_request,
                }],
                ResourceLimits::default(),
            )
            .expect_err("source arithmetic overflows before edge limits")
            .category(),
            ErrorCategory::ArithmeticOverflow
        );
    }

    #[test]
    fn prepared_image_cap_is_distinct_from_decoded_source_pixels() {
        let limits = ResourceLimits::default();
        let between_caps = Rgb8 {
            data: &[],
            height: 4_096,
            width: 4_097,
            row_stride: 4_097 * 3,
        };
        super::validate_rgb_resources(&between_caps, limits, 0)
            .expect("16,781,312 decoded source pixels are below 67,108,864");

        let encoded = [ImageInput::Encoded {
            data: &[1],
            format: crate::request::ImageFormat::Jpeg,
        }];
        let oversized_items = [ContentItem::Image(ImageRef {
            input_index: 0,
            options: ImageOptions {
                resized_height: Some(4_096),
                resized_width: Some(4_097),
                ..ImageOptions::default()
            },
        })];
        let oversized_messages = [Message {
            role: Role::User,
            content: MessageContent::Items(&oversized_items),
            tool_calls: &[],
            reasoning_content: None,
        }];
        let oversized = Request {
            messages: &oversized_messages,
            images: &encoded,
            videos: &[],
            options: RequestOptions::default(),
        };
        let registry = ProfileRegistry::bundled().expect("profiles");
        let error = preflight_batch(
            &registry,
            &[ProfiledRequest {
                profile_alias: "qwen3-vl-8b",
                request: oversized,
            }],
            limits,
        )
        .expect_err("prepared image exceeds the semantic occurrence cap");
        assert_eq!(error.category(), ErrorCategory::ResourceLimit);
        assert_eq!(
            error.context()["resource"].to_string(),
            "prepared_image_pixels_per_occurrence"
        );

        let oversized_budget_items = [ContentItem::Image(ImageRef {
            input_index: 0,
            options: ImageOptions {
                max_pixels: Some(16_777_217),
                ..ImageOptions::default()
            },
        })];
        let oversized_budget_messages = [Message {
            content: MessageContent::Items(&oversized_budget_items),
            ..oversized_messages[0]
        }];
        let oversized_budget = Request {
            messages: &oversized_budget_messages,
            ..oversized
        };
        assert_eq!(
            preflight_batch(
                &registry,
                &[ProfiledRequest {
                    profile_alias: "qwen3-vl-8b",
                    request: oversized_budget,
                }],
                limits,
            )
            .expect_err("image max_pixels cannot bypass the semantic cap")
            .category(),
            ErrorCategory::ResourceLimit
        );
    }

    #[test]
    fn prepared_image_cap_is_not_applied_to_video_options() {
        let frame_bytes = [0_u8; 3];
        let frames = [Rgb8 {
            data: &frame_bytes,
            height: 1,
            width: 1,
            row_stride: 3,
        }];
        let videos = [VideoInput { frames: &frames }];
        let items = [ContentItem::Video(VideoRef {
            input_index: 0,
            options: VideoOptions {
                resized_height: Some(4_096),
                resized_width: Some(4_097),
                max_pixels: Some(16_777_217),
                ..VideoOptions::default()
            },
        })];
        let messages = [Message {
            role: Role::User,
            content: MessageContent::Items(&items),
            tool_calls: &[],
            reasoning_content: None,
        }];
        let request = Request {
            messages: &messages,
            images: &[],
            videos: &videos,
            options: RequestOptions::default(),
        };
        preflight_batch(
            &ProfileRegistry::bundled().expect("profiles"),
            &[ProfiledRequest {
                profile_alias: "qwen3-vl-8b",
                request,
            }],
            ResourceLimits::default(),
        )
        .expect("video policy is independent from the prepared still-image cap");
    }

    #[test]
    fn structure_profile_options_and_resource_precedence_is_stable() {
        let registry = ProfileRegistry::bundled().expect("profiles");
        let empty_messages = Request {
            messages: &[],
            images: &[],
            videos: &[],
            options: RequestOptions {
                excluded: ExcludedOptions {
                    truncation: true,
                    ..ExcludedOptions::default()
                },
                ..RequestOptions::default()
            },
        };
        let error = preflight_batch(
            &registry,
            &[ProfiledRequest {
                profile_alias: "unknown",
                request: empty_messages,
            }],
            ResourceLimits::default()
                .lowered(LimitOverrides {
                    requests_per_batch: Some(0),
                    ..LimitOverrides::default()
                })
                .expect("lower"),
        )
        .expect_err("structure wins");
        assert_eq!(error.category(), ErrorCategory::InvalidRequest);

        let messages = [Message {
            role: Role::User,
            content: MessageContent::Text("x"),
            tool_calls: &[],
            reasoning_content: None,
        }];
        let request = Request {
            messages: &messages,
            ..empty_messages
        };
        let error = preflight_batch(
            &registry,
            &[ProfiledRequest {
                profile_alias: "unknown",
                request,
            }],
            ResourceLimits::default()
                .lowered(LimitOverrides {
                    requests_per_batch: Some(0),
                    ..LimitOverrides::default()
                })
                .expect("lower"),
        )
        .expect_err("profile wins over options and limits");
        assert_eq!(error.category(), ErrorCategory::ProfileMismatch);

        let error = preflight_batch(
            &registry,
            &[ProfiledRequest {
                profile_alias: "qwen3-vl-8b",
                request,
            }],
            ResourceLimits::default()
                .lowered(LimitOverrides {
                    requests_per_batch: Some(0),
                    ..LimitOverrides::default()
                })
                .expect("lower"),
        )
        .expect_err("options win over resources");
        assert_eq!(error.category(), ErrorCategory::UnsupportedOption);
    }

    #[test]
    fn rgb_stride_and_geometry_are_checked_without_output_allocation() {
        let data = [0_u8; 12];
        let images = [ImageInput::Rgb8(Rgb8 {
            data: &data,
            height: 2,
            width: 2,
            row_stride: 6,
        })];
        let items = [ContentItem::Image(ImageRef::default())];
        let messages = [Message {
            role: Role::User,
            content: MessageContent::Items(&items),
            tool_calls: &[],
            reasoning_content: None,
        }];
        let request = Request {
            messages: &messages,
            images: &images,
            videos: &[],
            options: RequestOptions::default(),
        };
        let registry = ProfileRegistry::bundled().expect("profiles");
        let summary = preflight_batch(
            &registry,
            &[ProfiledRequest {
                profile_alias: "qwen3-vl-8b",
                request,
            }],
            ResourceLimits::default(),
        )
        .expect("valid raw RGB");
        assert_eq!(summary.media_occurrences, 1);

        let invalid_images = [ImageInput::Rgb8(Rgb8 {
            row_stride: 5,
            ..match images[0] {
                ImageInput::Rgb8(rgb) => rgb,
                ImageInput::Encoded { .. } => unreachable!(),
            }
        })];
        let invalid_request = Request {
            images: &invalid_images,
            ..request
        };
        let error = preflight_batch(
            &registry,
            &[ProfiledRequest {
                profile_alias: "qwen3-vl-8b",
                request: invalid_request,
            }],
            ResourceLimits::default(),
        )
        .expect_err("short stride");
        assert_eq!(error.category(), ErrorCategory::MediaGeometry);
    }

    #[test]
    fn later_token_and_output_capacity_limits_are_checked() {
        let limits = ResourceLimits::default();
        limits
            .check_rendered_tokens(&[262_144, 1])
            .expect("both token limits permit this");
        assert_eq!(
            limits
                .check_rendered_tokens(&[262_145])
                .expect_err("per request token limit")
                .category(),
            ErrorCategory::ResourceLimit
        );
        assert_eq!(
            limits
                .check_materialized_output_bytes(4 * 1024 * 1024 * 1024 + 1)
                .expect_err("output limit")
                .category(),
            ErrorCategory::ResourceLimit
        );
    }
}
