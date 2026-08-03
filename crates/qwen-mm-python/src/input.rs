//! Python-schema validation and conversion into GIL-independent owned inputs.

use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::{
    buffer::PyBuffer,
    prelude::*,
    types::{PyAny, PyBool, PyDict, PyList, PyString},
};
use qwen_mm_core::{
    BufferClass, ErrorCategory, ExcludedOptions, ImageFormat, ImageOptions, LimitOverrides,
    ObservationRecorder, ObservationScope, QwenError, RequestOptions, ResourceLimits, Role,
    checked_add, checked_mul,
};

use crate::errors::{input_allocation_error, invalid_request, py_detail, type_name};

pub(crate) type BindingResult<T> = Result<T, QwenError>;

#[derive(Debug)]
pub(crate) struct OwnedRequest {
    pub(crate) messages: Vec<OwnedMessage>,
    pub(crate) images: Vec<OwnedImage>,
    pub(crate) options: OwnedRequestOptions,
}

pub(crate) fn drop_owned_media(requests: Vec<OwnedRequest>, recorder: &mut ObservationRecorder) {
    let releases = requests
        .iter()
        .enumerate()
        .flat_map(|(request_index, request)| {
            request
                .images
                .iter()
                .enumerate()
                .map(move |(input_index, image)| {
                    let bytes = match image {
                        OwnedImage::Encoded { data, .. } | OwnedImage::Rgb8 { data, .. } => {
                            data.len()
                        }
                    };
                    (request_index, input_index, bytes)
                })
        })
        .collect::<Vec<_>>();
    drop(requests);
    for (request_index, input_index, bytes) in releases {
        recorder.release_transient(
            "binding.owned_media",
            ObservationScope {
                request_index: Some(request_index),
                message_index: None,
                content_item_index: None,
                media_index: None,
                input_index: Some(input_index),
            },
            u64::try_from(bytes).unwrap_or(u64::MAX),
        );
    }
}

#[derive(Debug)]
pub(crate) struct OwnedMessage {
    pub(crate) role: Role,
    pub(crate) content: OwnedMessageContent,
    pub(crate) tool_calls: Vec<OwnedToolCall>,
    pub(crate) reasoning_content: Option<String>,
}

#[derive(Debug)]
pub(crate) enum OwnedMessageContent {
    Text(String),
    Items(Vec<OwnedContentItem>),
}

#[derive(Debug)]
pub(crate) enum OwnedContentItem {
    Text(String),
    Image {
        input_index: usize,
        options: ImageOptions,
    },
    Video {
        input_index: usize,
    },
}

#[derive(Debug)]
pub(crate) enum OwnedImage {
    Encoded {
        data: Vec<u8>,
        format: ImageFormat,
    },
    Rgb8 {
        data: Vec<u8>,
        height: usize,
        width: usize,
        row_stride: usize,
    },
}

#[derive(Debug)]
pub(crate) struct OwnedToolCall {
    pub(crate) wrapped: bool,
    pub(crate) id: Option<String>,
    pub(crate) name: String,
    pub(crate) arguments_json: String,
}

#[derive(Debug, Default)]
pub(crate) struct OwnedRequestOptions {
    pub(crate) add_generation_prompt: bool,
    pub(crate) add_vision_id: bool,
    pub(crate) tools_json: Vec<String>,
    pub(crate) enable_thinking: Option<bool>,
    pub(crate) excluded: OwnedExcludedOptions,
}

#[derive(Debug, Default)]
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct OwnedExcludedOptions {
    pub(crate) chat_template: Option<String>,
    pub(crate) documents_json: Option<String>,
    pub(crate) continue_final_message: bool,
    pub(crate) assistant_token_mask: bool,
    pub(crate) truncation: bool,
    pub(crate) left_padding: bool,
    pub(crate) pad_token_id: Option<i64>,
    pub(crate) load_audio_from_video: bool,
    pub(crate) audio_inputs: usize,
}

impl OwnedRequestOptions {
    pub(crate) fn borrowed<'a>(
        &'a self,
        tools: &'a [qwen_mm_core::ToolDefinition<'a>],
    ) -> RequestOptions<'a> {
        RequestOptions {
            add_generation_prompt: self.add_generation_prompt,
            add_vision_id: self.add_vision_id,
            tools,
            enable_thinking: self.enable_thinking,
            excluded: ExcludedOptions {
                chat_template: self.excluded.chat_template.as_deref(),
                documents_json: self.excluded.documents_json.as_deref(),
                continue_final_message: self.excluded.continue_final_message,
                assistant_token_mask: self.excluded.assistant_token_mask,
                truncation: self.excluded.truncation,
                left_padding: self.excluded.left_padding,
                pad_token_id: self.excluded.pad_token_id,
                load_audio_from_video: self.excluded.load_audio_from_video,
                audio_inputs: self.excluded.audio_inputs,
            },
        }
    }
}

pub(crate) fn parse_requests(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    limits: ResourceLimits,
    supports_thinking: bool,
    profile_alias: &str,
) -> BindingResult<Vec<OwnedRequest>> {
    parse_requests_internal(py, value, limits, supports_thinking, profile_alias, None)
}

pub(crate) fn parse_requests_observed(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    limits: ResourceLimits,
    supports_thinking: bool,
    profile_alias: &str,
    recorder: &mut ObservationRecorder,
) -> BindingResult<Vec<OwnedRequest>> {
    parse_requests_internal(
        py,
        value,
        limits,
        supports_thinking,
        profile_alias,
        Some(recorder),
    )
}

fn parse_requests_internal(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    limits: ResourceLimits,
    supports_thinking: bool,
    profile_alias: &str,
    mut recorder: Option<&mut ObservationRecorder>,
) -> BindingResult<Vec<OwnedRequest>> {
    let list = as_list(value, "requests")?;
    if list.is_empty() {
        return Err(invalid_request("batch must contain at least one request"));
    }
    let values = list.iter().collect::<Vec<_>>();
    let mut requests = values
        .iter()
        .enumerate()
        .map(|(request_index, request)| parse_request_structure(py, request, request_index))
        .collect::<BindingResult<Vec<_>>>()?;
    for (request_index, (request, value)) in requests.iter().zip(&values).enumerate() {
        let dict = as_dict(value, "request")?;
        let image_count = optional(dict, "images")?
            .map(|images| as_list(&images, "request.images").map(pyo3::types::PyListMethods::len))
            .transpose()?
            .unwrap_or(0);
        let video_count = optional(dict, "videos")?
            .map(|videos| as_list(&videos, "request.videos").map(pyo3::types::PyListMethods::len))
            .transpose()?
            .unwrap_or(0);
        validate_binding_structure(request, image_count, video_count, request_index)?;
    }
    for (request_index, (request, value)) in requests.iter_mut().zip(&values).enumerate() {
        let dict = as_dict(value, "request")?;
        request.options = match optional(dict, "options")? {
            Some(options) => parse_options(py, &options, request_index)?,
            None => OwnedRequestOptions::default(),
        };
    }
    validate_options(&requests, supports_thinking, profile_alias)?;
    preflight_python_resources(py, &values, &requests, limits)?;
    for (request_index, value) in values.iter().enumerate() {
        if has_key(as_dict(value, "request")?, "videos")? {
            return Err(QwenError::new(
                ErrorCategory::UnsupportedMedia,
                "video inputs are outside the Phase C Python binding",
            )
            .with_context("request_index", request_index));
        }
    }
    reject_video_items(&requests)?;
    for (request_index, (request, value)) in requests.iter_mut().zip(&values).enumerate() {
        let dict = as_dict(value, "request")?;
        let mut parsed = Vec::new();
        if let Some(images) = optional(dict, "images")? {
            for (image_index, image) in as_list(&images, "request.images")?.iter().enumerate() {
                let image = parse_image(py, &image, request_index, image_index)?;
                if let Some(recorder) = recorder.as_deref_mut() {
                    record_owned_image(&image, request_index, image_index, recorder);
                }
                parsed.push(image);
            }
        }
        request.images = parsed;
    }
    Ok(requests)
}

fn record_owned_image(
    image: &OwnedImage,
    request_index: usize,
    input_index: usize,
    recorder: &mut ObservationRecorder,
) {
    let bytes = match image {
        OwnedImage::Encoded { data, .. } | OwnedImage::Rgb8 { data, .. } => data.len(),
    };
    let bytes = u64::try_from(bytes).unwrap_or(u64::MAX);
    let scope = ObservationScope {
        request_index: Some(request_index),
        message_index: None,
        content_item_index: None,
        media_index: None,
        input_index: Some(input_index),
    };
    recorder.record_allocation("binding.owned_media", BufferClass::Transient, scope, bytes);
    recorder.record_copy("binding.owned_media", scope, bytes);
}

fn parse_request_structure(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    request_index: usize,
) -> BindingResult<OwnedRequest> {
    let dict = as_dict(value, "request")?;
    reject_unknown_keys(
        dict,
        &["messages", "images", "options", "videos"],
        "request",
    )?;
    let messages = as_list(&required(dict, "messages", "request")?, "request.messages")?
        .iter()
        .enumerate()
        .map(|(message_index, message)| parse_message(py, &message, request_index, message_index))
        .collect::<BindingResult<Vec<_>>>()?;
    if let Some(images) = optional(dict, "images")? {
        for (image_index, image) in as_list(&images, "request.images")?.iter().enumerate() {
            validate_image_schema(py, &image, request_index, image_index)?;
        }
    }
    if let Some(videos) = optional(dict, "videos")? {
        let _ = as_list(&videos, "request.videos")?;
    }
    Ok(OwnedRequest {
        messages,
        images: Vec::new(),
        options: OwnedRequestOptions::default(),
    })
}

#[allow(clippy::too_many_lines)]
fn validate_binding_structure(
    request: &OwnedRequest,
    image_count: usize,
    video_count: usize,
    request_index: usize,
) -> BindingResult<()> {
    if request.messages.is_empty() {
        return Err(invalid_request("request must contain at least one message")
            .with_context("request_index", request_index));
    }
    let mut referenced = vec![false; image_count];
    let mut referenced_videos = vec![false; video_count];
    let mut saw_system = false;
    for (message_index, message) in request.messages.iter().enumerate() {
        if message.role == Role::System {
            if saw_system || message_index != 0 {
                return Err(
                    invalid_request("system message must appear at most once and first")
                        .with_context("request_index", request_index)
                        .with_context("message_index", message_index),
                );
            }
            saw_system = true;
        }
        if message.role != Role::Assistant && !message.tool_calls.is_empty() {
            return Err(
                invalid_request("tool calls are only valid on assistant messages")
                    .with_context("request_index", request_index)
                    .with_context("message_index", message_index),
            );
        }
        if message.role != Role::Assistant && message.reasoning_content.is_some() {
            return Err(
                invalid_request("reasoning content is only valid on assistant messages")
                    .with_context("request_index", request_index)
                    .with_context("message_index", message_index),
            );
        }
        if let OwnedMessageContent::Items(items) = &message.content {
            if items.is_empty() {
                return Err(invalid_request("message content list must not be empty")
                    .with_context("request_index", request_index)
                    .with_context("message_index", message_index));
            }
            for (content_item_index, item) in items.iter().enumerate() {
                match item {
                    OwnedContentItem::Text(_) => {}
                    OwnedContentItem::Image { input_index, .. } => {
                        if message.role == Role::System {
                            return Err(invalid_request("system messages cannot contain visuals")
                                .with_context("request_index", request_index)
                                .with_context("message_index", message_index)
                                .with_context("content_item_index", content_item_index));
                        }
                        if message.role == Role::Assistant {
                            return Err(invalid_request(
                                "assistant content only supports text items",
                            )
                            .with_context("request_index", request_index)
                            .with_context("message_index", message_index)
                            .with_context("content_item_index", content_item_index));
                        }
                        let Some(slot) = referenced.get_mut(*input_index) else {
                            return Err(invalid_request("image reference index is out of range")
                                .with_context("request_index", request_index)
                                .with_context("message_index", message_index)
                                .with_context("content_item_index", content_item_index)
                                .with_context("input_index", *input_index));
                        };
                        *slot = true;
                    }
                    OwnedContentItem::Video { input_index } => {
                        if message.role == Role::System {
                            return Err(invalid_request("system messages cannot contain visuals")
                                .with_context("request_index", request_index)
                                .with_context("message_index", message_index)
                                .with_context("content_item_index", content_item_index));
                        }
                        if message.role == Role::Assistant {
                            return Err(invalid_request(
                                "assistant content only supports text items",
                            )
                            .with_context("request_index", request_index)
                            .with_context("message_index", message_index)
                            .with_context("content_item_index", content_item_index));
                        }
                        let Some(slot) = referenced_videos.get_mut(*input_index) else {
                            return Err(invalid_request("video reference index is out of range")
                                .with_context("request_index", request_index)
                                .with_context("message_index", message_index)
                                .with_context("content_item_index", content_item_index)
                                .with_context("input_index", *input_index));
                        };
                        *slot = true;
                    }
                }
            }
        }
    }
    if let Some(input_index) = referenced.iter().position(|referenced| !referenced) {
        return Err(
            invalid_request("every supplied image input must be referenced")
                .with_context("request_index", request_index)
                .with_context("input_index", input_index),
        );
    }
    if let Some(input_index) = referenced_videos.iter().position(|referenced| !referenced) {
        return Err(
            invalid_request("every supplied video input must be referenced")
                .with_context("request_index", request_index)
                .with_context("input_index", input_index),
        );
    }
    Ok(())
}

fn validate_options(
    requests: &[OwnedRequest],
    supports_thinking: bool,
    profile_alias: &str,
) -> BindingResult<()> {
    for (request_index, request) in requests.iter().enumerate() {
        let excluded = &request.options.excluded;
        let option = if excluded.chat_template.is_some() {
            Some("chat_template")
        } else if excluded.documents_json.is_some() {
            Some("documents")
        } else if excluded.continue_final_message {
            Some("continue_final_message")
        } else if excluded.assistant_token_mask {
            Some("assistant_token_mask")
        } else if excluded.load_audio_from_video || excluded.audio_inputs != 0 {
            Some("audio")
        } else if excluded.truncation {
            Some("truncation")
        } else if excluded.left_padding {
            Some("left_padding")
        } else if excluded.pad_token_id.is_some() {
            Some("pad_token_id")
        } else {
            None
        };
        if let Some(option) = option {
            return Err(QwenError::new(
                ErrorCategory::UnsupportedOption,
                "option is outside compatibility contract v1",
            )
            .with_context("option", option)
            .with_context("request_index", request_index));
        }
        if request.options.enable_thinking.is_some() {
            if !supports_thinking {
                return Err(QwenError::new(
                    ErrorCategory::UnsupportedOption,
                    "thinking is not supported by this profile",
                )
                .with_context("option", "enable_thinking")
                .with_context("profile", profile_alias)
                .with_context("request_index", request_index));
            }
            if !request.options.add_generation_prompt {
                return Err(
                    invalid_request("enable_thinking requires add_generation_prompt")
                        .with_context("request_index", request_index),
                );
            }
        }
        if !supports_thinking
            && request
                .messages
                .iter()
                .any(|message| message.reasoning_content.is_some())
        {
            return Err(QwenError::new(
                ErrorCategory::UnsupportedOption,
                "reasoning content is not supported by this profile",
            )
            .with_context("option", "reasoning_content")
            .with_context("profile", profile_alias)
            .with_context("request_index", request_index));
        }
    }
    Ok(())
}

fn reject_video_items(requests: &[OwnedRequest]) -> BindingResult<()> {
    for (request_index, request) in requests.iter().enumerate() {
        for (message_index, message) in request.messages.iter().enumerate() {
            let OwnedMessageContent::Items(items) = &message.content else {
                continue;
            };
            for (content_item_index, item) in items.iter().enumerate() {
                if matches!(item, OwnedContentItem::Video { .. }) {
                    return Err(QwenError::new(
                        ErrorCategory::UnsupportedMedia,
                        "video is outside the Phase C Python binding",
                    )
                    .with_context("request_index", request_index)
                    .with_context("message_index", message_index)
                    .with_context("content_item_index", content_item_index));
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn preflight_python_resources(
    py: Python<'_>,
    values: &[Bound<'_, PyAny>],
    requests: &[OwnedRequest],
    limits: ResourceLimits,
) -> BindingResult<()> {
    let request_count = to_u64("request count", requests.len())?;
    check_resource(
        "requests_per_batch",
        request_count,
        limits.requests_per_batch(),
        None,
    )?;
    let mut batch_media = 0_u64;
    let mut batch_encoded_bytes = 0_u64;
    for (request_index, (request, value)) in requests.iter().zip(values).enumerate() {
        check_resource(
            "messages_per_request",
            to_u64("message count", request.messages.len())?,
            limits.messages_per_request(),
            Some(request_index),
        )?;
        let mut content_items = 0_u64;
        let mut text_bytes = 0_u64;
        let mut media = 0_u64;
        for message in &request.messages {
            match &message.content {
                OwnedMessageContent::Text(text) => {
                    text_bytes = checked_add(
                        "request text byte count",
                        text_bytes,
                        to_u64("text bytes", text.len())?,
                    )?;
                }
                OwnedMessageContent::Items(items) => {
                    content_items = checked_add(
                        "request content item count",
                        content_items,
                        to_u64("content item count", items.len())?,
                    )?;
                    for item in items {
                        match item {
                            OwnedContentItem::Text(text) => {
                                text_bytes = checked_add(
                                    "request text byte count",
                                    text_bytes,
                                    to_u64("text bytes", text.len())?,
                                )?;
                            }
                            OwnedContentItem::Image { .. } | OwnedContentItem::Video { .. } => {
                                media = checked_add("request media occurrence count", media, 1)?;
                            }
                        }
                    }
                }
            }
            if let Some(reasoning) = &message.reasoning_content {
                text_bytes = checked_add(
                    "request reasoning byte count",
                    text_bytes,
                    to_u64("reasoning bytes", reasoning.len())?,
                )?;
            }
            for call in &message.tool_calls {
                for bytes in [
                    call.id.as_ref().map_or(0, String::len),
                    call.name.len(),
                    call.arguments_json.len(),
                ] {
                    text_bytes = checked_add(
                        "request tool-call byte count",
                        text_bytes,
                        to_u64("tool-call bytes", bytes)?,
                    )?;
                }
            }
        }
        for tool in &request.options.tools_json {
            text_bytes = checked_add(
                "request tool definition byte count",
                text_bytes,
                to_u64("tool definition bytes", tool.len())?,
            )?;
        }
        for (name, actual, limit) in [
            (
                "content_items_per_request",
                content_items,
                limits.content_items_per_request(),
            ),
            (
                "text_bytes_per_request",
                text_bytes,
                limits.text_bytes_per_request(),
            ),
            ("media_per_request", media, limits.media_per_request()),
        ] {
            check_resource(name, actual, limit, Some(request_index))?;
        }
        batch_media = checked_add("batch media occurrence count", batch_media, media)?;

        let dict = as_dict(value, "request")?;
        if let Some(images) = optional(dict, "images")? {
            for (image_index, image) in as_list(&images, "request.images")?.iter().enumerate() {
                match inspect_image_resources(py, &image, request_index, image_index)? {
                    ImageResource::Encoded(bytes) => {
                        check_resource(
                            "encoded_bytes_per_item",
                            bytes,
                            limits.encoded_bytes_per_item(),
                            Some(request_index),
                        )?;
                        batch_encoded_bytes =
                            checked_add("batch encoded bytes", batch_encoded_bytes, bytes)?;
                    }
                    ImageResource::Raw { height, width } => {
                        let pixels = checked_mul("decoded source pixels", height, width)?;
                        check_resource(
                            "decoded_source_pixels",
                            pixels,
                            limits.decoded_pixels_per_image_or_frame(),
                            Some(request_index),
                        )?;
                        check_resource(
                            "decoded_edge_length",
                            height,
                            limits.decoded_edge_length(),
                            Some(request_index),
                        )?;
                        check_resource(
                            "decoded_edge_length",
                            width,
                            limits.decoded_edge_length(),
                            Some(request_index),
                        )?;
                    }
                    ImageResource::UnsupportedObject => {}
                }
            }
        }
    }
    check_resource(
        "media_per_batch",
        batch_media,
        limits.media_per_batch(),
        None,
    )?;
    check_resource(
        "encoded_bytes_per_batch",
        batch_encoded_bytes,
        limits.encoded_bytes_per_batch(),
        None,
    )
}

enum ImageResource {
    Encoded(u64),
    Raw { height: u64, width: u64 },
    UnsupportedObject,
}

fn inspect_image_resources(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    request_index: usize,
    image_index: usize,
) -> BindingResult<ImageResource> {
    if let Ok(array) = value.cast::<PyUntypedArray>() {
        if array.ndim() == 1 {
            return encoded_buffer_length(py, value, request_index, image_index)
                .map(ImageResource::Encoded);
        }
        let shape = array.shape();
        return Ok(ImageResource::Raw {
            height: shape.first().copied().map_or(0, |value| value as u64),
            width: shape.get(1).copied().map_or(0, |value| value as u64),
        });
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        if has_key(dict, "rgb")? {
            let rgb = required(dict, "rgb", "raw RGB image")?;
            let array = rgb.cast::<PyUntypedArray>().map_err(|_| {
                invalid_request("raw RGB image must contain a NumPy array")
                    .with_context("request_index", request_index)
                    .with_context("image_index", image_index)
            })?;
            let shape = array.shape();
            return Ok(ImageResource::Raw {
                height: shape.first().copied().map_or(0, |value| value as u64),
                width: shape.get(1).copied().map_or(0, |value| value as u64),
            });
        }
        return encoded_buffer_length(
            py,
            &required(dict, "data", "encoded image")?,
            request_index,
            image_index,
        )
        .map(ImageResource::Encoded);
    }
    if is_recognized_unsupported_media_object(value) {
        return Ok(ImageResource::UnsupportedObject);
    }
    encoded_buffer_length(py, value, request_index, image_index).map(ImageResource::Encoded)
}

fn check_resource(
    name: &'static str,
    actual: u64,
    limit: u64,
    request_index: Option<usize>,
) -> BindingResult<()> {
    if actual <= limit {
        return Ok(());
    }
    let mut error = QwenError::new(ErrorCategory::ResourceLimit, "resource limit exceeded")
        .with_context("limit_name", name)
        .with_context("actual", actual)
        .with_context("limit", limit);
    if let Some(request_index) = request_index {
        error = error.with_context("request_index", request_index);
    }
    Err(error)
}

fn to_u64(operation: &'static str, value: usize) -> BindingResult<u64> {
    u64::try_from(value).map_err(|_| {
        QwenError::new(
            ErrorCategory::ArithmeticOverflow,
            "platform-sized value does not fit the compatibility counter",
        )
        .with_context("operation", operation)
    })
}

fn parse_message(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    request_index: usize,
    message_index: usize,
) -> BindingResult<OwnedMessage> {
    let dict = as_dict(value, "message")?;
    reject_unknown_keys(
        dict,
        &["role", "content", "tool_calls", "reasoning_content"],
        "message",
    )?;
    let role_text = extract_string(&required(dict, "role", "message")?, "message.role")?;
    let role = match role_text.as_str() {
        "system" => Role::System,
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        _ => {
            return Err(invalid_request("message role is not supported")
                .with_context("role", role_text)
                .with_context("request_index", request_index)
                .with_context("message_index", message_index));
        }
    };
    let content_value = required(dict, "content", "message")?;
    let content = if content_value.is_instance_of::<PyString>() {
        OwnedMessageContent::Text(extract_string(&content_value, "message.content")?)
    } else {
        let items = as_list(&content_value, "message.content")?
            .iter()
            .enumerate()
            .map(|(content_index, item)| {
                parse_content_item(&item, request_index, message_index, content_index)
            })
            .collect::<BindingResult<Vec<_>>>()?;
        OwnedMessageContent::Items(items)
    };
    let tool_calls = match optional(dict, "tool_calls")? {
        Some(calls) => as_list(&calls, "message.tool_calls")?
            .iter()
            .enumerate()
            .map(|(tool_index, call)| {
                parse_tool_call(py, &call, request_index, message_index, tool_index)
            })
            .collect::<BindingResult<Vec<_>>>()?,
        None => Vec::new(),
    };
    let reasoning_content = optional_string(dict, "reasoning_content", "message")?;
    Ok(OwnedMessage {
        role,
        content,
        tool_calls,
        reasoning_content,
    })
}

fn parse_content_item(
    value: &Bound<'_, PyAny>,
    request_index: usize,
    message_index: usize,
    content_index: usize,
) -> BindingResult<OwnedContentItem> {
    let dict = as_dict(value, "content item")?;
    let kind = extract_string(
        &required(dict, "type", "content item")?,
        "content item.type",
    )?;
    match kind.as_str() {
        "text" => {
            reject_unknown_keys(dict, &["type", "text"], "text content item")?;
            Ok(OwnedContentItem::Text(extract_string(
                &required(dict, "text", "text content item")?,
                "content item.text",
            )?))
        }
        "image" => {
            reject_unknown_keys(
                dict,
                &[
                    "type",
                    "input_index",
                    "image",
                    "buffer_index",
                    "options",
                    "min_pixels",
                    "max_pixels",
                    "resized_height",
                    "resized_width",
                ],
                "image content item",
            )?;
            let aliases = present_keys(dict, &["input_index", "image", "buffer_index"])?;
            if aliases.len() != 1 {
                return Err(invalid_request(
                    "image content item requires exactly one input reference field",
                )
                .with_context("actual_fields", aliases.join(",")));
            }
            let input_index = extract_usize(
                &required(dict, aliases[0], "image content item")?,
                "content item.input_index",
            )?;
            let option_dict = optional(dict, "options")?;
            let inline_options = present_keys(
                dict,
                &[
                    "min_pixels",
                    "max_pixels",
                    "resized_height",
                    "resized_width",
                ],
            )?;
            if option_dict.is_some() && !inline_options.is_empty() {
                return Err(invalid_request(
                    "image options cannot be supplied both inline and in options",
                ));
            }
            let options = match option_dict.as_ref() {
                Some(options) => parse_image_options(as_dict(options, "image options")?)?,
                None => extract_image_options(dict)?,
            };
            Ok(OwnedContentItem::Image {
                input_index,
                options,
            })
        }
        "video" => {
            reject_unknown_keys(
                dict,
                &["type", "input_index", "video", "buffer_index", "options"],
                "video content item",
            )?;
            let aliases = present_keys(dict, &["input_index", "video", "buffer_index"])?;
            if aliases.len() != 1 {
                return Err(invalid_request(
                    "video content item requires exactly one input reference field",
                ));
            }
            Ok(OwnedContentItem::Video {
                input_index: extract_usize(
                    &required(dict, aliases[0], "video content item")?,
                    "content item.input_index",
                )?,
            })
        }
        _ => Err(invalid_request("content item type is not supported")
            .with_context("type", kind)
            .with_context("request_index", request_index)
            .with_context("message_index", message_index)
            .with_context("content_item_index", content_index)),
    }
}

fn parse_image_options(dict: &Bound<'_, PyDict>) -> BindingResult<ImageOptions> {
    reject_unknown_keys(
        dict,
        &[
            "min_pixels",
            "max_pixels",
            "resized_height",
            "resized_width",
        ],
        "image options",
    )?;
    extract_image_options(dict)
}

fn extract_image_options(dict: &Bound<'_, PyDict>) -> BindingResult<ImageOptions> {
    Ok(ImageOptions {
        min_pixels: optional_u64(dict, "min_pixels", "image options")?,
        max_pixels: optional_u64(dict, "max_pixels", "image options")?,
        resized_height: optional_u64(dict, "resized_height", "image options")?,
        resized_width: optional_u64(dict, "resized_width", "image options")?,
    })
}

fn validate_image_schema(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    request_index: usize,
    image_index: usize,
) -> BindingResult<()> {
    if value.cast::<PyUntypedArray>().is_ok() || is_recognized_unsupported_media_object(value) {
        return Ok(());
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        let has_rgb = has_key(dict, "rgb")?;
        let has_data = has_key(dict, "data")?;
        let has_format = has_key(dict, "format")?;
        if has_rgb {
            reject_unknown_keys(dict, &["rgb"], "raw RGB image")?;
            if has_data || has_format {
                return Err(
                    invalid_request("raw RGB and encoded image fields cannot be combined")
                        .with_context("request_index", request_index)
                        .with_context("image_index", image_index),
                );
            }
            let rgb = required(dict, "rgb", "raw RGB image")?;
            if rgb.cast::<PyUntypedArray>().is_err() {
                return Err(invalid_request("raw RGB image must contain a NumPy array")
                    .with_context("request_index", request_index)
                    .with_context("image_index", image_index));
            }
            return Ok(());
        }
        reject_unknown_keys(dict, &["data", "format"], "encoded image")?;
        let data = required(dict, "data", "encoded image")?;
        let _ = extract_string(
            &required(dict, "format", "encoded image")?,
            "encoded image.format",
        )?;
        encoded_buffer_length(py, &data, request_index, image_index)?;
        return Ok(());
    }
    encoded_buffer_length(py, value, request_index, image_index).map(drop)
}

fn parse_image(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    request_index: usize,
    image_index: usize,
) -> BindingResult<OwnedImage> {
    if value
        .cast::<PyUntypedArray>()
        .is_ok_and(|array| array.ndim() != 1)
    {
        return parse_rgb_array(value, request_index, image_index);
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        let has_rgb = has_key(dict, "rgb")?;
        let has_data = has_key(dict, "data")?;
        let has_format = has_key(dict, "format")?;
        if has_rgb {
            reject_unknown_keys(dict, &["rgb"], "raw RGB image")?;
            if has_data || has_format {
                return Err(invalid_request(
                    "raw RGB and encoded image fields cannot be combined",
                ));
            }
            let rgb = required(dict, "rgb", "raw RGB image")?;
            return parse_rgb_array(&rgb, request_index, image_index);
        }
        reject_unknown_keys(dict, &["data", "format"], "encoded image")?;
        let data = required(dict, "data", "encoded image")?;
        let format = extract_string(
            &required(dict, "format", "encoded image")?,
            "encoded image.format",
        )?;
        let format = parse_image_format(&format, request_index, image_index)?;
        return Ok(OwnedImage::Encoded {
            data: copy_encoded_buffer(py, &data, request_index, image_index)?,
            format,
        });
    }
    if is_recognized_unsupported_media_object(value) {
        return Err(QwenError::new(
            ErrorCategory::UnsupportedMedia,
            "Python media source object kind is not supported",
        )
        .with_context("actual_type", type_name(value))
        .with_context("request_index", request_index)
        .with_context("image_index", image_index));
    }
    let data = copy_encoded_buffer(py, value, request_index, image_index)?;
    let format = infer_image_format(&data).ok_or_else(|| {
        QwenError::new(
            ErrorCategory::UnsupportedMedia,
            "bare encoded buffer does not contain a supported media signature",
        )
        .with_context("request_index", request_index)
        .with_context("image_index", image_index)
    })?;
    Ok(OwnedImage::Encoded { data, format })
}

fn parse_rgb_array(
    value: &Bound<'_, PyAny>,
    request_index: usize,
    image_index: usize,
) -> BindingResult<OwnedImage> {
    let array = value.cast::<PyArrayDyn<u8>>().map_err(|error| {
        QwenError::new(
            ErrorCategory::UnsupportedMedia,
            "raw RGB input must use NumPy uint8 storage",
        )
        .with_context("detail", error.to_string())
        .with_context("request_index", request_index)
        .with_context("image_index", image_index)
    })?;
    if array.ndim() != 3 {
        return Err(QwenError::new(
            ErrorCategory::UnsupportedMedia,
            "raw RGB array must have rank 3 [height, width, channels]",
        )
        .with_context("actual_rank", array.ndim())
        .with_context("request_index", request_index)
        .with_context("image_index", image_index));
    }
    let shape = array.shape();
    if shape[0] == 0 || shape[1] == 0 {
        return Err(QwenError::new(
            ErrorCategory::MediaGeometry,
            "raw RGB array dimensions must be non-zero",
        )
        .with_context("request_index", request_index)
        .with_context("image_index", image_index));
    }
    if shape[2] != 3 {
        return Err(QwenError::new(
            ErrorCategory::UnsupportedMedia,
            "raw RGB array must have exactly three channels",
        )
        .with_context("actual_channels", shape[2])
        .with_context("request_index", request_index)
        .with_context("image_index", image_index));
    }
    if !array.is_aligned() {
        return Err(QwenError::new(
            ErrorCategory::MediaGeometry,
            "raw RGB array must be aligned",
        )
        .with_context("request_index", request_index)
        .with_context("image_index", image_index));
    }
    let row_stride = shape[1].checked_mul(3).ok_or_else(|| {
        QwenError::new(
            ErrorCategory::ArithmeticOverflow,
            "raw RGB row stride overflowed",
        )
        .with_context("request_index", request_index)
        .with_context("image_index", image_index)
    })?;
    let strides = array.strides();
    if strides.iter().any(|stride| *stride <= 0)
        || strides[1] != 3
        || strides[2] != 1
        || usize::try_from(strides[0]).map_or(true, |stride| stride < row_stride)
    {
        return Err(QwenError::new(
            ErrorCategory::MediaGeometry,
            "raw RGB array must use positive HWC strides with packed RGB pixels and optional row padding",
        )
        .with_context("request_index", request_index)
        .with_context("image_index", image_index));
    }
    let data_len = shape[0].checked_mul(row_stride).ok_or_else(|| {
        QwenError::new(
            ErrorCategory::ArithmeticOverflow,
            "raw RGB byte length overflowed",
        )
        .with_context("request_index", request_index)
        .with_context("image_index", image_index)
    })?;
    let readonly = array.try_readonly().map_err(|error| {
        invalid_request("raw RGB array is already mutably borrowed")
            .with_context("detail", error.to_string())
            .with_context("request_index", request_index)
            .with_context("image_index", image_index)
    })?;
    let mut data = Vec::new();
    data.try_reserve_exact(data_len)
        .map_err(|_| input_allocation_error("raw RGB input", data_len))?;
    for row in readonly.as_array().outer_iter() {
        let source = row.as_slice().ok_or_else(|| {
            QwenError::new(
                ErrorCategory::MediaGeometry,
                "raw RGB array does not expose packed logical rows",
            )
            .with_context("request_index", request_index)
            .with_context("image_index", image_index)
        })?;
        data.extend_from_slice(source);
    }
    debug_assert_eq!(data.len(), data_len);
    Ok(OwnedImage::Rgb8 {
        data,
        height: shape[0],
        width: shape[1],
        row_stride,
    })
}

fn copy_encoded_buffer(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    request_index: usize,
    image_index: usize,
) -> BindingResult<Vec<u8>> {
    let buffer = PyBuffer::<u8>::get(value).map_err(|error| {
        invalid_request("encoded image must expose an unsigned-byte buffer")
            .with_context("detail", py_detail(&error))
            .with_context("actual_type", type_name(value))
            .with_context("request_index", request_index)
            .with_context("image_index", image_index)
    })?;
    if buffer.dimensions() != 1 || !buffer.is_c_contiguous() {
        return Err(invalid_request(
            "encoded image buffer must be one-dimensional and C-contiguous",
        )
        .with_context("request_index", request_index)
        .with_context("image_index", image_index));
    }
    let mut data = Vec::new();
    data.try_reserve_exact(buffer.item_count())
        .map_err(|_| input_allocation_error("encoded image input", buffer.item_count()))?;
    data.resize(buffer.item_count(), 0);
    buffer.copy_to_slice(py, &mut data).map_err(|error| {
        invalid_request("encoded image buffer could not be copied safely")
            .with_context("detail", py_detail(&error))
            .with_context("request_index", request_index)
            .with_context("image_index", image_index)
    })?;
    Ok(data)
}

fn encoded_buffer_length(
    _py: Python<'_>,
    value: &Bound<'_, PyAny>,
    request_index: usize,
    image_index: usize,
) -> BindingResult<u64> {
    let buffer = PyBuffer::<u8>::get(value).map_err(|error| {
        invalid_request("encoded image must expose an unsigned-byte buffer")
            .with_context("detail", py_detail(&error))
            .with_context("actual_type", type_name(value))
            .with_context("request_index", request_index)
            .with_context("image_index", image_index)
    })?;
    if buffer.dimensions() != 1 || !buffer.is_c_contiguous() {
        return Err(invalid_request(
            "encoded image buffer must be one-dimensional and C-contiguous",
        )
        .with_context("request_index", request_index)
        .with_context("image_index", image_index));
    }
    to_u64("encoded image bytes", buffer.len_bytes())
}

fn parse_tool_call(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    request_index: usize,
    message_index: usize,
    tool_index: usize,
) -> BindingResult<OwnedToolCall> {
    let outer = as_dict(value, "tool call")?;
    let function = optional(outer, "function")?;
    let (wrapped, call) = if let Some(function) = function.as_ref() {
        reject_unknown_keys(outer, &["id", "type", "function"], "wrapped tool call")?;
        if let Some(kind) = optional_string(outer, "type", "tool call.type")?
            && kind != "function"
        {
            return Err(invalid_request("wrapped tool call type must be function")
                .with_context("request_index", request_index)
                .with_context("message_index", message_index)
                .with_context("tool_index", tool_index));
        }
        let call = as_dict(function, "tool call.function")?;
        reject_unknown_keys(call, &["name", "arguments"], "tool call.function")?;
        (true, call)
    } else {
        reject_unknown_keys(outer, &["id", "name", "arguments"], "direct tool call")?;
        (false, outer)
    };
    let id = optional_string(outer, "id", "tool call")?;
    let name = extract_string(&required(call, "name", "tool call")?, "tool call.name")?;
    let arguments = required(call, "arguments", "tool call")?;
    let arguments_json = if arguments.is_instance_of::<PyString>() {
        let encoded = extract_string(&arguments, "tool call.arguments")?;
        validate_json_string(py, &encoded, "tool call arguments", false)?;
        encoded
    } else {
        json_dumps(py, &arguments, "tool call arguments")?
    };
    if name.is_empty() {
        return Err(invalid_request("tool call name must not be empty")
            .with_context("request_index", request_index)
            .with_context("message_index", message_index)
            .with_context("tool_index", tool_index));
    }
    Ok(OwnedToolCall {
        wrapped,
        id,
        name,
        arguments_json,
    })
}

fn parse_options(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    request_index: usize,
) -> BindingResult<OwnedRequestOptions> {
    let dict = as_dict(value, "request.options")?;
    reject_unknown_keys(
        dict,
        &[
            "add_generation_prompt",
            "add_vision_id",
            "tools",
            "enable_thinking",
            "chat_template",
            "documents",
            "continue_final_message",
            "assistant_token_mask",
            "truncation",
            "left_padding",
            "pad_token_id",
            "load_audio_from_video",
            "audio_inputs",
        ],
        "request.options",
    )
    .map_err(|error| error.with_context("request_index", request_index))?;
    let tools_json = match optional(dict, "tools")? {
        Some(tools) => as_list(&tools, "request.options.tools")?
            .iter()
            .enumerate()
            .map(|(tool_index, tool)| {
                let encoded = if tool.is_instance_of::<PyString>() {
                    extract_string(&tool, "tool definition")?
                } else {
                    json_dumps(py, &tool, "tool definition")?
                };
                validate_json_string(py, &encoded, "tool definition", true).map_err(|error| {
                    error
                        .with_context("request_index", request_index)
                        .with_context("tool_index", tool_index)
                })?;
                Ok(encoded)
            })
            .collect::<BindingResult<Vec<_>>>()?,
        None => Vec::new(),
    };
    let documents_json = match optional(dict, "documents")? {
        Some(documents) if documents.is_instance_of::<PyString>() => {
            Some(extract_string(&documents, "request.options.documents")?)
        }
        Some(documents) => Some(json_dumps(py, &documents, "request.options.documents")?),
        None => None,
    };
    let audio_inputs = match optional(dict, "audio_inputs")? {
        Some(value) => value.cast::<PyList>().map_or_else(
            |_| extract_usize(&value, "request.options.audio_inputs"),
            |items| Ok(items.len()),
        )?,
        None => 0,
    };
    let options = OwnedRequestOptions {
        add_generation_prompt: optional_bool(dict, "add_generation_prompt", "request.options")?
            .unwrap_or(false),
        add_vision_id: optional_bool(dict, "add_vision_id", "request.options")?.unwrap_or(false),
        tools_json,
        enable_thinking: optional_bool(dict, "enable_thinking", "request.options")?,
        excluded: OwnedExcludedOptions {
            chat_template: optional_string(dict, "chat_template", "request.options")?,
            documents_json,
            continue_final_message: optional_bool(
                dict,
                "continue_final_message",
                "request.options",
            )?
            .unwrap_or(false),
            assistant_token_mask: optional_bool(dict, "assistant_token_mask", "request.options")?
                .unwrap_or(false),
            truncation: optional_bool(dict, "truncation", "request.options")?.unwrap_or(false),
            left_padding: optional_bool(dict, "left_padding", "request.options")?.unwrap_or(false),
            pad_token_id: optional_i64(dict, "pad_token_id", "request.options")?,
            load_audio_from_video: optional_bool(dict, "load_audio_from_video", "request.options")?
                .unwrap_or(false),
            audio_inputs,
        },
    };
    Ok(options)
}

pub(crate) fn parse_limits(limits: Option<&Bound<'_, PyAny>>) -> BindingResult<ResourceLimits> {
    let Some(value) = limits else {
        return Ok(ResourceLimits::default());
    };
    let dict = as_dict(value, "limits")?;
    reject_unknown_keys(
        dict,
        &[
            "requests_per_batch",
            "messages_per_request",
            "content_items_per_request",
            "text_bytes_per_request",
            "media_per_request",
            "media_per_batch",
            "encoded_bytes_per_item",
            "encoded_bytes_per_batch",
            "decoded_pixels_per_image_or_frame",
            "decoded_edge_length",
            "prepared_image_pixels_per_occurrence",
            "raw_frames_per_video",
            "rendered_tokens_per_request",
            "rendered_tokens_per_batch",
            "materialized_output_bytes_per_batch",
        ],
        "limits",
    )?;
    ResourceLimits::default().lowered(LimitOverrides {
        requests_per_batch: optional_u64(dict, "requests_per_batch", "limits")?,
        messages_per_request: optional_u64(dict, "messages_per_request", "limits")?,
        content_items_per_request: optional_u64(dict, "content_items_per_request", "limits")?,
        text_bytes_per_request: optional_u64(dict, "text_bytes_per_request", "limits")?,
        media_per_request: optional_u64(dict, "media_per_request", "limits")?,
        media_per_batch: optional_u64(dict, "media_per_batch", "limits")?,
        encoded_bytes_per_item: optional_u64(dict, "encoded_bytes_per_item", "limits")?,
        encoded_bytes_per_batch: optional_u64(dict, "encoded_bytes_per_batch", "limits")?,
        decoded_pixels_per_image_or_frame: optional_u64(
            dict,
            "decoded_pixels_per_image_or_frame",
            "limits",
        )?,
        decoded_edge_length: optional_u64(dict, "decoded_edge_length", "limits")?,
        prepared_image_pixels_per_occurrence: optional_u64(
            dict,
            "prepared_image_pixels_per_occurrence",
            "limits",
        )?,
        raw_frames_per_video: optional_u64(dict, "raw_frames_per_video", "limits")?,
        rendered_tokens_per_request: optional_u64(dict, "rendered_tokens_per_request", "limits")?,
        rendered_tokens_per_batch: optional_u64(dict, "rendered_tokens_per_batch", "limits")?,
        materialized_output_bytes_per_batch: optional_u64(
            dict,
            "materialized_output_bytes_per_batch",
            "limits",
        )?,
    })
}

fn parse_image_format(
    format: &str,
    request_index: usize,
    image_index: usize,
) -> BindingResult<ImageFormat> {
    match format.to_ascii_lowercase().as_str() {
        "jpeg" | "jpg" => Ok(ImageFormat::Jpeg),
        "png" => Ok(ImageFormat::Png),
        "webp" => Ok(ImageFormat::WebP),
        _ => Err(QwenError::new(
            ErrorCategory::UnsupportedMedia,
            "encoded image format is outside compatibility contract v1",
        )
        .with_context("format", format)
        .with_context("request_index", request_index)
        .with_context("image_index", image_index)),
    }
}

fn infer_image_format(data: &[u8]) -> Option<ImageFormat> {
    if data.starts_with(&[0xff, 0xd8]) {
        Some(ImageFormat::Jpeg)
    } else if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some(ImageFormat::Png)
    } else if data.starts_with(b"RIFF") && data.get(8..12) == Some(b"WEBP") {
        Some(ImageFormat::WebP)
    } else {
        None
    }
}

fn json_dumps(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    label: &'static str,
) -> BindingResult<String> {
    let json = PyModule::import(py, "json").map_err(|error| {
        invalid_request("Python json module is unavailable")
            .with_context("detail", py_detail(&error))
    })?;
    let encoded = json
        .getattr("dumps")
        .and_then(|dumps| dumps.call1((value,)))
        .map_err(|error| {
            invalid_request("value is not JSON serializable")
                .with_context("field", label)
                .with_context("detail", py_detail(&error))
        })?;
    extract_string(&encoded, label)
}

fn validate_json_string(
    py: Python<'_>,
    encoded: &str,
    label: &'static str,
    require_object: bool,
) -> BindingResult<()> {
    let json = PyModule::import(py, "json").map_err(|error| {
        invalid_request("Python json module is unavailable")
            .with_context("detail", py_detail(&error))
    })?;
    let value = json
        .getattr("loads")
        .and_then(|loads| loads.call1((encoded,)))
        .map_err(|error| {
            invalid_request("string is not valid JSON")
                .with_context("field", label)
                .with_context("detail", py_detail(&error))
        })?;
    if require_object && value.cast::<PyDict>().is_err() {
        return Err(invalid_request("JSON value must be an object").with_context("field", label));
    }
    Ok(())
}

fn as_dict<'a>(
    value: &'a Bound<'_, PyAny>,
    label: &'static str,
) -> BindingResult<&'a Bound<'a, PyDict>> {
    value.cast::<PyDict>().map_err(|_| {
        invalid_request("Python binding schema expected a mapping")
            .with_context("field", label)
            .with_context("actual_type", type_name(value))
    })
}

fn as_list<'a>(
    value: &'a Bound<'_, PyAny>,
    label: &'static str,
) -> BindingResult<&'a Bound<'a, PyList>> {
    value.cast::<PyList>().map_err(|_| {
        invalid_request("Python binding schema expected a list")
            .with_context("field", label)
            .with_context("actual_type", type_name(value))
    })
}

fn required<'py>(
    dict: &Bound<'py, PyDict>,
    key: &'static str,
    label: &'static str,
) -> BindingResult<Bound<'py, PyAny>> {
    optional(dict, key)?.ok_or_else(|| {
        invalid_request("Python binding schema is missing a required field")
            .with_context("object", label)
            .with_context("field", key)
    })
}

fn optional<'py>(
    dict: &Bound<'py, PyDict>,
    key: &'static str,
) -> BindingResult<Option<Bound<'py, PyAny>>> {
    dict.get_item(key).map_err(|error| {
        invalid_request("Python mapping field lookup failed")
            .with_context("field", key)
            .with_context("detail", py_detail(&error))
    })
}

fn reject_unknown_keys(
    dict: &Bound<'_, PyDict>,
    allowed: &[&str],
    label: &'static str,
) -> BindingResult<()> {
    for (key, _) in dict.iter() {
        let key = extract_string(&key, label)?;
        if !allowed.contains(&key.as_str()) {
            return Err(
                invalid_request("Python binding schema contains an unknown field")
                    .with_context("object", label)
                    .with_context("field", key),
            );
        }
    }
    Ok(())
}

fn has_key(dict: &Bound<'_, PyDict>, key: &'static str) -> BindingResult<bool> {
    dict.contains(key).map_err(|error| {
        invalid_request("Python mapping field lookup failed")
            .with_context("field", key)
            .with_context("detail", py_detail(&error))
    })
}

fn present_keys<'a>(
    dict: &Bound<'_, PyDict>,
    keys: &'a [&'static str],
) -> BindingResult<Vec<&'a str>> {
    let mut present = Vec::new();
    for key in keys {
        if has_key(dict, key)? {
            present.push(*key);
        }
    }
    Ok(present)
}

fn is_recognized_unsupported_media_object(value: &Bound<'_, PyAny>) -> bool {
    if value.is_instance_of::<PyString>() || value.hasattr("__fspath__").unwrap_or(false) {
        return true;
    }
    value
        .get_type()
        .getattr("__module__")
        .and_then(|module| module.extract::<String>())
        .is_ok_and(|module| module == "PIL" || module.starts_with("PIL."))
}

fn extract_string(value: &Bound<'_, PyAny>, label: &'static str) -> BindingResult<String> {
    let py_string = value.cast::<PyString>().map_err(|error| {
        invalid_request("Python binding schema expected a string")
            .with_context("field", label)
            .with_context("actual_type", type_name(value))
            .with_context("detail", error.to_string())
    })?;
    let source = py_string.to_str().map_err(|error| {
        invalid_request("Python string is not valid UTF-8")
            .with_context("field", label)
            .with_context("detail", py_detail(&error))
    })?;
    let mut owned = String::new();
    owned
        .try_reserve_exact(source.len())
        .map_err(|_| input_allocation_error("Python string input", source.len()))?;
    owned.push_str(source);
    Ok(owned)
}

fn extract_bool(value: &Bound<'_, PyAny>, label: &'static str) -> BindingResult<bool> {
    if !value.is_instance_of::<PyBool>() {
        return Err(invalid_request("Python binding schema expected a bool")
            .with_context("field", label)
            .with_context("actual_type", type_name(value)));
    }
    value.extract::<bool>().map_err(|error| {
        invalid_request("Python bool could not be extracted")
            .with_context("field", label)
            .with_context("detail", py_detail(&error))
    })
}

fn extract_u64(value: &Bound<'_, PyAny>, label: &'static str) -> BindingResult<u64> {
    if value.is_instance_of::<PyBool>() {
        return Err(invalid_request("Python bool is not accepted as an integer")
            .with_context("field", label));
    }
    value.extract::<u64>().map_err(|error| {
        invalid_request("Python binding schema expected a non-negative integer")
            .with_context("field", label)
            .with_context("actual_type", type_name(value))
            .with_context("detail", py_detail(&error))
    })
}

fn extract_usize(value: &Bound<'_, PyAny>, label: &'static str) -> BindingResult<usize> {
    if value.is_instance_of::<PyBool>() {
        return Err(invalid_request("Python bool is not accepted as an integer")
            .with_context("field", label));
    }
    value.extract::<usize>().map_err(|error| {
        invalid_request("Python binding schema expected a non-negative platform-sized integer")
            .with_context("field", label)
            .with_context("actual_type", type_name(value))
            .with_context("detail", py_detail(&error))
    })
}

fn optional_string(
    dict: &Bound<'_, PyDict>,
    key: &'static str,
    label: &'static str,
) -> BindingResult<Option<String>> {
    optional(dict, key)?
        .map(|value| extract_string(&value, label))
        .transpose()
}

fn optional_bool(
    dict: &Bound<'_, PyDict>,
    key: &'static str,
    label: &'static str,
) -> BindingResult<Option<bool>> {
    optional(dict, key)?
        .map(|value| extract_bool(&value, label))
        .transpose()
}

fn optional_u64(
    dict: &Bound<'_, PyDict>,
    key: &'static str,
    label: &'static str,
) -> BindingResult<Option<u64>> {
    optional(dict, key)?
        .map(|value| extract_u64(&value, label))
        .transpose()
}

fn optional_i64(
    dict: &Bound<'_, PyDict>,
    key: &'static str,
    label: &'static str,
) -> BindingResult<Option<i64>> {
    optional(dict, key)?
        .map(|value| {
            if value.is_instance_of::<PyBool>() {
                return Err(invalid_request("Python bool is not accepted as an integer")
                    .with_context("field", label));
            }
            value.extract::<i64>().map_err(|error| {
                invalid_request("Python binding schema expected an integer")
                    .with_context("field", label)
                    .with_context("detail", py_detail(&error))
            })
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use numpy::PyUntypedArrayMethods;
    use pyo3::{
        prelude::*,
        types::{PyDict, PyList, PyModule},
    };
    use qwen_mm_core::{ErrorCategory, ResourceLimits};

    use super::{OwnedImage, parse_requests};

    fn parse_raw_image(
        py: Python<'_>,
        array: &Bound<'_, PyAny>,
    ) -> super::BindingResult<OwnedImage> {
        let message = PyDict::new(py);
        message.set_item("role", "user").expect("role");
        let image_item = PyDict::new(py);
        image_item.set_item("type", "image").expect("type");
        image_item.set_item("input_index", 0).expect("input index");
        message
            .set_item(
                "content",
                PyList::new(py, [image_item]).expect("content list"),
            )
            .expect("content");
        let request = PyDict::new(py);
        request
            .set_item(
                "messages",
                PyList::new(py, [message]).expect("messages list"),
            )
            .expect("messages");
        request.set_item("images", vec![array]).expect("images");
        let requests = PyList::new(py, [request]).expect("requests list");
        let mut parsed = parse_requests(
            py,
            requests.as_any(),
            ResourceLimits::default(),
            false,
            "qwen3-vl-8b",
        )?;
        Ok(parsed.remove(0).images.remove(0))
    }

    fn as_strided<'py>(
        numpy: &Bound<'py, PyModule>,
        source: &Bound<'py, PyAny>,
        shape: (usize, usize, usize),
        strides: (isize, isize, isize),
    ) -> Bound<'py, PyAny> {
        let kwargs = PyDict::new(numpy.py());
        kwargs.set_item("shape", shape).expect("shape");
        kwargs.set_item("strides", strides).expect("strides");
        numpy
            .getattr("lib")
            .expect("lib")
            .getattr("stride_tricks")
            .expect("stride tricks")
            .getattr("as_strided")
            .expect("as_strided")
            .call((source,), Some(&kwargs))
            .expect("strided array")
    }

    #[test]
    #[ignore = "requires NumPy importable by embedded CPython"]
    fn raw_arrays_are_copied_after_strict_validation() {
        Python::initialize();
        Python::attach(|py| {
            let numpy = PyModule::import(py, "numpy").expect("numpy");
            let array = numpy
                .getattr("arange")
                .expect("arange")
                .call1((12,))
                .expect("array")
                .call_method1("reshape", ((2, 2, 3),))
                .expect("reshape")
                .call_method1("astype", ("uint8",))
                .expect("uint8");
            let parsed = parse_raw_image(py, &array).expect("valid raw image");
            let OwnedImage::Rgb8 {
                data,
                height,
                width,
                ..
            } = &parsed
            else {
                panic!("raw image")
            };
            assert_eq!((*height, *width, data.len()), (2, 2, 12));
            assert!(
                array
                    .cast::<numpy::PyArrayDyn<u8>>()
                    .expect("array")
                    .is_c_contiguous()
            );
        });
    }

    #[test]
    #[ignore = "requires NumPy importable by embedded CPython"]
    fn positive_row_padding_is_copied_into_packed_owned_memory() {
        Python::initialize();
        Python::attach(|py| {
            let numpy = PyModule::import(py, "numpy").expect("numpy");
            let base = numpy
                .getattr("arange")
                .expect("arange")
                .call1((24,))
                .expect("array")
                .call_method1("reshape", ((2, 4, 3),))
                .expect("reshape")
                .call_method1("astype", ("uint8",))
                .expect("uint8");
            let padded = as_strided(&numpy, &base, (2, 2, 3), (12, 3, 1));
            assert!(
                !padded
                    .cast::<numpy::PyArrayDyn<u8>>()
                    .expect("array")
                    .is_c_contiguous()
            );

            let image = parse_raw_image(py, &padded).expect("row-padded HWC image");
            let OwnedImage::Rgb8 {
                data,
                height,
                width,
                row_stride,
            } = image
            else {
                panic!("raw image")
            };
            assert_eq!((height, width, row_stride), (2, 2, 6));
            assert_eq!(data.len(), 12);
            assert_eq!(data, [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17]);
        });
    }

    #[test]
    #[ignore = "requires NumPy importable by embedded CPython"]
    fn non_hwc_strides_remain_media_geometry_errors() {
        Python::initialize();
        Python::attach(|py| {
            let numpy = PyModule::import(py, "numpy").expect("numpy");
            let base = numpy
                .getattr("arange")
                .expect("arange")
                .call1((24,))
                .expect("array")
                .call_method1("reshape", ((2, 4, 3),))
                .expect("reshape")
                .call_method1("astype", ("uint8",))
                .expect("uint8");
            let negative_rows = numpy
                .getattr("flip")
                .expect("flip")
                .call1((&base, 0))
                .expect("negative rows");
            let transposed = numpy
                .getattr("transpose")
                .expect("transpose")
                .call1((&base, (1, 0, 2)))
                .expect("transposed");
            let channel_source = numpy
                .getattr("arange")
                .expect("arange")
                .call1((24,))
                .expect("array")
                .call_method1("reshape", ((2, 2, 6),))
                .expect("reshape")
                .call_method1("astype", ("uint8",))
                .expect("uint8");
            let channel_sliced = as_strided(&numpy, &channel_source, (2, 2, 3), (12, 6, 2));

            for invalid in [&negative_rows, &transposed, &channel_sliced] {
                let error = parse_raw_image(py, invalid).expect_err("invalid HWC strides");
                assert_eq!(error.category(), ErrorCategory::MediaGeometry);
            }
        });
    }
}
