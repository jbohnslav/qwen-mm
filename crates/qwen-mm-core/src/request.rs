//! Rust-native request, message, media, and option types.

use serde_json::Value;

use crate::{
    error::{ErrorCategory, QwenError, Result},
    profile::Profile,
};

/// A supported chat role.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Role {
    /// Optional first system message.
    System,
    /// User message.
    User,
    /// Assistant message, optionally with tool calls or reasoning.
    Assistant,
    /// Tool response message.
    Tool,
}

/// Supported encoded still-image formats.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ImageFormat {
    /// JPEG image.
    Jpeg,
    /// PNG image.
    Png,
    /// WebP image.
    WebP,
}

/// A caller-owned RGB8 raster with an explicit row stride.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Rgb8<'a> {
    /// Entire readable source byte span.
    pub data: &'a [u8],
    /// Height in rows.
    pub height: usize,
    /// Width in pixels.
    pub width: usize,
    /// Bytes from the beginning of one row to the next.
    pub row_stride: usize,
}

/// A still-image source accepted by the core v1 API.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageInput<'a> {
    /// JPEG, PNG, or WebP bytes. Decode validity is checked at the decode stage.
    Encoded {
        /// Encoded bytes.
        data: &'a [u8],
        /// Declared supported codec.
        format: ImageFormat,
    },
    /// Caller-owned RGB8 pixels.
    Rgb8(Rgb8<'a>),
}

/// A caller-owned raw-frame video accepted by the core v1 API.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VideoInput<'a> {
    /// Ordered RGB8 frames. All frames must have common dimensions.
    pub frames: &'a [Rgb8<'a>],
}

/// Per-occurrence still-image resize options.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ImageOptions {
    /// Optional effective minimum prepared pixels.
    pub min_pixels: Option<u64>,
    /// Optional effective maximum prepared pixels.
    pub max_pixels: Option<u64>,
    /// Explicit prepared height; must be paired with `resized_width`.
    pub resized_height: Option<u64>,
    /// Explicit prepared width; must be paired with `resized_height`.
    pub resized_width: Option<u64>,
}

/// Per-occurrence raw-frame video options.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VideoOptions {
    /// Requested sampling rate for prompt metadata.
    pub sample_fps: Option<f64>,
    /// Source frame rate.
    pub raw_fps: Option<f64>,
    /// Optional per-frame minimum pixels.
    pub min_pixels: Option<u64>,
    /// Optional per-frame maximum pixels.
    pub max_pixels: Option<u64>,
    /// Optional total prepared video pixel budget.
    pub total_pixels: Option<u64>,
    /// Explicit prepared frame height; must be paired with `resized_width`.
    pub resized_height: Option<u64>,
    /// Explicit prepared frame width; must be paired with `resized_height`.
    pub resized_width: Option<u64>,
}

/// An image occurrence referencing an entry in [`Request::images`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ImageRef {
    /// Zero-based input index. Reuse deliberately creates another occurrence.
    pub input_index: usize,
    /// Per-occurrence options.
    pub options: ImageOptions,
}

/// A video occurrence referencing an entry in [`Request::videos`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VideoRef {
    /// Zero-based input index. Reuse deliberately creates another occurrence.
    pub input_index: usize,
    /// Per-occurrence options.
    pub options: VideoOptions,
}

/// One item in ordered list content.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ContentItem<'a> {
    /// Text content.
    Text(&'a str),
    /// One ordered image occurrence.
    Image(ImageRef),
    /// One ordered raw-frame video occurrence.
    Video(VideoRef),
}

/// Either shorthand string content or an ordered content-item list.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MessageContent<'a> {
    /// Shorthand text content.
    Text(&'a str),
    /// Ordered structured content.
    Items(&'a [ContentItem<'a>]),
}

/// The normalized function-call payload used by both accepted input forms.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FunctionCall<'a> {
    /// Function name.
    pub name: &'a str,
    /// JSON-compatible arguments encoded as one JSON value.
    pub arguments_json: &'a str,
}

/// The two assistant tool-call shapes accepted by the pinned templates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolCall<'a> {
    /// Direct function fields.
    Direct {
        /// Optional call identifier.
        id: Option<&'a str>,
        /// Function call.
        call: FunctionCall<'a>,
    },
    /// `{ "function": ... }` wrapper form.
    Function {
        /// Optional call identifier.
        id: Option<&'a str>,
        /// Function call inside the wrapper.
        function: FunctionCall<'a>,
    },
}

impl ToolCall<'_> {
    pub(crate) const fn function_call(&self) -> &FunctionCall<'_> {
        match self {
            Self::Direct { call, .. } => call,
            Self::Function { function, .. } => function,
        }
    }

    pub(crate) const fn id(&self) -> Option<&str> {
        match self {
            Self::Direct { id, .. } | Self::Function { id, .. } => *id,
        }
    }
}

/// One JSON-compatible tool definition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ToolDefinition<'a> {
    /// A JSON object, retained byte-for-byte for the renderer.
    pub json: &'a str,
}

/// One ordered structured message.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Message<'a> {
    /// Message role.
    pub role: Role,
    /// Shorthand text or ordered items.
    pub content: MessageContent<'a>,
    /// Assistant tool calls; empty for other roles.
    pub tool_calls: &'a [ToolCall<'a>],
    /// Qwen3.5-only assistant reasoning content.
    pub reasoning_content: Option<&'a str>,
}

/// Deliberately represented exclusions, so adapters can return a stable error
/// instead of silently dropping an upstream option.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct ExcludedOptions<'a> {
    /// Arbitrary chat template override.
    pub chat_template: Option<&'a str>,
    /// JSON documents passed to a template.
    pub documents_json: Option<&'a str>,
    /// Continue the final message instead of normal rendering.
    pub continue_final_message: bool,
    /// Request an assistant-token mask.
    pub assistant_token_mask: bool,
    /// Request tokenizer truncation.
    pub truncation: bool,
    /// Request left padding.
    pub left_padding: bool,
    /// Supply a caller-selected pad token.
    pub pad_token_id: Option<i64>,
    /// Ask video processing to load audio.
    pub load_audio_from_video: bool,
    /// Number of supplied audio inputs.
    pub audio_inputs: usize,
}

/// Supported template options plus explicitly rejected upstream options.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RequestOptions<'a> {
    /// Add the template's assistant generation prompt.
    pub add_generation_prompt: bool,
    /// Add numbered vision identifiers.
    pub add_vision_id: bool,
    /// Ordered JSON-compatible tool definitions.
    pub tools: &'a [ToolDefinition<'a>],
    /// Qwen3.5-only thinking toggle. It only applies with a generation prompt.
    pub enable_thinking: Option<bool>,
    /// Excluded upstream options retained for categorized rejection.
    pub excluded: ExcludedOptions<'a>,
}

/// One non-empty ordered conversation and its caller-owned media inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Request<'a> {
    /// Ordered structured messages.
    pub messages: &'a [Message<'a>],
    /// Indexed still-image inputs.
    pub images: &'a [ImageInput<'a>],
    /// Indexed raw-frame video inputs.
    pub videos: &'a [VideoInput<'a>],
    /// Template and declared excluded options.
    pub options: RequestOptions<'a>,
}

/// Stable location of a media occurrence in batch traversal order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OccurrenceLocation {
    /// Request index in the batch.
    pub request_index: usize,
    /// Message index in the request.
    pub message_index: usize,
    /// Content-item index in the message.
    pub content_item_index: usize,
    /// Referenced input index in its modality array.
    pub input_index: usize,
}

pub(crate) fn validate_request_structure(
    request: &Request<'_>,
    request_index: usize,
) -> Result<()> {
    if request.messages.is_empty() {
        return Err(invalid("request must contain at least one message")
            .with_context("request_index", request_index));
    }

    let mut saw_system = false;
    for (message_index, message) in request.messages.iter().enumerate() {
        if message.role == Role::System {
            if saw_system || message_index != 0 {
                return Err(invalid("system message must appear at most once and first")
                    .with_context("request_index", request_index)
                    .with_context("message_index", message_index));
            }
            saw_system = true;
        }
        validate_message(message, request, request_index, message_index)?;
    }
    validate_all_inputs_are_referenced(request, request_index)
}

pub(crate) fn validate_request_options(
    request: &Request<'_>,
    profile: &Profile,
    request_index: usize,
) -> Result<()> {
    let excluded = &request.options.excluded;
    let excluded_name = if excluded.chat_template.is_some() {
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
    if let Some(option) = excluded_name {
        return Err(QwenError::new(
            ErrorCategory::UnsupportedOption,
            "option is outside compatibility contract v1",
        )
        .with_context("option", option)
        .with_context("request_index", request_index));
    }

    if request.options.enable_thinking.is_some() {
        if !profile.supports_thinking() {
            return Err(QwenError::new(
                ErrorCategory::UnsupportedOption,
                "thinking is not supported by this profile",
            )
            .with_context("option", "enable_thinking")
            .with_context("profile", profile.alias.as_str())
            .with_context("request_index", request_index));
        }
        if !request.options.add_generation_prompt {
            return Err(invalid("enable_thinking requires add_generation_prompt")
                .with_context("request_index", request_index));
        }
    }

    if !profile.supports_thinking()
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
        .with_context("profile", profile.alias.as_str())
        .with_context("request_index", request_index));
    }

    for (tool_index, tool) in request.options.tools.iter().enumerate() {
        validate_json_object(tool.json, "tool definition", request_index, tool_index)?;
    }
    Ok(())
}

fn validate_message(
    message: &Message<'_>,
    request: &Request<'_>,
    request_index: usize,
    message_index: usize,
) -> Result<()> {
    if let MessageContent::Items(items) = message.content {
        if items.is_empty() {
            return Err(invalid("content item list must not be empty")
                .with_context("request_index", request_index)
                .with_context("message_index", message_index));
        }
        for (item_index, item) in items.iter().enumerate() {
            match item {
                ContentItem::Text(_) => {}
                ContentItem::Image(reference) => {
                    if message.role == Role::System {
                        return Err(invalid("system messages cannot contain visuals")
                            .with_context("request_index", request_index)
                            .with_context("message_index", message_index)
                            .with_context("content_item_index", item_index));
                    }
                    if reference.input_index >= request.images.len() {
                        return Err(invalid("image reference is out of bounds")
                            .with_context("request_index", request_index)
                            .with_context("message_index", message_index)
                            .with_context("content_item_index", item_index)
                            .with_context("input_index", reference.input_index));
                    }
                }
                ContentItem::Video(reference) => {
                    if message.role == Role::System {
                        return Err(invalid("system messages cannot contain visuals")
                            .with_context("request_index", request_index)
                            .with_context("message_index", message_index)
                            .with_context("content_item_index", item_index));
                    }
                    if reference.input_index >= request.videos.len() {
                        return Err(invalid("video reference is out of bounds")
                            .with_context("request_index", request_index)
                            .with_context("message_index", message_index)
                            .with_context("content_item_index", item_index)
                            .with_context("input_index", reference.input_index));
                    }
                }
            }
            if message.role == Role::Assistant && !matches!(item, ContentItem::Text(_)) {
                return Err(invalid("assistant content only supports text items")
                    .with_context("request_index", request_index)
                    .with_context("message_index", message_index)
                    .with_context("content_item_index", item_index));
            }
        }
    }

    if message.role != Role::Assistant && !message.tool_calls.is_empty() {
        return Err(invalid("tool calls are only valid on assistant messages")
            .with_context("request_index", request_index)
            .with_context("message_index", message_index));
    }
    if message.role != Role::Assistant && message.reasoning_content.is_some() {
        return Err(
            invalid("reasoning content is only valid on assistant messages")
                .with_context("request_index", request_index)
                .with_context("message_index", message_index),
        );
    }
    for (tool_call_index, tool_call) in message.tool_calls.iter().enumerate() {
        let function = tool_call.function_call();
        if function.name.is_empty() {
            return Err(invalid("tool-call function name must not be empty")
                .with_context("request_index", request_index)
                .with_context("message_index", message_index)
                .with_context("tool_call_index", tool_call_index));
        }
        validate_json_value(
            function.arguments_json,
            "tool-call arguments",
            request_index,
            tool_call_index,
        )?;
    }
    Ok(())
}

fn validate_all_inputs_are_referenced(request: &Request<'_>, request_index: usize) -> Result<()> {
    for input_index in 0..request.images.len() {
        if !request.messages.iter().any(|message| {
            matches!(message.content, MessageContent::Items(items) if items.iter().any(|item| matches!(item, ContentItem::Image(reference) if reference.input_index == input_index)))
        }) {
            return Err(invalid("supplied image has no content occurrence")
                .with_context("request_index", request_index)
                .with_context("input_index", input_index));
        }
    }
    for input_index in 0..request.videos.len() {
        if !request.messages.iter().any(|message| {
            matches!(message.content, MessageContent::Items(items) if items.iter().any(|item| matches!(item, ContentItem::Video(reference) if reference.input_index == input_index)))
        }) {
            return Err(invalid("supplied video has no content occurrence")
                .with_context("request_index", request_index)
                .with_context("input_index", input_index));
        }
    }
    Ok(())
}

fn validate_json_object(
    json: &str,
    label: &str,
    request_index: usize,
    value_index: usize,
) -> Result<()> {
    let value = parse_json(json, label, request_index, value_index)?;
    if !value.is_object() {
        return Err(invalid(format!("{label} must be a JSON object"))
            .with_context("request_index", request_index)
            .with_context("value_index", value_index));
    }
    Ok(())
}

fn validate_json_value(
    json: &str,
    label: &str,
    request_index: usize,
    value_index: usize,
) -> Result<()> {
    parse_json(json, label, request_index, value_index).map(drop)
}

fn parse_json(json: &str, label: &str, request_index: usize, value_index: usize) -> Result<Value> {
    serde_json::from_str(json).map_err(|error| {
        invalid(format!("malformed {label}"))
            .with_context("request_index", request_index)
            .with_context("value_index", value_index)
            .with_context("detail", error.to_string())
    })
}

fn invalid(message: impl Into<String>) -> QwenError {
    QwenError::new(ErrorCategory::InvalidRequest, message)
}

#[cfg(test)]
mod tests {
    use super::{
        ContentItem, ExcludedOptions, ImageInput, ImageRef, Message, MessageContent, Request,
        RequestOptions, Role, ToolCall, ToolDefinition, validate_request_options,
        validate_request_structure,
    };
    use crate::{
        error::ErrorCategory,
        profile::{ProfileAlias, ProfileRegistry},
    };

    fn text_message(role: Role, text: &str) -> Message<'_> {
        Message {
            role,
            content: MessageContent::Text(text),
            tool_calls: &[],
            reasoning_content: None,
        }
    }

    #[test]
    fn repeated_media_reference_is_two_valid_occurrences() {
        let bytes = [1_u8];
        let images = [ImageInput::Encoded {
            data: &bytes,
            format: super::ImageFormat::Jpeg,
        }];
        let items = [
            ContentItem::Image(ImageRef::default()),
            ContentItem::Text("compare"),
            ContentItem::Image(ImageRef::default()),
        ];
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
        validate_request_structure(&request, 0).expect("reuse is explicitly supported");
    }

    #[test]
    fn invalid_role_content_and_media_combinations_are_rejected() {
        let visual_items = [ContentItem::Image(ImageRef::default())];
        let messages = [Message {
            role: Role::System,
            content: MessageContent::Items(&visual_items),
            tool_calls: &[],
            reasoning_content: None,
        }];
        let request = Request {
            messages: &messages,
            images: &[ImageInput::Encoded {
                data: &[1],
                format: super::ImageFormat::Png,
            }],
            videos: &[],
            options: RequestOptions::default(),
        };
        assert_eq!(
            validate_request_structure(&request, 0)
                .expect_err("system visual")
                .category(),
            ErrorCategory::InvalidRequest
        );

        let messages = [
            text_message(Role::User, "one"),
            text_message(Role::System, "late"),
        ];
        let request = Request {
            messages: &messages,
            images: &[],
            videos: &[],
            options: RequestOptions::default(),
        };
        assert!(validate_request_structure(&request, 0).is_err());
    }

    #[test]
    fn excluded_and_profile_conditional_options_have_stable_categories() {
        let registry = ProfileRegistry::bundled().expect("profiles");
        let messages = [text_message(Role::User, "hello")];
        let excluded = Request {
            messages: &messages,
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
        let error = validate_request_options(&excluded, registry.get(ProfileAlias::Qwen3Vl8b), 0)
            .expect_err("truncation is excluded");
        assert_eq!(error.category(), ErrorCategory::UnsupportedOption);
        assert_eq!(error.context()["option"].to_string(), "truncation");

        let clean = Request {
            messages: &messages,
            images: &[],
            videos: &[],
            options: RequestOptions::default(),
        };
        let thinking = Request {
            options: RequestOptions {
                add_generation_prompt: true,
                enable_thinking: Some(true),
                ..RequestOptions::default()
            },
            ..clean
        };
        let error = validate_request_options(&thinking, registry.get(ProfileAlias::Qwen3Vl8b), 0)
            .expect_err("thinking is qwen3.5 only");
        assert_eq!(error.category(), ErrorCategory::UnsupportedOption);

        let reasoning_messages = [Message {
            role: Role::Assistant,
            content: MessageContent::Text("answer"),
            tool_calls: &[],
            reasoning_content: Some("hidden"),
        }];
        let reasoning = Request {
            messages: &reasoning_messages,
            ..clean
        };
        assert_eq!(
            validate_request_options(&reasoning, registry.get(ProfileAlias::Qwen3Vl8b), 0)
                .expect_err("reasoning is qwen3.5 only")
                .category(),
            ErrorCategory::UnsupportedOption
        );
    }

    #[test]
    fn tools_are_rust_data_and_validate_json() {
        let call = ToolCall::Direct {
            id: Some("call-1"),
            call: super::FunctionCall {
                name: "weather",
                arguments_json: "{\"city\":\"Boston\"}",
            },
        };
        let messages = [Message {
            role: Role::Assistant,
            content: MessageContent::Text(""),
            tool_calls: &[call],
            reasoning_content: None,
        }];
        let tools = [ToolDefinition {
            json: "{\"type\":\"function\"}",
        }];
        let request = Request {
            messages: &messages,
            images: &[],
            videos: &[],
            options: RequestOptions {
                tools: &tools,
                ..RequestOptions::default()
            },
        };
        validate_request_structure(&request, 0).expect("valid direct tool call");
        let registry = ProfileRegistry::bundled().expect("profiles");
        validate_request_options(&request, registry.get(ProfileAlias::Qwen35_9b), 0)
            .expect("valid tool definition");
    }
}
