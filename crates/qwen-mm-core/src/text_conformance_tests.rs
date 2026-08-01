use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::{
    PlannedTextRequest, TextProcessor, VisualExpansion, expand_visuals, render_chat,
    validate_visual_plan,
};
use crate::{
    ContentItem, ErrorCategory, ExcludedOptions, FunctionCall, ImageFormat, ImageInput, ImageRef,
    Message, MessageContent, ProfileAlias, ProfileRegistry, ProfiledRequest, Request,
    RequestOptions, ResourceLimits, Rgb8, Role, ToolCall, ToolDefinition, VideoInput, VideoRef,
    preflight_batch,
};

const FIXTURE: &str = include_str!("../../../reference/conformance/v1/chat.json");
static DUMMY_IMAGE: [u8; 1] = [0];
static DUMMY_FRAME: [u8; 3] = [0, 0, 0];
static DUMMY_FRAMES: [Rgb8<'static>; 1] = [Rgb8 {
    data: &DUMMY_FRAME,
    height: 1,
    width: 1,
    row_stride: 3,
}];

#[derive(Debug, Deserialize)]
struct ChatFixture {
    schema_version: u32,
    contract_id: String,
    generator: String,
    generator_command: String,
    profiles: serde_json::Value,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    id: String,
    profile: String,
    requests: Vec<RequestSpec>,
    expected: Option<Expected>,
    expected_error: Option<ExpectedError>,
}

#[derive(Debug, Deserialize)]
struct RequestSpec {
    messages: Vec<MessageSpec>,
    #[serde(default)]
    options: OptionsSpec,
    #[serde(default)]
    visuals: Vec<VisualSpec>,
}

#[derive(Debug, Deserialize)]
struct MessageSpec {
    role: String,
    content: ContentSpec,
    #[serde(default)]
    tool_calls: Vec<ToolCallSpec>,
    reasoning_content: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ContentSpec {
    Text(String),
    Items(Vec<ContentItemSpec>),
}

#[derive(Debug, Deserialize)]
struct ContentItemSpec {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
    input_index: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ToolCallSpec {
    shape: String,
    name: String,
    arguments_json: String,
}

#[derive(Debug, Default, Deserialize)]
struct OptionsSpec {
    #[serde(default)]
    add_generation_prompt: bool,
    #[serde(default)]
    add_vision_id: bool,
    enable_thinking: Option<bool>,
    #[serde(default)]
    tools_json: Vec<String>,
    #[serde(default)]
    excluded: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct VisualSpec {
    kind: String,
    input_index: usize,
    grid_thw: [u64; 3],
    #[serde(default)]
    timestamps: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct Expected {
    rendered_prompts: Vec<String>,
    expanded_prompts: Vec<String>,
    rendered_sha256: Vec<String>,
    expanded_sha256: Vec<String>,
    input_ids: Vec<Vec<i64>>,
    attention_mask: Vec<Vec<i64>>,
    mm_token_type_ids: Vec<Vec<i64>>,
}

#[derive(Debug, Deserialize)]
struct ExpectedError {
    category: String,
}

fn fixture() -> ChatFixture {
    let fixture: ChatFixture = serde_json::from_str(FIXTURE).expect("chat conformance fixture");
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.contract_id, "qwen-mm-compat-v1");
    assert_eq!(fixture.generator, "qwen_mm_reference.chat_conformance");
    assert_eq!(
        fixture.generator_command,
        "./scripts/with-cargo.sh uv run --locked --no-sync --package qwen-mm-reference python -m qwen_mm_reference.chat_conformance"
    );
    assert!(fixture.profiles.is_object());
    fixture
}

fn role(value: &str) -> Role {
    match value {
        "system" => Role::System,
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        other => panic!("unknown fixture role {other}"),
    }
}

fn excluded_options(names: &[String]) -> ExcludedOptions<'static> {
    let mut options = ExcludedOptions::default();
    for name in names {
        match name.as_str() {
            "chat_template" => options.chat_template = Some("fixture override"),
            "truncation" => options.truncation = true,
            other => panic!("unknown fixture exclusion {other}"),
        }
    }
    options
}

fn fixture_content_items(spec: &RequestSpec) -> &[Vec<ContentItem<'_>>] {
    let content_items = spec
        .messages
        .iter()
        .map(|message| match &message.content {
            ContentSpec::Text(_) => Vec::new(),
            ContentSpec::Items(items) => items
                .iter()
                .map(|item| match item.kind.as_str() {
                    "text" => ContentItem::Text(item.text.as_deref().expect("fixture text")),
                    "image" => ContentItem::Image(ImageRef {
                        input_index: item.input_index.expect("fixture image index"),
                        ..ImageRef::default()
                    }),
                    "video" => ContentItem::Video(VideoRef {
                        input_index: item.input_index.expect("fixture video index"),
                        ..VideoRef::default()
                    }),
                    other => panic!("unknown fixture content kind {other}"),
                })
                .collect(),
        })
        .collect::<Vec<Vec<_>>>();
    Box::leak(content_items.into_boxed_slice())
}

fn fixture_tool_calls(spec: &RequestSpec) -> &[Vec<ToolCall<'_>>] {
    let tool_calls = spec
        .messages
        .iter()
        .map(|message| {
            message
                .tool_calls
                .iter()
                .map(|call| {
                    let function = FunctionCall {
                        name: &call.name,
                        arguments_json: &call.arguments_json,
                    };
                    match call.shape.as_str() {
                        "direct" => ToolCall::Direct {
                            id: None,
                            call: function,
                        },
                        "function" => ToolCall::Function { id: None, function },
                        other => panic!("unknown fixture tool-call shape {other}"),
                    }
                })
                .collect()
        })
        .collect::<Vec<Vec<_>>>();
    Box::leak(tool_calls.into_boxed_slice())
}

fn fixture_messages<'a>(
    spec: &'a RequestSpec,
    content_items: &'a [Vec<ContentItem<'a>>],
    tool_calls: &'a [Vec<ToolCall<'a>>],
) -> &'a [Message<'a>] {
    let messages = spec
        .messages
        .iter()
        .enumerate()
        .map(|(index, message)| Message {
            role: role(&message.role),
            content: match &message.content {
                ContentSpec::Text(text) => MessageContent::Text(text),
                ContentSpec::Items(_) => MessageContent::Items(&content_items[index]),
            },
            tool_calls: &tool_calls[index],
            reasoning_content: message.reasoning_content.as_deref(),
        })
        .collect::<Vec<_>>();
    Box::leak(messages.into_boxed_slice())
}

fn max_input_index(spec: &RequestSpec, kind: &str) -> Option<usize> {
    spec.messages
        .iter()
        .filter_map(|message| match &message.content {
            ContentSpec::Items(items) => items
                .iter()
                .filter(|item| item.kind == kind)
                .filter_map(|item| item.input_index)
                .max(),
            ContentSpec::Text(_) => None,
        })
        .max()
}

fn fixture_images<'a>(spec: &RequestSpec) -> &'a [ImageInput<'a>] {
    let images = max_input_index(spec, "image").map_or_else(Vec::new, |maximum| {
        vec![
            ImageInput::Encoded {
                data: &DUMMY_IMAGE,
                format: ImageFormat::Jpeg,
            };
            maximum + 1
        ]
    });
    Box::leak(images.into_boxed_slice())
}

fn fixture_videos<'a>(spec: &RequestSpec) -> &'a [VideoInput<'a>] {
    let videos = max_input_index(spec, "video").map_or_else(Vec::new, |maximum| {
        vec![
            VideoInput {
                frames: &DUMMY_FRAMES,
            };
            maximum + 1
        ]
    });
    Box::leak(videos.into_boxed_slice())
}

fn fixture_tools(spec: &RequestSpec) -> &[ToolDefinition<'_>] {
    let tools = spec
        .options
        .tools_json
        .iter()
        .map(|json| ToolDefinition { json })
        .collect::<Vec<_>>();
    Box::leak(tools.into_boxed_slice())
}

fn fixture_visuals(spec: &RequestSpec) -> &[VisualExpansion<'_>] {
    let visuals = spec
        .visuals
        .iter()
        .map(|visual| match visual.kind.as_str() {
            "image" => VisualExpansion::Image {
                input_index: visual.input_index,
                grid_thw: visual.grid_thw,
            },
            "video" => VisualExpansion::Video {
                input_index: visual.input_index,
                grid_thw: visual.grid_thw,
                timestamps: &visual.timestamps,
            },
            other => panic!("unknown fixture visual kind {other}"),
        })
        .collect::<Vec<_>>();
    Box::leak(visuals.into_boxed_slice())
}

fn build_request(spec: &RequestSpec) -> PlannedTextRequest<'_> {
    let content_items = fixture_content_items(spec);
    let tool_calls = fixture_tool_calls(spec);
    let messages = fixture_messages(spec, content_items, tool_calls);

    PlannedTextRequest {
        request: Request {
            messages,
            images: fixture_images(spec),
            videos: fixture_videos(spec),
            options: RequestOptions {
                add_generation_prompt: spec.options.add_generation_prompt,
                add_vision_id: spec.options.add_vision_id,
                tools: fixture_tools(spec),
                enable_thinking: spec.options.enable_thinking,
                excluded: excluded_options(&spec.options.excluded),
            },
        },
        visuals: fixture_visuals(spec),
    }
}

fn category(value: &str) -> ErrorCategory {
    match value {
        "invalid_request" => ErrorCategory::InvalidRequest,
        "unsupported_option" => ErrorCategory::UnsupportedOption,
        "profile_mismatch" => ErrorCategory::ProfileMismatch,
        other => panic!("unknown fixture error category {other}"),
    }
}

fn digest(value: &str) -> String {
    format!("{:x}", Sha256::digest(value.as_bytes()))
}

fn flatten(rows: &[Vec<i64>]) -> Vec<i64> {
    rows.iter().flatten().copied().collect()
}

fn asset_directory(alias: ProfileAlias) -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference/.cache/huggingface");
    match alias {
        ProfileAlias::Qwen3Vl8b => root.join(
            "models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        ),
        ProfileAlias::Qwen35_9b => {
            root.join("models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a")
        }
    }
}

#[test]
fn complete_chat_fixture_rendering_expansion_and_errors_are_always_on() {
    let fixture = fixture();
    let registry = ProfileRegistry::bundled().expect("profiles");
    for case in &fixture.cases {
        if let Some(expected_error) = &case.expected_error {
            if case.profile == "qwen-unknown" {
                let error = registry.resolve(&case.profile).expect_err(&case.id);
                assert_eq!(
                    error.category(),
                    category(&expected_error.category),
                    "{}",
                    case.id
                );
                continue;
            }
            let planned = case.requests.iter().map(build_request).collect::<Vec<_>>();
            let profiled = planned
                .iter()
                .map(|request| ProfiledRequest {
                    profile_alias: &case.profile,
                    request: request.request,
                })
                .collect::<Vec<_>>();
            if let Err(error) = preflight_batch(&registry, &profiled, ResourceLimits::default()) {
                assert_eq!(
                    error.category(),
                    category(&expected_error.category),
                    "{}",
                    case.id
                );
            } else {
                let profile = registry.resolve(&case.profile).expect("fixture profile");
                let mut observed = None;
                for (request_index, request) in planned.iter().enumerate() {
                    validate_visual_plan(&request.request, request.visuals, request_index)
                        .expect("fixture visual plan");
                    let rendered = render_chat(profile, &request.request, request_index)
                        .expect("fixture render");
                    if let Err(error) =
                        expand_visuals(profile, &rendered, request.visuals, request_index)
                    {
                        observed = Some(error);
                        break;
                    }
                }
                let error = observed.expect("fixture expected a post-preflight error");
                assert_eq!(
                    error.category(),
                    category(&expected_error.category),
                    "{}",
                    case.id
                );
            }
            continue;
        }

        let profile = registry.resolve(&case.profile).expect("fixture profile");
        let expected = case.expected.as_ref().expect("success expectation");
        assert_eq!(expected.rendered_prompts.len(), case.requests.len());
        for (request_index, spec) in case.requests.iter().enumerate() {
            let planned = build_request(spec);
            validate_visual_plan(&planned.request, planned.visuals, request_index)
                .expect("fixture visual plan");
            let rendered =
                render_chat(profile, &planned.request, request_index).expect("fixture render");
            let (expanded, _) = expand_visuals(profile, &rendered, planned.visuals, request_index)
                .expect("fixture expansion");
            assert_eq!(
                rendered.as_bytes(),
                expected.rendered_prompts[request_index].as_bytes(),
                "{}",
                case.id
            );
            assert_eq!(
                expanded.as_bytes(),
                expected.expanded_prompts[request_index].as_bytes(),
                "{}",
                case.id
            );
            assert_eq!(
                digest(&rendered),
                expected.rendered_sha256[request_index],
                "{}",
                case.id
            );
            assert_eq!(
                digest(&expanded),
                expected.expanded_sha256[request_index],
                "{}",
                case.id
            );
        }
    }
}

#[test]
#[ignore = "requires both hash-pinned local model snapshots under reference/.cache"]
fn complete_chat_fixture_token_arrays_match_both_pinned_oracles() {
    let fixture = fixture();
    let registry = ProfileRegistry::bundled().expect("profiles");
    for alias in [ProfileAlias::Qwen3Vl8b, ProfileAlias::Qwen35_9b] {
        let profile = registry.get(alias);
        let processor = TextProcessor::from_local_assets(profile, asset_directory(alias))
            .expect("pinned tokenizer assets");
        for case in fixture
            .cases
            .iter()
            .filter(|case| case.profile == alias.as_str() && case.expected.is_some())
        {
            let planned = case.requests.iter().map(build_request).collect::<Vec<_>>();
            let output = processor
                .prepare_batch(&planned, ResourceLimits::default())
                .expect("fixture text preparation");
            let expected = case.expected.as_ref().expect("success expectation");
            let columns = expected.input_ids.first().expect("fixture row").len();
            assert!(expected.input_ids.iter().all(|row| row.len() == columns));
            assert_eq!(
                output.input_ids.shape(),
                [expected.input_ids.len(), columns],
                "{}",
                case.id
            );
            assert_eq!(
                output.input_ids.as_slice(),
                flatten(&expected.input_ids),
                "{}",
                case.id
            );
            assert_eq!(
                output.attention_mask.as_slice(),
                flatten(&expected.attention_mask),
                "{}",
                case.id
            );
            assert_eq!(
                output.mm_token_type_ids.as_slice(),
                flatten(&expected.mm_token_type_ids),
                "{}",
                case.id
            );
        }
    }
}
