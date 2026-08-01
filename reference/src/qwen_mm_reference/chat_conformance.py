from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoProcessor

from .fixtures import repository_root, sha256_file

COMPATIBILITY_PATH = Path("reference/compatibility/v1.json")
DEFAULT_OUTPUT = Path("reference/conformance/v1/chat.json")
GENERATOR_COMMAND = (
    "./scripts/with-cargo.sh uv run --locked --no-sync --package "
    "qwen-mm-reference python -m qwen_mm_reference.chat_conformance"
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


TOOLS = [
    _json(
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Weather for a naïve café visitor",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    )
]
TOOL_ARGUMENTS = _json(
    {
        "city": "Montréal",
        "units": ["c", "f"],
        "flag": True,
        "none": None,
        "epsilon": 1e-7,
    }
)


def _text(role: str, content: str, **extra: Any) -> dict[str, Any]:
    return {"role": role, "content": content, **extra}


def _roles_tools(profile: str) -> dict[str, Any]:
    final: dict[str, Any] = _text("assistant", "done")
    options: dict[str, Any] = {
        "add_generation_prompt": True,
        "add_vision_id": False,
        "tools_json": TOOLS,
    }
    if profile == "qwen3.5-9b":
        final["reasoning_content"] = " compact hidden reasoning "
        options["enable_thinking"] = False
    return {
        "id": f"{profile}-roles-tools",
        "profile": profile,
        "requests": [
            {
                "messages": [
                    _text("system", "  concise café system  "),
                    _text("user", "weather in Montréal? 👩🏽‍💻"),
                    _text(
                        "assistant",
                        "checking",
                        tool_calls=[
                            {
                                "shape": "direct",
                                "name": "weather",
                                "arguments_json": TOOL_ARGUMENTS,
                            },
                            {
                                "shape": "function",
                                "name": "weather",
                                "arguments_json": _json({"city": "東京"}),
                            },
                        ],
                    ),
                    _text("tool", "first result"),
                    _text("tool", "second result"),
                    final,
                ],
                "options": options,
                "visuals": [],
            }
        ],
    }


def _visual_batch(profile: str) -> dict[str, Any]:
    first_items = [
        {"type": "image", "input_index": 0},
        {"type": "text", "text": "α"},
        {"type": "video", "input_index": 0},
    ]
    second_items = [
        {"type": "video", "input_index": 0},
        {"type": "text", "text": "interleave"},
        {"type": "image", "input_index": 0},
        {"type": "text", "text": "repeat"},
        {"type": "image", "input_index": 0},
    ]
    return {
        "id": f"{profile}-visual-padding",
        "profile": profile,
        "requests": [
            {
                "messages": [{"role": "user", "content": first_items}],
                "options": {
                    "add_generation_prompt": False,
                    "add_vision_id": False,
                },
                "visuals": [
                    {
                        "kind": "image",
                        "input_index": 0,
                        "grid_thw": [1, 4, 4],
                    },
                    {
                        "kind": "video",
                        "input_index": 0,
                        "grid_thw": [1, 4, 4],
                        "timestamps": [0.0],
                    },
                ],
            },
            {
                "messages": [{"role": "user", "content": second_items}],
                "options": {
                    "add_generation_prompt": True,
                    "add_vision_id": True,
                },
                "visuals": [
                    {
                        "kind": "video",
                        "input_index": 0,
                        "grid_thw": [2, 4, 4],
                        "timestamps": [0.0, 0.5],
                    },
                    {
                        "kind": "image",
                        "input_index": 0,
                        "grid_thw": [1, 4, 6],
                    },
                    {
                        "kind": "image",
                        "input_index": 0,
                        "grid_thw": [1, 6, 4],
                    },
                ],
            },
        ],
    }


def _qwen35_thinking(name: str, value: bool | None) -> dict[str, Any]:
    options: dict[str, Any] = {"add_generation_prompt": True}
    if value is not None:
        options["enable_thinking"] = value
    return {
        "id": f"qwen3.5-9b-thinking-{name}",
        "profile": "qwen3.5-9b",
        "requests": [
            {
                "messages": [_text("user", "think mode")],
                "options": options,
                "visuals": [],
            }
        ],
    }


def _qwen35_reasoning_extraction() -> dict[str, Any]:
    return {
        "id": "qwen3.5-9b-reasoning-extraction",
        "profile": "qwen3.5-9b",
        "requests": [
            {
                "messages": [
                    _text("user", "hello"),
                    _text(
                        "assistant",
                        "<think>\n internal chain \n</think>\nvisible answer",
                    ),
                ],
                "options": {"add_generation_prompt": False},
                "visuals": [],
            }
        ],
    }


def _error_cases() -> list[dict[str, Any]]:
    def rejected(case_id: str, profile: str, option: str) -> dict[str, Any]:
        return {
            "id": case_id,
            "profile": profile,
            "requests": [
                {
                    "messages": [_text("user", "rejected")],
                    "options": {"excluded": [option]},
                    "visuals": [],
                }
            ],
            "expected_error": {"category": "unsupported_option"},
        }

    return [
        {
            "id": "qwen3-vl-literal-image-token",
            "profile": "qwen3-vl-8b",
            "requests": [
                {
                    "messages": [_text("user", "literal <|image_pad|>")],
                    "options": {},
                    "visuals": [],
                }
            ],
            "expected_error": {"category": "invalid_request"},
        },
        {
            "id": "qwen3.5-literal-video-token",
            "profile": "qwen3.5-9b",
            "requests": [
                {
                    "messages": [_text("user", "literal <|video_pad|>")],
                    "options": {},
                    "visuals": [],
                }
            ],
            "expected_error": {"category": "invalid_request"},
        },
        rejected("qwen3-vl-rejected-template", "qwen3-vl-8b", "chat_template"),
        rejected("qwen3.5-rejected-truncation", "qwen3.5-9b", "truncation"),
        {
            "id": "qwen3-vl-rejected-thinking",
            "profile": "qwen3-vl-8b",
            "requests": [
                {
                    "messages": [_text("user", "thinking")],
                    "options": {
                        "add_generation_prompt": True,
                        "enable_thinking": True,
                    },
                    "visuals": [],
                }
            ],
            "expected_error": {"category": "unsupported_option"},
        },
        {
            "id": "qwen3-vl-rejected-reasoning",
            "profile": "qwen3-vl-8b",
            "requests": [
                {
                    "messages": [
                        _text("user", "reasoning"),
                        _text(
                            "assistant",
                            "answer",
                            reasoning_content="hidden",
                        ),
                    ],
                    "options": {},
                    "visuals": [],
                }
            ],
            "expected_error": {"category": "unsupported_option"},
        },
        {
            "id": "unknown-profile",
            "profile": "qwen-unknown",
            "requests": [],
            "expected_error": {"category": "profile_mismatch"},
        },
    ]


def cases() -> list[dict[str, Any]]:
    return [
        _roles_tools("qwen3-vl-8b"),
        _roles_tools("qwen3.5-9b"),
        _visual_batch("qwen3-vl-8b"),
        _visual_batch("qwen3.5-9b"),
        _qwen35_thinking("default", None),
        _qwen35_thinking("false", False),
        _qwen35_thinking("true", True),
        _qwen35_reasoning_extraction(),
        *_error_cases(),
    ]


def _snapshot_directory(profile: dict[str, Any]) -> Path:
    cache_name = "models--" + profile["model_id"].replace("/", "--")
    return (
        repository_root()
        / "reference"
        / ".cache"
        / "huggingface"
        / cache_name
        / "snapshots"
        / profile["revision"]
    )


def _validate_assets(profile: dict[str, Any], directory: Path) -> None:
    template = (
        "chat_template.jinja"
        if "chat_template.jinja" in profile["artifacts"]
        else "chat_template.json"
    )
    for name in ("tokenizer.json", "tokenizer_config.json", template):
        path = directory / name
        expected = profile["artifacts"][name]
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"{name}: expected {expected}, got {actual}")


def _official_message(message: dict[str, Any]) -> dict[str, Any]:
    content = message["content"]
    if isinstance(content, list):
        converted = []
        for item in content:
            if item["type"] == "text":
                converted.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image":
                converted.append({"type": "image", "image": "local-placeholder"})
            else:
                converted.append({"type": "video", "video": "local-placeholder"})
        content = converted
    result: dict[str, Any] = {"role": message["role"], "content": content}
    if "reasoning_content" in message:
        result["reasoning_content"] = message["reasoning_content"]
    if calls := message.get("tool_calls"):
        converted_calls = []
        for call in calls:
            function = {
                "name": call["name"],
                "arguments": json.loads(call["arguments_json"]),
            }
            converted_calls.append(
                {"function": function} if call["shape"] == "function" else function
            )
        result["tool_calls"] = converted_calls
    return result


def _replacement(visual: dict[str, Any]) -> str:
    temporal, height, width = visual["grid_thw"]
    if visual["kind"] == "image":
        return "<|image_pad|>" * (temporal * height * width // 4)
    frame_tokens = height * width // 4
    return "".join(
        f"<{timestamp:.1f} seconds><|vision_start|>"
        + "<|video_pad|>" * frame_tokens
        + "<|vision_end|>"
        for timestamp in visual["timestamps"]
    )


def _render_case(processor: Any, case: dict[str, Any]) -> dict[str, Any]:
    rendered = []
    image_replacements = []
    video_replacements = []
    for request in case["requests"]:
        options = request.get("options", {})
        kwargs = {
            "add_generation_prompt": options.get("add_generation_prompt", False),
            "add_vision_id": options.get("add_vision_id", False),
        }
        if "enable_thinking" in options:
            kwargs["enable_thinking"] = options["enable_thinking"]
        if tools := options.get("tools_json"):
            kwargs["tools"] = [json.loads(tool) for tool in tools]
        messages = [_official_message(message) for message in request["messages"]]
        rendered.append(processor.apply_chat_template(messages, tokenize=False, **kwargs))
        for visual in request.get("visuals", []):
            target = image_replacements if visual["kind"] == "image" else video_replacements
            target.append(_replacement(visual))

    expanded, _ = processor.get_text_with_replacements(
        rendered.copy(), image_replacements, video_replacements
    )
    encoded = processor.tokenizer(
        expanded,
        padding=True,
        truncation=False,
        add_special_tokens=True,
        return_attention_mask=True,
    )
    input_ids = encoded["input_ids"]
    return {
        "rendered_prompts": rendered,
        "expanded_prompts": expanded,
        "rendered_sha256": [
            hashlib.sha256(value.encode("utf-8")).hexdigest() for value in rendered
        ],
        "expanded_sha256": [
            hashlib.sha256(value.encode("utf-8")).hexdigest() for value in expanded
        ],
        "input_ids": input_ids,
        "attention_mask": encoded["attention_mask"],
        "mm_token_type_ids": processor.create_mm_token_type_ids(input_ids),
    }


def generate(output: Path) -> dict[str, Any]:
    root = repository_root()
    compatibility = json.loads((root / COMPATIBILITY_PATH).read_text(encoding="utf-8"))
    processors: dict[str, Any] = {}
    generated_cases = []
    for case in cases():
        copied = json.loads(json.dumps(case, ensure_ascii=False))
        if "expected_error" not in copied:
            alias = copied["profile"]
            if alias not in processors:
                profile = compatibility["profiles"][alias]
                directory = _snapshot_directory(profile)
                _validate_assets(profile, directory)
                processors[alias] = AutoProcessor.from_pretrained(directory, local_files_only=True)
            copied["expected"] = _render_case(processors[alias], copied)
        generated_cases.append(copied)

    document = {
        "schema_version": 1,
        "contract_id": compatibility["contract_id"],
        "generator": "qwen_mm_reference.chat_conformance",
        "generator_command": GENERATOR_COMMAND,
        "profiles": {
            alias: {
                "revision": profile["revision"],
                "fingerprint": profile["fingerprint"],
            }
            for alias, profile in compatibility["profiles"].items()
        },
        "cases": generated_cases,
    }
    output_path = root / output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return document


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate compact pinned chat conformance fixtures."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    document = generate(args.output)
    print(f"generated {len(document['cases'])} cases at {args.output}")


if __name__ == "__main__":
    main()
