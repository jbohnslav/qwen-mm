"""Small, explicitly invoked hosted Qwen contract smoke; never prints credentials."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import time
import types
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/official-example-rosetta-stone-v0.1.md"
API = "https://openrouter.ai/api/v1"
MODEL = "qwen/qwen3.5-9b"


def original_request():
    section = re.search(r"^## 6\..*?(?=^## 7\.)", DOC.read_text(), re.M | re.S)[0]
    code = re.findall(r"```python\n(.*?)```", section, re.S)[0]
    captured = {}
    module = types.ModuleType("openai")
    module.OpenAI = lambda: types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=captured.update))
    )
    with patch.dict("sys.modules", {"openai": module}):
        exec(compile(code, str(DOC), "exec"), {})
    extra = captured.pop("extra_body")
    assert extra.pop("chat_template_kwargs") == {"enable_thinking": False}
    return {**captured, **extra, "model": MODEL, "max_tokens": 128, "reasoning": {"enabled": False}}


def image_item(color):
    output = io.BytesIO()
    Image.new("RGB", (128, 128), color).save(output, format="PNG")
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-usd", type=float, default=1.99)
    args = parser.parse_args()
    assert 0 < args.budget_usd <= 1.99
    key = args.key_file.read_text().strip()
    with urllib.request.urlopen(f"{API}/models", timeout=30) as response:
        metadata = next(row for row in json.load(response)["data"] if row["id"] == MODEL)
    # Reserve a full context at the advertised rate before each call. The seven
    # tiny requests below use only synthetic content and the public Rosetta URL.
    pricing = metadata["pricing"]
    reserve = (
        metadata["context_length"] * float(pricing["prompt"])
        + 512 * float(pricing["completion"])
        + float(pricing.get("request", 0))
        + 2 * float(pricing.get("image", 0))
    )
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "document_sha256": hashlib.sha256(DOC.read_bytes()).hexdigest(),
        "pricing": pricing,
        "budget_usd": args.budget_usd,
        "scope": "Hosted request/media/tool behavior only; no local tensor parity or model-quality claim",
        "section_6_transport_changes": {
            "endpoint": API,
            "model": MODEL,
            "max_tokens": "32768 -> 128 for this bounded smoke",
            "extra_body": "flattened into HTTP JSON; chat_template_kwargs.enable_thinking=False -> reasoning.enabled=False",
            "messages": "executed from the original Markdown without changes, including the public image URL",
        },
        "cases": [],
    }
    spent = 0.0

    def save():
        report["total_cost_usd"] = spent
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def call(name, messages=None, *, check, **kwargs):
        nonlocal spent
        assert spent + reserve < args.budget_usd, "Insufficient remaining test budget"
        body = {
            "model": MODEL,
            "max_tokens": 128,
            "temperature": 0,
            "reasoning": {"enabled": False},
            "messages": messages,
            **kwargs,
        }
        request = urllib.request.Request(
            f"{API}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
        usage = result["usage"]
        spent += float(usage["cost"])
        message = result["choices"][0]["message"]
        passed = bool(check(message))
        record = {
            "name": name,
            "passed": passed,
            "id": result["id"],
            "model": result["model"],
            "provider": result.get("provider"),
            "elapsed_seconds": time.monotonic() - started,
            "usage": usage,
            "finish_reason": result["choices"][0]["finish_reason"],
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls"),
            "reasoning_returned": bool(
                message.get("reasoning") or message.get("reasoning_details")
            ),
        }
        report["cases"].append(record)
        save()
        print(json.dumps(record), flush=True)
        assert passed, name
        return message

    def prompt(text):
        return [{"role": "user", "content": text}]

    call("text", prompt("Reply exactly QWEN_OK."), check=lambda m: "QWEN_OK" in m["content"])
    for colors in (("red",), ("red", "blue")):
        content = [image_item(color) for color in colors]
        content.append(
            {"type": "text", "text": "Name the solid colors in image order, comma separated."}
        )
        call(
            "images_" + "_".join(colors),
            prompt(content),
            check=lambda m, colors=colors: (
                all(color in m["content"].lower() for color in colors)
                and (
                    len(colors) == 1
                    or m["content"].lower().index("red") < m["content"].lower().index("blue")
                )
            ),
        )
    call(
        "section_6_original_public_url",
        **original_request(),
        check=lambda m: bool(m.get("content")),
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "temperature",
                "description": "Read the temperature in a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    messages = prompt(
        "Use the temperature tool for Paris. Then tell me the temperature it returned."
    )

    def valid_tool(message):
        calls = message.get("tool_calls", [])
        return (
            len(calls) == 1
            and calls[0]["function"]["name"] == "temperature"
            and json.loads(calls[0]["function"]["arguments"]).get("city", "").lower() == "paris"
        )

    assistant = call(
        "tool_call",
        messages,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "temperature"}},
        check=valid_tool,
    )
    messages += [
        assistant,
        {
            "role": "tool",
            "tool_call_id": assistant["tool_calls"][0]["id"],
            "content": '{"temperature_c": 17}',
        },
    ]
    call(
        "tool_result",
        messages,
        tools=tools,
        tool_choice="none",
        check=lambda m: "17" in m["content"],
    )
    call(
        "thinking_enabled",
        prompt("What is 2 + 3? Answer briefly."),
        reasoning={"enabled": True},
        max_tokens=512,
        check=lambda m: (
            "5" in (m.get("content") or "")
            and bool(m.get("reasoning") or m.get("reasoning_details"))
        ),
    )
    report["all_required_passed"] = True
    save()


if __name__ == "__main__":
    main()
