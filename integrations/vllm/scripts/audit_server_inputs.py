"""Probe stock vLLM's HTTP schema and Qwen parser without loading model weights.

The HTTP endpoint here is an explicitly limited schema harness using vLLM's
real ChatCompletionRequest and content parser. It is not an inference server.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import vllm
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from vllm.entrypoints.chat_utils import _parse_chat_message_content_part
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.model_executor.models.qwen2_vl import Qwen2VLMultiModalDataParser


def audit() -> dict:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    def schema_probe(request: ChatCompletionRequest):
        try:
            for message in request.messages:
                parts = message.get("content", [])
                if isinstance(parts, str):
                    parts = [parts]
                for part in parts:
                    _parse_chat_message_content_part(
                        part,
                        SimpleNamespace(model_config=SimpleNamespace(enable_prompt_embeds=False)),
                        wrap_dicts=False,
                        interleave_strings=False,
                    )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"schema_and_text_parser_only": True}

    request = {
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        "max_tokens": 1,
    }
    client = TestClient(app)
    text = client.post("/v1/chat/completions", json=request)
    assert text.status_code == 200, text.text
    request["messages"][0]["content"] = [
        {"type": "image_pixels", "pixel_values": [[0.0]], "image_grid_thw": [[1, 4, 4]]}
    ]
    pixels = client.post("/v1/chat/completions", json=request)
    assert pixels.status_code in {400, 422}, pixels.text

    parser = Qwen2VLMultiModalDataParser(spatial_merge_size=2)
    try:
        parser.parse_mm_data(
            {
                "image": {
                    "pixel_values": torch.zeros((16, 1536)),
                    "image_grid_thw": torch.tensor([[1, 4, 4]]),
                }
            }
        )
    except ValueError as error:
        rejection = str(error)
        assert "image_embeds" in rejection, rejection
    else:
        raise AssertionError("stock Qwen parser unexpectedly accepted pre-encoder pixels")

    source = Path(vllm.__file__).parent
    files = (
        "entrypoints/chat_utils.py",
        "entrypoints/openai/chat_completion/protocol.py",
        "model_executor/models/qwen2_vl.py",
        "model_executor/models/qwen3_vl.py",
    )
    return {
        "scope": "stock HTTP schema/content-parser harness and real Qwen data parser; no inference",
        "packages": {
            name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers")
        },
        "stock_text": {"status": text.status_code, "body": text.json()},
        "prepared_pixel_http": {"status": pixels.status_code, "body": pixels.json()},
        "prepared_pixel_engine_parser": {"rejected": True, "reason": rejection},
        "source_sha256": {
            file: hashlib.sha256((source / file).read_bytes()).hexdigest() for file in files
        },
        "conclusion": "No stock prepared-pixel chat type; Qwen tensor dictionary requires post-encoder image_embeds",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
