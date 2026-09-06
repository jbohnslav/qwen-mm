"""Send a normal image request to stock vLLM or the qwen-mm server plugin."""

import argparse
import base64
import json
import mimetypes
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--prompt", default="Describe this image.")
    args = parser.parse_args()
    mime = mimetypes.guess_type(args.image.name)[0] or "application/octet-stream"
    data = base64.b64encode(args.image.read_bytes()).decode()
    body = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ],
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        args.url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        print(json.loads(response.read())["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
