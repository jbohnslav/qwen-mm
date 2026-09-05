"""Execute the Rosetta Stone's actual code fences against installed wheels."""

from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import copy
import functools
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import tempfile
import threading
import types
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.parse import unquote, urlsplit

import numpy as np
import torch
from PIL import Image
from qwen_mm import Processor, UnsupportedMediaError
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/official-example-rosetta-stone-v0.1.md"
SMALL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


@contextlib.contextmanager
def fixture_server(directory):
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=directory))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def snippets() -> dict[int, list[str]]:
    result = {}
    for section in re.split(r"^## ", DOC.read_text(), flags=re.M)[1:]:
        match = re.match(r"(\d+)\.", section)
        if match:
            result[int(match[1])] = re.findall(r"```python\n(.*?)```", section, flags=re.S)
    assert set(result) == set(range(13)), "Every numbered Rosetta section needs a suite entry"
    return result


def execute(source: str, namespace: dict, replacements: dict[str, str]) -> None:
    class BindFixtures(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in replacements:
                return ast.copy_location(ast.Constant(replacements[node.value]), node)
            return node

    tree = BindFixtures().visit(ast.parse(source))
    exec(compile(tree, str(DOC), "exec"), namespace)


def compare(reference, candidate) -> dict:
    assert list(reference) == list(candidate), (list(reference), list(candidate))
    differences = {}
    for key in reference:
        left, right = reference[key], candidate[key]
        if isinstance(left, torch.Tensor):
            left = left.cpu().numpy()
        if isinstance(right, torch.Tensor):
            right = right.cpu().numpy()
        assert left.shape == right.shape and left.dtype == right.dtype, key
        assert np.isfinite(right).all(), key
        error = float(np.max(np.abs(left.astype(np.float64) - right)))
        differences[key] = error
        if key != "pixel_values":
            np.testing.assert_array_equal(left, right, err_msg=key)
        else:
            # This checks the frozen maximum-error bound; the full resize-v2
            # suite remains responsible for all fidelity metrics.
            assert error <= 64 / 255 + 1e-6, (key, error)
    return differences


def composed(processor, messages, *, thinking=None, generation=False, images=None):
    messages = copy.deepcopy(messages)
    if images is not None:
        for message in messages:
            if isinstance(message.get("content"), list):
                for item in message["content"]:
                    if item.get("type") == "image" and isinstance(item.get("image"), int):
                        item["image"] = images[item["image"]]
    for message in messages:
        if isinstance(message.get("content"), list):
            for item in message["content"]:
                if item.get("type") == "image_url":
                    value = item.pop("image_url")
                    item.update(
                        type="image", image=value["url"] if isinstance(value, dict) else value
                    )
    kwargs = {"enable_thinking": thinking} if thinking is not None else {}
    text = processor.apply_chat_template(
        copy.deepcopy(messages), tokenize=False, add_generation_prompt=generation, **kwargs
    )
    image_inputs, videos = process_vision_info(messages, image_patch_size=16)
    assert videos is None
    return processor(text=[text], images=image_inputs, do_resize=False, return_tensors="pt")


def run(cache: Path, *, local_model: bool, live_urls: bool) -> dict:
    cases = snippets()
    records = []
    wheel = json.loads(importlib.metadata.distribution("qwen-mm").read_text("direct_url.json"))
    source = urlsplit(wheel["url"])
    assert source.scheme == "file", "Install the production wheel from a local file"
    wheel_path = Path(unquote(source.path))
    wheel["archive_info"]["hashes"] = {
        "sha256": hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    }
    report = {
        "platform": platform.platform(),
        "document_sha256": hashlib.sha256(DOC.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "wheel": wheel,
        "packages": {
            name: importlib.metadata.version(name) for name in ("qwen-mm", "torch", "transformers")
        },
        "fixture_policy": "Only placeholder paths, image URLs, and cache locations are bound; code and options execute as published",
        "live_urls": live_urls,
        "cases": records,
    }
    model = None
    if local_model:
        from verify_transformers_consumer import verify_compatible_assets

        small = cache / f"models--Qwen--Qwen3.5-0.8B/snapshots/{SMALL_REVISION}"
        profile = (
            cache / "models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
        )
        report["consumer_asset_witness"] = verify_compatible_assets(small, profile)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            small, local_files_only=True, dtype=torch.float32, attn_implementation="eager"
        ).eval()
        report["consumer"] = {
            "model": "Qwen/Qwen3.5-0.8B",
            "revision": SMALL_REVISION,
            "device": "cpu",
        }
    with (
        tempfile.TemporaryDirectory(prefix="qwen-mm-rosetta-") as temporary,
        fixture_server(temporary) as server,
    ):
        directory = Path(temporary)
        first, second = directory / "one.png", directory / "two.png"
        Image.new("RGB", (128, 128), "red").save(first)
        Image.new("RGB", (288, 256), "blue").save(second)
        y, x = np.mgrid[:128, :128]
        rgb = np.stack((x, y, (x + y) % 256), axis=-1).astype(np.uint8)
        replacements = {
            "file:///path/to/image1.jpg": first.as_uri(),
            "file:///path/to/image2.jpg": second.as_uri(),
            "file:///path/to/your/image.jpg": first.as_uri(),
            "mllm_demo_data/1.jpg": str(first),
        }
        if not live_urls:
            replacements.update(
                {
                    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg": f"{server}/one.png",
                    "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/RealWorld/RealWorld-04.png": f"{server}/two.png",
                }
            )
        report["fixture_bindings"] = {key: "generated PNG fixture" for key in replacements}
        for profile in Processor.supported_profiles():
            snapshot = (
                cache
                / f"models--{profile['model_id'].replace('/', '--')}/snapshots/{profile['revision']}"
            )
            official = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
            native = Processor(profile["profile"], snapshot)
            for number, blocks in cases.items():
                if number in (0, 9, 12) or (
                    number in (6, 10) and profile["profile"] != "qwen3.5-9b"
                ):
                    continue
                assert len(blocks) == 2, (number, len(blocks))
                env = {"processor": official, "rgb": rgb, "Processor": Processor}
                captured = {}
                upstream_failure = None
                if number == 6:
                    module = types.ModuleType("openai")

                    def capture(_target=captured, **kwargs):
                        _target.update(kwargs)

                    module.OpenAI = lambda: types.SimpleNamespace(
                        chat=types.SimpleNamespace(
                            completions=types.SimpleNamespace(create=capture)
                        )
                    )
                    with patch.dict("sys.modules", {"openai": module}):
                        execute(blocks[0], env, replacements)
                    reference = composed(official, env["messages"], thinking=False, generation=True)
                elif number == 8:
                    # The official video processor is outside the v0.1 envelope.
                    tree = ast.parse(blocks[0])
                    assignment = next(
                        node
                        for node in tree.body
                        if isinstance(node, ast.Assign)
                        and any(
                            isinstance(t, ast.Name) and t.id == "messages" for t in node.targets
                        )
                    )
                    exec(
                        compile(ast.Module(body=[assignment], type_ignores=[]), str(DOC), "exec"),
                        env,
                    )
                    reference = None
                else:
                    try:
                        execute(blocks[0], env, replacements)
                        reference = env.get("inputs")
                    except ValueError as error:
                        if number not in (3, 4) or "Incorrect image source" not in str(error):
                            raise
                        upstream_failure = "Pinned Transformers rejects the original file: URI; reference rerun binds those placeholders to plain paths. Native uses the original file: message form."
                        retry = dict(env)
                        plain_paths = {
                            **replacements,
                            "file:///path/to/image1.jpg": str(first),
                            "file:///path/to/image2.jpg": str(second),
                        }
                        execute(blocks[0], retry, plain_paths)
                        reference = retry["inputs"]
                env["processor"] = native
                error = None
                # Section 6's literal constructor uses the same pinned cache.
                construct = Processor.from_pretrained
                with patch.object(
                    Processor,
                    "from_pretrained",
                    side_effect=lambda name, _construct=construct: _construct(
                        name, cache_dir=cache, local_files_only=True
                    ),
                ):
                    try:
                        execute(blocks[1], env, replacements)
                    except UnsupportedMediaError as caught:
                        if number != 8:
                            raise
                        error = caught.category
                if number == 8:
                    assert error == "unsupported_media"
                    record = {
                        "section": number,
                        "profile": profile["profile"],
                        "passed": True,
                        "result": "original video message rejected as unsupported_media",
                    }
                else:
                    candidate = env["inputs"]
                    if number == 7:
                        reference = composed(official, env["messages"], images=env["row"]["images"])
                    record = {
                        "section": number,
                        "profile": profile["profile"],
                        "passed": True,
                        "input_differences": compare(reference, candidate),
                    }
                    if upstream_failure:
                        record["upstream_source_issue"] = upstream_failure
                    if captured:
                        record["transport"] = (
                            "official serving request captured; local oracle is the composed processor, not a server-output comparison"
                        )
                    if model is not None and profile["profile"] == "qwen3.5-9b":
                        # Execute both section 12 fences exactly; no substitute decoder.
                        outputs = {}
                        first_logits = {}
                        for label, inputs, decoder, code in (
                            ("official", reference, official, cases[12][0]),
                            ("qwen_mm", candidate, native, cases[12][1]),
                        ):
                            assert isinstance(next(iter(inputs.values())), torch.Tensor)
                            run_env = {"inputs": inputs, "processor": decoder, "model": model}
                            seen = []

                            def inspect_logits(_module, _args, output, _seen=seen):
                                assert torch.isfinite(output.logits).all()
                                if not _seen:
                                    _seen.append(output.logits[:, -1, :].detach().clone())

                            hook = model.register_forward_hook(inspect_logits)
                            try:
                                with torch.inference_mode():
                                    execute(code, run_env, {})
                            finally:
                                hook.remove()
                            assert seen
                            first_logits[label] = seen[0]
                            assert native.batch_decode(
                                run_env["new_tokens"], skip_special_tokens=True
                            ) == official.batch_decode(
                                run_env["new_tokens"], skip_special_tokens=True
                            )
                            outputs[label] = {
                                "tokens": run_env["new_tokens"].tolist(),
                                "text": run_env["output_text"],
                            }
                        if all(value == 0 for value in record["input_differences"].values()):
                            assert outputs["official"] == outputs["qwen_mm"]
                            torch.testing.assert_close(
                                first_logits["official"], first_logits["qwen_mm"], rtol=0, atol=0
                            )
                        record["generation"] = outputs
                        record["finite_logits"] = True
                        record["first_logits_max_absolute_difference"] = float(
                            (first_logits["official"] - first_logits["qwen_mm"]).abs().max()
                        )
                records.append(record)
                print(json.dumps(record), flush=True)
            # Exercise the same direct snippet with data URI and OpenAI source forms.
            for form in ("data_uri", "openai_url"):
                uri = "data:image/png;base64," + base64.b64encode(first.read_bytes()).decode()
                item = (
                    {"type": "image", "image": uri}
                    if form == "data_uri"
                    else {"type": "image_url", "image_url": {"url": uri}}
                )
                messages = [
                    {
                        "role": "user",
                        "content": [item, {"type": "text", "text": "Describe this image."}],
                    }
                ]
                candidate = native.prepare(
                    messages, min_pixels=65536, return_tensors="pt", add_generation_prompt=True
                )
                reference = official.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                )
                records.append(
                    {
                        "section": 2,
                        "variant": form,
                        "profile": profile["profile"],
                        "passed": True,
                        "input_differences": compare(reference, candidate),
                    }
                )
        # Execute every construction example literally, binding only the cache.
        real_native, real_official = Processor.from_pretrained, AutoProcessor.from_pretrained
        profiles = {record["model_id"]: record for record in Processor.supported_profiles()}

        def construct_official(name):
            record = profiles[name]
            return real_official(
                cache / f"models--{name.replace('/', '--')}/snapshots/{record['revision']}",
                local_files_only=True,
            )

        with (
            patch.object(
                Processor,
                "from_pretrained",
                side_effect=lambda name: real_native(name, cache_dir=cache, local_files_only=True),
            ),
            patch.object(AutoProcessor, "from_pretrained", side_effect=construct_official),
        ):
            namespace = {}
            for code in cases[0]:
                execute(code, namespace, {})
        records.append(
            {"section": 0, "passed": True, "result": "all literal constructor fences executed"}
        )
        records.append(
            {
                "section": 9,
                "passed": True,
                "result": "serving/plugin boundary; shared message schema exercised in section 6, no vLLM/SGLang execution claimed",
            }
        )
        records.append(
            {
                "section": 12,
                "passed": bool(model),
                "result": "both generation fences executed for Qwen3.5 cases"
                if model
                else "requires --local-model",
                "optional": model is None,
            }
        )
    report["all_required_passed"] = all(row["passed"] or row.get("optional") for row in records)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "reference/.cache/huggingface")
    parser.add_argument("--local-model", action="store_true")
    parser.add_argument("--live-urls", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/tmp/qwen-mm-rosetta.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    report = run(args.cache_dir.resolve(), local_model=args.local_model, live_urls=args.live_urls)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Rosetta suite passed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
