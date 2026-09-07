"""Bounded paired HTTP serving experiment, run inside one GPU allocation."""

from __future__ import annotations

import base64
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

import httpx
import numpy as np
import psutil
from PIL import Image

MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
BASE = "http://127.0.0.1:8000"


def payload(kind, seed):
    rng = np.random.default_rng(seed)
    sizes = {
        "text": [],
        "single": [(256, 256)],
        "multi": [(256, 256), (256, 512)],
        "heavy": [(1024, 1024)] * 4,
        "repeat": [(256, 256)] * 2,
    }[kind]
    content = [
        {
            "type": "text",
            "text": "Describe the image colors briefly." if sizes else "Reply with hello.",
        }
    ]
    encoded = []
    for h, w in sizes:
        data = rng.integers(0, 256, (h, w, 3), np.uint8)
        # Keep an aligned deterministic solid-color witness for exact equivalence.
        if kind in ("single", "repeat"):
            data[:] = (240, (20 + seed) % 256, 30 if kind == "single" else 60)
        stream = io.BytesIO()
        Image.fromarray(data).save(stream, format="PNG")
        encoded.append("data:image/png;base64," + base64.b64encode(stream.getvalue()).decode())
        content += [
            {"type": "image_url", "image_url": {"url": encoded[-1]}},
            {"type": "text", "text": "And this one."},
        ]
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 16,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def request(body):
    wire = json.dumps(body).encode()
    start = time.perf_counter()
    first = None
    chunks = []
    usage = None
    with (
        httpx.Client(timeout=180) as client,
        client.stream(
            "POST",
            BASE + "/v1/chat/completions",
            content=wire,
            headers={"Content-Type": "application/json"},
        ) as response,
    ):
        if response.is_error:
            response.read()
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:2000]}")
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                text = delta.get("content") or delta.get("reasoning") or ""
                if text:
                    first = first or time.perf_counter()
                    chunks.append(text)
    end = time.perf_counter()
    assert first is not None and usage is not None, (chunks, usage)
    return {
        "ttft_ms": (first - start) * 1000,
        "latency_ms": (end - start) * 1000,
        "payload_bytes": len(wire),
        "usage": usage,
        "text": "".join(chunks),
    }


def verify_events(events, mode):
    processors = [event for event in events if event["event"] == "processor" and event["grid"]]
    expected = "NativeImageProcessor" if mode == "native" else "Qwen3VLMultiModalProcessor"
    assert processors, "No image preprocessing observed; plugin audit was not loaded"
    assert all(event["implementation"] == expected for event in processors), (
        "Unexpected processor path"
    )
    assert not any(event["on_event_loop"] for event in processors), "CPU processing on event loop"
    vision = sum(event["event"] == "vision" for event in events)
    tower = sum(event["event"] == "vision_tower" for event in events)
    assert vision > 0 and vision == tower, "Expected one vision-tower call per pixel input batch"
    return {
        "image_processor_calls": len(processors),
        "vision_batches": vision,
        "vision_tower_calls": tower,
    }


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "model": MODEL,
        "revision": REVISION,
        "versions": {p: version(p) for p in ["vllm", "torch", "transformers", "qwen-mm", "numpy"]},
        "runs": [],
        "failures": [],
    }
    report["installed_packages"] = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    ).splitlines()
    import hashlib

    report["integration_sha256"] = {
        str(path.relative_to(Path(__file__).parents[1])): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in Path(__file__).parents[1].rglob("*.py")
        if "__pycache__" not in str(path)
    }
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    report["gpu"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    # ABBA order controls for systematic order effects. Each process starts empty caches.
    for index, mode in enumerate(["stock", "native", "native", "stock"]):
        print(f"Starting {index}: {mode}", flush=True)
        audit = output / f"{index}-{mode}-audit.jsonl"
        env = {
            **os.environ,
            "VLLM_PLUGINS": "qwen_mm_serving_audit"
            + (",qwen_mm_native_images" if mode == "native" else ""),
            "QWEN_MM_AUDIT_PATH": str(audit),
            "QWEN_MM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            # The Modal image supplies the matching CUDA development toolkit.
            "VLLM_USE_FLASHINFER_SAMPLER": "1",
        }
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            MODEL,
            "--revision",
            REVISION,
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "4096",
            "--max-num-seqs",
            "4",
            "--gpu-memory-utilization",
            "0.80",
            "--enforce-eager",
            "--no-enable-prefix-caching",
            "--limit-mm-per-prompt",
            '{"image":4,"video":0}',
            "--mm-processor-kwargs",
            '{"min_pixels":65536,"max_pixels":262144}',
            "--mm-processor-cache-gb",
            "1",
        ]
        samples, health = [], []
        stop = threading.Event()

        def monitor(server, stop=stop, health=health, samples=samples):
            with httpx.Client(timeout=3) as client:
                while not stop.wait(0.05):
                    start = time.perf_counter()
                    try:
                        client.get(BASE + "/health").raise_for_status()
                        health.append((time.perf_counter() - start) * 1000)
                    except httpx.HTTPError:
                        pass
                    try:
                        processes = [
                            psutil.Process(server.pid),
                            *psutil.Process(server.pid).children(recursive=True),
                        ]
                        samples.append(
                            sum(p.memory_info().rss for p in processes if p.is_running())
                        )
                    except psutil.Error:
                        pass

        with (output / f"{index}-{mode}-server.log").open("w") as log:
            server = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                deadline = time.monotonic() + 600
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        raise RuntimeError(f"{mode} exited {server.returncode}; inspect server log")
                    try:
                        if httpx.get(BASE + "/health", timeout=2).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(1)
                else:
                    raise TimeoutError("server readiness timeout")
                request(payload("text", 0))
                request(payload("single", 0))
                start_time = time.time()
                observer = threading.Thread(target=monitor, args=(server,), daemon=True)
                observer.start()
                rows = []
                for repeat in range(3):
                    for kind in ["text", "single", "multi", "repeat", "heavy"]:
                        body = payload(kind, 100 + repeat)
                        for state in ["cold", "warm"]:
                            # Each cold body has new content; its immediate repeat hits caches.
                            rows.append(
                                {
                                    "kind": kind,
                                    "requested_state": state,
                                    "repeat": repeat,
                                    **request(body),
                                }
                            )
                    bodies = [payload("heavy", 1000 + repeat * 4 + i) for i in range(4)]
                    begin = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        concurrent = list(pool.map(request, bodies))
                    duration = time.perf_counter() - begin
                    rows += [
                        {
                            "kind": "concurrent",
                            "requested_state": "cold",
                            "repeat": repeat,
                            "batch_seconds": duration,
                            **row,
                        }
                        for row in concurrent
                    ]
                stop.set()
                observer.join(timeout=4)
                events = [json.loads(line) for line in audit.read_text().splitlines()]
                witness = verify_events(
                    [event for event in events if event["time"] >= start_time], mode
                )
                report["runs"].append(
                    {
                        "witness": witness,
                        "index": index,
                        "mode": mode,
                        "command": command,
                        "start_time": start_time,
                        "rows": rows,
                        "health_latency_ms": health,
                        "peak_process_tree_rss_bytes": max(samples, default=0),
                    }
                )
            except Exception as exc:
                report["failures"].append({"index": index, "mode": mode, "error": repr(exc)})
                raise
            finally:
                stop.set()
                if server.poll() is None:
                    os.killpg(server.pid, signal.SIGTERM)
                    try:
                        server.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(server.pid, signal.SIGKILL)
                        server.wait()
                (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        time.sleep(3)
    return report


if __name__ == "__main__":
    run(Path(sys.argv[1]))
