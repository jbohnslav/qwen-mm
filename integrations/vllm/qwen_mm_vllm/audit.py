"""Opt-in serving experiment instrumentation; inactive without an output path."""

import asyncio
import json
import os
import threading
import time
from functools import wraps


def record(event, **fields):
    path = os.environ.get("QWEN_MM_AUDIT_PATH")
    if path:
        line = (
            json.dumps({"event": event, "time": time.time(), "pid": os.getpid(), **fields}) + "\n"
        )
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line.encode())
        finally:
            os.close(fd)


def register_audit():
    if not os.environ.get("QWEN_MM_AUDIT_PATH"):
        return
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor

    from .native import NativeImageProcessor

    def wrap(cls):
        original = cls._call_hf_processor
        if getattr(original, "_qwen_mm_audit", False):
            return

        @wraps(original)
        def call(self, *args, **kwargs):
            start = time.perf_counter()
            result = original(self, *args, **kwargs)
            try:
                asyncio.get_running_loop()
                on_loop = True
            except RuntimeError:
                on_loop = False
            grid = result.get("image_grid_thw")
            record(
                "processor",
                implementation=cls.__name__,
                ms=(time.perf_counter() - start) * 1000,
                grid=None if grid is None else grid.tolist(),
                on_event_loop=on_loop,
                thread=threading.current_thread().name,
            )
            return result

        call._qwen_mm_audit = True
        cls._call_hf_processor = call

    wrap(Qwen3VLMultiModalProcessor)
    wrap(NativeImageProcessor)
    cls = Qwen3_5ForConditionalGeneration
    original = cls._process_image_input
    if not getattr(original, "_qwen_mm_audit", False):

        @wraps(original)
        def vision(self, image_input):
            assert image_input["type"] != "image_embeds", "vision encoder bypass detected"
            record(
                "vision",
                grid=image_input["image_grid_thw"].tolist(),
                input_type=image_input["type"],
            )
            return original(self, image_input)

        vision._qwen_mm_audit = True
        cls._process_image_input = vision

    from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer

    forward = Qwen3_VisionTransformer.forward
    if not getattr(forward, "_qwen_mm_audit", False):

        @wraps(forward)
        def tower(self, *args, **kwargs):
            record("vision_tower")
            return forward(self, *args, **kwargs)

        tower._qwen_mm_audit = True
        Qwen3_VisionTransformer.forward = tower
