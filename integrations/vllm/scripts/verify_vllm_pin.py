#!/usr/bin/env python3
"""Verify the immutable vLLM source pin and the A6 seam assumptions."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("vllm_source", type=Path)
    parser.add_argument(
        "--pin",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "vllm-pin.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pin = json.loads(args.pin.read_text())
    source = args.vllm_source.resolve()
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert commit == pin["vllm"]["git_commit"], (commit, pin["vllm"]["git_commit"])

    for relative, expected in pin["source_sha256"].items():
        actual = hashlib.sha256((source / relative).read_bytes()).hexdigest()
        assert actual == expected, f"source hash mismatch for {relative}: {actual}"

    qwen = (source / "vllm/model_executor/models/qwen3_vl.py").read_text()
    parser = (source / "vllm/model_executor/models/qwen2_vl.py").read_text()
    renderer = (source / "vllm/renderers/base.py").read_text()
    processing = (source / "vllm/multimodal/processing/processor.py").read_text()
    registry = (source / "vllm/multimodal/registry.py").read_text()
    input_processor = (source / "vllm/v1/engine/input_processor.py").read_text()
    async_llm = (source / "vllm/v1/engine/async_llm.py").read_text()
    encoder_budget = (source / "vllm/multimodal/encoder_budget.py").read_text()
    qwen35 = (source / "vllm/model_executor/models/qwen3_5.py").read_text()

    assert "class Qwen3VLMultiModalProcessor(BaseMultiModalProcessor" in qwen
    assert 'required_fields={"image_embeds", "image_grid_thw"}' in parser
    assert "pixel_values=MultiModalFieldConfig.flat_from_sizes" in parser
    assert "class MultiModalRegistry" in registry
    assert "def register_processor(" in registry
    assert "mm_hashes = inputs.get_mm_hashes(self.info.model_id)" in processing
    assert "ThreadPoolExecutor(max_workers=pool_workers)" in renderer
    assert "self._process_multimodal, executor=self._mm_executor" in renderer
    assert "Multimodal preprocessing is always offloaded" in renderer
    assert "processed_inputs = self.input_preprocessor.preprocess(" in input_processor
    assert "request = self.input_processor.process_inputs(" in async_llm
    assert "mm_budget.reset_cache()  # Not used anymore" in input_processor
    assert "Qwen3VLMultiModalProcessor," in qwen35
    assert "info=Qwen3_5ProcessingInfo," in qwen35
    assert "cache = mm_registry.processor_only_cache_from_config" in encoder_budget
    print(f"verified vLLM {pin['vllm']['version']} at {commit}")


if __name__ == "__main__":
    main()
