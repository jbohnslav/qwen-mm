"""Check benchmark noise images against the existing resize-v2 quality gates.

Run with the repository's reference/src on PYTHONPATH and qwen-vl-utils 0.0.14.
This compares the pinned Transformers image processor at the serving budget;
it does not replace the separate frozen-corpus certification.
"""

import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image
from qwen_mm import Processor
from qwen_mm_reference.resize_conformance_v2 import unpatchify_rgb8
from qwen_mm_reference.resize_quality_v2 import GATES, channel_metrics
from transformers import AutoProcessor

MODELS = {
    "Qwen/Qwen3-VL-8B-Instruct": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
}


def run():
    rows = []
    for model, revision in MODELS.items():
        native = Processor.from_pretrained(model, thread_budget=1)
        hf = AutoProcessor.from_pretrained(model, revision=revision)
        for seed in [100, 101, 102, *range(1000, 1012)]:
            rng = np.random.default_rng(seed)
            for index in range(4):
                array = rng.integers(0, 256, (1024, 1024, 3), np.uint8)
                candidate = native.prepare(
                    [{"role": "user", "content": [{"type": "image", "image": 0}]}],
                    images=[array],
                    min_pixels=65536,
                    max_pixels=262144,
                )
                reference = hf.image_processor(
                    images=[Image.fromarray(array)],
                    size={"shortest_edge": 65536, "longest_edge": 262144},
                    return_tensors="np",
                )
                assert (
                    candidate["image_grid_thw"].tolist()
                    == reference["image_grid_thw"].tolist()
                    == [[1, 32, 32]]
                )
                expected = unpatchify_rgb8(reference["pixel_values"], 512, 512)
                actual = unpatchify_rgb8(candidate["pixel_values"], 512, 512)
                metrics = [channel_metrics(expected[:, :, c], actual[:, :, c]) for c in range(3)]
                rows.append({"model": model, "seed": seed, "image": index, "channels": metrics})
    return {
        "scope": "Both-profile benchmark noise images; frozen resize-v2 gates reused unchanged",
        "models": MODELS,
        "gates": GATES,
        "packages": {
            p: version(p) for p in ["qwen-mm", "transformers", "numpy", "torch", "torchvision"]
        },
        "rows": rows,
    }


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(json.dumps(run(), indent=2) + "\n")
