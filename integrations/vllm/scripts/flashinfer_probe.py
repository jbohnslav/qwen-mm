"""Validate the optional FlashInfer sampler against the image's wheel CUDA toolkit."""

import json
import os
import subprocess
from pathlib import Path

import flashinfer
import torch


def run():
    generator = torch.Generator(device="cuda").manual_seed(123)
    logits = torch.randn((4, 256), device="cuda", generator=generator)
    ids = flashinfer.sampling.top_k_top_p_sampling_from_logits(logits, 50, 0.9)
    torch.cuda.synchronize()
    assert ids.shape == (4,) and bool(((ids >= 0) & (ids < 256)).all())
    return {
        "cuda_home": os.environ["CUDA_HOME"],
        "nvcc": subprocess.check_output(["nvcc", "--version"], text=True),
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "sampled_ids": ids.tolist(),
    }


if __name__ == "__main__":
    import sys

    Path(sys.argv[1]).write_text(json.dumps(run(), indent=2) + "\n")
