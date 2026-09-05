"""Opt-in real-model contract smoke; inference latency is never a release gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from qwen_mm import Processor
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
PROFILE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
IDENTICAL_ASSETS = (
    "tokenizer.json",
    "merges.txt",
    "vocab.json",
    "preprocessor_config.json",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_compatible_assets(small: Path, profile: Path) -> dict:
    hashes = {}
    for filename in IDENTICAL_ASSETS:
        hashes[filename] = sha256(small / filename)
        assert hashes[filename] == sha256(profile / filename), filename
    small_tokenizer = json.loads((small / "tokenizer_config.json").read_text())
    profile_tokenizer = json.loads((profile / "tokenizer_config.json").read_text())
    small_tokenizer.pop("chat_template")
    profile_tokenizer.pop("chat_template")
    assert small_tokenizer == profile_tokenizer, "non-template tokenizer settings differ"
    small_config = json.loads((small / "config.json").read_text())
    profile_config = json.loads((profile / "config.json").read_text())
    for key in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
        assert small_config[key] == profile_config[key], key
    for key in ("patch_size", "temporal_patch_size", "spatial_merge_size", "in_channels"):
        assert small_config["vision_config"][key] == profile_config["vision_config"][key], key
    return hashes


def conversation(text: str, *images: Path, resize: bool = False) -> list[dict]:
    content = []
    for image in images:
        item = {"type": "image", "image": str(image)}
        if resize:
            item.update(resized_height=128, resized_width=160)
        content.append(item)
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def official_inputs(processor, conversations: list[list[dict]]) -> dict:
    # Match the repository's composed qwen-vl-utils -> Transformers oracle.
    text = [
        processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for messages in conversations
    ]
    images, videos = process_vision_info(conversations, image_patch_size=16)
    assert videos is None
    return dict(
        processor(
            text=text,
            images=images,
            padding=True,
            truncation=False,
            do_resize=False,
            return_tensors="np",
        )
    )


def run_case(model, official, native, name: str, conversations: list[list[dict]]) -> dict:
    reference = official_inputs(official, conversations)
    if len(conversations) == 1:
        prepared = native.prepare(
            conversations[0], add_generation_prompt=True, enable_thinking=False, padding_side="left"
        )
    else:
        prepared = native.prepare_batch(
            [
                {
                    "messages": messages,
                    "options": {"add_generation_prompt": True, "enable_thinking": False},
                }
                for messages in conversations
            ],
            padding_side="left",
        )
        assert any(row["left_padding"] > 0 for row in prepared.metadata["request_layouts"])
    assert list(prepared) == list(reference), (list(prepared), list(reference))
    differences = {}
    for key, value in prepared.items():
        expected = reference[key]
        assert value.shape == expected.shape and value.dtype == expected.dtype, key
        assert np.isfinite(value).all(), key
        differences[key] = float(np.max(np.abs(value.astype(np.float64) - expected)))
        if key != "pixel_values" or name != "resized_image":
            np.testing.assert_array_equal(value, expected, err_msg=f"{name}/{key}")

    generations = {}
    scores = {}
    elapsed = {}
    for label, arrays in (("official", reference), ("qwen_mm", prepared)):
        # No key filtering, repacking, dtype conversion, or second image processor.
        tensors = {key: torch.from_numpy(value) for key, value in arrays.items()}
        for key, tensor in tensors.items():
            assert tensor.data_ptr() == arrays[key].ctypes.data, key
        started = time.monotonic()
        with torch.inference_mode():
            output = model.generate(
                **tensors,
                max_new_tokens=8,
                do_sample=False,
                return_dict_in_generate=True,
                output_logits=True,
            )
        elapsed[label] = round(time.monotonic() - started, 3)
        assert output.logits and all(torch.isfinite(logits).all() for logits in output.logits)
        scores[label] = output.logits[0]
        suffix = output.sequences[:, tensors["input_ids"].shape[1] :]
        assert suffix.shape[1] > 0
        generations[label] = {
            "token_ids": suffix.tolist(),
            "text": official.batch_decode(suffix, skip_special_tokens=True),
        }
    logits_max_error = float((scores["official"] - scores["qwen_mm"]).abs().max())
    if name != "resized_image":
        torch.testing.assert_close(scores["official"], scores["qwen_mm"], rtol=0, atol=0)
        assert generations["official"]["token_ids"] == generations["qwen_mm"]["token_ids"]
    result = {
        "case": name,
        "passed": True,
        "shapes": {key: list(value.shape) for key, value in prepared.items()},
        "input_max_absolute_difference": differences,
        "first_token_logits_max_absolute_difference": logits_max_error,
        "generations": generations,
        "inference_seconds_diagnostic_only": elapsed,
        "equivalence": "structural only; resize-v2 owns pixel fidelity"
        if name == "resized_image"
        else "exact inputs, first-token logits, and greedy token IDs",
    }
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "reference/.cache/huggingface")
    parser.add_argument("--download", action="store_true", help="Fetch the pinned 0.8B weights")
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/qwen-mm-transformers-consumer.json")
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    small = Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=args.cache_dir,
            local_files_only=not args.download,
            allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors"],
        )
    )
    native = Processor.from_pretrained(
        "Qwen3.5", cache_dir=args.cache_dir, local_files_only=not args.download, thread_budget=2
    )
    assert native.revision == PROFILE_REVISION
    profile = args.cache_dir / f"models--Qwen--Qwen3.5-9B/snapshots/{PROFILE_REVISION}"
    assets = verify_compatible_assets(small, profile)
    official = AutoProcessor.from_pretrained(small, local_files_only=True)
    profile_official = AutoProcessor.from_pretrained(profile, local_files_only=True)
    official.tokenizer.padding_side = "left"
    print("Loading pinned Qwen3.5-0.8B on CPU in float32", flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        small, local_files_only=True, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    report = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "processor_profile": native.profile,
        "processor_revision": native.revision,
        "processor_fingerprint": native.profile_fingerprint,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "script_sha256": sha256(Path(__file__)),
        "uv_lock_sha256": sha256(ROOT / "uv.lock"),
        "platform": platform.platform(),
        "device": "cpu",
        "dtype": "float32",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("qwen-mm", "torch", "transformers", "numpy", "qwen-vl-utils")
        },
        "installed_distribution": json.loads(
            importlib.metadata.distribution("qwen-mm").read_text("direct_url.json") or "{}"
        ),
        "identical_processor_assets_sha256": assets,
        "model_weights_sha256": {
            path.name: sha256(path) for path in sorted(small.glob("*.safetensors"))
        },
        "thinking": "explicit False; 0.8B and 9B differ in their omitted-option default",
        "scope": "0.8B consumer of the 9B processor profile; not new model support or a speed gate",
        "cases": [],
    }
    with tempfile.TemporaryDirectory(prefix="qwen-mm-consumer-images-") as temporary:
        directory = Path(temporary)
        red, blue, patterned = (directory / name for name in ("red.png", "blue.png", "pattern.png"))
        Image.new("RGB", (256, 256), (255, 0, 0)).save(red)
        Image.new("RGB", (288, 256), (0, 0, 255)).save(blue)
        y, x = np.mgrid[:193, :257]
        pixels = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
        Image.fromarray(pixels).save(patterned)
        aligned_pattern = directory / "aligned-pattern.png"
        y, x = np.mgrid[:256, :256]
        pixels = np.stack((x % 256, y % 256, (3 * x + 7 * y) % 256), axis=-1).astype(np.uint8)
        Image.fromarray(pixels).save(aligned_pattern)
        text = conversation("Reply with the word hello.")
        single = conversation("What color is this image?", red)
        cases = [
            ("text", [text]),
            ("single_image", [single]),
            ("patterned_image", [conversation("Describe the colors.", aligned_pattern)]),
            ("multi_image", [conversation("Name the two image colors in order.", red, blue)]),
            ("left_padded_batch", [single, text]),
            ("resized_image", [conversation("Describe the colors.", patterned, resize=True)]),
        ]
        for name, conversations in cases:
            for messages in conversations:
                for thinking in (False, True):
                    kwargs = dict(
                        tokenize=False, add_generation_prompt=True, enable_thinking=thinking
                    )
                    assert official.apply_chat_template(messages, **kwargs) == (
                        profile_official.apply_chat_template(messages, **kwargs)
                    ), "small model chat rendering differs with explicit thinking"
            report["cases"].append(run_case(model, official, native, name, conversations))
    report["all_passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"All {len(report['cases'])} consumer cases passed; report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
