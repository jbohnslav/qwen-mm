from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version


FIXTURE_COUNT = 24
FIXTURE_WIDTH = 1023
FIXTURE_HEIGHT = 767
JPEG_OPTIONS: dict[str, Any] = {
    "format": "JPEG",
    "quality": 90,
    "subsampling": 2,
    "optimize": False,
    "progressive": False,
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def fixture_directory() -> Path:
    return repository_root() / "fixtures" / "baseline" / "image24"


def manifest_path() -> Path:
    return fixture_directory().parent / "manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_image(index: int) -> Image.Image:
    y, x = np.indices((FIXTURE_HEIGHT, FIXTURE_WIDTH), dtype=np.int32)
    phase = index + 1
    red = (3 * x + y + 17 * phase) % 256
    green = (x + 2 * y + 29 * phase) % 256
    blue = (2 * x + 3 * y + 43 * phase) % 256
    pixels = np.stack((red, green, blue), axis=-1).astype(np.int16)

    rng = np.random.default_rng(0x5157454E + index)
    noise = rng.integers(-6, 7, size=pixels.shape, dtype=np.int16)
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)

    image = Image.fromarray(pixels, mode="RGB")
    draw = ImageDraw.Draw(image)
    margin = 24 + index
    draw.rectangle(
        (margin, margin, FIXTURE_WIDTH - margin - 1, FIXTURE_HEIGHT - margin - 1),
        outline=((37 * phase) % 256, (71 * phase) % 256, (109 * phase) % 256),
        width=5,
    )
    radius = 50 + 3 * index
    center_x = 120 + (index * 73) % (FIXTURE_WIDTH - 240)
    center_y = 120 + (index * 47) % (FIXTURE_HEIGHT - 240)
    draw.ellipse(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
        fill=((97 * phase) % 256, (53 * phase) % 256, (19 * phase) % 256),
        outline=(255, 255, 255),
        width=3,
    )
    draw.line(
        (0, (index * 31) % FIXTURE_HEIGHT, FIXTURE_WIDTH - 1, (index * 83 + 211) % FIXTURE_HEIGHT),
        fill=(255, 255, 255),
        width=4,
    )
    return image


def generate() -> dict[str, Any]:
    directory = fixture_directory()
    directory.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    expected_names = {f"image-{index:02d}.jpg" for index in range(FIXTURE_COUNT)}
    for old_path in directory.glob("*.jpg"):
        if old_path.name not in expected_names:
            old_path.unlink()

    for index in range(FIXTURE_COUNT):
        filename = f"image-{index:02d}.jpg"
        path = directory / filename
        image = _make_image(index)
        image.save(path, **JPEG_OPTIONS)
        entries.append(
            {
                "filename": filename,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    manifest = {
        "schema_version": 1,
        "generator": "qwen_mm_reference.fixtures",
        "generator_version": 1,
        "numpy_version": np.__version__,
        "pillow_version": pillow_version,
        "count": FIXTURE_COUNT,
        "width": FIXTURE_WIDTH,
        "height": FIXTURE_HEIGHT,
        "mode": "RGB",
        "jpeg": {
            "quality": JPEG_OPTIONS["quality"],
            "subsampling": JPEG_OPTIONS["subsampling"],
            "optimize": JPEG_OPTIONS["optimize"],
            "progressive": JPEG_OPTIONS["progressive"],
        },
        "images": entries,
    }
    manifest_path().write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify() -> dict[str, Any]:
    path = manifest_path()
    if not path.exists():
        raise FileNotFoundError(f"missing fixture manifest: {path}")

    manifest = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("count") != len(manifest.get("images", [])):
        errors.append("manifest count does not match image entries")

    for entry in manifest.get("images", []):
        image_path = fixture_directory() / entry["filename"]
        if not image_path.exists():
            errors.append(f"missing {entry['filename']}")
            continue
        actual_size = image_path.stat().st_size
        actual_sha = sha256_file(image_path)
        if actual_size != entry["bytes"]:
            errors.append(
                f"{entry['filename']}: expected {entry['bytes']} bytes, got {actual_size}"
            )
        if actual_sha != entry["sha256"]:
            errors.append(f"{entry['filename']}: SHA-256 mismatch")
        with Image.open(image_path) as image:
            if image.size != (manifest["width"], manifest["height"]):
                errors.append(f"{entry['filename']}: unexpected dimensions {image.size}")
            if image.mode != manifest["mode"]:
                errors.append(f"{entry['filename']}: unexpected mode {image.mode}")

    if errors:
        raise RuntimeError("fixture verification failed:\n- " + "\n- ".join(errors))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "verify"))
    args = parser.parse_args()

    manifest = generate() if args.command == "generate" else verify()
    print(
        f"{args.command}: {manifest['count']} fixtures at "
        f"{manifest['width']}x{manifest['height']} ({manifest_path()})"
    )


if __name__ == "__main__":
    main()
