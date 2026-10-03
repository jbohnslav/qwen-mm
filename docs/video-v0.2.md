# Video preprocessing in v0.2

Video uses the same `Processor.prepare` and `prepare_batch` calls as images.
Both pinned Qwen3-VL and Qwen3.5 profiles produce `pixel_values_videos`,
`video_grid_thw`, and timestamp-expanded chat tokens. Images and videos can
appear together in any supported user/tool content list.

Install the v0.2 wheel with its optional file/image dependencies:

```shell
python3.11 -m pip install '/path/to/qwen_mm-0.2.0-cp311-abi3-<platform>.whl[video]'
```

The `video` extra installs PyAV and Pillow. NumPy clips need neither decoder
nor Torch. Torch output and Torch input require an independently installed
Torch. If a working TorchCodec installation exists, file decoding selects it
automatically; `video_backend="pyav"` or `"torchcodec"` selects explicitly.
TorchCodec installation must match the caller's Torch/FFmpeg environment.

```python
from PIL import Image
from qwen_mm import Processor

processor = Processor.from_pretrained("Qwen3.5", thread_budget=4)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": Image.open("overview.jpg")},
            {"type": "video", "video": "inspection.mp4", "fps": 2.0},
            {"type": "text", "text": "Describe changes relative to the overview."},
        ],
    }
]
prepared = processor.prepare(messages, add_generation_prompt=True, return_tensors="pt")
# model.generate(**prepared.to("cuda"), max_new_tokens=128)
```

For already extracted frames:

```python
frames = [Image.open(path) for path in frame_paths]
prepared = processor.prepare(
    [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames, "sample_fps": 2.0},
                {"type": "text", "text": "What changes during the clip?"},
            ],
        }
    ]
)
```

Plain frame lists follow the pinned VL Utils image-list path: each image is
converted to RGB, resized with the image-list factor, then the clip receives
the video resize. Rust performs both resizes, normalization, and patch packing.
Odd lists repeat their final frame and follow upstream's synthetic final frame
index. The existing still-image resize fidelity contract applies to the first
resize; pixel identity with Pillow is not promised.

| Source | Accepted representation | Timing and sampling |
| --- | --- | --- |
| MP4 and other FFmpeg-supported files | Path, `Path`, `file:`/HTTP(S) URL, or encoded bytes | Sample once using file FPS and rounded uniform indices |
| Frame list | Pillow objects, RGB arrays, image paths/URLs, or encoded images | Preserve frames; `sample_fps` defaults to 2; `raw_fps` defaults to `sample_fps` |
| Decoded clip | `uint8` NumPy THWC/TCHW or Torch TCHW/THWC | Preserve frames; odd clips repeat the final frame and its time |
| TorchCodec decoder | `VideoDecoder` or compatible object with metadata and `get_frames_at` | Retrieve selected indices in one batch |
| TorchCodec frame batch | `FrameBatch` with `.data` and `.pts_seconds` | Preserve actual presentation timestamps |
| VL Utils prepared video | `(integral_float_rgb_clip, video_metadata)` | Preserve prepared dimensions and metadata; no second resize |
| Explicit descriptor | `{"frames": clip, "timestamps": [...], "layout": "TCHW"}` | Explicit timestamps are seconds, ordered and nonnegative |

Metadata accepts `fps`, `frames_indices`, and `total_num_frames`, either in a
descriptor or the tuple's dictionary. `timestamps` takes precedence for prompt
time. A descriptor may set `preprocessed=True` for already sized frames; its
height and width must be multiples of 32. Ordinary decoded arrays receive the
video resize. Floating pixels must be integral, finite RGB values in `[0, 255]`;
rescaled/normalized floating tensors are rejected.

File/decoder items accept either `fps` (default 2) or `nframes`, plus
`min_frames`, `max_frames`, `video_start`, and `video_end` in seconds. These
follow the pinned VL Utils sampling rules. A one-frame file is an extension:
its frame is repeated for the temporal patch. Clip endpoints are inclusive in
frame-index space. Decoded clips use `sample_fps`, `raw_fps`, or explicit
metadata; file sampling options on a decoded clip raise an error.

Every video occurrence accepts `min_pixels`, `max_pixels`, `total_pixels`, and
paired `resized_height`/`resized_width`. Pixel budgets are per frame, with the
total budget used to derive the per-frame maximum. Defaults follow composed
VL Utils with patch size 16: minimum 131,072 pixels, maximum 786,432 pixels,
and a total budget of `128000 * 32**2 * .9`. Explicit dimensions are snapped
according to the upstream path. Supplying sizing controls to a prepared tuple
requests a new resize. The shared processor `min_pixels`/`max_pixels` keyword
defaults remain image controls; set video controls on the video item.

Separate `videos=[...]` inputs work with numeric `video`/`input_index` references,
including repeated occurrences. Every separate input must be referenced.
Heterogeneous batches can mix text-only, image-only, video-only, and mixed rows,
with either padding side. Arrays are contiguous float32 pixels and int64 grids
and tokens; Torch views share the NumPy storage. `prepared.metadata["videos"]`
and request/video layouts record source timing, geometry, and output ranges.

Resource ceilings apply before frame retrieval/conversion when dimensions are
available and before native copies/model-output allocation. Requests fail
atomically. File decode errors use `MediaDecodeError`; bad timing and geometry
use `MediaGeometryError`; malformed options use `InvalidRequestError`.
Remote sources share the existing encoded-byte budget and timeout behavior.
PyAV converts only selected frames to RGB but decodes preceding frames
sequentially. TorchCodec can retrieve sparse indices directly. Decoder threads
and Rust worker threads use the processor's bounded `thread_budget`.

Run `make video-test` to build a fresh production wheel and execute the
generated-fixture oracle, including lossless MP4s, both profiles, mixed media,
prepared tuples, timestamps, errors, output ownership, and concurrency. The
[video benchmark](../benchmarks/video-v1/README.md) separates decoded preprocessing from MP4 preparation and
records all timing samples; it does not measure model generation or GPU
training throughput. Historical still-image performance evidence remains
separate. The vLLM plugin retains its v0.1 still-image scope.
