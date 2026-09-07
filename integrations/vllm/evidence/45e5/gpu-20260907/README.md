# Real vLLM serving witness — 7 September 2026

**Result: the external native-image plugin works with a published vLLM wheel
and a real Qwen3.5-9B server.** All 168 measured HTTP requests succeeded.
A custom CUDA development image supports FlashInfer without building or
patching vLLM. Native image processing took less CPU time, while the measured
end-to-end gains were modest. Keep this an opt-in integration.

## Reproduction and provenance

- [Successful Modal run](https://modal.com/apps/jbohnslav/main/ap-893XraaKaN0grT125ahrQ9),
  one NVIDIA L40S, driver 580.95.05, 8 CPU, 32 GiB RAM, 2400-second function
  timeout. The run completed and a post-run container listing was empty.
- NVIDIA `cuda:13.0.2-devel-ubuntu24.04`, linux/amd64 digest
  `sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c`,
  with Modal-added Python 3.11 and `CUDA_HOME=/usr/local/cuda`.
- Published vLLM 0.23.0, Torch 2.11.0 (CUDA 13.0), Transformers 5.14.1,
  NumPy 2.3.5. No vLLM source build or source patch. qwen-mm's own Linux
  extension wheel and the external Python plugin were installed normally.
- Qwen/Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`;
  bf16, context 4096, four concurrent sequences, eager execution,
  prefix caching disabled, image processor cache 1 GiB, image pixel budget
  65,536–262,144, one native/Torch image-processing thread.
- FlashInfer sampler enabled on both paths. `flashinfer.json` records the
  successful GPU preflight and nvcc 13.0.88. First-use runtime kernel
  compilation is permitted; it is separate from building vLLM.
- `serving.tar.gz` retains all four server logs, processor/vision JSONL audits,
  per-request timings and outputs, health probes, memory observations, exact
  launch commands, installed-package versions and integration source hashes.
  Extract it, then run `scripts/summarize_serving.py RESULTS --output REPORT`.

The failed sampler startup, image-budget regression and mixed pip CUDA-toolkit
attempt are retained separately. The final runner exposes the matching toolkit
from the NVIDIA image. The native processor now merges server defaults and
request options through vLLM's context; both-profile regressions failed before
that fix and pass afterward.

## Serving behavior and correctness

Each of the four stock/native/native/stock processes handled 42 measured
requests plus two warmups. The corpus covers text, one image, unequal images,
repeated images, four large images and concurrency four. Each image case has
an immediate warm repeat. Each process recorded exactly 24 uncached image
processing calls, 24 pixel-input vision batches and 24 actual vision-tower
calls. All observed image processing ran off the event loop. Warm reuse caused
no extra image preprocessing or vision calls.

All 42 usage records, including prompt and completion token counts, matched
stock in each run. All 24 text/aligned-image decoded outputs matched exactly.
Resized random-noise captions were not text-identical: the two native runs
matched 28/42 and 26/42 total stock outputs. The native resized-pixel contract
is approximate, so this is **not a bit-exact replacement for stock resized
images**, and pixel-level closeness does not guarantee identical generation.

`resize-witness.json` checks every heavy/concurrent source image used in the
benchmark against pinned Transformers at the same output size: 60 images for
each profile, 120 image/profile cases and 360 channels. All channels pass the
existing, unchanged resize-v2 gates. Worst RGB8 absolute error is 8/255, worst
RMSE is 0.466/255, worst p99 absolute error is 1/255, and minimum canonical SSIM
is 0.999873. This supplements the existing frozen-corpus evidence; it is not a
new certification. Both profiles have preprocessing coverage; real generation
in this run is scoped to Qwen3.5-9B.

The final CPU integration suite passes 40 tests. Full repository checks pass,
including 133 reference tests, Rust checks/tests and wheel smoke; all 105
script tests pass. Logs are retained beside this report.

## Performance and adoption

See `measurements.md` for the unfiltered raw-run aggregates. Median image
processing latency fell from 46.06 to 28.43 ms, about **38% lower**. This mixes
one-, two- and four-image processing calls; most calls contain four images.
Median cold four-image TTFT fell from 291.99 to 283.33 ms (3.0%), and concurrent
TTFT from 816.19 to 760.09 ms (6.9%). The four-image gain is comparable to the
variation between the two stock runs, so it is not strong evidence by itself.

The raw aggregate concurrent throughput, 2.45 versus 2.84 requests/s, includes
a slow 2.74-second first stock batch. Do not present this as a sustained 16%
throughput gain. A transparent sensitivity calculation dropping repetition zero
from **every** process gives 2.883 versus 2.902 requests/s, only 0.65% apart;
concurrent TTFT in that subset is 779.75 versus 743.06 ms (4.7% lower). All
samples remain in the raw artifact and primary table. This post-hoc sensitivity
is descriptive, not a replacement benchmark or a new passing threshold.

Health-response p95 was effectively unchanged (2.84 versus 2.88 ms). Peak summed
process RSS was 12.03 versus 11.62 GiB; shared pages may be counted more than
once. Health response is a responsiveness proxy, not direct scheduling-lag
measurement. Health probe failures were not separately counted.

The maximum HTTP body was 16,807,280 bytes on both paths. Ordinary images stay
compressed over HTTP. Timers include local HTTP and client construction but
exclude fixture generation and JSON serialization before submission; WAN
latency was not measured. Copies remain in Pillow-to-NumPy materialization,
native ownership snapshots, downstream tensor assembly, IPC and GPU/dtype
transfer. `torch.from_numpy` shares native output storage, but the route is not
end-to-end zero-copy. The native public API's synthetic prompt work is included
in processor timings.

**Adoption decision:** expose the plugin for users whose CPU image processing
is a measured bottleneck and who accept resize-v2 semantics. Do not switch all
serving deployments on the strength of this small synthetic benchmark, promise
identical resized-image captions, or advertise a broad throughput improvement.
A larger representative-image workload and attribution of decode, hashing,
transport and GPU time would be the next useful performance study. Existing
D4 MISS and legacy Phase C evidence remain unchanged. Publication is outside
this ticket, and the earlier release candidate's artifacts remain tied to their
original source commit.

To reproduce the supplemental resize witness in the installed integration
environment, from the repository root:

```bash
uv pip install --python .venv-vllm/bin/python qwen-vl-utils==0.0.14
PYTHONPATH=reference/src .venv-vllm/bin/python \
  integrations/vllm/scripts/resize_witness.py /tmp/resize-witness.json
```
