# Verify the processor with a local Qwen model

This opt-in smoke test feeds qwen-mm's actual output arrays into
`Qwen3_5ForConditionalGeneration.generate`, using the pinned
[Qwen3.5-0.8B weights](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17).
It runs on CPU in float32. Model inference speed is not an acceptance criterion.

From the repository root, run:

```shell
sh scripts/test-transformers-consumer.sh --download
```

The first run downloads approximately 1.75 GB of model weights plus small
processor assets into `reference/.cache/huggingface`. Later runs can omit
`--download` to require the cached snapshots. `--cache-dir` selects another
cache. The runner builds a release wheel without test hooks, installs it and
the locked reference dependencies into a temporary environment, and removes
that environment after the test. The model cache remains reusable.

The default JSON report is `/tmp/qwen-mm-transformers-consumer.json`; pass
`--output /absolute/path/report.json` to retain it elsewhere. It records model
and processor revisions, asset and weight hashes, package versions, installed
wheel identity, input shapes/differences, logits differences, generated token
IDs, decoded text, and diagnostic inference times.

## Why the small model is a valid consumer

The public `Qwen3.5` processor still selects the existing pinned 9B profile.
The test does not add a public 0.8B profile or substitute unvalidated processor
assets. It first requires the 0.8B tokenizer vocabulary, merges, tokenizer
model, and image preprocessor configuration to match the 9B snapshot exactly.
Special image token IDs and vision patch geometry must also match.

The chat templates have different defaults for thinking mode. The test sets
`enable_thinking=False` explicitly and checks that both official templates
render each test conversation identically with either explicit thinking value.
Other tokenizer settings must be identical.

## What passes mean

The six cases cover text, a solid image, a patterned image without resize,
multiple images with different shapes, a left-padded mixed text/image batch,
and an image requiring resize. The official
side uses the repository's composed `qwen-vl-utils` and Transformers oracle;
the native side uses public `prepare` and `prepare_batch` calls.

Both sides must return identical keys, dtypes, shapes, and integer arrays.
For cases without resize, pixel arrays, first-token logits, and greedy generated
token IDs must also match exactly. Every generation must produce tokens and
finite logits. Resized pixels follow the existing
[resize-v2 fidelity contract](image-resize-contract-v2.md); their differences
and downstream logits are reported, and both paths must execute successfully,
but this smoke does not impose token equality or replace the fidelity suite.

The small model's answer quality is not a gate. This verifies real consumption
of the Qwen3.5 tensor contract, not 9B model quality, Qwen3-VL model execution,
video support, or preprocessing speed.

The [2026-09-05 macOS ARM report](../reference/consumer/v1/macos-arm64.json)
records all six cases passing against an installed release wheel. The five
cases without resize had exact inputs, first-token logits, and generated token
IDs. The resized case also generated the same tokens in this run, although its
pixels and logits differed; that token agreement is an observation, not a gate.

## The consumer handoff

The actual handoff is just:

```python
import torch

inputs = processor.prepare(
    messages, add_generation_prompt=True, enable_thinking=False, return_tensors="pt"
).to(model.device)
with torch.inference_mode():
    generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
new_tokens = generated[:, inputs["input_ids"].shape[1] :]
text = processor.batch_decode(new_tokens, skip_special_tokens=True)
```

Here `processor` is qwen-mm and `model` is the loaded model. Optional Torch output
shares CPU storage; the ergonomics test checks each array's pointer. No model-input keys are
filtered, no pixels are repacked, and no second image processor runs on the
qwen-mm side. The original six-case report above predates these convenience
methods and verifies the explicit NumPy bridge. The
[executable Rosetta suite](rosetta-verification.md) verifies this direct handoff,
including native decoding, using the actual published examples.
