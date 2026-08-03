# Deterministic bounded parallelism

Phase D2 gives each `Processor` its own finite Rayon pool. The compatibility
default remains one worker; production callers opt into a larger *total
qwen-mm budget* at construction:

```python
from qwen_mm import Processor

processor = Processor(
    "qwen3-vl-8b",
    "/path/to/pinned/snapshot",
    thread_budget=4,
)
assert processor.thread_budget == 4
```

The accepted range is 1 through 256. Zero and over-limit values are rejected,
not clamped. The resolved budget is immutable. Rust callers use
`QwenImageProcessor::from_local_assets_with_config` and
`ProcessorConfig::with_thread_budget`; the older constructor is the exact
one-thread compatibility path.

The pool is built with `rayon::ThreadPoolBuilder` and all parallel iterators and
joins run under that pool's `install` boundary. `RAYON_NUM_THREADS` therefore
cannot resize it, and qwen-mm never initializes or dispatches work to Rayon's
process-global pool. Dropping the processor drops its pool and worker handles.
Sharing one processor reuses the same pool across calls. Every public planning
and execution operation enters that pool, so concurrent calls on one shared
processor—including observed calls and ordered validation, preflight, text,
and metadata stages—are collectively capped by its `thread_budget` workers.

## Parallel boundary and determinism

The D1 profiles put 83.7% of ARM64 exclusive time and 86.1% of x86-64
exclusive time in visual stages. Resize alone represented 53.3%/55.2%, and
normalize/patchify/layout represented 19.0%/21.6%. D2 therefore parallelizes
independent visual occurrences at those two boundaries. Request validation,
global resource preflight, output layout, text expansion/tokenization, and all
metadata assembly remain in deterministic traversal order.

Decode/resize tasks return indexed private results. The caller waits for every
task and then applies the frozen category/occurrence precedence in serial
order. No official destination exists at that point. Normalize/patchify is
infallible after planning and writes directly into recursively split,
non-overlapping pixel and grid slices. This preserves request, occurrence,
token, patch, grid, sidecar, and destination order without a second patch
matrix or a completion-order merge.

`thread_budget=1` retains the original serial stage loops, but the complete
operation still runs or queues on the processor's sole owned worker. Supported
budgets 1 through N are required to produce byte-identical arrays and identical
metadata and errors.

## Observed calls

`prepare_batch_observed` intentionally retains a serial visual lane for Phase
D1 observation schema v1. `ObservationRecorder` is mutable, order-sensitive,
and bounded; it is never shared between workers. The observed lane therefore
keeps stable span sequence, parent links, event capacity, and allocation
lifetime accounting while returning exactly the same arrays as the parallel
production lane. Throughput measurements must use `prepare_batch`; observed
calls are diagnostic stage-attribution runs, not parallel scaling timings.

## One total host budget

`thread_budget` bounds qwen-mm native workers, not every thread in the host
process. qwen-mm invokes the tokenizers crate one prompt at a time and never
uses its batch-parallel API. It also does not mutate process-global environment
or PyTorch settings. Hosts must coordinate nested runtimes explicitly:

- Set `TOKENIZERS_PARALLELISM=false` when other code in the process may invoke
  Hugging Face batch tokenization.
- Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and Torch
  intra-op/inter-op thread counts to 1 while qwen-mm owns the parallel visual
  budget, unless a measured host-wide allocation divides the same finite CPU
  budget between them.
- Share one `Processor` per profile where practical. If an outer renderer or
  vLLM executor runs `K` processors concurrently, choose budgets such that
  `K * thread_budget` plus explicitly reserved outer-runtime workers does not
  exceed the host's intended total CPU budget.
- Do not set a vLLM renderer pool size and a full per-processor budget to the
  same host core count. One of those layers must be serial or the budget must
  be divided between them.

The paired benchmark adapter passes its protocol `thread_budget` directly to
both observed and unobserved processor factories. Environment variables remain
provenance; they are not a substitute for the processor-owned setting.

## Diagnostic status

D2 scaling runs are diagnostic rather than release certification. They use the
unobserved path, report throughput, process CPU utilization, and peak resident
memory for a representative many-image batch, and compare budgets 1 through 4.
The controlled-host pass/fail performance gates remain owned by D4.
