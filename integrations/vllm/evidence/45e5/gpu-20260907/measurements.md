# Paired serving results

GPU: NVIDIA L40S, 46068 MiB, 580.95.05
Model: `Qwen/Qwen3.5-9B` at `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.

ABBA process order on one GPU; three repeats per process. Latencies include
local HTTP and client construction, but exclude fixture generation and JSON
serialization before the HTTP timer. This is a bounded diagnostic, not D4/F3 certification.

| Request | Cache | n / mode | Stock TTFT ms | Native TTFT ms | Change | Stock total ms | Native total ms |
|---|---|---:|---:|---:|---:|---:|---:|
| text | cold | 6 | 68.30 | 66.78 | -2.2% | 112.63 | 111.29 |
| text | warm | 6 | 70.04 | 70.21 | +0.2% | 115.63 | 114.96 |
| single | cold | 6 | 84.43 | 83.60 | -1.0% | 437.59 | 436.43 |
| single | warm | 6 | 71.42 | 69.61 | -2.5% | 424.34 | 421.27 |
| multi | cold | 6 | 97.93 | 97.44 | -0.5% | 450.80 | 449.68 |
| multi | warm | 6 | 76.79 | 74.42 | -3.1% | 428.92 | 427.81 |
| repeat | cold | 6 | 86.46 | 84.21 | -2.6% | 441.01 | 437.25 |
| repeat | warm | 6 | 70.50 | 68.56 | -2.7% | 423.24 | 421.87 |
| heavy | cold | 6 | 291.99 | 283.33 | -3.0% | 660.50 | 650.57 |
| heavy | warm | 6 | 189.11 | 187.86 | -0.7% | 557.39 | 556.12 |
| concurrent | cold | 24 | 816.19 | 760.09 | -6.9% | 1286.34 | 1272.73 |

Negative change means lower median TTFT. Samples are small and requests
within a process share caches and hardware; these are descriptive medians.

| Mode | Image processing median / p95 ms | Health median / p95 ms | Peak summed process RSS GiB | Concurrent requests/s |
|---|---:|---:|---:|---:|
| stock | 46.06 / 59.35 | 2.12 / 2.84 | 12.03 | 2.45 |
| native | 28.43 / 32.86 | 2.12 / 2.88 | 11.62 | 2.84 |

Processor timing includes one processing call, which may contain multiple images.
Health-response latency is a responsiveness proxy, not direct event-loop lag.
RSS sums processes and may double-count shared pages. Concurrent throughput
includes the fixed 16-token generation budget, HTTP and queueing.

## Correctness witnesses

- Run 0 stock: 42/42 equal usage records, 42/42 equal decoded outputs versus run 0; audit {'image_processor_calls': 24, 'vision_batches': 24, 'vision_tower_calls': 24}.
- Run 1 native: 42/42 equal usage records, 28/42 equal decoded outputs versus run 0; audit {'image_processor_calls': 24, 'vision_batches': 24, 'vision_tower_calls': 24}.
- Run 2 native: 42/42 equal usage records, 26/42 equal decoded outputs versus run 0; audit {'image_processor_calls': 24, 'vision_batches': 24, 'vision_tower_calls': 24}.
- Run 3 stock: 42/42 equal usage records, 42/42 equal decoded outputs versus run 0; audit {'image_processor_calls': 24, 'vision_batches': 24, 'vision_tower_calls': 24}.

Equal decoded outputs and usage are bounded behavioral witnesses, not a logits
equivalence proof. Inspect differences, raw server logs and audit records before
making an adoption recommendation.
