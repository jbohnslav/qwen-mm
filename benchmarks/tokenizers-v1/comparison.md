# Before/after results

Local M4, release builds, one thread. Latencies are end-to-end preprocessing p50s, not pure tokenizer timings. Speedup is old qwen-mm latency divided by upgraded qwen-mm latency. See README.md for method and limits.

| Profile | Case | Before ms | After ms | Speedup | Before units/s | After units/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| qwen3-vl-8b | image1 | 5.283 | 5.343 | 0.99x | 189.3 | 187.1 |
| qwen3-vl-8b | jpeg24_requests | 124.174 | 125.774 | 0.99x | 193.3 | 190.8 |
| qwen3-vl-8b | text_long | 2.298 | 0.270 | 8.50x | 435.1 | 3699.1 |
| qwen3-vl-8b | text_short | 0.216 | 0.194 | 1.12x | 4632.3 | 5167.4 |
| qwen3.5-9b | image1 | 5.342 | 5.325 | 1.00x | 187.2 | 187.8 |
| qwen3.5-9b | jpeg24_requests | 125.292 | 125.404 | 1.00x | 191.6 | 191.4 |
| qwen3.5-9b | text_long | 2.334 | 0.471 | 4.96x | 428.4 | 2124.0 |
| qwen3.5-9b | text_short | 0.221 | 0.208 | 1.06x | 4520.2 | 4801.0 |

Units are prompts for text cases and images for image cases (24 images in 24 requests for jpeg24_requests). All 16 matched process/case pairs have byte-identical before/after output signatures, including token IDs, masks, and pixels. Both builds also passed paired oracle checks.

## Memory

Median process RSS census across two repetitions, in MiB. Includes reference/oracle allocations; this is not tokenizer-only memory or a hard bound.

| Profile | Case | Peak before | Peak after | Transient before | Transient after |
| --- | --- | ---: | ---: | ---: | ---: |
| qwen3-vl-8b | image1 | 771.8 | 757.5 | 0.0 | 0.0 |
| qwen3-vl-8b | jpeg24_requests | 4341.7 | 4124.2 | 1.6 | 56.0 |
| qwen3-vl-8b | text_long | 532.6 | 564.5 | 0.0 | 0.0 |
| qwen3-vl-8b | text_short | 532.8 | 553.6 | 0.0 | 0.1 |
| qwen3.5-9b | image1 | 838.9 | 996.2 | 0.1 | 0.0 |
| qwen3.5-9b | jpeg24_requests | 4226.8 | 4549.6 | 54.5 | 0.7 |
| qwen3.5-9b | text_long | 670.3 | 742.5 | 0.1 | 0.1 |
| qwen3.5-9b | text_short | 667.3 | 750.5 | 0.0 | 0.1 |
