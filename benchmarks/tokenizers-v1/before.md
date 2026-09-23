# qwen-mm paired benchmark v2

- Created: `2026-09-23T14:41:25.158767+00:00`
- Mode: `smoke`
- Architecture family: `arm64`
- Process repetitions: `2`
- Threshold enforcement: none; D4 owns release performance gates
- Performance status: `DIAGNOSTIC ONLY`
- Releasable: `false`
- Phase C correctness prerequisite: `invalid` (report_validation)

**DIAGNOSTIC ONLY:** these measurements are not a release performance claim. D4 owns performance certification.

Each measured workload passed paired pre/post output checks and measured-output stability. These checks do not replace the complete Phase C conformance gate.

| Profile | Case | Build | Threads | Reference p50 | Candidate p50 | Speedup | 95% CI |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| qwen3-vl-8b | image1 | shipping | 1 | 9.852 ms | 5.283 ms | 1.867x | 1.867–1.867x |
| qwen3-vl-8b | jpeg24_requests | shipping | 1 | 271.442 ms | 124.174 ms | 2.186x | 2.185–2.187x |
| qwen3-vl-8b | text_long | shipping | 1 | 3.697 ms | 2.298 ms | 1.607x | 1.599–1.616x |
| qwen3-vl-8b | text_short | shipping | 1 | 0.319 ms | 0.216 ms | 1.500x | 1.461–1.539x |
| qwen3.5-9b | image1 | shipping | 1 | 10.265 ms | 5.342 ms | 1.899x | 1.881–1.917x |
| qwen3.5-9b | jpeg24_requests | shipping | 1 | 273.045 ms | 125.292 ms | 2.178x | 2.175–2.181x |
| qwen3.5-9b | text_long | shipping | 1 | 3.857 ms | 2.334 ms | 1.649x | 1.649–1.650x |
| qwen3.5-9b | text_short | shipping | 1 | 0.359 ms | 0.221 ms | 1.625x | 1.582–1.667x |

p99 is reported only when an implementation has at least 100 raw samples. ARM and x86 measurements are intentionally stored in separate result files.
