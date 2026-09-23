# qwen-mm paired benchmark v2

- Created: `2026-09-23T14:44:18.373989+00:00`
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
| qwen3-vl-8b | image1 | shipping | 1 | 10.287 ms | 5.343 ms | 1.926x | 1.894–1.958x |
| qwen3-vl-8b | jpeg24_requests | shipping | 1 | 269.680 ms | 125.774 ms | 2.150x | 2.138–2.163x |
| qwen3-vl-8b | text_long | shipping | 1 | 3.814 ms | 0.270 ms | 14.007x | 13.753–14.262x |
| qwen3-vl-8b | text_short | shipping | 1 | 0.340 ms | 0.194 ms | 1.759x | 1.754–1.763x |
| qwen3.5-9b | image1 | shipping | 1 | 10.532 ms | 5.325 ms | 1.976x | 1.974–1.978x |
| qwen3.5-9b | jpeg24_requests | shipping | 1 | 273.367 ms | 125.404 ms | 2.182x | 2.170–2.194x |
| qwen3.5-9b | text_long | shipping | 1 | 3.897 ms | 0.471 ms | 8.169x | 8.150–8.188x |
| qwen3.5-9b | text_short | shipping | 1 | 0.379 ms | 0.208 ms | 1.806x | 1.804–1.809x |

p99 is reported only when an implementation has at least 100 raw samples. ARM and x86 measurements are intentionally stored in separate result files.
