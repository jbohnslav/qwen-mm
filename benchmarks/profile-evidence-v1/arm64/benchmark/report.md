# qwen-mm paired benchmark v2

- Created: `2026-08-03T14:29:07.237705+00:00`
- Mode: `smoke`
- Architecture family: `arm64`
- Process repetitions: `1`
- Threshold enforcement: none; D4 owns release performance gates
- Performance status: `DIAGNOSTIC ONLY`
- Releasable: `false`
- Phase C correctness prerequisite: `pass` (current passing correctness gate)

**DIAGNOSTIC ONLY:** these measurements are not a release performance claim. D4 owns performance certification.

Each measured workload passed paired pre/post output checks and measured-output stability. These checks do not replace the complete Phase C conformance gate.

| Profile | Case | Build | Threads | Reference p50 | Candidate p50 | Speedup | 95% CI |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| qwen3-vl-8b | image1 | profiled-release | 1 | 13.937 ms | 15.483 ms | 0.900x | 0.900–0.900x |
| qwen3-vl-8b | image1 | profiled-release | 4 | 12.311 ms | 15.448 ms | 0.797x | 0.797–0.797x |
| qwen3-vl-8b | image24 | profiled-release | 1 | 356.479 ms | 382.223 ms | 0.933x | 0.933–0.933x |
| qwen3-vl-8b | image24 | profiled-release | 4 | 277.129 ms | 385.118 ms | 0.720x | 0.720–0.720x |
| qwen3-vl-8b | ragged24 | profiled-release | 1 | 123.021 ms | 128.407 ms | 0.958x | 0.958–0.958x |
| qwen3-vl-8b | ragged24 | profiled-release | 4 | 111.487 ms | 128.805 ms | 0.866x | 0.866–0.866x |
| qwen3-vl-8b | rgb24 | profiled-release | 1 | 130.259 ms | 196.641 ms | 0.662x | 0.662–0.662x |
| qwen3-vl-8b | rgb24 | profiled-release | 4 | 113.525 ms | 192.150 ms | 0.591x | 0.591–0.591x |
| qwen3-vl-8b | text_long | profiled-release | 1 | 10.709 ms | 2.525 ms | 4.241x | 4.241–4.241x |
| qwen3-vl-8b | text_long | profiled-release | 4 | 10.584 ms | 2.468 ms | 4.289x | 4.289–4.289x |
| qwen3-vl-8b | text_short | profiled-release | 1 | 0.917 ms | 0.246 ms | 3.729x | 3.729–3.729x |
| qwen3-vl-8b | text_short | profiled-release | 4 | 0.911 ms | 0.253 ms | 3.603x | 3.603–3.603x |
| qwen3.5-9b | image1 | profiled-release | 1 | 13.930 ms | 15.239 ms | 0.914x | 0.914–0.914x |
| qwen3.5-9b | image1 | profiled-release | 4 | 12.615 ms | 16.200 ms | 0.779x | 0.779–0.779x |
| qwen3.5-9b | image24 | profiled-release | 1 | 328.719 ms | 385.721 ms | 0.852x | 0.852–0.852x |
| qwen3.5-9b | image24 | profiled-release | 4 | 280.971 ms | 382.236 ms | 0.735x | 0.735–0.735x |
| qwen3.5-9b | ragged24 | profiled-release | 1 | 124.728 ms | 131.288 ms | 0.950x | 0.950–0.950x |
| qwen3.5-9b | ragged24 | profiled-release | 4 | 112.854 ms | 129.981 ms | 0.868x | 0.868–0.868x |
| qwen3.5-9b | rgb24 | profiled-release | 1 | 131.443 ms | 197.680 ms | 0.665x | 0.665–0.665x |
| qwen3.5-9b | rgb24 | profiled-release | 4 | 113.923 ms | 195.558 ms | 0.583x | 0.583–0.583x |
| qwen3.5-9b | text_long | profiled-release | 1 | 10.924 ms | 2.552 ms | 4.280x | 4.280–4.280x |
| qwen3.5-9b | text_long | profiled-release | 4 | 10.812 ms | 2.487 ms | 4.348x | 4.348–4.348x |
| qwen3.5-9b | text_short | profiled-release | 1 | 1.105 ms | 0.251 ms | 4.403x | 4.403–4.403x |
| qwen3.5-9b | text_short | profiled-release | 4 | 1.112 ms | 0.243 ms | 4.573x | 4.573–4.573x |

p99 is reported only when an implementation has at least 100 raw samples. ARM and x86 measurements are intentionally stored in separate result files.
