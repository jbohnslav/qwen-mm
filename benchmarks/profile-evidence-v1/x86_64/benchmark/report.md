# qwen-mm paired benchmark v2

- Created: `2026-08-03T14:41:16.396194+00:00`
- Mode: `smoke`
- Architecture family: `x86_64`
- Process repetitions: `1`
- Threshold enforcement: none; D4 owns release performance gates
- Performance status: `DIAGNOSTIC ONLY`
- Releasable: `false`
- Phase C correctness prerequisite: `pass` (current passing correctness gate)

**DIAGNOSTIC ONLY:** these measurements are not a release performance claim. D4 owns performance certification.

Each measured workload passed paired pre/post output checks and measured-output stability. These checks do not replace the complete Phase C conformance gate.

| Profile | Case | Build | Threads | Reference p50 | Candidate p50 | Speedup | 95% CI |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| qwen3-vl-8b | image1 | profiled-release | 1 | 36.681 ms | 31.049 ms | 1.181x | 1.181–1.181x |
| qwen3-vl-8b | image1 | profiled-release | 4 | 34.013 ms | 30.098 ms | 1.130x | 1.130–1.130x |
| qwen3-vl-8b | image24 | profiled-release | 1 | 917.351 ms | 640.396 ms | 1.432x | 1.432–1.432x |
| qwen3-vl-8b | image24 | profiled-release | 4 | 770.816 ms | 634.757 ms | 1.214x | 1.214–1.214x |
| qwen3-vl-8b | ragged24 | profiled-release | 1 | 330.898 ms | 235.460 ms | 1.405x | 1.405–1.405x |
| qwen3-vl-8b | ragged24 | profiled-release | 4 | 266.875 ms | 241.697 ms | 1.104x | 1.104–1.104x |
| qwen3-vl-8b | rgb24 | profiled-release | 1 | 408.434 ms | 267.714 ms | 1.526x | 1.526–1.526x |
| qwen3-vl-8b | rgb24 | profiled-release | 4 | 282.872 ms | 272.038 ms | 1.040x | 1.040–1.040x |
| qwen3-vl-8b | text_long | profiled-release | 1 | 22.619 ms | 4.190 ms | 5.398x | 5.398–5.398x |
| qwen3-vl-8b | text_long | profiled-release | 4 | 22.324 ms | 3.897 ms | 5.729x | 5.729–5.729x |
| qwen3-vl-8b | text_short | profiled-release | 1 | 2.475 ms | 0.636 ms | 3.890x | 3.890–3.890x |
| qwen3-vl-8b | text_short | profiled-release | 4 | 2.474 ms | 0.568 ms | 4.356x | 4.356–4.356x |
| qwen3.5-9b | image1 | profiled-release | 1 | 34.070 ms | 31.359 ms | 1.086x | 1.086–1.086x |
| qwen3.5-9b | image1 | profiled-release | 4 | 32.315 ms | 32.637 ms | 0.990x | 0.990–0.990x |
| qwen3.5-9b | image24 | profiled-release | 1 | 824.719 ms | 622.138 ms | 1.326x | 1.326–1.326x |
| qwen3.5-9b | image24 | profiled-release | 4 | 790.508 ms | 634.333 ms | 1.246x | 1.246–1.246x |
| qwen3.5-9b | ragged24 | profiled-release | 1 | 334.699 ms | 234.196 ms | 1.429x | 1.429–1.429x |
| qwen3.5-9b | ragged24 | profiled-release | 4 | 261.815 ms | 220.160 ms | 1.189x | 1.189–1.189x |
| qwen3.5-9b | rgb24 | profiled-release | 1 | 363.759 ms | 271.459 ms | 1.340x | 1.340–1.340x |
| qwen3.5-9b | rgb24 | profiled-release | 4 | 275.349 ms | 278.033 ms | 0.990x | 0.990–0.990x |
| qwen3.5-9b | text_long | profiled-release | 1 | 22.621 ms | 4.566 ms | 4.954x | 4.954–4.954x |
| qwen3.5-9b | text_long | profiled-release | 4 | 23.191 ms | 4.481 ms | 5.176x | 5.176–5.176x |
| qwen3.5-9b | text_short | profiled-release | 1 | 2.600 ms | 0.638 ms | 4.076x | 4.076–4.076x |
| qwen3.5-9b | text_short | profiled-release | 4 | 2.649 ms | 0.614 ms | 4.317x | 4.317–4.317x |

p99 is reported only when an implementation has at least 100 raw samples. ARM and x86 measurements are intentionally stored in separate result files.
