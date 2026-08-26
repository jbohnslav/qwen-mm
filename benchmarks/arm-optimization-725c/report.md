# qwen-mm paired benchmark v2

- Created: `2026-08-26T15:44:06.627120+00:00`
- Mode: `dedicated`
- Architecture family: `arm64`
- Process repetitions: `3`
- Threshold enforcement: none; D4 owns release performance gates
- Performance status: `DIAGNOSTIC ONLY`
- Releasable: `false`
- Phase C correctness prerequisite: `pass` (current passing correctness gate)

**DIAGNOSTIC ONLY:** these measurements are not a release performance claim. D4 owns performance certification.

Each measured workload passed paired pre/post output checks and measured-output stability. These checks do not replace the complete Phase C conformance gate.

| Profile | Case | Build | Threads | Reference p50 | Candidate p50 | Speedup | 95% CI |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| qwen3-vl-8b | rgb24 | shipping | 1 | 90.916 ms | 50.441 ms | 1.803x | 1.801–1.804x |
| qwen3.5-9b | rgb24 | shipping | 1 | 91.576 ms | 50.693 ms | 1.808x | 1.802–1.810x |

p99 is reported only when an implementation has at least 100 raw samples. ARM and x86 measurements are intentionally stored in separate result files.
