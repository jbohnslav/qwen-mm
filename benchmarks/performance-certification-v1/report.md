# qwen-mm image performance certification v1

- Status: `MISS`
- Releasable: `false`
- Matrix: compact shipping-only matrix; the deferred native/exhaustive lanes are not claimed.
- Architectures: macOS ARM and Linux x86 are evaluated separately and never aggregated.
- Scope: CPU preprocessing only; this report makes no video or vLLM production claim.
- Cache workloads are outside the selected compact matrix.

## Gate conclusion

Certification missed 15 unchanged gate(s):

- `noise/x86_64/shipping/qwen3.5-9b/rgb24/t1/reference`: all 3 process medians retained; CV=0.053491; required <=0.050000
- `speed/arm64/qwen3-vl-8b/rgb24/t1`: shipping paired-bootstrap lower=0.819x; required >1.000x
- `efficiency/arm64/qwen3-vl-8b/image24/t8`: shipping E_8=0.5105; required >=0.6000
- `speed/arm64/qwen3.5-9b/rgb24/t1`: shipping paired-bootstrap lower=0.814x; required >1.000x
- `efficiency/arm64/qwen3.5-9b/image24/t8`: shipping E_8=0.5188; required >=0.6000
- `efficiency/x86_64/qwen3-vl-8b/image24/t8`: shipping E_8=0.2573; required >=0.6000
- `efficiency/x86_64/qwen3.5-9b/image24/t8`: shipping E_8=0.2570; required >=0.6000
- `memory/arm64/qwen3-vl-8b/images_16/t8`: shipping per-process candidate max(external RSS, exact native peak)/official external RSS ratio=max-paired=61.7033; every official must be >0 and every paired ratio <=0.5000
- `memory/arm64/qwen3-vl-8b/ragged24/t8`: shipping per-process candidate max(external RSS, exact native peak)/official external RSS ratio=max-paired=0.8617; every official must be >0 and every paired ratio <=0.5000
- `memory/arm64/qwen3-vl-8b/rgb24/t1`: shipping per-process candidate max(external RSS, exact native peak)/official external RSS ratio=max-paired=1.4184; every official must be >0 and every paired ratio <=0.5000
- `memory/arm64/qwen3.5-9b/image1/t1`: shipping per-process candidate max(external RSS, exact native peak)/official external RSS ratio=max-paired=164.8248; every official must be >0 and every paired ratio <=0.5000
- `memory/arm64/qwen3.5-9b/images_16/t8`: shipping per-process candidate max(external RSS, exact native peak)/official external RSS ratio=max-paired=1.1669; every official must be >0 and every paired ratio <=0.5000
- `memory/arm64/qwen3.5-9b/rgb24/t1`: shipping per-process candidate max(external RSS, exact native peak)/official external RSS ratio=max-paired=0.9403; every official must be >0 and every paired ratio <=0.5000
- `memory/x86_64/qwen3-vl-8b/image1/t1`: shipping per-process candidate max(external RSS, exact native peak)/official external RSS ratio=max-paired=0.5301; every official must be >0 and every paired ratio <=0.5000
- `memory/x86_64/qwen3.5-9b/image1/t1`: shipping per-process candidate max(external RSS, exact native peak)/official external RSS ratio=unmeasurable; every official must be >0 and every paired ratio <=0.5000

## Headline shipping results

| Architecture | Profile | Case | Candidate p50 | Speedup | 95% CI |
| --- | --- | --- | ---: | ---: | ---: |
| arm64 | qwen3-vl-8b | image24 | 37.792 ms | 6.426x | 6.416–6.486x |
| arm64 | qwen3-vl-8b | ragged24 | 17.681 ms | 4.750x | 4.723–4.754x |
| arm64 | qwen3.5-9b | image24 | 37.896 ms | 6.380x | 6.370–6.389x |
| arm64 | qwen3.5-9b | ragged24 | 17.925 ms | 4.700x | 4.691–4.708x |
| x86_64 | qwen3-vl-8b | image24 | 166.959 ms | 3.476x | 3.392–3.477x |
| x86_64 | qwen3-vl-8b | ragged24 | 80.936 ms | 2.565x | 2.526–2.647x |
| x86_64 | qwen3.5-9b | image24 | 166.725 ms | 3.626x | 3.563–3.640x |
| x86_64 | qwen3.5-9b | ragged24 | 80.444 ms | 2.572x | 2.466–2.604x |

Raw timing samples, separate scoped-memory observations, allocation census data, pre/post conformance witnesses, build artifacts, host placement, and reproduction commands are retained in the versioned JSON artifact.

