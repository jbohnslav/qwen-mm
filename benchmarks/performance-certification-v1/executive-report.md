# D4 CPU preprocessing status: ARM and x86

- Date: 2026-08-24
- Captured source: `5a77e78c85e0ef11da154d573fb405abb9c9780b`
- Certification result: **MISS — 107 of 122 gates passed; not yet releasable**

## Executive summary

The implementation is correct, reproducible, and substantially faster than the
official processor on most of the selected image workloads. The compact release
gate is still open because the contract requires every selected gate to pass,
and 15 do not.

The result is much more encouraging than a bare `MISS` suggests:

- **x86 speed is in very good shape.** Every selected x86 speed and text
  regression gate passed. Depending on the workload, qwen-mm is about
  `2.36x–3.70x` faster than the official path.
- **ARM encoded-image and batch speed is also strong.** Single-image and
  single-thread image batches are about `1.5x–1.8x` faster, while the selected
  eight-thread batches are about `4.7x–6.4x` faster than official.
- **The main ARM performance hole is caller-owned raw RGB.** That path has
  roughly `22%–23%` higher latency than official for both profiles and also
  misses its memory gate.
- **Eight-thread scaling needs work on both architectures.** Absolute t8
  performance is fast, but qwen-mm gets only about `51%` parallel efficiency on
  ARM and `26%` on x86; the gate is `60%`.
- **x86 batch memory is excellent.** Most selected x86 batch cases use only
  about `7%–25%` of the official path's transient-memory delta.
- **ARM memory is mixed.** The common image24 cases are excellent, but raw RGB
  and several ragged/multi-image cases exceed the frozen limit. Two extreme
  ratios are dominated by a near-zero official RSS denominator and need a
  measurement-quality fix as well as an allocation review.

In short: the core implementation and the high-value batch speedups are real.
The remaining work is concentrated in five well-defined areas rather than a
broad correctness or architecture rewrite.

## At-a-glance scorecard

| Area | ARM | x86 |
| --- | --- | --- |
| Pre/post semantic conformance | Pass | Pass |
| Architecture-specific gates | **50/60 pass** | **55/60 pass** |
| Single-thread encoded images | Pass; `1.50x–1.79x` | Pass; `2.36x–3.14x` |
| Single-thread raw RGB | **Fail; about `0.82x`** | Pass; `2.43x–2.55x` |
| Selected t8 batch speed vs official | Pass; `4.70x–6.43x` | Pass; `2.56x–3.70x` |
| image24 t8 parallel efficiency | **Fail; `51%–52%`** | **Fail; about `26%`** |
| Transient memory | Mixed; 6 of 12 fail | Strong overall; 2 of 12 fail |
| Timing noise | All pass | One slight official-baseline miss |
| Overall position | Fast, but raw RGB/scaling/memory remain | Speed-complete; scaling and two measurement edges remain |

The other two gates in the 122-gate total are the fixed M4 image24 point checks;
both pass comfortably. The candidate measures about `37.8 ms` against limits of
`111.0 ms` and `111.6 ms`.

## ARM: where we are

ARM is already compelling for encoded images and real batches. It passes every
selected speed gate except raw RGB, every noise gate, both text-regression
gates, and both fixed M4 point checks. Its ten architecture-specific misses are
two raw-RGB speed gates, two image24/t8 efficiency gates, and six memory gates.

The table shows median speedup versus official and the worst paired transient-
memory ratio. Speed must be above `1.0x`; memory must be at or below `0.5x`.

| ARM workload | Qwen3-VL speed / memory | Qwen3.5 speed / memory | Readout |
| --- | ---: | ---: | --- |
| image1, t1 | `1.52x` / `0.176x` | `1.51x` / `164.82x`* | Speed passes; one memory-floor problem |
| image24, t1 | `1.79x` / `0.070x` | `1.77x` / `0.062x` | Clear speed and memory win |
| image24, t8 | `6.43x` / `0.062x` | `6.38x` / `0.062x` | Excellent absolute result; scaling gate still misses |
| images_16, t8 | `5.28x` / `61.70x`* | `5.27x` / `1.17x` | Speed wins; both memory gates miss |
| ragged24, t8 | `4.75x` / `0.862x` | `4.70x` / `0.190x` | Speed wins; Qwen3-VL memory misses |
| rgb24, t1 | `0.819x` / `1.42x` | `0.815x` / `0.940x` | Primary ARM hot-path gap |

`*` These extreme ratios do not mean qwen-mm allocated hundreds of times more
memory in an absolute sense. The official process registered an extremely
small transient RSS increase in those repetitions. For example, the ARM
Qwen3.5 single-image candidate had an approximately 5 MB exact native peak,
while the paired official RSS delta could be only tens of kilobytes. The gate
correctly remains failed, but the next step must separate real candidate peak
use from RSS measurement-floor behavior.

The key ARM accomplishments are:

- image24 at t8 is approximately `37.8 ms`, or about `6.4x` faster than
  official;
- ragged24 is approximately `17.7–17.9 ms`, or about `4.7x` faster;
- image24 transient memory is only about `6%–7%` of official;
- long-text processing is about `1.62x` faster, so the image work did not create
  a text regression;
- the result is stable: all ARM timing-noise gates passed.

The ARM weaknesses are equally focused:

1. Raw caller-owned RGB takes about `112 ms` versus about `92 ms` official, so
   it needs a real single-thread hot-path optimization.
2. image24 improves by only about `4.1x` from qwen-mm t1 to t8, which is
   `51%–52%` eight-core efficiency rather than the required `60%`.
3. Several multi-image/ragged allocations live too long or are too large, even
   after discounting the measurement-floor cases.
4. The small-workload memory protocol needs to produce repeatably positive
   official denominators before those comparisons can be certified.

## x86: where we are

x86 is closer to completion. All 12 selected image speed gates pass, both text
regression gates pass, and ten of twelve memory gates pass. Its five misses are
two image24/t8 efficiency gates, two single-image memory gates, and one timing-
noise gate on the official Qwen3.5 raw-RGB baseline.

| x86 workload | Qwen3-VL speed / memory | Qwen3.5 speed / memory | Readout |
| --- | ---: | ---: | --- |
| image1, t1 | `2.36x` / `0.530x` | `2.38x` / unmeasurable* | Speed passes; both memory gates need closure |
| image24, t1 | `2.98x` / `0.075x` | `3.14x` / `0.070x` | Clear speed and memory win |
| image24, t8 | `3.48x` / `0.076x` | `3.63x` / `0.070x` | Fast and memory-light; scaling gate misses |
| images_16, t8 | `3.70x` / `0.070x` | `3.52x` / `0.074x` | Clear speed and memory win |
| ragged24, t8 | `2.56x` / `0.178x` | `2.57x` / `0.189x` | Clear speed and memory win |
| rgb24, t1 | `2.55x` / `0.252x` | `2.43x` / `0.251x` | Candidate wins; one official noise gate misses |

`*` The Qwen3.5 single-image memory gate is unmeasurable because two of the
three official-process RSS deltas were zero. Qwen3-VL is a genuine but small
miss at `0.530x` against the `0.500x` limit.

The key x86 accomplishments are:

- **every selected speed coordinate is faster than official with statistical
  support**, including raw RGB;
- single-thread image24 is about `3.0x–3.1x` faster;
- t8 image24 is about `167 ms`, or `3.5x–3.6x` faster than official;
- the ragged batch remains above the `2x` headline requirement;
- the important batch memory ratios are very strong, generally `0.07x–0.25x`;
- long-text processing is about `1.8x` faster.

The remaining x86 work is not basic speed. It is:

1. Improve internal t8 scaling. qwen-mm image24 only improves about `2.06x`
   from t1 to t8, which is roughly `26%` efficiency on an eight-core budget.
2. Fix or robustly characterize the single-image memory measurement, then trim
   the small real Qwen3-VL overage if it remains.
3. Stabilize one official Qwen3.5 raw-RGB timing series: its coefficient of
   variation was `0.053491` against a `0.050000` limit. This is a slight miss,
   but samples cannot be discarded or rerun selectively.

## Why fast t8 results can still fail the scaling gate

The absolute speed gate compares qwen-mm at t8 with the official processor at
t8. qwen-mm wins that comparison decisively because its underlying native path
is much faster.

The efficiency gate asks a different question: how much faster does qwen-mm
itself become when moving from one thread to eight? A perfect result would be
an `8x` improvement. The frozen gate requires at least `4.8x`, or 60%
efficiency.

- On ARM, qwen-mm goes from about `154–157 ms` at t1 to about `38 ms` at t8:
  roughly `4.1x` scaling, or `51%–52%` efficiency.
- On x86, it goes from about `343 ms` at t1 to about `167 ms` at t8: roughly
  `2.06x` scaling, or about `26%` efficiency.

So the current implementation is fast in absolute terms but leaves cores
underused, especially on x86. Likely investigation areas are work partitioning,
synchronization, allocator contention, resizer parallelism, and serial work
around the bounded pool. The follow-up must profile these rather than assume a
cause.

## Major work remaining

The 15 misses are completely covered by five Phase D tickets:

| Workstream | Ticket | What remains | Completion signal |
| --- | --- | --- | --- |
| ARM raw-RGB hot path | `725c` | Recover both ARM raw-RGB speed gates and both raw-RGB memory gates | Speed lower bounds `>1.0x`; memory ratios `<=0.5x` |
| Parallel scaling | `f49c` | Raise image24 t8 efficiency on ARM and x86 without oversubscription | All four `E_8 >= 0.6` |
| ARM batch memory | `b2a2` | Shorten/reuse scratch and arena lifetimes for images_16 and ragged24 | Three failed batch-memory ratios `<=0.5x` |
| Single-image memory | `8b8c` | Resolve zero/near-zero official RSS deltas and any real candidate peak excess | Three gates measurable with positive denominators and `<=0.5x` |
| x86 baseline stability | `82fe` | Explain and eliminate the slight official Qwen3.5 raw-RGB variance | All three medians retained with `CV <=0.05` |

These divide naturally into three parallel engineering lanes:

1. **CPU performance:** ARM raw RGB and cross-architecture t8 scaling.
2. **Memory:** single-image measurement quality plus real ARM batch peak
   reduction.
3. **Benchmark stability:** the one x86 official-baseline noise miss.

After those changes converge, D4 needs one new paired certification from an
identical exact source revision on ARM and x86. A new paid x86 run should be
requested only for that controlled final capture; no additional compute is
authorized by this report.

## Major accomplishments already banked

1. **Correctness survived optimization.** Both profiles passed pre- and
   post-run semantic conformance on both architectures. The performance work
   did not change public structure, ordering, dtypes, or stable errors.
2. **The core native image path is materially faster.** All x86 image speed
   gates and all ARM encoded/batch speed gates pass with statistical support.
3. **The highest-value batch workloads are strong.** ARM image24 and ragged24
   reach approximately `6.4x` and `4.7x`; x86 reaches approximately `3.5x` and
   `2.6x`.
4. **Most batch memory behavior is dramatically better.** The clearest cases
   use a small fraction of the official transient-memory delta rather than
   constructing another full float-CHW intermediate.
5. **Text did not regress.** Both profiles are faster on the long-text guard on
   both architectures.
6. **The evidence pipeline is now real and auditable.** ARM and x86 archives
   came from the same immutable source, retained raw samples and native memory
   counters, validated currentness, and reported all misses instead of stopping
   at the first one.
7. **The paid x86 workflow is operationally controlled.** The single approved
   Modal worker streamed structured stages and 30-second heartbeats, returned a
   validated archive, terminated, detached, and left no live container.

## What this result does not claim

This is the selected compact, shipping-only CPU preprocessing matrix. It does
not certify the deferred exhaustive A5 matrix, encoded video, vLLM integration,
or production end-to-end behavior. Those scopes remain separate and should not
be inferred from the strong image results.

For exact gate IDs, confidence intervals, raw measurements, provenance, and
reproduction commands, see the [generated certification report](report.md) and
the canonical [machine-readable result](result.json).
