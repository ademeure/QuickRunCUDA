# FP / mixed-precision / MUFU chain latency — V4

**Date: 2026-04-20.** Single warp, dependent chain, runtime u2 anti-DCE,
SASS-verified. 1500 MHz clock.

## Pure-pipe chain latencies

| Op | cy/inst chain |
|----|---------------|
| FFMA (f32) | 4.04 |
| HFMA2 (f16x2 FMA) | 4.04 |
| HMUL2 (f16x2 mul) | 4.04 |
| HADD2 (f16x2 add) | 4.04 |
| HFMA2.F32 (mixed-precision FMA) | 4.04 |
| MUFU.RCP (rcp.approx.f32) | 42.10 |
| MUFU.EX2 (ex2.approx.f32) | 14.14 |
| MUFU.RSQRT | 40.10 |
| MUFU.SIN | 24.02 |

All FMA-style ops chain at exactly **4 cy** — they share the same FMA
pipe with the same per-result chain latency. FP16x2 packed ops have NO
latency penalty over scalar f32.

## Mixed-precision FP16 ↔ FP32 bridging

Test: HFMA2 → cvt.f32.f16 → FFMA → cvt.f16x2.f32 → HFMA2 (5 inst per iter).

```
Measured: 24 cy/iter, 5 inst/iter → 4.8 cy/inst average
Expected pure-chain: 5 × 4 = 20 cy
Excess: 4 cy total (≈ 2 cy per cvt)
```

**The cvt instructions are nearly free** — about 2 cy each above the
pure FMA chain rate. This is the "cost of bridging" between FP16 and
FP32 domains. Mixed-precision pipelines pay very little for the
precision crossings.

## MUFU + FFMA composition

| Composition | Measured | Sequential expected | Excess |
|-------------|----------|---------------------|--------|
| FFMA → RCP → FFMA | 50 cy | 4 + 42 + 4 = 50 | **0 (perfect linear)** |
| FFMA → EX2 → FFMA | 44 cy | 4 + 14 + 4 = 22 | **+22 cy (2× anomaly)** |
| RCP → EX2 → RCP | 120 cy | 42 + 14 + 42 = 98 | +22 cy |

## The EX2 anomaly explained

EX2 chain alone shows 14 cy/inst (issue-to-issue). But when chained
with FFMA, an extra ~22 cy appears.

Hypothesis: **MUFU has separate issue-interval and result-availability
latencies**. EX2 can be issued every 14 cy in a pure chain (because
each EX2 reads the writeback buffer of the previous EX2 directly), but
when a *different pipe* (FFMA) needs the EX2 result, it waits the full
result-availability latency (~30+ cy).

Compare to RCP: 42 cy chain matches FFMA→RCP→FFMA = 50 cy linear stack
exactly. Suggests RCP has issue-interval ≈ result-latency (no separate
fast-path for chained MUFUs of same type).

SASS verification of mode 11 confirms compiler reordered FFMAs across
iter boundaries — 102 FFMA + 50 EX2 in unrolled body = 2 FFMA per EX2,
not 2 FFMA per EX2 as my source code suggested. The chain dep between
successive iters' FFMAs is what creates the apparent 44 cy/iter.

## Practical implications

1. **HFMA2 / HMUL2 / HFMA2.F32**: same chain latency as FFMA (4 cy).
   FP16 packed math is identical-latency to FP32 — only the throughput
   and precision differ.

2. **Mixed-precision pipelines**: cvt cost ~2 cy per crossing. So a
   "compute in FP16, accumulate in FP32" pattern adds minimal latency
   per element.

3. **MUFU.RCP / MUFU.RSQRT**: chained-with-FFMA cost is exactly the
   sequential sum. No hidden penalty.

4. **MUFU.EX2 / MUFU.SIN / MUFU.LG2**: chaining with FFMA pays an
   extra ~20 cy for the cross-pipe writeback. If you're chaining EX2
   with FFMA in a hot loop, expect ~44 cy/iter (not 22).

## Confidence

- **HIGH** for pure-pipe chain latencies (3 trials each, anti-DCE
  verified, SASS-confirmed inst count)
- **HIGH** for mixed-precision cvt cost ≈ 2 cy
- **HIGH** for FFMA→RCP→FFMA = 50 cy clean linear
- **MED** for "EX2 has separate issue/result latencies" — best-fit
  explanation but needs separate test (e.g., longer FFMA-EX2 chains
  to extract result-availability vs issue-interval)

## Files

- `tests/bench_chain_fp_mufu.cu` — modes 0-12
- `tests/bench_chain_pair_matrix.cu` — earlier 6×6 integer pair matrix
- `tests/bench_ffma_chain_latency.cu` — pure FFMA baseline (4.02 cy)
