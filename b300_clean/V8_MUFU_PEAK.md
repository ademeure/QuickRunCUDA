# V8: MUFU (rsqrt) peak = 99.49% XU pipe utilization

## Findings

Tested `rsqrt.approx.f32` with 8-chain ILP, 256 threads × 148 blocks, 100K outer iters.

- ncu `sm__pipe_xu_cycles_active.avg.pct_of_peak_sustained_active` = **99.49%**
- Time: 6.34 ms
- Thread-level MUFU issued: 8M per thread × 37888 threads = 303 G MUFUs
- Rate: 303e9 / 6.34e-3 = **47.8 G thread-MUFU/s**
- Per SM per cycle: 47.8e9 / (148 × 2.032e9) = 0.159 thread-MUFU per SM per cycle
  = 1 thread-MUFU per 6.3 cycles per SM (aggregate)

## 10-rule walk-through

1. **Theoretical**: MUFU pipe is XU (separate from FMA). Per literature, rsqrt
   issues at ~1 per 4 cycles per SMSP → per SM peak = 4/4 = 1 warp-MUFU/cy.
   At 148 SMs × 2.032 GHz × 32 thread-MUFU/warp = 9.62 T thread-MUFU/s theoretical.

2. **Measured**: 47.8 G thread-MUFU/s — only 0.5% of above theoretical estimate.
   Yet ncu reports XU pipe 99.49% active — CONFLICT.

3. **Rule 3**: not "over peak" but metrics disagree.

4. **Likely explanation**: MUFU rsqrt on Blackwell is actually much slower
   than 1/4 per SMSP. Might be 1 per 32 cycles per SMSP (shared resource).
   99.49% pipe util means XU is fully busy at its actual throughput (which
   is lower than old estimates).

5. **ncu XU pipe** is counter-direct. 99.49% means the XU was the bottleneck
   and was busy ~every active cycle.

6. **SASS** (`sass/bench_v8_mufu_peak.sass`): 128 `MUFU.RSQ R, R` in loop. ✓

7-10. **Confidence: MEDIUM** for the 47.8 G MUFU/s number; LOW for the
    "1 per 4 cy" theoretical peak comparison. Need more literature research
    for true Blackwell MUFU ratio.

## Practical takeaway

- rsqrt.approx.f32 peak on B300: **~48 GMUFU/s aggregate**
- For a kernel issuing 1 rsqrt per 10 FFMAs, MUFU is not bottleneck
- For rsqrt-heavy (e.g., norm layers), MUFU can dominate

## Open question (deferred to V9)

What is Blackwell's true MUFU issue rate per SMSP per cycle? The 99.49% pipe
utilization proves the kernel saturates the XU — but mapping to "MUFU per cycle"
requires knowing the internal throughput spec. Official NVIDIA docs are
ambiguous for sm_100+ on MUFU rates.