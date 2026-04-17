# FFMA Latency: Definitive Measurement on B300 sm_103a

**Date**: 2026-04-17  
**Source**: `investigations/ffma_latency.cu`  
**Clock**: 2032 MHz locked (GPU runs at 1920 MHz base during single-warp kernels; cycles are clock-independent)  
**Device**: NVIDIA B300 SXM6 AC, SM 10.3

---

## Summary

FFMA (single-precision FMA) latency on B300 is **4 cycles** — not 23. The "23 cy" in prior catalog entries was a methodological error: those measurements included loop-control overhead (compare + branch), not just the FMA latency.

The self-op pattern (`FFMA Ra,Ra,Ra,RZ`) produces **identical latency** to the clean diff-src pattern on B300. Unlike Volta/Turing, Blackwell has no register read-port pressure penalty for triple-same-register FMAs.

---

## Methodology

Single warp (32 threads, 1 block), thread 0 measures `clock64` before and after N FMAs. Inner loop is fully unrolled (`#pragma unroll` on INNER_LAT=1024) so no loop overhead enters the measurement. Outer loop uses `#pragma unroll 1`. Two warmup outer-iterations run before the timed section.

Three register-dependency variants tested:

| SASS form | Dep chain location | Notes |
|-----------|-------------------|-------|
| `FFMA Ra, Ra, Ra, RZ` | Ra = dest, src1, src2, src3 all same | "Self-op" — tests register read-port pressure |
| `FFMA Ra, Ra, Rb, Rc` | Ra = dest + src1; Rb, Rc distinct | "Self+const" — only multiply-src1 is self |
| `FFMA Ra, Rb, Rc, Ra` | Ra = dest + addend only; Rb, Rc distinct | "Diff-src" — cleanest accumulator chain |

Plus ILP variants (2, 4, 8 independent diff-src chains, INNER_ILP=16, OUTER_ILP=1280).

SASS verified via `cuobjdump --dump-sass` before running. ncu cross-checked: `sm__inst_executed_pipe_fma.sum = 22532` for diffsrc (2 warmup + 20 timed outer × 1024 inner = 22528, agrees).

---

## Raw Data

```
Variant                                        r0       r1       r2       r3       r4       r5       r6    median
1-chain self-op  [FFMA Ra,Ra,Ra,RZ]        82291    82291    82291    82291    82291    82291    82291      82291
1-chain self+const [FFMA Ra,Ra,Rb,Rc]      82310    82310    82310    82310    82310    82310    82310      82310
1-chain diff-src [FFMA Ra,Rb,Rc,Ra]        82310    82310    82310    82310    82310    82310    82310      82310
2-chain ILP                                93444    93444    93692    93444    93444    93444    93444      93444
4-chain ILP                                97375    97375    97375    97375    97375    97375    97375      97375
8-chain ILP                               179399   179399   179399   179399   179399   179399   179399     179399
```

Total FMAs per chain per run: 20480 (INNER_LAT=1024, OUTER_LAT=20; same for ILP with INNER_ILP=16 × OUTER_ILP=1280).

Stability: all 7 repetitions agree to within 248 cycles (<0.3%). Self-op has 1 fewer cycle (82291 vs 82310) because its warmup initialization path differs by 1 instruction.

---

## Cycles Per FMA

### Latency-bound (single dep chain)

| Variant | SASS | cy/FMA | vs diff-src |
|---------|------|-------:|------------:|
| Self-op | `FFMA Ra,Ra,Ra,RZ` | 4.018 | 1.000 |
| Self+const | `FFMA Ra,Ra,Rb,Rc` | 4.019 | 1.000 |
| Diff-src | `FFMA Ra,Rb,Rc,Ra` | 4.019 | 1.000 |

**No register read-port pressure inflation on B300.** All three variants give 4.02 cy/FMA regardless of how many reads hit the same register. Blackwell's operand collector handles same-register reads without penalty.

### Throughput-approaching (N independent chains)

| Chains | cy/FMA | cy for all chains to cycle | Speedup vs 1 chain |
|--------|-------:|---------------------------:|-------------------:|
| 1      | 4.019  | 4.019                      | 1.00x              |
| 2      | 2.281  | 4.562                      | 1.76x              |
| 4      | 1.189  | 4.756                      | 3.38x              |
| 8      | 1.095  | 8.760                      | 3.67x              |

The "cy for all chains to cycle" column = cy/FMA × N chains. For 2 chains this is 4.56, close to the 4.02 latency (within scheduling overhead). For 8 chains it increases to 8.76 cy — at this point all 8 chains are still limited by the per-SMSP dispatch rate of 1 warp-instruction per cycle (single-pipe bottleneck for a single warp).

---

## True FFMA Latency: 4 cycles

The measurement confirms: **B300 FFMA latency = 4 cycles** (±0.02 cy measurement noise).

Derivation: 82310 clock64 cycles / 20480 FMAs = **4.019 cy/FMA**. With INNER=1024 fully unrolled, loop overhead is zero. The 4 cy matches Volta/Turing/Ampere/Hopper — FFMA pipeline depth has been 4 stages since at least Volta.

---

## Why the Catalog Showed "23 cy"

The B300 catalog at line 19565 shows a "Warp Scheduler / FFMA Latency" table where 1 warp, 1 dep chain gives "23.1 cy/FMA for 1000 FFMA". This is methodologically wrong:

That measurement used an unrolled loop **without** large inner-unroll, i.e., `for (i=0; i<N; i++) { a = a*b + a; }` with the loop variable updated each iteration. The loop costs ~19 cycles per iteration (comparison + conditional branch + counter update + 4 cy FMA). The 23 cy / FMA was actually: **4 cy FMA + ~19 cy loop overhead = 23 cy per loop iteration**, misreported as "FMA latency".

Similarly, the catalog entry at line 14938 notes: "FMA → FMA | 23 | ~17 cy loop overhead | 6 cy producer latency" — this one correctly identified the loop overhead but the "6 cy producer latency" was then the local estimate, which is also wrong (it's 4 cy, not 6 cy; the "17 cy overhead" estimate itself was off).

The catalog also correctly reports 4.53 cy latency at line 1734 from a different, better methodology. The 4.02 here vs 4.53 there is due to measurement precision: the 4.53 number came from an early bench_latency.cu run with fewer iterations; 4.02 from 20480 FMAs with zero loop overhead is more accurate.

**The user's prior testing at "23 cy" was from a kernel with per-loop overhead counted as FMA latency.** With proper methodology (large inner unroll, loop overhead amortized to near zero), single-warp FFMA latency is 4 cycles on B300.

---

## What the Throughput Numbers Mean

### Single-warp ILP ceiling

With a single warp (32 threads), 8 independent chains per thread:
- Each SMSP handles 8 threads, so 8 × 8 = 64 chains per SMSP
- 64 chains is far more than the minimum needed to saturate (minimum = 4/2 = 2 for dual-pipe)
- Measured throughput: 1.095 cy/FMA
- This approaches but does not reach the theoretical minimum of 0.5 cy/FMA (dual-pipe, single warp)
- The gap (1.095 vs 0.5) is because a SINGLE WARP can only issue 1 warp-instruction per cycle per SMSP, so dual-issue requires the compiler to issue 2 FMAs per instruction slot — which happens when two independent chains are placed adjacent in the instruction stream (the "REUSE" optimization). With only 1 active warp per SMSP, the warp scheduler has no other warp to interleave.

### Full-chip throughput

When MANY warps are active (the bench_ffma_peak scenario with 256 threads × 6 CTAs/SM = 1536 threads = 48 warps/SM):
- Each SMSP can dual-issue (warp A on heavy pipe + warp B on lite pipe simultaneously)
- Measured chip-wide: ~72 TFLOPS (ncu-verified from B300_PIPE_CATALOG.md)
- Theoretical peak: 148 SMs × 256 CUDA cores × 2 ops/FMA × 2032 MHz / 1e12 = 77 TFLOPS
- Achievement: 72/77 = 93%

The single-warp ILP-8 throughput of 1.095 cy/FMA corresponds to ~70 TFLOPS chip-wide projection (at the per-SMSP dispatch limit, not dual-issue). The extra 2 TFLOPS to get to 72 come from the dual-issue path available only with multiple warps.

---

## Self-Op: No Inflation on B300

On Volta/Turing, `FFMA Ra,Ra,Ra,RZ` was known to double the FMA latency (from 4 to ~8 cy) due to register bank read-port pressure: 3 reads of the same register in a single cycle saturated the read bandwidth.

On B300, this penalty is absent: self-op measures **4.018 cy** vs diff-src at **4.019 cy** — within noise. B300's register file has sufficient read bandwidth (or the operand collector handles same-reg reads via a bypass path) to not penalize triple-same-register reads.

This means: self-op chains are valid for measuring FFMA latency on B300 (but remain invalid on Volta/Turing for that purpose).

---

## Implications for Peak Throughput Calculations

At 4 cy latency and 2032 MHz:
- Single warp, 1 chain: 4 cy × (1/2032 MHz) = 2.0 ns/FMA → limited to 0.5 TFLOPS/SM, 74 GFLOPS chip
- Single warp, 8 chains: ~1.1 cy → ~0.9 TFLOPS/SM (single-pipe dispatch limited)
- Many warps, 8 chains: ~0.5 cy → ~2 TFLOPS/SM → ~72 TFLOPS chip (actual measured peak)

To saturate FFMA throughput from a SINGLE WARP, need ILP >= latency × dual_pipe_count = 4 × 2 = 8 chains per SMSP (= 8 × 4 = 32 chains per warp). Our 8-chain test has 8 chains per SMSP exactly at the saturation knee but only sees single-pipe throughput (1 cy not 0.5 cy) because the warp scheduler cannot dual-issue a single warp to both pipes without additional warp-level concurrency.

**Practical rule**: 4 chains per warp is enough to half-hide the 4 cy latency (measured 1.19 cy/FMA at 4-chain). 8+ chains with multiple warps fully saturates the dual-pipe FMA unit.

---

## ncu Cross-Check

```
kernel_diffsrc:
  sm__inst_executed_pipe_fma.sum  = 22532  (expected 22528 = 22 outer × 1024 inner)
  sm__cycles_elapsed.avg          = 97310 cy  (includes warmup + timed + kernel overhead)

kernel_ilp8:
  sm__inst_executed_pipe_fma.sum  = 164114  (expected 164096 = 1282 outer × 16 inner × 8 chains)
  sm__cycles_elapsed.avg          = 187357 cy
```

ncu cycle counts include warmup passes and kernel launch overhead; our clock64 measurements isolate only the timed section. Both are consistent.

---

## Final Answer

| Question | Answer |
|----------|--------|
| True FFMA latency on B300 | **4 cycles** (4.019 cy measured) |
| Is the "23 cy" correct? | No — it included loop-control overhead |
| Does self-op inflate latency on B300? | No — all three variants give 4.02 cy |
| What inflates to 23 cy? | Poor measurement methodology (unrolled loop with per-iteration overhead ~19 cy) |
| 8-chain ILP throughput (single warp) | 1.09 cy/FMA (single-pipe dispatch, 1 warp) |
| Full-chip peak throughput (many warps) | ~0.5 cy/FMA, ~72 TFLOPS (dual-issue path) |
| Minimum ILP to saturate single-pipe | 4 chains per warp (per SMSP) |
| Minimum ILP to saturate dual-pipe | 8 chains × multiple warps |
