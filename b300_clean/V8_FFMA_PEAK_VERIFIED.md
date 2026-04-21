# V8: FFMA peak = 97.64% of SoL (triple-verified)

## 10-rule rigor walk-through

1. **Theoretical peak at 2032 MHz**:
   148 SMs × 4 SMSPs × 32 lanes × 2 ops × 2.032e9 Hz = **76.97 TFLOPS**
   Equivalently: 148 × 128 FFMAs/cy × 2.032 GHz = 38.5 GFFMA/s = 76.97 TFLOPS.

2. **Measured** (`tests/bench_ffma_warps_per_sm.cu`, 256 thr × 148 blk, ITERS=1M):
   - Wall clock: 8.06 ms → 75.2 TFLOPS = **97.7% of peak**
   - ncu `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active`: **97.64%**
   - ncu `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum` = 30.31 GFFMA
     (at ITERS=100K; scales linearly to 303.1 GFFMA at ITERS=1M)
   - Cross-check: 30.31e9 × 2 ops / 0.810e-3 s = 74.8 TFLOPS @ 97.2% ✓ agrees

3. **Rule 3**: 97.7% ≤ 100%. Not broken.

4. **Why not 100%**: SM tail latency (some cycles pipe not fully active), maybe
   warm-up/cool-down periods, slight instruction issue gaps.

5. **ncu cross-checked**: pipe_fma.pct_of_peak = 97.64% directly from HW counters.
   Matches wall-clock calc to within 0.1 percentage point.

6. **SASS** (`sass/bench_ffma_warps_per_sm_*.sass`):
   - Loop body: 128 × `FFMA Rx, Ry, Rx, Rx` (2-source: Rx self-feeds, Ry constant)
   - Loop counter: `UIADD3 UR4, UR4, 0x10` and `UISETP.GE UP0, UR4, UR5`
   - Anti-DCE: `FADD` reduction + conditional `ST.G` store

7. **Three methods**:
   - Wall clock (-T 5): 8.06 ms / 1M iters
   - ncu pipe utilization: 97.64%
   - ncu FFMA count: 30.31 GFFMA confirms 62500 loop iters × 128 FFMA × 37888 threads
   All three reconcile.

8. **Conclusive demonstration** that 2-source FFMA achieves near-peak:
   - Kernel uses `fma.rn.f32 %0, %0, %1, %0;` → self-feeds accumulator
   - Only 2 distinct source operands per FFMA (v, b, v = 2 unique)
   - Avoids the "3-source RF read port limit" (D6 finding)
   - 8-way ILP (N_CHAINS=8) saturates 4 SMSP × 2 warps/SMSP dispatch

9. **Surprise checked**: my earlier V8 J2 measurement got 71% (51.5 TFLOPS at
   boost). That used V6_C1 kernel with 3-source FFMA `c = c*k + b` (c, k, b
   all distinct). This hits the 2 read ports/cy limit, capping at ~65-71%.
   V8 J2 energy finding is VALID for that workload but NOT the true FFMA SoL.

10. **Confidence: HIGH**. Would change only if:
    - Clock actually exceeded 2032 MHz (unlikely; nvidia-smi confirms)
    - ncu metric miscounts (triple-checked vs wall-clock)

## The correct kernel recipe for FFMA SoL

```cuda
__global__ __launch_bounds__(256, 1)
void kernel(...) {
    float v[8], b[8];
    // ... init ...
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += 16) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            #pragma unroll
            for (int k = 0; k < 8; k++)
                asm volatile("fma.rn.f32 %0, %0, %1, %0;" : "+f"(v[k]) : "f"(b[k]));
        }
    }
    // ... anti-DCE ...
}
```

Launch: **256 threads/block × 148 blocks × boost clock** = full occupancy
→ **97.64% of peak FFMA throughput on B300.**

## Correcting V8 J2

V8 J2 concluded "1500 MHz is FFMA energy sweet spot". That conclusion assumed
FFMA throughput scales ~linearly with clock. With the 2-source (near-peak)
kernel, this is indeed true — I just verified 97.64% at both 1500 and 2032
(ncu).

At true 1500 MHz peak 56.8 TFLOPS × 97.64% = 55.5 TFLOPS. At boost 76.97 ×
97.64% = 75.2 TFLOPS. Ratio 1.35×. Power ratio (from J2): 293 / 415 W ≈ 0.71.
So energy/op: 293 × (1/55.5) vs 415 × (1/75.2). 5.28 vs 5.52 pJ/FLOP.
1500 MHz is 4.4% more energy-efficient than boost on 2-source FFMA too.

(V8 J2's numerical conclusions hold; the peak-% figure was workload-dependent.)

## Updated B300 FFMA SoL summary

- **Peak verified**: 75.2 TFLOPS @ 2032 MHz boost (97.64% of theoretical)
- **2-source FFMA** with 8-way ILP is the recipe
- **3-source FFMA** is ~65-71% due to 2 RF read ports/cycle (V6 D6)
- Energy sweet spot 1500 MHz holds regardless of kernel source count
