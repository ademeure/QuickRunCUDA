# V8: HMMA.F16 tensor peak = 99.90% of pipe (578.6 TFLOPS)

## 10-rule rigor walk-through

1. **Theoretical**: B300 legacy tensor pipe (HMMA.F16) spec-referenced at
   540-580 TFLOPS. More precisely, mma.sync.m16n8k16.f16.f16 issues 1 per
   N cycles per SMSP. At measured 99.9% pipe utilization:
   - 94.72M HMMAs ÷ 670 µs → 141 GHmma/s
   - 141 GHmma/s × 4096 FLOP/HMMA = 578.6 TFLOPS
   - ncu `sm__pipe_tensor_cycles_active.pct_of_peak = 99.9%` confirms SoL

2. **Measured** (mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16, 148 × 256 thr, 8 chains, ITERS=10K):
   - ncu `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` = **99.90%**
   - ncu `sm__inst_executed_pipe_tensor.sum` = 94,720,000
   - gpu_time = 670.37 µs
   - **578.6 TFLOPS = 99.9% of HMMA.F16 peak**

3. **Rule 3**: 99.9% ≤ 100%. Not broken.

4. **Why 99.9%**: tensor pipe nearly fully saturated. The 0.1% gap is noise
   from kernel startup/tail not counted as "active".

5. **ncu HIGH confidence**: direct pipe_tensor metric from HW.

6. **SASS** (`sass/bench_v8_hmma_f16_peak.sass`):
   - Loop body: 8 × `HMMA.16816.F16 R, R, R, R` (self-feeding accumulator)
   - All 8 distinct accumulators (C0[0..7])
   - Loop iterates ITERS times

7. **Three methods**:
   - Wall clock (0.672 ms) matches ncu (670 µs)
   - ncu tensor pipe % (99.90%)
   - HMMA count (94.72M) matches expected: 148 × 8 warps × 10000 × 8 chains
   - All three converge to 578 TFLOPS

8. **Conclusive demonstration**: 99.9% pipe saturation directly measured via
   HW counter. The 578 TFLOPS is not formula-derived; it's
   `count × 4096 FLOP / measured_time`.

9. **Surprise checked**: initial 128-thread test gave only 0.42 TFLOPS.
   Fixed by increasing to 256 threads × 8 chains for full occupancy.
   SASS showed only 4 HMMAs vs expected — N_CHAINS=4 was low-ILP.

10. **Confidence: HIGH**. Would change if:
    - Tensor pipe has higher-throughput instruction we didn't test (e.g., HMMA.16832)
    - FP16 accumulator (.F16.F16) vs FP32 accumulator (.F16.F32) differs — not tested

## Recipe for HMMA.F16 peak

```cuda
__global__ __launch_bounds__(256, 1)
void kernel(...) {
    // 8 chains of independent accumulators per warp
    unsigned c0[8], c1[8];
    // ... init A/B fragments, c0[]/c1[] ...
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
                " {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
                : "+r"(c0[k]), "+r"(c1[k])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
            );
        }
    }
    // ... anti-DCE ...
}
```

Launch: **256 threads/block × 148 blocks** at boost clock → **578.6 TFLOPS HMMA.F16**.

## Context

- B300 HMMA.F16 = 578 TFLOPS (this measurement, 2-accumulator .F16 variant)
- B300 BF16 tcgen05.mma (Blackwell tensor memory path) = ~1980 TFLOPS (CLAUDE.md)
- B300 FP8 tcgen05 = ~4500 TFLOPS (CLAUDE.md)

Legacy HMMA is ~1/3 of Blackwell's tcgen05 BF16 peak but still accessible via
widely-used `mma.sync` PTX. Good baseline for non-cuTLASS kernels.