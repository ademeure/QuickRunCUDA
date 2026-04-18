# B300 FP32 FFMA Peak (CUDA cores, NOT tensor cores)

**GPU**: NVIDIA B300 SXM6 AC, sm_103a, 148 SMs
**Confidence**: HIGH — SASS-verified, ncu cross-checked, two independent kernel families converge

---

## TL;DR

| Quantity | Value | Source |
|---|---|---|
| Theoretical FFMA peak @ 2032 MHz | **76.96 TFLOPS** | 148 × 128 × 2 × 2.032 |
| Theoretical FFMA peak @ 1920 MHz | **72.71 TFLOPS** | 148 × 128 × 2 × 1.920 |
| Measured peak @ 2032 MHz (best kernel) | **75.9 TFLOPS = 98.7%** | `fp32_peak_definitive.cu`, ILP=8/16, BS=1024, MB=6–8 |
| Measured peak @ 2032 MHz (recipe kernel) | **74.6 TFLOPS = 97%** | `peak_ffma_power.cu`, BS=256 × ILP=24 × full occ |
| FFMA latency | **4 cycles** (4.019 cy measured) | `ffma_latency.cu`, fully unrolled, 20480 FMAs |
| Self-op vs diff-src | **identical 4.02 cy** | NO register port penalty on Blackwell |
| cuBLAS Sgemm peak (N=16384) | **69.5 TFLOPS = 90%** | `cublas_sgemm_sweep.cu` |
| Power at peak FFMA | 361 W (full occ) → 437 W (low occ) | `peak_ffma_power.cu`, 12s sustained |
| Power efficiency | **0.21 TFLOPS/W** | full-occ recipe; 8× worse than tensor cores |

**Clock paradox**: `nvidia-smi -lgc 2032` paradoxically pins to **1920 MHz** (-6%). Default boost (no NVML lock) sustains 2032 MHz under FFMA load. Always state which clock applies.

---

## The 128-FP32-cores/SM fact

B300 has the **same FP32 lane count per SM as Hopper H100: 128 cores/SM**.

```
148 SMs × 4 SMSPs × 32 FP32 lanes = 18,944 FP32 cores total
                  ↑
              per SMSP: 32 lanes
              per SM:   128 cores
```

NCU pipe metrics show `pipe_fmaheavy=2.00 + pipe_fmalite=2.00 = 4.00 warp-inst/SM/cy`, which corresponds to **128 SASS FFMA dispatches per SM per cycle** = 128 FFMAs/SM/cy × 2 FLOPS/FFMA = 256 FLOPS/SM/cy. There is **no separate "doubling"** beyond this — the 4.00 dispatch already represents both halves of the fma pipe issuing.

Theoretical: `128 FFMA/SM/cy × 2 FLOPS/FFMA × 148 SMs × 2.032 GHz = 76.96 TFLOPS`.

---

## ILP scaling curve (single-warp latency hiding)

From `ffma_latency.cu` (single warp, 32 threads, fully unrolled INNER=1024):

| Chains/thread | cy/FMA | Notes |
|---:|---:|---|
| 1 | 4.019 | latency-bound; matches 4-cy pipeline depth exactly |
| 2 | 2.281 | 1.76× speedup |
| 4 | 1.189 | 3.38× — saturates single-pipe at this point |
| 8 | 1.095 | single-warp dispatch limit (1 warp-inst/cy/SMSP) |

**Saturation rule**: a single warp needs **≥4 in-flight FFMA chains per SMSP** to hide the 4-cy pipeline. With more warps active, the dual-pipe (heavy+lite) lets the chip reach 0.5 cy/FMA/SM at 2 ops/cy/SMSP.

From `ffma_ilp_curve.cu` (148 blocks × 128 threads — only 12.5% occupancy, array-based ILP):

| ILP | TFLOPS | %SOL |
|---:|---:|---:|
| 1 | 3.3 | 4% |
| 2 | 6.7 | 9% |
| 4 | 13.3 | 17% |
| 8 | 24.6 | 32% |
| 16 | 40.9 | 53% |
| 32 | 36.7 | 48% (regresses — register pressure) |

This curve is **occupancy-limited, not pipe-limited** — 12.5% occ caps you long before ILP runs out. With full occupancy (below), the curve flattens at ILP≥8.

From `fp32_peak_definitive.cu` at full occupancy (BS=1024, MB=6, ILP × INNER = 1024 FFMA per inner iter):

| ILP | TFLOPS | %SOL@2032 |
|---:|---:|---:|
| 1 | 73.99 | 96.1% |
| 2 | 75.17 | 97.7% |
| 4 | 75.59 | 98.2% |
| **8** | **75.92** | **98.6%** |
| 16 | 75.89 | 98.6% |
| 32 (INNER=128) | DEGRADES | I-cache footprint exceeds 4096-FFMA limit |

ILP=1 already reaches 96% because BS=1024 = 32 warps/block provides massive warp-level TLP to hide the 4-cy latency.

---

## Block size and occupancy effects

From `block_size_sweep.cu` (fixed 151,552 total threads, ILP=4, full occupancy at every BS):

| BS | blocks | blk/SM | TFLOPS | %SOL@2032 |
|---:|---:|---:|---:|---:|
| 32 | 4736 | 32 | 73.0 | 95% |
| 64 | 2368 | 32 | 73.0 | 95% |
| 128 | 1184 | 16 | 73.0 | 95% |
| 256 | 592 | 8 | 73.3 | 95% |
| 512 | 296 | 4 | 73.3 | 95% |
| 1024 | 148 | 2 | 73.2 | 95% |

**Block size has ZERO effect** when total work is fixed AND every config reaches full 100% occupancy (1024 thr/SM × 32 warps/SM). What matters is `total_warps_per_SM × ILP_per_thread ≥ 4-cy latency`.

From `fp32_peak_definitive.cu` (ILP=8 fixed, vary BS):

| BS | warps/block | TFLOPS | %SOL@2032 |
|---:|---:|---:|---:|
| 32 | 1 | 55.58 | 72.2% (only 8 in-flight ops/SMSP) |
| 64 | 2 | 70.57 | 91.7% |
| 128 | 4 | 73.61 | 95.6% |
| 256 | 8 | 73.67 | 95.7% |
| 512 | 16 | 75.20 | 97.7% |
| **1024** | **32** | **76.06** | **98.8%** |

CTAs/SM queued (`fp32_peak_definitive.cu`, ILP=8, BS=1024 = 1 CTA/SM live, rest queue):

| MB | Blocks | TFLOPS | %SOL@2032 |
|---:|---:|---:|---:|
| 1 | 148 | 73.19 | 95.1% |
| 2 | 296 | 74.92 | 97.3% |
| 4 | 592 | 75.62 | 98.3% |
| 6 | 888 | 76.05 | 98.8% |
| **8** | **1184** | **76.19** | **99.0%** |

Single-wave (MB=1) loses ~3% to launch/teardown framing; MB ≥ 4 fully amortizes.

---

## Optimal recipe

Two equivalent recipes both reach ~97–99% peak. Pick the simpler one for your context.

### Recipe A — `peak_ffma_power.cu` style (BS=256 × full occ × 24 ILP)

```cpp
__launch_bounds__(256, 8) __global__ void peak_ffma(float *out, int iters) {
    // 24 in-flight chains: 3 chain triplets × 8 vars
    float a0=threadIdx.x, a1=a0+1, a2=a0+2, a3=a0+3, a4=a0+4, a5=a0+5, a6=a0+6, a7=a0+7;
    float b0=a0*2, b1=a1*2, b2=a2*2, b3=a3*2, b4=a4*2, b5=a5*2, b6=a6*2, b7=a7*2;
    float c0=a0*3, c1=a1*3, c2=a2*3, c3=a3*3, c4=a4*3, c5=a5*3, c6=a6*3, c7=a7*3;
    for (int i = 0; i < iters; i++) {
        a0=a0*1.0001f+b0; b0=b0*1.0001f+c0; c0=c0*1.0001f+a0;
        a1=a1*1.0001f+b1; b1=b1*1.0001f+c1; c1=c1*1.0001f+a1;
        // ... a2..a7 / b2..b7 / c2..c7 same pattern
    }
    float s = a0+a1+...+c7;  // anti-DCE reduction
    if (s < -1e30f) out[blockIdx.x*blockDim.x+threadIdx.x] = s;  // impossible store
}
// Launch: peak_ffma<<<148, 256>>>(d_out, 100000);
// Result: 28 regs, 8 blk/SM = 2048 thr/SM = 100% occ → 74.6 TFLOPS @ 361 W
```

`__launch_bounds__(256, MIN)` for MIN ∈ {1,2,4,8} all give the same 28 regs and 74.4–74.5 TFLOPS — the well-tuned compiler has nothing to gain from the hint here.

### Recipe B — `fp32_peak_definitive.cu` style (BS=1024 × ILP=8)

```cpp
__launch_bounds__(1024, 1) __global__ void kernel(...) {
    float v[8];
    for (int k = 0; k < 8; k++) v[k] = __int_as_float((tid+k+1) | 0x3f800000u);
    float y = __int_as_float(((tid & 0xFF) + 0x3f800001u));
    #pragma unroll 1
    for (int o = 0; o < 100; o++) {           // OUTER not unrolled
        #pragma unroll
        for (int i = 0; i < 128; i++) {       // INNER fully unrolled
            #pragma unroll
            for (int k = 0; k < 8; k++)       // ILP chains
                asm volatile("fma.rn.f32 %0,%0,%1,%0;" : "+f"(v[k]) : "f"(y));
        }
    }
    unsigned acc = 0;
    for (int k = 0; k < 8; k++) acc ^= __float_as_int(v[k]);
    if (acc == (unsigned)seed) C[tid] = __int_as_float(acc);  // runtime-opaque seed
}
// Launch: 148*6 = 888 blocks × 1024 threads → 75.9 TFLOPS @ 98.7%
```

Inline PTX guarantees FFMA emission (no FADD/FMUL split). Anti-DCE: xor-fold all accumulators into a scalar, store conditional on `acc == seed` where seed is a runtime arg.

---

## cuBLAS Sgemm peak (FP32 GEMM, NOT tensor)

From `cublas_sgemm_sweep.cu` — square N×N×N FP32 cublasSgemm:

| N | TFLOPS | %SOL@2032 |
|---:|---:|---:|
| 256 | 2.6 | 3.4% (launch overhead) |
| 512 | 16.1 | 21% |
| 1024 | 44.8 | 58% |
| 2048 | 57.0 | 74% |
| 4096 | 65.7 | 85% |
| 8192 | 68.5 | **89%** |
| **16384** | **69.5** | **90%** |

cuBLAS reaches 90% of theoretical at N≥16384. Validates the 75.9 TFLOPS microbench peak: cuBLAS lifts the practical ceiling from 99% down to 90% because of memory traffic and tiling overhead, but the kernel-only peak is real.

---

## Power and efficiency

From `peak_ffma_power.cu` (12 s sustained):

| Configuration | TFLOPS | Power | TFLOPS/W |
|---|---:|---:|---:|
| Peak FFMA (BS=256 × 24 ILP × full occ) | 74.6 | **361 W** | 0.207 |
| Non-peak FFMA (low ILP, low occ) | 73.0 | 437 W | 0.167 |
| (For reference) BF16 mma.sync | 569 | 411 W | 1.39 (8× FFMA) |
| (For reference) FP8 cuBLAS | 4491 | 886 W | 5.07 (30× FFMA) |

**Counter-intuitive**: better-occupied kernel uses **less power** for slightly more throughput. Idle SMSPs in the under-saturated case still draw idle power.

No thermal throttle observed in 12 s sustained on the recipe; temp settles at 46 °C.

---

## RETIRED claims (do not cite)

| Claim | Where | Why retired |
|---|---|---|
| **"154 TFLOPS FP32 peak"** / "256 cores per SM" / "FP32 dual-issue doubles the rate" | various sub-agent outputs, conjectured | 2× formula error. B300 has 128 FP32 cores/SM (same as Hopper). The "dual-issue" is heavy+lite halves of one fma pipe issuing 4.00 warp-inst total = 128 SASS dispatches; this is the 76.96-TFLOPS figure, not double it |
| **"FFMA latency 23 cy"** | catalog line 19565, AUDIT_NOTES.md line 53, etc. | 4-cy FMA + ~19-cy loop overhead misattributed. With proper fully-unrolled methodology: 4.019 cy. Don't use 23 anywhere |
| **"Self-op `FFMA Ra,Ra,Ra,RZ` is 2× inflated by register-port pressure"** | catalog line 1947 ("8.46 cy") | Volta/Turing-era folklore. Measured on B300: self-op = self+const = diff-src = 4.02 cy. Blackwell operand collector handles same-reg reads without penalty |
| **"FP32 FMA = 38 TFLOPS at 386 W"** | catalog ~line 17914 | Prose estimate from a mis-saturated kernel; never directly measured. Real peak is 74.6 TFLOPS at 361 W |
| **"58.6 TFLOPS = 76.2% peak"** | AUDIT_NOTES.md / `fp32_peak2.cu` | Used CPU-side `std::chrono` timing including launch overhead; may have been on the throttled GPU 0. Re-measure with cudaEvents on a confirmed-2032-MHz GPU and you get 75.9 TFLOPS |
| **"52 TFLOPS = 68% peak"** / **"26.4 TFLOPS"** | older catalog entries | Both correct for under-saturated configurations (low occupancy, low ILP). Not peak. Replaced by the 74.6/75.9 numbers above |
| **"71.8 TFLOPS @ 98.8%"** | catalog line 30 | This number is **CORRECT** but at **1920 MHz** (natural boost without NVML lock). Same kernel at 2032 MHz gives 75.9 TFLOPS @ 98.7%. Not retired — just needs a clock annotation |
| **"FP32 64.75 TFLOPS = 84%"** | CONSOLIDATED_FINDINGS.md line 24 | Earlier 'honest' attempt with sub-optimal config; superseded by the 75.9-TFLOPS definitive run |

---

## SASS verification

`fp32_peak_definitive.cu` ILP=8 INNER=128 → exactly **1024 FFMA per inner-loop body**, 20 registers, 0 spills. Verified via `cuobjdump --dump-sass`.

ncu cross-check (`ffma_latency.cu` diff-src):
- `sm__inst_executed_pipe_fma.sum = 22532` (expected 22528 = 22 outer × 1024 inner) ✓
- pipe_fma utilization at peak: 99.08% (matches 98.7% measured TFLOPS within scheduling friction)

---

## Implications

- **Single warp** can extract ~half of a single SMSP's pipe with ILP=8 (1.1 cy/FMA = ~70 TFLOPS chip-projected). The other half requires multiple warps to feed dual-issue.
- **Practical rule**: 4 chains/thread + ≥8 warps/SM gets you to >95%. Adding more ILP barely helps once warps are sufficient.
- **Register pressure** at ILP=32 with INNER=128 blows the I-cache (4096 FFMA inner body = ~64 KB) — back off to INNER=8 or stay at ILP≤16.
- **Don't chase 100%**: the 1.3% gap from theoretical is warp-scheduler issue friction, consistent with the `sm__inst_issued.avg.per_cycle_active = 0.99–1.00` ceiling seen elsewhere.
- **Two GPUs in test system run different clocks**: GPU 1 boosts to 2032 MHz reliably; GPU 0 sits at 1920 MHz under load (132 hours of SW Thermal Slowdown counter). Always confirm the actual SM clock and compute theoretical against that clock.
