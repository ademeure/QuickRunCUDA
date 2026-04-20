# tcgen05 MMA cy/MMA vs N: BF16 / FP8 / NVFP4 throughput characterization

**Date: 2026-04-20.** Measured cy/MMA via clock64 for M=256, varying N,
across 3 tcgen05 MMA precisions. All using `__cluster_dims__(2,1,1)`.
Kernel uses `%laneid` early-exit for clean SASS (no ELECT between UTCOMMAs).

Sources:
- `tests/bench_bf16_n_lane_fast.cu` — kind::f16, K=16
- `tests/bench_fp8_n_lane_fast.cu` — kind::f8f6f4, K=32
- `tests/bench_nvfp4_k96_lane_fast.cu` — kind::mxf4nvf4.block_scale.block16, K=96

## cy/MMA scaling (M=256, 2-CTA)

| N   | BF16 K=16 | FP8 K=32 | NVFP4 K=96 |
|-----|-----------|----------|------------|
| 16  | 39        | 39       | —          |
| 32  | 39        | 39       | —          |
| 64  | 43        | 43       | 65         |
| 96  | 49        | 49       | 71         |
| 128 | 64        | 64       | 78         |
| 160 | 80        | 80       | 86         |
| 192 | 96        | 96       | 96         |
| 224 | 112       | 112      | 112        |
| 256 | 128       | 128      | 128        |

**BF16 and FP8 have IDENTICAL cycle counts at every N** — element size
does not affect timing. What differs is MAC/cy (twice as many FP8
elements fit in same bit-width).

## MAC throughput per cluster-cycle

| N   | BF16       | FP8        | NVFP4 K=96 |
|-----|-----------|------------|------------|
| 16  | 1,680     | 3,361      | —          |
| 32  | 3,361     | 6,722      | —          |
| 64  | 6,096     | 12,193     | 24,198     |
| 96  | 7,971     | 15,942     | 33,230     |
| 128 | **8,192** | **16,384** | 40,330     |
| 160 | 8,192     | 16,384     | 45,723     |
| 192 | 8,192     | 16,384     | **49,152** |
| 256 | 8,192     | 16,384     | 49,152     |

Saturation points: **BF16 N=128, FP8 N=128, NVFP4 N=192**.

## Peak throughput at 1005 MHz

All kernels reach 98.5% MFU at saturation-point N or larger:

| Format   | Peak MAC/cy | Peak PFLOPs (74 clusters) |
|----------|-------------|----------------------------|
| BF16 K=16 | 8,192      | **1.22 PF**                |
| FP8 K=32  | 16,384     | **2.44 PF** (= 2× BF16)    |
| NVFP4 K=96 (ULTRA) | 49,152 | **7.31 PF** (= 6× BF16) |

Ratio breakdown for NVFP4 / BF16 = 6×:
  4× element-size (16b → 4b) × 1.5× ULTRA K-factor (K=96 vs K=64 standard path)

## Structural observations

1. **Cycle count is element-size independent.** BF16 and FP8 run in
   exactly the same number of cycles. The multiplier array processes a
   fixed number of BITS per cycle; formats differ only in how many
   elements that bit-count represents.

2. **NVFP4 ULTRA has a wider pipe (N=192 vs N=128)** — 1.5× more N-lanes
   at 4-bit elements than BF16 has at 16-bit elements. The extra width
   is the "ULTRA" path.

3. **Small-N latency floor ~39 cy** for BF16/FP8 (N ≤ 32), ~65 cy for
   NVFP4. Below that, the pipe under-utilizes — rises from 20% to
   100% saturation as N approaches the sat-point.

4. **Post-saturation, cy scales linearly with N** (0.5 cy per N for
   BF16/FP8, 0.5 cy per N for NVFP4). Past sat, MAC/cy stays constant.

## Practical implications

- For peak throughput, choose N ≥ saturation point:
  - BF16/FP8: N ≥ 128
  - NVFP4 ULTRA: N ≥ 192
- Small-N GEMMs (e.g. single-token inference) under-utilize the pipe.
  At N=16 BF16 only gets 21% of peak throughput per cluster.
- Doubling N (to 256) when sat-point already reached gives NO extra
  throughput — just wastes power and SMEM.

## Confidence

- **HIGH** for the cy/MMA numbers — clock64 measurement, stable across
  trials, ELECT eliminated from inner loop via `%laneid` early-exit.
- **HIGH** for the BF16=FP8 cycle identity (direct A/B comparison).
- **HIGH** for peak PFLOPs numbers (98.5% MFU verified).
- **MED** for the explanation of saturation-point structure; could be
  verified by looking at UTCOMMA.2CTA.BLOCK16 instruction latency
  metadata from ncu if available.
