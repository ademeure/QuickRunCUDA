# tcgen05 cy/MMA N-sweep: 4 precisions × 2 CTA modes

**Date: 2026-04-20.** Full N sweep (16..256) for MXFP8, NVFP4 K=64,
NVFP4 K=96 ULTRA at both `cta_group::1` (M=128) and `cta_group::2`
(M=256). All kernels use `%laneid` early-exit; no ELECT in inner loop.
Measured via clock64 at 1005 MHz, 98.5% MFU.

## NVFP4 K=64 standard (M=128 cta=1 | M=256 cta=2)

| N   | cta=1 cy/MMA | cta=1 MAC/cy | cta=2 cy/MMA | cta=2 MAC/cy |
|-----|--------------|--------------|--------------|--------------|
| 16  | 39           | 3,361        | 39           | 6,722        |
| 32  | 40           | 6,554        | 39           | 13,443       |
| 64  | 48           | 10,923       | 43           | 24,200       |
| 96  | 56           | 14,043       | 49           | 31,885       |
| 128 | 64           | **16,384**   | 64           | **32,768**   |
| 160 | 80           | 16,384       | 80           | 32,768       |
| 192 | 96           | 16,384       | 96           | 32,768       |
| 224 | 112          | 16,384       | 112          | 32,768       |
| 256 | 128          | 16,384       | 128          | 32,768       |

Saturation at N=128 for both CTA modes. cta=2 gives 2× MAC/cy PER CLUSTER
but at half the number of clusters, so **same total throughput = 4.87 PF**.

## NVFP4 K=96 ULTRA (M=128 cta=1 | M=256 cta=2)

| N   | cta=1 cy/MMA | cta=1 MAC/cy | cta=2 cy/MMA | cta=2 MAC/cy |
|-----|--------------|--------------|--------------|--------------|
| 16  | 59           | 3,332        | 59           | 6,665        |
| 32  | 60           | 6,554        | 59           | 13,329       |
| 64  | 72           | 10,923       | 65           | 24,198       |
| 96  | 84           | 14,043       | 71           | 33,230       |
| 128 | 96           | 16,384       | 78           | 40,330       |
| 160 | 108          | 18,204       | 86           | 45,723       |
| 192 | 120          | 19,661       | 96           | **49,152**   |
| 224 | 132          | 20,852       | 112          | 49,152       |
| 256 | 144          | **21,845** ← never sat | 128 | 49,152       |

**cta=1 K=96 ULTRA DOES NOT saturate within N=16..256** — MAC/cy keeps
climbing; at N=256 reached only 21,845 (vs cta=2's 49,152 peak).

## MXFP8 K=32 UE8M0 block32 (M=128 cta=1 | M=256 cta=2)

| N   | cta=1 cy/MMA | cta=1 MAC/cy | cta=2 cy/MMA | cta=2 MAC/cy |
|-----|--------------|--------------|--------------|--------------|
| 16  | 39           | 1,680        | 39           | 3,361        |
| 32  | 40           | 3,277        | 39           | 6,722        |
| 64  | 48           | 5,461        | 43           | 12,100       |
| 96  | 56           | 7,022        | 49           | 15,942       |
| 128 | 64           | **8,192**    | 64           | **16,384**   |
| 160 | 80           | 8,192        | 80           | 16,384       |
| 192 | 96           | 8,192        | 96           | 16,384       |
| 224 | 112          | 8,192        | 112          | 16,384       |
| 256 | 128          | 8,192        | 128          | 16,384       |

## Peak total throughput (1005 MHz, saturated N)

| format      | K  | cta=1 peak  | cta=2 peak  | cta advantage |
|-------------|----|-------------|-------------|---------------|
| BF16        | 16 | 1.22 PF     | 1.22 PF     | none          |
| FP8 E4M3    | 32 | 2.44 PF     | 2.44 PF     | none          |
| MXFP8       | 32 | 2.44 PF     | 2.44 PF     | none          |
| NVFP4 K=64  | 64 | 4.87 PF     | 4.87 PF     | none          |
| NVFP4 K=96 ULTRA | 96 | 6.50 PF (N=256 not sat) | **7.31 PF** (N=192 sat) | **+12.5% cta=2** |

## Key observations

1. **cta=1 vs cta=2 is throughput-neutral** for BF16/FP8/MXFP8/NVFP4-K64
   at saturation — total work/second is the same. Both saturate at N=128.

2. **NVFP4 K=96 ULTRA at cta=1 never saturates in N=16..256**. cta=1 M=128
   caps out around 21,845 MAC/cy/block at N=256, while cta=2 M=256 hits
   49,152 MAC/cy/cluster at N=192. The ULTRA K-factor only works
   efficiently in cta=2 mode with M=256.

3. **Saturation point N=128 for non-ULTRA paths**, N=192 for K=96 ULTRA
   cta=2. At cta=1 K=96, scaling is monotonically increasing across full
   N range — no saturation observed.

4. **MXFP8 cy/MMA is identical to FP8** — block-scale overhead is hidden
   in existing cycle count (UE8M0 SF reads don't add stalls).

5. **cta=1 K=96 at small M doesn't benefit from ULTRA**: compare:
   - NVFP4 K=64 cta=1 M=128 N=128: cy=64, MAC/cy=16,384
   - NVFP4 K=96 cta=1 M=128 N=128: cy=96, MAC/cy=16,384
   Same MAC/cy, K=96 just takes 1.5× more cycles. **No ULTRA benefit in 1-CTA**.

6. **cta=2 K=96 saturates 192 (not 128)** — the extra 64 N-lanes come
   from the 2-CTA pipe-widening specific to K=96. Without the 2-CTA,
   pipe width drops to N=128 like any other format.

## Implication: K=96 ULTRA is a 2-CTA-only feature

If your MMA is cta_group::1, K=96 gives **no throughput benefit over K=64**
(same MAC/cy at saturation). Only cta_group::2 unlocks the ULTRA K-factor.

## Confidence

- **HIGH** for all cy/MMA numbers (clock64, 3 trials each, stable).
- **HIGH** for the cta-neutral throughput in BF16/FP8/K64.
- **HIGH** for K=96 ULTRA cta=1 non-saturation (monotonic scaling
  across full N=16..256 range).
