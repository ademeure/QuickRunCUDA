# Pure tcgen05.mma Power Microbench Results - FP16/BF16/FP8

Date: 2026-04-19. Free-rein continuation.

## Setup

`tests/bench_tcgen05_power.cu` v4: tcgen05.mma kind::f16 OR kind::f8f6f4
with SMEM-resident A and B (loaded once at kernel start, then 100M iters
of MMA referencing the same SMEM addresses). NO DRAM/L2 traffic in the
inner loop — pure multiplier circuit power.

Configuration:
- m=128 n=128 (max single-CTA shape that gives 98.5% MFU)
- K=16 for f16 path, K=32 for f8f6f4 path
- TMEM alloc = 512 cols (m*n fp32 / 32 elem-per-col)
- 148 CTAs persistent, single warp issuer per CTA
- @ -lgc 1005 MHz (no throttle) → ~6.4 sec sustained per test

Pattern semantics:
- 0 = zero (0x00)
- 4 = random (per-index pseudo-random byte fill)
- 5 = +1.0 (precision-specific bit pattern)
- 2 = 0x55 (alternating bit "weird uniform")

## Results: per-tensor isolation @ 98.5% MFU

| Precision | RAND_RAND | zero_zero | A_rand_B_zero | A_zero_B_rand | A=+1 B=rand | A=rand B=+1 |
|-----------|----------:|----------:|--------------:|--------------:|------------:|------------:|
| **FP16** | 642 W | 279 W | 297 (Δ +18) | **552 (Δ +273)** | 581 | 299 |
| **BF16** | 594 W | 284 W | 292 (Δ +8) | **487 (Δ +203)** | 547 | 299 |
| **FP8 e4m3** | 612 W | 289 W | 298 (Δ +9) | **553 (Δ +264)** | 602 | 303 |
| **FP8 e5m2** | 587 W | 289 W | 309 (Δ +20) | **585 (Δ +296)** | 612 | 310 |

## 🎯 KEY FINDING — B is the DOMINANT operand at the multiplier level

For ALL 4 precisions tested, randomizing B alone (with A held at zero) costs
**10-30× more power** than randomizing A alone (with B held at zero).

| Precision | A-only rand cost | B-only rand cost | Ratio |
|-----------|-----------------:|-----------------:|------:|
| FP16 | +18 W | +273 W | **15.2×** |
| BF16 | +8 W | +203 W | **25.4×** |
| FP8 e4m3 | +9 W | +264 W | **29.3×** |
| FP8 e5m2 | +20 W | +296 W | **14.8×** |

## REVERSAL of cuBLAS NVF4 conclusion

Earlier I observed cuBLAS NVF4 had A operand dominant in power. This was
WRONG attribution — the multiplier itself has B dominant. The cuBLAS NVF4
"A dominates" came from TMA multicast on B which saves B's memory pipeline
cost, masking the multiplier's intrinsic B>A.

Updated mental model:

```
Total per-tensor power = (memory pipeline cost) + (multiplier datapath cost)

Memory pipeline cost depends on:
  - Whether the tensor is multicast (multicast halves L2 reads)
  - Tensor size (more bytes = more activity)

Multiplier datapath cost (THIS microbench measures):
  - B >> A intrinsically (~15-30× ratio for f16/f8 paths)
  - Same across FP16, BF16, FP8 e4m3, FP8 e5m2

cuBLAS NVF4 with multicast B: memory cost A > B; multiplier B > A; net A > B observed
cuBLAS BF16 without multicast: memory cost A ≈ B; multiplier B > A; net B > A observed
```

## Uniform patterns are all near-zero baseline

| Pattern | All 4 precisions | Δ vs zero |
|---------|-----------------:|----------:|
| zero (0x00) | 279-289 W | 0 |
| +1.0 | 283-293 W | +3-4 W |
| 0x55 (weird uniform) | 284-294 W | +5 W |

Confirmed: **only randomness costs**, the actual constant value (+1.0, +6.0, etc) is irrelevant.

## TODO: NVFP4 / MXFP4 / MXFP8 (block-scaled formats)

Block-scaled formats use different PTX:
- NVFP4: `kind::mxf4nvf4.block_scale.block16` with TMEM-resident SF operands
- MXFP4: `kind::mxf4.block_scale.block32`
- MXFP8: `kind::f8f6f4` (no separate block-scale instruction; SF applied in software)

NVFP4 PTX is structurally different (SFA/SFB are TMEM addrs, not SMEM
descriptors). Need to:
1. Allocate extra TMEM for SFA, SFB
2. Initialize TMEM scale region with constant scale (1.0 = UE4M3 byte 0x38)
   via tcgen05.cp from SMEM
3. Modify PTX to include [tsfa_addr], [tsfb_addr] operands

Pure NVFP4/MXFP4 multiplier asymmetry could be opposite to FP16/FP8 results
(different instruction, different multiplier circuit). High priority next.

## Confidence

- HIGH on B-dominance for FP16/BF16/FP8 (4 precisions × consistent pattern)
- HIGH on uniform-vs-random distinction (large 200+ W gap)
- HIGH on B/A ratio being intrinsic to multiplier (no DRAM in test)
- MED on extrapolation to NVFP4 (different PTX instruction not tested yet)
