# tcgen05 perf/W ladder — 7 precisions @ 1500 MHz, M=N=256 cta=2

**Date: 2026-04-20.** Full precision ladder, sustained 10s tcgen05 MMA
compute (no memory traffic, A+B in SMEM, accum in tmem). Clock locked
1500 MHz (TDP-safe max — `-lgc 1500,1500`). Lane-0-only kernel
(`%laneid` early-exit). Peak PF from 7.31 × (clock/1005) × (format-specific).

## Peak throughput + power table

| format       | K  | cy/MMA | peak PF @ 1005 | pwr @ 1500 (W) | PF @ 1500 | **TF/W** |
|--------------|----|--------|-----------------|-----------------|-----------|----------|
| TF32         | 8  | 128    | 0.61            | 816             | 0.91      | 1.12     |
| FP16         | 16 | 128    | 1.22            | 940             | 1.82      | 1.94     |
| BF16         | 16 | 128    | 1.22            | 829             | 1.82      | **2.20** |
| FP8 E4M3     | 32 | 128    | 2.44            | 854             | 3.64      | 4.26     |
| MXFP8        | 32 | 128    | 2.44            | 851             | 3.64      | **4.28** |
| NVFP4 K=64   | 64 | 128    | 4.87            | 797             | 7.27      | 9.12     |
| **NVFP4 K=96** | 96 | 128  | **7.31**        | 795             | **10.91** | **🏆 13.72** |

## Key findings

### 1. Precision vs efficiency: monotonic improvement
Each halving of element size ~doubles TF/W (with ULTRA K-factor
giving an extra 1.5× on top for NVFP4 K=96):
- TF32 → BF16: 2.0× TF/W (K=8 → K=16, plus 19→16 bit mantissa)
- BF16 → FP8: 1.9× TF/W
- FP8 → NVFP4 K=64: 2.1× TF/W (halve element size)
- NVFP4 K=64 → K=96: 1.5× TF/W (pure K-factor, no element change)

**Total BF16 → NVFP4 K=96: 6.2× TF/W improvement.**

### 2. BF16 beats FP16 by 12% efficiency at same throughput
Both run at 128 cy/MMA, same 1.82 PF. But:
- FP16 burns 940 W
- BF16 burns 829 W (**-111 W**)

Root cause: BF16 has 7-bit mantissa; FP16 has 10-bit mantissa.
FP16's wider mantissa creates more multiplier bit-switching activity,
even with identical throughput.

### 3. NVFP4 K=96 ULTRA at 13.72 TF/W is the absolute best efficiency
No other precision/K combination tested gets above 10 TF/W. K=96 ULTRA
combines:
- 4-bit elements (minimum)
- K=96 (1.5× K=64)
- 2-CTA wider pipe (N=192 sat vs N=128 for smaller formats)

### 4. MXFP8 = FP8 to within 0.5%
Block-scale overhead (UE8M0 SF reads) is completely hidden at the cy
and power level. No practical difference between FP8 and MXFP8 for
throughput or efficiency.

## NVFP4 K=96 fine N sweep (192..256 step 8)

Past saturation at N=192, cy/MMA scales exactly linearly (0.5 cy per N):

| N   | cy/MMA | MAC/cy | PF @ 1005 |
|-----|--------|--------|-----------|
| 192 | 96     | 49,152 | 7.31 ← sweet spot |
| 208 | 104    | 49,152 | 7.31      |
| 224 | 112    | 49,152 | 7.31      |
| 240 | 120    | 49,152 | 7.31      |
| 256 | 128    | 49,152 | 7.31      |

**All multiples of 16 are valid. N=200, 216, 232, 248 fail with
illegal instruction.** For peak throughput with minimum latency per
MMA, prefer N=192.

## Confidence

- **HIGH** for all peak PF values (lane-0 kernel, clock64 measurement).
- **HIGH** for BF16 < FP16 power ordering (110 W gap is well outside
  measurement noise).
- **HIGH** for NVFP4 K=96 13.72 TF/W being best among tcgen05 precisions.
- **MED** for absolute PF @ 1500 assuming 98.5% MFU holds at 1500 MHz
  (extrapolated from 1005 MHz measurement via clock-scaling).

## Pitfall encountered

Measured cy/MMA was 8.5× inflated (1088 vs 128) when 5 other
QuickRunCUDA processes were still running from earlier tests.
`kill -9 $PID` wasn't cleaning up reliably. **Always `pkill -9
QuickRunCUDA` before a clean measurement.**
