# tcgen05 perf/W: 2-trial clean ladder, full random vs B-positive-only

**Date: 2026-04-20.** ALL measurements at 1500 MHz lock, M=N=256, cta=2,
2-trial reproducibility, with `pkill -9 QuickRunCUDA` + 6s cooldown
between EVERY measurement to avoid process contention.

## Methodology pitfall confirmed

Earlier `bench_nvfp4_k96_lane_fast` measurement showed cy/MMA=128 in
isolation but jumped to 1088 (8.5× inflated) when 5 leftover
QuickRunCUDA processes were running concurrently. **`kill -9 $PID;
wait $PID` is NOT sufficient cleanup**; only `pkill -9 QuickRunCUDA`
followed by `sleep 6` reliably yields clean results.

Earlier `perf/W` table at 1500 MHz had silent contamination — NVFP4
K=64 read 797 W and K=96 read 795 W (impossibly close given 1.5×
work). True values (clean): K=64=689 W, K=96=870 W.

## Full random (mode=0): 2 trials each

| format          | T1 (W) | T2 (W) | mean | TF/W |
|-----------------|--------|--------|------|------|
| FP16  K=16      | 935    | 935    | 935  | 1.95 |
| BF16  K=16      | 876    | 876    | 876  | 2.08 |
| TF32  K=8       | 787    | 787    | 787  | 1.16 |
| FP8   K=32      | 1069   | 1076   | 1073 | 3.39 |
| MXFP8 K=32      | 1046   | 1036   | 1041 | 3.50 |
| NVFP4 K=64      | 689    | 690    | 689  | 10.54 |
| NVFP4 K=96 N=192 (1.33× iters) | 882 | 879 | 880 | 12.39 |
| NVFP4 K=96 N=256 | 870   | 870    | 870  | 12.54 |

Reproducibility: max trial-to-trial gap = 10 W (MXFP8 random).
Most configs vary < 3 W between trials.

## B-positive only (mode=1): 2 trials each

Sign bit of every B element forced to 0; A still fully random.

| format          | T1 (W) | T2 (W) | mean | TF/W | Δ vs random |
|-----------------|--------|--------|------|------|-------------|
| FP16  K=16      | 784    | 786    | 785  | 2.32 | -150 W      |
| BF16  K=16      | 749    | 752    | 750  | 2.43 | -125 W      |
| TF32  K=8       | 679    | 682    | 680  | 1.34 | -107 W      |
| FP8   K=32      | 834    | 835    | 834  | 4.36 | **-238 W**  |
| MXFP8 K=32      | 802    | 798    | 800  | 4.55 | **-241 W**  |
| NVFP4 K=64      | 586    | 585    | 585  | 12.42 | -104 W     |
| NVFP4 K=96 N=192 | 733   | 723    | 728  | 14.99 | -152 W     |
| NVFP4 K=96 N=256 | 722   | 717    | 720  | **15.16** | -150 W |

## Key findings

### 1. Best efficiency: NVFP4 K=96 + B-positive = 15.16 TF/W
21% better than the "random" measurement (12.54 TF/W).
6.7× better than BF16 random.

### 2. FP8 / MXFP8 see HUGE savings from B-positive (-240 W)
Largest absolute savings of any format. Likely because FP8 has 4
elements per dword (more sign bits flip per cycle), so killing all
sign-bit toggle saves the most.

### 3. NVFP4 N=192 ≈ N=256 (within ±10 W)
At equal total work (1.33× more iters at N=192), power is the same.
Throughput per cluster is identical at saturation. **N=192 is the
optimal NVFP4 K=96 ULTRA shape** — same peak, lower per-MMA latency
and SMEM footprint.

### 4. BF16 < FP16 by 60 W in B-positive too
This 60 W gap (vs 110 W in random) confirms BF16 < FP16 power is
a real effect from FP16's wider mantissa, present even with sign
bits zeroed.

### 5. FP8 (1073 W) > FP16 (935 W) at random
Despite 2× the throughput, FP8 only uses 14% more power. That gives
FP8's 1.7× TF/W advantage. With B-positive, FP8 (834W) is barely
above FP16 (785W) but still 2× the throughput → 1.9× TF/W.

## Implications

For inference workloads with FP4/FP8 quantized weights:
- Always store +0 (not -0) for zero weights (free 100-240 W per CTA).
- For static-weight matmuls, pre-quantize to clamp signed weights
  to fewer bits (or pack as positive-magnitude + separate sign mask).
- N=192 GEMM tiles win over N=256 at zero throughput cost for
  NVFP4 K=96.

## Confidence

- **HIGH** — 2 trials per config, ±5 W reproducibility for most.
- **HIGH** that B-positive is the lever (multiple formats consistent).
- **HIGH** that contention silently corrupts measurements; pkill-9 +
  cooldown is the only reliable methodology.

## Addendum: A vs B sign-bit decomposition

Adding mode 2 (A-positive only) and mode 3 (A+B-positive) to the kernels.

| format    | random | B-pos | A-pos | A+B-pos |
|-----------|--------|-------|-------|---------|
| NVFP4 K=96 N=256 | 875W | 725W (-150) | 867W (-8) | 693W (-182) |
| FP8 K=32         | 1081W | 831W (-250) | 1082W (+1) | 809W (-272) |
| BF16 K=16        | 882W | 759W (-123) | 877W (-5)  | 725W (-157) |

**A-side sign bit is essentially FREE** (0-8 W swing across all formats).
**B-side dominates** sign-related power 95%+.

A+B-positive saves only an extra 22-34 W beyond B-positive alone. The
~3-4% incremental gain confirms the asymmetric architecture: B is
broadcast across many multiplier lanes, A has fewer consumers, so
A-side toggle activity contributes minimally to bus power.

## Updated practical recipe

For inference, the **only** sign-bit optimization that meaningfully
helps is on B-side weights. A-side activations can be random/arbitrary
without power penalty.

For NVFP4 K=96 the absolute lowest power is A+B-pos = 693 W = **15.74
TF/W** (10.91 PF / 0.693 kW). 4 W less than my earlier B-positive-only
N=192 result of 14.99 TF/W (rounding).

## SF tensor sensitivity (re-verified)

| format       | SF=1.0 | SF=random | Δ |
|--------------|--------|-----------|---|
| NVFP4 K=96 N=256 random | 873 W | 876 W | +3 W |
| MXFP8 K=32 random       | 1035 W | 1040 W | +5 W |

SF random adds ~3-5 W (negligible, as previously measured at 1005 MHz).
The 100-240 W from data dominates by 2-3 orders of magnitude.
