# NVFP4 K=64 / K=96 tcgen05 power at 1500 MHz (TDP-safe peak)

**Date: 2026-04-20.** Same kernels as `NVFP4_K96_AB_FULL.md` but at
higher clock. Goal: confirm power levers scale with clock and identify
clock range that does NOT hit TDP cap.

## Clock behavior

- `-lgc 2032,2032` → clock pins to 1920 MHz (known behavior, `base` clock)
- `-rgc` (unlocked) → boosts to 2032 idle but **throttles to 1740 MHz**
  under sustained random-FP4 K=96 load at 1092 W
- `-lgc 1800,1800` → **throttles to 1700 MHz** at 1092 W (still TDP cap)
- **`-lgc 1500,1500`** → holds at 1500 MHz at 1003 W (below 1100 W cap) ✓

For stable comparisons at peak-ish speed on FP4 workloads, **1500 MHz
is the highest safe clock on B300**.

## Measurements — K=96 ULTRA (M=N=256)

PFLOPs/s = 10.89 (98.5% MFU, scales 1.49× from 1005 MHz's 7.31).

| B data pattern                   | power W | active W | TFLOPs/W |
|----------------------------------|---------|----------|----------|
| 5-pos {+0..+2} SF=1.0            | 680     | 530      | **16.0** |
| 5-pos SF=rand                    | 695     | 545      | 15.7     |
| 8 positive (no signs)            | 750     | 600      | 14.5     |
| 5 centered {-1..+1}              | 787     | 637      | 13.8     |
| 16 random full                   | 905     | 755      | 12.0     |
| OLD p_n=32 sign-period           | 751     | 601      | 14.5     |
| OLD p_n=64 sign-period (worst)   | 999     | 849      | **10.9** |

## Measurements — K=64 standard (M=N=256)

PFLOPs/s = 7.27 (98.4% MFU, scales 1.49× from 1005 MHz's 4.87).

| B data pattern                   | power W | active W | TFLOPs/W |
|----------------------------------|---------|----------|----------|
| 5-pos {+0..+2}                   | 565     | 415      | **12.9** |
| 8 positive (no signs)            | 617     | 467      | 11.8     |
| 5 centered {-1..+1}              | 642     | 492      | 11.3     |
| 16 random                        | 724     | 574      | 10.0     |

## Clock-scaling summary

Total power at 1500 MHz / 1005 MHz for each pattern:

| pattern | 1005 W | 1500 W | ratio | expected (1.49×) |
|---------|--------|--------|-------|------------------|
| K=96 5-pos         | 430 | 680  | 1.58× | 1.49× (+6%) |
| K=96 16 random     | 552 | 905  | 1.64× | 1.49× (+10%) |
| K=96 p_n=64 worst  | 605 | 999  | 1.65× | 1.49× (+11%) |
| K=64 5-pos         | 368 | 565  | 1.53× | 1.49× (+3%) |
| K=64 16 random     | 455 | 724  | 1.59× | 1.49× (+7%) |

**Power grows super-linearly with clock**: 1.49× clock → 1.53-1.65×
power. Higher clock needs higher voltage (Vdd×freq²×Cload model) +
static-power scaling.

## TFLOPs/W cross-clock comparison

| config                 | 1005 MHz TFLOPs/W | 1500 MHz TFLOPs/W |
|------------------------|-------------------|-------------------|
| K=96 ULTRA + 5-pos     | **17.0** ← best   | 16.0              |
| K=96 ULTRA + random    | 13.2              | 12.0              |
| K=96 ULTRA + p_n=64    | 12.1 (est.)       | 10.9              |
| K=64 std + 5-pos       | 13.2              | 12.9              |
| K=64 std + random      | 10.7              | 10.0              |

**1005 MHz is more efficient than 1500 MHz** by 6-10% for the same
pattern, due to super-linear power scaling. But 1500 MHz gives 49%
more throughput. **For power-bounded sustained workloads: 1005 MHz.
For latency/peak-throughput: 1500 MHz**.

## Confidence

- **HIGH** that clock holds at 1500 MHz across all tested patterns
  (verified by sampling during runs, max 1003 W).
- **HIGH** for the 98.5% MFU both paths (128 cy/MMA measured).
- **HIGH** for TFLOPs/W ordering 5-pos > 8-pos > centered > random
  at both clocks.
- **MED** that 1.58-1.65× power/clock ratio is the true scaling curve
  (depends on ambient temperature during measurement).

## What would change conclusions

- Test at 510 MHz lock (B300 min) to see bottom of scaling curve.
- Measure voltage via NVML to validate super-linear V²f model.
- Test at higher TDP limit (if accessible): maybe 1800 MHz viable
  with 1200 W cap.
- Real model weights under this setup (predict ~16 TFLOPs/W for
  inference).
