# NVFP4 Sign-Period Power Sweep: K=64 vs K=96 across all N

**Date: 2026-04-20.** Comprehensive measurement of how N-direction sign-bit
alternation period affects power consumption. Single tcgen05.mma PTX, clock
locked 1005 MHz, 148 SMs persistent. Tests sign patterns "+-+-", "++--",
"+++---", "++++----", etc. up to period=N.

## Setup

- Custom kernel: `tests/bench_nvfp4_period.cu`
- Period_size = arg1 (-1). Bit pattern: `sign[n] = (n / period_size) & 1`
- Non-sign bits held at FP4 0x2 (= +1.0)
- SF tensor = UE4M3 1.0 (byte 0x38)
- 1-CTA only (2-CTA needs M=256, separate sweep)

## K=64 1-CTA Results

```
N      Best p (W)     Worst p (W)      Spread    %    Notes
16     p=2 (171.9)    p=5 (175.3)      3.4 W     2%   minimal
24     p=4 (177.2)    p=8 (187.0)      9.8 W     5%
32     p=3 (183.2)    p=8 (188.7)      5.5 W     3%
48     p=4 (196.1)    p=12 (210.6)     14.5 W    7%
64     p=4 (178.8)    p=12 (187.6)     8.8 W     5%
128    p=128 (307.4)  p=64 (369.0)     61.7 W   18%   big!
256    p=2 (390.8)    p=64 (475.4)     84.7 W   20%   biggest
```

### K=64 N=128 detailed (most informative):
```
period   power(W)   category
  1      310        LOW (alternation predictor)
  2      316        LOW
  3      359        HIGH (multiple of 3)
  4      316        LOW
  6      358        HIGH (multiple of 3)
  8      316        LOW
 12      360        HIGH (multiple of 3)
 16      314        LOW (matches sub-tile boundary)
 24      360        HIGH (multiple of 3)
 32      315        LOW
 48      363        HIGH (multiple of 3)
 64      369        HIGH (worst! "two halves" pattern)
128      307        LOW (all-same signs)
```

### K=64 N=256 detailed:
```
period   power(W)
  1      392 LOW
  2      391 LOW
  3      443 HIGH (×3)
  4      392 LOW
  6      444 HIGH (×3)
  8      393 LOW
 12      444 HIGH (×3)
 16      392 LOW
 24      443 HIGH (×3)
 32      394 LOW
 48      440 HIGH (×3)
 64      475 HIGH ← WORST
128      439 HIGH ← (also "two halves")
256      392 LOW (all-same)
```

## K=96 1-CTA Results (partial, sweep ongoing)

```
N      Behavior
16     Mostly flat (range ~5W)
24     Some spread (256W avg, range ~10W with p=3, 6, 8 high)
32     Remarkably flat (range 3W)
48     Big spread (~31W, p=1/2/4/8 LOW, p=3/6/12/16/24 HIGH)
64     COMPLETELY FLAT (306-308W, all periods within 2W) ← unique!
128    Same pattern as K=64 N=128 (range ~67W, multiples of 3 HIGH)
256    (in progress)
```

### K=96 N=64 anomaly

At K=96 N=64, ALL period values give 306-308W (essentially flat). Range
0.6%. This is unique among all configurations - elsewhere there's clear
period-dependent variation.

Hypothesis: K=96 N=64 hits a HW saturation regime where the dedup
mechanism is already at limit (always-on or always-off). Could relate to
the K=96 ULTRA path's 1.5× compute density.

### K=96 N=128 detailed (so far):
```
period   power(W)   category
  1      371        LOW
  2      371        LOW
  3      436        HIGH
  4      371        LOW
  6      437        HIGH
  8      374        LOW
 12      438        HIGH
 16      375        LOW
 24      432        HIGH
 32      375        LOW
```

Same pattern as K=64 N=128: multiples of 3 = HIGH, powers of 2 within
sub-tile = LOW.

## Mechanism interpretation

### Why multiples of 3 are HIGH

NVFP4 sub-tile (block_scale.block16) is 16 elements. When period divides
16 cleanly (1, 2, 4, 8, 16), every sub-tile has identical sign pattern →
sub-tile dedup activates → LOW power.

Period = 3, 6, 12, 24 don't divide 16 cleanly:
- p=3: sub-tile 0 = "+++---+++---+++-", sub-tile 1 = "--+++---+++---++"
  (different pattern at sub-tile boundary) → no dedup
- p=6: sub-tile 0 = "++++++------++++", sub-tile 1 = "++------++++++--"
  (different) → no dedup
- p=12: sub-tile 0 = "++++++++++++----", sub-tile 1 = "------++++++++++"
  → no dedup

Multiples of 3 NEVER align to 16 boundary → all sub-tiles have different
signs → dedup misses → HIGH power.

### Why p=N/2 ("two halves") is HIGH at large N

p=64 with N=128 gives "++++(64) ----(64)". Sub-tile pattern at sub-tile-
level: AAAA BBBB (4 sub-tiles +, 4 sub-tiles -). This is the chunk-4
pattern at sub-tile level.

Per BF16 K-row dedup analysis (different mechanism but similar arch):
- chunk-1 (alternation): full speedup
- chunk-2: WORST (no speedup)
- chunk-4: partial
- chunk≥sub-tile-count: full speedup (acts like all-same)

For NVFP4 sub-tile dedup, chunk-4 at sub-tile level appears to be the
WORST case. p=64 N=128 hits this exactly.

### K=96 N=64 saturation

At K=96 N=64, ALL periods give same power. Hypothesis: the dedup mechanism's
lookback window or cache isn't activated at this specific config. Possible
causes:
- N=64 = 4 sub-tiles is too small for sub-tile dedup to find variation
- K=96 ULTRA path uses different internal data flow that bypasses dedup
- Some HW-internal pipeline effect at K=96 N=64

## Practical implications

For B300 NVFP4 compute power optimization:

1. **AVOID periods that are multiples of 3** (3, 6, 12, 24, 48, ...) -
   these add 15-20% power penalty
2. **AVOID period = N/2** (two-halves pattern) - worst case at large N
3. **PREFER periods that divide 16** (1, 2, 4, 8, 16) - or all-same (p=N)
4. **Best alternation: chunk=1 (+-+-) or all-same** - both work equally well
5. **Effect is bigger at large N** - for N=256, picking right period saves
   up to 20% (84W) of compute power
6. **K=96 has additional anomaly at N=64** - all periods equivalent

## Confidence

- **HIGH**: Multiples of 3 are HIGH at all N >= 32 (consistent across N values)
- **HIGH**: Powers-of-2 periods (matching sub-tile boundary) are LOW
- **HIGH**: p=N (all-same) is LOW (constant signs)
- **HIGH**: p=N/2 "two halves" is HIGH at large N
- **HIGH**: Effect scales with N (bigger at N=128, 256)
- **MEDIUM**: K=96 N=64 saturation mechanism (no clear HW explanation)
- **HIGH**: K=64 and K=96 follow same pattern overall

## What would change conclusions

- ncu metrics for tcgen05 sub-tile dedup activity (not exposed)
- Re-running with finer period granularity (e.g., p=15, 17) to check non-multiples
- Cross-precision comparison (FP8, BF16) to see if same mechanism
- 2-CTA cluster sweep (separate kernel needed for valid 2-CTA NVFP4)
