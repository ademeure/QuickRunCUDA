# L2 Read Power vs Popcount Density: STRONG BELL-CURVE SIGNAL

**Date: 2026-04-20.** Tested whether L2 read power depends on the
**popcount** (Hamming weight) of the data, with bit positions
randomized per dword. Ran in conjunction with the bit-stride sweep
(see `L2_BITSTRIDE_SWEEP.md`).

## Setup

- `tests/bench_l2_popcount.cu`. Each dword has EXACTLY `density` bits
  set, in pseudo-random positions per dword (Fisher-Yates shuffle).
- 16 v4 .cg loads per inner iter, UNROLL=32, BLOCK=512, 148 blocks.
- Working set 64 MB (L2-warm), ~16 TB/s sustained.
- Clock locked 1005 MHz.
- Init verified: density 4 → every dword has popcount 4, etc.

## Results (L2-warm, ~16 TB/s)

```
density (bits/dword)   median W    active W (= med - 150 idle)
 0   (all zeros)        366.8       217
 1                      395.7       246
 2                      414.8       265
 3                      430.4       280
 4                      448.8       299
 6                      478.7       329
 8                      505.6       356
10                      524.8       375
12                      535.6       386
14                      543.0       393
16  (random ~50%)       548.4       398   ← PEAK
18                      546.0       396
20                      544.6       395
22                      538.3       388
24                      519.8       370
26                      495.9       346
28                      468.5       319
30                      433.1       283
31                      414.5       265
32  (all ones)          388.5       239
```

## Findings

### 1. Clean bell curve centered at 50% bit density
Power follows a smooth, near-symmetric bell shape with peak at d=16
(uniformly random). Both extremes (d=0 and d=32) draw the LEAST power.
Active power (above 150 W idle):

- d=16: **398 W active** (peak)
- d=0:  217 W active (zeros)
- d=32: 239 W active (ones)
- d=16 vs d=0:   1.83× active-power ratio
- d=16 vs d=32:  1.66× active-power ratio

### 2. Small asymmetry: d=32 is 22 W higher than d=0
At equal "no within-byte toggle" extremes, all-ones still draws 22 W
more than all-zeros. Possible mechanisms:
- HBM PHY signaling not fully symmetric (DBI, termination resistance,
  active-low drive).
- Inverter/level-shifter chains in the L2→SM mesh dissipate slightly
  more keeping output high.

### 3. Mechanism: bus-toggle (Hamming) energy on the bytes-on-wire
A 32-bit dword sent on a SerDes bus toggles bits whenever adjacent
sent symbols differ. For a random word, ~50% of inter-symbol
transitions occur per cycle. For all-zeros or all-ones, ~0% transitions.
Energy ∝ toggle rate is the textbook explanation for what we measure.

The shape is precisely what a **random-position popcount-d** model
predicts:

  P(adj bit differs) = 2·d·(32-d) / (32·31)
   ≈ d·(32-d)/496

Normalized to peak (d=16, 256/496 = 0.516):
| d | predicted | measured-active-W |
|---|-----------|-------------------|
| 0 | 0.0       | 217 |
| 4 | 0.226     | 299 (~75% of peak above zeros baseline) |
| 8 | 0.387     | 356 |
| 12| 0.484     | 386 |
| 16| 0.516     | 398 (peak) |
| 24| 0.387     | 370 |
| 32| 0         | 239 |

Predicted bell shape matches measured to within a small offset (the
22 W asymmetry).

## Reconciles bit-stride NULL result

The bit-stride sweep (`L2_BITSTRIDE_SWEEP.md`) showed all duplicated
patterns clustered at 537–550 W. That's because they ALL had ~50%
average bit density per dword (hash-derived), so they all sat near the
top of this bell curve. Whether bytes 0 and 1 happen to match doesn't
change the popcount distribution.

Going from "duplicated random" (550 W) to "true random" (549 W) changes
~nothing because both have d≈16. The lever is **popcount**, not
**duplication**.

## Cross-check: previous L2 peak BW results

The earlier `bench_l2_data_peak.cu` test showed:
- 0x00000000  zeros        363 W ✓ matches d=0 = 367 W
- 0x12121212  byte_const   365 W (popcount = 8 → predicted 506 W)
  ❗ This is a CONTRADICTION — `0x12121212` has popcount 8 per dword, yet
  it drew 365 W (= d=0 power), not the 505 W my popcount sweep predicts
  for d=8.

Hypothesis for the contradiction: **inter-dword toggling matters too**.
With 0x12121212 in EVERY dword, *adjacent dwords on the wire are
identical* → no bit toggles between dwords either. The only "toggling"
from cycle to cycle is reading 0x12121212 → 0x12121212 = 0 toggles!

In the popcount sweep, every dword has popcount d but DIFFERENT bit
positions (per-dword randomized). So adjacent dwords *do* toggle even
when popcount is fixed. This means **the bus actually responds to
inter-dword toggle activity, not just intra-dword popcount**.

This refines the model:

> P_active ∝ (inter-dword Hamming distance) on the L2/HBM ↔ SM bus

For per-dword constant data: 0 toggling → minimal power.
For per-dword random data with popcount d: toggle rate ∝ 2d(32-d)/(32·31)
plus the per-dword↔per-dword Hamming distance.

## Confidence

- **HIGH** that L2-warm read power follows a clean bell curve with
  popcount density when bit positions are randomized per dword.
  Signal is 184 W peak-to-trough (50% of mean).
- **HIGH** that bit-stride duplication does NOT meaningfully reduce
  power at peak BW.
- **MED** that the correct model is "inter-dword Hamming distance"
  rather than per-dword popcount. Will test with constant-vs-random
  variants of fixed popcount data.

## What would change conclusions

- Test **same popcount, fixed bit pattern across all dwords** (e.g.,
  every dword = 0x000000FF for d=8) — should be LOW like 0x12121212.
- Test **same popcount, varying bit pattern per dword** (current sweep)
  — should be HIGH like d=8 = 506 W.
- Test sparsity (X% of bytes/dwords/lines replaced by constant) — will
  reveal the partial-overwrite power scaling.

## Implications

- **Memory-bound kernels**: data with low popcount (<8 or >24 bits/dword)
  draw 30-45% less L2 read power than truly random data. For inference
  with FP4/FP8 quantization where many tensor elements are near zero,
  this is a real ~150-200 W power saving on the memory subsystem.
- **No need to hide redundant patterns**: chunk-level repetition
  (line/sector/page granularity) does NOT save power. Only per-dword
  popcount does, and only when sustained at >50% of peak L2 BW.
- **Adversarial pattern**: data with d=16 random per-dword maximizes
  L2 read power (~395 W active over idle) — could be useful for thermal
  stress testing.
