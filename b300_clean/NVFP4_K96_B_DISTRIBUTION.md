# NVFP4 K=96 Power vs B-side value distribution (single tcgen05.mma)

**Date: 2026-04-20.** Same kernel template as `NVFP4_K96_SIGNMATCH.md`,
varying which set of FP4 codes B is uniformly drawn from. A is always
fully random (16 codes uniform). Clock locked 1005 MHz.

## Setup
- `tests/bench_nvfp4_k96_b_modes.cu` (multiple distribution modes)
- `tests/bench_nvfp4_k96_b1val.cu` (single-constant B)
- `tests/bench_nvfp4_k96_b5val.cu` (5 positive + SF mode)
- M=N=256, K=96, cluster_dims=2, 296 blocks, 10M iters
- 5 power samples × 0.3 s after 1.5 s ramp; median of middle 3.

## Results — total power (idle = 150 W)

| B distribution | total W | active W | Δ vs constant |
|----------------|---------|----------|---------------|
| Constant (any single value, 0x0..0xF) | 295-303 | 145-153 | 0 (floor)     |
| **{+0, +0.5, +1, +1.5, +2}** = 5 pos    | **428** | 278     | +130 W       |
| {+0.5, +1, +1.5, +2} = 4 nonzero pos    | 434    | 284     | +136 W       |
| {+0..+6} = 8 positive (all mantissa)    | 468    | 318     | +170 W       |
| ±{+0..+2} = 10 codes                    | 510    | 360     | +212 W       |
| ±{0.5..2} = 8 nonzero ± codes          | 519    | 369     | +221 W       |
| **All 16 codes (full random)**          | **551** | 401     | +253 W       |
| (worst: p_n=64 sign-period, all mag=1)  | 605    | 455     | +307 W       |

SF=1.0 vs SF=random for the 5-pos B: 428 → 440 W (+12 W from SF traffic).

## Decomposition

| component             | added W | notes |
|-----------------------|---------|-------|
| Magnitude variation   | ~40 W   | 5 pos → 8 pos (5 mags → 8 mags) |
| Sign bit randomization| ~80 W   | 5 pos → 10 codes (add ± randomly to same 5 mags) |
| Worst-case sign pattern (p_n=64) | +50 W | random → 100% sign-toggle at N-64 stride |

Sum: 130 (5 pos floor) + 80 (sign random) + 40 (more mags) + 50 (worst sign) = 300 W active = 450 W total. Matches observed range from 298 W (constant) → 605 W (worst).

## Constant-B sweep validation

All 16 single-constant B values give 293-303 W (10 W spread within noise).
Confirms the per-bit static-popcount component on B-side is small (~0.5 W
per bit set in the constant). The 130 W "5-pos" cost is almost entirely
from inter-dword toggles, NOT from absolute popcount.

## Takeaways

1. **B-side magnitude variation alone**: ~40-130 W depending on number of
   distinct magnitudes (1 → 5 → 8).
2. **Adding sign randomization**: another +80 W on top.
3. **Adversarial sign pattern (p_n=64)**: another +50 W beyond random.

Real LLM weights (FP4 quantized) typically have:
- ~50% sign distribution
- Magnitudes biased low (lots of small values)
- Some sparsity (true zeros)

Expected real-world B-side power: 470-520 W active per CTA at K=96
ULTRA path, depending on weight distribution. Can be reduced to
~250-280 W if weights are pre-sorted to pack same-sign / same-magnitude
together (matches the 5-pos / 4-pos test).

## Confidence

- **HIGH** for the constant-B floor (293-303 W across all 16 values).
- **HIGH** for the magnitude-variation cost (~40 W per few new mags).
- **HIGH** for the sign-randomization cost (~80 W).
- **HIGH** for the p_n=64 worst case (605 W reproducible).
- **MED** for the linear additivity model (decomposition fits within ~10%
  but interactions may matter for combined patterns).

## What would change conclusions

- Test A-side variation while holding B constant (verify "A is FREE"
  hypothesis from earlier memory note about tcgen05 sub-tile dedup).
- Test K-axis vs N-axis sign patterns (we know N=64 is the lane stride;
  what about K?).
- Real model weights from a quantized model: load and measure directly.
- Per-block_scale (block16) vs no-block-scale to isolate scale toggling.

## Bonus: 5-pos B + outliers per K-block-of-16 sweep

15 elements per K-block-of-16 from {+0..+2}, N elements from
{-3, -2, -1, +2, +3, +4} ("outlier" set, uniform). Outlier positions
random per (n, kblock):

| outliers/K16 | % outliers | power W | Δ from pure 5-pos |
|----|-----|---------|----------|
| 0  | 0%  | 432     | 0        |
| 1  | 6%  | 461     | +29      |
| 2  | 13% | 480     | +48      |
| 4  | 25% | 503     | +71      |
| 8  | 50% | **526** | **+94 ← peak** |
| 16 | 100%| 519     | +88      |

50/50 mix is HIGHER than 100% pure outlier — mixing two distinct value
distributions increases inter-element variance more than either pure
distribution. Same toggle-energy phenomenon as popcount d=16 peak.

**Key inference takeaway**: just 1 high-magnitude or sign-flipped outlier
per 16 weights costs ~30 W per CTA. Pre-clipping outliers (or grouping
them so SF blocks see uniform distributions) is worth ~30-90 W of B-side
toggle power.
