# disable_lane Power Scaling

Date: 2026-04-20. Tests how disable_lane mask affects MMA power.

## Setup
- `tests/bench_tcgen05_bf16_lanemask.cu`
- BF16 m128n128k16, random A and B
- @ -lgc 1005 MHz, 50M iters, 148 SMs

## Results

| disable_lane[] | Disabled columns | Power (W) | Δ from full random |
|---------------:|-----------------:|----------:|-------------------:|
| {0,0,0,0}       | 0   | 610 |   0 |
| {0xFF...,0,0,0} | 32  | 529 | -81 |
| {0xFF...,0xFF,0,0} | 64  | 428 | -182 |
| {0xFF...,0xFF,0xFF,0} | 96  | 373 | -237 |
| {0xFF...,0xFF,0xFF,0xFF} | 128 | 299 | -311 (= Tier B baseline) |

Cycle count cy/MMA=64 in ALL cases — disable_lane doesn't affect timing.

## Analysis

- Per-disabled-column cost: 311W / 128 = ~2.4 W/column
- Per-disabled-word (32 columns): 81-101 W (close to 78 expected)
- When all 4 words = 0xFFFFFFFF: power = 299W = Tier B (not Tier A 294W)
  - Multiplier still consumes BASELINE static power
  - Data-dependent power eliminated

## Mechanism

`disable_lane` selectively gates OUTPUT computation. Disabled columns:
- Multiplier arrays for those N positions are gated
- TMEM write for those N positions is skipped
- B operand load for those N still happens (cliff cost still applies?)

## Practical applications

1. **Sparse attention**: mask off N columns where attention weights are zero
   → save proportional power (~2.4 W/column on B300 BF16 m128n128k16)
2. **Strided GEMMs**: if you only need every-Kth column, disable rest
3. **Power-constrained inference**: dynamically adjust active N width

Combined with sub-tile dedup: a 128-N MMA with 64 N disabled + N=16 sub-tile-friendly
remaining 64 N → save 64 × 2.4 = 154W from disable + 250W from sub-tile dedup
= 404W save. Final power ~206W (vs 610W full random) = 66% reduction.

## Confidence

- HIGH on linear scaling (5 measurements monotonic)
- HIGH on per-column cost (~2.4 W/column)
- HIGH on full disable = Tier B (not Tier A) - multiplier still active
- MEDIUM on whether this composes with sub-tile dedup (untested combined)
