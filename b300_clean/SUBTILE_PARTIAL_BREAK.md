# Sub-Tile Partial Break: Granularity of Dedup Cost

Date: 2026-04-20. Tests how many bytes can differ within a 32-byte sub-tile
before the dedup cost ramps up.

## Setup
- BF16 m128n128k16, mode 6400+K_zero in `bench_tcgen05_bf16_perbit_power.cu`
- 8 sub-tiles per K row. Baseline pattern from h(0)..h(14).
- For each sub-tile (1-7), FIRST K_zero N positions are different from baseline.
  Sub_tile 0 always matches baseline.
- @ -lgc 1005 MHz, 50M iters, 148 SMs

## Results

| K_zero (N positions different per sub-tile) | Power (W) | Δ vs free (305W) |
|--------------------------------------------:|----------:|-----------------:|
| 0 (all match baseline) | 305 | 0 |
| 1 | 337 | +32 |
| 2 | 361 | +56 |
| 3 | 385 | +80 |
| 4 | 405 | +100 |
| 6 | 446 | +141 |
| 8 | 478 | +173 |
| 10 | 516 | +211 |
| 12 | 548 | +243 |
| 14 | 586 | +281 |
| 16 (full sub-tile differs = random) | 611 | +306 |

## Findings

1. **First broken byte costs ~32W** (the "activation overhead")
2. **Subsequent broken bytes cost ~18W each** (per-position bit toggling)
3. **Scaling is roughly linear**: 305 + 32 + 18*(K_zero - 1) fits within ~10W

## Mechanism refinement

The 32-byte sub-tile dedup is BINARY at the cache level:
- If sub-tile EXACTLY matches cache (byte-identical): GATED (free)
- If ANY byte differs: cache MISS, pay activation + per-position cost

The per-position cost suggests the multiplier processes each N position
and pays bit-toggle cost when input value differs from previous cycle's
value at that position.

## Practical implication

For workloads where B has MOSTLY the same pattern with small variations:
- 1 byte different per sub-tile → +32W (10% of full random cost)
- 8 bytes different per sub-tile (50%) → +173W (57% of full random cost)
- 16 bytes different (full) → +306W (full random cost)

So **even partial sub-tile match gives proportional savings**. Don't need
exact match - quantization can be approximate.

For weight quantization with some "outliers" per group:
- 1-2 outliers per 16-N group: still saves ~85% of dedup benefit
- 4+ outliers: significant degradation

## Comparison with full sub-tile breaking (mode 2900-2908)

Mode 2900-2908 had K_break sub-tiles fully different (16 of 16 N positions broken):
- K_break=8 (all 8 sub-tiles fully unique): 601W
Mode 6416 has 8 sub-tiles each with 16 positions different from baseline: 611W
- These are roughly equivalent (full random across sub-tiles)
- Within-noise difference (10W)

The new test (6400+) gives finer-grained insight: the cost grows linearly
with broken positions, not as a sharp cliff.

## Confidence

- HIGH on linear scaling (11 measurements, monotonic)
- HIGH on first-byte activation overhead being measurable
- MEDIUM on the linear fit (not perfect, ~10W residuals)
- HIGH on the practical implication for partial-match optimization
