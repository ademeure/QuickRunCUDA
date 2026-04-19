# BF16 Sub-Tile Dedup: Major Finding

Date: 2026-04-19. Built `tests/bench_tcgen05_bf16_perbit_power.cu` modes 2700-2908.
Ran @ -lgc 1005 MHz, 50M iters, 148-block persistent, GPU 0.
Plateau power median (drop ramp samples by 95% threshold of max).

## Summary: Discovered N=16 sub-tile dedup mechanism

The "N-vary is free" claim from prior runs was a LOW-ENTROPY ARTIFACT.
With HIGH-ENTROPY N-vary values, a sharp cliff emerges based on number
of unique value patterns across N positions.

## Pure N-vary HIGH-ENTROPY (mode 2700-2707): N_unique sweep, K constant

| N_unique | Power (W) | Δ vs Tier B (299W) |
|---------:|----------:|-------------------:|
|        1 |       299 |                  0 |
|        2 |       301 |                 +2 |
|        4 |       299 |                  0 |
|        8 |       299 |                  0 |
|       16 |       302 |                 +3 |
|       32 |       605 | **+306 (CLIFF)** |
|       64 |       623 |               +324 |
|      128 |       621 |               +322 |

**16→32 cliff: +303W in one step** (matches near-full-random penalty).

## Fine N-vary cliff search (modes 2767..2798)

| N_unique | Power (W) |
|---------:|----------:|
|       17 |       591 |
|       19 |       603 |
|       21 |       608 |
|       23 |       613 |
|       25 |       594 |
|       27 |       593 |
|       29 |       599 |
|       31 |       592 |
|       33 |       595 |
|       35 |       607 |
|       40 |       603 |
|       48 |       531 |

**Cliff is between N_unique=16 and N_unique=17.**
At N_unique=17, power = 591W (already most of way to full random).

## Sub-tile breaking test (mode 2900-2908)

Layout: 8 sub-tiles of 16 N values each.
K_break = number of sub-tiles using UNIQUE pattern (others share pattern 0).

| K_break | Unique sub-tiles | Power (W) |
|--------:|-----------------:|----------:|
|       0 | none (all 8 same) |       301 |
|       1 |                1 |       302 |
|       2 |                2 |       304 |
|       3 |                3 |       302 |
|       4 |                4 |       **342** (+40 jump) |
|       5 |                5 |       472 |
|       6 |                6 |       555 |
|       7 |                7 |       608 |
|       8 |                8 |       601 |

## Mechanistic interpretation

1. **N=16 sub-tile is the dedup granularity**: matches MMA_N atomic tile.
   The multiplier hardware processes B in 16-N-wide chunks.

2. **HW caches up to ~4 distinct sub-tile patterns** for "free":
   - K_break=0..3 → ≤4 unique patterns total (1 shared + 0..3 unique)
   - All fit in some content-addressable structure → no register-toggle cost
   - Beyond 4 unique patterns: graceful degradation, then full cost

3. **The "N-vary is FREE" finding from low-entropy val table was AN ARTIFACT**:
   Low-entropy val table only has 16 distinct entries (N_unique ≤ 16), so
   sub-tile dedup hides the cost. When N_unique > 16 with high entropy, cost
   appears.

## Implications for prior K-vary findings

The "K-vary is the dominant cost" model needs revision. K-vary's cost
likely came from BREAKING THE PER-CYCLE DEDUP across K iterations within
a sub-tile (the SMEM B values changed across K rows of the operand).

True picture: power cost is driven by **per-cycle SUB-TILE PATTERN DIVERSITY**,
where:
- Sub-tile = 16 N values (likely the MMA atomic N tile)
- Up to ~4 distinct patterns can be cached/gated for free
- Beyond → power scales nonlinearly toward random baseline

## Next tests needed

1. **Verify position invariance**: alternate vs clustered unique sub-tiles
2. **Cross-precision**: does FP8 (K=32) have N=16 sub-tile too? Likely N=8
   for FP8 since N tile size differs.
3. **NVFP4**: N=16 packed in 8 FP4 per word (2 bytes wide).
4. **Combined K-cycle and sub-tile interaction**: K-vary HIGH ENTROPY +
   varying sub-tile uniformity.
5. **Discover the 4-slot "cache"**: does it scale with M? With cluster?
