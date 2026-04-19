# Cross-Precision Sub-Tile Dedup: 32-Byte Boundary CONFIRMED

Date: 2026-04-19. Tests across BF16, FP8 e4m3, NVFP4 confirm a UNIVERSAL
32-byte sub-tile granularity for B operand dedup.

@ -lgc 1005 MHz, 50M iters, 148 SMs, GPU 0. Plateau power median.

## N-vary HIGH-ENTROPY cliff search

| N_unique | BF16 (W) | FP8 (W) | NVFP4 (W) |
|---------:|---------:|--------:|----------:|
|        1 |      299 |     306 |       281 |
|        2 |      301 |     305 |       282 |
|        4 |      299 |     305 |       281 |
|        8 |      299 |     306 |       280 |
|       16 |      302 |     306 |       281 |
|       17 |  **591** |     -   |        -  |
|       32 |      605 |     309 |       281 |
|       33 |     595  | **642** |        -  |
|       64 |      623 |     599 |       285 |
|       65 |        - |     -   |   **463** |
|      128 |      621 |     657 |       457 |

| Precision | Cliff at N_unique | Bytes per sub-tile |
|-----------|-------------------|--------------------|
| BF16      | 16 → 17           | 16 × 2 = 32        |
| FP8 e4m3  | 32 → 33           | 32 × 1 = 32        |
| NVFP4     | 64 → 65           | 64 × 0.5 = 32      |

**ALL THREE precisions show the cliff at exactly 32 bytes of B sub-tile.**

## Sub-tile breaking test (K_break unique vs 8-K_break shared)

| K_break | BF16 (W) | FP8 (W) | NVFP4 (W) |
|--------:|---------:|--------:|----------:|
|       0 |      301 |     305 |       280 |
|       1 |      302 |     408 |       339 |
|       2 |      304 |     491 |       386 |
|       3 |      302 |     535 |       425 |
|       4 |      342 |     564 |       459 |
|       5 |      472 |     606 |       459 |
|       6 |      555 |     636 |       460 |
|       7 |      608 |     640 |       455 |
|       8 |      601 |     642 |       454 |

Sub-tile breaking shape differs per precision:
- **BF16**: 0-3 broken = FREE (4-slot dedup cache?); 4+ ramps to full
- **FP8**: GRADUAL ramp from start; ~80W per first broken sub-tile
- **NVFP4**: LINEAR ramp 0..3; saturates at K_break≥4 (~459W)

## Mechanism (BEST CURRENT MODEL)

1. **Sub-tile = 32 bytes of B operand** (= 16 BF16 = 32 FP8 = 64 NVFP4 N values)
2. **HW caches up to ~4 distinct sub-tile patterns** for "free" (BF16 only?)
   - Beyond 4 distinct patterns: spillover, full power
3. **Dedup is BYTE-PATTERN based** at 32-byte chunks, not value-based
4. **Position matters in BF16** (sticky activation): same-pattern sub-tiles at
   LOW N indices save more than mirror layout
5. **NVFP4 lower max power** (459W vs BF16 609W vs FP8 642W) — likely due to
   smaller value bit-width = less bit-toggling per multiply

## Cross-precision random baselines (recall)

- BF16 random: 609 W  (peak data dependence)
- FP8 random:  642 W  (highest! more multiplier energy per byte)
- NVFP4 random: 463 W  (lowest — fewer mantissa bits per multiply)

## Software optimization implications

1. **Pre-sort B columns** so identical 32-byte sub-tiles cluster at LOW N
2. **Prefer NVFP4 for power-constrained workloads** (random NVFP4 = BF16 const)
3. **For BF16, group columns by 16 N at a time** so up to 4 unique groups
   share (saves ~310 W vs unsorted)
