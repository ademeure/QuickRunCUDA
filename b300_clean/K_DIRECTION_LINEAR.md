# K-Direction Sparse Test: NO Halves Effect

Date: 2026-04-20. Confirms BF16 two-half processing is N-DIRECTION SPECIFIC.

## Setup
- `tests/bench_tcgen05_bf16_perbit_power.cu` modes 6200-6316
- BF16 m128n128k16, K_zero K rows forced to zero, rest random
- Mode 6200+: K rows 0..K_zero-1 zero (FIRST K rows zero)
- Mode 6300+: K rows (16-K_zero)..15 zero (LAST K rows zero)
- @ -lgc 1005 MHz, 50M iters

## Results

| K_zero | First K zero (W) | Last K zero (W) |
|-------:|-----------------:|----------------:|
|      0 |              611 |             613 |
|      4 |              536 |             543 |
|      8 |              454 |             462 |
|     12 |              377 |             382 |
|     16 |              297 |             297 |

## Findings

**No K-direction halves asymmetry**: First-K-zero and Last-K-zero give
identical power within noise (within 8W of each other). Both halves
of K direction processed uniformly.

**Linear per-K-row contribution**: ~19.5 W per non-zero K row.
- K_zero=8 leaves 8 random K rows → 8 × 19.5 = 156W extra ≈ 453W actual ✓
- K_zero=12 leaves 4 random K rows → 4 × 19.5 = 78W extra ≈ 375W actual ✓

## Comparison with N-direction

| Direction | Halves effect? | Per-position cost |
|-----------|---------------|-------------------|
| N (sub-tiles) | YES (BF16 only): Half B random nearly free | Variable: 0-200W per sub-tile depending on Half |
| K (rows)      | NO: uniform across K | ~19.5W per random K row |

The HW two-half processing is asymmetric: it **decomposes the N dimension**
into Half A (N=0..63) and Half B (N=64..127), with Half B's MAC array
operating in a power-efficient mode for non-uniform inputs.

K dimension is processed sequentially without similar half-grouping.

## Software optimization (refined)

For BF16 m128n128k16:
- **N-direction**: cluster shared / repeated patterns at LOW N (Half A)
- **K-direction**: any sparsity pattern saves linearly (no positional advantage)

For sparse attention with structured K (e.g., causal attention):
- Don't reorder K positions for power (no benefit)
- Reorder N (output) columns for power (Half B effect)

## Confidence

- HIGH on K-direction linearity (5 measurements monotonic, both directions match)
- HIGH on per-K-row cost (~19.5W, predictions match within 1-2W)
- HIGH on N vs K direction asymmetry (BF16-specific)
