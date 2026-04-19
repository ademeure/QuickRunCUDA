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

---

## Position-invariance test (modes 3000-3007): POSITION MATTERS

All have 4 unique sub-tiles + 4 shared, only placement varies:

| Mode | Mask | Sequence | Power (W) |
|-----:|:-----|:---------|----------:|
| 3000 | 0xAA | 0,1,0,3,0,5,0,7 | **614** (worst) |
| 3001 | 0x55 | 0,0,2,0,4,0,6,0 | 503 |
| 3002 | 0xF0 | 0,0,0,0,4,5,6,7 | **342** (best!) |
| 3003 | 0x0F | 0,1,2,3,0,0,0,0 | **611** (mirror of 3002, but full cost!) |
| 3004 | 0xCC | 0,0,2,3,0,0,6,7 | 542 |
| 3005 | 0x33 | 0,1,0,0,4,5,0,0 | 486 |
| 3006 | 0xE8 | 0,0,0,3,0,5,6,7 | 459 |
| 3007 | 0x17 | 0,1,2,0,4,0,0,0 | 557 |

### Key observation: 3002 vs 3003 mirror asymmetry

- **3002 (4 same FIRST, then 4 unique)**: 342 W (saves nearly all)
- **3003 (4 unique FIRST, then 4 same)**: 611 W (no savings!)

These have the SAME number of unique sub-tiles AND the SAME number
of adjacent-equal pairs (3 adj-eq each), but power differs by 269 W.

### Best-fit model (so far): "STICKY ACTIVATION"

The B operand port appears to start in a "low-power gated" state
and gets ACTIVATED upon encountering the first non-shared sub-tile.
Once activated, it STAYS ACTIVATED — even if subsequent sub-tiles
revert to a previously-seen pattern.

Predictions vs actual:
- 3000 (activates at t=1, runs full to t=7): predict 8 active = 605 W; actual 614 W ✓
- 3001 (activates at t=2): predict 6 active = ~518 W; actual 503 W ✓
- 3002 (activates at t=4): predict 4 active = ~474 W; actual 342 W (better)
- 3003 (activates at t=1): predict 7 active = ~566 W; actual 611 W ✓
- 3005 (activates at t=1): predict 7 active = 566 W; actual 486 W (off)

Sticky activation model fits 4 of 8 well; 3002 and 3005 underestimate
power saving. Likely there's an additional secondary gating mechanism
when the MMA runs N-direction sub-tile sweeps in chunks.

Key implication: **the order of N values matters for power consumption**.
Software that wants to minimize tensor core power can pre-sort B
values such that the most-repeating sub-tile pattern appears at low
N indices.
