# Sub-Tile Dedup Mechanism: Final Model

Date: 2026-04-19. Cross-precision pattern-count tests pin down the actual
hardware dedup behavior.

## Key insight: HW SUB-TILE = 32 BYTES UNIVERSALLY

Confirmed across BF16/FP8/NVFP4 by N-vary cliff at exactly 32 bytes. Each
precision has different N count per HW sub-tile:
- BF16: HW sub-tile = 16 N (= 32 bytes)
- FP8:  HW sub-tile = 32 N (= 32 bytes)
- NVFP4: HW sub-tile = 64 N (= 32 bytes)

For N=128 total: BF16 has 8 HW sub-tiles, FP8 has 4, NVFP4 has 2.

## Pattern count anomaly RESOLVED via HW granularity

Pattern-count test rotates `sub_tile_id % N_distinct` across MY 8
"pseudo-sub-tiles" (16 N each). At HW granularity:

| N_distinct | BF16 (8 HW sub) | FP8 (4 HW sub from my 2-pair groups) | NVFP4 (2 HW sub from my 4-pair groups) |
|-----------:|----------------:|------:|------:|
|          1 | (0,...,0)       | (0,0,0,0): 1 distinct | (0,0): 1 distinct |
|          2 | (0,1,0,1,...): 2 distinct | (0,1)(0,1)(0,1)(0,1): 1 dist | (0,1,0,1)(0,1,0,1): 1 dist |
|          3 | (0,1,2,0,1,2,0,1): 7 distinct | (0,1)(2,0)(1,2)(0,1): 3 dist | (0,1,2,0)(1,2,0,1): 2 dist |
|          4 | (0,1,2,3,0,1,2,3): 7 distinct | (0,1)(2,3)(0,1)(2,3): 2 dist | (0,1,2,3)(0,1,2,3): 1 dist |

Power vs HW-distinct count:

| HW distinct | BF16 (W) | FP8 (W) | NVFP4 (W) |
|------------:|---------:|--------:|----------:|
|           1 |      301 |     308 |       281 |
|           2 |        - |     630 |       284 |
|           3 |        - |     574 |       470 |

**Confirmed: HW dedup cache = 1 slot for FP8 / BF16; 2 slots for NVFP4** (or
NVFP4 just has wider single slot covering 64 bytes worth).

NVFP4 N_distinct=4 = 284W (FREE) because at HW level only 1 distinct pattern!

## BF16 special: even with 1 distinct HW pattern, alternation costs

BF16 N_distinct=2 (alternation): EVERY sub-tile differs from immediate prior
→ STICKY ACTIVATION fires at sub-tile 1, full power thereafter (622W).

BF16 K_break=1 (0,0,0,0,0,0,0,7): only ONE transition at sub-tile 7
→ activation comes too late to matter (302W ≈ free).

BF16 K_break=4 (0,0,0,0,4,5,6,7): activation at sub-tile 4 → middle ground (342W).

## FP8 sub-tile breaking re-interpretation

My K_break tests used 8 pseudo-sub-tiles (16 N each = 16 bytes), but FP8
HW sub-tile is 32 N (32 bytes = 2 of mine). So FP8 K_break=1 actually
breaks ONE byte-pair within HW sub-tile 3 → 0.5 sub-tile broken at HW.

| FP8 K_break | My sub-tile change | HW sub-tile pattern | Power (W) |
|------------:|--------------------|--------------------:|----------:|
|           0 | none               | all same            |       305 |
|           1 | last 1 broken      | HW3 partial broken  |       408 |
|           2 | last 2 broken      | HW3 fully broken    |       491 |
|           4 | last 4 broken      | HW2,HW3 broken      |       564 |
|           8 | all 8 broken       | all HW broken       |       642 |

The smooth ramp matches: each HW sub-tile broken adds ~80W cost.

## Final unified model

```
PER MMA POWER = P_static_base
              + Σ over HW-sub-tiles of:
                  if HW sub-tile == previous HW sub-tile: 0
                  else if HW sub-tile fits in 1-slot cache: small cost
                  else: full activation cost (~40W per sub-tile for BF16)
              + Σ over K iterations: per-cycle bit-toggle cost
                  (this is the K-vary ~50W cost we saw earlier)
```

## Software optimization implications (REVISED)

For tcgen05 GEMMs:

1. **Pre-sort B columns** so byte-identical 32-byte chunks cluster contiguously
   along N axis. Saves ~250-310 W per CTA at random baseline.

2. **Quantize columns to 32-byte granularity**: many ML workloads have
   structured B (e.g. weight quantization with per-group scales). If a group
   spans EXACTLY 32 bytes of B at the consumer's tile shape, dedup wins.

3. **Don't alternate B values at fine granularity**: an attention pattern
   that puts hot/cold tokens in alternating columns will pay full power.
   Cluster hot tokens contiguously instead.

4. **For NVFP4, period-4 distinct patterns at granularity 16-N is FREE**:
   maps to 1 distinct HW sub-tile.

## Confidence and limits

- HIGH on the "32-byte HW sub-tile = universal" finding (3 precisions agree)
- HIGH on the 1-slot cache for BF16/FP8, 2-slot equivalent for NVFP4
- MEDIUM on the exact model for partial activation (some unexplained gradients)
- LOW on whether this generalizes to cluster_group::2 / multi-CTA MMA
  (not yet tested)

## Open follow-ups

- Test A operand (broadcast structure differs)
- Test cluster_group::2 — does CTA pair share dedup cache?
- Cross-check with NCU `tpc__l1tex_*` or `sm__pipe_tensor_*` counters
- Verify with explicitly hand-written 64-byte patterns (rule out hash artifacts further)
