# Sub-Tile Position Halves: Major HW Discovery

Date: 2026-04-20. Single-unique-position test (mode 3020-3027) reveals
the sub-tile dedup mechanism processes B in TWO HALVES with different behaviors.

## Setup
- BF16 m128n128k16, 8 sub-tiles per K row, K constant per row
- Mode 3020+P: sub_tile P is unique (pattern_id P+1), all others use pattern 0
- @ -lgc 1005 MHz, 50M iters

## Results

| Unique pos P | Sub-tile sequence | Power (W) | Δ vs free (300W) |
|-------------:|-------------------|----------:|-----------------:|
|            0 | A B B B B B B B   |       426 |             +126 |
|            1 | B A B B B B B B   |       469 |             +169 |
|            2 | B B A B B B B B   |       461 |             +161 |
|            3 | B B B A B B B B   |       466 |             +166 |
|            4 | B B B B A B B B   |   **349** |              +49 |
|            5 | B B B B B A B B   |       304 |              +4  |
|            6 | B B B B B B A B   |       305 |              +5  |
|            7 | B B B B B B B A   |       306 |              +6  |

## SHARP transition between position 3 and position 5

Unique sub-tile in **first half** (pos 0-3): costs +126 to +169 W
Unique sub-tile at **position 4** (boundary): +49 W
Unique sub-tile in **second half** (pos 5-7): essentially free (+4 to +6 W)

## Hypothesis: Two-half processing

The B operand is processed in TWO HALVES:
- Half A: sub_tiles 0-3 (low N positions = N 0..63)
- Half B: sub_tiles 4-7 (high N positions = N 64..127)

Each half has its own dedup state. The behavior differs:
- Half A: activates AND stays sticky on any non-matching sub-tile
- Half B: dedup is MORE aggressive — single non-matching sub-tile barely costs

This is the BEST EXPLANATION for the observed asymmetry. Possible reason:
- Half A's "active sub-tiles" cost full pipeline activation
- Half B reuses Half A's pipeline state, only marginal transitions cost

## Implications for sticky activation

Earlier (mode 3000-3007) we observed sticky activation. The position-test
refines this:
- Sticky applies WITHIN a half
- Crossing the half boundary doesn't transfer sticky activation
- Half B is inherently CHEAPER to break than Half A

## Software optimization (REFINED)

The recipes from prior model:
1. **Cluster shared sub-tiles at LOW N** (Half A) — prevents Half A activation
2. **Put unique sub-tiles in HIGH N (Half B)** — minimal cost in Half B
3. **AVOID** isolated unique sub-tiles in Half A — costs +150W each

Quantitative example for BF16 m128n128:
- 4 unique sub-tiles clustered at positions 4-7: 342W (mode 2904)
- 4 unique at 0-3 (mirror): 611W (mode 3003)
- Difference: 269W save by putting uniques in second half

## Confidence

- HIGH on the position 3-5 cliff being real (8 measurements, monotonic trend)
- HIGH on the half-boundary at sub_tile 4 (= N=64 boundary)
- MEDIUM on the "two half processing" mechanism (best fit but not proven)
- LOW on extending this to FP8/NVFP4 (untested)

## Open follow-ups

- Test mode 3020+P for FP8 (FP8 has 8 sub-tiles too at my granularity)
- Test mode 3020+P with pos_unique pattern that matches half boundaries differently
- Verify with NCU counter (l1tex bank conflict counts may differ across halves)

---

## Cross-precision check: TWO-HALF is BF16-SPECIFIC

Repeated single-unique-position test for FP8 and NVFP4:

**FP8** (mode 3020-3027, my pseudo sub-tile = 16 N = HALF HW sub-tile):

| Position | Power (W) |
|---------:|----------:|
|        0 |       406 |
|        1 |       408 |
|        2 |       414 |
|        3 |       408 |
|        4 |       409 |
|        5 |       411 |
|        6 |       407 |
|        7 |       404 |

**FP8 HW position (mode 3030-3033, full 32-byte HW sub-tile unique)**:

| HW Pos | Power (W) |
|-------:|----------:|
|      0 |       501 |
|      1 |       490 |
|      2 |       501 |
|      3 |       493 |

**NVFP4** (mode 3020-3027):

| Position | Power (W) |
|---------:|----------:|
|        0 |       337 |
|        1 |       338 |
|        2 |       345 |
|        3 |       330 |
|        4 |       333 |
|        5 |       341 |
|        6 |       339 |
|        7 |       331 |

Both FP8 and NVFP4: **UNIFORM cost across positions**. No halves asymmetry.

## Updated conclusion: two-half is BF16-only

The two-half processing is ONLY observed in BF16 m128n128k16 (with K=16).
FP8 (K=32) and NVFP4 (K=64) treat all sub-tile positions equally.

Possible explanations:
1. BF16 m128n128 has a specific MAC array geometry (8 sub-tiles of N=16 with
   half-N processing)
2. FP8/NVFP4 have higher K → different pipeline depth → uniform processing
3. K-direction iteration count affects whether sub-tile pipeline can split

## Implications

The "Half B is free" optimization only works for BF16 m128n128. For other
precisions/shapes, the universal recipe still applies:
- 32-byte sub-tile dedup (within K row)
- Pairwise K-row dedup (across K)
- A operand free regardless of randomness

## Confidence

- HIGH on BF16 having clear two-half asymmetry (8 measurements monotonic)
- HIGH on FP8/NVFP4 NOT having halves asymmetry (uniform within ±10W)
- LOW on the explanation for why BF16 is special

---

## Cross-Half Mirror Test (mode 6700-6704): Half A is the dominant path

Test: Half B mirrors Half A's pattern. Does HW dedup Half B against Half A?

| K_unique (unique sub-tile pairs A↔B mirror) | Power (W) | Note |
|--------------------------------------------:|----------:|------|
| 0 (all baseline) | 305 | all 8 sub-tiles same pattern |
| 1 (sub_tile 3 + mirror 7 unique) | **465** | Half A activates! |
| 2 (sub_tiles 2,3 + mirror 6,7) | 543 | |
| 3 (sub_tiles 1,2,3 + mirror 5,6,7) | 619 | |
| 4 (all 4 unique + mirror) | 612 | full random |

vs K_break (only Half B unique):
| K_break | Power (W) |
|--------:|----------:|
| 0 | 305 |
| 1 | 304 (free!) |
| 4 | 347 |

## Key insight

Mirror=1 (unique at sub_tile 3 in Half A + mirror at 7) = 465W.
K_break=1 (unique at sub_tile 7 in Half B only) = 304W.

The DIFFERENCE: Mirror=1 has unique sub-tile in Half A (position 3).
K_break=1 doesn't.

**Even when Half B has MATCHING content to Half A's broken sub-tile,
no cross-half dedup occurs. The cost comes from the Half A activation alone.**

This refines the BF16 two-half model:
- **Half A is the "primary" / always-on multiplier path** — high power cost
  for any unique data
- **Half B is the "secondary" / gated path** — can be deduped to near-zero
  cost when patterns match a previous sub-tile WITHIN Half B
- **No cross-half dedup** — Half B can't reuse Half A's cached patterns

## Mechanism interpretation

The BF16 m128n128 multiplier likely has:
- ONE primary 64-N MAC array (for Half A) that's always active when used
- ONE secondary 64-N MAC array (for Half B) with aggressive power gating
- The two arrays operate INDEPENDENTLY with separate dedup state
- Half B's dedup cache is local to Half B; no shared state with Half A

## Practical implication (refined)

For BF16 m128n128 inference workloads:

1. **Pack high-entropy data into Half B (N=64..127)** - free because Half B
   gates aggressively
2. **Use Half A (N=0..63) for low-entropy / sparse / repeated patterns** -
   Half A pays full data-dep cost regardless of Half B's content
3. **Don't try cross-half mirroring** - doesn't help; HW doesn't dedup across halves

For example, for a row-major weight matrix:
- Sort columns so that columns 0..63 (Half A) are the most repetitive
- Columns 64..127 (Half B) can be arbitrary - they're mostly free
