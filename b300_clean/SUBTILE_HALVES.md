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
