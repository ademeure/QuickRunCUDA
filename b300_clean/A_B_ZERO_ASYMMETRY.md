# A vs B Zero Gating: Asymmetric Behavior

Date: 2026-04-20. Tests how A=0 vs B=0 affect power, comparing to random
to determine multiplier zero-detection behavior.

## Setup
- BF16 m128n128k16, @ -lgc 1005 MHz, 50M iters
- Mode 200: A rand, B rand
- Mode 300: A rand, B zero
- Mode 1799 (NEW): A zero, B rand

## Results

| Configuration | Power (W) | Δ vs random | Reduction |
|---------------|----------:|------------:|----------:|
| A rand + B rand | 611 | 0 | 0% |
| A rand + B zero | 298 | -313 | -51% (full save) |
| A zero + B rand | 490 | -121 | -20% (partial save) |

**B=0 fully gates the multiplier (313W save)**
**A=0 only partially gates (121W save)**

## Mechanism

When B=0:
- Per-MAC product is 0 for every cycle
- Multiplier datapath fully gated → power drops to Tier B baseline
- Accumulator updates are "0+C=C" → minimal toggle

When A=0:
- Per-MAC product is 0 for every cycle (same math result)
- BUT multiplier still receives B values per cycle
- B-side input pipeline still toggles (B varying)
- Only ~half the multiplier switching activity is saved

This confirms the **A is broadcast, B is distributed** model:
- B-side has per-cycle distinct values driven through MAC array
- A-side has broadcast value driven through fanout to all MACs
- B varies → multiplier inputs toggle (even if A=0 zeros the result)
- A varies → broadcast wires toggle (relatively cheap)

## A varying cost is CONDITIONAL on B varying

| A | B | Power (W) | A-vary marginal cost |
|---|---|----------:|---------------------:|
| const | const | 299 | 0 (baseline) |
| rand | const | 297-302 | ~0 (free with B const) |
| const | rand | 549 | +250 (B vary cost) |
| rand | rand | 611 | +61 (extra A cost when B varies) |
| zero | rand | 490 | -59 (A=0 saves vs A const +1.0) |

When B varies, A varying adds ~60W. When B const, A varying free.

## Refined model

```
P = baseline + B_vary_cost + (A_vary AND B_vary) * A_marginal
where:
  baseline ≈ 299W (Tier B)
  B_vary_cost ≈ 250W when B fully random
  A_marginal ≈ 60W only when B already varies
```

## Software optimization (refined again)

- **B is the dominant power knob**: 250W if random
- **A varying matters ~5x less but only when B also varies**: 60W marginal
- **A=0 trick**: 60W save over A=const when B random
- **B=0 trick**: 313W save (full gating)

For inference where activation A varies per token but weight B is fixed:
- A varying: free (with const B)
- Combined with structured B: full optimization stack

## Confidence

- HIGH on A=0 vs B=0 asymmetry (3 measurements clean)
- HIGH on A varying being conditional (cross-checked with mode 4007 = 297W)
- MEDIUM on the exact 60W A_marginal (only 1 measurement direction)
- LOW on whether this generalizes to FP8/NVFP4 (untested)
