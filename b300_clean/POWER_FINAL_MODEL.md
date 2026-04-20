# tcgen05.mma Power: Final Unified Model

Date: 2026-04-20. After extensive investigation across BF16/FP8/NVFP4 with
varying patterns on A and B, K-row × N-sub-tile interactions.

## The Model (in 4 components)

```
P_per_MMA = P_static_const          (Tier B: ~299 W BF16, ~305 W FP8, ~280 W NVFP4)
          + P_K_toggle               (max +50-85 W when K varies fully)
          + P_N_subtile_cliff        (jumps +250-310 W if per-row N pattern
                                       exceeds 32-byte cache)
          + P_per_byte_entropy       (mostly captured in randomness already)
```

These costs are **multiplicatively gated** rather than purely additive
(see worked examples below).

## Component 1: Static const baseline

| Precision | Tier B power (W) |
|-----------|-----------------:|
| BF16      |              299 |
| FP8 e4m3  |              305 |
| NVFP4     |              280 |

## Component 2: K-toggle cost (per-cycle multiplier switching)

K_unique = number of distinct B values in K direction (per N column).
Costs ramp from 0 to max as K_unique → K (full).

| Precision | K-vary low-ent max (W) | K-vary high-ent max (W) |
|-----------|----------------------:|------------------------:|
| BF16 (K=16) |  345 (+46) |  387 (+88) |
| FP8  (K=32) |  376 (+71) |  ~440 (+135) |
| NVFP4 (K=64) | 379 (+99) | est ~470 (+190) |

## Component 3: N sub-tile cliff cost

Per-K-row, the 32-byte sub-tile dedup cache holds ONE pattern. If sub-tiles
within a K row are identical → free. If diverse → full activation per
sub-tile.

| Precision | N_unique cliff | Sub-tile bytes | Cliff cost (W) |
|-----------|---------------:|---------------:|---------------:|
| BF16      | 16 → 17        |             32 |       +290     |
| FP8       | 32 → 33        |             32 |       +320     |
| NVFP4     | 64 → 65        |             32 |       +180     |

## Cache scope: PER-K-ROW (not per-MMA)

K-rotating test (mode 5000-5004): each K row has its own N pattern.

| K_unique | N_unique/row | Power (W) | Interpretation |
|---------:|-------------:|----------:|----------------|
|        1 |           16 |       303 | All K rows same → free |
|        2 |           16 |       380 | K alternates 2 patterns → K-toggle |
|        4 |           16 |       384 | K cycles 4 patterns |
|       16 |           16 |       387 | Every K row different |
|        1 |           32 |       607 | N already over cliff → full |
|       16 |           32 |       609 | Same |

**Insight: when within-row N_unique ≤ 16 (sub-tile cache hits), only K-toggle
applies. When N_unique > 16 (cliff), full power regardless of K structure.**

## A operand: completely free (broadcast)

| A pattern | B = const | Power (W) |
|-----------|-----------|----------:|
| const | const | 299 |
| FULL random per (m, k) | const | 297-302 |

A varies: ZERO cost. The sub-tile dedup mechanism applies ONLY to B side
(distributed operand). A is broadcast across N MACs → single value driven
through fanout buffer regardless of M variation.

## Worked examples

**Example 1: Quantized BF16 with N=16 group scaling**
Workload reads B with per-16-N-group scale factor. Within each 16-N group,
values share scale → highly correlated bytes. If scale is applied to fit
N_unique ≤ 16 per K row: P = const baseline + K-toggle = ~387 W.

**Example 2: Fully random GEMM**
P = const + K-toggle + N-cliff = 299 + 88 + 290 = 677 W (predicted).
Actual measured: 609 W (35% below additive prediction). Multiplicative
saturation.

**Example 3: ReLU-activated B**
Sign-bit forced 0 → -56 W from random baseline (=553 W). 18% savings due
to multiplier exponent NOT toggling sign side.

## Software optimization recipes

1. **Sort B columns to cluster identical 32-byte sub-tiles at LOW N**
   - "Sticky activation" model: HW activates at first non-matching sub-tile,
     stays active. Save up to 305 W per CTA.

2. **Pre-quantize B to fit ≤ 16 unique values per N=16 group (BF16)**
   - Sub-tile cache stays warm → free N variation.

3. **Put high-entropy operand on A side, low-entropy on B**
   - A randomness is FREE. B randomness costs full +250-310 W.

4. **For inference (decoder):**
   - Activations (high entropy, varies per request) → A operand
   - Weights (static, can be quantized for sub-tile dedup) → B operand
   - This is opposite of training where gradients vary fully on both sides.

## Confidence

- HIGH on 32-byte sub-tile cliff (cross-precision, multiple hashes, clean replication)
- HIGH on A vs B asymmetry (5+ measurements consistent)
- HIGH on cache scope = per-K-row (K-rotating test confirms)
- HIGH on sticky activation (position invariance test)
- MEDIUM on exact saturation function (multiplicative not additive)
- LOW on whether 2-CTA (cluster_group::2) shares cache
