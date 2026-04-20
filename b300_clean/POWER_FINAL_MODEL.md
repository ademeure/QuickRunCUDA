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

---

## K-row dedup mechanism: PAIRWISE-CONSECUTIVE (mode 5200-5216)

Test: K rows arranged in CONSECUTIVE GROUPS of identical content vs ALTERNATING.

| K_unique | Layout (16 K rows) | # transitions | Power (W) | vs free baseline (303W) |
|---------:|--------------------|--------------:|----------:|------------------------:|
|        1 | AAAAAAAAAAAAAAAA   |       0       |       306 | +3 |
|        2 | AAAAAAAA BBBBBBBB  |       1       |       314 | +11 |
|        4 | AAAA BBBB CCCC DDDD |       3       |       324 | +21 |
|        8 | AABB CCDD EEFF GGHH |       7       |       343 | +40 |
|       16 | ABCDEFGHIJKLMNOP   |      15       |       387 | +84 |

vs alternating (mode 5000-5004 K-rotating evenly):

| K_unique | Layout              | # transitions | Power (W) | Δ |
|---------:|---------------------|--------------:|----------:|---:|
|        2 | ABABABABABABABAB    |      15       |       380 | -7 vs all-different |
|        4 | ABCDABCDABCDABCD    |      15       |       384 | -3 |
|       16 | ABCDEFGHIJKLMNOP    |      15       |       387 | (same) |

**Cost is roughly +5W per K-row transition** (consecutive case).

Alternating ABABAB has 15 transitions → 380W ≈ 387W (= 15 × 5.6 + 303 = 387 ✓).

## REVISED model: K-row "active" count drives cost

```
P_per_MMA = P_baseline_const                              (~303 W BF16)
          + N_active_K_rows × P_per_active_K_row          (~5 W per row)
          + (N_unique_per_row > 16 ? N_K × N_subtile_cost : 0)
                                                          (~+19 W per K row × 16 rows = 305 W)
```

where `N_active_K_rows` = # of K rows differing from immediate predecessor.

## Software optimization update

For workloads where B varies along K (e.g., transformer attention with
position-dependent values):

1. **Group consecutive K rows by similarity**: if K rows can be reordered
   so consecutive rows have matching 32-byte sub-tile patterns, save
   ~5W × (# saved transitions). For K=16: max savings = 75W per CTA.

2. **Per-K-row sub-tile cache** still applies: keep N_unique per row ≤ 16
   to avoid the +305W cliff.

3. **Combined**: (consecutive K-row matching) × (sub-tile-friendly N pattern)
   → minimum power = ~303W (baseline only), even with all unique data.

---

## Cross-precision K-row consecutive grouping (modes 5200+ in each kernel)

**FP8** (K=32):
| K_unique | Group size | Power (W) | Δ vs free |
|---------:|-----------:|----------:|----------:|
|        1 |         32 |       306 | 0 |
|        2 |         16 |       316 | +10 |
|        4 |          8 |       328 | +22 |
|        8 |          4 |       348 | +42 |
|       16 |          2 |       389 | +83 |
|       32 |          1 |       417 | +111 |

**NVFP4** (K=64):
| K_unique | Group size | Power (W) | Δ vs free |
|---------:|-----------:|----------:|----------:|
|        1 |         64 |       286 | 0 |
|        2 |         32 |       292 | +6 |
|        4 |         16 |       299 | +13 |
|        8 |          8 |       311 | +25 |
|       16 |          4 |       332 | +46 |
|       32 |          2 |       347 | +61 |
|       64 |          1 |       396 | +110 |

## Per-K-row transition cost analysis

| Precision | Per-row cost (W) | K rows | Total K-vary (W) | Per-byte cost (W) |
|-----------|-----------------:|-------:|-----------------:|------------------:|
| BF16      |             5.25 |     16 |               84 |            0.0205 |
| FP8       |             3.50 |     32 |              111 |            0.0273 |
| NVFP4     |             1.72 |     64 |              110 |            0.0269 |

**Per-byte K-row cost ≈ 0.025 W/byte** (FP8 and NVFP4 agree, BF16 slightly less).
Total B operand volume is constant 4096 bytes across all 3 precisions →
total K-vary cost converges to ~110W.

## Global picture

```
Total power = baseline (~300W BF16, ~305W FP8, ~280W NVFP4)
            + K-vary cost (~85-110W when K rows fully randomized)
            + N sub-tile cliff (~250-310W when within-row N exceeds dedup)
```

Random data engages BOTH K-vary AND N-cliff costs:
- BF16 random:  300 + 85 + 290 - saturation = 609 W
- FP8 random:   305 + 111 + 320 - saturation = 642 W
- NVFP4 random: 280 + 110 + 180 - saturation = 463 W

Saturation factor ≈ 0.7-0.8 (multiplicative gating between K and N costs).

## Recipe to MINIMIZE power for arbitrary B (universal)

1. **Sort columns** so identical 32-byte sub-tiles cluster at low N AND
   so consecutive K rows have matching content where possible.
2. **Quantize B** so each K row has ≤ "32 bytes worth" unique values
   (16 BF16, 32 FP8, 64 NVFP4).
3. Aim for ~baseline power even with full-bit-entropy operand (only
   K-toggle cost remains, ~85W BF16 max).

This applies to GEMMs, attention, and any tcgen05.mma usage. Real ML
workloads with structured B (e.g. quantized weights, top-K sparsity)
naturally satisfy these constraints.
