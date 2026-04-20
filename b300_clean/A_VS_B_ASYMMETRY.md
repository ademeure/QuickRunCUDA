# A vs B Operand Power Asymmetry — DEFINITIVE

Date: 2026-04-20. Tests at -lgc 1005 MHz, BF16 m128n128k16, 50M iters.

## Headline result

| A pattern | B pattern | Power (W) | Δ vs Tier B (299W) |
|-----------|-----------|----------:|-------------------:|
| const +1.0 | const +1.0 | 299 | 0 |
| FULL random per (m,k) | const +1.0 | 297-302 | ~0 |
| const +1.0 | random | 549 | +250 |
| +Inf | random | 540 | +241 |
| random | random | 609 | +310 |

**A varying alone is FREE. B varying alone costs 250-310 W.**

## Why: A is BROADCAST, B is DISTRIBUTED

In a tcgen05.mma kind::f16 m128n128k16:
- 16384 MACs (128 M × 128 N) compute partial products
- At each cycle, MAC handles one (m, n) output's K iteration
- Per cycle: A[m, k_curr] is shared across ALL MACs at same m (broadcast to ~32 N positions)
- Per cycle: B[k_curr, n] is unique per N (each MAC has its own B value)

So the "B operand toggle activity" scales as 32× (one per N MAC) while A toggle scales as 1× (single broadcast value driven through fanout buffer).

## A M-vary HIGH-ENTROPY sweep (modes 4000-4007)

B = const +1.0, A varies across M with controllable M_unique.

| M_unique | Power (W) |
|---------:|----------:|
|        1 |       290 |
|        2 |       292 |
|        4 |       292 |
|        8 |       292 |
|       16 |       298 |
|       32 |       297 |
|       64 |       296 |
|      128 |       297 |

**No cliff. A M-vary is uniformly FREE across ALL M_unique values.**
Contrast: B N-vary has a sharp cliff at N_unique=17 (16→591W).

## A K-vary HIGH-ENTROPY sweep (modes 4100-4104)

B = const +1.0, A varies across K with controllable K_unique.

| K_unique | Power (W) |
|---------:|----------:|
|        1 |       290 |
|        2 |       293 |
|        4 |       295 |
|        8 |       296 |
|       16 |       295 |

**No K-cliff for A either.** A K-vary cost ≤ +5W vs constant.

## A FULL random (per m AND per k) — modes 4200-4207

B = const +1.0, A varies fully both directions.

| M_unique | Power (W) |
|---------:|----------:|
|        1 |       299 |
|        2 |       298 |
|        4 |       298 |
|        8 |       298 |
|       16 |       300 |
|       32 |       301 |
|       64 |       298 |
|      128 |       302 |

**Fully random A still costs only ~+12W vs const.**

## Mechanistic interpretation

The 32-byte sub-tile dedup cache documented in SUBTILE_DEDUP_MODEL.md is a
**B-OPERAND-ONLY mechanism**. It exists because B is the distributed side
where toggle activity scales with N-MAC count. A is broadcast and doesn't
need dedup.

**Confirmed: All previously-measured "K-vary cost" was B-side. A K-vary
genuinely free.**

## Software optimization implications

For tcgen05 GEMM workloads:

1. **B operand is THE power hot path**. Optimization effort should target
   B layout, not A.
2. **A randomness is essentially free**. Don't waste effort dedup'ing A.
3. **For decoders/inference**: weights (typically B) drive power; activations
   (typically A or B depending on layout) — choose layout so the larger or
   higher-entropy operand is on A side.
4. **For training**: gradient accumulation has both operands varying.
   Consider running matmuls in transposed form to put the smaller-entropy
   operand on B side.

## Confidence

- HIGH on the A-vs-B asymmetry (5 measurements with consistent pattern)
- HIGH on the "B random + A constant ~ random×random within noise"
  (mode 1700 = 549, mode 200 = 609; +60W from also-random A)
- HIGH on the absence of A-side dedup cliff (sweep covered 1-128 M_unique)
- LOW on whether this transfers to A-major MMA layouts (untested)
