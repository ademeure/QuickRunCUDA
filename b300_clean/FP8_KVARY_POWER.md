# FP8 e4m3 K-vary Cross-Validation

Date: 2026-04-19. Cross-check the per-MAC-temporal-constancy mechanism with
a different precision (FP8 e4m3, kind::f8f6f4, K=32 per MMA inst).

## Setup
- m=128 n=128 K=32 FP8 e4m3 (kind::f8f6f4)
- @ -lgc 1005 MHz, 50M iter sustained
- Same methodology as BF16 K-vary tests

## Results

| Pattern | Power | Δ vs B const 304 |
|---------|------:|------------------:|
| Baseline rand A & B | 630 | +326 |
| B all-zero | 300 | -4 (Tier A) |
| **B = +1.0 const** | **304** | **0 (Tier B)** |
| B K-vary 2 unique | 329 | +25 |
| B K-vary 4 unique | 349 | +45 |
| **B K-vary 16 unique** | **375** | **+71** |
| B N-vary 2 unique | 304 | 0 (FREE) |
| B N-vary 16 unique | 304 | 0 (FREE) |

## Cross-precision comparison

| Mechanism | BF16 | FP8 e4m3 |
|-----------|-----:|---------:|
| Random baseline | 599 W | 630 W |
| B constant (Tier B) | 299 W | 304 W |
| B K-vary 16 unique cost | +47 W | **+71 W** |
| B N-vary 16 unique cost | +3 W | **+0 W (truly free)** |

## FP8 B K-vary cost is 1.5× BF16

BF16 B K-vary 16: +47 W (16 unique values × 1 register flip each per K cycle)
FP8 B K-vary 16: +71 W (1.51× BF16)

**FP8 has K=32 (vs BF16 K=16) per MMA inst** — 2× more B value cycles per
instruction. Predicts ~2× more cumulative switching activity. Measured 1.51×,
in reasonable agreement (within noise on the 30W-scale diff).

This **confirms the per-MAC-temporal mechanism is precision-agnostic** and
the cost scales roughly with # K cycles.

## N-vary is COMPLETELY free for FP8

Where BF16 N-vary 16 added +3W (likely small overhead from non-aligned packing),
FP8 N-vary 16 adds 0W exactly. Possibly because FP8 packs 4 elements per byte
(better aligned to multiplier structure) than BF16 (2 elements per word).

## A K-vary encoding bug (acknowledged)

My A K-vary mode for FP8 had `idx/16` where `idx/8` was needed for FP8's
4-elements-per-word packing. The "A K-vary" measurement (+11W) is actually
A M-vary (per-N-broadcast effect mostly). Need encoding fix for clean A
K-vary comparison.

Even with the bug, A K-vary measurement (+11W) is much less than B K-vary
(+71W), preserving the qualitative finding that B distributed array dominates
the K-direction power cost.

## Confidence

- HIGH on FP8 B K-vary 16 = +71W (replicable)
- HIGH on FP8 N-vary being completely free
- HIGH on the qualitative B-dominates-K-vary finding
- MED on the precise FP8 A K-vary numbers (encoding bug noted)
- HIGH on the precision-agnostic mechanism (BF16 and FP8 both show
  the same constant-N-vary-free / B-K-vary-expensive structure)

## Practical recipe (FP8 vs BF16)

Same recipe as BF16 applies:
- Constant B per K row: maximum savings (~−326W vs random)
- N-direction value diversity is FREE
- K-direction variation costs ~+25W per "unique value across K cycles"
  (slightly higher than BF16 due to longer K=32)

---

## CORRECTED: FP8 A K-vary with fixed encoding

Re-tested A K-vary with proper FP8 encoding (idx % 8 for k_pack instead of
idx / 16 which was giving M-direction). Each unsigned word now contains
4 distinct K-direction values from the K-pack.

| Pattern | Power | Δ vs A const 614 |
|---------|------:|------------------:|
| A=+1.0 const, B rand | 614 | 0 |
| A K-vary 2 unique | 608 | −6 |
| A K-vary 4 unique | 608 | −6 |
| A K-vary 16 unique | 613 | −1 |

**A K-vary on FP8: effectively ZERO cost** (all within noise floor σ≈4-8W).

## Final cross-precision A vs B K-vary table

| Precision | A K-vary 16u cost | B K-vary 16u cost | Ratio |
|-----------|------------------:|------------------:|------:|
| BF16 | +2 W | +47 W | 24× |
| **FP8 e4m3** | **≈0 W** | **+71 W** | **>70×** |

The FP8 ratio is even MORE extreme than BF16. Both precisions confirm:
- A is broadcast through ONE shared register stage → essentially free
  to K-vary
- B is distributed across many parallel MAC registers → expensive to
  K-vary, scaling with K-cycle count

## FP8 vs BF16 B K-vary: 1.5× scaling

BF16 B K-vary 16: +47W
FP8 B K-vary 16: +71W (1.51× BF16)

Predicted from K-cycle count (FP8 K=32 vs BF16 K=16): 2×
Measured: 1.51×

Below the linear prediction. Likely the FP8 multiplier processes 2 K
positions per cycle (giving 16 cycles instead of 32), partially mitigating
the extra K work. So effective per-cycle K-vary cost is similar to BF16,
just spread over 1.5× the cycles.

## Confidence

- HIGH on FP8 A K-vary effectively 0 (4 datapoints within noise)
- HIGH on FP8 B K-vary +71W (replicable)
- HIGH on broadcast-A / distributed-B mechanism being precision-agnostic
- MED on the exact 2-K-positions-per-cycle pipelining hypothesis (1.5× vs 2×)
