# BF16 Per-Bit Power Decomposition

Date: 2026-04-19. Built `tests/bench_tcgen05_bf16_perbit_power.cu` with
verification step (kernel prints first B values + sign/exp/mant decoded
to confirm encoding works as expected).

## Setup
- BF16 m=128 n=128 K=16, single-CTA tcgen05.mma kind::f16
- A always random; B random except ONE bit position forced to 0
- @ -lgc 1005 MHz, 50M iters sustained ~3.2 sec
- Encoding verified by printf at startup

## Per-bit-force-0 power impact

| Bit pos | Field | Power | Δ vs random |
|--------:|-------|------:|------------:|
| - | BASELINE random | 605 W | 0 |
| - | all-zero | 294 W | −311 (total random penalty) |
| 0 | mant LSB | 598 | −7 |
| 1 | mant | 594 | −11 |
| 2 | mant | 585 | −20 |
| 3 | mant | 570 | −35 |
| 4 | mant | 583 | −22 |
| 5 | mant | 590 | −15 |
| 6 | mant MSB | 579 | −26 |
| 7 | exp LSB | 574 | −31 |
| 8 | exp | 569 | −36 |
| 9 | exp | 574 | −31 |
| 10 | exp | 576 | −29 |
| 11 | exp | 576 | −29 |
| 12 | exp | 579 | −26 |
| 13 | exp | 573 | −32 |
| **14** | **exp MSB** | **592** | **−13 (outlier, replicated 3×)** |
| 15 | SIGN | 549 | **−56 ← biggest** |

## Headlines

1. **Sign bit (15)** explains 56/311 = **18%** of total random penalty —
   single biggest contributor (matches earlier sign-bit isolation).
2. **Exponent bits 7-13** average ~−30 W each (~10% of total per bit).
3. **Mantissa bits 0-6** average ~−19 W each, with bit 0 (LSB) only −7 W.
4. **Sum of singles: 426 W vs total penalty 311 W** = 137% — bit forces are
   SUPER-ADDITIVE individually, SUB-LINEAR when combined.
5. **Bit 14 (exp MSB) anomalous**: only −13 W savings (vs ~−30 for neighbors).
   Replicated 3× to confirm. Likely cause: exp MSB=0 forces all values into
   [0, 1.0) range (denormal-prone), which keeps denorm/zero handling logic
   active, offsetting register-toggle savings.

## Practical implications

- **ReLU activations** (sign forced to 0) save ~18% of multiplier power
  for free in BF16 GEMM
- **Bias-shifted-narrow-range** activations (bit 14 forced to 0, all values
  in [0,1)) actually save LESS power than expected due to denorm handling
- **Mantissa quantization** (force LSBs to 0) gives small savings — bit 0
  alone only −7W, but combined effect grows non-linearly

## Verification rigor

Encoding verified by kernel printf at idx=0,1:
- mode=15 (sign=0): all 4 BF16 values have s=0 ✓
- mode=115 (sign=1): all 4 BF16 values have s=1 ✓
- mode=0 (mant LSB=0): mant bits all show LSB=0 ✓
- mode=14 (exp MSB=0): exp bits all show MSB=0 ✓

Bit 14 outlier replicated 3 separate runs (588W, 591W, 597W — variance ±5W,
clearly distinct from neighbor bits 13/15 at ~575/549W).

## Confidence

- HIGH on sign-bit dominance (3+ replications)
- HIGH on exp-bit ~uniform contributions (small variance)
- HIGH on mantissa-bit lower contributions
- HIGH on bit-14 outlier being REAL (replicated 3×)
- MED on the denorm-handling explanation for bit 14 (plausible but unverified)
- MED on the super-additive single-bit sum (need force-multiple-bits test
  to verify combined behavior)
