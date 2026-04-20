# CRITICAL CORRECTION: cuBLAS Speedup is Primarily BIT ENTROPY

Date: 2026-04-20. After GPTQ analysis appeared to show "K-row similarity" as
the speedup cause, applied rule #9 (suspect the test) and discovered the
ACTUAL primary mechanism is bit entropy of B.

## The smoking gun control test

| Mode | Description | TFLOPS |
|------|-------------|-------:|
| FULL_RANDOM | Random sign + 7-bit mantissa per element | 1503 |
| **RANDOM_4BIT_no_K_sim** | 4 mantissa bits per element, K-rows UNIQUE | **1755** |
| GPTQ_4BIT_with_K_sim | 4 bits + K-row similar (per-group scale) | 1690 |

**Random 4-bit-only is FASTER than GPTQ** — proving bit entropy dominates,
not K-row similarity.

## B-side bit entropy sweep (mantissa bits random, A always full random)

| B bits random | TFLOPS | Speedup |
|--------------:|-------:|--------:|
| 0 (constant) | 2208 | 1.32× |
| 1 | 2016 | 1.21× |
| 2 | 1934 | 1.16× |
| 3 | 1865 | 1.12× |
| 4 | 1808 | 1.08× |
| 5 | 1755 | 1.05× |
| 6 | 1709 | 1.02× |
| 7 (mantissa fully random) | 1669 | 1.00× |

Each random bit added to B: ~70 TFLOPS reduction.

## What about K-row identity then?

Looking at all data points:
- B all constant (0 bits): 2208 TFLOPS
- B K-row identical (N varies, K fixed): 2104 TFLOPS  
- B random per element with K-row similarity: ~1690-1755 TFLOPS
- B fully random per element: 1503 TFLOPS

So K-row identity provides ~600 TFLOPS over fully random, BUT this is also
explained by the N-direction values being CONSTANT across K (which reduces
the effective bit entropy seen by cuBLAS internal MMA per K iteration).

## Refined mechanism explanation

The cuBLAS speedup correlates with **per-cycle bit entropy** at the multiplier:
- Lower per-cycle bit toggle → less multiplier power → less throttle → faster
- "K-row identity" is ONE WAY to reduce per-cycle entropy (each K cycle sees same N pattern)
- "Reducing mantissa bits" is ANOTHER WAY (fewer random bits per element)
- They are NOT separate mechanisms; both reduce per-cycle multiplier activity

## Implication for practical workloads

For INT4-quantized weights:
- Per-element bit entropy = 4 mantissa bits ≈ 4 random bits
- From entropy sweep: 4-bit random (no K-row sim) = 1808 TFLOPS = 1.20× speedup
- Add K-row similarity: 1690 TFLOPS (slightly LESS due to GPTQ-specific structure)
- **Realistic INT4 inference**: ~1.10-1.20× automatic speedup from low bit entropy

For INT8-quantized weights:
- Per-element bit entropy ≈ 7 mantissa bits ≈ random
- ~1.06× speedup minimal

## Updated headline (CORRECTED)

Modern INT4-quantized LLM inference on B300 gets ~1.10-1.20× automatic
speedup, primarily because **INT4 has low per-element bit entropy** rather
than K-row similarity per se. The HW dedup mechanism manifests as reduced
multiplier power for low-entropy operands.

For maximally efficient inference:
- **Use INT4** instead of INT8 (more bit savings)
- Pre-quantize weights such that per-element entropy is minimized
- Focus on B-side (weights) optimization

## Confidence

- HIGH on bit entropy being the primary mechanism (clean control test)
- HIGH on each-bit cost (~70 TFLOPS per random mantissa bit)
- HIGH on K-row similarity being secondary (random_4bit > GPTQ_4bit)
- MEDIUM on whether the dedup mechanism in HW is "byte-level" vs "bit-level"
  (the empirical result is bit-entropy correlated; HW could be doing either)

## Methodology lesson

This is a textbook example of why rigor rule #9 matters. The original
"K-row similarity gives 1.13× speedup" finding was correct in measurement
but misattributed in cause. The actual mechanism is broader (bit entropy)
which has different practical implications.

The bit-entropy interpretation is BETTER NEWS for ML inference: ANY data
with low entropy benefits, not just specifically K-row-structured data.

---

## Bit POSITION sensitivity test

Does it matter WHICH bits are random within mantissa?

### 4 random bits at different starting positions:

| start_bit | TFLOPS |
|----------:|-------:|
| 0 | 1812 |
| 1 | 1801 |
| 2 | 1797 |
| 3 | 1798 |

### 1 random bit at different positions:

| start_bit | TFLOPS |
|----------:|-------:|
| 0 | 2008 |
| 1 | 2001 |
| 2 | 1993 |
| 3 | 1989 |
| 4 | 1985 |
| 5 | 1994 |
| 6 | 2003 |

**Bit position doesn't matter. Only bit COUNT matters.** Variance < 1% across positions.

## Implication for quantization design

Any quantization scheme that reduces bit count gives proportional benefit:
- 4-bit weights (INT4-style): ~1.20× speedup vs full random
- 1-bit weights: ~1.34× speedup
- 0-bit (constant): 1.47× speedup

This is precision-mantissa-bit-INDEPENDENT. The mechanism is purely
bit-count entropy at the multiplier level.

## Connect back to microbench

In microbench mode 0-15 (per-bit forcing), we saw:
- Sign bit (15) most impactful: -56W save when forced
- Exp bits 7-13: -30W each
- Mantissa bits 0-6: -15-25W each

So at MICROBENCH (per-bit) level, bit POSITION matters (sign > exp > mantissa).
But at cuBLAS level (with same exp=126 fixed and only mantissa varying), the
positions WITHIN mantissa are equivalent.

The reconciliation: bit-position effects manifest when comparing across
bit FIELDS (sign vs exp vs mantissa). Within a single field (mantissa),
positions are equivalent.

## Final mechanism summary

cuBLAS speedup = f(bit entropy of B) where:
- Each random sign bit: ~165 TFLOPS cost
- Each random exp bit: ~100-150 TFLOPS cost (estimated, untested)
- Each random mantissa bit: ~70 TFLOPS cost (uniform across positions)
- B all-constant: max speedup ~1.47× over fully random
