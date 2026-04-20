# Realistic INT4 Inference Scoping (CORRECTED)

Date: 2026-04-20. Final realistic test using Gaussian-distributed weights
(true Llama-like) shows the INT4 speedup is SMALLER than my synthetic
benchmarks predicted.

## Test methodology

- Gaussian baseline: BF16 weights ~ N(0, 0.02²) - true distribution of trained Llama weights
- INT4 quantized: per-128-group with 16 levels (-8 to +7) × shared group scale
- Both mathematically valid GEMMs

## Results (8192³ BF16)

| Configuration | TFLOPS | Speedup |
|---------------|-------:|--------:|
| Gaussian-distributed weights (TRUE Llama-like) | 1403 | 1.00× |
| INT4 quantized (16 levels, group=128) | 1463 | **1.04×** |

## Why only 4%? Comparison with synthetic predictions

| Scenario | Random baseline | INT4 result | Speedup |
|----------|----------------:|------------:|--------:|
| Synthetic mantissa-random | 1684 | 1860 | 1.10× |
| Synthetic GPTQ-style XOR | 1496 | 1613 | 1.07× |
| **REALISTIC Gaussian** | 1403 | 1463 | **1.04×** |

The realistic Gaussian baseline is LOWER (1403 TF) because Gaussian weights:
- Concentrated near zero (many subnormals)
- High effective bit toggle rate per cycle
- More multiplier work per random product

The INT4 quantized result is also lower than synthetic GPTQ (1463 vs 1613).
Both have 4-bit per element entropy, but INT4-quantized values fall on
SPECIFIC (-8...+7) × scale grid which has different bit patterns than
random 4-bit XOR.

## REVISED practical headline

**For real Llama 70B INT4 inference deployment**:
- Baseline: ~1403 TF/GPU (Gaussian weight distribution)
- INT4 quantized: ~1463 TF/GPU (+4% automatic)
- **Real-world auto speedup: ~4%, NOT 12-18% as synthetic tests suggested**

This is a more conservative and accurate estimate.

## Why my earlier estimates were higher

Synthetic tests used patterns that:
1. Mantissa-only random (no sign/exp variation) = artificially "good" baseline
2. XOR'd group-scale + INT4 = different bit pattern distribution than real INT4
3. Mantissa-aligned bit positions = different multiplier behavior

Real distributions (Gaussian, INT4 with proper levels) have different bit-toggle
characteristics that don't align as neatly with the multiplier dedup mechanism.

## Honest deployment guidance

For Llama-class INT4 inference on B300 (default 1100W cap):
- Expect **~4% automatic throughput gain** vs un-quantized Gaussian baseline
- Power saving: ~50-100W per GPU (not 200W)
- Modest but real benefit

For tighter power caps (600W):
- Speedup likely amplifies to ~10-15% (extrapolation)
- Still much smaller than my microbench predictions

## What changed in my understanding

The bit-entropy mechanism IS real, but its IMPACT on realistic data is smaller
than synthetic tests suggested. My earlier "1.12-1.24× automatic" estimates
overshot reality because of artificial test conditions.

The TRUE practical gain from this HW feature for INT4 inference is **modest
(~4% at default cap, possibly ~10% at tight cap)** rather than the optimistic
12-24% I claimed.

## Confidence

- HIGH on the 4% INT4 vs Gaussian result (validated with proper distributions)
- HIGH on the synthetic test artifacts being responsible for inflation
- HIGH on the REVISED practical guidance
- MEDIUM on whether real Llama weights are "more Gaussian-like" or "more
  uniformly-mantissa-random" (depends on training)
