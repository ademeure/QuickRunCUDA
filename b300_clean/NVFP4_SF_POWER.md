# NVFP4 Scale Factor (SF) Power Effect

Date: 2026-04-20. Tests if NVFP4's separate UE4M3 scale factor tensor
contributes to data-dependent power independently of A and B operands.

## Setup
- `tests/bench_tcgen05_nvfp4_kvary_power.cu` (extended with SF mode)
- NVFP4 m128n128k64 with kind::mxf4nvf4.block_scale.block16
- SF stored in TMEM (4 chunks × 128 × 32-bit per warp init)
- Use 'verify' arg to control SF: 0=1.0(default), 1=zeros, 2=random, 3=patterned
- @ -lgc 1005 MHz, 50M iters, 148 SMs

## Results

| SF pattern | B random (W) | B const +1.0 (W) |
|------------|-------------:|-----------------:|
| 1.0 (UE4M3=0x38) | 463 | 284 |
| 0 (zeros) | 437 | 279 |
| random | 479 | 311 |
| patterned (0xAAAA) | 469 | 284 |

## Findings

1. **SF random adds ~27W with B const** (independent SF-side cost)
2. **SF random adds ~16W on top of B random** (smaller marginal)
3. **SF=0 saves ~5W** (some products gated to zero)
4. **SF patterned (uniform)** has no effect (just like baseline)

## Mechanism

NVFP4 has THREE independent data inputs that affect power:
1. **A operand**: ~free (broadcast)
2. **B operand**: 32-byte sub-tile dedup applies (~180W max for NVFP4)
3. **SF operand**: ~27W random penalty (smaller scale)

Total NVFP4 random data baseline (463W) decomposes:
- Static baseline: ~280W
- B operand contribution: ~150-170W
- SF contribution: ~13-30W

## Implication for ML inference

For NVFP4 quantized inference:
- Activation scale factors (per-token) → typically MORE variation
- Weight scale factors (per-channel) → typically less variation
- Combined SF varies → ~30W extra power per CTA

## Confidence

- HIGH on SF random adding measurable power (4 measurements, consistent)
- HIGH on SF=0 reducing power slightly (consistent across B patterns)
- MEDIUM on the exact magnitude (only 1 random hash function tested)
- LOW on whether SF dedup mechanism exists (untested with N-vary on SF)
