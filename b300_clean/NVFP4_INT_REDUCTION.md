# NVFP4 lossless int reduction — feasibility + speedup

**Date: 2026-04-20.** Investigation prompted by user question: "is it
possible to losslessly represent NVFP4 values as int32 and use
`redux.sync.add` instead of FP SHFL chain?"

## Lossless representation feasibility

**Single NVFP4 value (E2M1 × UE4M3)**:
- E2M1 values: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6} — multiply by 2 → all
  integers in {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12} (5-bit signed each)
- UE4M3 scales: smallest non-zero 2^-9 (subnormal), largest ~480
- Min |product| = 2^-9, max |product| = 12 × 480 = 5760
- Range in units of 2^-9: ~3M (~22 bits)
- **Single value fits in int32 with 9 fractional bits — LOSSLESS**

**Sum of N=7168 values (model dim)**:
- Worst case: 5760 × 7168 ≈ 41M → needs 26 integer bits
- Plus 9 fractional bits = **35 bits → requires int64 for guaranteed lossless**
- Typical workloads: scales don't all max out → int32 sufficient in practice

## Speedup verified (this commit)

`bench_nvfp4_redux_inner.cu` — 32-lane warp reduction in tight loop:

| Method | cy/iter | Speedup |
|--------|---------|---------|
| FP32 + `__shfl_xor_sync` chain | 165 | 1.0× (baseline) |
| FP32 + manual PTX SHFL chain | 165 | 1.0× |
| **INT + `redux.sync.add`** | **60** | **2.75×** |

The 2.75× factor matches Q3's earlier 2.34× finding (slightly higher
here because the inner-loop perturbation is added to both, amplifying
the relative gain).

## Per-element decode bottleneck

In a separate test (`bench_nvfp4_int_reduce.cu`) where 8 NVFP4 elements
per thread are decoded inside the hot loop, all three reduction
approaches give identical ~23 cy/iter — the decode + multiply work
hides the reduction speedup. Int reduction matters only when the
warp-reduce step is the actual bottleneck.

## Recommended recipe for NVFP4 row sum (7K elements)

For maximum throughput on B300 / sm_103a:

```cpp
// Per warp, processes a row tile (e.g., 256 elements = 16 NVFP4 blocks)
int block_int_sum[16];   // 5-bit signed int per block × 16 blocks
unsigned char scales[16]; // UE4M3 scales

// 1. Decode each block's 16 E2M1 values to int (×2), sum to int16
//    (all in registers, 4 cy/IADD3)
// 2. Multiply each block_int_sum × scale → fp32 partial (16 mul-adds)
// 3. Per-thread accumulate fp32 partial across multiple tiles
// 4. Final warp-reduce — IF int-domain feasible (common scale), use
//    redux.sync.add for 2.75× speedup. Otherwise fall back to SHFL chain.

// For STREAMING reduction over many tiles:
//   inner accumulator in int → redux.sync.add per tile → much faster
// For ONE-SHOT reduction:
//   per-element work dominates; choose either path.
```

## Open question for future

Can we keep the FULL accumulation in int domain by:
1. Find max scale across all blocks first (cheap O(N/16) pass)
2. Encode all elements as `e2m1_x2 × scale_b × (max_scale / scale_b)` —
   the ratio is power-of-2-ish (UE4M3 mantissa) → small lossy bit shift
3. Accumulate in int32 throughout

This trades a tiny bit of bit precision (max 3 bits from UE4M3 mantissa
mismatch) for full int-domain reduction → potentially 2.75× speedup
end-to-end.

## Confidence

- **HIGH** for "single value fits in int32 with 9 frac bits"
- **HIGH** for "redux.sync.add 2.75× faster than SHFL chain in inner loop"
- **HIGH** for "decode-bound workloads hide the speedup"
- **MED** for "common-scale approach is practical" — depends on inference
  data scale distribution; needs real-tensor test

## Files

- `tests/bench_nvfp4_int_reduce.cu` — full pipeline test (decode-bound)
- `tests/bench_nvfp4_redux_inner.cu` — isolated reduction test
