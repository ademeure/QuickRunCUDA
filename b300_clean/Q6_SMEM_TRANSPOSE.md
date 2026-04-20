# 32×32 SHMEM transpose at SoL — V4 / Q6

**Date: 2026-04-20.** Test `bench_smem_transpose.cu`. Single warp,
1500 MHz, 10000 iters of full 32×32 read+write through SHMEM.

## Result

| Pattern | cy/iter | Speedup vs naive |
|---------|---------|------------------|
| **NAIVE** `smem[i][k]/smem[k][i]` (32×32) | 2191 | 1.0× |
| **+1 padding** `smem[32][33]` | **267** | **8.2× faster** |
| **Skewed** `smem[i][(i+k)&31]` | 268 | 8.2× faster |
| Same-bank baseline (no transpose) | 1024 | 2.1× faster |

## What's happening

NAIVE: 32 lanes write to one column of `smem[32][32]` (one bank per cell).
Then read from one row → all 32 lanes hit the SAME bank (column k of the
transposed view) → 32-way bank conflict, serial 32 cycles per warp inst.

PADDING: `smem[32][33]` shifts each row by 1 bank; now reading a "column"
of the transposed view hits 32 distinct banks → no conflict.

SKEWED: write `smem[i][(i+k)&31]` permutes elements; read at the matching
permutation reverses it. Same bank-distribution effect, no padding cost.

## Per-element cost

Each iter does:
- 32 stores per lane (1024 stores/warp)
- 32 loads per lane (1024 loads/warp)
- 2 syncwarp (≈ 2 cy total — see F6 finding)

NAIVE: 2191 cy / 32 lanes / 32 loads = ≈ 2.1 cy per load × 32 (conflict serial)
PADDING/SKEWED: 267 cy / 32 lanes / (32+32) ≈ 0.13 cy per access (interleaved)

The **32-way bank conflict serializes the read step into 32 cycles per warp
inst** instead of 1, accounting for the 8× slowdown.

## Recipe

For any 32-lane transpose pattern (matrix tile, register-to-shared
shuffle, etc.):

```cpp
// SLOW
__shared__ float smem[32][32];

// FAST: pad by one bank
__shared__ float smem[32][33];

// FAST: skewed stride (same speed, no extra memory)
__shared__ float smem[32][32];
// then index as smem[i][(i+k)&31] in BOTH writer and reader
```

The padded variant uses 4 KB extra SHMEM per tile (negligible). The
skewed variant uses no extra memory but requires careful index
arithmetic in both directions.

## Confidence

- **HIGH** for the 8.2× speedup (3 trials, stable, well-known
  bank-conflict pattern)
- **HIGH** that padding and skewing are equivalent in speed
- **HIGH** for the underlying mechanism (32-way conflict, `cy/ld=74`
  matches D5 finding for stride-32 access)

## Files

- `tests/bench_smem_transpose.cu` — 4 modes
- See also `b300_clean/D5_SMEM_BANK_BEHAVIOR.md` (basic bank rules)
