# Fast cross-warp reduction — V4 / Q3

**Date: 2026-04-20.** Test `bench_warp_reduce_methods.cu`. 1500 MHz,
single warp, NC=8 chained. Comparing four warp-reduction methods.

## Results: cycles per 32-lane sum-reduce

| Method | cy/reduce | Speedup vs SHFL chain |
|--------|-----------|----------------------|
| 5-step `SHFL.bfly` chain (manual PTX) | 27.19 | 1.0× (baseline) |
| 5-step `SHFL.up` chain (manual PTX)   | 27.19 | 1.0× |
| `__shfl_xor_sync` intrinsic chain (CUDA) | 27.19 | 1.0× |
| **`redux.sync.add.u32`** (single inst) | **11.61** | **2.34× faster** |

## SASS evidence

`redux.sync.add` compiles to `REDUX.SUM UR<n>, R<m>` — the destination
is the **uniform register file (URF)**, since the result is identical
across all 32 lanes. SHFL chain emits 5 SHFL.BFLY + 5 inferred adds.

Counts in SASS body (NC=8, UNROLL=16):
- SHFL chain methods: 640 SHFL ops (= 8 × 5 × 16) ✓
- redux.sync.add: 128 REDUX ops (= 8 × 16) ✓

## Why 2.34× faster

A 5-step SHFL chain has 5 sequential SHFL+ADD pairs, each ~5 cy
(SHFL latency), so chain length = ~25-27 cy. The single `REDUX.SUM`
instruction performs the entire 32→1 reduction in hardware in ~11 cy
— roughly 2× faster than the chain because it bypasses 4 of the 5
SHFL latencies.

## Practical recipe

For sum/min/max/and/or/xor across a warp (32 lanes):

```cpp
// SLOW (legacy CUDA pattern):
for (int offset = 16; offset > 0; offset /= 2)
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);

// FAST (Hopper+, sm_90+, single inst):
asm volatile("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;"
             : "=r"(sum) : "r"(sum));
```

`redux.sync` supports: `.add`, `.min`, `.max`, `.and`, `.or`, `.xor`
for `.u32`/`.s32`. It does NOT support FP — for FP reduction you
still need a SHFL chain (or first-bitcast trick).

## Implications

Any reduction kernel that uses warp-reduce as a building block
should switch:
- AllReduce primitives at warp boundary
- Per-warp absmax / max for quantization
- Per-warp histogram bin sum
- Block-level reduction's first stage (warp → SMEM → block)

For a typical "warp-reduce + SMEM-reduce + final atomic" kernel,
the warp-reduce is ~25% of total time (3 stages of ~equal cost),
so 2.34× faster warp-reduce = ~14% kernel speedup.

## Limitations

- INTEGER ONLY — no `.f32` variant of `redux.sync` exists. FP
  reduction must still use SHFL chain (or bitcast workaround).
- Single warp — for cross-warp (block-level) reduction, still need
  SMEM staging.
- The SHFL chain measurement assumes serial chain dep through `v[k]`
  to defeat compiler ILP. Actual reduction kernels may have
  different ILP profile.

## Confidence

- **HIGH** that REDUX.SUM is 2.34× faster than SHFL chain (3 trials,
  SASS-verified, clean cycle counts)
- **HIGH** for the per-reduce timings (clock64-internal, no launch noise)
- **HIGH** for the recipe (works on sm_90+ including B300/sm_103a)
- **MED** for "14% kernel speedup" estimate — depends on kernel profile

## Files

- `tests/bench_warp_reduce_methods.cu` — METHOD 0/1/2/3
