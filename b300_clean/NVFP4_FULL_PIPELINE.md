# NVFP4 → FP32 row-sum: complete optimization analysis

**Date: 2026-04-20.** Test `bench_nvfp4_full_real.cu`. Real NVFP4 pipeline:
16 E2M1 elements + 1 UE4M3 scale per block, accumulated to FP32.
Single warp = 32 blocks = 512 elements per tile. N_ITERS=10000 tiles.
1500 MHz clock-locked.

## Results (all 7 modes produce identical facc=16560000.0 — correctness verified)

| Mode | Strategy | cy/tile | Speedup | Notes |
|------|----------|---------|---------|-------|
| 0 | NAIVE: sw LUT E2M1 + sw UE4M3 + fp32 + final SHFL | 252 | 1.0× | baseline |
| 1 | HW `cvt.rn.f16x2.e2m1x2` + sw UE4M3 + fp32 + final SHFL | 140 | 1.80× | HW E2M1 decode only |
| **2** | **HW E2M1 + HW UE4M3 (`cvt.f16x2.e4m3x2`) + fp32 + final SHFL** | **85** | **2.96×** ★ | **OVERALL BEST** |
| 3 | full int domain: sw LUT + per-element IMAD + redux.sync | 248 | 1.02× | IMAD-per-element kills it |
| 4 | full int + HW cvt + per-element IMAD + redux.sync | 272 | 0.93× | cvt overhead makes it worse |
| 5 | HW E2M1+UE4M3 + fp32 + **per-tile SHFL** | 234 | 1.08× | per-tile SHFL reduce inside loop |
| 7 | HW E2M1 + HADD2 + 1 IMAD/block + **per-tile redux.sync.add** + int64 acc | 158 | 1.59× | per-tile redux inside loop |

## Two wins, independently verified

**Win 1 — HW decoders (structural change)**:
- `cvt.rn.f16x2.e2m1x2` on F2FP pipe at 64/SM/cy → 38.5 Telements/s
- `cvt.rn.f16x2.e4m3x2` for UE4M3 (UE4M3 ⊂ E4M3 with sign=0)
- Replaces sw LUT + conditional negate + FP32 scalar multiplies
- **1.80× from E2M1 alone, 2.96× when both decoders used**

**Win 2 — `redux.sync.add` vs SHFL chain**:
- Apples-to-apples with matching kernel structure (mode 5 vs 7 both do per-tile reduce)
- Mode 5 (fp32 SHFL chain per tile): 234 cy
- Mode 7 (int redux.sync.add per tile): 158 cy
- **1.47× speedup** for the warp-reduce step — confirms Q3 finding in realistic pipeline

## The decisive trick: DEFER the reduction

**Mode 2 beats mode 7** (85 vs 158 cy/tile) NOT because its reduce method is
faster, but because it **only reduces ONCE per kernel call** instead of per
tile. The full SHFL chain (5 SHFLs) at the end of a kernel is ~25 cycles
total — negligible vs doing it 10000 times.

- Per-tile reduce overhead dominates (even with fast redux)
- Per-thread fp32 accumulator (mode 2) skips the issue entirely
- The redux.sync speedup only matters when per-tile reduce is
  structurally forced (atomic output, streaming to global, etc.)

## Lossless integer representation — confirmed

To the user's original math question:

- **Single value** (E2M1 × UE4M3): every product is an exact integer
  multiple of 2^-10 (= 1/1024)
  - E2M1 values ×2 are integers in {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}
  - UE4M3 × 512 is always an exact integer (mantissa of 3 bits, exp shift)
  - Product fits in ~22 bits signed → **int32 LOSSLESS per value**
- **Sum of 7168 values**: worst case ~35 bits → needs int64 for strict
  lossless; int32 typical if scales don't all max out
- **Mode 7 uses int64 cross-tile accumulator** to guarantee lossless
  across any number of tiles

## Core decode code (mode 2, the winner)

```cpp
// Per thread (one 16-element block):
unsigned int dword0 = Au[threadIdx.x * 2 + 0];  // 8 nybbles
unsigned int dword1 = Au[threadIdx.x * 2 + 1];  // 8 nybbles
unsigned char scale_byte = Bs[threadIdx.x];     // UE4M3 scale

unsigned int sum_pair = 0;  // packed f16x2 accumulator
#pragma unroll
for (int n = 0; n < 4; n++) {
    unsigned short b = (unsigned short)((dword0 >> (n*8)) & 0xFF);
    unsigned int hpair;
    asm volatile("{ .reg .b8 _b,_p; mov.b16 {_b,_p}, %1; "
                 "cvt.rn.f16x2.e2m1x2 %0, _b; }"  // HW E2M1 decode
                 : "=r"(hpair) : "h"(b));
    asm volatile("add.rn.f16x2 %0, %0, %1;"       // HADD2 packed add
                 : "+r"(sum_pair) : "r"(hpair));
}
// ... same 4-loop for dword1 ...

// Promote packed f16 sum to fp32
float block_sum = __half2float(__ushort_as_half((unsigned short)(sum_pair & 0xFFFF)))
                + __half2float(__ushort_as_half((unsigned short)(sum_pair >> 16)));

// HW UE4M3 decode (UE4M3 byte as low 8 bits of E4M3 input → sign bit = 0)
unsigned short scale_pair_in = (unsigned short)scale_byte;
unsigned int scale_f16x2;
asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;"
             : "=r"(scale_f16x2) : "h"(scale_pair_in));
float scale = __half2float(__ushort_as_half(scale_f16x2 & 0xFFFF));

// Accumulate per-thread fp32 across all tiles
facc = fmaf(block_sum, scale, facc);
// ... loop next tile ...
// ONE final SHFL chain at end of kernel
```

## Core int-redux code (mode 7, for per-tile-reduce scenarios)

```cpp
// Same HW decode + HADD2 as mode 2, ending with sum_pair (packed f16x2)

// Multiply by 2.0 in f16 to get integer-valued halves
unsigned int two_pair = 0x40004000u;
asm volatile("mul.rn.f16x2 %0, %0, %1;" : "+r"(sum_pair) : "r"(two_pair));

// Convert each half to int16, sum to int
short v0, v1;
asm volatile("cvt.rni.s16.f16 %0, %1;" : "=h"(v0) : "h"(...));
asm volatile("cvt.rni.s16.f16 %0, %1;" : "=h"(v1) : "h"(...));
int block_sum_x2 = (int)v0 + (int)v1;

// UE4M3 → exact int (× 512)
int scale_int = (exp == 0) ? mant : (int)((8u + mant) << (exp - 1));

// ONE IMAD per block (vs 16 per-element IMADs in modes 3/4)
int per_tile_int = block_sum_x2 * scale_int;

// redux.sync.add per tile, accumulate to int64 cross-tile
int rsum_tile;
asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;"
             : "=r"(rsum_tile) : "r"(per_tile_int));
iacc64 += (long long)rsum_tile;

// At very end: convert int64 → fp32 via × 2^-10
facc = (float)iacc64 * (1.0f / 1024.0f);
```

## SASS (mode 2 inner loop)

```
F2FP.F16.E4M3.UNPACK_B R8, R12          // 1 inst UE4M3 decode
HADD2.F32 R8, -RZ, R8.H0_H0             // unpack to fp32

F2FP.F16.E2M1.UNPACK_B R11, R11         // 8 E2M1 decodes
F2FP.F16.E2M1.UNPACK_B R13, R12
HADD2 R12, R12, R13                     // packed f16 sum chain
F2FP.F16.E2M1.UNPACK_B R14, R14
F2FP.F16.E2M1.UNPACK_B R10, R10
HADD2 R10, R12, R10
...
HADD2.F32 R11, -RZ, R10.H0_H0           // unpack f16 sum to f32
HADD2.F32 R10, -RZ, R10.H1_H1
FFMA.FTZ R9, R10, R8, R9                // block_sum × scale + acc
```

9 F2FP + 7 HADD2 + 1 FFMA per block. No explicit IMADs, no redundant
moves. At F2FP's 64/SM/cy throughput, the decode work is ~0.14 cy/inst =
highly efficient.

## Confidence

- **HIGH** that mode 2 is optimal for "accumulate-many-tiles" scenarios
  (7 modes tested, all produce identical correct result, stable timings)
- **HIGH** that `redux.sync.add` is 1.47× faster than SHFL chain when
  reduce is per-tile (mode 5 vs 7 apples-to-apples)
- **HIGH** for "single NVFP4 value fits int32 lossless with 9 frac bits"
- **MED** for "int64 needed for 7168-sum lossless" — typical inference
  scale distributions might fit int32

## Files

- `tests/bench_nvfp4_full_real.cu` — all 7 modes
- `b300_clean/NVFP4_INT_REDUCTION.md` — lossless int feasibility analysis
- `b300_clean/Q3_WARP_REDUCE_RECIPES.md` — isolated redux.sync.add 2.34×
