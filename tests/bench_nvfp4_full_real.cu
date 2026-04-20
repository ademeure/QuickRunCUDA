// REAL NVFP4 row-sum: 16 E2M1 + 1 UE4M3 scale per block → FP32 sum.
// Each thread = 1 block (16 nybbles = 8 bytes data + 1 scale byte).
// 32-thread warp processes 32 blocks = 512 elements per row tile.
// N_ITERS row tiles per kernel call.
//
// Modes:
//   0: NAIVE — software LUT decode + FP32 chain + scale * partial + SHFL chain
//   1: OPTIMAL — HW cvt.f16x2.e2m1x2 + HADD2 chain + sw UE4M3 + f32 promote + SHFL chain
//   2: OPTIMAL+HW UE4M3 — uses cvt.rn.f16.e4m3 for scale (treats UE4M3 ≈ E4M3 since
//      sign bit is always 0 for valid UE4M3 codes)

#include <cuda_fp16.h>

#ifndef MODE
#define MODE 0
#endif
#ifndef N_ITERS
#define N_ITERS 1000
#endif

// Software UE4M3 → f32 decode
__device__ __forceinline__ float ue4m3_to_f32(unsigned int code) {
    unsigned int exp = (code >> 3) & 0xF;
    unsigned int mant = code & 0x7;
    if (exp == 0) {
        // subnormal: value = mant * 2^-9
        return (float)mant * (1.0f / 512.0f);
    } else {
        // normal: value = (1 + mant/8) * 2^(exp-7)
        float val = 1.0f + (float)mant * 0.125f;
        int shift = (int)exp - 7;
        if (shift >= 0) val *= (float)(1u << shift);
        else            val *= 1.0f / (float)(1u << (-shift));
        return val;
    }
}

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // A: data buffer (E2M1 nybbles). 8 bytes per thread.
    // B: scales (UE4M3 bytes). 1 byte per thread.
    unsigned int* Au = (unsigned int*)A;
    unsigned char* Bs = (unsigned char*)B;

    // Init data deterministically (just first time)
    if (threadIdx.x == 0) {
        for (int i = 0; i < 64; i++) {  // 32 threads * 2 dwords
            unsigned int v = 0;
            for (int n = 0; n < 8; n++) v |= ((i*8+n) & 7) << (n*4);
            Au[i] = v;
        }
        for (int i = 0; i < 32; i++) Bs[i] = 0x38 + (i & 0x7);  // ~1.0 area
    }
    __syncwarp();

    float facc = 0.0f;
    long long iacc64 = 0;  // for mode 7 (int64 to avoid overflow)

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        // Per-thread block of 16 E2M1 nybbles + 1 UE4M3 scale
        unsigned int dword0 = Au[threadIdx.x * 2 + 0];
        unsigned int dword1 = Au[threadIdx.x * 2 + 1];
        unsigned char scale_byte = Bs[threadIdx.x];

        // Defeat compiler hoisting per iter
        dword0 ^= ((unsigned)u2 * (unsigned)it);
        dword1 ^= ((unsigned)u2 * (unsigned)it);

#if MODE == 0
        // NAIVE: software decode 16 E2M1 + sw UE4M3 + FP32 multiply
        static const int lut[8] = {0, 1, 2, 3, 4, 6, 8, 12};
        float block_sum = 0.0f;
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            unsigned int code = (dword0 >> (n*4)) & 0xF;
            int mag = lut[code & 0x7];
            block_sum += (float)((code & 0x8) ? -mag : mag) * 0.5f;
        }
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            unsigned int code = (dword1 >> (n*4)) & 0xF;
            int mag = lut[code & 0x7];
            block_sum += (float)((code & 0x8) ? -mag : mag) * 0.5f;
        }
        float scale = ue4m3_to_f32(scale_byte);
        facc = fmaf(block_sum, scale, facc);

#elif MODE == 1
        // OPTIMAL DECODE: HW cvt + HADD2 chain in f16, scale via sw UE4M3, fp32 mul-add
        unsigned int sum_pair = 0;  // 2 packed f16
        // Decode + sum 4 byte-pairs from dword0 (= 8 elements)
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            unsigned short b = (unsigned short)((dword0 >> (n*8)) & 0xFF);
            unsigned int hpair;
            asm volatile("{ .reg .b8 _b,_p; mov.b16 {_b,_p}, %1; "
                         "cvt.rn.f16x2.e2m1x2 %0, _b; }"
                         : "=r"(hpair) : "h"(b));
            asm volatile("add.rn.f16x2 %0, %0, %1;"
                         : "+r"(sum_pair) : "r"(hpair));
        }
        // ... and 4 byte-pairs from dword1 (= 8 more elements)
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            unsigned short b = (unsigned short)((dword1 >> (n*8)) & 0xFF);
            unsigned int hpair;
            asm volatile("{ .reg .b8 _b,_p; mov.b16 {_b,_p}, %1; "
                         "cvt.rn.f16x2.e2m1x2 %0, _b; }"
                         : "=r"(hpair) : "h"(b));
            asm volatile("add.rn.f16x2 %0, %0, %1;"
                         : "+r"(sum_pair) : "r"(hpair));
        }
        // Sum the 2 halves of sum_pair, promote to fp32
        unsigned short lo_h = sum_pair & 0xFFFF;
        unsigned short hi_h = sum_pair >> 16;
        float block_sum = __half2float(__ushort_as_half(lo_h))
                        + __half2float(__ushort_as_half(hi_h));
        // Scale (software UE4M3) * block_sum, accumulate
        float scale = ue4m3_to_f32(scale_byte);
        facc = fmaf(block_sum, scale, facc);

#elif MODE == 3
        // FULL INT DOMAIN: single common scale = 2^-10 fixed-point unit.
        // Each (E2M1 × UE4M3) product is an EXACT integer multiple of 2^-10.
        // Sum entirely in int32, redux.sync.add for warp reduce, convert ONCE at end.
        //
        // Math:
        //   true_value = e2m1 × scale = (e2m1_x2 / 2) × scale
        //   in units of 2^-10:  true_value × 1024 = e2m1_x2 × (scale × 512)
        //                                         = e2m1_x2 × scale_int  (exact integer)
        //   For UE4M3 normal:    scale × 512 = (8+mant) × 2^(exp-1)         (always int)
        //   For UE4M3 subnormal: scale × 512 = mant                          (always int)

        // Step 1: UE4M3 -> int (scale × 512, exact)
        int scale_int;
        {
            unsigned int e = ((unsigned)scale_byte >> 3) & 0xF;
            unsigned int m = (unsigned)scale_byte & 0x7;
            scale_int = (e == 0) ? (int)m : (int)((8u + m) << (e - 1));
        }
        // Step 2: precompute 8 scaled magnitudes (1 IMAD/lookup per E2M1 magnitude)
        int scaled_mag[8];
        static const int lut_x2[8] = {0, 1, 2, 3, 4, 6, 8, 12};
        #pragma unroll
        for (int j = 0; j < 8; j++) scaled_mag[j] = lut_x2[j] * scale_int;
        // Step 3: decode 16 E2M1, lookup scaled magnitude, sign-flip, accumulate (INT)
        int block_int_sum = 0;
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            unsigned int code = (dword0 >> (n*4)) & 0xF;
            int v = scaled_mag[code & 0x7];
            block_int_sum += (code & 0x8) ? -v : v;
        }
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            unsigned int code = (dword1 >> (n*4)) & 0xF;
            int v = scaled_mag[code & 0x7];
            block_int_sum += (code & 0x8) ? -v : v;
        }
        // Step 4: warp redux.sync.add (s32) across all 32 blocks
        int warp_int_sum;
        asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;"
                     : "=r"(warp_int_sum) : "r"(block_int_sum));
        // Step 5: accumulate as INT across tiles (use int64 to avoid overflow at scale)
        // For this test the iterations don't drift, so int32 is fine.
        // Convert at very end.
        facc += (float)warp_int_sum * (1.0f / 1024.0f);

#elif MODE == 7
        // BEST OF BOTH: HW E2M1 cvt + HADD2 packed sum (fast per-element work)
        //   + 1 IMAD per BLOCK (not per element) for int rep
        //   + per-thread int accumulator across all tiles
        //   + ONE final redux.sync.add at very end
        unsigned int sum_pair = 0;
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            unsigned short b = (unsigned short)((dword0 >> (n*8)) & 0xFF);
            unsigned int hpair;
            asm volatile("{ .reg .b8 _b,_p; mov.b16 {_b,_p}, %1; "
                         "cvt.rn.f16x2.e2m1x2 %0, _b; }"
                         : "=r"(hpair) : "h"(b));
            asm volatile("add.rn.f16x2 %0, %0, %1;"
                         : "+r"(sum_pair) : "r"(hpair));
        }
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            unsigned short b = (unsigned short)((dword1 >> (n*8)) & 0xFF);
            unsigned int hpair;
            asm volatile("{ .reg .b8 _b,_p; mov.b16 {_b,_p}, %1; "
                         "cvt.rn.f16x2.e2m1x2 %0, _b; }"
                         : "=r"(hpair) : "h"(b));
            asm volatile("add.rn.f16x2 %0, %0, %1;"
                         : "+r"(sum_pair) : "r"(hpair));
        }
        // Multiply by 2.0 in packed f16 to get integer-valued halves
        unsigned int two_pair = 0x40004000u;
        asm volatile("mul.rn.f16x2 %0, %0, %1;" : "+r"(sum_pair) : "r"(two_pair));
        // Convert each half to s16, sum to s32
        short v0, v1;
        asm volatile("cvt.rni.s16.f16 %0, %1;" : "=h"(v0) : "h"((unsigned short)(sum_pair & 0xFFFF)));
        asm volatile("cvt.rni.s16.f16 %0, %1;" : "=h"(v1) : "h"((unsigned short)(sum_pair >> 16)));
        int block_sum_x2 = (int)v0 + (int)v1;
        // UE4M3 → int (× 512 = exact integer)
        int scale_int;
        {
            unsigned int e = ((unsigned)scale_byte >> 3) & 0xF;
            unsigned int m = (unsigned)scale_byte & 0x7;
            scale_int = (e == 0) ? (int)m : (int)((8u + m) << (e - 1));
        }
        // ONE IMAD per block (not per element)
        int per_tile_int = block_sum_x2 * scale_int;
        // Per-tile redux.sync.add to int32, accumulate to int64 cross-tile
        int rsum_tile;
        asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;"
                     : "=r"(rsum_tile) : "r"(per_tile_int));
        iacc64 += (long long)rsum_tile;

#elif MODE == 5
        // PER-TILE WARP REDUCE INSIDE LOOP: shows the SHFL chain cost
        // (forced apples-to-apples with mode 6 below)
        unsigned int sum_pair = 0;
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            unsigned short b = (unsigned short)((dword0 >> (n*8)) & 0xFF);
            unsigned int hpair;
            asm volatile("{ .reg .b8 _b,_p; mov.b16 {_b,_p}, %1; "
                         "cvt.rn.f16x2.e2m1x2 %0, _b; }"
                         : "=r"(hpair) : "h"(b));
            asm volatile("add.rn.f16x2 %0, %0, %1;"
                         : "+r"(sum_pair) : "r"(hpair));
        }
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            unsigned short b = (unsigned short)((dword1 >> (n*8)) & 0xFF);
            unsigned int hpair;
            asm volatile("{ .reg .b8 _b,_p; mov.b16 {_b,_p}, %1; "
                         "cvt.rn.f16x2.e2m1x2 %0, _b; }"
                         : "=r"(hpair) : "h"(b));
            asm volatile("add.rn.f16x2 %0, %0, %1;"
                         : "+r"(sum_pair) : "r"(hpair));
        }
        unsigned short lo_h = sum_pair & 0xFFFF;
        unsigned short hi_h = sum_pair >> 16;
        float block_sum = __half2float(__ushort_as_half(lo_h))
                        + __half2float(__ushort_as_half(hi_h));
        unsigned short scale_pair_in = (unsigned short)scale_byte;
        unsigned int scale_f16x2;
        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;"
                     : "=r"(scale_f16x2) : "h"(scale_pair_in));
        float scale = __half2float(__ushort_as_half(scale_f16x2 & 0xFFFF));
        float partial = block_sum * scale;
        // PER-TILE WARP REDUCE (fp32 SHFL chain) — done INSIDE the loop
        partial += __shfl_xor_sync(0xFFFFFFFF, partial, 16);
        partial += __shfl_xor_sync(0xFFFFFFFF, partial,  8);
        partial += __shfl_xor_sync(0xFFFFFFFF, partial,  4);
        partial += __shfl_xor_sync(0xFFFFFFFF, partial,  2);
        partial += __shfl_xor_sync(0xFFFFFFFF, partial,  1);
        facc += partial;

#elif MODE == 6
        // SAME AS MODE 5 but use INT redux.sync.add for the per-tile warp reduce.
        // Convert fp32 partial -> int (× 1024 fixed-point), redux, convert back.
        unsigned int sum_pair = 0;
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            unsigned short b = (unsigned short)((dword0 >> (n*8)) & 0xFF);
            unsigned int hpair;
            asm volatile("{ .reg .b8 _b,_p; mov.b16 {_b,_p}, %1; "
                         "cvt.rn.f16x2.e2m1x2 %0, _b; }"
                         : "=r"(hpair) : "h"(b));
            asm volatile("add.rn.f16x2 %0, %0, %1;"
                         : "+r"(sum_pair) : "r"(hpair));
        }
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            unsigned short b = (unsigned short)((dword1 >> (n*8)) & 0xFF);
            unsigned int hpair;
            asm volatile("{ .reg .b8 _b,_p; mov.b16 {_b,_p}, %1; "
                         "cvt.rn.f16x2.e2m1x2 %0, _b; }"
                         : "=r"(hpair) : "h"(b));
            asm volatile("add.rn.f16x2 %0, %0, %1;"
                         : "+r"(sum_pair) : "r"(hpair));
        }
        unsigned short lo_h = sum_pair & 0xFFFF;
        unsigned short hi_h = sum_pair >> 16;
        // Use int decode of UE4M3 (× 512 = exact integer)
        int scale_int;
        {
            unsigned int e = ((unsigned)scale_byte >> 3) & 0xF;
            unsigned int m = (unsigned)scale_byte & 0x7;
            scale_int = (e == 0) ? (int)m : (int)((8u + m) << (e - 1));
        }
        // Convert f16 sums to int (× 2 then cvt for whole-int)
        // partial × scale = block_sum × scale
        // In units of 2^-10:  block_sum_f16 × scale_int × 2^(11-something)
        // Cleaner: compute block_sum as int via cvt(2 × f16) → s16
        unsigned short doubled_lo, doubled_hi;
        unsigned int two_pair = 0x40004000u;
        unsigned int doubled_pair = sum_pair;
        asm volatile("mul.rn.f16x2 %0, %0, %1;" : "+r"(doubled_pair) : "r"(two_pair));
        doubled_lo = doubled_pair & 0xFFFF;
        doubled_hi = doubled_pair >> 16;
        short v0, v1;
        asm volatile("cvt.rni.s16.f16 %0, %1;" : "=h"(v0) : "h"(doubled_lo));
        asm volatile("cvt.rni.s16.f16 %0, %1;" : "=h"(v1) : "h"(doubled_hi));
        // block_sum_x2 = sum of e2m1_x2 values; product with scale_int gives units of 2^-10
        int partial_int = ((int)v0 + (int)v1) * scale_int;
        int rsum;
        asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;"
                     : "=r"(rsum) : "r"(partial_int));
        facc += (float)rsum * (1.0f / 1024.0f);

#elif MODE == 4
        // FULL INT DOMAIN + HW E2M1 cvt: same as mode 3 but use HW decode
        // Step 1: same UE4M3 → int
        int scale_int;
        {
            unsigned int e = ((unsigned)scale_byte >> 3) & 0xF;
            unsigned int m = (unsigned)scale_byte & 0x7;
            scale_int = (e == 0) ? (int)m : (int)((8u + m) << (e - 1));
        }
        // Step 2: HW decode 16 E2M1 → 16 f16 (× 2 to get integer-valued f16) → 16 int
        int block_int_sum = 0;
        // Process 4 byte-pairs per dword (8 nybbles per dword)
        #pragma unroll
        for (int dw = 0; dw < 2; dw++) {
            unsigned int dword = (dw == 0) ? dword0 : dword1;
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                unsigned short b = (unsigned short)((dword >> (n*8)) & 0xFF);
                unsigned int hpair;
                asm volatile("{ .reg .b8 _b,_p; mov.b16 {_b,_p}, %1; "
                             "cvt.rn.f16x2.e2m1x2 %0, _b; }"
                             : "=r"(hpair) : "h"(b));
                // Each f16 in hpair is a value in {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
                // Multiply by 2 → integer-valued f16 in {0, ±1, ±2, ..., ±12}
                unsigned int two_pair = 0x40004000u;  // {2.0, 2.0} packed
                unsigned int doubled;
                asm volatile("mul.rn.f16x2 %0, %0, %1;"
                             : "+r"(hpair) : "r"(two_pair));
                doubled = hpair;
                // Convert each half to s16 via cvt.rni
                short v0, v1;
                asm volatile("cvt.rni.s16.f16 %0, %1;" : "=h"(v0) : "h"((unsigned short)(doubled & 0xFFFF)));
                asm volatile("cvt.rni.s16.f16 %0, %1;" : "=h"(v1) : "h"((unsigned short)(doubled >> 16)));
                block_int_sum += ((int)v0 + (int)v1) * scale_int;
            }
        }
        // Step 3: warp redux.sync.add
        int warp_int_sum;
        asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;"
                     : "=r"(warp_int_sum) : "r"(block_int_sum));
        facc += (float)warp_int_sum * (1.0f / 1024.0f);

#elif MODE == 2
        // OPTIMAL + try HW UE4M3 via cvt.rn.f16.e4m3 (UE4M3 ⊂ E4M3 without sign)
        unsigned int sum_pair = 0;
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            unsigned short b = (unsigned short)((dword0 >> (n*8)) & 0xFF);
            unsigned int hpair;
            asm volatile("{ .reg .b8 _b,_p; mov.b16 {_b,_p}, %1; "
                         "cvt.rn.f16x2.e2m1x2 %0, _b; }"
                         : "=r"(hpair) : "h"(b));
            asm volatile("add.rn.f16x2 %0, %0, %1;"
                         : "+r"(sum_pair) : "r"(hpair));
        }
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            unsigned short b = (unsigned short)((dword1 >> (n*8)) & 0xFF);
            unsigned int hpair;
            asm volatile("{ .reg .b8 _b,_p; mov.b16 {_b,_p}, %1; "
                         "cvt.rn.f16x2.e2m1x2 %0, _b; }"
                         : "=r"(hpair) : "h"(b));
            asm volatile("add.rn.f16x2 %0, %0, %1;"
                         : "+r"(sum_pair) : "r"(hpair));
        }
        unsigned short lo_h = sum_pair & 0xFFFF;
        unsigned short hi_h = sum_pair >> 16;
        float block_sum = __half2float(__ushort_as_half(lo_h))
                        + __half2float(__ushort_as_half(hi_h));
        // HW decode UE4M3 → f16 via cvt.rn.f16x2.e4m3x2 (treats high byte as 0)
        unsigned short scale_pair_in = (unsigned short)scale_byte;  // low byte = scale, high = 0
        unsigned int scale_f16x2;
        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;"
                     : "=r"(scale_f16x2) : "h"(scale_pair_in));
        float scale = __half2float(__ushort_as_half(scale_f16x2 & 0xFFFF));
        facc = fmaf(block_sum, scale, facc);
#endif
    }

#if MODE == 0 || MODE == 1 || MODE == 2
    // Modes 0/1/2: per-thread fp32 partials, single final SHFL reduce
    facc += __shfl_xor_sync(0xFFFFFFFF, facc, 16);
    facc += __shfl_xor_sync(0xFFFFFFFF, facc,  8);
    facc += __shfl_xor_sync(0xFFFFFFFF, facc,  4);
    facc += __shfl_xor_sync(0xFFFFFFFF, facc,  2);
    facc += __shfl_xor_sync(0xFFFFFFFF, facc,  1);
#elif MODE == 7
    // Already reduced per tile + accumulated in int64; convert to fp32
    facc = (float)iacc64 * (1.0f / 1024.0f);
#endif
    // Modes 3/4/5/6 already reduced (per-tile or per-kernel) — facc is uniform across lanes.

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if ((int)facc == seed) ((unsigned*)C)[blockIdx.x] = (unsigned)facc;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d N_ITERS=%d clk=%llu cy/iter=%.3f facc=%.4f\n",
               MODE, N_ITERS, t1 - t0, (double)(t1-t0)/(double)N_ITERS, facc);
    }
}
