// Full optimized NVFP4 row-sum pipeline:
//   Mode 0: NAIVE — LUT decode + FP32 chain + SHFL chain reduce
//   Mode 1: HW DECODE — cvt.rn.f16x2.e2m1x2 + HADD2 + final SHFL chain
//   Mode 2: FULL OPTIMIZED — HW decode → int conversion (multiply by 2 nybble→int4)
//           → int accumulate → redux.sync.add per row
//
// 32-thread warp processes 256 NVFP4 elements per row, N_ITERS rows.

#include <cuda_fp16.h>

#ifndef MODE
#define MODE 0
#endif
#ifndef N_ITERS
#define N_ITERS 10000
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int* Au = (unsigned int*)A;
    if (threadIdx.x == 0) {
        for (int i = 0; i < 32; i++) {
            unsigned int v = 0;
            for (int n = 0; n < 8; n++) v |= ((i*8+n) & 7) << (n*4);
            Au[i] = v;
        }
    }
    __syncwarp();

    float ftotal = 0.0f;
    int   itotal = 0;
    unsigned int dword = Au[threadIdx.x];

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        unsigned int d = dword ^ ((unsigned)u2 * (unsigned)it);

#if MODE == 0
        // NAIVE: LUT decode + FP32 + SHFL chain
        static const int lut[8] = {0, 1, 2, 3, 4, 6, 8, 12};
        float partial = 0.0f;
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            unsigned int code = (d >> (n*4)) & 0xF;
            int mag = lut[code & 0x7];
            int signed_val = (code & 0x8) ? -mag : mag;
            partial += (float)signed_val * 0.5f;
        }
        // Warp reduce via SHFL chain
        partial += __shfl_xor_sync(0xFFFFFFFF, partial, 16);
        partial += __shfl_xor_sync(0xFFFFFFFF, partial,  8);
        partial += __shfl_xor_sync(0xFFFFFFFF, partial,  4);
        partial += __shfl_xor_sync(0xFFFFFFFF, partial,  2);
        partial += __shfl_xor_sync(0xFFFFFFFF, partial,  1);
        ftotal += partial;
#elif MODE == 1
        // HW DECODE: cvt + HADD2 packed + SHFL chain reduce
        unsigned int sum_pair_bits = 0;
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            unsigned short byte = (unsigned short)((d >> (n*8)) & 0xFF);
            unsigned int hpair;
            asm volatile("{ .reg .b8 _b, _p; mov.b16 {_b,_p}, %1; cvt.rn.f16x2.e2m1x2 %0, _b; }"
                         : "=r"(hpair) : "h"(byte));
            asm volatile("add.rn.f16x2 %0, %0, %1;"
                         : "+r"(sum_pair_bits) : "r"(hpair));
        }
        unsigned short lo16 = sum_pair_bits & 0xFFFF;
        unsigned short hi16 = sum_pair_bits >> 16;
        float partial = __half2float(__ushort_as_half(lo16))
                      + __half2float(__ushort_as_half(hi16));
        partial += __shfl_xor_sync(0xFFFFFFFF, partial, 16);
        partial += __shfl_xor_sync(0xFFFFFFFF, partial,  8);
        partial += __shfl_xor_sync(0xFFFFFFFF, partial,  4);
        partial += __shfl_xor_sync(0xFFFFFFFF, partial,  2);
        partial += __shfl_xor_sync(0xFFFFFFFF, partial,  1);
        ftotal += partial;
#elif MODE == 2
        // FULL OPTIMIZED: int domain throughout
        // Each E2M1 ×2 is in {0,±1,±2,±3,±4,±6,±8,±12} fitting in int8.
        // Sum 8 of them per thread = int range ~96. Use redux.sync.add
        // per row, then convert at end (×0.5 for the implicit ÷2).
        // Use PRMT to extract bytes + LUT via small constant (PRMT.F4E)
        // No that doesn't exist. Use cvt to f16x2 → cast to int via simple bit ops.
        // Simpler: still need the LUT for sign expansion + magnitude.
        // Alternative: use cvt.rn.f16x2.e2m1x2 → cast to int by bit-equivalent
        // Actually the cleanest int decode: use the LUT but accumulate as int.
        static const int lut[8] = {0, 1, 2, 3, 4, 6, 8, 12};
        int partial = 0;
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            unsigned int code = (d >> (n*4)) & 0xF;
            int mag = lut[code & 0x7];
            int signed_val = (code & 0x8) ? -mag : mag;
            partial += signed_val;
        }
        // Warp reduce via redux.sync.add (INT only)
        int rsum;
        asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;"
                     : "=r"(rsum) : "r"(partial));
        itotal += rsum;
#elif MODE == 3
        // HW cvt → int via f16 bits → int via cvt → redux.sync.add
        // cvt.rn.f16x2.e2m1x2 produces exact (×0.5) representable values
        // The values are all multiples of 0.5; ×2 gives integers in
        // {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}. We can use cvt.rni.s32.f16
        // to convert each f16 to int (rounded, but exact since values
        // are integer × 0.5 = whole or half-integer; ×2 first then convert).
        int partial = 0;
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            unsigned short byte = (unsigned short)((d >> (n*8)) & 0xFF);
            unsigned int hpair;
            asm volatile("{ .reg .b8 _b, _p; mov.b16 {_b,_p}, %1; cvt.rn.f16x2.e2m1x2 %0, _b; }"
                         : "=r"(hpair) : "h"(byte));
            // Multiply by 2 (shift exp by 1) then convert to int
            // Easier: cvt h→int, multiply by 2 in int domain
            unsigned short hlo = hpair & 0xFFFF;
            unsigned short hhi = hpair >> 16;
            short v0, v1;
            asm volatile("cvt.rni.s16.f16 %0, %1;" : "=h"(v0) : "h"(hlo));
            asm volatile("cvt.rni.s16.f16 %0, %1;" : "=h"(v1) : "h"(hhi));
            // v0/v1 are exact half-integer; multiply by 2 to get whole ints
            // Wait — cvt.rni rounds, so we LOSE the 0.5 step. Better cvt to
            // f16, multiply by 2.0 first, then cvt to int. But that's an
            // extra op. Alternative: extract magnitude+sign from the
            // f16 bit representation directly via bit ops.
            // For now, just use s32 conversion of (h * 2) — done in f16:
            __half h0 = __ushort_as_half(hlo);
            __half h1 = __ushort_as_half(hhi);
            int i0 = __half2int_rn(__hmul(h0, __float2half(2.0f)));
            int i1 = __half2int_rn(__hmul(h1, __float2half(2.0f)));
            partial += i0 + i1;
        }
        int rsum;
        asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;"
                     : "=r"(rsum) : "r"(partial));
        itotal += rsum;
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if ((int)ftotal == seed && itotal == seed) ((unsigned*)C)[blockIdx.x] = itotal;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d N_ITERS=%d clk=%llu cy/iter=%.3f ftotal=%.4f itotal=%d\n",
               MODE, N_ITERS, t1 - t0, (double)(t1-t0)/(double)N_ITERS, ftotal, itotal);
    }
}
