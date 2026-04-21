// Compare NVFP4 decode methods: LUT vs arithmetic vs hardware cvt.
// Each thread decodes 8 E2M1 values (1 dword), produces sum of float values.
// MODE 0: LUT-based (original approach)
// MODE 1: arithmetic decode (avoid LUT)
// MODE 2: hardware cvt.rn.f16x2.e2m1x2 (4 cvt insts → 8 f16 → 8 f32 sum)
// MODE 3: hardware cvt + hadd2 (avoid f16→f32 conversion)

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

    float facc = 0.0f;
    unsigned int dword = Au[threadIdx.x];

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        // Per-iter perturbation in lower bits (defeats hoisting)
        unsigned int d = dword ^ ((unsigned)u2 * (unsigned)it);

#if MODE == 0
        // LUT-based decode (8 elements)
        static const int lut[8] = {0, 1, 2, 3, 4, 6, 8, 12};
        float partial = 0.0f;
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            unsigned int code = (d >> (n*4)) & 0xF;
            int mag = lut[code & 0x7];
            int signed_val = (code & 0x8) ? -mag : mag;
            partial += (float)signed_val * 0.5f;
        }
        facc += partial;
#elif MODE == 1
        // Arithmetic decode (no LUT)
        // E2M1: sign|exp|exp|mant; mag×2 = (mant+2)<<exp if exp>0 else mant
        float partial = 0.0f;
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            unsigned int code = (d >> (n*4)) & 0xF;
            unsigned int sign = code >> 3;
            unsigned int exp  = (code >> 1) & 0x3;
            unsigned int mant = code & 0x1;
            unsigned int mag = (exp != 0) ? ((mant + 2u) << exp) : mant;
            int signed_val = sign ? -(int)mag : (int)mag;
            partial += (float)signed_val * 0.5f;
        }
        facc += partial;
#elif MODE == 2
        // Hardware cvt: split dword into 4 bytes, cvt each → f16x2, sum
        float partial = 0.0f;
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            unsigned short byte = (unsigned short)((d >> (n*8)) & 0xFF);
            unsigned int hpair;
            asm volatile("{ .reg .b8 _b, _p; mov.b16 {_b,_p}, %1; cvt.rn.f16x2.e2m1x2 %0, _b; }"
                         : "=r"(hpair) : "h"(byte));
            __half h0 = __ushort_as_half((unsigned short)(hpair & 0xFFFF));
            __half h1 = __ushort_as_half((unsigned short)(hpair >> 16));
            partial += __half2float(h0) + __half2float(h1);
        }
        facc += partial;
#elif MODE == 3
        // cvt + hadd2 packed sum (stay in f16 longer)
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
        facc += __half2float(__ushort_as_half(lo16)) + __half2float(__ushort_as_half(hi16));
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if ((int)facc == seed) ((unsigned*)C)[blockIdx.x] = (unsigned)facc;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d N_ITERS=%d clk=%llu cy/iter=%.3f facc=%.4f\n",
               MODE, N_ITERS, t1 - t0, (double)(t1-t0)/(double)N_ITERS, facc);
    }
}
