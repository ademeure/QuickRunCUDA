// Scalar FP16 FMA vs packed
#include <cuda_fp16.h>
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned short hv = 0x3C00 + (unsigned)u2;
    unsigned short hb = 0x3C01;
    unsigned int hv2 = 0x3C003C00u + (unsigned)u2;
    unsigned int hb2 = 0x3C013C01u;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // Scalar fp16 FMA
        asm("fma.rn.f16 %0, %0, %1, %0;" : "+h"(hv) : "h"(hb));
#elif MODE == 1
        // Packed f16x2 FMA (= 2 elements per inst)
        asm("fma.rn.f16x2 %0, %0, %1, %0;" : "+r"(hv2) : "r"(hb2));
#elif MODE == 2
        // Scalar bf16 FMA
        unsigned short bv = hv;
        asm("fma.rn.bf16 %0, %0, %1, %0;" : "+h"(bv) : "h"(hb));
        hv = bv;
#elif MODE == 3
        // Packed bf16x2 FMA
        asm("fma.rn.bf16x2 %0, %0, %1, %0;" : "+r"(hv2) : "r"(hb2));
#endif
    }

    if (hv == (unsigned short)seed && hv2 == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = hv2;
}
