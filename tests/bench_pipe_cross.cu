// Cross-pipe contention tests — verify which ops share a pipe
// by using BASE_OP + COMPANION_OP with independent registers.
//
// BASE_OP: the op being measured
//   0 = FFMA.f32
//   1 = IMAD.u32
//   2 = FMUL.f32
//   3 = LOP3.xor.b32
//   4 = IADD3.u32
//   5 = SHL.b32
//   6 = MUFU.ex2.approx.f32
//   7 = F2FP.unpack (cvt.rn.f16x2.e4m3x2)
//
// COMPANION_OP: same options as BASE_OP

#ifndef BASE_OP
#define BASE_OP 0
#endif
#ifndef COMP_OP
#define COMP_OP 0
#endif
#ifndef N_BASE
#define N_BASE 4
#endif
#ifndef N_COMP
#define N_COMP 0
#endif
#ifndef UNROLL
#define UNROLL 32
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

#define EMIT_OP(OP, reg_f, reg_u) do { \
    if (OP == 0)      asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(reg_f) : "f"(1.00001f), "f"(0.999f)); \
    else if (OP == 1) asm volatile("mad.lo.u32 %0, %0, 3, 1;" : "+r"(reg_u)); \
    else if (OP == 2) asm volatile("mul.rn.f32 %0, %0, %1;" : "+f"(reg_f) : "f"(1.00001f)); \
    else if (OP == 3) asm volatile("xor.b32 %0, %0, 0xAAAAAAAA;" : "+r"(reg_u)); \
    else if (OP == 4) asm volatile("add.u32 %0, %0, 1;" : "+r"(reg_u)); \
    else if (OP == 5) asm volatile("shl.b32 %0, %0, 1;" : "+r"(reg_u)); \
    else if (OP == 6) asm volatile("ex2.approx.f32 %0, %0;" : "+f"(reg_f)); \
    else if (OP == 7) { unsigned int tmp; asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(tmp) : "h"((unsigned short)reg_u)); reg_u = tmp; } \
} while(0)

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int unused_2) {
    float base_f[N_BASE];
    unsigned int base_u[N_BASE];
    #pragma unroll
    for (int k = 0; k < N_BASE; k++) {
        base_f[k] = 1.0001f + (float)(threadIdx.x + k) * 0.0001f;
        base_u[k] = 0x3C013C01u ^ (threadIdx.x + k);
    }

#if N_COMP > 0
    float comp_f[N_COMP];
    unsigned int comp_u[N_COMP];
    #pragma unroll
    for (int m = 0; m < N_COMP; m++) {
        comp_f[m] = 2.0001f + (float)(threadIdx.x + m) * 0.0001f;
        comp_u[m] = 0x3C023C02u ^ (threadIdx.x + m);
    }
#endif

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_BASE; k++) {
                EMIT_OP(BASE_OP, base_f[k], base_u[k]);
            }
#if N_COMP > 0
            #pragma unroll
            for (int m = 0; m < N_COMP; m++) {
                EMIT_OP(COMP_OP, comp_f[m], comp_u[m]);
            }
#endif
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_BASE; k++) acc ^= __float_as_int(base_f[k]) ^ base_u[k];
#if N_COMP > 0
    #pragma unroll
    for (int m = 0; m < N_COMP; m++) acc ^= __float_as_int(comp_f[m]) ^ comp_u[m];
#endif
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
