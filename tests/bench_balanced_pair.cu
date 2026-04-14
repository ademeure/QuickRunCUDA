// Balanced-ILP pair test with GUARANTEED-independent register chains.
// Uses distinct register arrays for each op and ensures compiler can't CSE.
// For UNPACK: distinct XOR seeds; chains are truly independent.
//
// Designed specifically to measure pipe-sharing for 2 ops in balanced demand.

#ifndef N_UNPACK
#define N_UNPACK 0
#endif
#ifndef N_PACK
#define N_PACK 0
#endif
#ifndef N_EX2
#define N_EX2 0
#endif
#ifndef N_RSQ
#define N_RSQ 0
#endif
#ifndef N_FFMA
#define N_FFMA 0
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 2
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // Each chain uses a DISTINCT seed constant mixed with tid, and feedback
    // with a chain-specific XOR of 0xAB0001, 0xAB0002, ... so the compiler
    // can NEVER merge chains (each has unique loop-dependent trajectory).

#if N_UNPACK > 0
    unsigned short u_h[N_UNPACK];
    #pragma unroll
    for (int k = 0; k < N_UNPACK; k++)
        u_h[k] = (unsigned short)((0x3C00 + k * 0x11) ^ (threadIdx.x * 131 + k));
#endif
#if N_PACK > 0
    unsigned int p_u[N_PACK];
    #pragma unroll
    for (int k = 0; k < N_PACK; k++)
        p_u[k] = ((0x3C003C01u + k * 0x100) ^ (threadIdx.x * 137 + k));
#endif
#if N_EX2 > 0
    float e_f[N_EX2];
    #pragma unroll
    for (int k = 0; k < N_EX2; k++)
        e_f[k] = 1.00001f + 0.0001f * (threadIdx.x + k * 17);
#endif
#if N_RSQ > 0
    float r_f[N_RSQ];
    #pragma unroll
    for (int k = 0; k < N_RSQ; k++)
        r_f[k] = 2.00001f + 0.0001f * (threadIdx.x + k * 19);
#endif
#if N_FFMA > 0
    float f_f[N_FFMA];
    #pragma unroll
    for (int k = 0; k < N_FFMA; k++)
        f_f[k] = 1.0001f + 0.0001f * (threadIdx.x + k * 23);
#endif

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if N_UNPACK > 0
            #pragma unroll
            for (int k = 0; k < N_UNPACK; k++) {
                unsigned int tmp;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(tmp) : "h"(u_h[k]));
                // Chain-specific XOR prevents CSE
                u_h[k] = (unsigned short)(tmp ^ (0xAB00u + k));
            }
#endif
#if N_PACK > 0
            #pragma unroll
            for (int k = 0; k < N_PACK; k++) {
                unsigned short tmp;
                asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(tmp) : "r"(p_u[k]));
                p_u[k] = ((unsigned int)tmp ^ (0xCD000000u + k));
            }
#endif
#if N_EX2 > 0
            #pragma unroll
            for (int k = 0; k < N_EX2; k++) {
                asm volatile("ex2.approx.f32 %0, %0;" : "+f"(e_f[k]));
            }
#endif
#if N_RSQ > 0
            #pragma unroll
            for (int k = 0; k < N_RSQ; k++) {
                asm volatile("rsqrt.approx.f32 %0, %0;" : "+f"(r_f[k]));
            }
#endif
#if N_FFMA > 0
            #pragma unroll
            for (int k = 0; k < N_FFMA; k++) {
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f_f[k]) : "f"(1.000001f), "f"(0.9999f));
            }
#endif
        }
    }

    unsigned int acc = 0;
#if N_UNPACK > 0
    #pragma unroll
    for (int k = 0; k < N_UNPACK; k++) acc ^= (unsigned int)u_h[k];
#endif
#if N_PACK > 0
    #pragma unroll
    for (int k = 0; k < N_PACK; k++) acc ^= p_u[k];
#endif
#if N_EX2 > 0
    #pragma unroll
    for (int k = 0; k < N_EX2; k++) acc ^= __float_as_int(e_f[k]);
#endif
#if N_RSQ > 0
    #pragma unroll
    for (int k = 0; k < N_RSQ; k++) acc ^= __float_as_int(r_f[k]);
#endif
#if N_FFMA > 0
    #pragma unroll
    for (int k = 0; k < N_FFMA; k++) acc ^= __float_as_int(f_f[k]);
#endif
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
