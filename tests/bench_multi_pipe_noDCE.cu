// Multi-pipe kernel with GUARANTEED-to-execute ops.
// Each "chain" has its output feeding back as the next input via asm volatile
// with "+r" / "+f" constraints, which forces every instance to emit.
// SASS-verified.

#ifndef N_CVT
#define N_CVT 8
#endif
#ifndef N_FFMA
#define N_FFMA 0
#endif
#ifndef N_IADD
#define N_IADD 0
#endif
#ifndef N_LOP3
#define N_LOP3 0
#endif
#ifndef N_HFMA2
#define N_HFMA2 0
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
    // Each companion gets its own feedback chain — each output becomes next input,
    // so compiler can't fold anything (each iteration's result differs).
    unsigned short h_cvt[N_CVT];
    unsigned int p_cvt[N_CVT];
    #pragma unroll
    for (int k = 0; k < N_CVT; k++) h_cvt[k] = 0x3C01 ^ (threadIdx.x + k);

#if N_FFMA > 0
    float f_ffma[N_FFMA];
    #pragma unroll
    for (int k = 0; k < N_FFMA; k++) f_ffma[k] = 1.0001f + 0.0001f * (threadIdx.x + k);
#endif
#if N_IADD > 0
    unsigned int x_iadd[N_IADD];
    #pragma unroll
    for (int k = 0; k < N_IADD; k++) x_iadd[k] = 0xBEEF0000u ^ (threadIdx.x + k);
#endif
#if N_LOP3 > 0
    unsigned int x_lop3[N_LOP3];
    #pragma unroll
    for (int k = 0; k < N_LOP3; k++) x_lop3[k] = 0xDEAD0000u ^ (threadIdx.x + k);
#endif
#if N_HFMA2 > 0
    unsigned int x_hfma2[N_HFMA2];
    #pragma unroll
    for (int k = 0; k < N_HFMA2; k++) x_hfma2[k] = 0x3C003C01u ^ (threadIdx.x + k);
#endif

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            // F2FP unpack — feedback via truncate (zero-cost)
            #pragma unroll
            for (int k = 0; k < N_CVT; k++) {
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(p_cvt[k]) : "h"(h_cvt[k]));
                h_cvt[k] = (unsigned short)p_cvt[k];
            }
#if N_FFMA > 0
            // FFMA — feedback via self-accumulate with constants
            #pragma unroll
            for (int k = 0; k < N_FFMA; k++) {
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f_ffma[k]) : "f"(1.00001f), "f"(0.999f));
            }
#endif
#if N_IADD > 0
            // IADD — feedback XOR-add self with a thread-dep constant
            #pragma unroll
            for (int k = 0; k < N_IADD; k++) {
                asm volatile("add.u32 %0, %0, %0;" : "+r"(x_iadd[k]));
                // self-add creates a chain that CAN'T be simplified to x<<1 because PTX doesn't see constant
                // Oh wait it can — simpler to use "add.u32 %0, %0, 1"
            }
            #pragma unroll
            for (int k = 0; k < N_IADD; k++) {
                asm volatile("add.u32 %0, %0, 1;" : "+r"(x_iadd[k]));
            }
#endif
#if N_LOP3 > 0
            #pragma unroll
            for (int k = 0; k < N_LOP3; k++) {
                asm volatile("xor.b32 %0, %0, 0xAAAAAAAA;" : "+r"(x_lop3[k]));
            }
#endif
#if N_HFMA2 > 0
            // HFMA2 — f16x2 multiply-add self
            #pragma unroll
            for (int k = 0; k < N_HFMA2; k++) {
                asm volatile("fma.rn.f16x2 %0, %0, %1, %1;" : "+r"(x_hfma2[k]) : "r"((unsigned int)0x3C003C00u));
            }
#endif
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CVT; k++) acc ^= p_cvt[k];
#if N_FFMA > 0
    #pragma unroll
    for (int k = 0; k < N_FFMA; k++) acc ^= __float_as_int(f_ffma[k]);
#endif
#if N_IADD > 0
    #pragma unroll
    for (int k = 0; k < N_IADD; k++) acc ^= x_iadd[k];
#endif
#if N_LOP3 > 0
    #pragma unroll
    for (int k = 0; k < N_LOP3; k++) acc ^= x_lop3[k];
#endif
#if N_HFMA2 > 0
    #pragma unroll
    for (int k = 0; k < N_HFMA2; k++) acc ^= x_hfma2[k];
#endif
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
