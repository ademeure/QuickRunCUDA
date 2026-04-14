// Triple-pipe saturation: run FFMA + F2FP.unpack + LOP3 simultaneously.
// If all three pipes are truly independent, aggregate thread-op rate should be
// near FFMA_peak + F2FP_peak + LOP3_peak = 128 + 64 + ~128 = 320/SM/clk.
// Uses completely independent register chains.

#ifndef N_FFMA
#define N_FFMA 8
#endif
#ifndef N_F2FP
#define N_F2FP 4
#endif
#ifndef N_LOP3
#define N_LOP3 8
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
#if N_FFMA > 0
    float ffma_r[N_FFMA];
    #pragma unroll
    for (int k = 0; k < N_FFMA; k++) ffma_r[k] = 1.0001f + (float)(threadIdx.x + k) * 0.0001f;
    const float ca = 1.00001f, cb = 0.9999f;
#endif

#if N_F2FP > 0
    unsigned short f2fp_h[N_F2FP];
    unsigned int f2fp_p[N_F2FP];
    #pragma unroll
    for (int k = 0; k < N_F2FP; k++) f2fp_h[k] = 0x3C01 ^ (threadIdx.x + k);
#endif

#if N_LOP3 > 0
    unsigned int lop3_x[N_LOP3];
    #pragma unroll
    for (int k = 0; k < N_LOP3; k++) lop3_x[k] = 0xDEADBEEF ^ (threadIdx.x + k);
#endif

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if N_FFMA > 0
            #pragma unroll
            for (int k = 0; k < N_FFMA; k++) {
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(ffma_r[k]) : "f"(ca), "f"(cb));
            }
#endif
#if N_F2FP > 0
            #pragma unroll
            for (int k = 0; k < N_F2FP; k++) {
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f2fp_p[k]) : "h"(f2fp_h[k]));
                f2fp_h[k] = (unsigned short)f2fp_p[k];
            }
#endif
#if N_LOP3 > 0
            #pragma unroll
            for (int k = 0; k < N_LOP3; k++) {
                asm volatile("xor.b32 %0, %0, %1;" : "+r"(lop3_x[k]) : "r"(i+j+k));
            }
#endif
        }
    }
    unsigned int acc = 0;
#if N_FFMA > 0
    #pragma unroll
    for (int k = 0; k < N_FFMA; k++) acc ^= __float_as_int(ffma_r[k]);
#endif
#if N_F2FP > 0
    #pragma unroll
    for (int k = 0; k < N_F2FP; k++) acc ^= f2fp_p[k];
#endif
#if N_LOP3 > 0
    #pragma unroll
    for (int k = 0; k < N_LOP3; k++) acc ^= lop3_x[k];
#endif
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
