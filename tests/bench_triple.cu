// Triple-op contention: FFMA2 (pipe_fma), UNPACK (pipe_alu), LOP3 (pipe_alu).
// LOP3 uses a loop-carried RUNTIME mask `s` that LCG-updates between every
// asm. This prevents the compiler from fusing multiple XORs into a 3-input
// LOP3.LUT (the 0x66 collapse problem). The mask update is a single IMAD
// (fmaheavy pipe) per LOP3 — which contaminates fmaheavy measurement but
// leaves pipe_alu measurement of LOP3 clean.

#ifndef N_FFMA2
#define N_FFMA2 0
#endif
#ifndef N_UNPACK
#define N_UNPACK 0
#endif
#ifndef N_LOP3
#define N_LOP3 0
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
#if N_FFMA2 > 0
    unsigned long long f[N_FFMA2];
    #pragma unroll
    for (int k = 0; k < N_FFMA2; k++) {
        unsigned int ulo = __float_as_int(1.0001f + 0.0001f*(threadIdx.x + k*23));
        unsigned int uhi = __float_as_int(1.0002f + 0.0001f*(threadIdx.x + k*29));
        f[k] = ((unsigned long long)uhi << 32) | ulo;
    }
    unsigned int c1_u = __float_as_int(1.000001f);
    unsigned int c0_u = __float_as_int(0.9999f);
    unsigned long long c1 = ((unsigned long long)c1_u << 32) | c1_u;
    unsigned long long c0 = ((unsigned long long)c0_u << 32) | c0_u;
#endif
#if N_UNPACK > 0
    unsigned int u[N_UNPACK];
    #pragma unroll
    for (int k = 0; k < N_UNPACK; k++) u[k] = 0x3C003C01u ^ (threadIdx.x * 137 + k * 23);
#endif
#if N_LOP3 > 0
    unsigned int l[N_LOP3];
    #pragma unroll
    for (int k = 0; k < N_LOP3; k++) l[k] = 0xDEAD0000u + (threadIdx.x * 131 + k * 17);
    // Runtime mask initialised from seed so compiler cannot compute at compile time.
    unsigned int s = (unsigned int)(seed ^ (threadIdx.x * 0xdeadbeefu));
#endif

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if N_FFMA2 > 0
            #pragma unroll
            for (int k = 0; k < N_FFMA2; k++) {
                asm volatile("fma.rn.f32x2 %0, %0, %1, %2;" : "+l"(f[k]) : "l"(c1), "l"(c0));
            }
#endif
#if N_UNPACK > 0
            #pragma unroll
            for (int k = 0; k < N_UNPACK; k++) {
                unsigned int tmp;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(tmp) : "h"((unsigned short)u[k]));
                u[k] = tmp;
            }
#endif
#if N_LOP3 > 0
            #pragma unroll
            for (int k = 0; k < N_LOP3; k++) {
                asm volatile("xor.b32 %0, %0, %1;" : "+r"(l[k]) : "r"(s));
                // Update s so next asm sees a different mask — prevents fusion.
                s = s * 0x5DEECE66Du + 0xBu;
            }
#endif
        }
    }

    unsigned long long acc = 0;
#if N_FFMA2 > 0
    #pragma unroll
    for (int k = 0; k < N_FFMA2; k++) acc ^= f[k];
#endif
#if N_UNPACK > 0
    #pragma unroll
    for (int k = 0; k < N_UNPACK; k++) acc ^= (unsigned long long)u[k];
#endif
#if N_LOP3 > 0
    #pragma unroll
    for (int k = 0; k < N_LOP3; k++) acc ^= (unsigned long long)l[k];
    acc ^= s;  // keep s live
#endif
    if (acc == (unsigned long long)seed) ((unsigned long long*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
