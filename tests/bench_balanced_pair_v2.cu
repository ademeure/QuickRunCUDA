// Full-topology balanced pair kernel: add LOP3/IADD3/IMAD/HFMA2/SHL/FMUL.
// Uses per-chain unique XOR feedback to prevent CSE.
// All chains should be SASS-independent. Verify with sass_count.

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
#ifndef N_FMUL
#define N_FMUL 0
#endif
#ifndef N_LOP3
#define N_LOP3 0
#endif
#ifndef N_IADD3
#define N_IADD3 0
#endif
#ifndef N_IMAD
#define N_IMAD 0
#endif
#ifndef N_SHL
#define N_SHL 0
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
#if N_UNPACK > 0
    unsigned short u_h[N_UNPACK];
    #pragma unroll
    for (int k = 0; k < N_UNPACK; k++) u_h[k] = (unsigned short)((0x3C00 + k*0x11) ^ (threadIdx.x*131 + k));
#endif
#if N_PACK > 0
    unsigned int p_u[N_PACK];
    #pragma unroll
    for (int k = 0; k < N_PACK; k++) p_u[k] = (0x3C003C01u + k*0x100) ^ (threadIdx.x*137 + k);
#endif
#if N_EX2 > 0
    float e_f[N_EX2];
    #pragma unroll
    for (int k = 0; k < N_EX2; k++) e_f[k] = 1.0001f + 0.0001f*(threadIdx.x + k*17);
#endif
#if N_RSQ > 0
    float r_f[N_RSQ];
    #pragma unroll
    for (int k = 0; k < N_RSQ; k++) r_f[k] = 2.0001f + 0.0001f*(threadIdx.x + k*19);
#endif
#if N_FFMA > 0
    float ffma_f[N_FFMA];
    #pragma unroll
    for (int k = 0; k < N_FFMA; k++) ffma_f[k] = 1.0001f + 0.0001f*(threadIdx.x + k*23);
#endif
#if N_FMUL > 0
    float fmul_f[N_FMUL];
    #pragma unroll
    for (int k = 0; k < N_FMUL; k++) fmul_f[k] = 1.001f + 0.0001f*(threadIdx.x + k*29);
#endif
#if N_LOP3 > 0
    unsigned int lop3_u[N_LOP3];
    #pragma unroll
    for (int k = 0; k < N_LOP3; k++) lop3_u[k] = 0xDEAD0000u + (threadIdx.x + k*31);
#endif
#if N_IADD3 > 0
    unsigned int iadd3_u[N_IADD3];
    #pragma unroll
    for (int k = 0; k < N_IADD3; k++) iadd3_u[k] = 0xBEEF0000u + (threadIdx.x + k*37);
#endif
#if N_IMAD > 0
    unsigned int imad_u[N_IMAD];
    #pragma unroll
    for (int k = 0; k < N_IMAD; k++) imad_u[k] = (threadIdx.x + k*41 + 1);
#endif
#if N_SHL > 0
    unsigned int shl_u[N_SHL];
    #pragma unroll
    for (int k = 0; k < N_SHL; k++) shl_u[k] = 0x1u << (k % 16);
#endif
#if N_HFMA2 > 0
    unsigned int hfma2_u[N_HFMA2];
    #pragma unroll
    for (int k = 0; k < N_HFMA2; k++) hfma2_u[k] = 0x3C003C01u ^ (threadIdx.x + k*43);
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
            for (int k = 0; k < N_EX2; k++) asm volatile("ex2.approx.f32 %0, %0;" : "+f"(e_f[k]));
#endif
#if N_RSQ > 0
            #pragma unroll
            for (int k = 0; k < N_RSQ; k++) asm volatile("rsqrt.approx.f32 %0, %0;" : "+f"(r_f[k]));
#endif
#if N_FFMA > 0
            #pragma unroll
            for (int k = 0; k < N_FFMA; k++) {
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(ffma_f[k]) : "f"(1.000001f), "f"(0.9999f));
            }
#endif
#if N_FMUL > 0
            #pragma unroll
            for (int k = 0; k < N_FMUL; k++) {
                asm volatile("mul.rn.f32 %0, %0, %1;" : "+f"(fmul_f[k]) : "f"(1.0000001f));
            }
#endif
#if N_LOP3 > 0
            #pragma unroll
            for (int k = 0; k < N_LOP3; k++) {
                // Chain-unique XOR constant to avoid folding
                asm volatile("xor.b32 %0, %0, %1;" : "+r"(lop3_u[k]) : "r"((unsigned)(0xAAAAAAABu + k*17)));
            }
#endif
#if N_IADD3 > 0
            #pragma unroll
            for (int k = 0; k < N_IADD3; k++) {
                asm volatile("add.u32 %0, %0, %1;" : "+r"(iadd3_u[k]) : "r"((unsigned)(k + 1)));
            }
#endif
#if N_IMAD > 0
            #pragma unroll
            for (int k = 0; k < N_IMAD; k++) {
                asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(imad_u[k]) : "r"((unsigned)3), "r"((unsigned)(k+1)));
            }
#endif
#if N_SHL > 0
            #pragma unroll
            for (int k = 0; k < N_SHL; k++) {
                asm volatile("shl.b32 %0, %0, 1;" : "+r"(shl_u[k]));
            }
#endif
#if N_HFMA2 > 0
            #pragma unroll
            for (int k = 0; k < N_HFMA2; k++) {
                asm volatile("fma.rn.f16x2 %0, %0, %1, %1;" : "+r"(hfma2_u[k]) : "r"((unsigned)(0x3C003C00u + k)));
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
    for (int k = 0; k < N_FFMA; k++) acc ^= __float_as_int(ffma_f[k]);
#endif
#if N_FMUL > 0
    #pragma unroll
    for (int k = 0; k < N_FMUL; k++) acc ^= __float_as_int(fmul_f[k]);
#endif
#if N_LOP3 > 0
    #pragma unroll
    for (int k = 0; k < N_LOP3; k++) acc ^= lop3_u[k];
#endif
#if N_IADD3 > 0
    #pragma unroll
    for (int k = 0; k < N_IADD3; k++) acc ^= iadd3_u[k];
#endif
#if N_IMAD > 0
    #pragma unroll
    for (int k = 0; k < N_IMAD; k++) acc ^= imad_u[k];
#endif
#if N_SHL > 0
    #pragma unroll
    for (int k = 0; k < N_SHL; k++) acc ^= shl_u[k];
#endif
#if N_HFMA2 > 0
    #pragma unroll
    for (int k = 0; k < N_HFMA2; k++) acc ^= hfma2_u[k];
#endif
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
