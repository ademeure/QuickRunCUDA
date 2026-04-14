// Pipe topology matrix kernel. All 12 op classes as opt-in chains.
// Each chain is DCE-proof via per-chain-unique XOR/IADD feedback that mixes loop-dep state.
// Use N_<OP> = 0 to disable an op class entirely (no SASS emitted for it).
//
// Signature expected by QuickRunCUDA:
//   extern "C" __global__ void kernel(float*A, float*B, float*C, int arg0, int arg1, int arg2);
// arg0 = ITERS, arg1 = seed, arg2 unused.
//
// KEY DESIGN: For ops that are idempotent or self-canceling with constant operands
// (LOP3 xor, SHL by const, IADD3 add const), we mix in a per-unroll-iteration varying
// value derived from another chain's state so the compiler cannot CSE across the unrolled
// inner loop. We keep feedback LIGHT so companion op overhead is negligible.

#ifndef N_FFMA
#define N_FFMA 0
#endif
#ifndef N_FMUL
#define N_FMUL 0
#endif
#ifndef N_IMAD
#define N_IMAD 0
#endif
#ifndef N_IADD3
#define N_IADD3 0
#endif
#ifndef N_LOP3
#define N_LOP3 0
#endif
#ifndef N_SHL
#define N_SHL 0
#endif
#ifndef N_HFMA2
#define N_HFMA2 0
#endif
#ifndef N_HADD2
#define N_HADD2 0
#endif
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
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

#if N_FFMA > 0
    float ffma[N_FFMA];
    #pragma unroll
    for (int k = 0; k < N_FFMA; k++) ffma[k] = 1.0001f + 1e-5f * (tid + k * 23);
#endif
#if N_FMUL > 0
    float fmul[N_FMUL];
    #pragma unroll
    for (int k = 0; k < N_FMUL; k++) fmul[k] = 1.00003f + 1e-6f * (tid + k * 29);
#endif
#if N_IMAD > 0
    unsigned int imad[N_IMAD];
    #pragma unroll
    for (int k = 0; k < N_IMAD; k++) imad[k] = tid * 1315423911u + k * 0x9E3779B1u;
#endif
#if N_IADD3 > 0
    unsigned int iadd[N_IADD3];
    #pragma unroll
    for (int k = 0; k < N_IADD3; k++) iadd[k] = tid * 2654435761u + k * 0x85EBCA6Bu;
#endif
#if N_LOP3 > 0
    unsigned int lop3[N_LOP3];
    #pragma unroll
    for (int k = 0; k < N_LOP3; k++) lop3[k] = tid * 0xDEADBEEFu + k * 0xCAFEBABEu;
#endif
#if N_SHL > 0
    unsigned int shl[N_SHL];
    #pragma unroll
    for (int k = 0; k < N_SHL; k++) shl[k] = tid * 0xC2B2AE35u + k * 0x27D4EB2Fu;
#endif
#if N_HFMA2 > 0
    unsigned int hfma2[N_HFMA2];
    #pragma unroll
    for (int k = 0; k < N_HFMA2; k++) hfma2[k] = 0x3C003C00u ^ (tid * 131u + k);
#endif
#if N_HADD2 > 0
    unsigned int hadd2[N_HADD2];
    #pragma unroll
    for (int k = 0; k < N_HADD2; k++) hadd2[k] = 0x3C003C00u ^ (tid * 137u + k);
#endif
#if N_UNPACK > 0
    unsigned short unp[N_UNPACK];
    #pragma unroll
    for (int k = 0; k < N_UNPACK; k++) unp[k] = (unsigned short)((0x3C00 + k * 0x11) ^ (tid * 131 + k));
#endif
#if N_PACK > 0
    unsigned int pac[N_PACK];
    #pragma unroll
    for (int k = 0; k < N_PACK; k++) pac[k] = (0x3C003C01u + k * 0x100u) ^ (tid * 137u + k);
#endif
#if N_EX2 > 0
    float ex2v[N_EX2];
    #pragma unroll
    for (int k = 0; k < N_EX2; k++) ex2v[k] = 1.00001f + 1e-4f * (tid + k * 17);
#endif
#if N_RSQ > 0
    float rsqv[N_RSQ];
    #pragma unroll
    for (int k = 0; k < N_RSQ; k++) rsqv[k] = 2.00001f + 1e-4f * (tid + k * 19);
#endif

    // Opaque per-thread "spice" that tracks iteration index, used to prevent inner-loop CSE
    // on ops whose body is otherwise a constant op.
    unsigned int spice = tid | 1u;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            // Advance spice with a cheap unique step (not counted as a pipe op).
            // We DON'T count this toward any op; ensure it doesn't dominate.
            spice = spice * 1664525u + 1013904223u;

#if N_FFMA > 0
            #pragma unroll
            for (int k = 0; k < N_FFMA; k++) {
                asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                             : "+f"(ffma[k])
                             : "f"(1.0000001f + 1e-9f * k), "f"(0.9999999f - 1e-9f * k));
            }
#endif
#if N_FMUL > 0
            #pragma unroll
            for (int k = 0; k < N_FMUL; k++) {
                asm volatile("mul.rn.f32 %0, %0, %1;"
                             : "+f"(fmul[k])
                             : "f"(1.0000001f + 1e-9f * k));
            }
#endif
#if N_IMAD > 0
            #pragma unroll
            for (int k = 0; k < N_IMAD; k++) {
                asm volatile("mad.lo.u32 %0, %0, %1, %2;"
                             : "+r"(imad[k])
                             : "r"(1315423911u + k), "r"(k + 1u));
            }
#endif
#if N_IADD3 > 0
            // IADD3 with per-iter varying addend so compiler cannot fold
            #pragma unroll
            for (int k = 0; k < N_IADD3; k++) {
                asm volatile("add.u32 %0, %0, %1;" : "+r"(iadd[k]) : "r"(spice));
            }
#endif
#if N_LOP3 > 0
            // Per-iter varying XOR mask via spice
            #pragma unroll
            for (int k = 0; k < N_LOP3; k++) {
                asm volatile("xor.b32 %0, %0, %1;" : "+r"(lop3[k]) : "r"(spice));
            }
#endif
#if N_SHL > 0
            // For each chain, do SHL by variable amount (from spice & 0x1f). This prevents folding
            // the shift sequence into a constant multiplier.
            #pragma unroll
            for (int k = 0; k < N_SHL; k++) {
                asm volatile("shl.b32 %0, %0, %1;" : "+r"(shl[k]) : "r"(spice));
            }
#endif
#if N_HFMA2 > 0
            // Use spice as the 2nd/3rd operand to avoid compiler synthesizing constants via extra HFMA2s.
            #pragma unroll
            for (int k = 0; k < N_HFMA2; k++) {
                asm volatile("fma.rn.f16x2 %0, %0, %1, %2;"
                             : "+r"(hfma2[k]) : "r"(spice), "r"(spice));
            }
#endif
#if N_HADD2 > 0
            #pragma unroll
            for (int k = 0; k < N_HADD2; k++) {
                asm volatile("add.rn.f16x2 %0, %0, %1;"
                             : "+r"(hadd2[k]) : "r"(spice));
            }
#endif
#if N_UNPACK > 0
            #pragma unroll
            for (int k = 0; k < N_UNPACK; k++) {
                unsigned int tmp;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(tmp) : "h"(unp[k]));
                unp[k] = (unsigned short)(tmp ^ spice);
            }
#endif
#if N_PACK > 0
            #pragma unroll
            for (int k = 0; k < N_PACK; k++) {
                unsigned short tmp;
                asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(tmp) : "r"(pac[k]));
                pac[k] = ((unsigned int)tmp ^ spice);
            }
#endif
#if N_EX2 > 0
            #pragma unroll
            for (int k = 0; k < N_EX2; k++) {
                asm volatile("ex2.approx.f32 %0, %0;" : "+f"(ex2v[k]));
            }
#endif
#if N_RSQ > 0
            #pragma unroll
            for (int k = 0; k < N_RSQ; k++) {
                asm volatile("rsqrt.approx.f32 %0, %0;" : "+f"(rsqv[k]));
            }
#endif
        }
    }

    unsigned int acc = spice;
#if N_FFMA > 0
    #pragma unroll
    for (int k = 0; k < N_FFMA; k++) acc ^= __float_as_int(ffma[k]);
#endif
#if N_FMUL > 0
    #pragma unroll
    for (int k = 0; k < N_FMUL; k++) acc ^= __float_as_int(fmul[k]);
#endif
#if N_IMAD > 0
    #pragma unroll
    for (int k = 0; k < N_IMAD; k++) acc ^= imad[k];
#endif
#if N_IADD3 > 0
    #pragma unroll
    for (int k = 0; k < N_IADD3; k++) acc ^= iadd[k];
#endif
#if N_LOP3 > 0
    #pragma unroll
    for (int k = 0; k < N_LOP3; k++) acc ^= lop3[k];
#endif
#if N_SHL > 0
    #pragma unroll
    for (int k = 0; k < N_SHL; k++) acc ^= shl[k];
#endif
#if N_HFMA2 > 0
    #pragma unroll
    for (int k = 0; k < N_HFMA2; k++) acc ^= hfma2[k];
#endif
#if N_HADD2 > 0
    #pragma unroll
    for (int k = 0; k < N_HADD2; k++) acc ^= hadd2[k];
#endif
#if N_UNPACK > 0
    #pragma unroll
    for (int k = 0; k < N_UNPACK; k++) acc ^= (unsigned int)unp[k];
#endif
#if N_PACK > 0
    #pragma unroll
    for (int k = 0; k < N_PACK; k++) acc ^= pac[k];
#endif
#if N_EX2 > 0
    #pragma unroll
    for (int k = 0; k < N_EX2; k++) acc ^= __float_as_int(ex2v[k]);
#endif
#if N_RSQ > 0
    #pragma unroll
    for (int k = 0; k < N_RSQ; k++) acc ^= __float_as_int(rsqv[k]);
#endif
    if ((int)acc == seed) ((unsigned int*)C)[tid] = acc;
}
