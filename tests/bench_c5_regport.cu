// C5 Register-port hypothesis: FFMA2 + UNPACK gives u=1.67 vs FFMA2+PRMT u=1.95.
// Hypothesis (user 2026-04-24): F2FP UNPACK exceeds SMSP RF read-port budget per
// cycle when paired with FFMA2; PRMT may stay within the limit due to operand
// reuse / different register read pattern.
//
// We test FFMA2:OP = 1:1 ratio (same as catalog claim) with:
//   FFMA2 form: fma.rn.f32x2 d, a, b, c  (3 distinct sources per inst; 6 sub-reads via lo/hi pair)
//
// Variants (set with -DVARIANT=N):
//
//   0 = FFMA2 alone (baseline; hits ~2.0 inst/SM/cy = full pipe)
//   1 = FFMA2 + F2FP.UNPACK_B (DISTINCT src per F2FP)         <-- catalog "1.67"
//   2 = FFMA2 + PRMT.b32 (2 srcs + imm selector)              <-- catalog "1.95"
//   3 = F2FP.UNPACK_B alone
//   4 = PRMT alone
//   5 = FFMA2 + F2FP, F2FP source SAME as one of FFMA2 operands (port sharing test)
//   6 = FFMA2 + F2FP_lo using already-issued register (reuse test)
//   7 = FFMA2 + 2× F2FP per FFMA2  (over-pressure: should slow further)
//   8 = FFMA2 + LOP3 (catalog "free" co-issue)                <-- control
//   9 = FFMA2 + IADD3 (1 src + imm)
//  10 = FFMA2 + F2FP where F2FP src is loop-invariant (compiler should mark .reuse)
//  11 = F2FP + PRMT 1:1 (no FFMA2; compares two ALU ops directly)

#ifndef VARIANT
#define VARIANT 0
#endif
#ifndef N_PAIRS
#define N_PAIRS 8
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

    // FFMA2 state: N_PAIRS distinct fp32x2 accumulators, with 2 distinct multiplicands
    unsigned long long f[N_PAIRS];
    unsigned long long b_[N_PAIRS];
    unsigned long long c_[N_PAIRS];
    #pragma unroll
    for (int k = 0; k < N_PAIRS; k++) {
        unsigned int alo = __float_as_int(1.0001f + 0.0001f*(threadIdx.x + k*23));
        unsigned int ahi = __float_as_int(1.0002f + 0.0001f*(threadIdx.x + k*29));
        unsigned int blo = __float_as_int(0.9999f + 0.0001f*(threadIdx.x + k*7));
        unsigned int bhi = __float_as_int(0.9998f + 0.0001f*(threadIdx.x + k*11));
        unsigned int clo = __float_as_int(1.000001f + 0.0000001f*k);
        unsigned int chi = __float_as_int(1.000002f + 0.0000001f*k);
        f[k]  = ((unsigned long long)ahi << 32) | alo;
        b_[k] = ((unsigned long long)bhi << 32) | blo;
        c_[k] = ((unsigned long long)chi << 32) | clo;
    }

    // ALU state: N_PAIRS distinct registers for unpack/PRMT
    unsigned int u[N_PAIRS];
    unsigned int v[N_PAIRS];   // second source for PRMT
    #pragma unroll
    for (int k = 0; k < N_PAIRS; k++) {
        u[k] = 0x3C003C01u ^ (threadIdx.x * 137 + k * 23);
        v[k] = 0xCAFEBABEu ^ (threadIdx.x * 71  + k * 19);
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_PAIRS; k++) {

#if VARIANT == 0
                // FFMA2 alone — 3 distinct sources per inst (a, b, c)
                asm volatile("fma.rn.f32x2 %0, %0, %1, %2;" : "+l"(f[k]) : "l"(b_[k]), "l"(c_[k]));

#elif VARIANT == 1
                // FFMA2 + F2FP UNPACK_B (distinct dest), 1:1
                asm volatile("fma.rn.f32x2 %0, %0, %1, %2;" : "+l"(f[k]) : "l"(b_[k]), "l"(c_[k]));
                unsigned int tmp;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(tmp) : "h"((unsigned short)u[k]));
                u[k] = tmp;

#elif VARIANT == 2
                // FFMA2 + PRMT, 1:1
                asm volatile("fma.rn.f32x2 %0, %0, %1, %2;" : "+l"(f[k]) : "l"(b_[k]), "l"(c_[k]));
                asm volatile("prmt.b32 %0, %0, %1, 0x7531;" : "+r"(u[k]) : "r"(v[k]));

#elif VARIANT == 3
                // F2FP UNPACK_B alone
                unsigned int tmp;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(tmp) : "h"((unsigned short)u[k]));
                u[k] = tmp;

#elif VARIANT == 4
                // PRMT alone
                asm volatile("prmt.b32 %0, %0, %1, 0x7531;" : "+r"(u[k]) : "r"(v[k]));

#elif VARIANT == 5
                // FFMA2 + F2FP — F2FP source is THE SAME register as FFMA2's first operand
                // (low half). Tests whether sharing a source register helps.
                asm volatile("fma.rn.f32x2 %0, %0, %1, %2;" : "+l"(f[k]) : "l"(b_[k]), "l"(c_[k]));
                unsigned int tmp;
                // hijack low 16 bits of f[k] (.lo is hi when LE long-long; just need *some*
                // bits from the FFMA2 dest to make srcs collide at scheduler view)
                unsigned short low_h;
                asm volatile("cvt.u16.u64 %0, %1;" : "=h"(low_h) : "l"(f[k]));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(tmp) : "h"(low_h));
                u[k] = tmp;

#elif VARIANT == 6
                // FFMA2 + F2FP — F2FP source set to a value just produced by FFMA2 in *previous*
                // iteration. Simulates close producer-consumer + exposes .reuse opportunity.
                asm volatile("fma.rn.f32x2 %0, %0, %1, %2;" : "+l"(f[k]) : "l"(b_[k]), "l"(c_[k]));
                unsigned int tmp;
                unsigned short fbits = (unsigned short)((unsigned int)(f[k] & 0xFFFFu) ^ u[k]);
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(tmp) : "h"(fbits));
                u[k] = tmp;

#elif VARIANT == 7
                // FFMA2 + 2× F2FP per FFMA2
                asm volatile("fma.rn.f32x2 %0, %0, %1, %2;" : "+l"(f[k]) : "l"(b_[k]), "l"(c_[k]));
                unsigned int tmp1, tmp2;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(tmp1) : "h"((unsigned short)u[k]));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(tmp2) : "h"((unsigned short)v[k]));
                u[k] = tmp1; v[k] = tmp2;

#elif VARIANT == 8
                // FFMA2 + LOP3 (3 src, well-known free co-issue)
                asm volatile("fma.rn.f32x2 %0, %0, %1, %2;" : "+l"(f[k]) : "l"(b_[k]), "l"(c_[k]));
                asm volatile("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(u[k]) : "r"(v[k]), "r"(u[(k+1)%N_PAIRS]));

#elif VARIANT == 9
                // FFMA2 + IADD3 (only 1 src + 0 imm; minimal RF reads on ALU side)
                asm volatile("fma.rn.f32x2 %0, %0, %1, %2;" : "+l"(f[k]) : "l"(b_[k]), "l"(c_[k]));
                asm volatile("add.u32 %0, %0, %1;" : "+r"(u[k]) : "r"(v[k]));

#elif VARIANT == 10
                // FFMA2 + F2FP, F2FP src is loop-invariant scalar from u2 (compiler should
                // hoist + mark .reuse aggressively).
                asm volatile("fma.rn.f32x2 %0, %0, %1, %2;" : "+l"(f[k]) : "l"(b_[k]), "l"(c_[k]));
                unsigned int tmp;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(tmp) : "h"((unsigned short)u2));
                u[k] ^= tmp;

#elif VARIANT == 11
                // F2FP + PRMT 1:1 (no FFMA2; both on ALU pipe — pure pipe contention)
                unsigned int tmp;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(tmp) : "h"((unsigned short)u[k]));
                u[k] = tmp;
                asm volatile("prmt.b32 %0, %0, %1, 0x7531;" : "+r"(v[k]) : "r"(u[k]));
#endif
            }
        }
    }

    // Anti-DCE
    unsigned long long acc = 0;
    #pragma unroll
    for (int k = 0; k < N_PAIRS; k++) {
        acc ^= f[k];
        acc ^= (unsigned long long)u[k];
        acc ^= (unsigned long long)v[k];
    }
    if (acc == (unsigned long long)seed)
        ((unsigned long long*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
