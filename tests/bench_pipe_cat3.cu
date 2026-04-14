// Catalog v3 — clean kernels (u32 storage, no 64-bit merge LOP3 tax).
// Split by data-type so each op runs solo without ALU contamination.

#ifndef N_CHAINS
#define N_CHAINS 8
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
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // For FP32 / INT u32 chains — store as u32 directly
    unsigned int v[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) v[k] = __float_as_int(1.0001f + 0.0001f*(threadIdx.x + k*17));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                unsigned int x = v[k];
                unsigned int nxt = v[(k+1) & (N_CHAINS-1)];  // runtime data

#if OP == 0   // FFMA scalar — clean, no 64-bit tax
                float fv = __int_as_float(x);
                float fn = __int_as_float(nxt);
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(1.000001f), "f"(fn));
                x = __float_as_int(fv);
#elif OP == 1  // FMUL scalar
                float fv = __int_as_float(x);
                float fn = __int_as_float(nxt);
                asm volatile("mul.rn.f32 %0, %0, %1;" : "+f"(fv) : "f"(fn));
                x = __float_as_int(fv);
#elif OP == 2  // FADD scalar
                float fv = __int_as_float(x);
                float fn = __int_as_float(nxt);
                asm volatile("add.rn.f32 %0, %0, %1;" : "+f"(fv) : "f"(fn));
                x = __float_as_int(fv);
#elif OP == 3  // FMIN with data-dep RHS — no fold
                float fv = __int_as_float(x);
                float fn = __int_as_float(nxt);
                asm volatile("min.f32 %0, %0, %1;" : "+f"(fv) : "f"(fn));
                x = __float_as_int(fv);
#elif OP == 4  // FMAX data-dep
                float fv = __int_as_float(x);
                float fn = __int_as_float(nxt);
                asm volatile("max.f32 %0, %0, %1;" : "+f"(fv) : "f"(fn));
                x = __float_as_int(fv);
#elif OP == 5  // FMIN.NaN (propagating)
                float fv = __int_as_float(x);
                float fn = __int_as_float(nxt);
                asm volatile("min.NaN.f32 %0, %0, %1;" : "+f"(fv) : "f"(fn));
                x = __float_as_int(fv);
#elif OP == 6  // FABS data-dep via sub
                float fv = __int_as_float(x) - __int_as_float(nxt);
                asm volatile("abs.f32 %0, %0;" : "+f"(fv));
                x = __float_as_int(fv);
#elif OP == 7  // FNEG data-dep
                float fv = __int_as_float(x) + __int_as_float(nxt);
                asm volatile("neg.f32 %0, %0;" : "+f"(fv));
                x = __float_as_int(fv);
#elif OP == 8  // MUFU.RCP
                float fv = __int_as_float(x) * 0.5f + 0.25f;
                asm volatile("rcp.approx.f32 %0, %0;" : "+f"(fv));
                x = __float_as_int(fv);
#elif OP == 9  // MUFU.SQRT
                float fv = __int_as_float(x) * 0.5f + 1.0f;
                asm volatile("sqrt.approx.f32 %0, %0;" : "+f"(fv));
                x = __float_as_int(fv);
#elif OP == 10  // MUFU.EX2 (re-test clean)
                float fv = __int_as_float(x) * 0.0001f + 0.1f;
                asm volatile("ex2.approx.f32 %0, %0;" : "+f"(fv));
                x = __float_as_int(fv);
#elif OP == 11  // XMAD (legacy) — skipped, use IMAD.HI
                asm volatile("mul.hi.u32 %0, %0, %1;" : "+r"(x) : "r"(nxt));
#elif OP == 12  // IADD carry-in chain
                asm volatile("add.cc.u32 %0, %0, %1;" : "+r"(x) : "r"(nxt));
#elif OP == 13  // IMUL.LO scalar (just mul, no add)
                asm volatile("mul.lo.u32 %0, %0, %1;" : "+r"(x) : "r"(nxt));
#elif OP == 14  // HADD2 data-dep f16x2
                asm volatile("add.rn.f16x2 %0, %0, %1;" : "+r"(x) : "r"(nxt));
#elif OP == 15  // HMIN f16x2 data-dep
                asm volatile("min.f16x2 %0, %0, %1;" : "+r"(x) : "r"(nxt));
#elif OP == 16  // HMAX f16x2 data-dep
                asm volatile("max.f16x2 %0, %0, %1;" : "+r"(x) : "r"(nxt));
#elif OP == 17  // HMNMX2 Nan (propagating)
                asm volatile("max.NaN.f16x2 %0, %0, %1;" : "+r"(x) : "r"(nxt));
#elif OP == 18  // BF16 FMA pair
                asm volatile("fma.rn.bf16x2 %0, %0, %1, %1;" : "+r"(x) : "r"(nxt));
#elif OP == 19  // BF16 ADD pair
                asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(x) : "r"(nxt));
#elif OP == 20  // BF16 MIN pair
                asm volatile("min.bf16x2 %0, %0, %1;" : "+r"(x) : "r"(nxt));
#elif OP == 21  // COPYSIGN
                asm volatile("copysign.f32 %0, %1, %0;" : "+f"(*(float*)&x) : "f"(*(float*)&nxt));
#elif OP == 22  // CVT.sat.u8.f32
                asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "+r"(x) : "f"(*(float*)&x));
#elif OP == 23  // CVT.RN.f16.f32  (F2F narrow)
                asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(*(unsigned short*)&x) : "f"(*(float*)&nxt));
#elif OP == 24  // CVT.RN.bf16.f32
                asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(*(unsigned short*)&x) : "f"(*(float*)&nxt));
#elif OP == 25  // vote.ballot
                asm volatile("{ .reg .pred p; setp.ne.u32 p, %0, 0; vote.sync.ballot.b32 %0, p, 0xFFFFFFFF; }" : "+r"(x));
#elif OP == 26  // match.any.sync
                asm volatile("match.any.sync.b32 %0, %0, 0xFFFFFFFF;" : "+r"(x));
#endif
                v[k] = x;
            }
        }
    }
    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= v[k];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
