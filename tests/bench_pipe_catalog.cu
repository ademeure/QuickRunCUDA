// Pipe-catalog: solo-peak for many ops, to map each to its pipe via ncu.
// Feedback pattern per op is chosen to NOT contaminate other pipes.
// N_CHAINS × UNROLL asm volatiles per thread; SASS static count must match.

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
    // Storage depends on op type; pick widest (u64) for FP32 pair / FFMA2 etc.
    unsigned long long w[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        unsigned int lo = __float_as_int(1.0001f + 0.0001f*(threadIdx.x + k*23));
        unsigned int hi = __float_as_int(1.0002f + 0.0001f*(threadIdx.x + k*29));
        w[k] = ((unsigned long long)hi << 32) | lo;
    }
    unsigned int c1_u = __float_as_int(1.000001f);
    unsigned int c0_u = __float_as_int(0.9999f);
    unsigned long long c1_64 = ((unsigned long long)c1_u << 32) | c1_u;
    unsigned long long c0_64 = ((unsigned long long)c0_u << 32) | c0_u;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
#if OP == 0   // PRMT (pipe_alu baseline)
                unsigned int next = (unsigned int)w[(k+1) & (N_CHAINS-1)];
                unsigned int cur = (unsigned int)w[k];
                asm volatile("prmt.b32 %0, %0, %1, 0x3210;" : "+r"(cur) : "r"(next));
                w[k] = (w[k] & 0xFFFFFFFF00000000ULL) | cur;

#elif OP == 1  // FFMA scalar (pipe_fma)
                float fv = __int_as_float((unsigned int)w[k]);
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(1.000001f), "f"(0.9999f));
                w[k] = (w[k] & 0xFFFFFFFF00000000ULL) | __float_as_int(fv);

#elif OP == 2  // FFMA2 (vec2 FP32 FMA)
                asm volatile("fma.rn.f32x2 %0, %0, %1, %2;" : "+l"(w[k]) : "l"(c1_64), "l"(c0_64));

#elif OP == 3  // IMAD 32-bit (suspected pipe_fmaheavy)
                unsigned int x = (unsigned int)w[k];
                asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(x) : "r"(3u), "r"(1u));
                w[k] = (w[k] & 0xFFFFFFFF00000000ULL) | x;

#elif OP == 4  // IADD3 (add 3 operands)
                unsigned int x = (unsigned int)w[k];
                unsigned int y = (unsigned int)w[(k+1) & (N_CHAINS-1)];
                asm volatile("add.u32 %0, %0, %1;" : "+r"(x) : "r"(y));
                w[k] = (w[k] & 0xFFFFFFFF00000000ULL) | x;

#elif OP == 5  // SHL
                unsigned int x = (unsigned int)w[k];
                asm volatile("shl.b32 %0, %0, 1;" : "+r"(x));
                w[k] = (w[k] & 0xFFFFFFFF00000000ULL) | x;

#elif OP == 6  // POPC (popcount)
                unsigned int x = (unsigned int)w[k];
                unsigned int y;
                asm volatile("popc.b32 %0, %1;" : "=r"(y) : "r"(x));
                w[k] = (w[k] & 0xFFFFFFFF00000000ULL) | (x + y);  // keep both live

#elif OP == 7  // HADD2 (FP16 pair add)
                unsigned int x = (unsigned int)w[k];
                asm volatile("add.rn.f16x2 %0, %0, %1;" : "+r"(x) : "r"(0x3C003C00u));
                w[k] = (w[k] & 0xFFFFFFFF00000000ULL) | x;

#elif OP == 8  // HMUL2 (FP16 pair mul)
                unsigned int x = (unsigned int)w[k];
                asm volatile("mul.rn.f16x2 %0, %0, %1;" : "+r"(x) : "r"(0x3C003C00u));
                w[k] = (w[k] & 0xFFFFFFFF00000000ULL) | x;

#elif OP == 9  // HFMA2 (FP16 pair FMA)
                unsigned int x = (unsigned int)w[k];
                asm volatile("fma.rn.f16x2 %0, %0, %1, %1;" : "+r"(x) : "r"(0x3C003C00u));
                w[k] = (w[k] & 0xFFFFFFFF00000000ULL) | x;

#elif OP == 10  // MUFU.EX2
                float fv = __int_as_float((unsigned int)w[k]) * 0.5f + 0.25f;  // bound input
                asm volatile("ex2.approx.f32 %0, %0;" : "+f"(fv));
                w[k] = (w[k] & 0xFFFFFFFF00000000ULL) | __float_as_int(fv);

#elif OP == 11  // MUFU.RSQ
                float fv = __int_as_float((unsigned int)w[k]) * 0.5f + 1.0f;
                asm volatile("rsqrt.approx.f32 %0, %0;" : "+f"(fv));
                w[k] = (w[k] & 0xFFFFFFFF00000000ULL) | __float_as_int(fv);

#elif OP == 12  // F2I int conversion
                float fv = __int_as_float((unsigned int)w[k]);
                int x;
                asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(x) : "f"(fv));
                w[k] = (w[k] & 0xFFFFFFFF00000000ULL) | (unsigned int)x;

#elif OP == 13  // I2F
                int x = (int)(unsigned int)w[k];
                float fv;
                asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(fv) : "r"(x));
                w[k] = (w[k] & 0xFFFFFFFF00000000ULL) | __float_as_int(fv);

#elif OP == 14  // F2F half-float conversion (f16 -> f32)
                unsigned int x = (unsigned int)w[k];
                float fv;
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(fv) : "h"((unsigned short)x));
                w[k] = (w[k] & 0xFFFFFFFF00000000ULL) | __float_as_int(fv);

#elif OP == 15  // FADD (scalar FP32 add)
                float fv = __int_as_float((unsigned int)w[k]);
                asm volatile("add.rn.f32 %0, %0, %1;" : "+f"(fv) : "f"(0.00001f));
                w[k] = (w[k] & 0xFFFFFFFF00000000ULL) | __float_as_int(fv);

#elif OP == 16  // FMUL scalar
                float fv = __int_as_float((unsigned int)w[k]);
                asm volatile("mul.rn.f32 %0, %0, %1;" : "+f"(fv) : "f"(1.0000001f));
                w[k] = (w[k] & 0xFFFFFFFF00000000ULL) | __float_as_int(fv);

#elif OP == 17  // IMAD.WIDE 32x32->64
                unsigned int x = (unsigned int)w[k];
                unsigned long long y = w[k];
                asm volatile("mad.wide.u32 %0, %1, %2, %0;" : "+l"(y) : "r"(x), "r"(3u));
                w[k] = y;

#elif OP == 18  // F2FP UNPACK (pipe_alu reference)
                unsigned int tmp;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(tmp) : "h"((unsigned short)w[k]));
                w[k] = (w[k] & 0xFFFFFFFF00000000ULL) | tmp;
#endif
            }
        }
    }

    unsigned long long acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= w[k];
    if (acc == (unsigned long long)seed) ((unsigned long long*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
