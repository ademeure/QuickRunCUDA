// Extended chain pair latency: includes FP16, mixed precision, MUFU.
// OP codes:
//   0 = FFMA (f32 chain)
//   1 = IMAD (i32 chain)
//   2 = LOP3 (b32 chain)
//   3 = HFMA2 (f16x2 chain)
//   4 = HMUL2 (f16x2 mul, no add)
//   5 = HFMA2.F32 (f16x2 inputs → f32 accum, like HMMA prep)
//   6 = CVT_F16_F32 (f32 → f16)
//   7 = CVT_F32_F16 (f16 → f32)
//   8 = MUFU.RCP (f32 reciprocal)
//   9 = MUFU.EX2 (f32 exp2)
//  10 = MUFU.RSQRT (f32 rsqrt)

#ifndef N_INNER
#define N_INNER 1000
#endif
#ifndef N_OUTER
#define N_OUTER 100
#endif
#ifndef OP_A
#define OP_A 0
#endif
#ifndef OP_B
#define OP_B 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    // Multiple representations of same chain register — bridges between domains.
    unsigned int v32 = (unsigned)threadIdx.x + 1u;
    unsigned int b32 = 0xDEADBEEFu;
    unsigned int c32 = 0xC0FFEE00u;

    float ff = (float)threadIdx.x + 1.0f;
    float fb = 1.000001f;
    float fc = 0.000001f;

    unsigned int hh = 0x3C003C00u;  // {1.0, 1.0} f16x2
    unsigned int hb = 0x3C013C01u;  // ~1.0 + epsilon
    unsigned int hc = 0x38003800u;  // {0.5, 0.5}

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // Helper: do OP and return updated chain val into v
    #define DO_OP(op, vint, vfloat, vh) do { \
        if (op == 0) asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(vfloat) : "f"(fb), "f"(fc)); \
        else if (op == 1) asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(vint) : "r"(b32), "r"(c32)); \
        else if (op == 2) asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(vint) : "r"(b32), "r"(c32)); \
        else if (op == 3) asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(vh) : "r"(hb), "r"(hc)); \
        else if (op == 4) asm volatile("mul.rn.f16x2 %0, %0, %1;" : "+r"(vh) : "r"(hb)); \
        else if (op == 5) asm volatile("fma.rn.f32x2 %0, %0, %1, %2;" : "+f"(vfloat) : "f"(fb), "f"(fc)); \
        else if (op == 6) asm volatile("cvt.rn.satfinite.f16x2.f32 %0, %2, %1;" : "=r"(vh) : "f"(vfloat), "f"(vfloat)); \
        else if (op == 7) asm volatile("{ .reg .f16 _t; mov.b16 _t, %1; cvt.f32.f16 %0, _t; }" : "=f"(vfloat) : "h"((unsigned short)(vh & 0xFFFF))); \
        else if (op == 8) asm volatile("rcp.approx.f32 %0, %0;" : "+f"(vfloat)); \
        else if (op == 9) asm volatile("ex2.approx.f32 %0, %0;" : "+f"(vfloat)); \
        else if (op == 10) asm volatile("rsqrt.approx.f32 %0, %0;" : "+f"(vfloat)); \
    } while (0)

    #pragma unroll 1
    for (int i = 0; i < N_OUTER; i++) {
        #pragma unroll N_INNER
        for (int j = 0; j < N_INNER; j++) {
            DO_OP(OP_A, v32, ff, hh);
            DO_OP(OP_B, v32, ff, hh);
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v32 == (unsigned)seed && (int)ff == seed && (int)hh == seed) ((unsigned*)C)[blockIdx.x] = v32;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long total = (unsigned long long)N_OUTER * (unsigned long long)N_INNER;
        printf("OP_A=%2d OP_B=%2d clk=%llu cy/pair=%.4f\n",
               OP_A, OP_B, t1 - t0, (double)(t1-t0)/(double)total);
    }
}
