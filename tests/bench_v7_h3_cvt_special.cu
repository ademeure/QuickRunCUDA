// V7 H3: cvt FP exception handling (overflow/inf/NaN)
// Test how narrow-FP cvt handles special inputs
extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x != 0) return;

    // Test special FP32 → FP8 e4m3 conversions
    union { float f; unsigned int u; } u;

    // (a) Normal value
    u.f = 1.0f;
    unsigned short s_normal;
    float src_a = u.f, src_b = u.f;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(s_normal) : "f"(src_a), "f"(src_b));

    // (b) +Inf
    u.u = 0x7F800000;
    float inf_v = u.f;
    unsigned short s_inf;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(s_inf) : "f"(inf_v), "f"(inf_v));

    // (c) -Inf
    u.u = 0xFF800000;
    float ninf_v = u.f;
    unsigned short s_ninf;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(s_ninf) : "f"(ninf_v), "f"(ninf_v));

    // (d) NaN
    u.u = 0x7FC00000;
    float nan_v = u.f;
    unsigned short s_nan;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(s_nan) : "f"(nan_v), "f"(nan_v));

    // (e) Overflow value (far above FP8 max)
    u.f = 1e30f;
    float over_v = u.f;
    unsigned short s_over;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(s_over) : "f"(over_v), "f"(over_v));

    // (f) Underflow value (subnormal)
    u.f = 1e-30f;
    float under_v = u.f;
    unsigned short s_under;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(s_under) : "f"(under_v), "f"(under_v));

    // (g) Zero
    unsigned short s_zero;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(s_zero) : "f"(0.0f), "f"(0.0f));

    if (blockIdx.x == 0) {
        printf("FP32 → FP8 e4m3 special-case conversions:\n");
        printf("  1.0     → 0x%04x\n", (unsigned)s_normal);
        printf("  +Inf    → 0x%04x  (saturates to FP8 max?)\n", (unsigned)s_inf);
        printf("  -Inf    → 0x%04x  (saturates to FP8 -max?)\n", (unsigned)s_ninf);
        printf("  NaN     → 0x%04x  (FP8 NaN?)\n", (unsigned)s_nan);
        printf("  1e30    → 0x%04x  (overflow → max)\n", (unsigned)s_over);
        printf("  1e-30   → 0x%04x  (underflow → 0?)\n", (unsigned)s_under);
        printf("  0.0     → 0x%04x\n", (unsigned)s_zero);
    }
}
