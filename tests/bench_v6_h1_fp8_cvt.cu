// V6 H1: FP8 cvt latency comparison
// MODE 0: 8× cvt.rn.satfinite.e4m3x2.f32
// MODE 1: 8× cvt.rn.satfinite.e5m2x2.f32
// MODE 2: 8× cvt.rn.f16x2.f32 (baseline)
// Each cvt has DIFFERENT inputs to prevent CSE.
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    float v[16];
    #pragma unroll
    for (int j = 0; j < 16; j++) v[j] = (float)(threadIdx.x + j + 1) * 0.5f;

    unsigned short s[8] = {0};
    unsigned int r[8] = {0};

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // Make each iter's inputs different to prevent CSE
        float fi = (float)i;
#if MODE == 0
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(s[0]) : "f"(v[0]+fi), "f"(v[1]+fi));
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(s[1]) : "f"(v[2]+fi), "f"(v[3]+fi));
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(s[2]) : "f"(v[4]+fi), "f"(v[5]+fi));
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(s[3]) : "f"(v[6]+fi), "f"(v[7]+fi));
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(s[4]) : "f"(v[8]+fi), "f"(v[9]+fi));
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(s[5]) : "f"(v[10]+fi), "f"(v[11]+fi));
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(s[6]) : "f"(v[12]+fi), "f"(v[13]+fi));
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(s[7]) : "f"(v[14]+fi), "f"(v[15]+fi));
#elif MODE == 1
        asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(s[0]) : "f"(v[0]+fi), "f"(v[1]+fi));
        asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(s[1]) : "f"(v[2]+fi), "f"(v[3]+fi));
        asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(s[2]) : "f"(v[4]+fi), "f"(v[5]+fi));
        asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(s[3]) : "f"(v[6]+fi), "f"(v[7]+fi));
        asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(s[4]) : "f"(v[8]+fi), "f"(v[9]+fi));
        asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(s[5]) : "f"(v[10]+fi), "f"(v[11]+fi));
        asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(s[6]) : "f"(v[12]+fi), "f"(v[13]+fi));
        asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(s[7]) : "f"(v[14]+fi), "f"(v[15]+fi));
#elif MODE == 2
        asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r[0]) : "f"(v[0]+fi), "f"(v[1]+fi));
        asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r[1]) : "f"(v[2]+fi), "f"(v[3]+fi));
        asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r[2]) : "f"(v[4]+fi), "f"(v[5]+fi));
        asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r[3]) : "f"(v[6]+fi), "f"(v[7]+fi));
        asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r[4]) : "f"(v[8]+fi), "f"(v[9]+fi));
        asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r[5]) : "f"(v[10]+fi), "f"(v[11]+fi));
        asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r[6]) : "f"(v[12]+fi), "f"(v[13]+fi));
        asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r[7]) : "f"(v[14]+fi), "f"(v[15]+fi));
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    unsigned int ssum = 0; for (int j = 0; j < 8; j++) ssum ^= s[j];
    unsigned int rsum = 0; for (int j = 0; j < 8; j++) rsum ^= r[j];
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ((unsigned int*)C)[1] = ssum;
        ((unsigned int*)C)[2] = rsum;
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d ITERS=%d cy/iter=%.3f cy/cvt=%.3f\n",
               MODE, ITERS, (double)(t1-t0)/(double)ITERS,
               (double)(t1-t0)/(double)ITERS/8.0);
    }
}
