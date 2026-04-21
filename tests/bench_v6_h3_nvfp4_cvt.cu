// V6 H3: NVFP4 cvt latency
// MODE 0: cvt FP32 → e2m1x4 (NVFP4, packed 4 in b16)
// MODE 1: cvt FP32 → e2m1x2 (packed 2 in b8)
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

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        float fi = (float)i;
#if MODE == 0
        // NVFP4 e2m1x4 (4 values packed in b16)
        // PTX: cvt.rn.satfinite.e2m1x4.f32 %dst, {%a, %b, %c, %d};
        asm volatile("cvt.rn.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4};"
                     : "=h"(s[0]) : "f"(v[0]+fi), "f"(v[1]+fi), "f"(v[2]+fi), "f"(v[3]+fi));
        asm volatile("cvt.rn.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4};"
                     : "=h"(s[1]) : "f"(v[4]+fi), "f"(v[5]+fi), "f"(v[6]+fi), "f"(v[7]+fi));
        asm volatile("cvt.rn.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4};"
                     : "=h"(s[2]) : "f"(v[8]+fi), "f"(v[9]+fi), "f"(v[10]+fi), "f"(v[11]+fi));
        asm volatile("cvt.rn.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4};"
                     : "=h"(s[3]) : "f"(v[12]+fi), "f"(v[13]+fi), "f"(v[14]+fi), "f"(v[15]+fi));
#elif MODE == 1
        // FP4 e2m1x2 packed-2
        asm volatile("cvt.rn.satfinite.e2m1x2.f32 %0, %1, %2;" : "=h"(s[0]) : "f"(v[0]+fi), "f"(v[1]+fi));
        asm volatile("cvt.rn.satfinite.e2m1x2.f32 %0, %1, %2;" : "=h"(s[1]) : "f"(v[2]+fi), "f"(v[3]+fi));
        asm volatile("cvt.rn.satfinite.e2m1x2.f32 %0, %1, %2;" : "=h"(s[2]) : "f"(v[4]+fi), "f"(v[5]+fi));
        asm volatile("cvt.rn.satfinite.e2m1x2.f32 %0, %1, %2;" : "=h"(s[3]) : "f"(v[6]+fi), "f"(v[7]+fi));
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    unsigned int ssum = 0; for (int j = 0; j < 8; j++) ssum ^= s[j];
    if (threadIdx.x == 0 && blockIdx.x == 0) ((unsigned int*)C)[1] = ssum;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int n_cvts = (MODE == 0) ? 4 : 4;  // 4 ops per iter either way
        printf("MODE=%d ITERS=%d cy/iter=%.3f cy/cvt=%.3f cy/scalar=%.3f\n",
               MODE, ITERS, (double)(t1-t0)/(double)ITERS,
               (double)(t1-t0)/(double)ITERS/(double)n_cvts,
               (double)(t1-t0)/(double)ITERS/(double)(MODE == 0 ? 16 : 8));
    }
}
