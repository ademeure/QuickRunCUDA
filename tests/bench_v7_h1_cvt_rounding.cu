// V7 H1: cvt rounding mode latency for narrow FP (FP8 e4m3)
// MODE 0: cvt.rn (round to nearest, default)
// MODE 1: cvt.rz (round toward zero)
// MODE 2: cvt.rm (round toward -inf)
// MODE 3: cvt.rp (round toward +inf)
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
#define RND "rn"
#elif MODE == 1
#define RND "rz"
#elif MODE == 2
#define RND "rm"
#elif MODE == 3
#define RND "rp"
#endif
        // Note: only rn supports satfinite for e4m3x2; others may not. Try without satfinite.
        asm volatile("cvt." RND ".satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(s[0]) : "f"(v[0]+fi), "f"(v[1]+fi));
        asm volatile("cvt." RND ".satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(s[1]) : "f"(v[2]+fi), "f"(v[3]+fi));
        asm volatile("cvt." RND ".satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(s[2]) : "f"(v[4]+fi), "f"(v[5]+fi));
        asm volatile("cvt." RND ".satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(s[3]) : "f"(v[6]+fi), "f"(v[7]+fi));
        asm volatile("cvt." RND ".satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(s[4]) : "f"(v[8]+fi), "f"(v[9]+fi));
        asm volatile("cvt." RND ".satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(s[5]) : "f"(v[10]+fi), "f"(v[11]+fi));
        asm volatile("cvt." RND ".satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(s[6]) : "f"(v[12]+fi), "f"(v[13]+fi));
        asm volatile("cvt." RND ".satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(s[7]) : "f"(v[14]+fi), "f"(v[15]+fi));
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    unsigned int ssum = 0; for (int j = 0; j < 8; j++) ssum ^= s[j];
    if (threadIdx.x == 0 && blockIdx.x == 0) ((unsigned int*)C)[1] = ssum;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d (rounding=%s) cy/iter=%.3f cy/cvt=%.3f\n",
               MODE, RND, (double)(t1-t0)/(double)ITERS, (double)(t1-t0)/(double)ITERS/8.0);
    }
}
