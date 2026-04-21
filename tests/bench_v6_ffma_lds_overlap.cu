// V6 A4: FFMA (fma pipe) + LDS (lsu pipe) overlap test
// MODE 0: 8× FFMA (ILP=8)
// MODE 1: 8× LDS (different addresses for diversity)
// MODE 2: combined
// Test theory: if LDS queues through L1TEX (doesn't block issue), then ~73% overlap; else ~30%
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    __shared__ __align__(16) float smem[1024];
    if (threadIdx.x == 0) {
        for (int i = 0; i < 1024; i++) smem[i] = (float)i;
    }
    __syncwarp();

    float a = (float)(threadIdx.x + 1) * 0.001f;
    float b = (float)(threadIdx.x + 2) * 0.001f;
    float f0=a, f1=a, f2=a, f3=a, f4=a, f5=a, f6=a, f7=a;
    float k0=b*0.99f, k1=b*0.98f, k2=b*0.97f, k3=b*0.96f;
    float k4=b*0.95f, k5=b*0.94f, k6=b*0.93f, k7=b*0.92f;

    float load0 = 0;
    int idx = threadIdx.x;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0 || MODE == 2
        // 8 FFMA, ILP=8
        f0 = f0 * k0 + b;
        f1 = f1 * k1 + b;
        f2 = f2 * k2 + b;
        f3 = f3 * k3 + b;
        f4 = f4 * k4 + b;
        f5 = f5 * k5 + b;
        f6 = f6 * k6 + b;
        f7 = f7 * k7 + b;
#endif
#if MODE == 1 || MODE == 2
        // 8 LDS
        #pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            float v;
            unsigned int sa = (unsigned int)__cvta_generic_to_shared(&smem[(idx + kk * 32 + i) & 1023]);
            asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(sa));
            load0 += v;
        }
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

#if MODE == 0 || MODE == 2
    float fsum = f0+f1+f2+f3+f4+f5+f6+f7;
    if (fsum == 1.234567e-30f) C[blockIdx.x] = fsum;
#endif
#if MODE == 1 || MODE == 2
    if (load0 == 1.234567e-30f) ((float*)C)[blockIdx.x + 1] = load0;
#endif

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d ITERS=%d total_cy=%llu cy/iter=%.3f\n",
               MODE, ITERS, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
