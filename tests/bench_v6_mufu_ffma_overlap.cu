// V6 A_extra: MUFU.RCP (xu) + FFMA (fma) overlap
// Both compute pipes — both supposedly block scheduler issue
// MODE 0: 8× MUFU
// MODE 1: 8× FFMA
// MODE 2: combined
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    float a = (float)(threadIdx.x + 1) * 0.001f;
    float b = (float)(threadIdx.x + 2) * 0.001f;

    float r0=a, r1=a*1.1f, r2=a*1.2f, r3=a*1.3f;
    float r4=a*1.4f, r5=a*1.5f, r6=a*1.6f, r7=a*1.7f;

    float f0=a, f1=a, f2=a, f3=a, f4=a, f5=a, f6=a, f7=a;
    float k0=b*0.99f, k1=b*0.98f, k2=b*0.97f, k3=b*0.96f;
    float k4=b*0.95f, k5=b*0.94f, k6=b*0.93f, k7=b*0.92f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0 || MODE == 2
        asm volatile("rcp.approx.f32 %0, %0;" : "+f"(r0));
        asm volatile("rcp.approx.f32 %0, %0;" : "+f"(r1));
        asm volatile("rcp.approx.f32 %0, %0;" : "+f"(r2));
        asm volatile("rcp.approx.f32 %0, %0;" : "+f"(r3));
        asm volatile("rcp.approx.f32 %0, %0;" : "+f"(r4));
        asm volatile("rcp.approx.f32 %0, %0;" : "+f"(r5));
        asm volatile("rcp.approx.f32 %0, %0;" : "+f"(r6));
        asm volatile("rcp.approx.f32 %0, %0;" : "+f"(r7));
#endif
#if MODE == 1 || MODE == 2
        f0 = f0 * k0 + b;
        f1 = f1 * k1 + b;
        f2 = f2 * k2 + b;
        f3 = f3 * k3 + b;
        f4 = f4 * k4 + b;
        f5 = f5 * k5 + b;
        f6 = f6 * k6 + b;
        f7 = f7 * k7 + b;
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

#if MODE == 0 || MODE == 2
    float rsum = r0+r1+r2+r3+r4+r5+r6+r7;
    if (rsum == 1.234567e-30f) C[blockIdx.x] = rsum;
#endif
#if MODE == 1 || MODE == 2
    float fsum = f0+f1+f2+f3+f4+f5+f6+f7;
    if (fsum == 1.234567e-30f) ((float*)C)[blockIdx.x+1] = fsum;
#endif

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d ITERS=%d total_cy=%llu cy/iter=%.3f\n",
               MODE, ITERS, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
