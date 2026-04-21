// V6 A_extra2: FFMA + IADD3 — both compute pipes, both block issue
// PREDICTION: NEAR-ZERO overlap (combined ≈ sum of individuals)
// MODE 0: 8× FFMA only
// MODE 1: 8× IADD3 only
// MODE 2: combined
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    float a = (float)(threadIdx.x + 1) * 0.001f;
    float b = (float)(threadIdx.x + 2) * 0.001f;
    float f0=a, f1=a, f2=a, f3=a, f4=a, f5=a, f6=a, f7=a;
    float k0=b*0.99f, k1=b*0.98f, k2=b*0.97f, k3=b*0.96f;
    float k4=b*0.95f, k5=b*0.94f, k6=b*0.93f, k7=b*0.92f;

    int x = threadIdx.x + 1;
    int kk = blockIdx.x + 1;
    int i0=x, i1=x, i2=x, i3=x, i4=x, i5=x, i6=x, i7=x;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0 || MODE == 2
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
        asm volatile("add.s32 %0, %0, %1;" : "+r"(i0) : "r"(kk+i));
        asm volatile("add.s32 %0, %0, %1;" : "+r"(i1) : "r"(kk+i+1));
        asm volatile("add.s32 %0, %0, %1;" : "+r"(i2) : "r"(kk+i+2));
        asm volatile("add.s32 %0, %0, %1;" : "+r"(i3) : "r"(kk+i+3));
        asm volatile("add.s32 %0, %0, %1;" : "+r"(i4) : "r"(kk+i+4));
        asm volatile("add.s32 %0, %0, %1;" : "+r"(i5) : "r"(kk+i+5));
        asm volatile("add.s32 %0, %0, %1;" : "+r"(i6) : "r"(kk+i+6));
        asm volatile("add.s32 %0, %0, %1;" : "+r"(i7) : "r"(kk+i+7));
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

#if MODE == 0 || MODE == 2
    float fsum = f0+f1+f2+f3+f4+f5+f6+f7;
    if (fsum == 1.234567e-30f) C[blockIdx.x] = fsum;
#endif
#if MODE == 1 || MODE == 2
    int isum = i0+i1+i2+i3+i4+i5+i6+i7;
    if (isum == 0xCAFEBABE) ((int*)C)[blockIdx.x+1] = isum;
#endif

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d ITERS=%d total_cy=%llu cy/iter=%.3f\n",
               MODE, ITERS, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
