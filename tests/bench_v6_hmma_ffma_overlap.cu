// V6: HMMA (tensor pipe) + FFMA (fma pipe) overlap test
// MODE 0: 4× HMMA only
// MODE 1: 8× FFMA only (8 chains for ILP saturation)
// MODE 2: 4× HMMA + 8× FFMA combined (overlap test)
// Prediction: 90%+ overlap (different pipes)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    // HMMA inputs
    unsigned int a0=0x3F803F80, a1=0x3F803F80, a2=0x3F803F80, a3=0x3F803F80;
    unsigned int b0=0x3F803F80, b1=0x3F803F80;
    float c0=0.0f, c1=0.0f, c2=0.0f, c3=0.0f;

    // FFMA inputs (8 chains for ILP=8)
    float a = (float)(threadIdx.x + 1) * 0.001f;
    float b = (float)(threadIdx.x + 2) * 0.001f;
    float f0=a, f1=a, f2=a, f3=a, f4=a, f5=a, f6=a, f7=a;
    float k0=b*0.99f, k1=b*0.98f, k2=b*0.97f, k3=b*0.96f;
    float k4=b*0.95f, k5=b*0.94f, k6=b*0.93f, k7=b*0.92f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0 || MODE == 2
        // 4 HMMA
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
#endif
#if MODE == 1 || MODE == 2
        // 8 FFMA (1 per chain — 8 ILP)
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

    // Anti-DCE
#if MODE == 0 || MODE == 2
    if (c0 == 1.234567e-30f) C[blockIdx.x] = c0 + c1 + c2 + c3;
#endif
#if MODE == 1 || MODE == 2
    float fsum = f0+f1+f2+f3+f4+f5+f6+f7;
    if (fsum == 1.234567e-30f) ((float*)C)[blockIdx.x + 1] = fsum;
#endif

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d ITERS=%d total_cy=%llu cy/iter=%.3f\n",
               MODE, ITERS, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
