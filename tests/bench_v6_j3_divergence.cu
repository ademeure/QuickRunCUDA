// V6 J3: Branch divergence cost
// MODE 0: All 32 threads same path (no divergence)
// MODE 1: 1 thread else (31 if + 1 else)
// MODE 2: 16 + 16 split
// MODE 3: 31 + 1 (worst case for the LAST thread)
// MODE 4: All 32 in else (no divergence, opposite path)
//
// Theoretical: warp divergence serializes. Each unique path costs 1 issue slot.
// 2-way diverge => 2× cost; if both branches are 8 FFMAs, total = ~16 cy/iter
// 1-way (uniform) = 8 cy/iter
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    float a = (float)(threadIdx.x + 1) * 0.001f;
    float b = (float)(threadIdx.x + 2) * 0.001f;

    // Both branches do same amount of work (8 FFMAs) for fair comparison
    float f0=a, f1=a, f2=a, f3=a, f4=a, f5=a, f6=a, f7=a;
    float k0=b*0.99f, k1=b*0.98f, k2=b*0.97f, k3=b*0.96f;
    float k4=b*0.95f, k5=b*0.94f, k6=b*0.93f, k7=b*0.92f;

    // (cond moved into loop to prevent compiler hoisting)

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // Compute cond INSIDE loop using i to defeat hoisting
#if MODE == 0
        bool cond = (i & 1);  // alternates per iter but uniform across warp
#elif MODE == 1
        bool cond = ((threadIdx.x + i) & 1) || (threadIdx.x != 0);  // 31 if, 1 else
#elif MODE == 2
        bool cond = ((threadIdx.x + i) & 16) == ((i & 16));  // 16/16 split varying
#elif MODE == 3
        bool cond = (threadIdx.x != ((i & 31)));  // exactly 1 lane else, varies
#elif MODE == 4
        bool cond = (i & 0);  // always false
#endif
        if (cond) {
            // Branch A: 8 FFMA chains
            f0 = f0 * k0 + b;
            f1 = f1 * k1 + b;
            f2 = f2 * k2 + b;
            f3 = f3 * k3 + b;
            f4 = f4 * k4 + b;
            f5 = f5 * k5 + b;
            f6 = f6 * k6 + b;
            f7 = f7 * k7 + b;
        } else {
            // Branch B: 8 FFMA chains (different mults to defeat compiler merge)
            f0 = f0 * k0 + a;
            f1 = f1 * k1 + a;
            f2 = f2 * k2 + a;
            f3 = f3 * k3 + a;
            f4 = f4 * k4 + a;
            f5 = f5 * k5 + a;
            f6 = f6 * k6 + a;
            f7 = f7 * k7 + a;
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    // Anti-DCE
    float fsum = f0+f1+f2+f3+f4+f5+f6+f7;
    if (fsum == 1.234567e-30f) C[blockIdx.x] = fsum;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d ITERS=%d cy/iter=%.3f\n",
               MODE, ITERS, (double)(t1-t0)/(double)ITERS);
    }
}
