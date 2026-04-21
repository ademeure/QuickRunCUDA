// V7 M8: Test scheduler look-ahead / out-of-order execution
// Pattern: long-latency LDG followed by independent FFMA chain
// If scheduler reorders, FFMA chain should progress while LDG waits
// Compare back-to-back vs interleaved latency
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    float a = (float)(threadIdx.x + 1);
    float b = (float)(threadIdx.x + 2);
    float c0=a, c1=a, c2=a, c3=a, c4=a, c5=a, c6=a, c7=a;
    float k0=b*0.99f, k1=b*0.98f, k2=b*0.97f, k3=b*0.96f;
    float k4=b*0.95f, k5=b*0.94f, k6=b*0.93f, k7=b*0.92f;
    float ldg_result = 0;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // Long-latency LDG (cold cache line)
        float load_val = A[(i * 32 + threadIdx.x) & 0xFFFF];

#if MODE == 0
        // Serial: wait for LDG before FFMA
        ldg_result += load_val;
        c0 = c0 * k0 + ldg_result;  // FFMA depends on load
        c1 = c1 * k1 + c0;
        c2 = c2 * k2 + c1;
        c3 = c3 * k3 + c2;
#elif MODE == 1
        // Interleaved: FFMA chain independent of LDG — scheduler should overlap
        c0 = c0 * k0 + b;  // independent
        c1 = c1 * k1 + b;
        c2 = c2 * k2 + b;
        c3 = c3 * k3 + b;
        ldg_result += load_val;  // use LDG at end
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    float sum = c0+c1+c2+c3+ldg_result;
    if (sum == 1.234567e-30f) C[blockIdx.x] = sum;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d cy/iter=%.3f\n", MODE, (double)(t1-t0)/(double)ITERS);
    }
}
