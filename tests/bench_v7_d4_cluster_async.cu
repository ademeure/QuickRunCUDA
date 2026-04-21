// V7 D4: Async cluster.barrier — arrive only, do work, wait later
// MODE 0: arrive then immediately wait (sync)
// MODE 1: arrive, do work, then wait (async — overlap)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(128, 1)
__cluster_dims__(4, 1, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float a = (float)(threadIdx.x + 1) * 0.001f;
    float b = (float)(threadIdx.x + 2) * 0.001f;
    float c0=a, c1=a, c2=a, c3=a;
    float k0=b*0.99f, k1=b*0.98f, k2=b*0.97f, k3=b*0.96f;

    unsigned long long t0, t1;
    if (blockIdx.x == 0 && threadIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // Sync — arrive + immediate wait
        asm volatile("barrier.cluster.arrive.aligned;");
        asm volatile("barrier.cluster.wait.aligned;");
        // Then compute
        c0 = c0 * k0 + b;
        c1 = c1 * k1 + b;
        c2 = c2 * k2 + b;
        c3 = c3 * k3 + b;
#elif MODE == 1
        // Async — arrive, do work in parallel, then wait
        asm volatile("barrier.cluster.arrive.aligned;");
        c0 = c0 * k0 + b;
        c1 = c1 * k1 + b;
        c2 = c2 * k2 + b;
        c3 = c3 * k3 + b;
        asm volatile("barrier.cluster.wait.aligned;");
#endif
    }

    if (blockIdx.x == 0 && threadIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (c0+c1+c2+c3 == 1.234567e-30f) C[blockIdx.x] = c0;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d cy/iter=%.3f\n", MODE, (double)(t1-t0)/(double)ITERS);
    }
}
