// V8 J4: Cluster size auto-pick for FFMA-heavy workload
// Test cluster size 1, 2, 4, 8 on same FFMA workload; measure per-iter cost
#ifndef CSIZE
#define CSIZE 1
#endif

extern "C" __global__ __launch_bounds__(128, 1)
#if CSIZE > 1
__cluster_dims__(CSIZE, 1, 1)
#endif
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float a = (float)(threadIdx.x + 1) * 0.001f;
    float b = (float)(blockIdx.x + 1) * 0.001f;
    float c0=a, c1=a, c2=a, c3=a;
    float k0=b, k1=b, k2=b, k3=b;

    unsigned long long t0, t1;
    if (threadIdx.x == 0 && blockIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll 32
        for (int k = 0; k < 256; k++) {
            c0 = c0 * k0 + b;
            c1 = c1 * k1 + b;
            c2 = c2 * k2 + b;
            c3 = c3 * k3 + b;
        }
#if CSIZE > 1
        // Per iter: cluster sync
        asm volatile("barrier.cluster.arrive.aligned;");
        asm volatile("barrier.cluster.wait.aligned;");
#endif
    }

    if (threadIdx.x == 0 && blockIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    float sum = c0+c1+c2+c3;
    if (sum == 1.234567e-30f) C[blockIdx.x] = sum;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("CSIZE=%d cy/iter=%.3f\n", CSIZE, (double)(t1-t0)/(double)ITERS);
    }
}
