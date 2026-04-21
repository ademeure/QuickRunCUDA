// V8 J1: Auto-find optimal block size for a workload
// Compile with different THREADS values; report which is fastest
#ifndef THREADS
#define THREADS 128
#endif

extern "C" __global__ __launch_bounds__(THREADS, 1)
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
    }

    if (threadIdx.x == 0 && blockIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    float sum = c0+c1+c2+c3;
    if (sum == 1.234567e-30f) C[blockIdx.x * blockDim.x + threadIdx.x] = sum;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("THREADS=%d cy/iter=%.3f\n", THREADS, (double)(t1-t0)/(double)ITERS);
    }
}
