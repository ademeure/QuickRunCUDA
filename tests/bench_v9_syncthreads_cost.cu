// V9: __syncthreads() latency — barrier cost per call
#ifndef CHAIN_LEN
#define CHAIN_LEN 1000
#endif
#ifndef THREADS
#define THREADS 256
#endif

extern "C" __global__ __launch_bounds__(THREADS, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // All threads participate in barrier. Thread 0 measures.
    unsigned long long t0, t1;

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    __syncthreads();  // sync t0 read across warps

    // Chain of syncthreads calls
    #pragma unroll 1
    for (int i = 0; i < CHAIN_LEN; i++) {
        __syncthreads();
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        ((unsigned long long*)C)[blockIdx.x] = t1 - t0;
    }
}
