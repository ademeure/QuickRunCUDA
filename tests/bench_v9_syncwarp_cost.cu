// V9: __syncwarp() latency
#ifndef CHAIN_LEN
#define CHAIN_LEN 1000
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    __syncwarp();

    // Chain of syncwarp calls
    #pragma unroll 1
    for (int i = 0; i < CHAIN_LEN; i++) {
        __syncwarp();
    }
    __syncwarp();

    if (threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        ((unsigned long long*)C)[blockIdx.x] = t1 - t0;
    }
}
