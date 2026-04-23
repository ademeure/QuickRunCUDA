// Audit: SHFL latency via warp self-chain
// All 32 threads run, but each thread chain-uses its own SHFL result.
// Measure thread 0.
#ifndef CHAIN_LEN
#define CHAIN_LEN 4096
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(int* A, int* B, int* C, int ITERS, int seed, int u2) {
    int v = threadIdx.x + seed + 1;

    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 32
    for (int i = 0; i < CHAIN_LEN; i++) {
        // Self-chain SHFL: each lane reads from (lane+1) % 32
        v = __shfl_sync(0xFFFFFFFF, v, (threadIdx.x + 1) & 31);
    }

    if (threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        ((unsigned long long*)C)[0] = t1 - t0;
        ((int*)C)[2] = v;
    }
}
