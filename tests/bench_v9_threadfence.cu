// V9: __threadfence variants — cost of memory fences
#ifndef CHAIN_LEN
#define CHAIN_LEN 1000
#endif
#ifndef FENCE
#define FENCE 0  // 0=block, 1=gpu, 2=system, 3=none (baseline)
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x != 0) return;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < CHAIN_LEN; i++) {
#if FENCE == 0
        __threadfence_block();
#elif FENCE == 1
        __threadfence();
#elif FENCE == 2
        __threadfence_system();
#elif FENCE == 3
        // no-op (baseline — loop overhead only)
        __syncwarp();  // cheap op to prevent loop elision
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (blockIdx.x == 0) ((unsigned long long*)C)[0] = t1 - t0;
}
