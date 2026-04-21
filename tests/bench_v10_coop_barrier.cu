// V10: grid-level cooperative barrier cost
// Compare: grid_group().sync() vs just __syncthreads() in persistent kernel
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#ifndef CHAIN_LEN
#define CHAIN_LEN 100
#endif
#ifndef MODE
#define MODE 0  // 0=grid.sync, 1=syncthreads
#endif

extern "C" __global__ __launch_bounds__(128, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    cg::grid_group grid = cg::this_grid();

    unsigned long long t0, t1;
    if (blockIdx.x == 0 && threadIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
#if MODE == 0
    grid.sync();
#elif MODE == 1
    __syncthreads();
#endif

    #pragma unroll 1
    for (int i = 0; i < CHAIN_LEN; i++) {
#if MODE == 0
        grid.sync();
#elif MODE == 1
        __syncthreads();
#endif
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        ((unsigned long long*)C)[0] = t1 - t0;
    }
}
