// membar.sys with N warps per SM each doing write+membar.
// Each participating warp's lane 0 measures per-iter latency.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef ACTIVE_WARPS
#define ACTIVE_WARPS 16
#endif
#ifndef ITERS
#define ITERS 100
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned warp_id = threadIdx.x >> 5;
    unsigned lane = threadIdx.x & 31;
    if (warp_id >= ACTIVE_WARPS) return;

    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned* my_addr = A + tid;

    // Each active warp writes its own median cy to C
    unsigned long long t0, t1;
    unsigned total = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        *(volatile unsigned*)my_addr = i + seed;
#ifdef USE_GL
        asm volatile("membar.gl;");
#else
        asm volatile("membar.sys;");
#endif
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        total += (unsigned)(t1 - t0);
    }
    if (lane == 0) {
        // Output: C[blockIdx * ACTIVE_WARPS + warp_id]
        C[blockIdx.x * ACTIVE_WARPS + warp_id] = total / ITERS;
    }
}
