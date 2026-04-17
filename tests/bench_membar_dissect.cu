// Dissect membar.sys cost: is it warps/SM (8-channel limit) or total writes or per-warp writes?
// Explicitly control: WARPS_PER_SM, WRITES_PER_THREAD.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef ACTIVE_WARPS
#define ACTIVE_WARPS 32  // how many warps in the block are "active" (the rest early-exit)
#endif
#ifndef NWRITES
#define NWRITES 1
#endif
#ifndef ITERS
#define ITERS 200
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned warp_id = threadIdx.x >> 5;
    unsigned lane = threadIdx.x & 31;
    if (warp_id >= ACTIVE_WARPS) return;
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned* my_addr = A + tid * NWRITES;

    unsigned long long t0, t1;
    unsigned total = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
#if NWRITES > 0
        #pragma unroll
        for (int j = 0; j < NWRITES; j++) ((volatile unsigned*)my_addr)[j] = i + seed + j;
#endif
        asm volatile("fence.sc.sys;");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        total += (unsigned)(t1 - t0);
    }
    if (lane == 0) {
        C[blockIdx.x * ACTIVE_WARPS + warp_id] = total / ITERS;
    }
}
