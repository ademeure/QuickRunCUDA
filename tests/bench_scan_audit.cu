// Warp-level inclusive scan (prefix sum) implementations.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef ITERS
#define ITERS 1024
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned lane = threadIdx.x & 31;
    unsigned v = (unsigned)seed + lane;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int iter = 0; iter < ITERS; iter++) {
#if OP == 0
        // Standard shfl_up-based Kogge-Stone (log2(32) = 5 levels)
        unsigned x = v;
        #pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            unsigned y = __shfl_up_sync(0xFFFFFFFF, x, offset);
            if (lane >= offset) x += y;
        }
        v = x;
#elif OP == 1
        // Using __reduce_add_sync (won't give per-lane prefix, just total)
        unsigned total = __reduce_add_sync(0xFFFFFFFF, v);
        v = total;
#elif OP == 2
        // Brent-Kung style (O(n log n) work but fewer shfl levels conceptually)
        unsigned x = v;
        #pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            unsigned y = __shfl_up_sync(0xFFFFFFFF, x, offset, 32);
            if (lane >= offset) x += y;
        }
        v = x;
#elif OP == 3
        // Using CREDUX (not directly for scan, but for baseline comparison)
        unsigned mx = __reduce_max_sync(0xFFFFFFFF, v);
        v ^= mx;
#endif
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        C[2] = v;
    }
}
