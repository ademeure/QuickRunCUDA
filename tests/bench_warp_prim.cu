// Warp primitives audit: throughput + SASS for shfl, vote, ballot, etc.
//
// OP=0 : __shfl_sync with butterfly pattern
// OP=1 : __shfl_xor_sync (most common reduction)
// OP=2 : __shfl_up_sync
// OP=3 : __shfl_down_sync
// OP=4 : __ballot_sync
// OP=5 : __any_sync
// OP=6 : __all_sync
// OP=7 : __activemask
// OP=8 : __reduce_sum_sync (Hopper+ warp reduce)
// OP=9 : __reduce_min_sync

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef ITERS
#define ITERS 4096
#endif
#ifndef OP
#define OP 0
#endif

// Don't need cooperative_groups for these — __reduce_*_sync is a builtin

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned v = tid + seed;
    unsigned acc = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if OP == 0
        v = __shfl_sync(0xFFFFFFFF, v, (i + 1) & 31);
        acc ^= v;
#elif OP == 1
        v = __shfl_xor_sync(0xFFFFFFFF, v, 1);
        acc ^= v;
#elif OP == 2
        v = __shfl_up_sync(0xFFFFFFFF, v, 1);
        acc ^= v;
#elif OP == 3
        v = __shfl_down_sync(0xFFFFFFFF, v, 1);
        acc ^= v;
#elif OP == 4
        unsigned mask = __ballot_sync(0xFFFFFFFF, v & 1);
        acc ^= mask;
#elif OP == 5
        unsigned r = __any_sync(0xFFFFFFFF, v & 1);
        acc ^= r;
#elif OP == 6
        unsigned r = __all_sync(0xFFFFFFFF, v & 1);
        acc ^= r;
#elif OP == 7
        unsigned r = __activemask();
        acc ^= r;
#elif OP == 8
        unsigned r = __reduce_add_sync(0xFFFFFFFF, v);
        acc ^= r;
#elif OP == 9
        unsigned r = __reduce_min_sync(0xFFFFFFFF, v);
        acc ^= r;
#elif OP == 10
        // bar.warp.sync (synchronization only)
        __syncwarp();
        acc ^= v;
#endif
        v = v * 1664525u + 1013904223u;  // LCG to keep v changing
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        C[2] = acc;
        C[3] = v;
    }
}
