// E5: Atomic latency vs L2 partition
// Each warp does atomicAdd on a unique address; measure per-warp time
// Vary which address bits change between warps to detect L2 partition effects

#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    // Each block handles a different address; vary by block index
    // MODE 0: addresses spaced 4 KB apart (potential same partition)
    // MODE 1: addresses spaced 1 KB apart (rotates partitions)
    // MODE 2: addresses spaced 256 B apart (cache-line granularity)
    // MODE 3: addresses spaced 32 KB apart (super-line)
#if MODE == 0
    unsigned int addr_word_off = blockIdx.x * (4096 / 4);
#elif MODE == 1
    unsigned int addr_word_off = blockIdx.x * (1024 / 4);
#elif MODE == 2
    unsigned int addr_word_off = blockIdx.x * (256 / 4);
#elif MODE == 3
    unsigned int addr_word_off = blockIdx.x * (32768 / 4);
#endif

    unsigned int* target = (unsigned int*)C + addr_word_off;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // Each thread does ITERS atomic adds (chained — atomic returns old value)
    unsigned int sink = (unsigned)u2;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        sink ^= atomicAdd(target, 1u + sink);
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (sink == (unsigned)seed) ((unsigned*)C)[8192*1024 + blockIdx.x] = sink;
    if (threadIdx.x == 0) {
        // Each block records its time at end of buffer
        ((unsigned long long*)C)[8192*1024 + 1024 + blockIdx.x] = t1 - t0;
    }
}
