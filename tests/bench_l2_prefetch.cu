// Compare L2 warming patterns: cp.async.bulk.prefetch vs normal LDG vs nothing.
// Access a fresh 16 MB region, then time subsequent reads to see hit rate.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = gridDim.x * blockDim.x;
    unsigned acc = 0;
    // 16 MB working set = 4M u32 addresses
    unsigned mask = 0xFFFFFF;  // 16 MB

    // WARM-UP phase
    for (int i = 0; i < 4096; i++) {
        unsigned long long off = ((unsigned long long)tid * 4 + (unsigned long long)i * 4 * total) & mask;
        unsigned long long addr = (unsigned long long)A + off;
#if OP == 0  // no warm-up
#elif OP == 1  // LDG warm
        unsigned x;
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(addr));
        acc ^= x;
#elif OP == 2  // cp.async.bulk.prefetch (L2-only warm)
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], 128;" :: "l"(addr) : "memory");
#elif OP == 3  // prefetch with 16B granule
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], 16;" :: "l"(addr) : "memory");
#endif
    }
    __syncthreads();

    // Now time READS of the same region (should be L2 hit if warm)
    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    for (int i = 0; i < 4096; i++) {
        unsigned long long off = ((unsigned long long)tid * 4 + (unsigned long long)i * 4 * total) & mask;
        unsigned long long addr = (unsigned long long)A + off;
        unsigned x;
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(addr));
        acc ^= x;
    }
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (threadIdx.x == 0) ((unsigned long long*)C)[blockIdx.x] = t1 - t0;
    if (acc == seed) C[blockIdx.x * BLOCK_SIZE + threadIdx.x + 1024] = acc;
}
