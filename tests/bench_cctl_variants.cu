// Test CCTL (cache control) variants via discard.global and other PTX constructs.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef ITERS
#define ITERS 200
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    if (threadIdx.x != 0) return;
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long t0, t1;
    unsigned total = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
#if OP == 0
        // applypriority.global.L2 (soft cache hint)
        asm volatile("applypriority.global.L2::evict_normal [%0], 128;" :: "l"(A + tid * 32));
#elif OP == 1
        // discard.global (hint to drop from cache)
        asm volatile("discard.global.L2 [%0], 128;" :: "l"(A + tid * 32));
#elif OP == 2
        // cp.async.bulk.prefetch.L2 (manual prefetch)
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], 128;" :: "l"(A + tid * 32) : "memory");
#elif OP == 3
        // cp.async.bulk.commit_group (no payload)
        asm volatile("cp.async.bulk.commit_group;");
#elif OP == 4
        // prefetch.global.L1
        asm volatile("prefetch.global.L1 [%0];" :: "l"(A + tid * 32));
#elif OP == 5
        // prefetch.global.L2
        asm volatile("prefetch.global.L2 [%0];" :: "l"(A + tid * 32));
#elif OP == 6
        // ld.global.L1::evict_last — cache hint
        unsigned r;
        asm volatile("ld.global.L1::evict_last.u32 %0, [%1];" : "=r"(r) : "l"(A + tid * 32));
        total += r;
#elif OP == 7
        // ld.global.L1::evict_first
        unsigned r;
        asm volatile("ld.global.L1::evict_first.u32 %0, [%1];" : "=r"(r) : "l"(A + tid * 32));
        total += r;
#endif
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        total += (unsigned)(t1 - t0);
    }
    C[blockIdx.x] = total / ITERS;
}
