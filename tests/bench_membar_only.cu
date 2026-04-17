// Try to use ONLY MEMBAR.SC.SYS (no CCTL.IVALL) via inline asm.
// This isolates the cost of the fence vs cache invalidate.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef ITERS
#define ITERS 100
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    if (threadIdx.x != 0) return;
    unsigned long long t0, t1;
    unsigned total = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
#if OP == 0
        // Full fence.sc.sys (emits 4 insts)
        asm volatile("fence.sc.sys;");
#elif OP == 1
        // Just MEMBAR.SC.SYS via inline SASS (no CCTL)
        // PTX can't directly emit MEMBAR.SC.SYS without the ERRBAR/CCTL tail
        // Try raw `bar.sync.aligned` or other alternatives...
        // Actually PTX doesn't let us bypass, so fence.sc.sys is the floor
        asm volatile("fence.sc.sys;");  // placeholder
#elif OP == 2
        // CCTL.IVALL alone (emit via PTX?)
        // discard.global or cp.async.bulk.prefetch-related?
        // Just test cache invalidation via unique ld to refresh
        asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(total) : "l"(C));
#elif OP == 3
        // barrier.cta.sync (just bar, no mem)
        asm volatile("bar.cta.sync 0, 32;");
#endif
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        total += (unsigned)(t1 - t0);
    }
    C[blockIdx.x] = total / ITERS;
}
