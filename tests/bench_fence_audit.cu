// Memory fence costs — different scopes and orderings.

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
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    unsigned acc = seed;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if OP == 0
        // membar.cta (deprecated but still emitted)
        asm volatile("membar.cta;");
#elif OP == 1
        // membar.gl
        asm volatile("membar.gl;");
#elif OP == 2
        // membar.sys
        asm volatile("membar.sys;");
#elif OP == 3
        // fence.sc.cta (sequential consistency)
        asm volatile("fence.sc.cta;");
#elif OP == 4
        // fence.sc.gpu
        asm volatile("fence.sc.gpu;");
#elif OP == 5
        // fence.acq_rel.cta
        asm volatile("fence.acq_rel.cta;");
#elif OP == 6
        // fence.acq_rel.gpu
        asm volatile("fence.acq_rel.gpu;");
#elif OP == 7
        // bar.cta.sync
        asm volatile("bar.cta.sync 0, 32;");
#elif OP == 8
        // bar.cta.arrive + wait separately
        asm volatile("bar.cta.arrive 0;");
        asm volatile("bar.cta.wait 0;");
#elif OP == 9
        // __syncwarp via bar.warp.sync
        asm volatile("bar.warp.sync 0xFFFFFFFF;");
#endif
        acc ^= i;
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        C[2] = acc;
    }
}
