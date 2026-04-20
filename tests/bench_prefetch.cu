// PTX prefetch instruction - does it work?
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;
    unsigned int* p = (unsigned int*)A;
    unsigned int acc = 0;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < 256; i++) {
        unsigned int idx = (i * 32 + threadIdx.x) & 0x1FFFFF;
#if MODE == 0
        // No prefetch
        unsigned int x;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x) : "l"(p + idx));
        acc ^= x;
#elif MODE == 1
        // PTX prefetch.L2 + load
        asm volatile("prefetch.global.L2 [%0];" :: "l"(p + idx));
        unsigned int x;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x) : "l"(p + idx));
        acc ^= x;
#elif MODE == 2
        // PTX prefetch.L1 + load
        asm volatile("prefetch.global.L1 [%0];" :: "l"(p + idx));
        unsigned int x;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x) : "l"(p + idx));
        acc ^= x;
#elif MODE == 3
        // Prefetch FAR ahead (test prefetch ahead of time)
        unsigned int prefetch_idx = ((i + 64) * 32 + threadIdx.x) & 0x1FFFFF;
        asm volatile("prefetch.global.L2 [%0];" :: "l"(p + prefetch_idx));
        unsigned int x;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x) : "l"(p + idx));
        acc ^= x;
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (acc == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = acc;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/load=%.3f\n",
               MODE, t1 - t0, (double)(t1-t0)/256.0);
    }
}
