// V5 E1: PTX prefetch hints
// MODE 0: load only (baseline)
// MODE 1: prefetchu.L1 (prefetch into L1) before load
// MODE 2: prefetch.global.L1 before load
// MODE 3: prefetch.global.L2 before load
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;
    unsigned int v = (unsigned)u2;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // Address shifts each iteration to defeat L1 caching for cold-miss test
        unsigned int* addr = (unsigned int*)A + ((i * 32) & ((1 << 20) - 1));
#if MODE == 1
        asm volatile("prefetchu.L1 [%0];" :: "l"(addr));
#elif MODE == 2
        asm volatile("prefetch.global.L1 [%0];" :: "l"(addr));
#elif MODE == 3
        asm volatile("prefetch.global.L2 [%0];" :: "l"(addr));
#endif
        unsigned int x;
        asm volatile("ld.global.u32 %0, [%1];" : "=r"(x) : "l"(addr));
        v ^= x;
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/load=%.2f\n",
               MODE, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
