// Chain latency through clock64 register reads
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;
    unsigned long long acc = 0;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // Pure clock64 read chain
        unsigned long long c;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c));
        acc = c + acc;
#elif MODE == 1
        // clock64 + cvt to f64 + back (harder chain)
        unsigned long long c;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c));
        acc = c ^ (acc >> 1);
#elif MODE == 2
        // clock + clock64 alternating
        unsigned int c1;
        unsigned long long c2;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c1));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c2));
        acc = (unsigned long long)c1 + c2;
#elif MODE == 3
        // %globaltimer (ns)
        unsigned long long c;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(c));
        acc = c + acc;
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if ((int)acc == seed) ((unsigned*)C)[blockIdx.x] = (unsigned)acc;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.3f\n", MODE, t1 - t0, (double)(t1-t0)/(double)ITERS);
    }
}
