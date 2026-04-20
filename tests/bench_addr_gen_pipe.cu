// Address generation pipeline test: can LDG addr-ALU overlap with prior LDG?

#ifndef MODE
#define MODE 0
#endif
#ifndef N_LOADS
#define N_LOADS 64
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int* p = (unsigned int*)A;
    unsigned int idx = threadIdx.x;
    unsigned int acc = 0;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < N_LOADS; i++) {
#if MODE == 0
        // Simple chain: addr depends on prev result
        unsigned int x;
        asm volatile("ld.global.ca.u32 %0, [%1];"
                     : "=r"(x) : "l"(p + (idx & 0x3FF)));
        idx = x;
        acc ^= x;
#elif MODE == 1
        // Chain with extra ALU work between LDGs (simulating address calc)
        unsigned int x;
        asm volatile("ld.global.ca.u32 %0, [%1];"
                     : "=r"(x) : "l"(p + (idx & 0x3FF)));
        // 4 ALU ops between LDGs
        x = x * 31u + (unsigned)i;
        x = x ^ (x >> 5);
        x = x * 17u + (unsigned)u2;
        x = x ^ (x >> 7);
        idx = x;
        acc ^= x;
#elif MODE == 2
        // Many ALU ops between LDGs (test overlap with address calc)
        unsigned int x;
        asm volatile("ld.global.ca.u32 %0, [%1];"
                     : "=r"(x) : "l"(p + (idx & 0x3FF)));
        // 16 ALU ops
        for (int j = 0; j < 16; j++) x = x * 31u + (unsigned)j;
        idx = x;
        acc ^= x;
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (acc == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = acc;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d N_LOADS=%d clk=%llu cy/load=%.3f\n",
               MODE, N_LOADS, t1 - t0, (double)(t1-t0)/(double)N_LOADS);
    }
}
