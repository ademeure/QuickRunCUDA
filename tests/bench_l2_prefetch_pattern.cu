// L2 prefetcher detection: does B300 detect stride patterns?

#ifndef MODE
#define MODE 0
#endif
#ifndef N_LOADS
#define N_LOADS 1024
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int* Au = (unsigned int*)A;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    unsigned int acc = 0;
    #pragma unroll 1
    for (int i = 0; i < N_LOADS; i++) {
        unsigned int idx;
#if MODE == 0
        idx = (i * 32 + threadIdx.x) & 0x1FFFFF;  // sequential 8MB working set
#elif MODE == 1
        idx = ((N_LOADS - 1 - i) * 32 + threadIdx.x) & 0x1FFFFF;  // reverse
#elif MODE == 2
        unsigned int hash = (i * 0x9E3779B1u) ^ ((unsigned)i >> 16);
        idx = ((hash & (N_LOADS - 1)) * 32 + threadIdx.x) & 0x1FFFFF;  // shuffled
#elif MODE == 3
        idx = (i * 32 * 16 + threadIdx.x) & 0x1FFFFF;  // stride 64B
#endif
        idx ^= ((unsigned)u2 * acc);
        unsigned int x;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x) : "l"(Au + idx));
        acc ^= x;
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (acc == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d N_LOADS=%d clk=%llu cy/load=%.3f\n",
               MODE, N_LOADS, t1 - t0, (double)(t1-t0)/(double)N_LOADS);
    }
}
