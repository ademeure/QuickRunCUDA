// V5 E1': K-ahead pipelined prefetch — does prefetching K iters ahead beat back-to-back?
#ifndef K_AHEAD
#define K_AHEAD 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;
    unsigned int v = (unsigned)u2;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // Each iter: prefetch[i+K_AHEAD], load[i]
        unsigned int* load_addr = (unsigned int*)A + ((i * 32) & ((1 << 20) - 1));
#if K_AHEAD > 0
        unsigned int* pf_addr = (unsigned int*)A + (((i + K_AHEAD) * 32) & ((1 << 20) - 1));
        asm volatile("prefetch.global.L1 [%0];" :: "l"(pf_addr));
#endif
        unsigned int x;
        asm volatile("ld.global.u32 %0, [%1];" : "=r"(x) : "l"(load_addr));
        v ^= x;
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("K_AHEAD=%d clk=%llu cy/load=%.2f\n",
               K_AHEAD, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
