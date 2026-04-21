// SMEM load chain latency
#ifndef MODE
#define MODE 0
#endif
#ifndef N_LOADS
#define N_LOADS 1024
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;
    __shared__ unsigned int smem[1024];

    if (threadIdx.x < 32) {
        for (int i = threadIdx.x; i < 1024; i += 32) smem[i] = (i + 1) & 0x3FF;
    }
    __syncwarp();

    unsigned int idx = threadIdx.x;
    unsigned int acc = 0;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < N_LOADS; i++) {
#if MODE == 0
        // Chain through SMEM (each load addr = prev result)
        unsigned int x;
        asm volatile("ld.shared.u32 %0, [%1];"
                     : "=r"(x) : "r"((unsigned)__cvta_generic_to_shared(smem + (idx & 0x3FF))));
        idx = x;
        acc ^= x;
#elif MODE == 1
        // No chain — use i for addr (compiler may hoist)
        unsigned int x;
        asm volatile("ld.shared.u32 %0, [%1];"
                     : "=r"(x) : "r"((unsigned)__cvta_generic_to_shared(smem + ((i + (unsigned)u2 * acc) & 0x3FF))));
        acc ^= x;
#elif MODE == 2
        // 4-way ILP through 4 chains
        if ((i & 3) == 0) {
            unsigned int x;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"((unsigned)__cvta_generic_to_shared(smem + (idx & 0x3FF))));
            idx = x;
            acc ^= x;
        }
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (acc == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d N=%d clk=%llu cy/load=%.3f\n",
               MODE, N_LOADS, t1 - t0, (double)(t1-t0)/(double)N_LOADS);
    }
}
