// cp.async vs LDG+STS for SMEM staging
#ifndef MODE
#define MODE 0
#endif
#ifndef N_LOADS
#define N_LOADS 8
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;
    __shared__ unsigned int smem[1024];
    unsigned int* p = (unsigned int*)A;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // LDG + STS (synchronous staging)
        #pragma unroll
        for (int k = 0; k < N_LOADS; k++) {
            unsigned int idx = (i * 32 + threadIdx.x + k * 32) & 0x3FF;
            unsigned int x;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(p + idx));
            asm volatile("st.shared.u32 [%0], %1;"
                         :: "r"((unsigned)__cvta_generic_to_shared(smem + (k & 1023))), "r"(x));
        }
        __syncwarp();
#elif MODE == 1
        // cp.async (async staging)
        #pragma unroll
        for (int k = 0; k < N_LOADS; k++) {
            unsigned int idx = (i * 32 + threadIdx.x + k * 32) & 0x3FF;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 4;"
                         :: "r"((unsigned)__cvta_generic_to_shared(smem + (k & 1023))),
                            "l"(p + idx));
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        __syncwarp();
#elif MODE == 2
        // cp.async with 16-byte transfers
        #pragma unroll
        for (int k = 0; k < N_LOADS; k++) {
            unsigned int idx = (i * 32 + threadIdx.x + k * 32) & 0x3FC;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                         :: "r"((unsigned)__cvta_generic_to_shared(smem + (k * 4 & 1020))),
                            "l"(p + idx));
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        __syncwarp();
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (smem[0] == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = smem[0];
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long total = (unsigned long long)ITERS * N_LOADS;
        printf("MODE=%d N_LOADS=%d clk=%llu cy/load=%.3f\n",
               MODE, N_LOADS, t1 - t0, (double)(t1-t0)/(double)total);
    }
}
