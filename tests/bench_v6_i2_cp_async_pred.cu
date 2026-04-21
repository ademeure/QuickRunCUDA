// V6 I2: cp.async with predicate
// MODE 0: cp.async always executes
// MODE 1: cp.async @P0 (predicate true)
// MODE 2: cp.async @P0 (predicate false — should be skipped)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    __shared__ __align__(16) unsigned int smem[1024];
    unsigned int smem_addr_base = (unsigned int)__cvta_generic_to_shared(smem);
    int pflag = (seed > 0) ? 1 : 0;
#if MODE == 1
    pflag = 1;
#elif MODE == 2
    pflag = 0;
#endif

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     "cp.async.commit_group;\n"
                     "cp.async.wait_all;"
                     :: "r"(smem_addr_base + threadIdx.x * 16),
                        "l"(A + (i * 32 + threadIdx.x) * 4));
#else
        asm volatile("{ .reg .pred p;\n"
                     "  setp.ne.s32 p, %2, 0;\n"
                     "  @p cp.async.cg.shared.global [%0], [%1], 16;\n"
                     "  cp.async.commit_group;\n"
                     "  cp.async.wait_all; }"
                     :: "r"(smem_addr_base + threadIdx.x * 16),
                        "l"(A + (i * 32 + threadIdx.x) * 4),
                        "r"(pflag));
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (smem[threadIdx.x] == 0xCAFEBABE) C[blockIdx.x] = (float)smem[threadIdx.x];

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d ITERS=%d cy/iter=%.3f\n", MODE, ITERS, (double)(t1-t0)/(double)ITERS);
    }
}
