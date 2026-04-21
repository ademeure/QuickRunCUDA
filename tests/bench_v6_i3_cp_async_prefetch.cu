// V6 I3: cp.async + explicit L2 prefetch combo
// MODE 0: cp.async only (baseline)
// MODE 1: prefetch.L2 then cp.async (warm L2 first)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    __shared__ __align__(16) unsigned int smem[1024];
    unsigned int smem_addr_base = (unsigned int)__cvta_generic_to_shared(smem);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 1
        // Issue L2 prefetch ahead of current iter
        asm volatile("prefetch.global.L2 [%0];"
                     :: "l"(A + ((i + 4) * 32 + threadIdx.x) * 4));
#endif
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     "cp.async.commit_group;\n"
                     "cp.async.wait_all;"
                     :: "r"(smem_addr_base + threadIdx.x * 16),
                        "l"(A + (i * 32 + threadIdx.x) * 4));
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (smem[threadIdx.x] == 0xCAFEBABE) C[blockIdx.x] = (float)smem[threadIdx.x];

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d ITERS=%d cy/iter=%.3f\n",
               MODE, ITERS, (double)(t1-t0)/(double)ITERS);
    }
}
