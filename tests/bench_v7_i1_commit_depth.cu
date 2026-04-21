// V7 I1: cp.async.commit_group depth
// Test how many groups in flight can hide latency
// MODE 0: 1 group (depth 1) — sequential
// MODE 1: 4 groups in flight
// MODE 2: 16 groups in flight
#ifndef MODE
#define MODE 0
#endif

#if MODE == 0
#define DEPTH 1
#elif MODE == 1
#define DEPTH 4
#elif MODE == 2
#define DEPTH 16
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    __shared__ __align__(16) unsigned int smem[1024];
    unsigned int smem_addr_base = (unsigned int)__cvta_generic_to_shared(smem);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // Issue DEPTH groups, then wait for them all (pipeline depth)
    #pragma unroll 1
    for (int i = 0; i < ITERS / DEPTH; i++) {
        // Issue DEPTH cp.async + commit_group
        #pragma unroll
        for (int g = 0; g < DEPTH; g++) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         "cp.async.commit_group;"
                         :: "r"(smem_addr_base + threadIdx.x * 16),
                            "l"(A + ((i * DEPTH + g) * 32 + threadIdx.x) * 4));
        }
        // Wait for all
        asm volatile("cp.async.wait_all;");
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (smem[threadIdx.x] == 0xCAFEBABE) C[blockIdx.x] = (float)smem[threadIdx.x];

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d DEPTH=%d cy/iter=%.3f cy/cp_async=%.3f\n",
               MODE, DEPTH, (double)(t1-t0)/(double)(ITERS/DEPTH),
               (double)(t1-t0)/(double)ITERS);
    }
}
