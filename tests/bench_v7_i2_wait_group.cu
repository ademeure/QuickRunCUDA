// V7 I2: cp.async.wait_group N vs wait_all
// MODE 0: wait_all after all 16 commits (drain everything)
// MODE 1: wait_group 8 (leave 8 in flight)
// MODE 2: wait_group 12 (leave 12 in flight, very deep pipeline)
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
        // Issue 16 cp.async + commit
        #pragma unroll
        for (int g = 0; g < 16; g++) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         "cp.async.commit_group;"
                         :: "r"(smem_addr_base + threadIdx.x * 16),
                            "l"(A + ((i * 16 + g) * 32 + threadIdx.x) * 4));
        }
#if MODE == 0
        asm volatile("cp.async.wait_all;");
#elif MODE == 1
        asm volatile("cp.async.wait_group 8;");  // leave 8 in flight
#elif MODE == 2
        asm volatile("cp.async.wait_group 12;");  // leave 12 in flight
#endif
    }
    asm volatile("cp.async.wait_all;");

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (smem[threadIdx.x] == 0xCAFEBABE) C[blockIdx.x] = (float)smem[threadIdx.x];

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d cy/iter=%.3f\n", MODE, (double)(t1-t0)/(double)ITERS);
    }
}
