// cp.async.commit_group / wait_all overhead
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(128, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ unsigned int smem[1024];
    unsigned int* p = (unsigned int*)A;

    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    for (int i = 0; i < ITERS; i++) {
        unsigned int idx = (i * 128 + threadIdx.x) & 0xFF;
#if MODE == 0
        // No async — just LDG + STS
        unsigned int x;
        asm("ld.global.cg.u32 %0, [%1];" : "=r"(x) : "l"(p + idx));
        asm("st.shared.u32 [%0], %1;"
            :: "r"((unsigned)__cvta_generic_to_shared(smem + idx)), "r"(x));
#elif MODE == 1
        // cp.async 16-byte
        asm("cp.async.cg.shared.global [%0], [%1], 16;"
            :: "r"((unsigned)__cvta_generic_to_shared(smem + (idx & 0xF0))), "l"(p + (idx & 0xF0)));
#elif MODE == 2
        // cp.async 16B + commit + wait_all
        asm("cp.async.cg.shared.global [%0], [%1], 16;"
            :: "r"((unsigned)__cvta_generic_to_shared(smem + (idx & 0xF0))), "l"(p + (idx & 0xF0)));
        asm("cp.async.commit_group;");
        asm("cp.async.wait_all;");
#elif MODE == 3
        // cp.async 16B + commit + wait_group<0>
        asm("cp.async.cg.shared.global [%0], [%1], 16;"
            :: "r"((unsigned)__cvta_generic_to_shared(smem + (idx & 0xF0))), "l"(p + (idx & 0xF0)));
        asm("cp.async.commit_group;");
        asm("cp.async.wait_group 0;");
#endif
    }

    __syncthreads();
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (smem[0] == (unsigned)seed) C[blockIdx.x] = (float)smem[0];
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.3f\n", MODE, t1 - t0, (double)(t1-t0)/(double)ITERS);
    }
}
