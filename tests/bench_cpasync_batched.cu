// cp.async pipelined: issue many, commit, work, wait
#ifndef MODE
#define MODE 0
#endif
#ifndef N_STAGES
#define N_STAGES 4
#endif

extern "C" __global__ __launch_bounds__(128, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ unsigned int smem[2048];
    unsigned int* p = (unsigned int*)A;
    unsigned int v = (unsigned)threadIdx.x;

    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // Sync LDG+STS baseline (16B)
        uint4 x = ((uint4*)p)[(i * 128 + threadIdx.x) & 0xFF];
        ((uint4*)smem)[(i * 128 + threadIdx.x) & 0xFF] = x;
        __syncthreads();
#elif MODE == 1
        // cp.async + immediate wait (bad pattern)
        unsigned int idx = ((i * 128 + threadIdx.x) & 0xF0);
        asm("cp.async.cg.shared.global [%0], [%1], 16;"
            :: "r"((unsigned)__cvta_generic_to_shared(smem + idx)), "l"(p + idx));
        asm("cp.async.commit_group;");
        asm("cp.async.wait_all;");
        __syncthreads();
#elif MODE == 2
        // Batched: issue N, commit each, do compute, wait
        #pragma unroll
        for (int k = 0; k < N_STAGES; k++) {
            unsigned int idx = ((i * 128 * N_STAGES + k * 128 + threadIdx.x) & 0xF0);
            asm("cp.async.cg.shared.global [%0], [%1], 16;"
                :: "r"((unsigned)__cvta_generic_to_shared(smem + idx)), "l"(p + idx));
        }
        asm("cp.async.commit_group;");
        // Some compute work between issue and wait
        v = v * 31u + (unsigned)i;
        v = v * 17u + (unsigned)i;
        v = v * 13u + (unsigned)i;
        v = v * 7u + (unsigned)i;
        asm("cp.async.wait_all;");
        __syncthreads();
#endif
    }

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (smem[0] == (unsigned)seed && v == (unsigned)seed) C[blockIdx.x] = (float)smem[0];
    if (blockIdx.x == 0 && threadIdx.x == 0) {
#if MODE == 2
        unsigned long long bytes_per_iter = 16ull * N_STAGES * 128;
#else
        unsigned long long bytes_per_iter = 16ull * 128;
#endif
        printf("MODE=%d N_STAGES=%d clk=%llu cy/iter=%.3f bytes/iter=%llu cy/byte=%.3f\n",
               MODE, N_STAGES, t1 - t0, (double)(t1-t0)/(double)ITERS,
               bytes_per_iter, (double)(t1-t0)/(double)(ITERS * bytes_per_iter));
    }
}
