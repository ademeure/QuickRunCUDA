// V7 I4: Double-buffered async copy (ping-pong)
// MODE 0: single buffer — load, wait, compute, repeat
// MODE 1: double buffer — load N+1 while computing N
// "Compute" = read SMEM and accumulate (~10 cy of work per element)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    __shared__ __align__(16) float smem[2][1024];
    unsigned int smem_addr0 = (unsigned int)__cvta_generic_to_shared(&smem[0][0]);
    unsigned int smem_addr1 = (unsigned int)__cvta_generic_to_shared(&smem[1][0]);

    float acc = 0.0f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

#if MODE == 0
    // Single buffer: load → wait → compute, repeat
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // Load 16 B per thread = 512 B per warp
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     "cp.async.commit_group;\n"
                     "cp.async.wait_all;"
                     :: "r"(smem_addr0 + threadIdx.x * 16),
                        "l"(A + (i * 32 + threadIdx.x) * 4));
        // Compute: simple SMEM read + accumulate
        float v;
        unsigned int sa = smem_addr0 + threadIdx.x * 16;
        asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(sa));
        acc += v * 1.001f + 0.001f;  // ~5 cy of work
    }
#elif MODE == 1
    // Double buffer: prime first load, then loop
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 "cp.async.commit_group;"
                 :: "r"(smem_addr0 + threadIdx.x * 16),
                    "l"(A + threadIdx.x * 4));

    #pragma unroll 1
    for (int i = 0; i < ITERS - 1; i++) {
        unsigned int dest = (i & 1) ? smem_addr0 + threadIdx.x * 16 : smem_addr1 + threadIdx.x * 16;
        unsigned int read_buf = (i & 1) ? smem_addr1 + threadIdx.x * 16 : smem_addr0 + threadIdx.x * 16;

        // Issue next load
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     "cp.async.commit_group;"
                     :: "r"(dest), "l"(A + ((i + 1) * 32 + threadIdx.x) * 4));

        // Wait ONLY for previous (cp.async.wait_group leaves 1 in flight)
        asm volatile("cp.async.wait_group 1;");

        // Compute on previous buffer
        float v;
        asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(read_buf));
        acc += v * 1.001f + 0.001f;
    }
    // Drain final
    asm volatile("cp.async.wait_all;");
    float v;
    unsigned int last_buf = ((ITERS - 1) & 1) ? smem_addr0 + threadIdx.x * 16 : smem_addr1 + threadIdx.x * 16;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(last_buf));
    acc += v * 1.001f + 0.001f;
#endif

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (acc == 1.234567e-30f) C[blockIdx.x] = acc;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d ITERS=%d cy/iter=%.3f\n",
               MODE, ITERS, (double)(t1-t0)/(double)ITERS);
    }
}
