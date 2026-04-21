// V5 A6: mbarrier.arrive without wait — fire-and-forget cost
// MODE 0: baseline empty loop
// MODE 1: arrive only (no wait)
// MODE 2: arrive + wait (full sync)
// MODE 3: __syncthreads (compare with regular bar.sync)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(128, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(8) unsigned long long bar;
    unsigned int bar_addr = __cvta_generic_to_shared(&bar);
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 128;" :: "r"(bar_addr));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 1
        // fire-and-forget arrive
        asm volatile("{ .reg .b64 state;\n"
                     "  mbarrier.arrive.shared::cta.b64 state, [%0]; }"
                     :: "r"(bar_addr));
#elif MODE == 2
        // arrive + try_wait (full sync)
        asm volatile("{ .reg .b64 state;\n"
                     "  mbarrier.arrive.shared::cta.b64 state, [%0]; }"
                     :: "r"(bar_addr));
        asm volatile("{ .reg .pred p;\n"
                     "L_w_%=: mbarrier.try_wait.shared::cta.b64 p, [%0], 0;\n"
                     "  @!p bra L_w_%=; }"
                     :: "r"(bar_addr));
#elif MODE == 3
        __syncthreads();
#endif
    }

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.3f\n",
               MODE, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
