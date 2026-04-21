// V5 A4: Multiple mbarrier objects in one CTA — do they share resources?
// MODE 0: 1 mbarrier, arrive only
// MODE 1: 4 mbarriers, arrive on each (round-robin)
// MODE 2: 16 mbarriers
// MODE 3: 64 mbarriers
#ifndef MODE
#define MODE 0
#endif

#if MODE == 0
#define N_BARS 1
#elif MODE == 1
#define N_BARS 4
#elif MODE == 2
#define N_BARS 16
#elif MODE == 3
#define N_BARS 64
#endif

extern "C" __global__ __launch_bounds__(128, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(8) unsigned long long bars[N_BARS];

    if (threadIdx.x == 0) {
        for (int i = 0; i < N_BARS; i++) {
            unsigned int bar_addr = __cvta_generic_to_shared(&bars[i]);
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 128;" :: "r"(bar_addr));
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // Round-robin arrive on N_BARS barriers
        unsigned int bar_addr = __cvta_generic_to_shared(&bars[i % N_BARS]);
        asm volatile("{ .reg .b64 state;\n"
                     "  mbarrier.arrive.shared::cta.b64 state, [%0]; }"
                     :: "r"(bar_addr));
    }

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d N_BARS=%d clk=%llu cy/arrive=%.3f\n",
               MODE, N_BARS, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
