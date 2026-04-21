// F3: mbarrier.arrive_drop semantics
// Test: does .arrive_drop reduce the EXPECTED arrival count, or just decrement?
// Behavior under PTX 7.0+: arrive_drop decrements expected by 1 AND arrives once.
// MODE 0: regular arrive
// MODE 1: arrive_drop on half the warp — does the bar still fire correctly?
// MODE 2: arrive_drop on ALL — bar should fire after each drop
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(128, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ unsigned long long bar_storage[1];

    unsigned int bar_addr_init = __cvta_generic_to_shared(bar_storage);
    if (threadIdx.x == 0) {
        // Init mbarrier with expected = 128 (full block)
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 128;" :: "r"(bar_addr_init));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    unsigned int bar_addr = __cvta_generic_to_shared(bar_storage);

#if MODE == 0
    // Pure arrive — every thread arrives, total 128 arrivals = expected 128
    asm volatile("{ .reg .b64 state;\n"
                 "  mbarrier.arrive.shared::cta.b64 state, [%0]; }" :: "r"(bar_addr));
#elif MODE == 1
    // Half-warp does arrive_drop (which decrements expected too); other half does regular
    if (threadIdx.x < 64) {
        asm volatile("{ .reg .b64 state;\n"
                     "  mbarrier.arrive_drop.shared::cta.b64 state, [%0]; }" :: "r"(bar_addr));
    } else {
        asm volatile("{ .reg .b64 state;\n"
                     "  mbarrier.arrive.shared::cta.b64 state, [%0]; }" :: "r"(bar_addr));
    }
#elif MODE == 2
    // All threads do arrive_drop (decrements expected to 0; immediate complete)
    asm volatile("{ .reg .b64 state;\n"
                 "  mbarrier.arrive_drop.shared::cta.b64 state, [%0]; }" :: "r"(bar_addr));
#endif

    // Wait for barrier to complete
    asm volatile("{ .reg .pred p;\n"
                 "L_wait_%=: mbarrier.try_wait.shared::cta.b64 p, [%0], 0;\n"
                 "  @!p bra L_wait_%=; }" :: "r"(bar_addr));

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d clk=%llu (mbarrier wait time)\n", MODE, t1-t0);
    }
}
