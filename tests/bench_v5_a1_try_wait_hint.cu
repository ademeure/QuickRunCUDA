// V5 A1: mbarrier.try_wait suspend time hint
// PTX: mbarrier.try_wait.shared::cta.b64 p, [bar], phase, suspendTimeHint
// suspendTimeHint = uint64 — hint for HW for how long to wait/suspend
// Test various values
#ifndef HINT
#define HINT 0
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

    unsigned long long phase = 0;
    unsigned int suspendHint = (unsigned int)HINT;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // arrive (all 128)
        asm volatile("{ .reg .b64 state;\n"
                     "  mbarrier.arrive.shared::cta.b64 state, [%0]; }"
                     :: "r"(bar_addr));
        // try_wait with phase + suspendTimeHint
        unsigned int done = 0;
        do {
            asm volatile("{ .reg .pred p;\n"
                         "  mbarrier.try_wait.shared::cta.b64 p, [%1], %2, %3;\n"
                         "  selp.b32 %0, 1, 0, p; }"
                         : "=r"(done) : "r"(bar_addr), "l"(phase), "r"(suspendHint));
        } while (!done);
        phase ^= 1;
    }

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("HINT=%lld clk=%llu cy/iter=%.3f\n",
               (long long)HINT, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
