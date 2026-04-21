// V5 A2: mbarrier wait power vs busy spin power
// Per-block test: thread 0 is the "trigger", other threads wait
//   MODE 0: busy spin (volatile load)
//   MODE 1: mbarrier.try_wait.parity
//   MODE 2: __nanosleep loop
// Run with persistent kernel (148 blocks); thread 0 holds for ITERS µs
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(8) unsigned long long bar;
    __shared__ volatile unsigned int signal;
    unsigned int bar_addr = __cvta_generic_to_shared(&bar);

    if (threadIdx.x == 0) {
        signal = 0;
#if MODE == 1
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                     :: "r"(bar_addr), "r"((unsigned int)256));
        asm volatile("fence.mbarrier_init.release.cluster;");
#endif
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // Trigger thread: hold for ITERS microseconds then release
        for (int i = 0; i < ITERS; i++) {
            __nanosleep(1000);
        }
        signal = 1;
#if MODE == 1
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                     :: "r"(bar_addr));
#endif
    } else {
#if MODE == 0
        while (signal == 0) {}
#elif MODE == 1
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                     :: "r"(bar_addr));
        asm volatile("{ .reg .pred p;\n"
                     "L_w_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n"
                     "  @!p bra L_w_%=; }"
                     :: "r"(bar_addr));
#elif MODE == 2
        while (signal == 0) {
            __nanosleep(100);
        }
#endif
    }

    if (signal == 0xDEADBEEF) C[blockIdx.x] = 1.0f;
}
