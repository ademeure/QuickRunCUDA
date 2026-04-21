// V9: mbarrier (async barrier) latency
// Test: arrive + wait round trip in a single thread
#ifndef CHAIN_LEN
#define CHAIN_LEN 1000
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(unsigned long long* A, unsigned long long* B, unsigned long long* C,
            int ITERS, int seed, int u2) {
    __shared__ alignas(8) unsigned long long bar;
    if (threadIdx.x != 0) return;

    unsigned int bar_addr = (unsigned)__cvta_generic_to_shared(&bar);

    // Initialize mbarrier with arrival count = 1
    asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(bar_addr));

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // Chain: arrive + wait, repeat
    #pragma unroll 1
    for (int i = 0; i < CHAIN_LEN; i++) {
        unsigned long long state;
        // arrive returns the previous state token (used for wait)
        asm volatile(
            "mbarrier.arrive.shared.b64 %0, [%1];\n"
            : "=l"(state) : "r"(bar_addr)
        );
        // wait: spins until barrier completes
        asm volatile(
            "{\n"
            ".reg .pred P;\n"
            "L_wait_%=:\n"
            "mbarrier.test_wait.shared.b64 P, [%0], %1;\n"
            "@!P bra L_wait_%=;\n"
            "}\n"
            :: "r"(bar_addr), "l"(state)
        );
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (blockIdx.x == 0) ((unsigned long long*)C)[0] = t1 - t0;
}
