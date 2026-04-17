// Test membar.sys under various concurrency scenarios.
//
// OP=0: 1 SM (1 CTA, 1 warp), 100 sequential membar.sys, no writes
// OP=1: 1 SM, 100 sequential membar.sys, with 1 GPU write before each
// OP=2: All SMs (148 CTAs), 100 sequential membar.sys each, no writes
// OP=3: All SMs, 100 sequential membar.sys, with 1 GPU write each iter
// OP=4: 1 SM with membar.sys, other 147 SMs doing heavy GPU writes (contention)

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef ITERS
#define ITERS 100
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned* my_addr = A + tid;

#if OP == 4
    // Mode 4: SM 0 does membar.sys, others do heavy writes
    if (blockIdx.x == 0) {
        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        #pragma unroll 1
        for (int i = 0; i < ITERS; i++) {
            asm volatile("membar.sys;");
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        if (threadIdx.x == 0) {
            ((unsigned long long*)C)[0] = t1 - t0;
        }
    } else {
        // Other SMs: heavy global writes
        for (int i = 0; i < 100000; i++) {
            *(volatile unsigned*)my_addr = i;
        }
    }
#else
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    unsigned acc = seed;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if OP == 0
        // 1 SM, just membar.sys, no writes
        asm volatile("membar.sys;");
#elif OP == 1
        // 1 SM, 1 write + membar.sys
        *(volatile unsigned*)my_addr = i;
        asm volatile("membar.sys;");
#elif OP == 2
        // All SMs, just membar.sys
        asm volatile("membar.sys;");
#elif OP == 3
        // All SMs, 1 write + membar.sys
        *(volatile unsigned*)my_addr = i;
        asm volatile("membar.sys;");
#endif
        acc ^= i;
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        C[2] = acc;
    }
#endif
}
