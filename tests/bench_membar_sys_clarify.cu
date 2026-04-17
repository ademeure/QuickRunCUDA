// Clarify "no traffic" membar.sys by testing different backgrounds:
// OP=0: 1 SM only, rest IDLE (no kernel on other SMs)
// OP=1: 1 SM doing membar, other SMs SPIN but no writes
// OP=2: 1 SM doing membar, other SMs IDLE (actually just return)
// OP=3: Full chip, every SM does membar.sys simultaneously (baseline full-chip no-writes)

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

#if OP == 0 || OP == 2
    // Only block 0 measures; others just exit
    if (blockIdx.x != 0) return;
    if (threadIdx.x != 0) return;
    unsigned long long t0, t1;
    unsigned total = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        asm volatile("membar.sys;");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        total += (unsigned)(t1 - t0);
    }
    if (threadIdx.x == 0) ((unsigned long long*)C)[0] = total / ITERS;
#elif OP == 1
    // Block 0 measures, others spin (no writes) forever
    if (blockIdx.x == 0) {
        if (threadIdx.x != 0) return;
        unsigned long long t0, t1;
        unsigned total = 0;
        #pragma unroll 1
        for (int i = 0; i < ITERS; i++) {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
            asm volatile("membar.sys;");
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
            total += (unsigned)(t1 - t0);
        }
        ((unsigned long long*)C)[0] = total / ITERS;
    } else {
        // Spin — no memory traffic
        volatile int x = tid;
        for (int i = 0; i < 1000000; i++) x = x * 1664525 + i;
        if (x == 0xDEADBEEF) C[1] = x;  // prevent DCE
    }
#elif OP == 3
    // All SMs do membar.sys
    if (threadIdx.x != 0) return;
    unsigned long long t0, t1;
    unsigned total = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        asm volatile("membar.sys;");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        total += (unsigned)(t1 - t0);
    }
    C[blockIdx.x] = total / ITERS;
#endif
}
