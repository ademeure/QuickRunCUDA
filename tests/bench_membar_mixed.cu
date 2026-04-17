// Mixed-load test: some SMs have 1 write, others have 16 writes.
// Measure fence cost on each subset independently.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef ACTIVE_CTAS
#define ACTIVE_CTAS 148
#endif
#ifndef LIGHT_SMS
#define LIGHT_SMS 74  // first 74 SMs do W=1
#endif
#ifndef HEAVY_W
#define HEAVY_W 16
#endif
#ifndef ITERS
#define ITERS 200
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    if (blockIdx.x >= ACTIVE_CTAS) return;
    if (threadIdx.x != 0) return;
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned* my_addr = A + tid * HEAVY_W;

    // LIGHT SMs: W=1. HEAVY SMs: W=HEAVY_W
    int nw = (blockIdx.x < LIGHT_SMS) ? 1 : HEAVY_W;

    unsigned long long t0, t1;
    unsigned total = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        for (int j = 0; j < nw; j++) ((volatile unsigned*)my_addr)[j] = i + seed + j;
        asm volatile("fence.sc.sys;");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        total += (unsigned)(t1 - t0);
    }
    C[blockIdx.x] = total / ITERS;
}
