// Asymmetric: some SMs are "HEAVY" (bs=1024 × W=16 = 10K+ tier),
// others "LIGHT" (bs=32 × W=1 = 5K tier). Measure per-SM fence cost.

#ifndef ITERS
#define ITERS 100
#endif
#ifndef HEAVY_SMS
#define HEAVY_SMS 74  // first N CTAs are heavy (bs=1024)
#endif
#ifndef LIGHT_BS
#define LIGHT_BS 32
#endif

extern "C" __global__ __launch_bounds__(1024, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned* my_addr = A + tid * 16;
    bool is_heavy = (blockIdx.x < HEAVY_SMS);

    // HEAVY: all 1024 threads write + fence
    // LIGHT: only first LIGHT_BS threads (=1 warp) write + fence, rest return
    if (!is_heavy && threadIdx.x >= LIGHT_BS) return;

    int nw = is_heavy ? 16 : 1;

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
    // 1 thread per CTA reports
    if (threadIdx.x == 0) C[blockIdx.x] = total / ITERS;
}
