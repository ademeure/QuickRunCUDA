// Scoreboard slot count test.
// Issue N independent LDGs from one warp, then consume them all.
// If N <= scoreboard depth, all issue back-to-back at peak rate.
// If N > scoreboard depth, the (depth+1)th issue stalls until earlier
// completes. This shows up as cy/load increasing.

#ifndef N_LOADS
#define N_LOADS 8
#endif
#ifndef N_OUTER
#define N_OUTER 1000
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int* ai = (unsigned int*)A;
    unsigned int regs[32];  // up to 32 in-flight loads

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < N_OUTER; i++) {
        // Issue N_LOADS LDGs to distinct cache lines (no addr dep on each other)
        // Each load index depends on i so DCE can't hoist across outer iters.
        #pragma unroll
        for (int k = 0; k < N_LOADS; k++) {
            unsigned int idx = (i * 256 + k * 256 + threadIdx.x) & 0x3FFF;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(regs[k]) : "l"(ai + idx));
        }
        // Now consume them — XOR all into regs[0] which is checked
        #pragma unroll
        for (int k = 1; k < N_LOADS; k++) {
            regs[0] ^= regs[k];
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if ((int)regs[0] == seed) ((unsigned*)C)[blockIdx.x] = regs[0];
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long total_loads = (unsigned long long)N_OUTER * N_LOADS;
        printf("N_LOADS=%d N_OUTER=%d clk=%llu cy/load=%.3f\n",
               N_LOADS, N_OUTER, t1 - t0, (double)(t1-t0)/(double)total_loads);
    }
}
