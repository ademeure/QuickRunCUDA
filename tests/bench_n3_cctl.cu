// N3: PTX cctl (cache control) instructions
// Test: cctl.ivall (invalidate all L1 lines), cctl.wb (writeback dirty lines)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int v = (unsigned)u2;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // Touch some memory to dirty L1
        asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(A + (i & 1023)));
#if MODE == 0
        // Baseline: no cctl
#elif MODE == 1
        // cctl.ivall (invalidate all L1) - all lines dropped
        asm volatile("cctl.ivall.L1;");
#elif MODE == 2
        // cctl.wb.L1 (writeback dirty lines)
        asm volatile("cctl.wb.L1;");
#elif MODE == 3
        // cctl.iv (invalidate single addr)
        asm volatile("cctl.iv.L1 [%0];" :: "l"(A));
#elif MODE == 4
        // discard (newer)
        asm volatile("discard.global.L2 [%0], 128;" :: "l"(A));
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.3f\n",
               MODE, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
