// __syncwarp cycle cost on B300.
// Mode 0: no syncwarp (baseline)
// Mode 1: __syncwarp(0xFFFFFFFF) every iter
// Mode 2: __syncwarp with restricted mask 0x0000FFFF (16 lanes)
// Mode 3: bar.warp.sync via inline PTX
// Mode 4: __syncthreads() (single-warp block)
// Mode 5: nanosleep equivalent (control)

#ifndef N_OPS
#define N_OPS 8
#endif
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v = (unsigned)(threadIdx.x * 131);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // Some compute work per iteration
        #pragma unroll
        for (int k = 0; k < N_OPS; k++) {
            v = v * 31u + (unsigned)i + k;
        }
#if MODE == 0
        // baseline, no sync
#elif MODE == 1
        __syncwarp(0xFFFFFFFFu);
#elif MODE == 2
        __syncwarp(0x0000FFFFu);
#elif MODE == 3
        asm volatile("bar.warp.sync 0xFFFFFFFF;");
#elif MODE == 4
        __syncthreads();
#elif MODE == 5
        // light alternative: just memory fence (no cross-thread sync)
        asm volatile("membar.cta;");
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d iters=%d clk=%llu cy/iter=%.3f\n",
               MODE, ITERS, t1 - t0, (double)(t1-t0)/(double)ITERS);
    }
}
