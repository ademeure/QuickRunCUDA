// V5 E4: bar.warp variants
// MODE 0: bar.warp.sync (full warp, all)
// MODE 1: bar.warp.sync 0xFFFFFFFF (explicit mask)
// MODE 2: bar.warp.sync.aligned 0xFFFFFFFF
// MODE 3: __syncwarp() (C wrapper)
// MODE 4: bar.warp.arrive (Hopper+ split-phase)
// MODE 5: bar.warp.wait
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
        v ^= i;
#if MODE == 0
        asm volatile("bar.warp.sync 0xFFFFFFFF;");
#elif MODE == 1
        asm volatile("bar.warp.sync 0xFFFFFFFF;");
#elif MODE == 2
        // bar.warp.sync.aligned doesn't exist in PTX; bar.sync 0 has aligned variant
        asm volatile("bar.warp.sync 0xFFFFFFFF;");
#elif MODE == 3
        __syncwarp();
#elif MODE == 4
        // bar.warp.arrive — does this PTX exist?
        asm volatile("bar.warp.arrive 0xFFFFFFFF;");
#elif MODE == 5
        asm volatile("bar.warp.wait 0xFFFFFFFF;");
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.3f\n",
               MODE, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
