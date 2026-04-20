// Named barrier test: bar.sync 0..15 — independent or shared resources?
// 4-warp block (128 threads). Each warp uses different barrier IDs.
//
// MODE 0: bar.sync 0 only (full block barrier, all 4 warps participate)
// MODE 1: bar.sync 1 (different barrier ID, same all-warp barrier)
// MODE 2: alternating bar.sync 0, 1 each iteration
// MODE 3: bar.sync 0 with 64-thread participation (split-warp pattern)
// MODE 4: 4 different barriers (0,1,2,3) cycled

#ifndef MODE
#define MODE 0
#endif
#ifndef N_ITERS
#define N_ITERS 10000
#endif

extern "C" __global__ __launch_bounds__(128, 8)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v = (unsigned)threadIdx.x;

    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < N_ITERS; i++) {
        v = v * 31u + (unsigned)i;
#if MODE == 0
        asm volatile("bar.sync 0, 128;");
#elif MODE == 1
        asm volatile("bar.sync 1, 128;");
#elif MODE == 2
        if (i & 1) asm volatile("bar.sync 0, 128;");
        else       asm volatile("bar.sync 1, 128;");
#elif MODE == 3
        // Split: warps 0-1 use bar 0, warps 2-3 use bar 1, both with 64 threads
        if (threadIdx.x < 64) asm volatile("bar.sync 0, 64;");
        else                  asm volatile("bar.sync 1, 64;");
#elif MODE == 4
        // 4 different barriers cycled
        switch (i & 3) {
            case 0: asm volatile("bar.sync 0, 128;"); break;
            case 1: asm volatile("bar.sync 1, 128;"); break;
            case 2: asm volatile("bar.sync 2, 128;"); break;
            case 3: asm volatile("bar.sync 3, 128;"); break;
        }
#endif
    }

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d N_ITERS=%d clk=%llu cy/iter=%.3f\n",
               MODE, N_ITERS, t1 - t0, (double)(t1-t0)/(double)N_ITERS);
    }
}
