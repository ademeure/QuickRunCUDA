// bar.arrive (split-phase) vs bar.sync (full barrier)
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
        // Full sync barrier (everyone arrives + waits)
        asm volatile("bar.sync 0, 128;");
#elif MODE == 1
        // Split: arrive then immediate sync (same as full)
        asm volatile("bar.arrive 0, 128;");
        asm volatile("bar.sync 0, 128;");
#elif MODE == 2
        // Split with work between arrive and sync
        asm volatile("bar.arrive 0, 128;");
        v = v * 17u + (unsigned)i;  // do unrelated work
        v = v * 13u + (unsigned)i;
        v = v * 11u + (unsigned)i;
        v = v * 7u + (unsigned)i;
        asm volatile("bar.sync 0, 128;");
#elif MODE == 3
        // bar.arrive only (consumer pattern - producer doesn't wait)
        if (threadIdx.x < 64) {
            asm volatile("bar.arrive 0, 128;");
        } else {
            asm volatile("bar.sync 0, 128;");
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
