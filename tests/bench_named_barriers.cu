// bar.sync 0..15 named barriers: independent or shared?
// Use 8 warps (256 threads). Test cycle cost of bar.sync on different barrier IDs.

#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v = (unsigned)(threadIdx.x);

    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        v = v * 31u + (unsigned)i;
#if MODE == 0
        // No barrier
#elif MODE == 1
        // bar.sync 0 (full block barrier)
        asm volatile("bar.sync 0;");
#elif MODE == 2
        // bar.sync 1
        asm volatile("bar.sync 1;");
#elif MODE == 3
        // bar.sync 2
        asm volatile("bar.sync 2;");
#elif MODE == 4
        // Alternating bar.sync 0, 1
        if (i & 1) asm volatile("bar.sync 0;");
        else       asm volatile("bar.sync 1;");
#elif MODE == 5
        // bar.arrive 0 + bar.sync 0 (split-mode pipelining)
        asm volatile("bar.arrive 0, 256;");
        asm volatile("bar.sync 0, 256;");
#elif MODE == 6
        // 4 different barriers cycled
        switch (i & 3) {
            case 0: asm volatile("bar.sync 0;"); break;
            case 1: asm volatile("bar.sync 1;"); break;
            case 2: asm volatile("bar.sync 2;"); break;
            case 3: asm volatile("bar.sync 3;"); break;
        }
#endif
    }

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d iters=%d clk=%llu cy/iter=%.3f\n",
               MODE, ITERS, t1 - t0, (double)(t1-t0)/(double)ITERS);
    }
}
