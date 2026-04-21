// V9: nanosleep — per-lane or per-warp behavior?
// Test: half warp sleeps long, half sleeps short. Does whole warp wait for max?
#ifndef MODE
#define MODE 0  // 0=all sleep same, 1=divergent sleep, 2=lane 0 only sleeps
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(unsigned long long* A, unsigned long long* B, unsigned long long* C,
            int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;
    int lane = threadIdx.x & 31;

    unsigned long long t0, t1;
    if (lane == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    __syncwarp();

    // Loop for stable timing
    #pragma unroll 1
    for (int i = 0; i < 100; i++) {
#if MODE == 0
        asm volatile("nanosleep.u32 1000;");
#elif MODE == 1
        int N = (lane < 16) ? 1000 : 100;
        asm volatile("nanosleep.u32 %0;" :: "r"(N));
#elif MODE == 2
        if (lane == 0) asm volatile("nanosleep.u32 1000;");
#elif MODE == 3
        asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)(100 + lane * 100)));
#endif
        __syncwarp();
    }
    if (lane == 0) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        ((unsigned long long*)C)[0] = t1 - t0;
    }
}
