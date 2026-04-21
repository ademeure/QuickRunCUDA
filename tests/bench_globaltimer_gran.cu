// globaltimer granularity test — check min tick
extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned long long ticks[64];
    for (int i = 0; i < 64; i++) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ticks[i]));
    }

    // Print all deltas
    printf("globaltimer ticks (consecutive reads, ns):\n");
    for (int i = 1; i < 16; i++) {
        printf("  delta[%d] = %llu ns\n", i, ticks[i] - ticks[i-1]);
    }

    // Find min nonzero delta
    unsigned long long min_delta = 0xFFFFFFFFFFFFFFFFull;
    for (int i = 1; i < 64; i++) {
        unsigned long long d = ticks[i] - ticks[i-1];
        if (d > 0 && d < min_delta) min_delta = d;
    }
    printf("MIN nonzero delta: %llu ns\n", min_delta);

    if (ticks[63] == (unsigned long long)seed) C[0] = (float)ticks[63];
}
