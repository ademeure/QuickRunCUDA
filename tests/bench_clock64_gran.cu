// clock64 granularity check
extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned long long ticks[64];
    for (int i = 0; i < 64; i++) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(ticks[i]));
    }

    printf("clock64 deltas (cycles):\n");
    for (int i = 1; i < 16; i++) {
        printf("  delta[%d] = %llu cy\n", i, ticks[i] - ticks[i-1]);
    }

    unsigned long long min_delta = 0xFFFFFFFFFFFFFFFFull;
    int count_zero = 0;
    for (int i = 1; i < 64; i++) {
        unsigned long long d = ticks[i] - ticks[i-1];
        if (d == 0) count_zero++;
        if (d > 0 && d < min_delta) min_delta = d;
    }
    printf("MIN nonzero delta: %llu cycles\n", min_delta);
    printf("Zero-delta reads: %d / 63\n", count_zero);

    if (ticks[63] == (unsigned long long)seed) C[0] = (float)ticks[63];
}
