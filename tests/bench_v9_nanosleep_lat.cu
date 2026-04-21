// V9: __nanosleep(N) actual latency
// Documented max = 1 ms. Measure real timings at various N values.
extern "C" __global__ __launch_bounds__(32, 1)
void kernel(unsigned long long* A, unsigned long long* B, unsigned long long* C,
            int ITERS, int seed, int u2) {
    if (threadIdx.x != 0) return;

    // Measurements across N sweeps
    unsigned long long t0, t1;
    unsigned long long results[16] = {0};

    // N values to test: 100, 500, 1000, 5000, 10000, 50000, 100000, 500000 ns
    const int ns_values[8] = {100, 500, 1000, 5000, 10000, 50000, 100000, 500000};

    for (int i = 0; i < 8; i++) {
        int N = ns_values[i];
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
        #pragma unroll 1
        for (int r = 0; r < 100; r++) {
            asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)N));
        }
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        results[i] = (t1 - t0) / 100;   // avg ns per nanosleep
    }

    if (blockIdx.x == 0) {
        for (int i = 0; i < 8; i++) C[i] = results[i];
    }
}
