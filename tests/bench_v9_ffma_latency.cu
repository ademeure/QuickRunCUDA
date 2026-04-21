// V9: FFMA pipeline latency — serial dependency chain
// Single-thread, single-warp measurement to avoid SMSP contention.
// Each FFMA in chain depends on previous result → latency-bound.
#ifndef CHAIN_LEN
#define CHAIN_LEN 1024
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // Only 1 warp runs; other threads idle. Thread 0 measures.
    if (threadIdx.x != 0) return;

    float a = (float)(seed + 1) * 0.001f;
    float b = (float)(seed + 2) * 0.001f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // Serial dependency chain: each FFMA depends on prev
    #pragma unroll 32
    for (int i = 0; i < CHAIN_LEN; i++) {
        asm volatile("fma.rn.f32 %0, %0, %1, %0;" : "+f"(a) : "f"(b));
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    unsigned long long cycles = t1 - t0;

    if (blockIdx.x == 0) {
        // Write cycles and final a
        ((unsigned long long*)C)[0] = cycles;
        ((float*)C)[2] = a;
    }
}
