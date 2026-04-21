// V9: HMMA single-instruction latency via serial dependency chain
// Each mma.sync depends on prev accum result → latency-bound
#ifndef CHAIN_LEN
#define CHAIN_LEN 1024
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;  // 1 warp only

    // HMMA.F16.F32 fragments
    unsigned a0 = (unsigned)(threadIdx.x * 17 + seed);
    unsigned a1 = (unsigned)(threadIdx.x * 23 + seed + 1);
    unsigned a2 = (unsigned)(threadIdx.x * 29 + seed + 2);
    unsigned a3 = (unsigned)(threadIdx.x * 31 + seed + 3);
    unsigned b0 = (unsigned)(threadIdx.x * 37 + seed + 4);
    unsigned b1 = (unsigned)(threadIdx.x * 41 + seed + 5);

    // F32 accumulator: 4 × f32 per thread
    float c0 = (float)(threadIdx.x);
    float c1 = (float)(threadIdx.x + 10);
    float c2 = (float)(threadIdx.x + 100);
    float c3 = (float)(threadIdx.x + 1000);

    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // Serial HMMA chain — each depends on prev
    #pragma unroll 16
    for (int i = 0; i < CHAIN_LEN; i++) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            " {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1)
        );
    }

    if (threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        ((unsigned long long*)C)[0] = t1 - t0;
        ((float*)C)[2] = c0 + c1 + c2 + c3;  // Anti-DCE
    }
}
