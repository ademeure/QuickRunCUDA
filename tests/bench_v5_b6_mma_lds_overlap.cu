// V5 B6: mma.sync + LDS overlap test
// Hypothesis: tensor pipe and LSU pipe are independent → can run concurrently
//   MODE 0: HMMA only (8 chained mma.sync ops, ILP=8 baseline)
//   MODE 1: LDS only (shared memory loads, equivalent count)
//   MODE 2: HMMA + LDS interleaved (test overlap)
//
// If they overlap: MODE 2 ≈ max(MODE 0, MODE 1) (parallelism)
// If serialized: MODE 2 ≈ MODE 0 + MODE 1 (sum)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    __shared__ __align__(16) float smem[1024];
    if (threadIdx.x == 0) {
        for (int i = 0; i < 1024; i++) smem[i] = (float)i;
    }
    __syncwarp();

    // mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 d, a, b, c
    // We need a (4 fp32), b (2 fp32), c (4 fp32), produces d (4 fp32)
    unsigned int a0 = 0x3F803F80;  // BF16(1.0, 1.0)
    unsigned int a1 = 0x3F803F80;
    unsigned int a2 = 0x3F803F80;
    unsigned int a3 = 0x3F803F80;
    unsigned int b0 = 0x3F803F80;
    unsigned int b1 = 0x3F803F80;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    float load0 = 0, load1 = 0, load2 = 0, load3 = 0;
    int idx = threadIdx.x;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0 || MODE == 2
        // HMMA chain (8 ops)
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
#endif
#if MODE == 1 || MODE == 2
        // LDS chain (8 ops)
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float v;
            unsigned int sa = (unsigned int)__cvta_generic_to_shared(&smem[(idx + k * 32 + i) & 1023]);
            asm volatile("ld.shared.f32 %0, [%1];"
                         : "=f"(v) : "r"(sa));
            load0 += v;
        }
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    // Anti-DCE
    float sum = c0 + c1 + c2 + c3 + load0 + load1 + load2 + load3;
    if (sum == 1.234567e-30f) C[blockIdx.x] = sum;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d ITERS=%d total_cy=%llu cy/iter=%.3f\n",
               MODE, ITERS, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
