// V8 D1: 2:4 sparse mma.sp.sync vs dense mma.sync
// MODE 0: dense mma.sync.m16n8k16 f32.bf16.bf16.f32 (ILP 4)
// MODE 1: sparse mma.sp.sync.m16n8k32 with 2:4 sparsity metadata
// For sparse: k doubles to 32, so 2× throughput if HW supports
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int a0=0x3F803F80, a1=0x3F803F80, a2=0x3F803F80, a3=0x3F803F80;
    unsigned int b0=0x3F803F80, b1=0x3F803F80;
    // Sparse metadata: 4 bits per 2-of-4 selection; 32 threads × 4 lanes × ...
    unsigned int meta = 0xEEEEEEEE;  // "11 10 11 10..." pattern (element 2 non-zero, 0 zero, etc.)

    // 4 independent chains for ILP=4
    float c0[4][4];
    #pragma unroll
    for (int i=0; i<4; i++) { c0[i][0]=0; c0[i][1]=0; c0[i][2]=0; c0[i][3]=0; }

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
#if MODE == 0
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+f"(c0[j][0]), "+f"(c0[j][1]), "+f"(c0[j][2]), "+f"(c0[j][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif MODE == 1
            // 2:4 sparse mma.sp — m16n8k32 (2× k depth with sparsity)
            asm volatile(
                "mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%0, %1, %2, %3}, %12, 0x0;"
                : "+f"(c0[j][0]), "+f"(c0[j][1]), "+f"(c0[j][2]), "+f"(c0[j][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b0), "r"(b1), "r"(meta));
#endif
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    float sum = 0;
    #pragma unroll
    for (int i=0; i<4; i++) sum += c0[i][0];
    if (sum == 1.234567e-30f) C[blockIdx.x] = sum;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double cy_per_iter = (double)(t1-t0)/(double)ITERS;
        printf("MODE=%d cy/iter=%.3f cy/mma=%.3f (ILP=4 chains)\n",
               MODE, cy_per_iter, cy_per_iter/4.0);
    }
}
