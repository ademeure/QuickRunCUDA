// V6 J1: HMMA latency vs ILP sweep
// MODE selects ILP from 1, 2, 4, 8, 16
// Each chain is 64 mma.sync ops; total = ILP × 64 instructions
#ifndef MODE
#define MODE 0
#endif

#if MODE == 0
#define ILP 1
#elif MODE == 1
#define ILP 2
#elif MODE == 2
#define ILP 4
#elif MODE == 3
#define ILP 8
#elif MODE == 4
#define ILP 16
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int a0=0x3F803F80, a1=0x3F803F80, a2=0x3F803F80, a3=0x3F803F80;
    unsigned int b0=0x3F803F80, b1=0x3F803F80;

    // ILP independent accumulators
    float c[ILP][4];
    #pragma unroll
    for (int j = 0; j < ILP; j++) {
        c[j][0] = 0.0f; c[j][1] = 0.0f; c[j][2] = 0.0f; c[j][3] = 0.0f;
    }

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // Issue ILP independent HMMA chains, 64 deep each
        #pragma unroll
        for (int k = 0; k < 64; k++) {
            #pragma unroll
            for (int j = 0; j < ILP; j++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                    : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    float sum = 0;
    #pragma unroll
    for (int j = 0; j < ILP; j++) {
        sum += c[j][0] + c[j][1] + c[j][2] + c[j][3];
    }
    if (sum == 1.234567e-30f) C[blockIdx.x] = sum;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // cy/iter / (ILP × 64) = cy/HMMA
        double cy_per_iter = (double)(t1-t0) / (double)ITERS;
        double cy_per_mma = cy_per_iter / (ILP * 64.0);
        printf("MODE=%d ILP=%d cy/iter=%.1f cy/mma=%.3f\n",
               MODE, ILP, cy_per_iter, cy_per_mma);
    }
}
