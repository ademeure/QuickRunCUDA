// V6 A6: 2 distinct HMMA chains in same warp
// Q: Can the tensor pipe execute multiple HMMA from one warp in parallel?
// MODE 0: 4× HMMA chain A only
// MODE 1: 4× HMMA chain B only (independent regs)
// MODE 2: 4× chain A + 4× chain B interleaved
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    // Chain A
    unsigned int aA0=0x3F803F80, aA1=0x3F803F80, aA2=0x3F803F80, aA3=0x3F803F80;
    unsigned int bA0=0x3F803F80, bA1=0x3F803F80;
    float cA0=0.0f, cA1=0.0f, cA2=0.0f, cA3=0.0f;

    // Chain B (independent regs)
    unsigned int aB0=0x3F803F80, aB1=0x3F803F80, aB2=0x3F803F80, aB3=0x3F803F80;
    unsigned int bB0=0x3F803F80, bB1=0x3F803F80;
    float cB0=0.0f, cB1=0.0f, cB2=0.0f, cB3=0.0f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0 || MODE == 2
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+f"(cA0), "+f"(cA1), "+f"(cA2), "+f"(cA3)
                : "r"(aA0), "r"(aA1), "r"(aA2), "r"(aA3), "r"(bA0), "r"(bA1));
        }
#endif
#if MODE == 1 || MODE == 2
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+f"(cB0), "+f"(cB1), "+f"(cB2), "+f"(cB3)
                : "r"(aB0), "r"(aB1), "r"(aB2), "r"(aB3), "r"(bB0), "r"(bB1));
        }
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

#if MODE == 0 || MODE == 2
    if (cA0 == 1.234567e-30f) C[blockIdx.x] = cA0+cA1+cA2+cA3;
#endif
#if MODE == 1 || MODE == 2
    if (cB0 == 1.234567e-30f) ((float*)C)[blockIdx.x+1] = cB0+cB1+cB2+cB3;
#endif

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d ITERS=%d total_cy=%llu cy/iter=%.3f\n",
               MODE, ITERS, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
