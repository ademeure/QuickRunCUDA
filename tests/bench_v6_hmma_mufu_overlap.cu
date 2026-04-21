// V6 A3: HMMA (tensor) + MUFU (xu pipe) overlap test
// MODE 0: 4× HMMA only
// MODE 1: 8× MUFU.RCP only
// MODE 2: combined
// Prediction: ~30% if scheduler-bound (matches A1/A2)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int a0=0x3F803F80, a1=0x3F803F80, a2=0x3F803F80, a3=0x3F803F80;
    unsigned int b0=0x3F803F80, b1=0x3F803F80;
    float c0=0.0f, c1=0.0f, c2=0.0f, c3=0.0f;

    float a = (float)(threadIdx.x + 1) * 0.001f;
    float r0=a, r1=a*1.1f, r2=a*1.2f, r3=a*1.3f;
    float r4=a*1.4f, r5=a*1.5f, r6=a*1.6f, r7=a*1.7f;

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
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
#endif
#if MODE == 1 || MODE == 2
        // 8 MUFU.RCP via PTX rcp.approx — this maps to MUFU.RCP SASS
        asm volatile("rcp.approx.f32 %0, %0;" : "+f"(r0));
        asm volatile("rcp.approx.f32 %0, %0;" : "+f"(r1));
        asm volatile("rcp.approx.f32 %0, %0;" : "+f"(r2));
        asm volatile("rcp.approx.f32 %0, %0;" : "+f"(r3));
        asm volatile("rcp.approx.f32 %0, %0;" : "+f"(r4));
        asm volatile("rcp.approx.f32 %0, %0;" : "+f"(r5));
        asm volatile("rcp.approx.f32 %0, %0;" : "+f"(r6));
        asm volatile("rcp.approx.f32 %0, %0;" : "+f"(r7));
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

#if MODE == 0 || MODE == 2
    if (c0 == 1.234567e-30f) C[blockIdx.x] = c0+c1+c2+c3;
#endif
#if MODE == 1 || MODE == 2
    float rsum = r0+r1+r2+r3+r4+r5+r6+r7;
    if (rsum == 1.234567e-30f) ((float*)C)[blockIdx.x+1] = rsum;
#endif

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d ITERS=%d total_cy=%llu cy/iter=%.3f\n",
               MODE, ITERS, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
