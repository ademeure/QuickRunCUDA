// V6 A2: HMMA (tensor pipe) + IADD3 (alu pipe) overlap test
// MODE 0: 4× HMMA only
// MODE 1: 8× IADD3 only (8 chains for ILP)
// MODE 2: combined
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int a0=0x3F803F80, a1=0x3F803F80, a2=0x3F803F80, a3=0x3F803F80;
    unsigned int b0=0x3F803F80, b1=0x3F803F80;
    float c0=0.0f, c1=0.0f, c2=0.0f, c3=0.0f;

    int x = threadIdx.x + 1;
    int k = blockIdx.x + 1;
    int i0=x, i1=x, i2=x, i3=x, i4=x, i5=x, i6=x, i7=x;

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
        // 8 add.s32 (compiles to IADD3); volatile prevents CSE/DCE
        asm volatile("add.s32 %0, %0, %1;" : "+r"(i0) : "r"(k+i));
        asm volatile("add.s32 %0, %0, %1;" : "+r"(i1) : "r"(k+i+1));
        asm volatile("add.s32 %0, %0, %1;" : "+r"(i2) : "r"(k+i+2));
        asm volatile("add.s32 %0, %0, %1;" : "+r"(i3) : "r"(k+i+3));
        asm volatile("add.s32 %0, %0, %1;" : "+r"(i4) : "r"(k+i+4));
        asm volatile("add.s32 %0, %0, %1;" : "+r"(i5) : "r"(k+i+5));
        asm volatile("add.s32 %0, %0, %1;" : "+r"(i6) : "r"(k+i+6));
        asm volatile("add.s32 %0, %0, %1;" : "+r"(i7) : "r"(k+i+7));
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

#if MODE == 0 || MODE == 2
    if (c0 == 1.234567e-30f) C[blockIdx.x] = c0+c1+c2+c3;
#endif
#if MODE == 1 || MODE == 2
    int isum = i0+i1+i2+i3+i4+i5+i6+i7;
    if (isum == 0xCAFEBABE) ((int*)C)[blockIdx.x+1] = isum;
#endif

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d ITERS=%d total_cy=%llu cy/iter=%.3f\n",
               MODE, ITERS, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
