// mma.sync per-K throughput at different K sizes
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int a0 = 0x3F803F80u, a1 = 0x3F803F80u, a2 = 0x3F803F80u, a3 = 0x3F803F80u;
    unsigned int b0 = 0x3F803F80u, b1 = 0x3F803F80u;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // m16n8k16 BF16 (large K)
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif MODE == 1
        // m16n8k8 BF16 (smaller K — needs 2 a regs, 1 b reg)
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(b0));
#elif MODE == 2
        // m16n8k16 TF32
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if ((int)c0 == seed) C[blockIdx.x] = c0 + c1 + c2 + c3;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
#if MODE == 0
        int k=16;
#elif MODE == 1
        int k=8;
#else
        int k=8;
#endif
        printf("MODE=%d k=%d clk=%llu cy/iter=%.3f cy_per_K=%.3f\n",
               MODE, k, t1 - t0, (double)(t1-t0)/(double)ITERS,
               (double)(t1-t0)/(double)ITERS/(double)k);
    }
}
