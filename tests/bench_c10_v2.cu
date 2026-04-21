// C10 v2: derive R2P (setp) and P2R (selp) costs separately
// Each test chained through v so no DCE
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int v = (unsigned)(threadIdx.x ^ u2);
    unsigned int x = 0xDEADBEEFu ^ (unsigned)u2;
    unsigned int y = 0xCAFEBABEu;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll 16
        for (int u = 0; u < 16; u++) {
#if MODE == 0
            // 1 setp + 1 selp (round-trip R2P+P2R chained)
            asm volatile("{ .reg .pred p;\n"
                         "  setp.ne.b32 p, %0, %1;\n"
                         "  selp.b32 %0, %1, %2, p; }"
                         : "+r"(v) : "r"(x), "r"(y));
#elif MODE == 1
            // 1 setp + 4 selps (amortize R2P; expose P2R cost)
            asm volatile("{ .reg .pred p;\n"
                         "  setp.ne.b32 p, %0, %1;\n"
                         "  selp.b32 %0, %1, %2, p;\n"
                         "  selp.b32 %0, %0, %1, p;\n"
                         "  selp.b32 %0, %2, %0, p;\n"
                         "  selp.b32 %0, %0, %2, p; }"
                         : "+r"(v) : "r"(x), "r"(y));
#elif MODE == 2
            // 4 setps + 1 selp (amortize P2R; expose R2P cost)
            asm volatile("{ .reg .pred p;\n"
                         "  setp.ne.b32 p, %0, %1;\n"
                         "  setp.eq.b32 p, %0, %2;\n"
                         "  setp.lt.u32 p, %0, %1;\n"
                         "  setp.gt.u32 p, %0, %2;\n"
                         "  selp.b32 %0, %1, %2, p; }"
                         : "+r"(v) : "r"(x), "r"(y));
#elif MODE == 3
            // 1 IADD3 baseline (Cluster B simple, similar register flow)
            asm volatile("add.u32 %0, %0, %1;" : "+r"(v) : "r"(x));
#elif MODE == 4
            // 1 LOP3 baseline
            asm volatile("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(v) : "r"(x), "r"(y));
#elif MODE == 5
            // 5 IADD3 (5-op baseline for MODE 1/2 normalization)
            asm volatile("add.u32 %0, %0, %1;" : "+r"(v) : "r"(x));
            asm volatile("add.u32 %0, %0, %1;" : "+r"(v) : "r"(x));
            asm volatile("add.u32 %0, %0, %1;" : "+r"(v) : "r"(x));
            asm volatile("add.u32 %0, %0, %1;" : "+r"(v) : "r"(x));
            asm volatile("add.u32 %0, %0, %1;" : "+r"(v) : "r"(x));
#endif
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.3f cy_per_inner=%.3f\n",
               MODE, t1-t0, (double)(t1-t0)/(double)ITERS,
               (double)(t1-t0)/(double)ITERS/16.0);
    }
}
