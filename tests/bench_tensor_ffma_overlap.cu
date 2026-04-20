// Tensor + FFMA + IMAD simultaneous test.
// Mode 0: pure mma.sync chain
// Mode 1: pure FFMA chain (2-source to avoid port limit)
// Mode 2: pure IMAD chain
// Mode 3: mma.sync + FFMA
// Mode 4: mma.sync + IMAD
// Mode 5: mma.sync + FFMA + IMAD all three

#include <cuda_bf16.h>

#ifndef N_OPS
#define N_OPS 8
#endif
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    // mma fragments
    unsigned int a0 = 0x3F803F80u, a1 = 0x3F803F80u, a2 = 0x3F803F80u, a3 = 0x3F803F80u;
    unsigned int b0 = 0x3F803F80u, b1 = 0x3F803F80u;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    // FFMA chains (2 source per inst)
    float fv[N_OPS], fb[N_OPS];
    // IMAD chains
    int   iv[N_OPS], ib[N_OPS];

    #pragma unroll
    for (int k = 0; k < N_OPS; k++) {
        fv[k] = (float)(threadIdx.x + k);
        fb[k] = (float)(threadIdx.x * 2 + k);
        iv[k] = (int)(threadIdx.x + k);
        ib[k] = (int)(threadIdx.x * 2 + k);
    }

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0 || MODE == 3 || MODE == 4 || MODE == 5
        // mma.sync chain
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
#if MODE == 1 || MODE == 3 || MODE == 5
        // FFMA chain (2-source, peak)
        #pragma unroll
        for (int k = 0; k < N_OPS; k++) {
            asm volatile("fma.rn.f32 %0, %0, %1, %0;" : "+f"(fv[k]) : "f"(fb[k]));
        }
#endif
#if MODE == 2 || MODE == 4 || MODE == 5
        // IMAD chain (2-source, fits in 1 IMAD)
        #pragma unroll
        for (int k = 0; k < N_OPS; k++) {
            asm volatile("mad.lo.s32 %0, %0, %1, %0;" : "+r"(iv[k]) : "r"(ib[k]));
        }
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    float facc = 0.0f; int iacc = 0;
    #pragma unroll
    for (int k = 0; k < N_OPS; k++) { facc += fv[k]; iacc ^= iv[k]; }
    if (((int)c0 == seed) && ((int)c1 == seed) && ((int)facc == seed) && (iacc == seed))
        C[blockIdx.x] = c0 + c1 + c2 + c3 + facc + (float)iacc;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d iters=%d clk=%llu cy/iter=%.3f\n",
               MODE, ITERS, t1 - t0, (double)(t1-t0)/(double)ITERS);
    }
}
