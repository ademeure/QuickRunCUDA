// Tensor core warmup cost: is first MMA after kernel launch slower?
// Single warp, instrumented per-MMA with clock64. Logs first 8 MMAs.
// Mode 0: mma.sync m16n8k16 BF16 chain
// Mode 1: with explicit warmup loop before timed MMAs (control)

#include <cuda_bf16.h>

#ifndef N_TIMED
#define N_TIMED 8
#endif
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    // Setup matrix fragments — load some BF16 input data
    unsigned int a0 = 0x3F803F80u;  // bf16 1.0 packed
    unsigned int a1 = 0x3F803F80u;
    unsigned int a2 = 0x3F803F80u;
    unsigned int a3 = 0x3F803F80u;
    unsigned int b0 = 0x3F803F80u;
    unsigned int b1 = 0x3F803F80u;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    unsigned long long times[N_TIMED + 1];

#if MODE == 1
    // Warmup loop: 16 untimed MMAs first
    #pragma unroll 1
    for (int i = 0; i < 16; i++) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
#endif

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(times[0]));

    #pragma unroll
    for (int t = 0; t < N_TIMED; t++) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(times[t+1]));
    }

    // Use accumulator to defeat DCE
    if (((int)c0 == seed) && ((int)c1 == seed) && ((int)c2 == seed) && ((int)c3 == seed))
        C[blockIdx.x] = c0 + c1 + c2 + c3;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d ", MODE);
        for (int t = 0; t < N_TIMED; t++) {
            printf("mma%d=%llucy ", t, times[t+1] - times[t]);
        }
        printf("\n");
    }
}
