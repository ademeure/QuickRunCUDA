// EX2 + other-MUFU co-issue test.
// EX2 runs at 32/SM/clk (2-wide), other MUFU (rsqrt/sin/tanh) at 16/SM/clk (1-wide).
// Hypothesis: combined = 32/SM/clk (each uses 1 slot) — parallel: pack/unpack case.
//
// N_EX2 independent EX2 chains + N_MUFU other-MUFU chains per iter.

#ifndef N_EX2
#define N_EX2 8
#endif
#ifndef N_MUFU
#define N_MUFU 0
#endif
#ifndef MUFU_ASM
#define MUFU_ASM rsqrt.approx.f32
#endif
#ifndef UNROLL
#define UNROLL 32
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1
#endif

#define _S(x) #x
#define S(x) _S(x)

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int unused_2) {
    float e[N_EX2];
    #pragma unroll
    for (int k = 0; k < N_EX2; k++) e[k] = (float)(threadIdx.x + k + 1) * 0.001f;

#if N_MUFU > 0
    float m[N_MUFU];
    #pragma unroll
    for (int k = 0; k < N_MUFU; k++) m[k] = (float)(threadIdx.x + k + 100) * 0.01f + 1.0f;
#endif

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_EX2; k++) {
                asm volatile("ex2.approx.f32 %0, %0;" : "+f"(e[k]));
            }
#if N_MUFU > 0
            #pragma unroll
            for (int k = 0; k < N_MUFU; k++) {
                asm volatile(S(MUFU_ASM) " %0, %0;" : "+f"(m[k]));
            }
#endif
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_EX2; k++) acc ^= __float_as_int(e[k]);
#if N_MUFU > 0
    #pragma unroll
    for (int k = 0; k < N_MUFU; k++) acc ^= __float_as_int(m[k]);
#endif
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
