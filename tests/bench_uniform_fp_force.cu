// Try VERY hard to force UFFMA/UFADD/UFMUL emission.

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 4
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // Warp-uniform FP chain via blockIdx-dependent initialization
    float f0 = (float)blockIdx.x * 0.001f + 1.0f;
    float c = (float)(seed & 0xFF) * 0.0001f + 1.0001f;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0  // Inline UFFMA-style via explicit uniform PTX
            asm volatile("{.reg .f32 t; fma.rn.f32 t, %0, %1, %1; mov.b32 %0, t;}"
                : "+f"(f0) : "f"(c));
#elif OP == 1  // Warp-invariant FMA through explicit uniform reg
            asm volatile("fma.rn.f32 %0, %0, %1, %1;" : "+f"(f0) : "f"(c));
#elif OP == 2  // Independent uniform variables
            float f1 = c + 0.001f;
            f0 = f0 * c + f1;
            c = c * 1.0000001f + 0.0001f;
#elif OP == 3  // Known-uniform min
            f0 = fminf(f0, c);
            c = c + 0.0001f;
#endif
        }
    }
    // Must use f0 in result keyed on threadIdx so full warp still participates
    if (__float_as_int(f0) == seed)
        ((float*)C)[blockIdx.x * blockDim.x + threadIdx.x] = f0;
}
