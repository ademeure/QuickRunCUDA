// F2F pure throughput: is it 32 or 64 per SM/clk?
// Test with independent parallel chains of single F2F (one-way f32->f16 only).
#ifndef UNROLL
#define UNROLL 32
#endif
#ifndef N_CHAINS
#define N_CHAINS 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int unused_2) {
    float inp[N_CHAINS];
    unsigned int acc[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        inp[k] = (float)(threadIdx.x + k + 1) * 1.0001f;
        acc[k] = 0;
    }
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                unsigned short h;
                // F2F: cvt.rn.f16.f32 emits F2F.F16.F32 (distinct opcode from F2FP).
                // Use volatile + separate write to avoid coalescing.
                asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(inp[k]));
                acc[k] ^= (unsigned int)h;
                // Mutate inp[k] via aliasing so it doesn't become loop-invariant.
                // Use IMAD-style op hidden in asm to defeat hoisting.
                unsigned int bits = __float_as_int(inp[k]);
                asm volatile("xor.b32 %0, %0, %1;" : "+r"(bits) : "r"(j+k));
                inp[k] = __int_as_float(bits);
            }
        }
    }
    unsigned int final = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) final ^= acc[k];
    unsigned int acc_out = final;
    if ((int)acc_out == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc_out;
}
