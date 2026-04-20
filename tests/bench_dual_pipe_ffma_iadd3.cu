// FFMA + IADD3 dual-pipe issue test.
// Mode 0: pure FFMA chain. Expect ~14 TIOPS at 1500 MHz (76 TFLOPS/2032×1500/2 = 28.4? actually FP32 FFMA is FMA pipe, 0.5/SMSP/cy, but 64 cores).
// Mode 1: pure IADD3 chain. Expect ~14 TIOPS.
// Mode 2: mixed FFMA + IADD3 in same loop. If different pipes, total ~28 (sum).
//          If same pipe, total ~14 (split).

#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 4
#endif
#ifndef MIX_MODE
#define MIX_MODE 2
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv[N_CHAINS];
    float fb[N_CHAINS];
    float fc[N_CHAINS];
    unsigned int iv[N_CHAINS];
    unsigned int ib[N_CHAINS];
    unsigned int ic[N_CHAINS];

    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        fv[k] = (float)(threadIdx.x + k);
        fb[k] = (float)(threadIdx.x * 2 + k);
        fc[k] = (float)(threadIdx.x * 3 + k);
        iv[k] = 0xDEAD0000u + (threadIdx.x * 131 + k * 17);
        ib[k] = 0xBEEF0000u + (threadIdx.x * 271 + k * 23);
        ic[k] = 0xCAFE0000u + (threadIdx.x * 419 + k * 41);
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
#if MIX_MODE == 0
                // pure FFMA
                asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                             : "+f"(fv[k]) : "f"(fb[k]), "f"(fc[k]));
#elif MIX_MODE == 1
                // pure IADD3
                asm volatile("add.s32 %0, %0, %1;\n\tadd.s32 %0, %0, %2;"
                             : "+r"(iv[k]) : "r"(ib[k]), "r"(ic[k]));
#else
                // mixed: 1 FFMA + 1 IADD3 per chain step
                asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                             : "+f"(fv[k]) : "f"(fb[k]), "f"(fc[k]));
                asm volatile("add.s32 %0, %0, %1;\n\tadd.s32 %0, %0, %2;"
                             : "+r"(iv[k]) : "r"(ib[k]), "r"(ic[k]));
#endif
            }
        }
    }

    float facc = 0.0f;
    unsigned int iacc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) { facc += fv[k]; iacc ^= iv[k]; }
    if (((int)iacc == seed) && ((int)facc == seed))
        ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = iacc;
}
