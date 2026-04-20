// IADD3 (3-input integer add) throughput.
// Catalog claim: 2.46/SM/cy (above LOP3's 2.0/SM/cy) — implies dual-issue.
// Question: is IADD3 truly faster than LOP3, or does the claim reflect
// a different test condition?

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

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v[N_CHAINS];
    unsigned int b_src[N_CHAINS];
    unsigned int c_src[N_CHAINS];

    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        v[k]     = 0xDEAD0000u + (threadIdx.x * 131 + k * 17);
        b_src[k] = 0xBEEF0000u + (threadIdx.x * 271 + k * 23);
        c_src[k] = 0xCAFE0000u + (threadIdx.x * 419 + k * 41);
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                // PTX add3 → SASS IADD3
                asm volatile("add.s32 %0, %0, %1;\n\tadd.s32 %0, %0, %2;"
                             : "+r"(v[k]) : "r"(b_src[k]), "r"(c_src[k]));
            }
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= v[k];
    if ((int)acc == seed)
        ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
