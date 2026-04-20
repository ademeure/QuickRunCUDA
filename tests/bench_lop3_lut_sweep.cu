// LOP3.LUT throughput sweep across all 256 truth tables.
// Question: is LOP3 throughput data-independent (just a LUT-per-bit), or do
// some imms have fast paths / slow paths?
//
// Compile with -H "#define LOP3_IMM 0x96" etc. Kernel does N_CHAINS
// independent LOP3 chains with 3 distinct register sources, ILP-bound to
// reach pipe peak. ITERS controls runtime.

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
#ifndef LOP3_IMM
#define LOP3_IMM 0x96  // default: 3-input XOR
#endif

#define _STR(x) #x
#define STR(x) _STR(x)

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
                // PTX lop3.b32: d = LUT[a, b, c] where LUT is the 8-bit imm
                asm volatile("lop3.b32 %0, %0, %1, %2, " STR(LOP3_IMM) ";"
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
