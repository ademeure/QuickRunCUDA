// LOP3 register port pressure test.
// Mode 0: 1 unique read source per LOP3 (a,a,a)
// Mode 1: 2 unique reads per LOP3 (a,b,a) — chain in `a`
// Mode 2: 3 unique reads per LOP3 (a,b,c) — chain in `a`
// All same chain length, same imm. If RF has port limit, mode 2 should throttle.

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
#define LOP3_IMM 0x96
#endif
#ifndef PORT_MODE
#define PORT_MODE 2
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
#if PORT_MODE == 0
                // 1 unique read: lop3(v, v, v, imm)
                asm volatile("lop3.b32 %0, %0, %0, %0, " STR(LOP3_IMM) ";"
                             : "+r"(v[k]));
#elif PORT_MODE == 1
                // 2 unique reads: lop3(v, b, v, imm)
                asm volatile("lop3.b32 %0, %0, %1, %0, " STR(LOP3_IMM) ";"
                             : "+r"(v[k]) : "r"(b_src[k]));
#else
                // 3 unique reads: lop3(v, b, c, imm)
                asm volatile("lop3.b32 %0, %0, %1, %2, " STR(LOP3_IMM) ";"
                             : "+r"(v[k]) : "r"(b_src[k]), "r"(c_src[k]));
#endif
            }
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= v[k];
    if ((int)acc == seed)
        ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
