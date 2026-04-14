// Pure F2FP throughput test: nothing but F2FP in the hot loop.
// Each inner iteration is N_CHAINS independent asm blocks; each block contains
// CHAIN_PAIRS back-to-back e4m3x2↔f16x2 round-trips (= 2×CHAIN_PAIRS F2FPs).
// Between asm blocks there is NO LOP3/XOR/shuffle — just the next block's
// asm volatile on a different register.
//
// At compile time this gives the scheduler full ILP across N_CHAINS registers,
// while each block is a CHAIN_PAIRS-deep fully-dependent F2FP sequence.

#ifndef N_CHAINS
#define N_CHAINS 4
#endif
#ifndef CHAIN_PAIRS
#define CHAIN_PAIRS 4
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

#define ASM_PAIR "cvt.rn.satfinite.e4m3x2.f16x2 _h, %0;\ncvt.rn.f16x2.e4m3x2 %0, _h;\n"

#if   CHAIN_PAIRS == 1
  #define CHAIN_BODY ASM_PAIR
#elif CHAIN_PAIRS == 2
  #define CHAIN_BODY ASM_PAIR ASM_PAIR
#elif CHAIN_PAIRS == 4
  #define CHAIN_BODY ASM_PAIR ASM_PAIR ASM_PAIR ASM_PAIR
#elif CHAIN_PAIRS == 8
  #define CHAIN_BODY ASM_PAIR ASM_PAIR ASM_PAIR ASM_PAIR ASM_PAIR ASM_PAIR ASM_PAIR ASM_PAIR
#elif CHAIN_PAIRS == 16
  #define CHAIN_BODY ASM_PAIR ASM_PAIR ASM_PAIR ASM_PAIR ASM_PAIR ASM_PAIR ASM_PAIR ASM_PAIR \
                     ASM_PAIR ASM_PAIR ASM_PAIR ASM_PAIR ASM_PAIR ASM_PAIR ASM_PAIR ASM_PAIR
#endif

#define DO_CHAIN(reg) asm volatile("{ .reg .b16 _h;\n" CHAIN_BODY "}" : "+r"(reg))

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int unused_2) {
    unsigned int p[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) p[k] = 0x3C003C01u ^ (threadIdx.x + k);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            // N_CHAINS independent chains, each CHAIN_PAIRS pairs deep = 2*CHAIN_PAIRS F2FPs.
            // NO LOP3 / XOR / shuffle between them.
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) DO_CHAIN(p[k]);
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= p[k];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
