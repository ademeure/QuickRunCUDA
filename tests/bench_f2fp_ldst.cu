// F2FP + LDG/STG contention. Each thread does F2FP round-trips and
// optionally issues N_LDG loads and/or N_STG stores from/to independent memory.
//
// Addresses: each thread writes to unique C[tid + lane] positions to avoid
// conflicts; reads from unique A[tid + lane] positions.

#ifndef N_CHAINS
#define N_CHAINS 4
#endif
#ifndef CHAIN_PAIRS
#define CHAIN_PAIRS 4
#endif
#ifndef N_LDG
#define N_LDG 0
#endif
#ifndef N_STG
#define N_STG 0
#endif
#ifndef LDG_HIT_L1
#define LDG_HIT_L1 1   // 1 = keep addresses in a tight range (L1/L2 hit); 0 = span a large region (DRAM)
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
#endif

#define DO_F2FP(reg) asm volatile("{ .reg .b16 _h;\n" CHAIN_BODY "}" : "+r"(reg))

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(const float* __restrict__ A, float* B,
            float* __restrict__ C, int ITERS, int seed, int unused_2) {
    unsigned int p[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) p[k] = 0x3C003C01u ^ (threadIdx.x + k);

#if N_LDG > 0
    // Each thread reads N_LDG float values from stable offsets
    float ld_val[N_LDG];
    const float* my_A_base = A +
        (LDG_HIT_L1 ? (threadIdx.x & 0xFF) : (blockIdx.x * blockDim.x + threadIdx.x) * 32);
#endif
#if N_STG > 0
    float st_src[N_STG];
    #pragma unroll
    for (int m = 0; m < N_STG; m++) st_src[m] = (float)(threadIdx.x + m) * 0.001f;
    // Writes use per-block bump region so different blocks don't collide on same cache line
    float* my_C_base = C +
        (LDG_HIT_L1 ? (threadIdx.x & 0xFF) : (blockIdx.x * blockDim.x + threadIdx.x) * 32) + 1024;
#endif

    unsigned int acc = 0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            // F2FP work
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) DO_F2FP(p[k]);

#if N_LDG > 0
            #pragma unroll
            for (int m = 0; m < N_LDG; m++) {
                asm volatile("ld.global.nc.f32 %0, [%1];"
                             : "=f"(ld_val[m]) : "l"(my_A_base + m * 32));
            }
            #pragma unroll
            for (int m = 0; m < N_LDG; m++) acc ^= __float_as_int(ld_val[m]);
#endif
#if N_STG > 0
            #pragma unroll
            for (int m = 0; m < N_STG; m++) {
                asm volatile("st.global.f32 [%0], %1;"
                             :: "l"(my_C_base + m * 32), "f"(st_src[m]));
            }
#endif
        }
    }

    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= p[k];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
