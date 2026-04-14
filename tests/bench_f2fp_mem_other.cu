// F2FP + various "other" ops to find contention.
// OP_TYPE:
//   0 = ld.shared.f32       (shared load)
//   1 = ld.global.ca.v4.f32 (wide LDG)
//   2 = st.global.v4.f32    (wide STG, dup)
//   3 = atom.global.add.f32 (global atomic)
//   4 = ldmatrix.sync       (tensor-core setup LDS matrix load)
//   5 = bar.sync.aligned 0  (barrier every N ops)
//   6 = membar.gl           (global memory barrier)

#ifndef OP_TYPE
#define OP_TYPE 0
#endif
#ifndef N_CHAINS
#define N_CHAINS 4
#endif
#ifndef CHAIN_PAIRS
#define CHAIN_PAIRS 4
#endif
#ifndef N_OP
#define N_OP 0
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
#endif
#define DO_F2FP(reg) asm volatile("{ .reg .b16 _h;\n" CHAIN_BODY "}" : "+r"(reg))

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(const float* __restrict__ A, float* B,
            float* __restrict__ C, int ITERS, int seed, int unused_2) {
    unsigned int p[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) p[k] = 0x3C003C01u ^ (threadIdx.x + k);

#if OP_TYPE == 0 || OP_TYPE == 4
    __shared__ __align__(16) float smem[BLOCK_SIZE * 32 + 32];
    // Init some shared data
    if (threadIdx.x < 32) {
        #pragma unroll
        for (int m = 0; m < 32; m++) smem[threadIdx.x + m * BLOCK_SIZE] = (float)(m + 1);
    }
    __syncthreads();
#endif

#if OP_TYPE == 1 || OP_TYPE == 2
    float4* my_base = (float4*)(C + blockIdx.x * BLOCK_SIZE * 32) + threadIdx.x;
#endif
#if OP_TYPE == 3
    float* atom_loc = C + blockIdx.x;
#endif

    unsigned int acc = 0;
    float ldval = 0.0f;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) DO_F2FP(p[k]);

#if N_OP > 0
            #pragma unroll
            for (int m = 0; m < N_OP; m++) {
  #if OP_TYPE == 0
                asm volatile("ld.shared.f32 %0, [%1];" : "=f"(ldval)
                             : "r"((unsigned)__cvta_generic_to_shared(&smem[threadIdx.x + m * 32])));
                acc ^= __float_as_int(ldval);
  #elif OP_TYPE == 1
                float4 v;
                asm volatile("ld.global.ca.v4.f32 {%0,%1,%2,%3}, [%4];"
                             : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                             : "l"(my_base + m));
                acc ^= __float_as_int(v.x);
  #elif OP_TYPE == 2
                asm volatile("st.global.v4.f32 [%0], {1.0f, 2.0f, 3.0f, 4.0f};" :: "l"(my_base + m));
  #elif OP_TYPE == 3
                asm volatile("atom.global.add.f32 %0, [%1], %2;" : "=f"(ldval)
                             : "l"(atom_loc), "f"(1.0f));
                acc ^= __float_as_int(ldval);
  #elif OP_TYPE == 5
                asm volatile("bar.sync.aligned 0;" ::);
  #elif OP_TYPE == 6
                asm volatile("membar.gl;" ::);
  #endif
            }
#endif
        }
    }

    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= p[k];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
