// Pure F2FP interleaved + independent companion ops on separate registers.
// N_CHAINS asm blocks each doing CHAIN_PAIRS round-trips. N_COMP companion ops
// on separate register chain q[]. Designed for the cleanest coissue measurement.

#ifndef N_CHAINS
#define N_CHAINS 4
#endif
#ifndef CHAIN_PAIRS
#define CHAIN_PAIRS 4
#endif
#ifndef N_COMP
#define N_COMP 0
#endif
#ifndef COMP_TYPE
#define COMP_TYPE 0   // 0=FFMA 1=FMUL 2=IMAD 3=LOP3 4=MUFU.ex2 5=SHL 6=IADD3
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
void kernel(float* A, float* B, float* C, int ITERS, int seed, int unused_2) {
    unsigned int p[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) p[k] = 0x3C003C01u ^ (threadIdx.x + k);

#if N_COMP > 0
    #if COMP_TYPE == 0 || COMP_TYPE == 1 || COMP_TYPE == 4
      float q[N_COMP];
      #pragma unroll
      for (int m = 0; m < N_COMP; m++) q[m] = 1.0001f + (float)(threadIdx.x+m)*0.001f;
      const float ca = 1.0000001f;
    #else
      unsigned int q[N_COMP];
      #pragma unroll
      for (int m = 0; m < N_COMP; m++) q[m] = 0xBEEFCAFEu ^ (threadIdx.x*31+m);
    #endif
#endif

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) DO_F2FP(p[k]);
#if N_COMP > 0
            #pragma unroll
            for (int m = 0; m < N_COMP; m++) {
  #if COMP_TYPE == 0
                asm volatile("fma.rn.f32 %0, %0, %1, %1;" : "+f"(q[m]) : "f"(ca));
  #elif COMP_TYPE == 1
                asm volatile("mul.rn.f32 %0, %0, %1;" : "+f"(q[m]) : "f"(ca));
  #elif COMP_TYPE == 2
                asm volatile("mad.lo.u32 %0, %0, 3, 1;" : "+r"(q[m]));
  #elif COMP_TYPE == 3
                asm volatile("xor.b32 %0, %0, 0xAAAAAAAA;" : "+r"(q[m]));
  #elif COMP_TYPE == 4
                asm volatile("ex2.approx.f32 %0, %0;" : "+f"(q[m]));
  #elif COMP_TYPE == 5
                asm volatile("shl.b32 %0, %0, 1;" : "+r"(q[m]));
  #elif COMP_TYPE == 6
                asm volatile("add.u32 %0, %0, 1;" : "+r"(q[m]));
  #endif
            }
#endif
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= p[k];
#if N_COMP > 0
    #pragma unroll
    for (int m = 0; m < N_COMP; m++) {
  #if COMP_TYPE == 0 || COMP_TYPE == 1 || COMP_TYPE == 4
        acc ^= __float_as_int(q[m]);
  #else
        acc ^= q[m];
  #endif
    }
#endif
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
