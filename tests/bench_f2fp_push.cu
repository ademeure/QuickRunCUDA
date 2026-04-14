// F2FP throughput microbench — push toward 32 ops/SM/clk.
//
// Parameters via -H:
//   CVT_ASM       PTX opcode (e.g. cvt.rn.satfinite.e4m3x2.f16x2)
//   CVT_B8        defined if the output is .b8 (FP4) and needs a b16 pack
//   N_CHAINS      independent CVTs per iteration per thread (default 16)
//   UNROLL        outer unroll factor (default 16)
//   BLOCK_SIZE    threads per block (default 256)
//   MIN_BLOCKS    min blocks per SM for __launch_bounds__ (default 1)
//   SINK_MODE     0 = no sink (let compiler DCE if it can — use at own risk)
//                 1 = impossible-branch store of XOR
//                 2 = volatile asm consumes (no store)
//   INPUT_TYPE    0 = u32 via "r" (for f16x2/bf16x2 sources)
//                 1 = f32 pair via "f" (for f32 sources)

#ifndef CVT_ASM
#define CVT_ASM cvt.rn.satfinite.e4m3x2.f16x2
#endif
#ifndef N_CHAINS
#define N_CHAINS 16
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1
#endif
#ifndef SINK_MODE
#define SINK_MODE 1
#endif
#ifndef INPUT_TYPE
#define INPUT_TYPE 0   // u32 source for f16x2/bf16x2
#endif

#define _S(x) #x
#define S(x) _S(x)

#if INPUT_TYPE == 0
  #define SRC_REG "r"
  #define SRC_T unsigned int
  #define SRC_INIT(i) (0x3C003C00u + (i) + tid)
  #define CVT_ONE(dst, src) S(CVT_ASM) " " dst ", " src ";\n\t"
#elif INPUT_TYPE == 1
  // f32 pair -> packed narrow
  #define SRC_REG "f"
  #define SRC_T float
  #define SRC_INIT(i) ((float)((i) + threadIdx.x) * 0.01f + 1.0f)
  #define CVT_ONE(dst, src_lo, src_hi) S(CVT_ASM) " " dst ", " src_hi ", " src_lo ";\n\t"
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int unused_2) {
    unsigned int tid = threadIdx.x;
    unsigned int acc = 0;

    // Initialise N_CHAINS independent "chains" (each just one value — we re-feed
    // a tid-dependent input every iter so these are N_CHAINS independent CVTs)
    SRC_T in[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) in[k] = (SRC_T)SRC_INIT(k);

    unsigned short out[N_CHAINS];

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            // Thread-dependent modulation so compiler can't CSE across iters
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
#if INPUT_TYPE == 0
                in[k] = (SRC_T)(in[k] + i + j + k + tid);
#else
                in[k] = in[k] + (float)(i + j + k) * 0.00001f;
#endif
            }

#if INPUT_TYPE == 0
  #ifdef CVT_B8
            // b8 output — use a block asm with internal b8 regs and final b16 pack
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                unsigned short t;
                asm volatile(
                    "{ .reg .b8 _b;\n\t"
                    S(CVT_ASM) " _b, %1;\n\t"
                    "mov.b16 %0,{_b,0}; }"
                    : "=h"(t) : "r"(in[k]));
                out[k] = t;
            }
  #else
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                unsigned short t;
                asm volatile(S(CVT_ASM) " %0, %1;" : "=h"(t) : "r"(in[k]));
                out[k] = t;
            }
  #endif
#else   /* INPUT_TYPE == 1, f32 pair source */
  #ifdef CVT_B8
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                unsigned short t;
                float lo = in[k];
                float hi = in[k] + 0.5f;
                asm volatile(
                    "{ .reg .b8 _b;\n\t"
                    S(CVT_ASM) " _b, %2, %1;\n\t"
                    "mov.b16 %0,{_b,0}; }"
                    : "=h"(t) : "f"(lo), "f"(hi));
                out[k] = t;
            }
  #else
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                unsigned short t;
                float lo = in[k];
                float hi = in[k] + 0.5f;
                asm volatile(S(CVT_ASM) " %0, %2, %1;" : "=h"(t) : "f"(lo), "f"(hi));
                out[k] = t;
            }
  #endif
#endif

            // Accumulate via xor so results can't be DCE'd
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) acc ^= out[k];
        }
    }

#if SINK_MODE == 1
    // Impossible branch store — seed unlikely to match
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + tid] = acc;
#elif SINK_MODE == 2
    asm volatile("" : : "r"(acc));
#endif
}
