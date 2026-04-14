// F2FP one-way throughput — SYMMETRIC feedback to fairly compare directions.
// Both forward and reverse use a single-register feedback loop through the
// output, with minimal-cost type-coercion: forward uses zero-extend (u16→u32),
// reverse uses truncate (u32→u16). Both are "free" register-width coercions
// at SIMT level, so the F2FP pipe is the only real work on the critical path.

#ifndef DIRECTION
#define DIRECTION 0
#endif
#ifndef N_CHAINS
#define N_CHAINS 8
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

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int unused_2) {
#if DIRECTION == 0
    // Forward: u32 in → u16 out → u32 (zero-extend) → next forward. Loop-carried
    // dep forces compiler to emit every F2FP. Zero-extend is a free MOV in SIMT.
    unsigned int p[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) p[k] = 0x3C003C01u ^ (threadIdx.x + k);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                unsigned short h;
                asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
                             : "=h"(h) : "r"(p[k]));
                p[k] = (unsigned int)h;   // zero-extend (free)
            }
        }
    }

#elif DIRECTION == 1
    // Reverse: u16 in → u32 out → u16 (truncate) → next reverse.
    unsigned short h[N_CHAINS];
    unsigned int p[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) h[k] = 0x3C01 ^ (threadIdx.x + k);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;"
                             : "=r"(p[k]) : "h"(h[k]));
                h[k] = (unsigned short)p[k];  // truncate (free)
            }
        }
    }

#elif DIRECTION == 2
    // Interleaved round-trip
    unsigned int p[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) p[k] = 0x3C003C01u ^ (threadIdx.x + k);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                asm volatile("{ .reg .b16 _h;\n"
                             "  cvt.rn.satfinite.e4m3x2.f16x2 _h, %0;\n"
                             "  cvt.rn.f16x2.e4m3x2 %0, _h; }"
                             : "+r"(p[k]));
            }
        }
    }
#endif

    unsigned int acc = 0;
#if DIRECTION == 0 || DIRECTION == 2
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= p[k];
#elif DIRECTION == 1
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= (unsigned int)h[k];
#endif
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
