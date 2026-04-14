// F2FP one-way with INDEPENDENT LOP3s added. Test if LOP3 on a separate
// register chain affects F2FP throughput. Should not, if truly separate pipes.

#ifndef DIRECTION
#define DIRECTION 0
#endif
#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef N_LOP3
#define N_LOP3 0
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
    // F2FP state (one of 3 modes)
#if DIRECTION == 0
    unsigned int p[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) p[k] = 0x3C003C01u ^ (threadIdx.x + k);
#elif DIRECTION == 1
    unsigned short h[N_CHAINS];
    unsigned int p[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) h[k] = 0x3C01 ^ (threadIdx.x + k);
#elif DIRECTION == 2
    unsigned int p[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) p[k] = 0x3C003C01u ^ (threadIdx.x + k);
#endif

    // Independent LOP3 state — separate registers, not touched by F2FP chain
#if N_LOP3 > 0
    unsigned int q[N_LOP3];
    #pragma unroll
    for (int k = 0; k < N_LOP3; k++) q[k] = 0xBEEFCAFEu ^ (threadIdx.x * 31 + k);
#endif

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            // F2FP ops
#if DIRECTION == 0
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                unsigned short _h;
                asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
                             : "=h"(_h) : "r"(p[k]));
                p[k] = (unsigned int)_h;
            }
#elif DIRECTION == 1
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;"
                             : "=r"(p[k]) : "h"(h[k]));
                h[k] = (unsigned short)p[k];
            }
#elif DIRECTION == 2
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                asm volatile("{ .reg .b16 _h;\n"
                             "  cvt.rn.satfinite.e4m3x2.f16x2 _h, %0;\n"
                             "  cvt.rn.f16x2.e4m3x2 %0, _h; }"
                             : "+r"(p[k]));
            }
#endif
            // Independent LOP3s — pure int pipe, no shared registers with F2FP
#if N_LOP3 > 0
            #pragma unroll
            for (int k = 0; k < N_LOP3; k++) {
                asm volatile("xor.b32 %0, %0, 0xAAAAAAAA;" : "+r"(q[k]));
            }
#endif
        }
    }

    unsigned int acc = 0;
#if DIRECTION == 0 || DIRECTION == 2
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= p[k];
#elif DIRECTION == 1
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= (unsigned int)h[k];
#endif
#if N_LOP3 > 0
    #pragma unroll
    for (int k = 0; k < N_LOP3; k++) acc ^= q[k];
#endif
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
