// Forward-only or reverse-only or interleaved, with optional STG per iter
#ifndef DIRECTION
#define DIRECTION 0
#endif
#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef N_STG
#define N_STG 0
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
void kernel(const float* __restrict__ A, float* B,
            float* __restrict__ C, int ITERS, int seed, int unused_2) {
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

#if N_STG > 0
    float st_src[N_STG];
    #pragma unroll
    for (int m = 0; m < N_STG; m++) st_src[m] = (float)(threadIdx.x + m) * 0.001f;
    float* my_C_base = C + (threadIdx.x & 0xFF) + 1024;
#endif

    unsigned int acc = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if DIRECTION == 0
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                unsigned short _h;
                asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_h) : "r"(p[k]));
                p[k] = (unsigned int)_h;
            }
#elif DIRECTION == 1
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(p[k]) : "h"(h[k]));
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
#if N_STG > 0
            #pragma unroll
            for (int m = 0; m < N_STG; m++) {
                asm volatile("st.global.f32 [%0], %1;"
                             :: "l"(my_C_base + m * 32), "f"(st_src[m]));
            }
#endif
        }
    }
#if DIRECTION == 0 || DIRECTION == 2
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= p[k];
#elif DIRECTION == 1
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= (unsigned int)h[k];
#endif
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
