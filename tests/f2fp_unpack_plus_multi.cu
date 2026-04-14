// Unpack + MULTIPLE companions on different pipes
// To probe: can SMSP dispatch SFU (unpack) + FMA (FFMA) + INT (LOP3/IMAD) + HALF (HFMA2) all in 1 cycle?

#ifndef N_CVT
#define N_CVT 8
#endif
#ifndef N_FFMA
#define N_FFMA 0
#endif
#ifndef N_IADD
#define N_IADD 0
#endif
#ifndef N_HFMA
#define N_HFMA 0
#endif
#ifndef N_LOP3
#define N_LOP3 0
#endif
#ifndef UNROLL
#define UNROLL 32
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int unused_2) {
    unsigned short h[N_CVT];
    unsigned int p[N_CVT];
    #pragma unroll
    for (int k = 0; k < N_CVT; k++) h[k] = 0x3C01 ^ (threadIdx.x + k);

    float cf[(N_FFMA>0?N_FFMA:1)];
    unsigned int ci[(N_IADD>0?N_IADD:1)];
    unsigned int cx[(N_LOP3>0?N_LOP3:1)];
    unsigned int ch[(N_HFMA>0?N_HFMA:1)];
    #pragma unroll
    for (int m = 0; m < N_FFMA; m++) cf[m] = (float)(threadIdx.x+m+1)*1.0001f;
    #pragma unroll
    for (int m = 0; m < N_IADD; m++) ci[m] = threadIdx.x+m+0xBEEF;
    #pragma unroll
    for (int m = 0; m < N_LOP3; m++) cx[m] = threadIdx.x+m+0xDEAD;
    #pragma unroll
    for (int m = 0; m < N_HFMA; m++) ch[m] = 0x3C003C01u ^ (threadIdx.x+m+11);
    const float ca = 1.0000001f;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CVT; k++) {
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(p[k]) : "h"(h[k]));
                h[k] = (unsigned short)p[k];
            }
            #pragma unroll
            for (int m = 0; m < N_FFMA; m++)
                asm volatile("fma.rn.f32 %0, %0, %1, %1;" : "+f"(cf[m]) : "f"(ca));
            #pragma unroll
            for (int m = 0; m < N_IADD; m++)
                asm volatile("add.u32 %0, %0, 1;" : "+r"(ci[m]));
            #pragma unroll
            for (int m = 0; m < N_LOP3; m++)
                asm volatile("xor.b32 %0, %0, 0x1;" : "+r"(cx[m]));
            #pragma unroll
            for (int m = 0; m < N_HFMA; m++)
                asm volatile("fma.rn.f16x2 %0, %0, %0, %0;" : "+r"(ch[m]));
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CVT; k++) acc ^= (unsigned int)h[k];
    #pragma unroll
    for (int m = 0; m < N_FFMA; m++) acc ^= __float_as_int(cf[m]);
    #pragma unroll
    for (int m = 0; m < N_IADD; m++) acc ^= ci[m];
    #pragma unroll
    for (int m = 0; m < N_LOP3; m++) acc ^= cx[m];
    #pragma unroll
    for (int m = 0; m < N_HFMA; m++) acc ^= ch[m];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
