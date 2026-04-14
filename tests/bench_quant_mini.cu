// Realistic quant-like kernel: F2FP PACK + FFMA + STG ratio (like quant+scale+write)
#ifndef N_PACK
#define N_PACK 4
#endif
#ifndef N_FFMA
#define N_FFMA 0
#endif
#ifndef N_STG
#define N_STG 0
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(const float* A, float* B, float* C, int ITERS, int seed, int unused_2) {
    unsigned int p[N_PACK];
    #pragma unroll
    for (int k = 0; k < N_PACK; k++) p[k] = 0x3C003C01u ^ (threadIdx.x + k);
#if N_FFMA > 0
    float f[N_FFMA];
    #pragma unroll
    for (int m = 0; m < N_FFMA; m++) f[m] = 1.0f + (float)m * 0.001f;
    float ca = 1.00001f, cb = 0.99999f;
#endif
#if N_STG > 0
    float* my_C_base = C + blockIdx.x * BLOCK_SIZE * 64 + threadIdx.x;
    float val = (float)threadIdx.x;
#endif

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            // PACK only (forward f16x2 → e4m3x2)
            #pragma unroll
            for (int k = 0; k < N_PACK; k++) {
                unsigned short h;
                asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(h) : "r"(p[k]));
                p[k] = (unsigned int)h;
            }
#if N_FFMA > 0
            #pragma unroll
            for (int m = 0; m < N_FFMA; m++) {
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f[m]) : "f"(ca), "f"(cb));
            }
#endif
#if N_STG > 0
            #pragma unroll
            for (int m = 0; m < N_STG; m++) {
                asm volatile("st.global.f32 [%0], %1;"
                             :: "l"(my_C_base + m * BLOCK_SIZE), "f"(val));
            }
#endif
        }
    }
    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_PACK; k++) acc ^= p[k];
#if N_FFMA > 0
    #pragma unroll
    for (int m = 0; m < N_FFMA; m++) acc ^= __float_as_int(f[m]);
#endif
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
