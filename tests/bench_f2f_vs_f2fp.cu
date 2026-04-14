// Mix F2F (cvt.rn.f16.f32) with F2FP pack (cvt.rn.satfinite.e4m3x2.f16x2).
// Determine if they share SFU slots.
#ifndef UNROLL
#define UNROLL 32
#endif
#ifndef N_F2FP
#define N_F2FP 16
#endif
#ifndef N_F2F
#define N_F2F 0
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int unused_2) {
    unsigned int p[N_F2FP];
    #pragma unroll
    for (int k = 0; k < N_F2FP; k++) p[k] = 0x3C003C01u ^ (threadIdx.x + k);

    float finp[N_F2F > 0 ? N_F2F : 1];
    unsigned int facc[N_F2F > 0 ? N_F2F : 1];
    #pragma unroll
    for (int m = 0; m < N_F2F; m++) { finp[m] = (float)(threadIdx.x + m + 1) * 1.0001f; facc[m] = 0; }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_F2FP; k++) {
                // F2FP round trip (pack + unpack = 2 SFU slots)
                asm volatile(
                    "{ .reg .b16 _h;\n\t"
                    "  cvt.rn.satfinite.e4m3x2.f16x2 _h, %0;\n\t"
                    "  cvt.rn.f16x2.e4m3x2 %0, _h; }"
                    : "+r"(p[k]));
            }
            #pragma unroll
            for (int m = 0; m < N_F2F; m++) {
                unsigned short h;
                asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(finp[m]));
                facc[m] ^= (unsigned int)h;
                unsigned int bits = __float_as_int(finp[m]);
                asm volatile("xor.b32 %0, %0, %1;" : "+r"(bits) : "r"(j+m));
                finp[m] = __int_as_float(bits);
            }
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_F2FP; k++) acc ^= p[k];
    #pragma unroll
    for (int m = 0; m < N_F2F; m++) acc ^= facc[m];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
