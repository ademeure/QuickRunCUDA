// FP16 packed ops vs FP32 throughput
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv = (float)threadIdx.x + 0.5f;
    float fb = 1.000001f + (float)u2 * 1e-9f;
    unsigned int hv = 0x3C003C00u + (unsigned)u2;
    unsigned int hb = 0x3C013C01u + (unsigned)u2;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // Scalar fp32 FFMA (2-src)
        asm("fma.rn.f32 %0, %0, %1, %0;" : "+f"(fv) : "f"(fb));
#elif MODE == 1
        // FP16x2 packed FMA (HFMA2)
        asm("fma.rn.f16x2 %0, %0, %1, %0;" : "+r"(hv) : "r"(hb));
#elif MODE == 2
        // bf16x2 packed FMA
        asm("fma.rn.bf16x2 %0, %0, %1, %0;" : "+r"(hv) : "r"(hb));
#elif MODE == 3
        // 4 simultaneous fp16x2 (max ILP)
        unsigned int hv2 = hv ^ 0xAAAA;
        unsigned int hv3 = hv ^ 0x5555;
        unsigned int hv4 = hv ^ 0xFFFF;
        asm("fma.rn.f16x2 %0, %0, %4, %0;" : "+r"(hv) : "r"(hb));
        asm("fma.rn.f16x2 %0, %0, %4, %0;" : "+r"(hv2) : "r"(hb));
        asm("fma.rn.f16x2 %0, %0, %4, %0;" : "+r"(hv3) : "r"(hb));
        asm("fma.rn.f16x2 %0, %0, %4, %0;" : "+r"(hv4) : "r"(hb));
        hv ^= hv2 ^ hv3 ^ hv4;
#endif
    }

    if ((int)fv == seed && hv == (unsigned)seed) C[blockIdx.x * blockDim.x + threadIdx.x] = fv;
}
