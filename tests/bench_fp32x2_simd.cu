// Does B300 support packed FP32x2 SIMD?
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv = (float)threadIdx.x + 0.5f;
    float fb = 1.000001f + (float)u2 * 1e-9f;
#if MODE == 1 || MODE == 2
    // Try f32x2 packed (Blackwell+?)
    float2 v2 = make_float2((float)threadIdx.x, (float)threadIdx.x + 1);
    float2 b2 = make_float2(1.000001f, 1.000002f);
#endif

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // FFMA scalar
        asm("fma.rn.f32 %0, %0, %1, %0;" : "+f"(fv) : "f"(fb));
#elif MODE == 1
        // PTX fma.rn.f32x2 — does this exist on B300?
        asm("fma.rn.f32x2 {%0, %1}, {%0, %1}, {%2, %3}, {%0, %1};"
            : "+f"(v2.x), "+f"(v2.y) : "f"(b2.x), "f"(b2.y));
#elif MODE == 2
        // CUDA float2 ops (what does compiler emit?)
        v2.x = v2.x * b2.x + v2.x;
        v2.y = v2.y * b2.y + v2.y;
#endif
    }

    if ((int)fv == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = fv;
}
