// FFMA .reuse modifier - does it help?
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv[8];
    float fb = 1.000001f + (float)u2 * 1e-9f;
    float fc = 0.5f;

    #pragma unroll
    for (int k = 0; k < 8; k++) fv[k] = (float)threadIdx.x + 1.0f + k;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // 8 chains, all FFMAs reuse fb (which compiler should mark .reuse)
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            asm volatile("fma.rn.f32 %0, %0, %1, %0;" : "+f"(fv[k]) : "f"(fb));
        }
#elif MODE == 1
        // Each FFMA uses different fb (no .reuse possible)
        float fbs[8] = {1.001f, 1.002f, 1.003f, 1.004f, 1.005f, 1.006f, 1.007f, 1.008f};
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            asm volatile("fma.rn.f32 %0, %0, %1, %0;" : "+f"(fv[k]) : "f"(fbs[k]));
        }
#elif MODE == 2
        // 3-source: chain uses 3 distinct, fb shared
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv[k]) : "f"(fb), "f"(fc));
        }
#endif
    }

    float acc = 0;
    #pragma unroll
    for (int k = 0; k < 8; k++) acc += fv[k];
    if ((int)acc == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
