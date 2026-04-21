// V9: FP32 subnormal handling cost
// QuickRunCUDA uses -use_fast_math → should flush to zero (FTZ).
// Test by running FFMA on subnormal vs normal inputs.
#ifndef SUBNORMAL
#define SUBNORMAL 0  // 0=normal, 1=subnormal values
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float v[8], b[8];

    #pragma unroll
    for (int k = 0; k < 8; k++) {
#if SUBNORMAL == 0
        // Normal values ~ 1e-3 scale
        v[k] = (float)(threadIdx.x + k) * 0.001f;
        b[k] = (float)(threadIdx.x * 2 + k) * 0.001f;
#elif SUBNORMAL == 1
        // Subnormal values (< 1e-38)
        // 1e-40 is subnormal in FP32 (smallest normal ~1.18e-38)
        union { unsigned u; float f; } conv_v, conv_b;
        conv_v.u = 0x00000100 | threadIdx.x;  // small denormal-like bit pattern
        conv_b.u = 0x00000200 | k;
        v[k] = conv_v.f;
        b[k] = conv_b.f;
#endif
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += 16) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                asm volatile("fma.rn.f32 %0, %0, %1, %0;" : "+f"(v[k]) : "f"(b[k]));
            }
        }
    }

    float acc = 0;
    #pragma unroll
    for (int k = 0; k < 8; k++) acc += v[k];
    if ((int)acc == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
