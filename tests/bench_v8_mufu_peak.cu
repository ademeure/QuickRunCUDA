// V8: MUFU (math unit) peak via rsqrt.approx.f32
// Theoretical: MUFU pipe = 1/4 of FP32 rate typically
#ifndef N_CHAINS
#define N_CHAINS 8
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float v[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) v[k] = (float)(threadIdx.x + k + 1);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += 16) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                asm volatile("rsqrt.approx.f32 %0, %0;" : "+f"(v[k]));
            }
        }
    }

    float acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc += v[k];
    if ((int)acc == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
