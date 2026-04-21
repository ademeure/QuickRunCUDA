// V10: FMA + XU (rsqrt) pipe mixing test
// FFMA on FMA pipe + rsqrt on XU pipe — truly different pipes
#ifndef N_FMA
#define N_FMA 8
#endif
#ifndef N_XU
#define N_XU 8
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv[N_FMA], fb[N_FMA];
    float xv[N_XU];

    #pragma unroll
    for (int k = 0; k < N_FMA; k++) {
        fv[k] = (float)(threadIdx.x + k) * 0.001f;
        fb[k] = (float)(threadIdx.x * 2 + k) * 0.001f;
    }
    #pragma unroll
    for (int k = 0; k < N_XU; k++) {
        xv[k] = (float)(threadIdx.x + k + 1);
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += 16) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            #pragma unroll
            for (int k = 0; k < N_FMA; k++) {
                asm volatile("fma.rn.f32 %0, %0, %1, %0;" : "+f"(fv[k]) : "f"(fb[k]));
            }
            #pragma unroll
            for (int k = 0; k < N_XU; k++) {
                asm volatile("rsqrt.approx.f32 %0, %0;" : "+f"(xv[k]));
            }
        }
    }

    float facc = 0;
    float xacc = 0;
    #pragma unroll
    for (int k = 0; k < N_FMA; k++) facc += fv[k];
    #pragma unroll
    for (int k = 0; k < N_XU; k++) xacc += xv[k];

    if ((int)(facc + xacc) == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = facc + xacc;
}
