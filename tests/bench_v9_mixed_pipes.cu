// V9: Mixed FFMA + IADD — verify FMA and ALU pipes run in parallel
#ifndef N_FMA
#define N_FMA 8
#endif
#ifndef N_INT
#define N_INT 8
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv[N_FMA], fb[N_FMA];
    int   iv[N_INT], ib[N_INT];

    #pragma unroll
    for (int k = 0; k < N_FMA; k++) {
        fv[k] = (float)(threadIdx.x + k) * 0.001f;
        fb[k] = (float)(threadIdx.x * 2 + k) * 0.001f;
    }
    #pragma unroll
    for (int k = 0; k < N_INT; k++) {
        iv[k] = threadIdx.x + k;
        ib[k] = threadIdx.x * 2 + k;
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += 16) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            // FMA pipe ops
            #pragma unroll
            for (int k = 0; k < N_FMA; k++) {
                asm volatile("fma.rn.f32 %0, %0, %1, %0;" : "+f"(fv[k]) : "f"(fb[k]));
            }
            // ALU pipe ops (IADD)
            #pragma unroll
            for (int k = 0; k < N_INT; k++) {
                asm volatile("add.s32 %0, %0, %1;" : "+r"(iv[k]) : "r"(ib[k]));
            }
        }
    }

    float facc = 0;
    int iacc = 0;
    #pragma unroll
    for (int k = 0; k < N_FMA; k++) facc += fv[k];
    #pragma unroll
    for (int k = 0; k < N_INT; k++) iacc += iv[k];

    if ((int)facc + iacc == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = facc + (float)iacc;
}
